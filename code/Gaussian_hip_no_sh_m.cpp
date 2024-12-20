#include <png.h>
#include <zlib.h>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <iostream>
#include <sstream>  // For stringstream
#include <chrono>   // For timing
#include "hip/hip_runtime.h"

#define MASK_X 20   //mask size
#define MASK_Y MASK_X
#define BLK_SIZE 8
// #define DEBUG
double kernel[MASK_X][MASK_Y];
__device__ __constant__ double d_kernel[MASK_X][MASK_Y];

// Function to generate a filename based on image size (width and height)
std::string generateFilename(char* inputname)
{
    // Use stringstream to construct a string with width and height as part of the filename
    std::stringstream filename;
    filename << inputname << "_[" << MASK_X << "x" << MASK_Y << "]_out.png";
    return filename.str();  // Convert the stringstream to a string and return
}

void set_filter(double sigma)
{
    double sum = 0.0; // to normalize
    double r, s = 2.0 * sigma * sigma; 
    int i,j;
  
    // generating 5x5 kernel 
    for (i = -MASK_X/2; i < MASK_X/2 + MASK_X%2; i++) { 
        for (j = -MASK_Y/2; j < MASK_Y/2 + MASK_Y%2; j++) { 
            r = sqrt(i * i + j * j); 
            kernel[i + MASK_X/2][j + MASK_X/2] = (exp(-(r * r) / s)) / (M_PI * s); 
            sum += kernel[i + MASK_X/2][j + MASK_X/2]; 
        } 
    } 

    double sum_ = 0.0;
    for (i = 0; i < MASK_Y; i++) {
        for (j = 0; j < MASK_X; j++) {
            kernel[i][j] /= sum;
        }
    }
    #ifdef DEBUG
    std::cout << "[Filter]" << std::endl;
    for (i = 0; i < MASK_Y; i++) {
        std::cout << "[";
        for (j = 0; j < MASK_X; j++) {
            std::cout << std::setw(10) << std::setprecision(3) << kernel[i][j];
        }
        std::cout << "]" << std::endl;
    }
    #endif

    return;
}


int read_png(const char* filename, unsigned char** image, unsigned* height, unsigned* width, unsigned* channels, unsigned* imgsize) {
    FILE* infile = fopen(filename, "rb");
    if (!infile) return 1;

    unsigned char sig[8];
    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8)) {
        fclose(infile);
        return 1;
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fclose(infile);
        return 4;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        fclose(infile);
        return 4;
    }

    png_set_sig_bytes(png_ptr, 8);
    png_init_io(png_ptr, infile);

    png_read_info(png_ptr, info_ptr);
    *width = png_get_image_width(png_ptr, info_ptr);
    *height = png_get_image_height(png_ptr, info_ptr);
    *channels = png_get_channels(png_ptr, info_ptr);
    int bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    if (bit_depth == 16) png_set_strip_16(png_ptr);
    if (*channels < 3) png_set_expand_gray_1_2_4_to_8(png_ptr);

    png_set_palette_to_rgb(png_ptr);
    png_read_update_info(png_ptr, info_ptr);

    png_uint_32 rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *imgsize = rowbytes * *height;

    *image = (unsigned char*)malloc(*imgsize);
    png_bytep row_pointers[*height];
    for (unsigned int i = 0; i < *height; ++i) {
        row_pointers[i] = *image + i * rowbytes;
    }

    png_read_image(png_ptr, row_pointers);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(infile);
    return 0;
}


void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) return;

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fclose(fp);
        return;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, NULL);
        fclose(fp);
        return;
    }

    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    png_bytep row_pointers[height];
    for (unsigned int i = 0; i < height; ++i) {
        row_pointers[i] = image + i * width * channels;
    }

    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0); // Low compression for speed
    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

__global__ void Gaussian(unsigned char* d_s, unsigned char* d_tg, unsigned height, unsigned width, unsigned channels) {
    __shared__ unsigned char sharedMem[BLK_SIZE + MASK_Y][BLK_SIZE + MASK_X][3];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int xBound  = MASK_X / 2;
    int yBound  = MASK_Y / 2;

    int padded_y, padded_x;

    if(x < width && y < height) {
        double R, G, B;
        double val[3] = {0.0};
        #pragma unroll
        for (int v = -yBound; v < ((MASK_Y % 2) ? 1 : 0); ++v) {
            for (int u = -xBound; u < xBound + ((MASK_X % 2) ? 1 : 0); ++u) {
                // padding y
                if ((x + u) < 0) padded_x = -(x + u + 1);
                else if ((x + u) >= width) padded_x = width - (x + u - width + 1);
                else padded_x = x + u;
                // padding y
                if (y + v < 0) padded_y = -(y + v + 1);
                else if (y + v >= height) padded_y = height - (y + v - height + 1);
                else padded_y = y + v;
                R = d_s[channels * (width * padded_y + padded_x) + 2];
                G = d_s[channels * (width * padded_y + padded_x) + 1];
                B = d_s[channels * (width * padded_y + padded_x) + 0];
                val[2] += R * d_kernel[u][v];
                val[1] += G * d_kernel[u][v];
                val[0] += B * d_kernel[u][v];
            }
        }
        d_tg[channels * (width * y + x) + 2] = (val[2] > 255.0) ? 255 : val[2];
        d_tg[channels * (width * y + x) + 1] = (val[1] > 255.0) ? 255 : val[1];
        d_tg[channels * (width * y + x) + 0] = (val[0] > 255.0) ? 255 : val[0];
    }
}

void print_time_n(std::chrono::nanoseconds time) {
    int seconds = time.count() / 1000;
    int milliseconds = time.count() % 1000;
    std::cout << "Execution time: " << time.count() << " ns" << std::endl;
}
void print_time(std::chrono::milliseconds time) {
    int seconds = time.count() / 1000;
    int milliseconds = time.count() % 1000;
    std::cout << "Execution time: " << seconds << " s " << milliseconds << " ms" << std::endl;
}

int main(int argc, char** argv) {
    if(!(argc > 1 && argc < 4)){
        std::cerr << "[Usage] ./Gaussian input.png [optional output.png]" << std::endl;
        return 1;
    }
    

    unsigned height, width, channels, imgsize;
    unsigned char* src_img,  *d_src_img, *d_dst_img;;

    read_png(argv[1], &src_img, &height, &width, &channels, &imgsize);
    std::cout << "File Name      : " << argv[1] << std::endl;
    std::cout << "Channel        : " << channels << std::endl;
    if(channels != 3){
        std::cerr << "[Usage] please use convert.py to make it 3-channel" << std::endl;
        return 1;
    }

    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();
    // Generate the filter in gpu
    hipMalloc((void**)&d_src_img, imgsize);
    hipMemcpyAsync(d_src_img, src_img, imgsize, hipMemcpyHostToDevice);
    set_filter(1.0);
    hipMemcpyToSymbol(d_kernel, &kernel, MASK_X * MASK_Y * sizeof(double));
    hipMalloc((void**)&d_dst_img, imgsize);
    // Define block size
    dim3 blockSize = (BLK_SIZE,BLK_SIZE); // Set block size (experiment for optimal performance)
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

    auto slice_1 = std::chrono::high_resolution_clock::now();
    // Apply Gaussian filter two times
    hipLaunchKernelGGL(Gaussian, gridSize, blockSize, 0, 0, d_src_img, d_dst_img, height, width, channels);
    hipLaunchKernelGGL(Gaussian, gridSize, blockSize, 0, 0, d_dst_img, d_src_img, height, width, channels);
    auto slice_2 = std::chrono::high_resolution_clock::now();
    //copy back the result image
    unsigned char *dst_img = (unsigned char *)malloc(imgsize);
    hipMemcpyAsync(dst_img, d_src_img, imgsize, hipMemcpyDeviceToHost);

    // End the timer
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "BLK_SIZE       : " << BLK_SIZE << std::endl;
    std::cout << "total runtime  : ";
    print_time(std::chrono::duration_cast<std::chrono::milliseconds>(end - start));
    std::cout << "Copy to GPU    : ";
    print_time(std::chrono::duration_cast<std::chrono::milliseconds>(slice_1 - start));
    std::cout << "Kernel runtime : ";
    print_time_n(std::chrono::duration_cast<std::chrono::nanoseconds>(slice_2 - slice_1));
    std::cout << "Copy to CPU    : ";
    print_time(std::chrono::duration_cast<std::chrono::milliseconds>(end - slice_2));
    std::cout << std::endl;

    if (argc == 3)
        write_png(argv[2], dst_img, height, width, channels);
    else
        write_png(generateFilename(argv[1]).c_str(), dst_img, height, width, channels);

    hipFree(d_src_img);
    hipFree(d_dst_img);

    return 0;
}
