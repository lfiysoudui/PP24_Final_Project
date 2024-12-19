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
#define BLK_SIZE 64
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
    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8)) return 1; /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) return 4; /* out of memory */

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4; /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32 i, rowbytes;
    png_bytep row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int)png_get_channels(png_ptr, info_ptr);
    *imgsize = rowbytes * *height;
    if ((*image = (unsigned char*)malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0; i < *height; ++i) {
        row_pointers[i] = *image + i * rowbytes;
    }

    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width,
    const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 0);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
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

    // Load pixels to shared memory
    int x_start = (threadIdx.x == 0)? -xBound : 0;
    int x_end   = (threadIdx.x == blockDim.x - 1)? xBound + ((MASK_X % 2) ? 1 : 0) : 1;
    int y_start = (threadIdx.y == 0)? -yBound : 0;
    int y_end   = (threadIdx.y == blockDim.y - 1)? yBound + ((MASK_Y % 2) ? 1 : 0) : 1;

    int padded_y, padded_x;
    for(int v = y_start; v < y_end; ++v){
        for(int u = x_start; u < x_end; ++u){
            // padding x
            if ((x + u) < 0) padded_x = -(x + u + 1);
            else if ((x + u) >= width) padded_x = width - (x + u - width + 1);
            else padded_x = x + u;
            // padding y
            if (y + v < 0) padded_y = -(y + v + 1);
            else if (y + v >= height) padded_y = height - (y + v - height + 1);
            else padded_y = y + v;

            for (int c = 0; c < channels; c++) {
                sharedMem[yBound + v + threadIdx.y][xBound + u + threadIdx.x][c]
                    = d_s[channels * (width * padded_y + padded_x) + c];
            }
        }
    }
    __syncthreads();

    if(x < width && y < height) {
        double R, G, B;
        double val[3] = {0.0};
        #pragma unroll
        for (int v = 0; v < MASK_Y; ++v) {
            #pragma unroll
            for (int u = 0; u < MASK_X; ++u) {
                R = sharedMem[threadIdx.y + v][threadIdx.x + u][2];
                G = sharedMem[threadIdx.y + v][threadIdx.x + u][1];
                B = sharedMem[threadIdx.y + v][threadIdx.x + u][0];
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

int main(int argc, char** argv) {
    if(!(argc > 1 && argc < 4)){
        std::cerr << "[Usage] ./Gaussian input.png [optional output.png]" << std::endl;
        return 1;
    }
    

    unsigned height, width, channels, imgsize;
    unsigned char* src_img,  *d_src_img, *d_intermediate_img, *d_dst_img;;

    read_png(argv[1], &src_img, &height, &width, &channels, &imgsize);
    std::cout << "channel : " << channels << std::endl;
    if(channels != 3){
        std::cerr << "[Usage] please use convert.py to make it 3-channel" << std::endl;
        return 1;
    }
    hipMalloc((void**)&d_src_img, imgsize);
    hipMemcpy(d_src_img, src_img, imgsize, hipMemcpyHostToDevice);

    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();
    // Generate the filter in gpu
    set_filter(1.0);
    hipMemcpyToSymbol(d_kernel, &kernel, MASK_X * MASK_Y * sizeof(double));

    hipMalloc((void**)&d_intermediate_img, imgsize);
    hipMalloc((void**)&d_dst_img, imgsize);

    // Define block size
    dim3 blockSize = (BLK_SIZE,BLK_SIZE); // Set block size (experiment for optimal performance)
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

    // Apply Gaussian filter two times
    hipLaunchKernelGGL(Gaussian, gridSize, blockSize, 0, 0, d_src_img, d_intermediate_img, height, width, channels);
    hipLaunchKernelGGL(Gaussian, gridSize, blockSize, 0, 0, d_intermediate_img, d_dst_img, height, width, channels);

    //copy back the result image
    unsigned char *dst_img = (unsigned char *)malloc(imgsize);
    hipMemcpy(dst_img, d_dst_img, imgsize, hipMemcpyDeviceToHost);

    // End the timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    int seconds = duration.count() / 1000;
    int milliseconds = duration.count() % 1000;
    std::cout << "Execution time: " << seconds << " s " << milliseconds << " ms" << std::endl;

    if (argc == 3)
        write_png(argv[2], dst_img, height, width, channels);
    else
        write_png(generateFilename(argv[1]).c_str(), dst_img, height, width, channels);

    hipFree(d_src_img);
    hipFree(d_intermediate_img);
    hipFree(d_dst_img);

    return 0;
}
