#include <png.h>
#include <zlib.h>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <iostream>
#include <sstream>  // For stringstream
#include <chrono>   // For timing
#include <cuda_runtime.h>

#define MASK_X 10   //mask size
#define ITER_T 3    //iteration time
#define MASK_Y MASK_X
double** kernel;
__constant__ double d_kernel[MASK_X][MASK_Y]; // 使用 constant memory 儲存 kernel

// Function to generate a filename based on image size (width and height)
std::string generateFilename(char* inputname)
{
    // Use stringstream to construct a string with width and height as part of the filename
    std::stringstream filename;
    filename << inputname << "_[" << MASK_X << "x" << MASK_Y << "].png";
    return filename.str();  // Convert the stringstream to a string and return
}

void set_filter(double sigma)
{
    kernel = (double**)malloc(sizeof(double*) * MASK_Y);
    for(int i = 0; i < MASK_Y; i++) kernel[i] = (double*)malloc(sizeof(double) * MASK_X);

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
    #ifdef debug
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

int read_png(const char* filename, unsigned char** image, unsigned* height, unsigned* width,
    unsigned* channels) {
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

__global__ void GaussianKernel(unsigned char* d_s, unsigned char* d_t, unsigned height, unsigned width, unsigned channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return; // 確保不處理無效像素

    int adjustX = (MASK_X % 2) ? 1 : 0;
    int adjustY = (MASK_Y % 2) ? 1 : 0;
    int xBound = MASK_X / 2;
    int yBound = MASK_Y / 2;

    double val[3] = {0.0};

    for (int v = -yBound; v < yBound + adjustY; ++v) {
        for (int u = -xBound; u < xBound + adjustX; ++u) {
            int nx = x + u;
            int ny = y + v;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                unsigned char R = d_s[channels * (width * ny + nx) + 2];
                unsigned char G = d_s[channels * (width * ny + nx) + 1];
                unsigned char B = d_s[channels * (width * ny + nx) + 0];

                val[2] += R * d_kernel[u + xBound][v + yBound];
                val[1] += G * d_kernel[u + xBound][v + yBound];
                val[0] += B * d_kernel[u + xBound][v + yBound];
            }
        }
    }

    d_t[channels * (width * y + x) + 2] = (val[2] > 255.0) ? 255 : val[2];
    d_t[channels * (width * y + x) + 1] = (val[1] > 255.0) ? 255 : val[1];
    d_t[channels * (width * y + x) + 0] = (val[0] > 255.0) ? 255 : val[0];
}

void GaussianCUDA(unsigned char* h_s, unsigned char* h_t, unsigned height, unsigned width, unsigned channels) {
    unsigned char *d_s, *d_t;
    size_t imageSize = height * width * channels * sizeof(unsigned char);

    cudaMalloc(&d_s, imageSize);
    cudaMalloc(&d_t, imageSize);
    cudaMemcpy(d_s, h_s, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_kernel, kernel, MASK_X * MASK_Y * sizeof(double));

    dim3 blockSize(MASK_X, MASK_Y); // 直接把 block size 設成跟 filter 的 size 一樣，配合複製到 device 上的 mem
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);


    GaussianKernel<<<gridSize, blockSize>>>(d_s, d_t, height, width, channels);

    cudaMemcpy(h_t, d_t, imageSize, cudaMemcpyDeviceToHost);
    cudaFree(d_s);
    cudaFree(d_t);
}

int main(int argc, char** argv) {
    assert((argc < 2, "[Usage] ./Gaussian input.png [optional output.png]"));

    unsigned height, width, channels;
    unsigned char* src_img = NULL;

    read_png(argv[1], &src_img, &height, &width, &channels);
    assert(channels == 3);

    unsigned char* dst_img =
        (unsigned char*)malloc(height * width * channels * sizeof(unsigned char));

    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();
    set_filter(1.0);
    GaussianCUDA(src_img, dst_img, height, width, channels);

    // End the timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    int seconds = duration.count() / 1000;
    int milliseconds = duration.count() % 1000;
    std::cout << "Execution time: " << seconds << " s " << milliseconds << " ms" << std::endl;
    if(argc == 3)
        write_png(argv[2], dst_img, height, width, channels);
    else
        write_png(generateFilename(argv[1]).c_str(), dst_img, height, width, channels);

    free(src_img);
    free(dst_img);

    return 0;
}
