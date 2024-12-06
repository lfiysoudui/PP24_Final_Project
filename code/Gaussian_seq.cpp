#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <png.h>

using namespace std;

// 定義 Array, Matrix, 和 Image 的型態，方便處理圖像數據
typedef vector<double> Array;        // 用一維數組表示每個像素通道（紅、綠、藍）
typedef vector<Array> Matrix;        // 用二維數組表示一個通道的圖像（例如，紅色通道）
typedef vector<Matrix> Image;        // 用三維數組表示完整的圖像，包含紅、綠、藍三個通道

// 計算高斯濾波器（卷積核）
Matrix getGaussian(int height, int width, double sigma)
{
    Matrix kernel(height, Array(width));  // 初始化大小為 (height, width) 的高斯濾波器
    double sum = 0.0;  // 用於計算高斯濾波器所有元素的總和，用於歸一化
    int i, j;

    // 計算高斯濾波器的每個值
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            // 高斯公式計算濾波器中的每個值
            kernel[i][j] = exp(-(i*i + j*j) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            sum += kernel[i][j];  // 累加總和
        }
    }

    // 歸一化濾波器，使其總和為 1
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            kernel[i][j] /= sum;
        }
    }

    return kernel;  // 返回計算好的高斯濾波器
}

// 加載圖像並轉換為 Image 格式
Image loadImage(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        cerr << "Error opening file!" << endl;
        exit(1);
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);

    if (setjmp(png_jmpbuf(png))) {
        cerr << "Error reading PNG file!" << endl;
        exit(1);
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    int width = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);

    if (color_type != PNG_COLOR_TYPE_RGB) {
        cerr << "Unsupported color type!" << endl;
        exit(1);
    }

    Image imageMatrix(3, Matrix(height, Array(width)));
    png_bytepp row_pointers = (png_bytepp)malloc(sizeof(png_bytep) * height);

    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_bytep)malloc(png_get_rowbytes(png, info));
    }

    png_read_image(png, row_pointers);
    fclose(fp);

    for (int y = 0; y < height; y++) {
        png_bytep row = row_pointers[y];
        for (int x = 0; x < width; x++) {
            imageMatrix[0][y][x] = row[x * 3];     // 紅色通道
            imageMatrix[1][y][x] = row[x * 3 + 1]; // 綠色通道
            imageMatrix[2][y][x] = row[x * 3 + 2]; // 藍色通道
        }
        free(row_pointers[y]);
    }
    free(row_pointers);

    return imageMatrix;  // 返回圖像矩陣
}

// 保存圖像
void saveImage(Image &image, const char *filename)
{
    assert(image.size() == 3);  // 確保圖像有三個通道

    int height = image[0].size();  // 圖像的高度
    int width = image[0][0].size();  // 圖像的寬度

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        cerr << "Error opening file!" << endl;
        exit(1);
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);
    if (setjmp(png_jmpbuf(png))) {
        cerr << "Error writing PNG file!" << endl;
        exit(1);
    }

    png_init_io(png, fp);

    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_ADAM7, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    png_bytepp row_pointers = (png_bytepp)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) {
        png_bytep row = (png_bytep)malloc(width * 3 * sizeof(png_byte));
        for (int x = 0; x < width; x++) {
            row[x * 3] = image[0][y][x];    // 紅色通道
            row[x * 3 + 1] = image[1][y][x];  // 綠色通道
            row[x * 3 + 2] = image[2][y][x];   // 藍色通道
        }
        row_pointers[y] = row;
    }

    png_write_image(png, row_pointers);
    png_write_end(png, nullptr);

    fclose(fp);

    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);
}


// 應用濾波器（卷積）對圖像進行處理
Image applyFilter(Image &image, Matrix &filter)
{
    assert(image.size() == 3 && filter.size() != 0);  // 確保圖像和濾波器有效

    int height = image[0].size();  // 圖像高度
    int width = image[0][0].size();  // 圖像寬度
    int filterHeight = filter.size();  // 濾波器高度
    int filterWidth = filter[0].size();  // 濾波器寬度
    int newImageHeight = height - filterHeight + 1;  // 濾波後圖像的高度
    int newImageWidth = width - filterWidth + 1;    // 濾波後圖像的寬度
    int d, i, j, h, w;

    // 創建一個新的圖像（空的）
    Image newImage(3, Matrix(newImageHeight, Array(newImageWidth)));

    // 對每個顏色通道進行濾波操作
    for (d = 0; d < 3; d++) {
        for (i = 0; i < newImageHeight; i++) {
            for (j = 0; j < newImageWidth; j++) {
                for (h = i; h < i + filterHeight; h++) {
                    for (w = j; w < j + filterWidth; w++) {
                        // 將濾波器對應區域的像素與濾波器值相乘並累加
                        newImage[d][i][j] += filter[h - i][w - j] * image[d][h][w];
                    }
                }
            }
        }
    }

    return newImage;  // 返回濾波後的新圖像
}

// 重複應用濾波器多次
Image applyFilter(Image &image, Matrix &filter, int times)
{
    Image newImage = image;  // 初始化新圖像為原圖像
    for (int i = 0; i < times; i++) {
        newImage = applyFilter(newImage, filter);  // 每次應用濾波器
    }
    return newImage;  // 返回處理後的圖像
}

int main()
{
    // 從文件加載圖像
    Image image = loadImage("input.png");

    // 計算高斯濾波器
    Matrix gaussian = getGaussian(5, 5, 1.0);

    // 將濾波器應用於圖像
    Image filteredImage = applyFilter(image, gaussian, 3);

    // 保存結果圖像
    saveImage(filteredImage, "output.png");

    return 0;
}
