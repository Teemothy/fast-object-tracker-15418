#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <cuda_runtime.h>

#include "object_tracker.h"

void draw_design(float* image, int height, int width,
                 int y, int x)
{
    image[(y+1)*width + (x+2)] = 1.0;
    image[(y+2)*width + (x+2)] = 1.0;
    image[(y+1)*width + (x+4)] = 1.0;
    image[(y+2)*width + (x+4)] = 1.0;
    image[(y+2)*width + (x+3)] = 0.0;
    image[(y+3)*width + (x+3)] = 0.0;
    image[(y+4)*width + (x+3)] = 1.0;
    image[(y+5)*width + (x+0)] = 1.0;
    image[(y+5)*width + (x+2)] = 1.0;
    image[(y+5)*width + (x+3)] = 1.0;
    image[(y+5)*width + (x+4)] = 1.0;
    image[(y+5)*width + (x+6)] = 1.0;
}

void clear_image(float* image, int height, int width)
{
    for(int i = 0; i < height * width; i++)
        image[i] = 0.5;
}

int main(int argc, char** argv)
{
    int height = 256;
    int width = 256;
    int num = height * width;
    int size = num * sizeof(float);

    float* image = (float*)malloc(size);

    int y = 128;
    int x = 128+30;
    clear_image(image, height, width);
    draw_design(image, height, width, y, x);

    float* device_image;
    cudaMalloc(&device_image, size);
    cudaMemcpy(device_image, image, size, cudaMemcpyHostToDevice);

    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            //printf("%f,", input[i*width + j]);
        }
        //printf("\n");
    }

    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            //printf("%f,", output[i*width + j]);
        }
        //printf("\n");
    }

    object_tracker z = object_tracker(6,7,
                                      height, width,
                                      16, 16,
                                      200,
                                      5, 10,
                                      50);

    int neg_y[8] = {1,2,3,6,15,18,20,25};
    int neg_x[8] = {3,7,0,1,20,10,15,12};
    int num_neg = 8;
    int* device_neg_y;
    int* device_neg_x;
    cudaMalloc(&device_neg_y, 8 * sizeof(int));
    cudaMalloc(&device_neg_x, 8 * sizeof(int));
    cudaMemcpy(device_neg_y, neg_y, 8 * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(device_neg_x, neg_x, 8 * sizeof(int),cudaMemcpyHostToDevice);
    z.initial_train(device_image,
                    y, x,
                    device_neg_y, device_neg_x, num_neg);
    for (int i = 0; i < 60; i++)
    {
        y = ((int)(128.0 + 30.0*sin((float)i/20.0)));
        x = ((int)(128.0 + 30.0*cos((float)i/20.0)));
        clear_image(image, height, width);
        draw_design(image, height, width, y, x);

        cudaMemcpy(device_image, image, size, cudaMemcpyHostToDevice);
        std::pair<int, int> coord = z.track_frame(device_image);
        printf("Step %d: Put at position %d, %d. Found at position %d, %d\n",
               i, y, x, coord.first, coord.second);

    }
}
