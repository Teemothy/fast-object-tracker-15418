#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <cuda_runtime.h>

#include "object_tracker.h"
#include "cycleTimer.h"

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
    int height = 1026;
    int width = 1024;
    int num = height * width;
    int size = num * sizeof(float);

    float radius = 20.0;
    float period = 80.0;
    int num_iter = 10000;

    int bounding_box_height = 32;
    int bounding_box_width = 32;
    int search_box_height = 16;
    int search_box_width = 16;
    int classifiers_count = 250;
    int steps_to_discard = 10;
    int discard_count = 50;
    int best_count = 50;

    float* image = (float*)malloc(size);

    int y = height/2;
    int x = width/2 + (int)radius;
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

    object_tracker z = object_tracker(bounding_box_height,bounding_box_width,
                                      height, width,
                                      search_box_height, search_box_width,
                                      classifiers_count,
                                      steps_to_discard, discard_count,
                                      best_count);

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

    double timeElapsed = 0.0;
    double transferTime = 0.0;

    for (int i = 0; i < num_iter; i++)
    {
        y = ((int)(height/2 + radius*sin((float)i/period)));
        x = ((int)(width/2 + radius*cos((float)i/period)));
        clear_image(image, height, width);
        draw_design(image, height, width, y, x);

        double startTime = CycleTimer::currentSeconds();
        cudaMemcpy(device_image, image, size, cudaMemcpyHostToDevice);
        double middleTime = CycleTimer::currentSeconds();
        std::pair<int, int> coord = z.track_frame(device_image);
        double endTime = CycleTimer::currentSeconds();
        if (y != coord.first || x != coord.second)
            printf("Error at step:%d\n", i);
        timeElapsed += endTime - startTime;
        transferTime += middleTime - startTime;
    }

    printf("Frames per second: %f\n", ((float)num_iter)/timeElapsed);
    printf("Transfer fraction: %f\n", transferTime/timeElapsed);
}
