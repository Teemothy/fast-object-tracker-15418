#include <stdlib.h>
#include <stdio.h>
#include <cmath>

#include "cycleTimer.h"
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
    image[(y+5)*width + (x+2)] = 0.0;
    image[(y+5)*width + (x+3)] = 1.0;
    image[(y+5)*width + (x+4)] = 1.0;
    image[(y+5)*width + (x+6)] = 1.0;
    image[(y+11)*width + (x+2)] = 1.0;
    image[(y+12)*width + (x+2)] = 0.0;
    image[(y+11)*width + (x+4)] = 1.0;
    image[(y+12)*width + (x+4)] = 1.0;
    image[(y+12)*width + (x+3)] = 0.0;
    image[(y+13)*width + (x+3)] = 0.0;
    image[(y+14)*width + (x+3)] = 1.0;
    image[(y+15)*width + (x+0)] = 1.0;
    image[(y+15)*width + (x+2)] = 0.0;
    image[(y+15)*width + (x+3)] = 1.0;
    image[(y+15)*width + (x+4)] = 0.0;
    image[(y+15)*width + (x+6)] = 1.0;
    image[(y+11)*width + (x+12)] = 1.0;
    image[(y+12)*width + (x+12)] = 0.0;
    image[(y+11)*width + (x+14)] = 1.0;
    image[(y+12)*width + (x+14)] = 1.0;
    image[(y+12)*width + (x+13)] = 0.0;
    image[(y+13)*width + (x+8)] = 0.0;
    image[(y+14)*width + (x+4)] = 1.0;
    image[(y+15)*width + (x+9)] = 0.0;
    image[(y+15)*width + (x+7)] = 1.0;
    image[(y+15)*width + (x+10)] = 1.0;
    image[(y+15)*width + (x+5)] = 1.0;
    image[(y+15)*width + (x+6)] = 1.0;
}

void clear_image(float* image, int height, int width)
{
    for(int i = 0; i < height * width; i++)
        image[i] = 0.5;
}

int main(int argc, char** argv)
{
    int height = 1024;
    int width = 1024;
    int num = height * width;
    int size = num * sizeof(float);

    float* image = (float*)malloc(size);

    int y = 512;
    int x = 512+100;
    clear_image(image, height, width);
    draw_design(image, height, width, y, x);

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

    object_tracker z = object_tracker(16,16,
                                      height, width,
                                      32, 32,
                                      500,
                                      5, 10,
                                      50);

    int neg_y[8] = {1,2,3,6,15,18,20,25};
    int neg_x[8] = {3,7,0,1,20,10,15,12};
    int num_neg = 8;
    z.initial_train(image,
                    y, x,
                    neg_y, neg_x, num_neg);
    double timeElapsed = 0.0;
    for (int i = 0; i < 6000; i++)
    {
        y = ((int)(512.0 + 100.0*sin((float)i/20.0)));
        x = ((int)(512.0 + 100.0*cos((float)i/20.0)));
        clear_image(image, height, width);
        draw_design(image, height, width, y, x);

        double startTime = CycleTimer::currentSeconds();
        std::pair<int, int> coord = z.track_frame(image);
        double endTime = CycleTimer::currentSeconds();
        timeElapsed += (endTime - startTime);
        printf("Step %d: Put at position %d, %d. Found at position %d, %d\n",
               i, y, x, coord.first, coord.second);
        printf("Time elapsed:%f\n", endTime - startTime);
    }
    printf("Avg time:%f\n", timeElapsed/6000.0);
}
