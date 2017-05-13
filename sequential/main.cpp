#include <stdlib.h>
#include <stdio.h>
#include <cmath>

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
    int height = 512;
    int width = 1024;
    int num = height * width;
    int size = num * sizeof(float);

    float radius = 200.0;
    float period = 40.0;
    int num_iter = 10000;

    int bounding_box_height = 15;
    int bounding_box_width = 15;
    int search_box_height = 16;
    int search_box_width = 16;
    int classifiers_count = 250;
    int steps_to_discard = 10;
    int discard_count = 50;
    int best_count = 100;

    float* image = (float*)malloc(size);

    int y = height/2;
    int x = width/2 + (int)radius;
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

    object_tracker z = object_tracker(bounding_box_height,bounding_box_width,
                                      height, width,
                                      search_box_height, search_box_width,
                                      classifiers_count,
                                      steps_to_discard, discard_count,
                                      best_count);

    int neg_y[8] = {1,2,3,6,15,18,20,25};
    int neg_x[8] = {3,7,0,1,20,10,15,12};
    int num_neg = 8;
    z.initial_train(image,
                    y, x,
                    neg_y, neg_x, num_neg);

    double timeElapsed = 0.0;

    for (int i = 0; i < num_iter; i++)
    {
        y = ((int)(height/2 + radius*sin((float)i/period)));
        x = ((int)(width/2 + radius*cos((float)i/period)));
        clear_image(image, height, width);
        draw_design(image, height, width, y, x);

        double startTime = CycleTimer::currentSeconds();
        std::pair<int, int> coord = z.track_frame(image);
        double endTime = CycleTimer::currentSeconds();
        timeElapsed += endTime - startTime;
    }

    printf("Frames per second: %f\n", ((float)num_iter)/timeElapsed);
}
