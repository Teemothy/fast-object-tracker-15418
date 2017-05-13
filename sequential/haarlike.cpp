#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "haarlike.h"

float rect_sum(float* integral_image, int height, int width,
               int y, int x, int bheight, int bwidth)
{
    float A = integral_image[y*width + x];
    float B = integral_image[y*width + (x+bwidth)];
    float C = integral_image[(y+bheight)*width + x];
    float D = integral_image[(y+bheight)*width + (x+bwidth)];
    return D + A - B - C;
}

float
haarlike_classifier::eval(float* integral_image, int height, int width,
                          int y, int x)
{
    float sum = 0.0;
    for(int i = 0; i < 3; i++)
    {
        sum += box_weights[i] * rect_sum(integral_image, height, width,
                                         y+box_y[i], x+box_x[i],
                                         box_height[i], box_width[i]);
    }
    return sum;
}

bool
haarlike_classifier::classify(float* integral_image, int height, int width,
                              int y, int x)
{
    return ((polarity*eval(integral_image, height, width, y, x)) > polarity*threshold);
}

haarlike_classifier::haarlike_classifier(int bounding_box_height,
                                         int bounding_box_width)
{
    bounding_height = bounding_box_height;
    bounding_width = bounding_box_width;

    srand(time(NULL));

    int num_boxes = 2 + (rand() % 2);

    for(int i = 0; i < 3; i++)
    {
        if(i < num_boxes)
        {
            box_height[i] = 1 + (rand() % bounding_height);
            box_width[i] = 1 + (rand() % bounding_width);
            box_y[i] = (rand() % (bounding_height-box_height[i]+1));
            box_x[i] = (rand() % (bounding_width-box_width[i]+1));
            box_weights[i] = (2.0*(rand()%2)-1.0) /
                             ((float)box_height[i]*box_width[i]);
        }
        else
        {
            box_height[i] = 0;
            box_width[i] = 0;
            box_y[i] = 0;
            box_x[i] = 0;
            box_weights[i] = 0.0;
        }
    }

    mean_pos = 0.0;
    pos_count = 0;
    mean_neg = 0.0;

    polarity = 0.0;
    threshold = 0.0;
}

void
haarlike_classifier::initial_train(float* integral_image, int height, int width,
                                   int pos_y, int pos_x,
                                   int* neg_y, int* neg_x, int num_neg)
{
    mean_pos = eval(integral_image, height, width,
                    pos_y, pos_x);
    pos_count = 1;

    for(int i = 0; i < num_neg; i++)
    {
        mean_neg += eval(integral_image, height, width,
                         neg_y[i], neg_x[i]);
    }
    mean_neg /= (float)num_neg;

    if (mean_pos > mean_neg)
        polarity = 1.0;
    else
        polarity = -1.0;

    threshold = (mean_pos + mean_neg)/2.0;
}

void
haarlike_classifier::classify_region(float* integral_image, int height, int width,
                                     bool* output)
{
    int output_height = height - bounding_height + 1;
    int output_width = width - bounding_width + 1;
    for(int i = 0; i < output_height; i++)
    {
        for(int j = 0; j < output_width; j++)
        {
            output[i*output_width + j] = classify(integral_image, height, width,
                                                  i, j);
        }
    }
}

void
haarlike_classifier::update(float* integral_image, int height, int width,
                            int pos_y, int pos_x)
{
    float value = eval(integral_image, height, width,
                       pos_y, pos_x);
    pos_count++;
    mean_pos = (mean_pos + value)/((float)pos_count);

    if (mean_pos > mean_neg)
        polarity = 1.0;
    else
        polarity = -1.0;

    threshold = (mean_pos + mean_neg)/2.0;
}

void
haarlike_classifier::print_params()
{
    printf("%d,%d\n", bounding_height, bounding_width);
    for(int i = 0; i < 3; i++)
    {
        printf("%d,%d,%d,%d,%f\n", box_y[i], box_x[i], box_height[i], box_width[i],
                                   box_weights[i]);
    }
    printf("%f,%d,%f\n", mean_pos, pos_count, mean_neg);
    printf("%f,%f\n", polarity, threshold);
}
