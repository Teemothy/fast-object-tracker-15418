#include <cstdlib>
#include <utility>
#include <cmath>
#include <stdio.h>

#include "haar_pool.h"
#include "integral_image.h"

haarlike_pool::haarlike_pool(int bounding_box_height, int bounding_box_width,
                             int image_height, int image_width,
                             int search_box_height, int search_box_width,
                             int classifiers_count,
                             int steps_to_discard, int discard_count)
{
    bounding_height = bounding_box_height;
    bounding_width = bounding_box_width;

    height = image_height;
    width = image_width;
    integral_image_cache = (float*)malloc(height * width * sizeof(float));

    search_height = search_box_height;
    search_width = search_box_width;

    num_classifiers = classifiers_count;

    pool = (haarlike_classifier*)malloc(num_classifiers * sizeof(haarlike_classifier));
    correct = (int*)malloc(num_classifiers * sizeof(int));
    seen = (int*)malloc(num_classifiers * sizeof(int));
    error = (float*)malloc(num_classifiers * sizeof(float));
    alpha = (float*)malloc(num_classifiers * sizeof(float));
    idxs = (int*)malloc(num_classifiers * sizeof(int));
    error_copy = (float*)malloc(num_classifiers * sizeof(float));

    for(int i = 0; i < num_classifiers; i++)
    {
        pool[i] = haarlike_classifier(bounding_height, bounding_width);
        correct[i] = 2;
        seen[i] = 3;
        error[i] = 1.0/3.0;
        alpha[i] = 0.5 * logf((1.0 - error[i])/error[i]);
        idxs[i] = i;
        error_copy[i] = 1.0/3.0;
    }


    confidence_maps = (float*)malloc(
               num_classifiers * search_height * search_width * sizeof(float));
    discard_steps = steps_to_discard;
    num_discard = discard_count;

    steps = 0;
}

void
haarlike_pool::initial_train(float* image,
                             int pos_y, int pos_x,
                             int* neg_y, int* neg_x, int num_neg)
{
    integral_image(image, integral_image_cache, height, width);
    for(int i = 0; i < num_classifiers; i++)
        pool[i].initial_train(integral_image_cache, height, width,
                              pos_y, pos_x,
                              neg_y, neg_x, num_neg);
}

void
haarlike_pool::classify_region(float* image, int search_y, int search_x)
{
    integral_image(image, integral_image_cache, height, width);
    for(int c = 0; c < num_classifiers; c++)
    {
        for(int i = 0; i < search_height; i++)
        {
            for(int j = 0; j < search_width; j++)
            {
                float* confidence_map = confidence_maps+(c*search_height*search_width);
                confidence_map[i*search_width+j] = (float)
                    pool[c].classify(integral_image_cache, height, width,
                            search_y+i, search_x+j);
            }
        }
    }
}

void sort_by_value(int* keys, float* values, int num)
{
    for(int i = 0; i < num; i++)
    {
        int min_ind = i;
        int min_val = values[i];
        for(int j = i; j < num; j++)
        {
            if(values[j] < min_val)
            {
                min_val = values[j];
                min_ind = j;
            }
        }
        int tmp_key = keys[i];
        int tmp_val = values[i];
        keys[i] = keys[min_ind];
        values[i] = values[min_ind];
        keys[min_ind] = tmp_key;
        values[min_ind] = tmp_val;
    }
}

void
haarlike_pool::update(int pos_y, int pos_x)
{
    steps++;

    for(int i = 0; i < num_classifiers; i++)
    {
        bool is_correct = pool[i].classify(integral_image_cache, height, width, pos_y, pos_x);
        pool[i].update(integral_image_cache, height, width, pos_y, pos_x);
        if (is_correct)
            correct[i]++;
        seen[i]++;
        error[i] = 1.0 - ((float)correct[i])/seen[i];
        alpha[i] = 0.5 * logf((1.0 - error[i])/error[i]);
        idxs[i] = i;
        error_copy[i] = error[i];
    }

    sort_by_value(idxs, error_copy, num_classifiers);

    if(steps % discard_steps == 0)
    {
        for(int i = num_classifiers - num_discard; i < num_classifiers; i++)
        {
            int idx = idxs[i];
            pool[idx] = haarlike_classifier(bounding_height, bounding_width);
            correct[idx] = 1;
            seen[idx] = 2;
            error[idx] = 0.5;
            alpha[idx] = 0.5 * logf((1.0 - error[i])/error[i]);
            error_copy[i] = 0.5;
        }
    }
}

int
haarlike_pool::get_ith_best(int i)
{
    return idxs[i];
}

float
haarlike_pool::get_ith_error(int i)
{
    return error[i];
}

float
haarlike_pool::get_ith_alpha(int i)
{
    return alpha[i];
}

float*
haarlike_pool::get_ith_confidence_map(int i)
{
    return confidence_maps + (i * search_height * search_width);
}

void
haarlike_pool::print_params()
{
    for(int i = 0; i < num_classifiers; i++)
    {
        printf("Classifier #%d\n", i);
        pool[i].print_params();
        printf("Correct:%d, Seen:%d, Error:%f, Alpha:%f\n", correct[i], seen[i], error[i],
                                                            alpha[i]);
        for(int j = 0; j < search_height; j++)
        {
            for(int k = 0; k < search_width; k++)
            {
                printf("%f,", confidence_maps[i*search_height*search_width+j*search_width+k]);
            }
            printf("\n");
        }
    }
    for(int i = 0; i < num_classifiers; i++)
    {
        printf("%dth best index,error:%d,%f\n", i, idxs[i], error_copy[i]);
    }
}

