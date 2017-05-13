#include <cstdlib>

#include "strong_classifier.h"

strong_classifier::strong_classifier(int bounding_box_height, int bounding_box_width,
                                     int image_height, int image_width,
                                     int search_box_height, int search_box_width,
                                     int classifiers_count,
                                     int steps_to_discard, int discard_count,
                                     int best_count) :
    haar_pool(bounding_box_height, bounding_box_width,
              image_height, image_width,
              search_box_height, search_box_width,
              classifiers_count,
              steps_to_discard, discard_count)
{
    search_height = search_box_height;
    search_width = search_box_width;
    confidence_map = (float*)malloc(search_height * search_width * sizeof(float));

    num_best = best_count;
}

void
strong_classifier::initial_train(float* image,
                                 int pos_y, int pos_x,
                                 int* neg_y, int* neg_x, int num_neg)
{
    haar_pool.initial_train(image,
                            pos_y, pos_x,
                            neg_y, neg_x, num_neg);
}

std::pair<int,int>
strong_classifier::find_and_update(float* image,
                                   int search_y, int search_x)
{
    haar_pool.classify_region(image,
                              search_y, search_x);

    int max_y = 0;
    int max_x = 0;
    float max_conf = 0.0;

    for(int y = 0; y < search_height; y++)
    {
        for(int x = 0; x < search_width; x++)
        {
            float local_confidence = 0.0;
            for(int i = 0; i < num_best; i++)
            {
                int index = haar_pool.get_ith_best(i);
                local_confidence += haar_pool.get_ith_alpha(index) *
                                    haar_pool.get_ith_confidence_map(index)[y*search_width+x];
            }
            if (local_confidence > max_conf)
            {
                max_y = y;
                max_x = x;
                max_conf = local_confidence;
            }
        }
    }

    std::pair<int, int> result (max_y+search_y, max_x+search_x);

    haar_pool.update(result.first, result.second);

    return result;
}

void
strong_classifier::print_confidence_map()
{
}
