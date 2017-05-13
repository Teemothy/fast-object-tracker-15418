#include <stdio.h>

#include "object_tracker.h"

object_tracker::object_tracker(int bounding_box_height, int bounding_box_width,
                               int image_height, int image_width,
                               int search_box_height, int search_box_width,
                               int classifiers_count,
                               int steps_to_discard, int discard_count,
                               int best_count) :
    classifier(bounding_box_height, bounding_box_width,
               image_height, image_width,
               search_box_height, search_box_width,
               classifiers_count,
               steps_to_discard, discard_count,
               best_count)
{
    height = image_height;
    width = image_width;

    search_height = search_box_height;
    search_width = search_box_width;

    max_search_y = height - bounding_box_height - search_box_height + 1;
    max_search_x = width - bounding_box_width - search_box_width - 1;
}

void
object_tracker::initial_train(float* image,
                              int pos_y, int pos_x,
                              int* neg_y, int* neg_x, int num_neg)
{
    prev_y = pos_y;
    prev_x = pos_x;

    next_y = pos_y;
    next_x = pos_x;

    classifier.initial_train(image,
                             pos_y, pos_x,
                             neg_y, neg_x, num_neg);
}

std::pair<int,int>
object_tracker::track_frame(float* image)
{
    int to_look_y = next_y - search_height/2;
    int to_look_x = next_x - search_width/2;
    if (to_look_y < 0)
        to_look_y = 0;
    else if (to_look_y > max_search_y)
        to_look_y = max_search_y;
     if (to_look_x < 0)
        to_look_x = 0;
    else if (to_look_x > max_search_x)
        to_look_x = max_search_x;

    std::pair<int, int> coord = classifier.find_and_update(image,
                                                           to_look_y,
                                                           to_look_x);

    next_y = coord.first + (coord.first - prev_y)/2;
    next_x = coord.second + (coord.second - prev_x)/2;

    prev_y = coord.first;
    prev_x = coord.second;

    return coord;
}
