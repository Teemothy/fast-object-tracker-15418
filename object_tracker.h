#include <utility>

#include "strong_classifier.h"

class object_tracker {

private:

    strong_classifier classifier;

    int height;
    int width;

    int search_height;
    int search_width;

    int max_search_y;
    int max_search_x;

    int prev_y;
    int prev_x;

    int next_y;
    int next_x;

public:

    object_tracker(int bounding_box_height, int bounding_box_width,
                   int image_height, int image_width,
                   int search_box_height, int search_box_width,
                   int classifiers_count,
                   int steps_to_discard, int discard_count,
                   int best_count);

    void initial_train(float* image,
                       int pos_y, int pos_x,
                       int* neg_y, int* neg_x, int num_neg);

    std::pair<int,int> track_frame(float* image);
};
