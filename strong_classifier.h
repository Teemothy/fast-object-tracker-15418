#include <utility>

#include "haar_pool.h"

class strong_classifier {

private:

    haarlike_pool haar_pool;

    int search_height;
    int search_width;
    float* confidence_map;

    int num_best;

public:

    strong_classifier(int bounding_box_height, int bounding_box_width,
                      int image_height, int image_width,
                      int search_box_height, int search_box_width,
                      int classifiers_count,
                      int steps_to_discard, int discard_count,
                      int best_count);

    void initial_train(float* image,
                       int pos_y, int pos_x,
                       int* neg_y, int* neg_x, int num_neg);

    std::pair<int,int> find_and_update(float* image,
                                       int search_y, int search_x);

    void print_confidence_map();
};
