#include "haarlike.h"

class haarlike_pool {

private:

    int bounding_height;
    int bounding_width;

    float* integral_image_cache;
    int height;
    int width;

    int search_height;
    int search_width;

    int num_classifiers;

    haarlike_classifier* pool;
    int* correct;
    int* seen;
    float* error;
    float* alpha;

    float* confidence_maps;

    int* idxs;
    float* error_copy;

    int discard_steps;
    int num_discard;

    int steps;

public:

    haarlike_pool(int bounding_box_height, int bounding_box_width,
                  int image_height, int image_width,
                  int search_box_height, int search_box_width,
                  int classifiers_count,
                  int steps_to_discard, int discard_count);

    void initial_train(float* image,
                       int pos_y, int pos_x,
                       int* neg_y, int* neg_x, int num_neg);

    void classify_region(float* image,
                         int search_y, int search_x);

    CUDA_DEV int get_ith_best(int i);
    CUDA_DEV float get_ith_error(int i);
    CUDA_DEV float get_ith_alpha(int i);
    CUDA_DEV float* get_ith_confidence_map(int i);

    void update(int pos_y, int pos_x);

    void print_params();
};
