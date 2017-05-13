#ifdef __CUDACC__
#define CUDA_DEV __device__
#else
#define CUDA_DEV
#endif

class haarlike_classifier {

private:

    int bounding_height;
    int bounding_width;

    int box_y[3];
    int box_x[3];
    int box_height[3];
    int box_width[3];
    float box_weights[3];

    float mean_pos;
    int pos_count;
    float mean_neg;

    float polarity;
    float threshold;

    public:

    CUDA_DEV  float eval(float* integral_image, int height, int width,
               int y, int x);


    CUDA_DEV haarlike_classifier(int bounding_box_height, int bounding_box_width);

    CUDA_DEV void initial_train(float* integral_image, int height, int width,
                       int pos_y, int pos_x,
                       int* neg_y, int* neg_x, int num_neg);

    CUDA_DEV void classify_region(float* integral_image, int height, int width,
                         bool* output);

    CUDA_DEV void update(float* integral_image, int height, int width,
                int pos_y, int pos_x);

    CUDA_DEV bool classify(float* integral_image, int height, int width,
                  int y, int x);

    CUDA_DEV void print_params();
};
