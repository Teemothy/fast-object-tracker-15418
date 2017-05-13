#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "haar_pool.h"
#include "integral_image.h"

__global__ void initialize_classifiers(haarlike_classifier* pool,
                                       int bounding_height, int bounding_width,
                                       int* correct, int* seen, float* error, float* alpha,
                                       int* idxs, float* error_copy)
{
    int tdx = threadIdx.x;
    pool[tdx] = haarlike_classifier(bounding_height, bounding_width);
    correct[tdx] = 2;
    seen[tdx] = 3;
    error[tdx] = 1.0/3.0;
    alpha[tdx] = 0.5 * logf((1.0 - error[tdx])/error[tdx]);
    idxs[tdx] = tdx;
    error_copy[tdx] = 1.0/3.0;
}

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
    cudaMalloc(&integral_image_cache, height * width * sizeof(float));

    search_height = search_box_height;
    search_width = search_box_width;

    num_classifiers = classifiers_count;

    cudaMalloc(&pool, num_classifiers * sizeof(haarlike_classifier));
    cudaMalloc(&correct, num_classifiers * sizeof(int));
    cudaMalloc(&seen, num_classifiers * sizeof(int));
    cudaMalloc(&error, num_classifiers * sizeof(float));
    cudaMalloc(&alpha, num_classifiers * sizeof(float));
    cudaMalloc(&idxs, num_classifiers * sizeof(int));
    cudaMalloc(&error_copy, num_classifiers * sizeof(float));
    initialize_classifiers<<<1, num_classifiers>>>(pool, bounding_height, bounding_width,
                                                   correct, seen, error, alpha,
                                                   idxs, error_copy);
    cudaDeviceSynchronize();

    cudaMalloc(&confidence_maps,
               num_classifiers * search_height * search_width * sizeof(float));
    discard_steps = steps_to_discard;
    num_discard = discard_count;

    steps = 0;
}

__global__ void initial_train_kernel(haarlike_classifier* pool,
                                     float* integral_image, int height, int width,
                                     int pos_y, int pos_x,
                                     int* neg_y, int* neg_x, int num_neg)
{
    int tdx = threadIdx.x;
    pool[tdx].initial_train(integral_image, height, width,
                            pos_y, pos_x,
                            neg_y, neg_x, num_neg);
}

void
haarlike_pool::initial_train(float* image,
                             int pos_y, int pos_x,
                             int* neg_y, int* neg_x, int num_neg)
{
    integral_image(image, integral_image_cache, height, width);
    initial_train_kernel<<<1, num_classifiers>>>(pool,
                                                 integral_image_cache, height, width,
                                                 pos_y, pos_x,
                                                 neg_y, neg_x, num_neg);
}

__global__ void classify_kernel(haarlike_classifier* pool,
                                float* integral_image, int height, int width,
                                int search_y, int search_x,
                                int search_height, int search_width,
                                float* confidence_maps)
{
    int bdx = blockIdx.x;
    int tdy = threadIdx.y;
    int tdx = threadIdx.x;


    float* confidence_map = confidence_maps+(bdx*search_height*search_width);
    confidence_map[tdy*search_width + tdx] = (float)
        pool[bdx].classify(integral_image, height, width,
                            search_y+tdy, search_x+tdx);

}

void
haarlike_pool::classify_region(float* image, int search_y, int search_x)
{
    integral_image(image, integral_image_cache, height, width);
    classify_kernel<<<num_classifiers, dim3(search_width, search_height)>>>
        (pool, integral_image_cache, height, width,
         search_y, search_x,
         search_height, search_width,
         confidence_maps);
}

__global__ void update_kernel(haarlike_classifier* pool,
                              int* correct, int* seen, float* error, float* alpha,
                              int* idxs, float* error_copy,
                              float* integral_image, int height, int width,
                              int pos_y, int pos_x)
{
    int tdx = threadIdx.x;
    bool is_correct = pool[tdx].classify(integral_image, height, width, pos_y, pos_x);
    pool[tdx].update(integral_image, height, width, pos_y, pos_x);
    if (is_correct)
        correct[tdx]++;
    seen[tdx]++;
    error[tdx] = 1.0 - ((float)correct[tdx])/seen[tdx];
    alpha[tdx] = 0.5 * logf((1.0 - error[tdx])/error[tdx]);
    idxs[tdx] = tdx;
    error_copy[tdx] = error[tdx];
}

__global__ void discard_classifiers(haarlike_classifier* pool,
                                       int bounding_height, int bounding_width,
                                       int* correct, int* seen, float* error, float* alpha,
                                       int* idxs, float* error_copy,
                                       int num_classifiers, int num_discard)
{
    int tdx = threadIdx.x;
    if (tdx >= num_classifiers - num_discard)
    {
        int idx = idxs[tdx];
        pool[idx] = haarlike_classifier(bounding_height, bounding_width);
        correct[idx] = 1;
        seen[idx] = 2;
        error[idx] = 0.5;
        alpha[idx] = 0.5 * logf((1.0 - error[tdx])/error[tdx]);
        error_copy[tdx] = 0.5;
    }
}


void
haarlike_pool::update(int pos_y, int pos_x)
{
    steps++;
    update_kernel<<<1, num_classifiers>>>(pool,
                                          correct, seen, error, alpha,
                                          idxs, error_copy,
                                          integral_image_cache, height, width,
                                          pos_y, pos_x);

    thrust::stable_sort_by_key(thrust::device, error_copy, error_copy+num_classifiers, idxs);

    if(steps % discard_steps == 0)
    {
        discard_classifiers<<<1, num_classifiers>>>(pool,
                                                    bounding_height, bounding_width,
                                                    correct, seen, error, alpha,
                                                    idxs, error_copy,
                                                    num_classifiers, num_discard);
    }
}

__device__
int
haarlike_pool::get_ith_best(int i)
{
    return idxs[i];
}

__device__
float
haarlike_pool::get_ith_error(int i)
{
    return error[i];
}

__device__
float
haarlike_pool::get_ith_alpha(int i)
{
    return alpha[i];
}

__device__
float*
haarlike_pool::get_ith_confidence_map(int i)
{
    return confidence_maps + (i * search_height * search_width);
}

__global__ void print_kernel(float* integral_image_cache, int height, int width,
                        int search_width, int search_height,
                        int num_classifiers,
                        haarlike_classifier* pool, int* correct, int* seen, float* error,
                        float* alpha, int* idxs, float* error_copy,
                        float* confidence_maps)

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

void
haarlike_pool::print_params()
{
    print_kernel<<<1,1>>>(integral_image_cache, height, width,
                          search_width, search_height,
                          num_classifiers,
                          pool, correct, seen, error,
                          alpha, idxs, error_copy,
                          confidence_maps);
    cudaDeviceSynchronize();
}
