#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

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
    cudaMalloc(&confidence_map, search_height * search_width * sizeof(float));

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

__global__ void combine_maps(haarlike_pool haar_pool,
                             int num_best,
                             float* output_map, int search_height, int search_width)
{
    int x = threadIdx.x;
    int y = threadIdx.y;

    float local_confidence = 0.0;

    for (int i = 0; i < num_best; i++)
    {
        int index = haar_pool.get_ith_best(i);
        local_confidence += haar_pool.get_ith_alpha(index) *
                            haar_pool.get_ith_confidence_map(index)[y * search_width + x];
    }

    output_map[y * search_width + x] = local_confidence;
}


std::pair<int,int>
strong_classifier::find_and_update(float* image,
                                   int search_y, int search_x)
{
    haar_pool.classify_region(image,
                              search_y, search_x);

    combine_maps<<<1, dim3(search_width, search_height)>>>
        (haar_pool,
         num_best,
         confidence_map, search_height, search_width);

    int index = ((int)(thrust::max_element(thrust::device,
                                    confidence_map,
                                    confidence_map + (search_height * search_width)) -
                  confidence_map));

    std::pair<int, int> result (index/search_width+search_y, index%search_width+search_x);

    haar_pool.update(result.first, result.second);

    return result;
}

__global__ void print_kernel(float* confidence_map, int height, int width)
{
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            printf("%f,",confidence_map[i*height + j]);
        }
        printf("\n");
    }
}
void
strong_classifier::print_confidence_map()
{
    print_kernel<<<1,1>>>(confidence_map, search_height, search_width);
    cudaDeviceSynchronize();
}
