#include <stdio.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#define BLOCK_DIM 16
#define UPDIV(n, d) (((n)+(d)-1)/(d))

/*
 * horizontal_exclusive_scan: exclusive scans a 2d array horizontally
 * REQUIRES: threads per block == width/2
 * REQUIRES: num blocks == height
 * REQUIRES: input, output have size height x width
 * REQUIRES: 2 * threads_per_block  * sizeof(float) shared memory
 *
 * TODO: Currently, this function assigns one row per block which may
 * be inefficient. Also this has a max width of 2048 based on 1024 max
 * on threads per block. Odd number width may also be accomodated.
 *
 * Code adapted from MIT paper
 */
__global__ void horizontal_exclusive_scan(float* input, float* output,
                                          int height, int width)
{
    extern __shared__ float temp[];
    int tdx = threadIdx.x;
    int bdx = blockIdx.x;
    int offset = 1;

    temp[2*tdx] = input[bdx*width + 2*tdx];
    temp[2*tdx+1] = input[bdx*width + 2*tdx+1];

    for(int d = width>>1; d > 0; d >>= 1)
    {
        __syncthreads();
        if(tdx < d)
        {
            int ai = offset*(2*tdx+1)-1;
            int bi = offset*(2*tdx+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if(tdx == 0) temp[width - 1] = 0;

    for(int d = 1; d < width; d *= 2)
    {

        offset >>= 1;
        __syncthreads();

        if(tdx < d)
        {
            int ai = offset*(2*tdx+1)-1;
            int bi = offset*(2*tdx+2)-1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
        __syncthreads();
    }

    output[bdx*width + 2*tdx] = temp[2*tdx];
    output[bdx*width + 2*tdx+1] = temp[2*tdx+1];
}

/*
  transpone: transposes a 2d matrix
 * REQUIRES: block dimensions that cover the input matrix
 *
 * Code borrowed from MIT paper
 */
__global__ void transpose(float* input, float* output, int height, int width)
{
    int y = blockIdx.y * BLOCK_DIM + threadIdx.y;
    int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
    if(y < height && x < width)
        output[x * height + y] = input[y * width + x];

}

/*
 * integral_image: computes the exclusive integral image of input
 * REQUIRES: input, output on device memory and have size height x width
 * REQUIRES: width % 2 == 0
 *
 * IMPORTANT: destructive on input
 *
 * TODO: Currently uses a block for each row, but this may be improved for
 * performance.
 * TODO: May need to change input, output to be on host memory
 */
void integral_image(float* input, float* output, int height, int width)
{
    int size1 = width * sizeof(float);
    int size2 = height * sizeof(float);
    horizontal_exclusive_scan<<<height, width/2, size1>>>(input, input, height, width);
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid1(UPDIV(width, BLOCK_DIM), UPDIV(height, BLOCK_DIM));
    transpose<<<dimGrid1, dimBlock>>>(input, output, height, width);
    horizontal_exclusive_scan<<<width, height/2, size2>>>(output, input, width, height);
    dim3 dimGrid2(UPDIV(height, BLOCK_DIM), UPDIV(width, BLOCK_DIM));
    transpose<<<dimGrid2, dimBlock>>>(input, output, width, height);
}

/*
 * integral_image_loc: computes the exclusive integral image of input
 * REQUIRES: input, output on host memory and have size height x width
 * REQUIRES: width % 2 == 0
 */
void integral_image_host(float* input, float* output, int height, int width)
{
    int num = height * width;
    int size = num * sizeof(float);
    float* cudaInput;
    float* cudaOutput;
    cudaMalloc(&cudaInput, size);
    cudaMalloc(&cudaOutput, size);
    cudaMemcpy(cudaInput, input, size, cudaMemcpyHostToDevice);
    integral_image(cudaInput, cudaOutput, height, width);
    cudaMemcpy(output, cudaOutput, size, cudaMemcpyDeviceToHost);
}
