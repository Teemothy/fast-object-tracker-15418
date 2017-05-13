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
    for(int j = 0; j < width; j++)
        output[j] = 0.0;
    for(int i = 0; i < height; i++)
        output[i*width] = 0.0;

    for (int i = 1; i < height; i++)
    {
        for(int j = 1; j < width; j++)
        {
            output[i*width + j] = output[(i-1)*width + j] + output[i*width + (j-1)] -
                                  output[(i-1)*width + (j-1)] + input[(i-1)*width + (j-1)];
        }
    }
}
