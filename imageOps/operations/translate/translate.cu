#include "utils.cu"

extern "C" __global__ 
void translate(float* image, float* translate, float* out, unsigned int* dims, unsigned int fillMode)
{
 size_t dstx = blockIdx.x * blockDim.x + threadIdx.x;
 size_t dsty = blockIdx.y * blockDim.y + threadIdx.y;

 float srcx = float(dstx) - translate[0];
 float srcy = float(dsty) - translate[1];

 unsigned int height = dims[0];
 unsigned int width = dims[1];

 if (dstx < width && dsty < height) {

   size_t outIdx = dsty*width + dstx;
   out[outIdx] = sample2d(image,srcx,srcy,dims,fillMode,INTERPOLATION_MODE_POINT);
 }
}