#include "utils.cu"

extern "C" __global__ 
void translate(float* image, float* translate, float* out, unsigned int* dims, unsigned int fillMode,unsigned int interpolationMode)
{
 size_t dstx = blockIdx.x * blockDim.x + threadIdx.x;
 size_t dsty = blockIdx.y * blockDim.y + threadIdx.y;

 float srcx = float(dstx) - translate[0];
 float srcy = float(dsty) - translate[1];

 unsigned int height = dims[0];
 unsigned int width = dims[1];
 unsigned int channels = dims[2];

 if (dstx >= width || dsty >= height)
    return;

  size_t outIdx = dsty*width + dstx;

  sampleAndAssign(image,out,make_float2(srcx,srcy),outIdx,dims,fillMode,interpolationMode);
  
}