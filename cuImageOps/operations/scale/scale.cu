#include "utils.cu"

extern "C" __global__ 
void scale(float* image,float* out, float* scale, float* pivot, unsigned int* dims, unsigned int fillMode, unsigned int interpolationMode)
{
 size_t dstx = blockIdx.x * blockDim.x + threadIdx.x;
 size_t dsty = blockIdx.y * blockDim.y + threadIdx.y;

 unsigned int height = dims[0];
 unsigned int width = dims[1];
 unsigned int channels = dims[2];

 float pivotx = pivot[0] * (width-1);
 float pivoty = pivot[1] * (height-1);

 float srcx = ((float(dstx) - pivotx)/ scale[0]) + pivotx;
 float srcy = ((float(dsty) - pivoty)/ scale[1]) + pivoty;


 if (dstx >= width || dsty >= height)
    return;

  size_t outIdx = dsty*width + dstx;

  sampleAndAssign(image,out,make_float2(srcx,srcy),outIdx,dims,fillMode,interpolationMode);
  
}