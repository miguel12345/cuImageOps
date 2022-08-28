#include "utils.cu"

extern "C" __global__ 
void affine(unsigned char* output, unsigned char* image, float* matrix, unsigned int* dims, unsigned int fillMode, unsigned int interpolationMode)
{
 size_t dstx = blockIdx.x * blockDim.x + threadIdx.x;
 size_t dsty = blockIdx.y * blockDim.y + threadIdx.y;

 unsigned int height = dims[0];
 unsigned int width = dims[1];

 // multiply matrix by point
 float srcx = matrix[0]* dstx + matrix[1]* dsty + matrix[2];
 float srcy = matrix[3]* dstx + matrix[4]* dsty + matrix[5];


 if (dstx >= width || dsty >= height)
    return;

  size_t outIdx = dsty*width + dstx;

  sampleAndAssign_uchar(image,output,make_float2(srcx,srcy),outIdx,dims,fillMode,interpolationMode);
  
}