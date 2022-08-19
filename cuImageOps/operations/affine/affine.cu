#include "utils.cu"

extern "C" __global__ 
void affine(float* image,float* out, float* matrix, unsigned int* dims, unsigned int fillMode, unsigned int interpolationMode)
{
 size_t dstx = blockIdx.x * blockDim.x + threadIdx.x;
 size_t dsty = blockIdx.y * blockDim.y + threadIdx.y;

 unsigned int height = dims[0];
 unsigned int width = dims[1];
 unsigned int channels = dims[2];

 // multiply matrix by point
 float srcx = matrix[0]* dstx + matrix[1]* dsty + matrix[2];
 float srcy = matrix[3]* dstx + matrix[4]* dsty + matrix[5];


 if (dstx >= width || dsty >= height)
    return;

  size_t outIdx = dsty*width + dstx;

  if(channels == 4){
    float4* image4c = (float4*)&image[0];
    float4* out4c = (float4*)&out[0];
    out4c[outIdx] = sample2d<float4>(image4c,srcx,srcy,dims,fillMode,interpolationMode,make_float4(0.0f,0.0f,0.0f,0.0f));
  }
  else if(channels == 3){
    float3* image3c = (float3*)&image[0];
    float3* out3c = (float3*)&out[0];
    out3c[outIdx] = sample2d<float3>(image3c,srcx,srcy,dims,fillMode,interpolationMode,make_float3(0.0f,0.0f,0.0f));
  }
  else{
    out[outIdx] = sample2d<float>(image,srcx,srcy,dims,fillMode,interpolationMode,0.0f);
  }

  
}