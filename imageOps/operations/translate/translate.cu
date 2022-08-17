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
 unsigned int channels = dims[2];

 if (dstx >= width || dsty >= height)
    return;

  size_t outIdx = dsty*width + dstx;

  if(channels == 3){
    float3* image3c = (float3*)&image[0];
    float3* out3c = (float3*)&out[0];
    out3c[outIdx] = sample2d<float3>(image3c,srcx,srcy,dims,fillMode,INTERPOLATION_MODE_POINT,make_float3(0.0f,0.0f,0.0f));
  }
  else{
    out[outIdx] = sample2d<float>(image,srcx,srcy,dims,fillMode,INTERPOLATION_MODE_POINT,0.0f);
  }

  
}