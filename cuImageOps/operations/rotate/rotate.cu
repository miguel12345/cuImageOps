#include "utils.cu"

extern "C" __global__ 
void rotate(float* image,float* out, float theta, float* _pivot, unsigned int* dims, unsigned int fillMode,unsigned int interpolationMode)
{
 size_t dstx = blockIdx.x * blockDim.x + threadIdx.x;
 size_t dsty = blockIdx.y * blockDim.y + threadIdx.y;

 unsigned int height = dims[0];
 unsigned int width = dims[1];
 unsigned int channels = dims[2];

 float2 point = make_float2(float(dstx),float(dsty));
 float2 pivot = make_float2(_pivot[0]* (width-1),_pivot[1]* (height-1));

 float2 srcPoint = rotate(make_float2(point.x - pivot.x,point.y - pivot.y),-theta);
 srcPoint = make_float2(srcPoint.x + pivot.x,srcPoint.y + pivot.y);

 if (dstx >= width || dsty >= height)
    return;

  size_t outIdx = dsty*width + dstx;

  if(channels == 4){
    float4* image4c = (float4*)&image[0];
    float4* out4c = (float4*)&out[0];
    out4c[outIdx] = sample2d<float4>(image4c,srcPoint.x,srcPoint.y,dims,fillMode,interpolationMode,make_float4(0.0f,0.0f,0.0f,0.0f));
  }
  else if(channels == 3){
    float3* image3c = (float3*)&image[0];
    float3* out3c = (float3*)&out[0];
    out3c[outIdx] = sample2d<float3>(image3c,srcPoint.x,srcPoint.y,dims,fillMode,interpolationMode,make_float3(0.0f,0.0f,0.0f));
  }
  else{
    out[outIdx] = sample2d<float>(image,srcPoint.x,srcPoint.y,dims,fillMode,interpolationMode,0.0f);
  }

  
}