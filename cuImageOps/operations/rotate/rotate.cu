#include "utils.cu"

extern "C" __global__ 
void rotate(float* output, float* input, float theta, float* _pivot, unsigned int* dims, unsigned int fillMode,unsigned int interpolationMode)
{
 size_t dstx = blockIdx.x * blockDim.x + threadIdx.x;
 size_t dsty = blockIdx.y * blockDim.y + threadIdx.y;

 unsigned int height = dims[0];
 unsigned int width = dims[1];

 float2 point = make_float2(float(dstx),float(dsty));
 float2 pivot = make_float2(_pivot[0]* (width-1),_pivot[1]* (height-1));

 float2 srcPoint = rotate(make_float2(point.x - pivot.x,point.y - pivot.y),-theta);
 srcPoint = make_float2(srcPoint.x + pivot.x,srcPoint.y + pivot.y);

 if (dstx >= width || dsty >= height)
    return;

  size_t outIdx = dsty*width + dstx;

  sampleAndAssign(input,output,srcPoint,outIdx,dims,fillMode,interpolationMode);
  
}