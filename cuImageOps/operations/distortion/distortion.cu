#include "utils.cu"

extern "C" __global__ 
void distortion(unsigned char* output, unsigned char* image, float k1, float k2, float k3, float p1, float p2, unsigned int* dims, unsigned int fillMode, unsigned int interpolationMode)
{
 size_t dstx = blockIdx.x * blockDim.x + threadIdx.x;
 size_t dsty = blockIdx.y * blockDim.y + threadIdx.y;

 unsigned int height = dims[0];
 unsigned int width = dims[1];

 if (dstx >= width || dsty >= height)
    return;

  size_t outIdx = dsty*width + dstx;

  float xcn = (float)dstx/(float)width - 0.5f;
  float ycn = (float)dsty/(float)height - 0.5f;
  float r_2 = xcn * xcn + ycn * ycn;
  float r_4 = r_2 * r_2;
  float r_6 = r_4 * r_2;

  float xcn_d = (xcn * (1 + k1*(r_2) + k2*r_4 + k3*r_6)) + 2*p1*xcn*ycn + p2*(r_2 + 2*(xcn*xcn));
  float ycn_d = (ycn * (1 + k1*(r_2) + k2*r_4 + k3*r_6)) + p1*(r_2 + 2*(ycn*ycn)) + 2*p2*xcn*ycn;
  
  float srcx = (xcn_d + 0.5f) * width;
  float srcy = (ycn_d + 0.5f) * height;

  sampleAndAssign_uchar(image,output,make_float2(srcx,srcy),outIdx,dims,fillMode,interpolationMode);
  
}