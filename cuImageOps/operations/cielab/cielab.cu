#include "utils.cu"


template<typename data_type> __device__ data_type bgr2CIELab(data_type rgb);


inline __device__ float CIELabf(float input){
  if(input > 0.008856f){
    return pow(input,1.f/3.f);
  }
  
  return (7.787f * input + 16) / 116.f;
}

template<> __device__ float3 bgr2CIELab<float3>(float3 bgr) {

  float m00 = 0.412453f;
  float m01 = 0.357580f;
  float m02 = 0.180423f;
  float m10 = 0.212671f;
  float m11 = 0.715160f;
  float m12 = 0.072169f;
  float m20 = 0.019334f;
  float m21 = 0.119193f;
  float m22 = 0.950227f;

  float b = bgr.x;
  float g = bgr.y;
  float r = bgr.z;

  float x = r*m00 + g*m01 + r*m02;
  float y = r*m10 + g*m11 + r*m12;
  float z = r*m20 + g*m21 + r*m22;

  x = x / 0.950456f;
  z = z / 1.088754f;

  float l = 0f;

  if(y > 0.008856f){
    l = 116 * pow(y,1.f/3.f) - 16;
  }
  else{
    l = 903.3 * y;
  }

  float a = 500 * (CIELabf(x) - CIELabf(y));
  float b = 200 * (CIELabf(y) - CIELabf(z));

  return make_float3(l,a,b);

}


template<>__device__ uchar3 bgr2CIELab<uchar3>(uchar3 bgr) {

  float3 bgr_f = make_float3(bgr.x / 255.f,bgr.y / 255.f,bgr.z / 255.f);

  float3 lab_f = bgr2CIELab<float3(bgr_f);

  uchar3 lab = make_uchar3( 
    __float2uint_rn((lab_f.x/100.f) * 255),
    lab_f.y + 128f,
    lab_f.y + 128f);

}

extern "C" __global__ 
void CIELab(unsigned char* output_image, unsigned char* input_image, unsigned int* input_image_dims)
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