#include "utils.cu"

#define KERNEL_2D 1
#define KERNEL_1D_HORIZONTAL 2
#define KERNEL_1D_VERTICAL 3

template<typename input_type,typename output_type, typename accum_type> __device__ void kernelSampleAndAssign(input_type* input, output_type* output,unsigned int* dims,uint2 dstPoint, float* kernel, unsigned int kernelSize,unsigned int kernelType, unsigned int fillMode,unsigned int interpolationMode, input_type fillConstant) {

  int halfKernelSize = (kernelSize-1)/2;
  unsigned int kernelCellIdx = 0;

  unsigned int width = dims[1];

  unsigned int outputIdx = dstPoint.y*width + dstPoint.x;

  accum_type aggregatedVal = accum_type();


  if(kernelType == KERNEL_2D) {
    for(int kernelCellX = -halfKernelSize; kernelCellX <= halfKernelSize; kernelCellX++){
        for(int kernelCellY = -halfKernelSize; kernelCellY <= halfKernelSize; kernelCellY++){
            
            float kernelWeight = kernel[kernelCellIdx];

            input_type kernelCellVal = sample2d(input, (float)((int)dstPoint.x + kernelCellX), (float)((int)dstPoint.y + kernelCellY),dims,fillMode,interpolationMode,fillConstant);

            aggregatedVal = aggregatedVal + kernelCellVal * kernelWeight;

            kernelCellIdx += 1;
        }
      }
  }
  else if(kernelType == KERNEL_1D_HORIZONTAL) {
    for(int kernelCellX = -halfKernelSize; kernelCellX <= halfKernelSize; kernelCellX++){
          float kernelWeight = kernel[kernelCellIdx];
          input_type kernelCellVal = sample2d(input, (float)((int)dstPoint.x + kernelCellX), (float)dstPoint.y,dims,fillMode,interpolationMode,fillConstant);
          aggregatedVal = aggregatedVal + kernelCellVal * kernelWeight;
          kernelCellIdx += 1;
      }
  }
  else if(kernelType == KERNEL_1D_VERTICAL) {
    for(int kernelCellY = -halfKernelSize; kernelCellY <= halfKernelSize; kernelCellY++){
          float kernelWeight = kernel[kernelCellIdx];
          input_type kernelCellVal = sample2d(input, (float)dstPoint.x, (float)((int)dstPoint.y + kernelCellY),dims,fillMode,interpolationMode,fillConstant);
          aggregatedVal = aggregatedVal + kernelCellVal * kernelWeight;
          kernelCellIdx += 1;
      }
  }
  

  output[outputIdx] = convert<accum_type,output_type>(aggregatedVal);

}

extern "C" __global__ 
void blur(unsigned char* output, unsigned char* input, float* kernel, unsigned int kernelSize, unsigned int* dims, unsigned int fillMode,unsigned int interpolationMode)
{
  size_t dstx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t dsty = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int height = dims[0];
  unsigned int width = dims[1];

  if (dstx >= width || dsty >= height)
      return;

  uint2 dstPoint = make_uint2(dstx,dsty);

  unsigned int channels = dims[2];


  if(channels == 1){
    kernelSampleAndAssign<unsigned char,unsigned char, float>(input,output,dims,dstPoint,kernel,kernelSize,KERNEL_2D,fillMode,interpolationMode,0);
  }
  else if(channels == 3){
    
    uchar3* input3c = (uchar3*)&input[0];
    uchar3* output3c = (uchar3*)&output[0];

    kernelSampleAndAssign<uchar3,uchar3, float3>(input3c,output3c,dims,dstPoint,kernel,kernelSize,KERNEL_2D,fillMode,interpolationMode,make_uchar3(0,0,0));
  }
  else if(channels == 4){

    uchar4* input4c = (uchar4*)&input[0];
    uchar4* output4c = (uchar4*)&output[0];

    kernelSampleAndAssign<uchar4,uchar4, float4>(input4c,output4c,dims,dstPoint,kernel,kernelSize,KERNEL_2D,fillMode,interpolationMode,make_uchar4(0,0,0,0));
    
  }
  
}

extern "C" __global__ void blurHorizontal(float* output,unsigned char* input, float* kernel, unsigned int kernelSize, unsigned int* dims, unsigned int fillMode,unsigned int interpolationMode)
{
  size_t dstx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t dsty = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int height = dims[0];
  unsigned int width = dims[1];

  if (dstx >= width || dsty >= height)
      return;

  uint2 dstPoint = make_uint2(dstx,dsty);

  unsigned int channels = dims[2];


  if(channels == 1){
    kernelSampleAndAssign<unsigned char,float,float>(input,output,dims,dstPoint,kernel,kernelSize,KERNEL_1D_HORIZONTAL,fillMode,interpolationMode,0);
  }
  else if(channels == 3){
    
    uchar3* input3c = (uchar3*)&input[0];
    float3* output3c = (float3*)&output[0];

    kernelSampleAndAssign<uchar3,float3,float3>(input3c,output3c,dims,dstPoint,kernel,kernelSize,KERNEL_1D_HORIZONTAL,fillMode,interpolationMode,make_uchar3(0,0,0));
  }
  else if(channels == 4){

    uchar4* input4c = (uchar4*)&input[0];
    float4* output4c = (float4*)&output[0];

    kernelSampleAndAssign<uchar4,float4,float4>(input4c,output4c,dims,dstPoint,kernel,kernelSize,KERNEL_1D_HORIZONTAL,fillMode,interpolationMode,make_uchar4(0,0,0,0));
    
  }
}

extern "C" __global__ void blurVertical(unsigned char* output,float* input, float* kernel, unsigned int kernelSize, unsigned int* dims, unsigned int fillMode,unsigned int interpolationMode)
{
  size_t dstx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t dsty = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int height = dims[0];
  unsigned int width = dims[1];

  if (dstx >= width || dsty >= height)
      return;

  uint2 dstPoint = make_uint2(dstx,dsty);

  unsigned int channels = dims[2];


  if(channels == 1){
    kernelSampleAndAssign<float,unsigned char,float>(input,output,dims,dstPoint,kernel,kernelSize,KERNEL_1D_VERTICAL,fillMode,interpolationMode,0);
  }
  else if(channels == 3){
    
    float3* input3c = (float3*)&input[0];
    uchar3* output3c = (uchar3*)&output[0];

    kernelSampleAndAssign<float3,uchar3,float3>(input3c,output3c,dims,dstPoint,kernel,kernelSize,KERNEL_1D_VERTICAL,fillMode,interpolationMode,make_float3(0,0,0));
  }
  else if(channels == 4){

    float4* input4c = (float4*)&input[0];
    uchar4* output4c = (uchar4*)&output[0];

    kernelSampleAndAssign<float4,uchar4,float4>(input4c,output4c,dims,dstPoint,kernel,kernelSize,KERNEL_1D_VERTICAL,fillMode,interpolationMode,make_float4(0,0,0,0));
    
  }
}