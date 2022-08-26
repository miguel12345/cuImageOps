#include "utils.cu"

#define NUM_BINS 256

template<unsigned char num_channels> __device__ void _partial_histogram(unsigned int* partial_histograms,const unsigned char* input_image, const unsigned int* input_image_dims)
{

  size_t thread_global_x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t thread_global_y = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int input_image_height = input_image_dims[0];
  unsigned int input_image_width = input_image_dims[1];

  // grid dimensions
  int num_global_threads_x = blockDim.x * gridDim.x; 
  int num_global_threads_y = blockDim.y * gridDim.y;


  // linear thread index within 2D block
  int block_thread_idx = threadIdx.x + threadIdx.y * blockDim.x; 

  // total threads in 2D block
  int num_threads_in_block = blockDim.x * blockDim.y; 

  // linear block index within 2D grid
  unsigned int global_block_index = blockIdx.x + blockIdx.y * gridDim.x;

  __shared__ unsigned int smem[NUM_BINS * num_channels];
  for (int i = block_thread_idx * num_channels; i < NUM_BINS; i += num_threads_in_block) {
    for (int c = 0; c < num_channels; c += 1) {
      smem[i + c] = 0;
    }
  }
  __syncthreads();

  for (size_t x = thread_global_x; x < input_image_width; x += num_global_threads_x) {
    for (size_t y = thread_global_y; y < input_image_height; y += num_global_threads_y) { 
      for (int c = 0; c < num_channels; c += 1) {
          //Sample image and convert float to uint
          unsigned char val = input_image[(y*input_image_width + x)*num_channels + c];

          //Increment the counter for val
          atomicAdd(&smem[val*num_channels + c], 1);
      }
    }
}

  __syncthreads();

  //Write block counters to output
  unsigned int* block_partial_histogram = partial_histograms + (global_block_index * NUM_BINS * num_channels);


  for (unsigned int i = block_thread_idx; i < NUM_BINS ; i += num_threads_in_block) {
    for (int c = 0; c < num_channels; c += 1) {
      block_partial_histogram[i * num_channels + c] = smem[i * num_channels + c];
    }
  }

}

extern "C" __global__ void partial_histogram(unsigned int* partial_histograms,const unsigned char* input_image, const unsigned int* input_image_dims, unsigned char num_channels) {

  if(num_channels == 1){
    _partial_histogram<1>(partial_histograms,input_image,input_image_dims);
  }
  else if(num_channels == 3){
    _partial_histogram<3>(partial_histograms,input_image,input_image_dims);
  }

}

extern "C" __global__ void global_histogram(unsigned int* global_histogram,unsigned int* partial_histograms, unsigned int num_partial_histograms, unsigned char num_channels){

    unsigned int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(global_thread_idx >= (num_partial_histograms* NUM_BINS * num_channels))
      return;

    for(unsigned int i = 0; i < num_partial_histograms; i++ ){
      for (int c = 0; c < num_channels; c += 1) {
        global_histogram[global_thread_idx*num_channels + c] += partial_histograms[i * NUM_BINS* num_channels + global_thread_idx*num_channels + c];
      }
    }


}