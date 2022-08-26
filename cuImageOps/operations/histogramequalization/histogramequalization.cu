#include "utils.cu"

//Computes a cumulative distribution function
extern "C" __global__ void cumulative_distribution(unsigned int* cumulative_distribution_output, unsigned int* global_histogram,unsigned int num_bins){
    cumulative_distribution_output[0] = global_histogram[0]

    for(unsigned int i = 1; i < num_bins; i++ ){
      cumulative_distribution_output[i] = global_histogram[i] + cumulative_distribution_output[i+1]
    }
}


extern "C" __global__ void histogram_equalization(unsigned char* output_image, unsigned char* input_image,const unsigned int* input_image_dims, unsigned char num_bins, unsigned int* cumulative_distribution) {

  size_t thread_global_x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t thread_global_y = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int input_image_height = input_image_dims[0];
  unsigned int input_image_width = input_image_dims[1];
  unsigned int num_channels = input_image_dims[2];

  // grid dimensions
  int num_global_threads_x = blockDim.x * gridDim.x; 
  int num_global_threads_y = blockDim.y * gridDim.y;

  //For each pixel in the input image, fetch the level(s), get the its cumulative distribution value, normalize it, and convert to uint8

    for (size_t x = thread_global_x; x < input_image_width; x += num_global_threads_x) {
    for (size_t y = thread_global_y; y < input_image_height; y += num_global_threads_y) { 
      for (int c = 0; c < num_channels; c += 1) {

          //Sample image
          unsigned char level = input_image[(y*input_image_width + x)*num_channels + c];

          //Check its cumulative value
          unsigned int cumulative_val = cumulative_distribution[level]

          //Normalize it
          float normalized_cumulative_val = (float)cumulative_val / (float)num_bins;

          //Convert to uint8
          unsigned char equalized_level = (unsigned char) __float2uint_rd(normalized_cumulative_val * (num_bins-1));
      }
    }
}
}