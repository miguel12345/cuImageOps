#include "utils.cu"

//Computes a cumulative distribution function
extern "C" __global__ void cumulative_distribution(unsigned int* cumulative_distribution_output, unsigned int* global_histogram,unsigned int num_bins,unsigned int* cumulative_distribution_min){
    cumulative_distribution_output[0] = global_histogram[0];

    for(unsigned int i = 1; i < num_bins; i++ ){
      if(cumulative_distribution_output[i-1] > 0 && cumulative_distribution_min[0] == 0){
        cumulative_distribution_min[0] = cumulative_distribution_output[i-1];
      }
      cumulative_distribution_output[i] = global_histogram[i] + cumulative_distribution_output[i-1];
    }
}


extern "C" __global__ void histogram_equalization(unsigned char* output_image, unsigned char* input_image,const unsigned int* input_image_dims, unsigned int num_bins, unsigned int* cumulative_distribution, unsigned int* cumulative_distribution_min) {

  int thread_global_x = blockIdx.x * blockDim.x + threadIdx.x;
  int thread_global_y = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int input_image_height = input_image_dims[0];
  unsigned int input_image_width = input_image_dims[1];
  unsigned int input_image_len = input_image_height * input_image_width;
  unsigned int num_channels = input_image_dims[2];

  // grid dimensions
  int num_global_threads_x = blockDim.x * gridDim.x;
  int num_global_threads_y = blockDim.y * gridDim.y;


  //For each pixel in the input image, fetch the level(s), get the its cumulative distribution value, normalize it, and convert to uint8
    for (int x = thread_global_x; x < input_image_width; x += num_global_threads_x) {
      for (int y = thread_global_y; y < input_image_height; y += num_global_threads_y) { 
        for (int c = 0; c < num_channels; c += 1) {

            //Sample image
            unsigned char level = input_image[(y*input_image_width + x)*num_channels + c];

            //Check its cumulative value
            unsigned int cumulative_val = cumulative_distribution[level];

            //Normalize it
            float normalized_cumulative_val = (float)(cumulative_val-cumulative_distribution_min[0]) / (float)(input_image_len-cumulative_distribution_min[0]);

            //Convert to uint8
            unsigned char equalized_level = (unsigned char) (__float2uint_rn(normalized_cumulative_val * (num_bins-1)));

            output_image[(y*input_image_width + x)*num_channels + c] = equalized_level;
            
        }
      }
    }
}