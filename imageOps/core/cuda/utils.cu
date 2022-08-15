#define FILL_MODE_CONSTANT 1
#define FILL_MODE_REFLECTION 2

#define INTERPOLATION_MODE_POINT 1

__device__ float sample2d(float* image, float x, float y, unsigned int* dims, unsigned int fillMode, unsigned int interpolationMode)
{

    unsigned int height = dims[0];
    unsigned int width = dims[1];

    int xInt = __float2int_rn(x);
    int yInt = __float2int_rn(y);


    if(xInt < 0 || xInt >= width || yInt < 0 || yInt >= height) {
        if (fillMode == FILL_MODE_CONSTANT) {
            return 0.0;
        }
        else if (fillMode == FILL_MODE_REFLECTION)
        {
            if(xInt < 0) {
                xInt = -xInt;
                xInt = xInt%width;
            }
            else if (xInt >= width)
            {
                xInt = xInt%width;
            }
            
            if(yInt < 0) {
                yInt = -yInt;
                yInt = yInt%height;
                yInt = (height-1) - yInt;
            }
            else if (yInt >= height)
            {
                yInt = yInt%height;
                yInt = (height-1) - yInt;
            }
        }
    }
    
    return image[yInt*width + xInt];
}