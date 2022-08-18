#define FILL_MODE_CONSTANT 1
#define FILL_MODE_REFLECTION 2

#define INTERPOLATION_MODE_POINT 1

__device__ float radians(float a)
{
  return 0.017453292 * a;
}

__device__ float2 rotate(float2 point, float theta){
    float thetarad = radians(theta);
    float thetasin = sin(thetarad);
    float thetacos = cos(thetarad);

    float rotX = point.x * thetacos - point.y * thetasin;
    float rotY = point.y * thetacos + point.x * thetasin;

    return make_float2(rotX,rotY);
}

template<typename type> __device__ type sample2d(type* image, float x, float y, unsigned int* dims, unsigned int fillMode, unsigned int interpolationMode, type fillConstant)
{

    unsigned int height = dims[0];
    unsigned int width = dims[1];

    int xInt = __float2int_rn(x);
    int yInt = __float2int_rn(y);


    if(xInt < 0 || xInt >= width || yInt < 0 || yInt >= height) {
        if (fillMode == FILL_MODE_CONSTANT) {
            return fillConstant;
        }
        else if (fillMode == FILL_MODE_REFLECTION)
        {
            if(xInt < 0) {
                xInt = -xInt;
                xInt = (xInt-1)%width;
            }
            else if (xInt >= width)
            {
                xInt = (xInt-1)%width;
                xInt = (width-1) - xInt;
            }
            
            if(yInt < 0) {
                yInt = -yInt;
                yInt = (yInt-1)%height;
                yInt = (height-1) - yInt;
            }
            else if (yInt >= height)
            {
                yInt = (yInt-1)%height;
            }
        }
    }
    
    return image[yInt*width + xInt];
}