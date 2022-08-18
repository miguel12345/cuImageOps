#define FILL_MODE_CONSTANT 1
#define FILL_MODE_REFLECTION 2

#define INTERPOLATION_MODE_POINT 1
#define INTERPOLATION_MODE_LINEAR 2

inline __device__ float3 operator*(float3 a,float b) {
    return make_float3(a.x*b,a.y*b,a.z*b);
}

inline __device__ float3 operator+(float3 a,float3 b) {
    return make_float3(a.x+b.x,a.y+b.y,a.z+b.z);
}

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

template<typename type> __device__ type pointsample2d(type* image, float x, float y, unsigned int* dims, unsigned int fillMode,type fillConstant)
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


template<typename type> __device__ type bilinearsample2d(type* image, float x, float y, unsigned int* dims, unsigned int fillMode, type fillConstant)
{

    

    //Determine the four corners

    float2 tl = make_float2(floorf(x),floorf(y));
    float2 tr = make_float2(floorf(x+1),floorf(y));
    float2 bl = make_float2(floorf(x),floorf(y+1));
    float2 br = make_float2(floorf(x+1),floorf(y+1));

    //Sample the four corners
    
    type tlval = pointsample2d(image,tl.x,tl.y,dims,fillMode,fillConstant);
    type trval = pointsample2d(image,tr.x,tr.y,dims,fillMode,fillConstant);
    type blval = pointsample2d(image,bl.x,bl.y,dims,fillMode,fillConstant);
    type brval = pointsample2d(image,br.x,br.y,dims,fillMode,fillConstant);

    float area = (tr.x-bl.x)*(bl.y-tl.y);

    //Calculate interpolation weights

    float wtl = ((br.x-x)*(br.y-y))/area;
    float wtr = ((x-bl.x)*(bl.y-y))/area;
    float wbl = ((tr.x-x)*(y-tr.y))/area;
    float wbr = ((x-tl.x)*(y-tl.y))/area;

    //Return interpolated result
    return tlval*wtl + trval*wtr + blval*wbl + brval*wbr;

}

template<typename type> __device__ type sample2d(type* image, float x, float y, unsigned int* dims, unsigned int fillMode,unsigned int interpolationMode, type fillConstant) {

    if(interpolationMode == INTERPOLATION_MODE_POINT) {
        return pointsample2d(image,x,y,dims,fillMode,fillConstant);
    }
    else if(interpolationMode == INTERPOLATION_MODE_LINEAR) {
        return bilinearsample2d(image,x,y,dims,fillMode,fillConstant);
    }
}