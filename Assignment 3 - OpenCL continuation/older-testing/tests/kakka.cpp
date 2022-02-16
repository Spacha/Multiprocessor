#define CL_USE_DEPRECATED_OPENCL_2_0_APIS // required for using olf APIs

#include "lodepng.h"

#include <windows.h>
#include <CL/cl.hpp>
#include <iostream>
#include <stdio.h>
#include <math.h>

using namespace cl;

////////////////////////////////////////////////////////////////////////////////

struct pixel_s
{
    unsigned char red;
    unsigned char green;
    unsigned char blue;
    unsigned char alpha;
};
typedef pixel_s pixel_t;


/**
 * This is my wrapper for lodepng.h that simplifies the handling of PNGs a lot.
 * The object contains image metadata and offers
 * functions for manipulating the image.
 **/
class Image
{
public:
    std::vector<unsigned char> image;   // image pixels (RGBA)
    std::string name;                   // image file name
    unsigned int height;                // image height
    unsigned int width;                 // image width
    LodePNGColorType colorType;         // color type (see LodePNGColorType in lodepng.h)

    /**
     * Initializes the object and sets color type
     * E.g. LCT_RGBA, LCT_GREY
     **/
    Image(LodePNGColorType colorType = LCT_RGBA)
    {
        this->colorType = colorType;
    }

    /**
     * Creates an empty image of given size.
     **/
    void createEmpty(size_t width, size_t height)
    {
        this->width = width;
        this->height = height;

        this->image.clear();
        this->image.reserve(4 * this->width * this->height);
    }

    /**
     * Load PNG file from disk to memory first, then decode to raw pixels in memory.
     * Returns true on success, false on fail.
     **/
    bool load(const std::string &filename)
    {
        this->name = filename;

        unsigned err;
        std::vector<unsigned char> png;

        // load and decode
        cout << "Loading image...";
        err = lodepng::load_file(png, filename);
        cout << " Done." << endl;

        if (err)
        {
            cout << "Image load error " << err << ": " << lodepng_error_text(err) << endl;
            return false;
        }

        cout << "Decoding image...";
        err = lodepng::decode(this->image, this->width, this->height, png);
        cout << " Done." << endl;

        // the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA...

        if (err)
        {
            cout << "Decode error " << err << ": " << lodepng_error_text(err) << endl;
            return false;
        }

        return true;
    }

    /**
     * Encode PNG and save it to the disk.
     * Returns true on success, false on fail.
     **/
    bool save(const std::string &filename)
    {
        cout << this->image.size() << endl;

        unsigned err;
        std::vector<unsigned char> png;

        cout << "Encoding image...";
        err = lodepng::encode(png, this->image, this->width, this->height);
        cout << " Done." << endl;

        if (err) {
            cout << "Encode error " << err << ": "<< lodepng_error_text(err) << endl;
            return false;
        }

        cout << "Saving image...";
        lodepng::save_file(png, filename);
        cout << " Done." << endl;

        if (err) {
            cout << "Image save error " << err << ": " << lodepng_error_text(err) << endl;
            return false;
        }

        return true;
    }

    /**
     * Puts given RGBA (4 channel) pixel to (x,y).
     **/
    void putPixel(unsigned int x, unsigned int y, pixel_t pixel)
    {
        if (y > (this->height-1) || x > (this->width-1))
            throw;

        unsigned int i = 4*(y*width + x);
        this->image[i]   = pixel.red;
        this->image[i+1] = pixel.green;
        this->image[i+2] = pixel.blue;
        this->image[i+3] = pixel.alpha;
    }

    /**
     * Puts given grayscale (1 channel) pixel to (x,y).
     **/
    void putPixel(unsigned int x, unsigned int y, unsigned char grey)
    {
        pixel_t pixel = { grey, grey, grey, 0xff};
        this->putPixel(x, y, pixel);
    }

    /**
     * Returns a pixel struct containing the color values of each pixel in (x,y).
     **/
    pixel_t getPixel(unsigned int x, unsigned int y)
    {
        if (y > (height-1) || x > (width-1))
            throw;

        unsigned int i = 4*(y*width + x);

        // RGBA
        return { image[i], image[i+1], image[i+2], image[i+3] };
    }

    /**
     * Print hex color value of pixel at (x, y).
     **/
    void printPixel(unsigned int x, unsigned int y)
    {
        pixel_t pixel = getPixel(x, y);
        printf("Pixel (%u, %u):\tRGBA: #%02x%02x%02x%02x\n", x, y, pixel.red, pixel.green, pixel.blue, pixel.alpha);
    }
};

float *createBlurMask(float sigma, int * maskSizePointer) {
    int maskSize = (int)ceil(3.0f*sigma);
    float * mask = new float[(maskSize*2+1)*(maskSize*2+1)];
    float sum = 0.0f;
    for(int a = -maskSize; a < maskSize+1; a++) {
        for(int b = -maskSize; b < maskSize+1; b++) {
            float temp = exp(-((float)(a*a+b*b) / (2*sigma*sigma)));
            sum += temp;
            mask[a+maskSize+(b+maskSize)*(maskSize*2+1)] = temp;
        }
    }
    // Normalize the mask
    for(int i = 0; i < (maskSize*2+1)*(maskSize*2+1); i++)
        mask[i] = mask[i] / sum;

    *maskSizePointer = maskSize;

    return mask;
}


int main(int argc, char ** argv) {
    // Load image
    Image image;
    img.load("simple.png");

    // Create OpenCL context
    Context context = createCLContextFromArguments(argc, argv);

    // Compile OpenCL code
    Program program = buildProgramFromSource(context, "gaussian_blur.cl");
    
    // Select device and create a command queue for it
    VECTOR_CLASS<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    CommandQueue queue = CommandQueue(context, devices[0]);

    // Create an OpenCL Image / texture and transfer data to the device
    Image2D clImage = Image2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ImageFormat(CL_RGBA, CL_UNORM_INT8), image.width, image.height, 0, (void*)image.image);

    return 0;

    // Create a buffer for the result
    Buffer clResult = Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*image.width*image.height);

    // Create Gaussian mask
    int maskSize;
    float * mask = createBlurMask(10.0f, &maskSize);
    
    // Create buffer for mask and transfer it to the device
    Buffer clMask = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned char)*(maskSize*2+1)*(maskSize*2+1), mask);

    // Run Gaussian kernel
    Kernel gaussianBlur = Kernel(program, "gaussian_blur");
    gaussianBlur.setArg(0, clImage);
    gaussianBlur.setArg(1, clMask);
    gaussianBlur.setArg(2, clResult);
    gaussianBlur.setArg(3, maskSize);

    queue.enqueueNDRangeKernel(
        gaussianBlur,
        NullRange,
        NDRange(image.width, image.height),
        NullRange
    );

    // Transfer image back to host
    // float* data = new float[image->getWidth()*image->getHeight()];
    vector<unsigned char> data;
    data.reserve(4*width*height);
    queue.enqueueReadBuffer(clResult, CL_TRUE, 0, sizeof(unsigned char)*imagewidth*image.height, &data); 
    // image->setData(data);
    image.data = data;

    // Save image to disk
    image->save("images/result.jpg", "jpeg");
    image->display();
}
