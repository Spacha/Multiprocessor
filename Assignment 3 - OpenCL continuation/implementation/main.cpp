#define CL_USE_DEPRECATED_OPENCL_2_0_APIS // required for using olf APIs

#include <windows.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

#include "lodepng.h"
#include "MiniOCL.hpp"

using std::cout;
using std::endl;

/**
 * Compile & run:
 *   cls && g++ main.cpp MiniOCL.cpp lodepng.cpp %OCL_ROOT%/lib/x86_64/opencl.lib -Wall -I %OCL_ROOT%\include -o image-filter.exe && image-filter.exe img/im0.png
 **/

// These are options for COMPUTE_DEVICE.
#define TARGET_NONE 0                   // Don't use OpenCL
#define TARGET_GPU  1                   // OpenCL on GPU
#define TARGET_CPU  2                   // OpenCL on CPU

///////////////////////////////////////////////////////////////////////////////
// Parameters:
///////////////////////////////////////////////////////////////////////////////

#define COMPUTE_DEVICE TARGET_GPU      // TARGET_NONE / TARGET_GPU / TARGET_CPU

///////////////////////////////////////////////////////////////////////////////
// DEFINITIONS & MACROS
///////////////////////////////////////////////////////////////////////////////

struct filter_s
{
    const size_t size;      // size of the mask (height or width)
    const float divisor;    // the mask is divided by this
    const float *mask;      // the actual filter mask
};
typedef struct filter_s filter_t;

#define CHECK_ERROR(success, msg)                           \
    if (!success) {                                         \
        cout << msg << endl;                                \
        return EXIT_FAILURE;                                \
    }

#if COMPUTE_DEVICE == TARGET_CPU
# define TARGET_DEVICE_TYPE CL_DEVICE_TYPE_CPU
#else
# define TARGET_DEVICE_TYPE CL_DEVICE_TYPE_GPU
#endif

#if COMPUTE_DEVICE != TARGET_NONE
# define USE_OCL
#endif

///////////////////////////////////////////////////////////////////////////////
// GLOBAL VARIABLES
///////////////////////////////////////////////////////////////////////////////

const char *kernelFileName = "kernels.cl";
// const std::string imgName  = "img/simple.png";

///////////////////////////////////////////////////////////////////////////////
// FILTERS
///////////////////////////////////////////////////////////////////////////////

const size_t maskSize = 5;

// Mean/averaging filter (5x5)
const float meanFilterMask[maskSize*maskSize] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f
};
const filter_t meanFilter = {
    .size = maskSize,
    .divisor = 25.0f,
    .mask = meanFilterMask
};

// Gaussian filter (5x5)
const float gaussianFilterMask[maskSize*maskSize] = {
     1.0f,  4.0f,  7.0f,  4.0f,  1.0f,
     4.0f, 16.0f, 26.0f, 16.0f,  4.0f,
     7.0f, 26.0f, 41.0f, 26.0f,  7.0f,
     4.0f, 16.0f, 26.0f, 16.0f,  4.0f,
     1.0f,  4.0f,  7.0f,  4.0f,  1.0f
};
const filter_t gaussianFilter = {
    .size = maskSize,
    .divisor = 273.0f,
    .mask = gaussianFilterMask
};

// Emboss filter (5x5)
const float embossFilterMask[maskSize*maskSize] = {
     -1.0f,  0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, -1.0f, 0.0f, 0.0f, 0.0f,
      0.0f,  0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,  0.0f, 0.0f, 1.0f, 0.0f,
      0.0f,  0.0f, 0.0f, 0.0f, 1.0f
};
const filter_t embossFilter = {
    .size = maskSize,
    .divisor = 1.0f,
    .mask = embossFilterMask
};

const filter_t filters[] = { meanFilter, gaussianFilter, embossFilter };


///////////////////////////////////////////////////////////////////////////////
// HELPER FUNCTIONS
///////////////////////////////////////////////////////////////////////////////

/**
 * Returns the current compute device name.
 **/
std::string computeDeviceStr()
{
    switch (COMPUTE_DEVICE)
    {
        case TARGET_GPU:
            return "OpenCL (GPU)";
            break;
        case TARGET_CPU:
            return "OpenCL (CPU)";
            break;
        default:
            return "CPU (no parallelization)";
            break;
    }
}

///////////////////////////////////////////////////////////////////////////////
// PerfTimer
///////////////////////////////////////////////////////////////////////////////

/**
 * Used to measure performance (i.e. execution times) of different operations.
 * Requires <windows.h>
 **/
class PerfTimer
{
public:
    LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
    LARGE_INTEGER Frequency;

    /**
     * Start/reset performance counter.
     **/
    void reset()
    {
        QueryPerformanceFrequency(&Frequency);
        QueryPerformanceCounter(&StartingTime);
    }

    /**
     * Get a snapshot of the delta time in microseconds. 
     **/
    long long int getMicroseconds()
    {
        QueryPerformanceCounter(&EndingTime);
        ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
        ElapsedMicroseconds.QuadPart *= 1000000;
        ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;

        return ElapsedMicroseconds.QuadPart;
    }

    /**
     * A helper for printing the execution time since start.
     **/
    void printTime()
    {
        double us = this->getMicroseconds();

        std::string unitsStr;
        double divisor;

        if (us < 1000) {            // microseconds
            unitsStr = "us";
            divisor = 1;
        } else if (us < 1000000) {  // milliseconds
            unitsStr = "ms";
            divisor = 1000;
        } else {                    // seconds
            unitsStr = "s";
            divisor = 1000000;
        }

        printf("\t=> time: %0.3f %s\n", us / divisor, unitsStr.c_str());
    }
};

///////////////////////////////////////////////////////////////////////////////
// Image
///////////////////////////////////////////////////////////////////////////////

/* Struct representing a pixel (0-255) */
struct pixel_s
{
    unsigned char red;
    unsigned char green;
    unsigned char blue;
    unsigned char alpha;
};
typedef struct pixel_s pixel_t;

/* Same struct but as float (0.0f - 1.0f) */
struct pixelf_s
{
    float red;
    float green;
    float blue;
    float alpha;
};
typedef struct pixelf_s pixelf_t;

/**
 * Convert the floating point representation
 * to the integer representation.
 */
pixel_t pixelf_to_pixel(pixelf_t p)
{
    return {
        (unsigned char)(255.0f * p.red),
        (unsigned char)(255.0f * p.green),
        (unsigned char)(255.0f * p.blue),
        (unsigned char)(255.0f * p.alpha)
    };
}
/**
 * Convert the integer representation
 * to the floating point representation.
 */
pixelf_t pixel_to_pixelf(pixel_t p)
{
    return {
        (float)(  p.red / 255.0f),
        (float)(p.green / 255.0f),
        (float)( p.blue / 255.0f),
        (float)(p.alpha / 255.0f)
    };
}


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
    size_t width;                       // image width
    size_t height;                      // image height
    LodePNGColorType colorType;         // color type (see LodePNGColorType in lodepng.h)
    MiniOCL *ocl = nullptr;             // Handle to OpenCL wrapper class for parallel execution

    /**
     * Initializes the object and sets color type.
     * E.g. LCT_RGBA, LCT_GREY
     **/
    Image(LodePNGColorType colorType = LCT_RGBA)
    {
        this->colorType = colorType;
    }

    /**
     * In case parallel execution is to be used, an OpenCL (MiniOCL) instance is needed.
     * This method sets an already initialized MiniOCL object as a property.
     **/
    void setOpenCL(MiniOCL *ocl)
    {
        this->ocl = ocl;
    }

    /**
     * Creates an empty image of given size.
     **/
    void createEmpty(size_t width, size_t height)
    {
        this->width = width;
        this->height = height;

        this->image.clear();
        this->image.resize(this->sizeBytes(), (unsigned char)0);
        this->image.resize(this->sizeBytes(), (unsigned char)0);
    }

    /**
     * Returns the size of the image in bytes.
     **/
    size_t sizeBytes()
    {
        // TODO: Can we guarantee always having 4 channels?
        return 4 * sizeof(unsigned char) * this->width * this->height;
    }

    /**
     * Replaces the current image with given image @newImage.
     * The image sizes must match exactly.
     **/
    void replaceImage(std::vector<unsigned char> &newImage)
    {
        if (newImage.size() != this->image.size())
            throw;

        this->image = std::move(newImage);
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
        cout << "Loading image... ";
        err = lodepng::load_file(png, filename);
        cout << "Done." << endl;

        if (err) {
            cout << "Image load error " << err << ": " << lodepng_error_text(err) << endl;
            return false;
        }

        cout << "Decoding image... ";
        err = lodepng::decode(this->image,
            (unsigned &)this->width, (unsigned &)this->height, png);
        cout << "Done." << endl;

        // the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA...

        if (err) {
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
        unsigned err;
        std::vector<unsigned char> png;

        cout << "Encoding image... ";
        err = lodepng::encode(png, this->image,
            (unsigned)this->width, (unsigned)this->height);
        cout << "Done." << endl;

        if (err) {
            cout << "Encode error " << err << ": "<< lodepng_error_text(err) << endl;
            return false;
        }

        cout << "Saving image... ";
        lodepng::save_file(png, filename);
        cout << "Done." << endl;

        if (err) {
            cout << "Image save error " << err << ": " << lodepng_error_text(err) << endl;
            return false;
        }

        return true;
    }

    /**
     * Converts the image to grayscale and makes the image opaque.
     * Uses the NTSC formula.
     **/
    bool convertToGrayscale()
    {
        bool success = true;
        cout << "Transforming image to grayscale... ";

#ifdef USE_OCL /* OpenCL (GPU or CPU) */

        if (!this->ocl) {
            cout << "Cannot do parallel execution without instance of MiniOCL." << endl;
            return false;
        }

        // set up OpenCL for execution

        success = ocl->buildKernel(kernelFileName, "grayscale");

        ocl->setInputImageBuffer(
            0, static_cast<void *>(image.data()), width, height);   // image in
        ocl->setOutputImageBuffer(
            1, static_cast<void *>(image.data()), width, height);   // image out

        success = this->ocl->executeKernel(width, height, 16, 16);

#else /* No parallelization */

        for (unsigned int y = 0; y < this->height; y++)
        {
            for (unsigned int x = 0; x < this->width; x++)
            {
                pixel_t p = this->getPixel(x, y);

                // replace the pixel with a gray one
                this->putPixel(x, y, ceil(0.299*p.red + 0.587*p.green + 0.114*p.blue));
            }
        }

#endif
        // this->colorType = LCT_GREY; // update color type
        cout << "Done." << endl;
        return success;
    }

    /**
     * Filters the image using the given mask.
     **/
    bool filter(const filter_t &filter)
    {
        bool success = true;
        cout << "Filtering image using a mask... ";

        // the mask size must be odd
        if (filter.size % 2 == 0) {
            cout << "Error: mask size is not odd." << endl;
            return false;
        }

#ifdef USE_OCL /* OpenCL (GPU or CPU) */

        if (!this->ocl) {
            cout << "Cannot do parallel execution without instance of MiniOCL." << endl;
            return false;
        }

        success = this->ocl->buildKernel(kernelFileName, "filter");

        ocl->setInputImageBuffer(
            0, static_cast<void *>(image.data()), width, height);               // image in
        ocl->setOutputImageBuffer(
            1, static_cast<void *>(image.data()), width, height);               // image out
        ocl->setInputBuffer(
            2, (void *)filter.mask, filter.size * filter.size * sizeof(float)); // filter mask
        ocl->setValue(
            3, (void *)&filter.size, sizeof(int));                              // filter size
        ocl->setValue(
            4, (void *)&filter.divisor, sizeof(float));                         // filter divisor

        success = ocl->executeKernel(width, height, 16, 16);

#else /* No parallelization */

        // TODO: This implementation could use different edge handling techniques.
        // std::vector<unsigned char> tempImage(4 * width * height);
        Image tempImage;
        tempImage.createEmpty(width, height);

        int d = filter.size / 2; // kernel's "edge thickness"

        for (unsigned int cy = 0; cy < height; cy++)
        {
            for (unsigned int cx = 0; cx < width; cx++)
            {
                unsigned int weight = 0;
                pixelf_t newClr = { 0.0f, 0.0f, 0.0f, 255.0f };

                // iterate over each element in the mask
                for (int y = cy - d; y <= (int)(cy + d); y++)
                {
                    for (int x = cx - d; x <= (int)(cx + d); x++)
                    {
                        // get pixel if within image
                        pixel_t p = validCoordinates(x, y)
                            ? this->getPixel(x, y)
                            : (pixel_t){ 0, 0, 0, 0 };

                        newClr.red   += filter.mask[weight] * (float)p.red;
                        newClr.green += filter.mask[weight] * (float)p.green;
                        newClr.blue  += filter.mask[weight] * (float)p.blue;
                        weight++;
                    }
                }

                // convert back to integers (floats are used in computation)
                pixel_t pixelInt = {
                    (unsigned char)(newClr.red / filter.divisor),
                    (unsigned char)(newClr.green / filter.divisor),
                    (unsigned char)(newClr.blue / filter.divisor),
                    (unsigned char)(newClr.alpha)
                };

                // replace the pixel in the center of the mask
                tempImage.putPixel(cx, cy, pixelInt);
            }
        }

        // update the image
        this->replaceImage(tempImage.image);

#endif
        cout << "Done." << endl;
        return success;
    }

    /**
     * Puts given RGBA (4 channel) pixel to position (x,y).
     **/
    void putPixel(unsigned int x, unsigned int y, pixel_t pixel)
    {
        if (!validCoordinates(x, y))
            throw;

        unsigned int i = 4*(y*width + x);
        this->image[i]   = pixel.red;
        this->image[i+1] = pixel.green;
        this->image[i+2] = pixel.blue;
        this->image[i+3] = pixel.alpha;
    }

    /**
     * Puts given grayscale (1 channel) pixel to position (x,y).
     **/
    void putPixel(unsigned int x, unsigned int y, unsigned char grey)
    {
        pixel_t pixel = { grey, grey, grey, 0xff};
        this->putPixel(x, y, pixel);
    }

    /**
     * Returns a pixel struct containing the color values
     * of each pixel in position (x,y).
     **/
    pixel_t getPixel(unsigned int x, unsigned int y)
    {
        if (!validCoordinates(x, y))
            throw;

        // RGBA
        unsigned int i = 4*(y*width + x);
        return { image[i], image[i+1], image[i+2], image[i+3] };
    }

    /**
     * Returns true if given coordinates point to
     * a pixel coordinate, otherwise false.
     **/
    bool validCoordinates(unsigned int x, unsigned int y)
    {
        return !(x > (width-1) || y > (height-1) || x < 0 || y < 0);
    }

    /**
     * Print hex color value of pixel in position (x,y).
     **/
    void printPixel(unsigned int x, unsigned int y)
    {
        pixel_t pixel = getPixel(x, y);
        printf("Pixel (%u, %u):\tRGBA: #%02x%02x%02x%02x\n", x, y, pixel.red, pixel.green, pixel.blue, pixel.alpha);
    }
};



///////////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    bool success;
    Image img;              // holds the image name that is being handled
    PerfTimer ptimer;       // used for measuring execution time
    std::string imgName;    // the image name that is being handled

    // if an argument is provided, use as image name
    if (argc > 1)
    {
        imgName = argv[1];
    }
    else
    {
        cout << "Image name is required as an argument!" << endl;
        return EXIT_FAILURE;
    }

    // seems to be typically around 100-300 us
    cout << "NOTE: The execution times include some printing to console." << endl;
    cout << "Image manipulation is done using " << computeDeviceStr() << "." << endl;

#ifdef USE_OCL
    double kernelTime;

    // initialize OpenCL if necessary
    MiniOCL ocl;
    ocl.initialize(TARGET_DEVICE_TYPE);
    img.setOpenCL(&ocl);

    ocl.displayDeviceInfo();
#endif /* USE_OCL */

    // 1. load image from disk
    ptimer.reset();
    success = img.load(imgName);
    ptimer.printTime();
    cout << "Image '" << imgName << "', size "
         << img.width << "x" << img.height << "." << endl;
    CHECK_ERROR(success, "Error loading image from disk.")

    // 2. convert the image to grayscale
    ptimer.reset();
    success = img.convertToGrayscale();
    ptimer.printTime();
    CHECK_ERROR(success, "Error transforming image to grayscale.")

#ifdef USE_OCL
    // print the actual kernel execution time
    kernelTime = ocl.getExecutionTime();
    printf("\t=> Kernel execution time: %0.3f ms \n", kernelTime / 1000.0f);
#endif /* USE_OCL */

    ptimer.reset();
    success = img.save("img/gray.png");
    ptimer.printTime();
    CHECK_ERROR(success, "Error saving image to disk.")

    // 3. filter image
    ptimer.reset();
    success = img.filter(embossFilter); // meanFilter / gaussianFilter / embossFilter
    ptimer.printTime();
    CHECK_ERROR(success, "Error filtering the image.")

#ifdef USE_OCL
    // print the actual kernel execution time
    kernelTime = ocl.getExecutionTime();
    printf("\t=> Kernel execution time: %0.3f ms \n", kernelTime / 1000.0f);
#endif /* USE_OCL */

    // 4. save filtered image
    ptimer.reset();
    img.save("img/filtered.png");
    ptimer.printTime();
    CHECK_ERROR(success, "Error saving the image to disk.")

    return EXIT_SUCCESS;
}
