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

// These are options for COMPUTE_DEVICE.
#define TARGET_NONE 0                   // Don't use OpenCL
#define TARGET_GPU  1                   // OpenCL on GPU
#define TARGET_CPU  2                   // OpenCL on CPU

///////////////////////////////////////////////////////////////////////////////
// Parameters:
///////////////////////////////////////////////////////////////////////////////

#define COMPUTE_DEVICE TARGET_GPU 		// TARGET_NONE / TARGET_GPU / TARGET_CPU

///////////////////////////////////////////////////////////////////////////////
// DEFINITIONS & MACROS
///////////////////////////////////////////////////////////////////////////////

#define CHECK_ERROR(success, msg) 							\
	if (!success) {				  							\
		cout << msg << endl;								\
		return EXIT_FAILURE;						 		\
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

// See OpenCL Programming Guide p.342.
// OpenCL kernel. Each work item takes care of one element of C.

const char *kernelSource = "\n" \
"__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |                 \n" \
"                               CLK_ADDRESS_CLAMP_TO_EDGE   |                 \n" \
"                               CLK_FILTER_NEAREST;                           \n" \
"__kernel void RGBA2grayscale(__read_only image2d_t in,                       \n" \
"                         __write_only image2d_t out)                         \n" \
"{                                                                            \n" \
"    // get our global thread ID                                              \n" \
"    int id = (get_global_id(1) * get_global_size(0)) + get_global_id(0);     \n" \
"                                                                             \n" \
"    /*c[id] = a[id] + b[id];*/                                               \n" \
"    //float4 clr = (0.5f, 0.5f, 0.5f, 0.5f);                                 \n" \
"    int2 coord = (int2)(get_global_id(0), get_global_id(1));                 \n" \
"    float4 clr = read_imagef(in, sampler, coord);                            \n" \
"                                                                             \n" \
"    // write_imagef(out, coord, (float4)((coord[0]+1)/6.0f, (coord[1]+1)/3.0f,0.75f,1.0f));               \n" \
"    write_imagef(out, coord, clr);                                           \n" \
"}                                                                            \n" \
"\n";

// Mean filter (5x5)
const size_t maskSize = 5;
const float mask[maskSize*maskSize] = { // 0.04 = 1/25 = 1/(5*5)
	0.04f, 0.04f, 0.04f, 0.04f, 0.04f,
	0.04f, 0.04f, 0.04f, 0.04f, 0.04f,
	0.04f, 0.04f, 0.04f, 0.04f, 0.04f
};

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

/* Requires <windows.h> */
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
    	double delta_us = this->getMicroseconds();

    	std::string unitsStr;
    	double divisor;

    	if (delta_us < 1000) { 				// microseconds
			unitsStr = "us";
    		divisor = 1;
    	} else if (delta_us < 1000000) {	// milliseconds
			unitsStr = "ms";
    		divisor = 1000;
    	} else {
    		unitsStr = "s"; 				// seconds
    		divisor = 1000000;
    	}

    	printf("\t=> time: %0.3f %s\n", delta_us/divisor, unitsStr.c_str());
    }
};

///////////////////////////////////////////////////////////////////////////////
// Image
///////////////////////////////////////////////////////////////////////////////

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
    size_t width;                 		// image width
    size_t height;                		// image height
    LodePNGColorType colorType;         // color type (see LodePNGColorType in lodepng.h)
    MiniOCL *ocl = nullptr;				// Handle to OpenCL wrapper class for parallel execution

    /**
     * Initializes the object and sets color type
     * E.g. LCT_RGBA, LCT_GREY
     **/
    Image(LodePNGColorType colorType = LCT_RGBA)
    {
        this->colorType = colorType;
    }

    /**
     * In case parallel execution is to be used, an OpenCL (MiniOCL) instance is needed.
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
        // cout << this->image.size() << endl;

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
     * Creates a greyscale copy of the image to greyImg and makes the image opaque.
     * Uses NTSC formula.
     **/
    bool convertToGrayscale()
    {
    	bool success = true;
        cout << "Transforming image to grayscale... ";

        std::string name = "RGBA2grayscale";
        success = this->ocl->buildKernel(&name, &kernelSource);

        success = this->ocl->setImageBuffers(
        	static_cast<void *>(this->image.data()), // input image
        	static_cast<void *>(this->image.data()), // output image (overwrite)
        	width,
        	height);

        success = this->ocl->executeKernel(16, 16);

#ifdef USE_OCL /* OpenCL (GPU or CPU) */

        // cout << "Kernel execution time: " << microSeconds << endl;
        if (!this->ocl) {
        	cout << "Cannot do parallel execution without instance of MiniOCL." << endl;
        	return false;
        }

#else /* No parallelization */

		pixel_t p;
        unsigned char grayVal;
        unsigned int i;

        for (unsigned int y = 0; y < this->height; y++)
        {
            for (unsigned int x = 0; x < this->width; x++)
            {
                p = this->getPixel(x, y);
                grayVal = ceil(0.299*p.red + 0.587*p.green + 0.114*p.blue);

                i = 4*(y*width + x); // could also just use: i++
                this->image[i]   = grayVal; // could use putPixel
                this->image[i+1] = grayVal;
                this->image[i+2] = grayVal;
                this->image[i+3] = 0xff;
            }
        }

        this->colorType = LCT_GREY; // update color type

#endif

        cout << "Done." << endl;
        return success;
    }

    /**
     * Filters the image using the mask provided.
     **/
    bool filter(const float *mask, size_t maskSize)
    {
        cout << "Filtering image using a mask... ";

#ifdef USE_OCL /* OpenCL (GPU or CPU) */

        // ...

#else /* No parallelization */

        // ...

#endif

        cout << "Done." << endl;
        return true;
    }

	/**
     * Puts given RGBA (4 channel) pixel to position (x,y).
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
        if (y > (height-1) || x > (width-1))
            throw;

        // RGBA
        unsigned int i = 4*(y*width + x);
        return { image[i], image[i+1], image[i+2], image[i+3] };
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



/******************************************************************************
* MAIN
******************************************************************************/

int main()
{
	bool success;
	Image img;
	PerfTimer ptimer; // used for measuring execution time


	std::string imgName = "simple.png"; // the image to load from disk

	// seems to be typically around 100-300 us
	cout << "NOTE: The execution times include some printing to console." << endl;
	cout << "Image manipulation is done using " << computeDeviceStr() << "." << endl;

#ifdef USE_OCL
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
	cout << "Image '" << imgName << "', size " << img.width << "x" << img.height << endl;
	CHECK_ERROR(success, "Error loading image from disk.")

	// 2. convert the image to grayscale
	ptimer.reset();
	success = img.convertToGrayscale();
	ptimer.printTime();
	CHECK_ERROR(success, "Error transforming image to grayscale.")

	// print the actual kernel execution time
	double microSeconds = ocl.getExecutionTime();
    printf("\t=> Kernel execution time: %0.3f us \n", microSeconds);

	ptimer.reset();
	success = img.save("gray.png");
	ptimer.printTime();
	CHECK_ERROR(success, "Error saving image to disk.")

	// 3. filter image
	ptimer.reset();
	success = img.filter(mask, maskSize);
	ptimer.printTime();
	CHECK_ERROR(success, "Error filtering the image.")

	// 4. save filtered image
	ptimer.reset();
	img.save("filtered.png");
	ptimer.printTime();
	CHECK_ERROR(success, "Error saving the image to disk.")

	return EXIT_SUCCESS;
}
