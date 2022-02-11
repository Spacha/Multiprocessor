#define CL_USE_DEPRECATED_OPENCL_2_0_APIS // required for using olf APIs

#include "lodepng.h"

#include <windows.h>
#include <CL/cl.h>
#include <iostream>
#include <stdio.h>
#include <math.h>

using namespace std;

// These are options for COMPUTE_DEVICE.
#define TARGET_NONE 0                   // Don't use OpenCL
#define TARGET_GPU  1                   // OpenCL on GPU
#define TARGET_CPU  2                   // OpenCL on CPU

///////////////////////////////////////////////////////////////////////////////
// Parameters:
#define USE_OPENCL      0               // 0 = Filter image without parallelization, 1 = Use OpenCL (GPU/CPU)
#define COMPUTE_DEVICE  TARGET_NONE     // TARGET_NONE, TARGET_GPU, TARGET_CPU; Compute device to be used in parallelization
#define VERBOSITY       1               // 0 = Essential, 1 = Extra information, 2 = Too much information
///////////////////////////////////////////////////////////////////////////////

#if USE_CPU
#define TARGET_DEVICE_TYPE CL_DEVICE_TYPE_CPU
#else
#define TARGET_DEVICE_TYPE CL_DEVICE_TYPE_GPU
#endif

/*
@link https://lodev.org/lodepng/

Compile using:
    g++ main.cpp lodepng.cpp -Wall -o main.exe
For optimized compilation (takes more time):
    g++ main.cpp lodepng.cpp -Wall -Wextra -pedantic -ansi -O3 -o main.exe
*/

/*
     0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   ...
0    RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA ...
1    RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA ...
2    RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA RGBA ...
     ...
*/

////////////////////////////////////////////////////////////////////////////////

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

void printIntro()
{
    cout << "Using " << computeDeviceStr() << "." << endl;
    // TODO: Show compute device info...
    cout << endl;
}

///////////////////////////////////////////////////////////////////////////////
/* Requires <windows.h> */
class PerfTimer
{
public:
    LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
    LARGE_INTEGER Frequency;

    /**
     * Start performance counter. If it is already started, calling this restarts it.
     **/
    void start()
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
};

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
     * Initializes the object and sets color type.
     * LCT_RGBA => 4 channels
     * LCT_GREY => 1 channel
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
        this->image.reserve(this->pixelSize() * this->width * this->height);
    }

    size_t pixelSize()
    {
        switch(this->colorType)
        {
            case LCT_RGBA:
                return 4;
                break;
            case LCT_GREY:
                return 1;
                break;
            default:
                return 0;
                break;
        }
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
        unsigned err;
        std::vector<unsigned char> png;

        cout << "Encoding image...";
        err = lodepng::encode(png, image, width, height, colorType);
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
        // this function is only applicable for RGBA images
        if (y > (this->height-1) || x > (this->width-1) || this->colorType != LCT_RGBA)
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
    void putPixel(unsigned int x, unsigned int y, unsigned char pixel)
    {
        // this function is only applicable for greyscale images
        if (y > (this->height-1) || x > (this->width-1) || this->colorType != LCT_GREY)
            throw;

        unsigned int i = y*width + x;
        this->image[i] = pixel;
    }

    /**
     * Returns a pixel struct containing the color values of each pixel in (x,y).
     **/
    pixel_t getPixel(unsigned int x, unsigned int y)
    {
        if (y > (height-1) || x > (width-1))
        {
            throw;
        }

        unsigned int i = this->pixelSize()*(y*width + x);
        if (this->colorType == LCT_GREY)
        {
            // grayscale...
            return { image[i], image[i], image[i], 0xff };
        }

        // RGBA
        return { image[i], image[i+1], image[i+2], image[i+3] };
    }

    /**
     * Print hex color value of pixel at (x, y).
     **/
    void printPixel(unsigned int x, unsigned int y)
    {
        pixel_t pixel = getPixel(x, y);
        if (this->colorType == LCT_GREY)
        {
            printf("Pixel (%u, %u):\tGrey (1-255): %u\n", x, y, pixel.red);
        } else {
            printf("Pixel (%u, %u):\tRGBA: #%02x%02x%02x%02x\n", x, y, pixel.red, pixel.green, pixel.blue, pixel.alpha);
        }
    }

    /**
     * Creates a greyscale copy of the image to greyImg and makes the image opaque.
     * Uses NTSC formula.
     **/
    void getGrayScale(Image *grayImg)
    {
        pixel_t p;
        unsigned char grayVal;

        cout << "Transforming image to grayscale...";

#if COMPUTE_DEVICE == TARGET_NONE /* No parallelization */

        for (unsigned int y = 0; y < this->height; y++)
        {
            for (unsigned int x = 0; x < this->width; x++)
            {
                p = this->getPixel(x, y);
                grayVal = ceil(0.299*p.red + 0.587*p.green + 0.114*p.blue);

                grayImg->putPixel(x, y, grayVal);
                // grayImg[i] = grayVal << 24 | grayVal << 16 << grayVal << 8 | 0xff;
                /*
                grayImg->image.push_back( grayVal );
                grayImg->image.push_back( grayVal );
                grayImg->image.push_back( grayVal );
                grayImg->image.push_back( 0xff );
                */
            }
        }

#else /* OpenCL (GPU or CPU) */
        cout << "[ Not yet implemented] ";
#endif

        cout << " Done." << endl;
    }

    /**
     * Creates a greyscale copy of the image to greyImg and makes the image opaque.
     * Uses NTSC formula.
     **/
    void getFiltered(Image *grayImg)
    {
        cout << "Filtering image...";

#if COMPUTE_DEVICE == TARGET_NONE
        // No parallelization
#else
        // OpenCL (GPU or CPU)
#endif

        cout << " Done." << endl;
    }
};

///////////////////////////////////////////////////////////////////////////////

int main()
{
    Image img;
    PerfTimer ptimer;
    long long int delta_us;

    printIntro();

    ptimer.start();
    //img.load("im0.png");
    img.load("simple.png");
    delta_us = ptimer.getMicroseconds();
    printf("\t=> time: %0.3f ms\n", delta_us/1000.0);

    printf("Image: %ux%u\n", img.width, img.height);
    // img.printPixel(2189, 1323);
    
    Image imgGray(LCT_GREY);      // contains the grayscale image
    Image imgFiltered(LCT_GREY);  // contains the filtered image

    imgGray.createEmpty(img.width, img.height);
    imgFiltered.createEmpty(img.width, img.height);

    // convert the original image to grayscale and store it to imgGray
    ptimer.start();
    img.getGrayScale(&imgGray);
    delta_us = ptimer.getMicroseconds();
    printf("\t=> time: %0.3f ms\n", delta_us/1000.0);

    for (unsigned int y = 0; y < imgGray.height; y++)
    {
        for (unsigned int x = 0; x < imgGray.width; x++)
        {
            imgGray.printPixel(x,y);
        }
    }

    // store the gray image to disk
    ptimer.start();
    imgGray.save("gray.png");
    delta_us = ptimer.getMicroseconds();
    printf("\t=> time: %0.3f ms\n", delta_us/1000.0);

    // convert the original image to grayscale and store it to imgGray
    ptimer.start();
    img.getFiltered(&imgFiltered);
    delta_us = ptimer.getMicroseconds();
    printf("\t=> time: %0.3f ms\n", delta_us/1000.0);

    // store the gray image to disk
    ptimer.start();
    imgGray.save("filtered.png");
    delta_us = ptimer.getMicroseconds();
    printf("\t=> time: %0.3f ms\n", delta_us/1000.0);

    // Measure the image size in memory before and after grayscaling.

    return 0;
}
