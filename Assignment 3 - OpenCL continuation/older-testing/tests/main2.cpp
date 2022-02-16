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

int main()
{
    Image img;
    PerfTimer ptimer;
    long long int delta_us;

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
    // imgFiltered.createEmpty(img.width, img.height);

    vector<unsigned char> grayVect;
    grayVect.reserve(img.width * img.height); // only 1 channel

    // convert the original image to grayscale and store it to imgGray
    ptimer.start();
    img.getGrayScale(grayVect);
    delta_us = ptimer.getMicroseconds();
    printf("\t=> time: %0.3f ms\n", delta_us/1000.0);

#if 0
    for (unsigned int y = 0; y < imgGray.height; y++)
    {
        for (unsigned int x = 0; x < imgGray.width; x++)
        {
            imgGray.printPixel(x,y);
        }
    }
#endif

    // store the gray image to disk
    ptimer.start();
    imgGray.save("gray.png");
    delta_us = ptimer.getMicroseconds();
    printf("\t=> time: %0.3f ms\n", delta_us/1000.0);

#if 0
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
#endif

    // Measure the image size in memory before and after grayscaling.

    return 0;
}
