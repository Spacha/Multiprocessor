#include "Application.hpp"
#include "PerfTimer.hpp"
#include "MiniOCL.hpp"
#include "Image.hpp"

using std::cout;
using std::endl;

///////////////////////////////////////////////////////////////////////////////
// Parameters:
///////////////////////////////////////////////////////////////////////////////

// COMPUTE_DEVICE options:
//  TARGET_NONE = Sequential - don't use OpenCL
//  TARGET_GPU  = OpenCL on GPU
//  TARGET_CPU  = OpenCL on CPU
#define COMPUTE_DEVICE TARGET_GPU

/**
 * Compile & run:
 *   cls && g++ main.cpp MiniOCL.cpp lodepng.cpp %OCL_ROOT%/lib/x86_64/opencl.lib -Wall -I %OCL_ROOT%\include -o image-filter.exe && image-filter.exe img/im0.png
 **/

///////////////////////////////////////////////////////////////////////////////
// DEFINITIONS & MACROS
///////////////////////////////////////////////////////////////////////////////

/**
 * If @success is false, orints out an error message and returns.
 **/
#define CHECK_ERROR(success, msg)                           \
    if (!success) {                                         \
        cout << msg << endl;                                \
        return EXIT_FAILURE;                                \
    }

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
    success = img.filter(gaussianFilter); // meanFilter / gaussianFilter / embossFilter
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
