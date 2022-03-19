#include "Application.hpp"
#include "PerfTimer.hpp"
#include "MiniOCL.hpp"
#include "Filters.hpp"
#include "Image.hpp"

using std::cout;
using std::endl;

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
        case TARGET_PTHREAD:
            return "Pthread (CPU)";
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
    /*
    As a small note, the disparity map should be scaled to the range of 0-255
    based on the minimum and maximum values in the image.
    Also: using single channel image in calculations would be more efficient,
    CPU-wise, but especially memory-wise.

    TODO:
        (1) SCALING THE MAP TO 0-255
        (2) ONLY COMPILE KERNEL ONCE
        (3) NO UNNECESSARY TRANSFERS TO DEVICE
    */
    bool success;
    PerfTimer ptimer;
    std::string leftImgName;
    std::string rightImgName;
    unsigned int windowSize = 9;
    unsigned int maxSearchD = 32;
    unsigned int ccThreshold = 8;
    unsigned int downscaleFactor = 4;

    Image *leftImg = new Image();               // left stereo image
    Image *rightImg = new Image();              // right stereo image
    Image finalImg;                             // final image after cross-checking

    // if an arguments are provided, use them as image name
    if (argc > 2)
    {
        leftImgName = argv[1];
        rightImgName = argv[2];

        // the rest are optional
        if (argc > 3)
            windowSize = (unsigned int)atoi( argv[3] );
        if (argc > 4)
            maxSearchD = (unsigned int)atoi( argv[4] );
        if (argc > 5)
            ccThreshold = (unsigned int)atoi( argv[5] );
        if (argc > 6)
            downscaleFactor = (unsigned int)atoi( argv[6] );
    }
    else
    {
        CHECK_ERROR(false, "Left and right image names are required as an argument!");
    }

    // seems to be typically around 100-300 us
    cout << "NOTE: The execution times include some printing to console." << endl;
    cout << "Image manipulation is done using " << computeDeviceStr() << "." << endl;

#ifdef USE_OCL
    double kernelTime;

    // initialize OpenCL if necessary
    MiniOCL ocl(kernelFileName);
    ocl.initialize(TARGET_DEVICE_TYPE);
    leftImg->setOpenCL(&ocl);
    rightImg->setOpenCL(&ocl);
    finalImg.setOpenCL(&ocl);

    ocl.displayDeviceInfo();
#endif /* USE_OCL */

    // 1. Load both images from disk

    ptimer.reset();
    success = leftImg->load(leftImgName);
    CHECK_ERROR(success, "Error loading the left image from disk.")
    success = rightImg->load(rightImgName);
    CHECK_ERROR(success, "Error loading the right image from disk.")
    ptimer.printTime();

    cout << "Left image '" << leftImgName << "', size "
         << leftImg->width << "x" << leftImg->height << "." << endl;
    cout << "Right image '" << rightImgName << "', size "
         << rightImg->width << "x" << rightImg->height << "." << endl;

    // 2. Downscale (resize) the both images
    ptimer.reset();
    success = leftImg->downScale(downscaleFactor);
    CHECK_ERROR(success, "Error downscaling the left image.")
    success = rightImg->downScale(downscaleFactor);
    CHECK_ERROR(success, "Error downscaling the right image.")
    ptimer.printTime();

    // 3. Convert both images to grayscale
    ptimer.reset();
    success = leftImg->convertToGrayscale();
    CHECK_ERROR(success, "Error transforming the left image to grayscale.")
    success = rightImg->convertToGrayscale();
    CHECK_ERROR(success, "Error transforming the right image to grayscale.")
    ptimer.printTime();

#ifdef USE_OCL
    // print the actual kernel execution time
    kernelTime = ocl.getExecutionTime();
    printf("\t=> Right image kernel execution time: %0.3f ms \n", kernelTime / 1000.0f);
#endif /* USE_OCL */

    ptimer.reset();
    success = leftImg->save("img/1-gray-l.png");
    CHECK_ERROR(success, "Error saving the left image to disk.")
    success = rightImg->save("img/1-gray-r.png");
    CHECK_ERROR(success, "Error saving the right image to disk.")
    ptimer.printTime();

    // 4. Calculate stereo disparity (ZNCC) for both images

    Image *leftDispImg = new Image();      // contains the left-to-right disparity map
    Image *rightDispImg = new Image();     // contains the right-to-left disparity map

    ptimer.reset();
    success = leftImg->calcZNCC(*rightImg, leftDispImg, windowSize, maxSearchD);
    CHECK_ERROR(success, "Error calculating ZNCC for the left image.")
    success = rightImg->calcZNCC(*leftImg, rightDispImg, windowSize, maxSearchD, true);
    CHECK_ERROR(success, "Error calculating ZNCC for the right image.")
    ptimer.printTime();

#ifdef USE_OCL
    // print the actual kernel execution time
    kernelTime = ocl.getExecutionTime();
    printf("\t=> Kernel execution time: %0.3f ms \n", kernelTime / 1000.0f);
#endif /* USE_OCL */

    // these have become unnecessary at this point
    delete leftImg;
    delete rightImg;

    ptimer.reset();
    success = leftDispImg->save("img/2-disparity-l.png");
    CHECK_ERROR(success, "Error saving the left image to disk.")
    success = rightDispImg->save("img/2-disparity-r.png");
    CHECK_ERROR(success, "Error saving the right image to disk.")
    ptimer.printTime();

    // 5. Cross checking

    ptimer.reset();
    finalImg.crossCheck(*leftDispImg, *rightDispImg, ccThreshold);
    CHECK_ERROR(success, "Error in cross checking.")
    ptimer.printTime();

#ifdef USE_OCL
    // print the actual kernel execution time
    kernelTime = ocl.getExecutionTime();
    printf("\t=> Kernel execution time: %0.3f ms \n", kernelTime / 1000.0f);
#endif /* USE_OCL */

    // these have become unnecessary at this point
    delete leftDispImg;
    delete rightDispImg;

    ptimer.reset();
    success = finalImg.save("img/3-cross-checked.png");
    CHECK_ERROR(success, "Error saving image to disk.")
    ptimer.printTime();

    // 6. Occlusion filling

    ptimer.reset();
    success = finalImg.occlusionFill();
    CHECK_ERROR(success, "Error in occlusion filling.")
    ptimer.printTime();

#ifdef USE_OCL
    // print the actual kernel execution time
    kernelTime = ocl.getExecutionTime();
    printf("\t=> Kernel execution time: %0.3f ms \n", kernelTime / 1000.0f);
#endif /* USE_OCL */

    ptimer.reset();
    success = finalImg.save("img/4-occlusion-filled.png");
    CHECK_ERROR(success, "Error saving image to disk.")
    ptimer.printTime();

    return EXIT_SUCCESS;
}
