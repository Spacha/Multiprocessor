#include "Application.hpp"
#include "PerfTimer.hpp"
#include "MiniOCL.hpp"
#include "Filters.hpp"
#include "Image.hpp"

using std::cout;
using std::endl;

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
    PerfTimer ptimer;
    std::string leftImgName;
    std::string rightImgName;

    Image *leftImg = new Image();               // left stereo image
    Image *rightImg = new Image();              // right stereo image
    Image finalImg;                             // final image after cross-checking

    const int downscaleFactor = 4;

    // if an arguments are provided, use them as image name
    if (argc > 2)
    {
        leftImgName = argv[1];
        rightImgName = argv[2];
    }
    else
    {
        cout << "Left and right image names are required as an argument!" << endl;
        return EXIT_FAILURE;
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
    success = leftImg->calcZNCC(*rightImg, leftDispImg);
    CHECK_ERROR(success, "Error calculating ZNCC for the left image.")
    success = rightImg->calcZNCC(*leftImg, rightDispImg);
    CHECK_ERROR(success, "Error calculating ZNCC for the right image.")
    ptimer.printTime();

    // these have become unnecessary at this point
    delete leftImg; // works only if we used: Image leftImage = new Image();
    delete rightImg;

    // TODO: If we use OCL, put benchmark here.

    ptimer.reset();
    success = leftDispImg->save("img/2-disparity-l.png");
    CHECK_ERROR(success, "Error saving the left image to disk.")
    success = rightDispImg->save("img/2-disparity-r.png");
    CHECK_ERROR(success, "Error saving the right image to disk.")
    ptimer.printTime();

    return 0;

    // 5. Cross checking

    ptimer.reset();
    finalImg.crossCheck(*leftDispImg, *rightDispImg);
    CHECK_ERROR(success, "Error in cross checking.")
    ptimer.printTime();

    // these have become unnecessary at this point
    delete leftDispImg;
    delete rightDispImg;

    // TODO: If we use OCL, put benchmark here.

    ptimer.reset();
    success = finalImg.save("img/3-cross-checked.png");
    CHECK_ERROR(success, "Error saving image to disk.")
    ptimer.printTime();

    return EXIT_SUCCESS;  // TODO: Testing only!!

    // 6. Occlusion filling

    ptimer.reset();
    success = finalImg.occlusionFill();
    CHECK_ERROR(success, "Error in occlusion filling.")
    ptimer.printTime();

    // TODO: If we use OCL, put benchmark here.

    ptimer.reset();
    success = finalImg.save("img/4-occlusion-filled.png");
    CHECK_ERROR(success, "Error saving image to disk.")
    ptimer.printTime();

    return EXIT_SUCCESS;
}
