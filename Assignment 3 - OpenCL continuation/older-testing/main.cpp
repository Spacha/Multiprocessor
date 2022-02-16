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
#define COMPUTE_DEVICE  TARGET_GPU      // TARGET_NONE, TARGET_GPU, TARGET_CPU; Compute device to be used in parallelization
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

// OpenCL kernel. Each work item takes care of one element of C.
const char *kernelGrayscale = "\n" \
"__kernel void grayscale(__global unsigned int *in,                           \n" \
"                        __global unsigned int *out                           \n" \
"                        const unsigned int n)                                \n" \
"{                                                                            \n" \
"    // get our global thread ID                                              \n" \
"    int id = (get_global_id(1) * get_global_size(0)) + get_global_id(0);     \n" \
"    if (if % 4)                                                              \n" \
"       return;                                                               \n" \
"                                                                             \n" \
"    if (id < n)                                                              \n" \
"        out[id] = in[id];                                                    \n" \
"}                                                                            \n";

const char *kernelGaussian = "\n" \
"__kernel void gaussian_filter(__read_only image2d_t srcImg,                \n" \
"                              __write_only image2d_t dstImg,               \n" \
"                              sampler_t sampler,                           \n" \
"                              int width, int height)                       \n" \
"{                                                                          \n" \
"    float4 clr;                                                            \n" \
"    float2 coords;                                                         \n" \
"                                                                           \n" \
"    clr = read_imagef(srcImg, sampler, coords);                            \n" \
"    write_imagef(dstImg, coords, &clr);                                    \n" \
"                                                                           \n" \
"}                                                                          \n"
"\n";

const char *kernelGaussianEiToimi = "\n" \
"__kernel void gaussian_filter(__read_only image2d_t srcImg,                \n" \
"                      __write_only image2d_t dstImg,                       \n" \
"                      sampler_t sampler,                                   \n" \
"                      int width, int height)                               \n" \
"{                                                                          \n" \
"    // Gaussian Kernel is:                                                 \n" \
"    // 1 2 1                                                               \n" \
"    // 2 4 2                                                               \n" \
"    // 1 2 1                                                               \n" \
"    float kernelWeights[9] = { 1.0f, 2.0f, 1.0f,                           \n" \
"        2.0f, 4.0f, 2.0f,                                                  \n" \
"        1.0f, 2.0f, 1.0f };                                                \n" \
"    int2 startImageCoord = (int2) (get_global_id(0) - 1,                   \n" \
"                                   get_global_id(1) - 1);                  \n" \
"    int2 endImageCoord = (int2) (get_global_id(0) + 1,                     \n" \
"                                 get_global_id(1) + 1);                    \n" \
"    int2 outImageCoord = (int2) (get_global_id(0),                         \n" \
"                                 get_global_id(1));                        \n" \
"    if (outImageCoord.x < width && outImageCoord.y < height)               \n" \
"    {                                                                      \n" \
"        uchar4 outColor = (uchar4)(100, 100, 100, 100);                    \n" \
"                                                                           \n" \
"        // Write the output value to image                                 \n" \
"        write_imagef(dstImg, outImageCoord, outColor);                     \n" \
"    }                                                                      \n" \
"}                                                                          \n";

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
class OpenCL
{
public:
    void init()
    {
        //
    }

    void execute()
    {
        //
    }

    void teardown()
    {
        //
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

    /**
     * Creates a greyscale copy of the image to greyImg and makes the image opaque.
     * Uses NTSC formula.
     **/
    void convertToGrayscale()
    {
        pixel_t p;
        unsigned char grayVal;

        cout << "Transforming image to grayscale...";

#if COMPUTE_DEVICE == TARGET_NONE /* No parallelization */
        unsigned int i;

        for (unsigned int y = 0; y < this->height; y++)
        {
            for (unsigned int x = 0; x < this->width; x++)
            {
                p = this->getPixel(x, y);
                grayVal = ceil(0.299*p.red + 0.587*p.green + 0.114*p.blue);

                // this->putPixel(x, y, grayVal);
                i = 4*(y*width + x); // could also just use: i++
                this->image[i]   = grayVal;
                this->image[i+1] = grayVal;
                this->image[i+2] = grayVal;
                this->image[i+3] = 0xff;

                // grayImg[i] = grayVal << 24 | grayVal << 16 << grayVal << 8 | 0xff;
                /*
                grayImg->image.push_back( grayVal );
                grayImg->image.push_back( grayVal );
                grayImg->image.push_back( grayVal );
                grayImg->image.push_back( 0xff );
                */
            }
        }

        this->colorType = LCT_GREY; // update color type

#else /* OpenCL (GPU or CPU) */

        ////////////////////////////////////////////////////////////////////////////////
        /*
        2988 = 36*83
        2008 = 8 *251
        */

        cl_int err;

        cl_platform_id platform;            // OpenCL platform
        cl_device_id device_id;             // device ID
        cl_context context;                 // context
        cl_command_queue queue;             // command queue
        cl_program program;                 // program
        cl_kernel kernel;                   // kernel

        // Connect to a compute device
        //
        // bind to platform
        err = clGetPlatformIDs(1, &platform, NULL);

        // get ID for the device
        err = clGetDeviceIDs(platform, TARGET_DEVICE_TYPE, 1, &device_id, NULL);


        // Create a compute context
        //
        context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);


        // Create a command commands
        //
        queue = clCreateCommandQueue(context, device_id, NULL, &err);


        // Create the compute program from the source buffer
        //
        program = clCreateProgramWithSource(context, 1, (const char **)&kernelGaussian, NULL, &err);

        // Build the program executable
        //
        err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        cout << "Kernel created" << endl;

        // Create the compute kernel in the program we wish to run
        //
        kernel = clCreateKernel(program, "gaussian_filter", &err);
        if (!kernel || err != CL_SUCCESS)
        {
            printf("Error (%d): Failed to create compute kernel!\n", err);
            exit(1);
        }

        size_t static width = 2;
        size_t static height = 3;
        unsigned char *buffer = new unsigned char[width * height * 4](); // create an empty array of bytes
        // unsigned char *buffer = (unsigned char *)malloc(sizeof(unsigned char)*width * height * 4);

        // static const cl_image_format format = { CL_RGBA, CL_UNORM_INT8 };

        vector<unsigned char> input;
        vector<unsigned char> output;
        for (unsigned int i = 0; i < (int)width*height; i++)
        {
            input.push_back(0xff);
            input.push_back(0xff);
            input.push_back(0xff);
            input.push_back(0xff);
        }

        cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 0, NULL, &err);
        cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 0, NULL, &err);

        // write the data set into the input array in device memory
        err = clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0,
                                   4 * width * height, (void *)&input, 0, NULL, NULL);

        /*
        cl_mem inputImage = clCreateImage2D(context,
                                             CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format,
                                             width, height, 0,
                                             buffer,
                                             &err); //deprecated


        cl_mem outputImage = clCreateImage2D(context,
                                             CL_MEM_WRITE_ONLY, &format,
                                             width, height, 0,
                                             (void *)&this->image,
                                             &err); //deprecated
        //cl_mem outputImage = clCreateFromGLTexture(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, img, NULL);
        */
        if (err != CL_SUCCESS)
        {
            printf("Error creating CL image object");
            return;
        }

        cl_sampler sampler = clCreateSampler(context,
                                  CL_TRUE, // Non-normalized coordinates
                                  CL_ADDRESS_CLAMP_TO_EDGE,
                                  CL_FILTER_NEAREST,
                                  &err);
        if (err != CL_SUCCESS)
        {
            printf("Error creating CL sampler object");
            return;
        }


        // Write our data set into the input array in device memory
        //

        // Set the arguments to our compute kernel
        //
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_sampler), &sampler);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_int), &width);
        err |= clSetKernelArg(kernel, 4, sizeof(cl_int), &height);

        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to set kernel arguments! %d\n", err);
            return;
        }

        size_t globalSizeArr[2] = { width, height };
        err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSizeArr, NULL, 0, NULL, NULL);

        clFinish(queue);

        // read the results from the device
        clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, width * height * 4, (void *)&this->image, 0, NULL, NULL );

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
    
    // imgGray.createEmpty(img.width, img.height);
    // imgFiltered.createEmpty(img.width, img.height);

    vector<unsigned char> grayVect;
    grayVect.reserve(img.width * img.height); // only 1 channel

#if 0
    ///////////////////////////////////////////////////////////////////////////
    // OpenCL
    ///////////////////////////////////////////////////////////////////////////

     // device input buffers
    cl_mem d_a;
    cl_mem d_b;
    // device output buffer
    cl_mem d_c;
 
    cl_platform_id platform;            // OpenCL platform
    cl_device_id device_id;             // device ID
    cl_context context;                 // context
    cl_command_queue queue;             // command queue
    cl_program program;                 // program
    cl_kernel kernel;                   // kernel
 
    // size, in bytes, of each matrix
    size_t bytes = N*N*sizeof(double);

    // VECTORS HERE...
 
    size_t globalSize, localSize;
    cl_int err;
 
    // number of work items in each local work group
    localSize = 10;
 
    // number of total work items - localSize must be devisor
    // globalSize = ceil(n/(float)localSize)*localSize;
    globalSize = N;     // NOTE: globalSize must be divisible to localSize
 
    // bind to platform
    err = clGetPlatformIDs(1, &platform, NULL);
 
    // get ID for the device
    err = clGetDeviceIDs(platform, TARGET_DEVICE_TYPE, 1, &device_id, NULL);
 
    // create a context  
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
 
    // create a command queue 
    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
 
    // create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1,
                            (const char **)&kernelSource, NULL, &err);
 
    // build the program executable 
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
 
    // create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "add_matrix", &err);
 
    // create the input and output arrays in device memory for our calculation
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
 
    // write the data set into the input array in device memory
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                                   bytes, h_a, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                   bytes, h_b, 0, NULL, NULL);
 
    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
#endif
    ///////////////////////////////////////////////////////////////////////////

    // convert the original image to grayscale and store it to imgGray
    ptimer.start();
    img.convertToGrayscale();
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
    img.save("gray.png");
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
