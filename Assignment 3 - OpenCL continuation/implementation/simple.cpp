#define CL_USE_DEPRECATED_OPENCL_2_0_APIS // required for using olf APIs

#include <windows.h>
#include <iostream>
#include <vector>
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define VERBOSE 0       // show detailed info
#define USE_CPU 0       // use CPU instead of GPU

#if USE_CPU
#define TARGET_DEVICE_TYPE CL_DEVICE_TYPE_CPU
#else
#define TARGET_DEVICE_TYPE CL_DEVICE_TYPE_GPU
#endif

#define CHECK_ERROR(e) if (err) { printf("Error: %d\n", err); err = 0; }

/**
 * Print matrix @mtx having @rows rows and @cols columns.
 */
void Print_Matrix(double *mtx, size_t rows, size_t cols)
{
    for (unsigned int i = 0; i < rows*cols; i++)
    {
        printf("%f", *(mtx++));

        if ((i+1) % cols == 0) 
            printf("\n");       // new row
        else
            printf(" ");        // new column
    }
}

////////////////////////////////////////////////////////////////////////////////

/**
 * Display OpenCL device information.
 */
cl_int displayDeviceInfo(cl_device_id device)
{
    cl_int status;

    char dev_name[512] = {0};
    cl_device_type dev_type;
    cl_uint vendor_id;
    cl_uint max_freq;
    char driver_version[512] = {0};
    char dev_c_version[512] = {0};
    cl_uint comp_units;
    cl_uint max_work_dim;
    size_t workitem_size[3];
    size_t workgroup_size;

    printf("Compute device info:\n");

    status = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dev_name), &dev_name, NULL);
    status = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, NULL);
    status = clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID, sizeof(vendor_id), &vendor_id, NULL);
    status = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(max_freq), &max_freq, NULL);
    status = clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(driver_version), &driver_version, NULL);
    status = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(dev_c_version), &dev_c_version, NULL);
    status = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(comp_units), &comp_units, NULL);
    status = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_dim), &max_work_dim, NULL);
    status = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
    status = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL);

    // translate device type id to text
    char dev_type_str[16];
    switch (dev_type)
    {
    case CL_DEVICE_TYPE_DEFAULT:
        strcpy(dev_type_str, "Default");
        break;
    case CL_DEVICE_TYPE_CPU:
        strcpy(dev_type_str, "CPU");
        break;
    case CL_DEVICE_TYPE_GPU:
        strcpy(dev_type_str, "GPU");
        break;
    case CL_DEVICE_TYPE_ACCELERATOR:
        strcpy(dev_type_str, "Accelerator");
        break;
    default:
        strcpy(dev_type_str, "Unknown");
        break;
    }

    printf("\tDevice name:                   %s\n", dev_name);
    printf("\tDevice type:                   %s\n", dev_type_str);
    printf("\tVendor ID:                     %u\n", vendor_id);
    printf("\tMaximum frequency:             %u MHz\n", max_freq);
    printf("\tDriver version:                %s\n", driver_version);
    printf("\tDevice C version:              %s\n", dev_c_version);
    printf("\tCompute units:                 %u\n", comp_units);
    printf("\tMax. work item dimensions:     %u\n", max_work_dim);
    printf("\tMax. work item sizes:          %u/%u/%u\n", (unsigned int)workitem_size[0], (unsigned int)workitem_size[1], (unsigned int)workitem_size[2]);
    printf("\tMax. work group size:          %u\n", (unsigned int)workgroup_size);

    return status;
}

////////////////////////////////////////////////////////////////////////////////

// See OpenCL Programming Guide p.342.

// OpenCL kernel. Each work item takes care of one element of C.
const char *kernelSource = "\n" \
"__kernel void pixel_stuff(__read_only image2d_t in,                          \n" \
"                         __write_only image2d_t out,                         \n" \
"                         unsigned int width,                                 \n" \
"                         __constant float *ftest,                           \n" \
"                         sampler_t sampler)                                  \n" \
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

int main()
{
    cl_int err;

    // device (image) buffers
    cl_mem inputImg;
    cl_mem outputImg;
 
    cl_platform_id platform;            // OpenCL platform
    cl_device_id device_id;             // device ID
    cl_context context;                 // context
    cl_command_queue queue;             // command queue
    cl_program program;                 // program
    cl_kernel kernel;                   // kernel

    // image dimensions
    size_t width = 2;
    size_t height = 3;
    size_t bytes = 4 * width * height;

    /*
    const unsigned char imgArr[bytes] = {
        0xff,0xff,0xff,0xff,  0xff,0x00,0x00,0xff,  0xff,0xff,0xff,0xff,  0xff,0x00,0x00,0xff,  0xff,0xff,0xff,0xff,  0xff,0x00,0x00,0xff,
        0x40,0x40,0x40,0xff,  0x00,0xff,0x00,0xff,  0x40,0x40,0x40,0xff,  0x00,0xff,0x00,0xff,  0x40,0x40,0x40,0xff,  0x00,0xff,0x00,0xff,
        0x00,0x00,0x00,0xff,  0x00,0x00,0xff,0xff,  0x00,0x00,0x00,0xff,  0x00,0x00,0xff,0xff,  0x00,0x00,0x00,0xff,  0x00,0x00,0xff,0xff
    };
    */
    const unsigned char imgArr[bytes] = {
        0xff,0xff,0xff,0xff,  0xff,0x00,0x00,0xff,
        0x40,0x40,0x40,0xff,  0x00,0xff,0x00,0xff,
        0x00,0x00,0x00,0xff,  0x00,0x00,0xff,0xff
    };
    std::vector <unsigned char> imgData(imgArr, imgArr + bytes);

    unsigned char *imgOut = (unsigned char *)malloc(bytes);

    // allocate memory for each vector on host
    // h_a = (unsigned char *)malloc(bytes);
    // h_b = (unsigned char *)malloc(bytes);
    // h_c = (unsigned char *)malloc(bytes);

    //strncpy((char *)h_a, (char *)img, bytes);

    // Image...

 
    // bind to platform
    err = clGetPlatformIDs(1, &platform, NULL);
 
    // get ID for the device (currently takes the first device automatically)
    err = clGetDeviceIDs(platform, TARGET_DEVICE_TYPE, 1, &device_id, NULL);
 
    // create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
 
    // create a command queue
    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
 
    // create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &err);

    CHECK_ERROR(err)
 
    // build the program executable 
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    CHECK_ERROR(err);
 
    // create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "pixel_stuff", &err);

    // Pixel format: RGBA, each pixel channel is unsigned 8-bit integer
    static const cl_image_format imgFormat = { CL_RGBA, CL_UNORM_INT8 };
    static const cl_image_desc imgDesc = {
        CL_MEM_OBJECT_IMAGE2D,                      // cl_mem_object_type image_type
        width,                                      // size_t image_width
        height,                                     // size_t image_height
        0,                                          // size_t image_depth
        0,                                          // size_t image_array_size
        0,                                          // size_t image_row_pitch; 0 => calculated automatically
        0,                                          // size_t image_slice_pitch; 0 => calculated automatically
        0,                                          // cl_uint num_mip_levels
        0,                                          // cl_uint num_samples
        NULL                                        // cl_mem buffer
    };
    inputImg = clCreateImage(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,    // host can only read this image
        &imgFormat,                                 // image (array) format
        &imgDesc,                                   // image description
        static_cast<void*>(imgData.data()),         // the image data is sourced from here
        &err);
    outputImg = clCreateImage(
        context,
        CL_MEM_WRITE_ONLY,
        &imgFormat,
        &imgDesc,
        NULL,
        &err);

    CHECK_ERROR(err);
 
    // create the input and output arrays in device memory for our calculation
    // d_in = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    // d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);


#if 0
    // NOTE: This is not necessary if CL_MEM_COPY_HOST_PTR is used!
    err |= clEnqueueWriteImage(
        queue,                                  // cl_command_queue command_queue
        inputImg,                               // cl_mem image
        CL_TRUE,                                // cl_bool blocking_write
        origin,                                 // const size_t *origin[3]
        region,                                 // const size_t *region[3]
        0,                                      // size_t input_row_pitch
        0,                                      // size_t input_slice_pitch
        static_cast<void*>(imgData.data()),     // const void *ptr
        0,                                      // cl_uint num_events_in_wait_list
        NULL,                                   // const cl_eventevent_wait_list
        NULL);                                  // cl_event *event
#endif
    CHECK_ERROR(err);

    size_t origin[] = {0,0,0};
    size_t region[] = {width, height, 1};

    /*
    // write the data set into the input array in device memory
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                                   bytes, inputImg, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                   bytes, h_b, 0, NULL, NULL);
    */

    // create sampler for sampling the input image
    cl_sampler sampler = clCreateSampler(
        context,
        CL_FALSE,                               // Coordinate system; non-normalized
        CL_ADDRESS_CLAMP_TO_EDGE,               // Disallow sampling over the edge
        CL_FILTER_NEAREST,                      // Take the nearest true pixel value
        &err);

    unsigned int width_test = 123;
    // float ftest[16] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
    float *ftest = (float *)malloc(16 * sizeof(float));
    ftest[0] = 1;
    ftest[1] = 2;
    ftest[2] = 3;
    ftest[3] = 4;
    ftest[4] = 5;
    ftest[5] = 6;
    ftest[6] = 7;
    ftest[7] = 8;
    ftest[8] = 9;
    ftest[9] = 10;
    ftest[10] = 11;
    ftest[11] = 12;
    ftest[12] = 13;
    ftest[13] = 14;
    ftest[14] = 15;
    ftest[15] = 16;

    cl_mem fbuf = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY,
        16 * sizeof(float),
        NULL,
        NULL);
    clEnqueueWriteBuffer(queue, fbuf, CL_TRUE, 0, 16 * sizeof(float), ftest, 0, NULL, NULL);

    /*
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
                                bytes, h_c, 0, NULL, NULL );
    */
 
    // Set the arguments to our compute kernel
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImg);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputImg);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &width_test);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &fbuf);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_sampler), &sampler);

    CHECK_ERROR(err);

    // size_t localWorkSize[2] = { 16, 16 };
    // size_t globalWorkSize[2] = { RoundUp(localWorkSize[0], width), RoundUp(localWorkSize[1], height) };
    size_t localWorkSize[2] = { 16, 16 }; // 16*16 => 256
    size_t globalWorkSize[2] = {
        (size_t)ceil(width/(float)localWorkSize[0]) * localWorkSize[0],
        (size_t)ceil(height/(float)localWorkSize[1]) * localWorkSize[1]
    };

    // Execute the kernel over the entire range of the data set
    cl_event kernelDoneEvt;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, localWorkSize,
                                 0, NULL, &kernelDoneEvt);


    // wait for the kernel to finish
    clWaitForEvents(1, &kernelDoneEvt);
 
    // wait for the command queue to get serviced before reading back results
    clFinish(queue);

    cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(kernelDoneEvt, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(kernelDoneEvt, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    // display information
    err = displayDeviceInfo(device_id);
    double microSeconds = (time_end - time_start)/1000.0;
    printf("Execution time: %0.3f us \n", microSeconds);

    // read the results from the device
    err |= clEnqueueReadImage(
        queue,                      // cl_command_queue command_queue
        outputImg,                  // cl_mem image
        CL_TRUE,                    // cl_bool blocking_read
        origin,                     // const size_t *origin[3]
        region,                     // const size_t *region[3]
        0,                          // size_t input_row_pitch
        0,                          // size_t input_slice_pitch
        imgOut,                     // const void *ptr
        0,                          // cl_uint num_events_in_wait_list
        NULL,                       // const cl_event event_wait_list
        NULL);                      // cl_event *event

    for (unsigned int i = 0; i < bytes; i++)
    {
        //printf("#%02x%02x%02x%02x ", imgOut[i], imgOut[i+1], imgOut[i+2], imgOut[i+3]);
        printf("%02x", imgOut[i]);
        if ((i+1) % (4*width) == 0)
            printf("\n");
        else if ((i+1) % 4 == 0)
            printf(" ");

    }
    printf("\n");

    printf("OK.\n");
    return 0;

#if VERBOSE
    printf("A:\n");
    Print_Matrix(h_a, N, N);
    printf("B:\n");
    Print_Matrix(h_b, N, N);
    printf("A + B:\n");
    Print_Matrix(h_c, N, N);
#endif

    // release OpenCL resources
    clReleaseMemObject(inputImg);
    clReleaseMemObject(outputImg);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
 
    // release host memory
    free(imgOut);
 
    return EXIT_SUCCESS;
}
