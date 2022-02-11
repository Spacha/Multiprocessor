#define CL_USE_DEPRECATED_OPENCL_2_0_APIS // required for using olf APIs

#include <windows.h>
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

/**
 * Print matrix @mtx having @rows rows and @cols columns.
 */
void Print_Matrix(double *mtx, size_t rows, size_t cols)
{
    for (int i = 0; i < rows*cols; i++)
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
    printf("\tMax. work item sizes:          %I64u/%I64u/%I64u\n", workitem_size[0], workitem_size[1], workitem_size[2]);
    printf("\tMax. work group size:          %I64u\n", workgroup_size);

    return status;
}

////////////////////////////////////////////////////////////////////////////////

// OpenCL kernel. Each work item takes care of one element of C.
const char *kernelSource = "\n" \
"__kernel void add_matrix(__global double *a,                                 \n" \
"                         __global double *b,                                 \n" \
"                         __global double *c)                                 \n" \
"{                                                                            \n" \
"    // get our global thread ID                                              \n" \
"    int id = (get_global_id(1) * get_global_size(0)) + get_global_id(0);     \n" \
"                                                                             \n" \
"    c[id] = a[id] + b[id];                                                   \n" \
"}                                                                            \n" \
"\n";

int main( int argc, char* argv[] )
{
    // size of a side of the matrices (N*N)
    unsigned int N = 100;
 
    // host input vectors
    double *h_a;
    double *h_b;
    // host output vector
    double *h_c;
 
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
 
    // allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);
 
    // populate the matrices
    int i;
    for (i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_a[i+j*N] = i+j*N;
            h_b[i+j*N] = i+j*N;
        }
    }
 
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

    // Execute the kernel over the entire range of the data set
    cl_event event;
    size_t globalSizeArr[2] = { globalSize, globalSize };       // 100x100 global work groups
    size_t localSizeArr[2] = { localSize, localSize };          // 10x10 local work items
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSizeArr, localSizeArr,
                                 0, NULL, &event);

    // wait for the kernel to finish
    clWaitForEvents(1, &event);
 
    // wait for the command queue to get serviced before reading back results
    clFinish(queue);

    cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    // display information
    printf("Sum two %dx%d matrices using OpenCL on %s.\n", N, N, TARGET_DEVICE_TYPE == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU");
    err = displayDeviceInfo(device_id);
    double nanoSeconds = time_end - time_start;
    printf("Execution time: %0.3f us \n", nanoSeconds/1000.0);
 
    // read the results from the device
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
                                bytes, h_c, 0, NULL, NULL );

#if VERBOSE
    printf("A:\n");
    Print_Matrix(h_a, N, N);
    printf("B:\n");
    Print_Matrix(h_b, N, N);
    printf("A + B:\n");
    Print_Matrix(h_c, N, N);
#endif

    // release OpenCL resources
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
 
    // release host memory
    free(h_a);
    free(h_b);
    free(h_c);
 
    return EXIT_SUCCESS;
}
