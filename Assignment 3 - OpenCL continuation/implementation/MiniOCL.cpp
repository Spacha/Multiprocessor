#include "MiniOCL.hpp"

using std::cout;
using std::endl;

MiniOCL::MiniOCL()
{
    // initialize the object...
}
MiniOCL::~MiniOCL()
{
    // destroy the object, release memory...
}

/**
 * Initializes OpenCL.
 **/
bool MiniOCL::initialize(cl_device_type device_type)
{
    cl_int err = CL_SUCCESS;

    this->device_type = device_type;

    // bind to platform
    err |= clGetPlatformIDs(1, &platform, NULL);
 
    // get ID for the device (currently takes the first device automatically)
    err |= clGetDeviceIDs(platform, device_type, 1, &device_id, NULL);

    if (err != CL_SUCCESS)
        return false;
 
    // create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
 
    // create a command queue
    // this is a list of consecutive pairs of "key" and "value" terminated by 0
    cl_queue_properties properties[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
        0};
    queue = clCreateCommandQueueWithProperties(context, device_id, properties, &err);

    return err == CL_SUCCESS;
}

/**
 * Reads the kernel source code from a file and builds the kernel.
 **/
bool MiniOCL::buildKernel(const char *fileName, const char *kernelName)
{
    cl_int err = CL_SUCCESS;

    // read the kernel source from the file
    std::ifstream kernelFile(fileName);
    std::string source(std::istreambuf_iterator<char>(kernelFile), (std::istreambuf_iterator<char>()));

    // create the compute program from the source buffer
    const char* sourceStr = source.c_str();
    size_t sourceSizes[] = { strlen(sourceStr) };
    program = clCreateProgramWithSource(context, 1, &sourceStr, sourceSizes, &err);

    // build the program executable
    err |= clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, kernelName, &err);

    return err == CL_SUCCESS;
}

/**
 * Executes the initialized and built kernel.
 **/
bool MiniOCL::executeKernel(size_t globalWidth, size_t globalHeight, size_t localWidth, size_t localHeight)
{
    cl_int err = CL_SUCCESS;

    // set work sizes (based on local work size)
    const size_t localWorkSize[2] = { localWidth, localHeight };
    const size_t globalWorkSize[2] = {
        (size_t)ceil(globalWidth/(float)localWorkSize[0]) * localWorkSize[0],
        (size_t)ceil(globalHeight/(float)localWorkSize[1]) * localWorkSize[1]};

    // Execute the kernel over the entire range of the data set
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, localWorkSize,
                                 0, NULL, &kernelEvent);

    // wait for the kernel to finish
    clWaitForEvents(1, &kernelEvent);
 
    // wait for the command queue to get serviced before reading back results
    clFinish(queue);
    this->readOutput();

    return err == CL_SUCCESS;
}

/**
 * Reads the computing output from the OpenCL kernel
 * and stores it to the output address.
 **/
bool MiniOCL::readOutput()
{
    cl_int err = CL_SUCCESS;

    err |= clEnqueueReadImage(queue,
        outImg.buffer, CL_TRUE,
        outImg.origin,
        outImg.region, 0, 0,
        outImg.data, 0, NULL, NULL);

    return err == CL_SUCCESS;
}

/**
 * Sets a simple value as a kernel argument.
 **/
bool MiniOCL::setValue(cl_uint argIndex, void *value, size_t size)
{
    cl_int err = CL_SUCCESS;

    err = clSetKernelArg(kernel, argIndex, size, value);

    return err == CL_SUCCESS;
}

/**
 * Sets an input buffer as a kernel argument.
 **/
bool MiniOCL::setInputBuffer(cl_uint argIndex, void *data, size_t size)
{
    cl_int err = CL_SUCCESS;

    cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR;
    cl_mem buffer = clCreateBuffer(context, flags, size, data, &err);
    //buffers.push_back( clCreateBuffer(context, flags, size, data, &err) );

    err |= clSetKernelArg(kernel, argIndex, sizeof(cl_mem), &buffer);
    // this->setValue(argIndex, (void *)&buffer, sizeof(cl_mem));

    return err == CL_SUCCESS;
}

/**
 * Sets an output buffer as a kernel argument.
 **/
bool MiniOCL::setOutputBuffer(cl_uint argIndex, void *data, size_t size)
{
    cl_int err = CL_SUCCESS;

    err = CL_INVALID_VALUE; // Not implemented!

    // don't forget to use output_buffer_t

    return err == CL_SUCCESS;
}

/**
 * Sets an input image buffer as a kernel argument.
 **/
bool MiniOCL::setInputImageBuffer(cl_uint argIndex, void *data, size_t width, size_t height)
{
    cl_int err = CL_SUCCESS;

    // Pixel format: RGBA, each pixel channel is unsigned 8-bit integer
    static const cl_image_format format = { CL_RGBA, CL_UNORM_INT8 };

    static const cl_image_desc description = {
        CL_MEM_OBJECT_IMAGE2D, width, height, 0, 0, 0, 0, 0, 0, NULL
    };

    cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
    // cl_mem buffer = clCreateImage(context, flags, &format, &description, data, &err);
    cl_mem buffer = clCreateImage(context, flags, &format, &description, data, &err);

    // set the arguments to the kernel
    err |= clSetKernelArg(kernel, argIndex, sizeof(cl_mem), &buffer);
    
    return err == CL_SUCCESS;
}

/**
 * Sets an output image buffer as a kernel argument.
 **/
bool MiniOCL::setOutputImageBuffer(cl_uint argIndex, void *data, size_t width, size_t height)
{
    cl_int err = CL_SUCCESS;

    // Pixel format: RGBA, each pixel channel is unsigned 8-bit integer
    static const cl_image_format format = { CL_RGBA, CL_UNORM_INT8 };

    static const cl_image_desc description = {
        CL_MEM_OBJECT_IMAGE2D, width, height, 0, 0, 0, 0, 0, 0, NULL
    };

    size_t origin[3] = { 0, 0, 0 };
    size_t region[3] = { width, height, 1 };

    cl_mem_flags flags = CL_MEM_WRITE_ONLY;
    outImg.buffer = clCreateImage(context, flags, &format, &description, NULL, &err);
    outImg.data = data;

    // set the image origin and region
    std::copy(origin, origin + 3, outImg.origin);
    std::copy(region, region + 3, outImg.region);

    err |= clSetKernelArg(kernel, argIndex, sizeof(cl_mem), &outImg.buffer);

    return err == CL_SUCCESS;
}

/**
 * Displays OpenCL device information.
 */
bool MiniOCL::displayDeviceInfo(cl_device_id device_id /* = NULL */)
{
    // by default, use the object's target device id
    device_id = device_id ? device_id : this->device_id;
    cl_int err = CL_SUCCESS;

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

    err |= clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(dev_name), &dev_name, NULL);
    err |= clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(dev_type), &dev_type, NULL);
    err |= clGetDeviceInfo(device_id, CL_DEVICE_VENDOR_ID, sizeof(vendor_id), &vendor_id, NULL);
    err |= clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(max_freq), &max_freq, NULL);
    err |= clGetDeviceInfo(device_id, CL_DRIVER_VERSION, sizeof(driver_version), &driver_version, NULL);
    err |= clGetDeviceInfo(device_id, CL_DEVICE_OPENCL_C_VERSION, sizeof(dev_c_version), &dev_c_version, NULL);
    err |= clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(comp_units), &comp_units, NULL);
    err |= clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_dim), &max_work_dim, NULL);
    err |= clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
    err |= clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL);

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

    return err == CL_SUCCESS;
}

/**
 * Returns last kernel execution time in microseconds.
 **/
double MiniOCL::getExecutionTime()
{
    cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    return (time_end - time_start) / 1000.0;
}
