#include <iostream>
#include <algorithm>
#include <iterator>

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS // required for using olf APIs
#ifdef __APPLE__
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

int main()
{
    cl_context context = clCreateContext(contextProperties, deviceIdCount, deviceIds.data (), nullptr, nullptr, nullptr);

    cl::ImageFormat format(CL_RGBA, CL_UNSIGNED_INT8);
    cl::Image2D imgIn(context, CL_MEM_READ_ONLY, format, 4, 4);
    return 0;
}
