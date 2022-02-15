#pragma once

#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

/**
 * A simple wrapper class for accessing OpenCL.
 **/
class MiniOCL
{
	unsigned int device_type;
	std::string kernelName;
	// const char **kernelSource;

	// host (image) locations
    void *inputData;
    void *outputData;

	// device (image) buffers
    cl_mem inputBuffer;
    cl_mem outputBuffer;

    // OpenCL objects
    cl_platform_id platform;            // OpenCL platform
    cl_device_id device_id;             // device ID
    cl_context context;                 // context
    cl_command_queue queue;             // command queue
    cl_program program;                 // program
    cl_kernel kernel;                   // kernel
    cl_event kernelEvent;				// kernel event (profiling)

    size_t imageWidth;
	size_t imageHeight;

public:
	MiniOCL();
	bool initialize(cl_device_type device_type);
	bool buildKernel(const std::string *name, const char **source);
	bool setWorkGroupSize(size_t localWidth, size_t localHeight);
	bool setImageBuffers(void *in, void *out, size_t width, size_t height);
	bool executeKernel(size_t localWidth, size_t localHeight);
	bool displayDeviceInfo(cl_device_id device_id = NULL);
	double getExecutionTime();
	~MiniOCL();
};
