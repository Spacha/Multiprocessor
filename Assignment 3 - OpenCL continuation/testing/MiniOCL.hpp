#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

struct output_buffer_s
{
	cl_mem buffer;
	void *data;
	size_t origin[3];
	size_t region[3];
};
typedef struct output_buffer_s output_buffer_t;

/**
 * A simple wrapper class for accessing OpenCL.
 **/
class MiniOCL
{
	unsigned int device_type;
	std::string kernelName;

    std::vector<cl_mem> buffers;

    cl_mem inputBuffer;
    cl_mem outputBuffer;
    void *out;
    size_t width;
    size_t height;

    // OpenCL objects
    cl_platform_id platform;            // OpenCL platform
    cl_device_id device_id;             // device ID
    cl_context context;                 // context
    cl_command_queue queue;             // command queue
    cl_program program;                 // program
    cl_kernel kernel;                   // kernel
    cl_event kernelEvent;				// kernel event (profiling)

public:
	MiniOCL();
	bool initialize(cl_device_type device_type);
	// bool buildKernel(const std::string *name, const char **source);
	bool buildKernel(const char *fileName, const char *kernelName);
	bool executeKernel(size_t globalWidth, size_t globalHeight, size_t localWidth, size_t localHeight);

	bool readOutput();

	// values
	bool setValue(cl_uint argIndex, void *value, size_t size);
	// buffers
	bool setInputBuffer(cl_uint argIndex, void *data, size_t size);
	bool setOutputBuffer(cl_uint argIndex, void *data, size_t size);
	// image buffers
	bool setInputImageBuffer(cl_uint argIndex, void *data, size_t width, size_t height);
	bool setOutputImageBuffer(cl_uint argIndex, void *data, size_t width, size_t height);

	// bool setWorkGroupSize(size_t localWidth, size_t localHeight);
bool setImageBuffers(void *in, void *out, size_t width, size_t height);
	// bool setInputBuffer(const void *data, size_t size);
	// bool setArg(const void *data, size_t size);
	bool displayDeviceInfo(cl_device_id device_id = NULL);
	double getExecutionTime();

	cl_context getContext();
	cl_command_queue getQueue();
	cl_kernel getKernel();
	~MiniOCL();
};
