#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VERBOSE 0
#define MTX_SIZE 100

////////////////////////////////////////////////////////////////////////////////
// Macros form measuring execution time.

#define START_TIMING()                                                          \
    LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;                \
    LARGE_INTEGER Frequency;                                                    \
    QueryPerformanceFrequency(&Frequency);                                      \
    QueryPerformanceCounter(&StartingTime);
#define STOP_TIMING(delta_us)                                                   \
    QueryPerformanceCounter(&EndingTime);                                       \
    ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart; \
    ElapsedMicroseconds.QuadPart *= 1000000;                                    \
    ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;                         \
    delta_us = ElapsedMicroseconds.QuadPart;

////////////////////////////////////////////////////////////////////////////////

/**
 * Print matrix @mtx having @rows rows and @cols columns.
 */
void Print_Matrix(double *mtx, size_t rows, size_t cols)
{
	for (int i = 0; i < rows*cols; i++)
	{
		printf("%f", *(mtx++));

		if ((i+1) % cols == 0)
			printf("\n");
		else
			printf(" ");
	}
}

////////////////////////////////////////////////////////////////////////////////

/**
 * Add_Matrix
 *  Sum two matrices
 * 	input:
 * 		@res the resulting matrix
 *  	@A the first matrix to be summed
 *  	@B the second matrix to be summed
 * 	output: the execution time
 */
long long int Add_Matrix(double *res, double *A, double *B, size_t rows, size_t cols)
{
	START_TIMING();

	// traverse rows
	for (int i = 0; i < rows; i++)
	{
		// traverse cols
		for (int j = 0; j < cols; j++)
		{
			// sum corresponding elements in A and B and save the result to C
			res[j*cols+i] = A[j*cols+i] + B[j*cols+i];
		}
	}

	long long int us;
	STOP_TIMING(us);
	return us;
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
	double *mtxA = (double *)malloc(MTX_SIZE*MTX_SIZE*sizeof(double));
	double *mtxB = (double *)malloc(MTX_SIZE*MTX_SIZE*sizeof(double));
	double *mtxC = (double *)malloc(MTX_SIZE*MTX_SIZE*sizeof(double));

	// populate the matrices
	for (int i = 0; i < MTX_SIZE; i++)
	{
		for (int j = 0; j < MTX_SIZE; j++)
		{
			mtxA[i+j*MTX_SIZE] = i+j*MTX_SIZE;
			mtxB[i+j*MTX_SIZE] = i+j*MTX_SIZE;
		}	
	}


	long long int us = Add_Matrix(mtxC, mtxA, mtxB, MTX_SIZE, MTX_SIZE);

#if VERBOSE
	Print_Matrix(mtxA, MTX_SIZE, MTX_SIZE);
	printf("\n");
	Print_Matrix(mtxB, MTX_SIZE, MTX_SIZE);
	printf("\n");
	Print_Matrix(mtxC, MTX_SIZE, MTX_SIZE);
#endif /* VERBOSE */

	printf("Sum two %dx%d matrices without using OpenCL.\n", MTX_SIZE, MTX_SIZE);
	printf("Execution time: %0.3f us", (double)us);
	
	free(mtxA);
	free(mtxB);
	free(mtxC);

    return EXIT_SUCCESS;
}
