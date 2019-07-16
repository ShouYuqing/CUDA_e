// implementation of some operations using CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>

__global__ void show()
{
	printf("block_id.x: %d , block_id.y: %d ,block_id.z: %d ,thread_id.x: %d ,thread_id.y: %d ,thread_id.z: %d \n",
		blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}

// 1d block && 2d grid index
__global__ void array_2d_grid(int * arr)
{
	int b_d = blockDim.x;
	int offset1 = b_d*(blockIdx.x);
	int offset2 = b_d*gridDim.x*blockIdx.y;
	int gid = threadIdx.x + offset1 + offset2;
	printf("thread_id: %d, block_x: %d, block_y: %d, gid: %d : %d \n", threadIdx.x, blockIdx.x, blockIdx.y, gid, arr[gid]);
}

// 2d block && 2d grid index
__global__ void array_2dblock(int * arr)
{
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int offset1 = blockDim.x*blockDim.y*(blockIdx.x);
	int offset2 = blockDim.x*blockDim.y*gridDim.x*blockIdx.y;
	int gid = tid + offset1 + offset2;
	printf("gid: %d: %d \n", gid, arr[gid]);
}

// add argument: size 
__global__ void array_2dblock2(int * arr, int size)
{
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
	int offset1 = blockDim.x*blockDim.y*(blockIdx.x);
	int offset2 = blockDim.x*blockDim.y*(gridDim.x*blockIdx.y);
	int gid = tid + offset1 + offset2;
	if (gid < size)
		printf("the gid is smaller than size! gid: %d: %d \n", gid, arr[gid]);
	else
		;
}

// 3d grid && 3d block index
__global__ void array_3d(int * arr, int size)
{
	int tid = threadIdx.y*(blockDim.x*blockDim.z) + threadIdx.z*blockDim.x + threadIdx.x;
	int offset1 = blockDim.x*blockDim.y*blockDim.z*(blockIdx.x);;
	int offset2 = blockDim.x*blockDim.y*blockDim.z*(blockIdx.z*gridDim.x);
	int offset3 = blockDim.x*blockDim.y*blockDim.z*(blockIdx.y*gridDim.x*gridDim.z);
	int gid = tid + offset1 + offset2 + offset3;
	if (gid < size)
		printf("gid: %d : %d \n", gid, arr[gid]);
}

// 2d index to sum an array
__global__ void sum_array(int *a, int * b, int * arr, int size)
{
	int gid = threadIdx.x + blockIdx.x*blockDim.x;
	if (gid < size)
	{
		arr[gid] = a[gid] + b[gid];
		printf("gid: %d, a: %d, b: %d, a+b=%d\n", gid, a[gid], b[gid], arr[gid]);
	}
}
int main()
{
	int * arr1;
	int * arr2;
	int * arr3;
	arr1 = (int*)malloc(sizeof(int) * 64);
	arr2 = (int*)malloc(sizeof(int) * 64);
	arr3 = (int*)malloc(sizeof(int) * 64);
	for (int i = 0; i < 64; i++)
	{
		arr1[i] = (int)(rand() & 0xff);
		arr2[i] = (int)(rand() & 0xff);
	}
	int arr_size = 64;
	int arr_byte_size = sizeof(int) * arr_size;
	// device memory
	int *d_arr1;
	cudaMalloc((void**)&d_arr1, arr_byte_size);
	int *d_arr2;
	cudaMalloc((void**)&d_arr2, arr_byte_size);
	int *d_arr3;
	cudaMalloc((void**)&d_arr3, arr_byte_size);
	// copy the data from host to device
	cudaMemcpy(d_arr1, arr1, arr_byte_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_arr2, arr2, arr_byte_size, cudaMemcpyHostToDevice);
	dim3 grid(16);
	dim3 block(4);
	sum_array << <grid, block >> > (d_arr1, d_arr2, d_arr3, 64);
	cudaMemcpy(arr3, d_arr3, arr_byte_size, cudaMemcpyDeviceToHost);
	for (int i = 0; i<64; i++)
	{
		printf("result%d: %d\n", i, arr3[i]);
	}
	cudaDeviceSynchronize();
	cudaFree(d_arr1);
	cudaFree(d_arr2);
	cudaFree(d_arr3);
	cudaDeviceReset();
	return 0;
}

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//
//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//    int i = threadIdx.x;
//    c[i] = a[i] + b[i];
//}
//
//int main()
//{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
