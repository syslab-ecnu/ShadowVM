/*
 * tools.hpp
 *
 *  Created on: 2019年6月28日
 *      Author: lizhif
 */

#ifndef TOOLS_HPP_
#define TOOLS_HPP_

__device__ inline int GetPhyGlobalThreadID() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ inline int GetPhyLocalThreadID() {
	return threadIdx.x;
}

__device__ inline int GetPhyThreadNumPerBlock() {
	return blockDim.x ;
}

__device__ inline int GetPhyBlockID() {
	return blockIdx.x;
}

__device__ inline int GetPhyBlockNum() {
	return gridDim.x;
}

#define ErrChk(code) { Assert((code), __FILE__, __LINE__); }
static inline void Assert(cudaError_t  code, const char *file, int line){
	if(code!=cudaSuccess) {
		printf("CUDA Runtime Error: %s:%d:'%s'\n", file, line, cudaGetErrorString(code));
		exit(EXIT_FAILURE);
	}
}

#define KernelErrChk(){\
		cudaError_t errSync  = cudaGetLastError();\
		cudaError_t errAsync = cudaDeviceSynchronize();\
		if (errSync != cudaSuccess) {\
			  printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));\
			  exit(EXIT_FAILURE);\
		}\
		if (errAsync != cudaSuccess){\
			printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));\
			exit(EXIT_FAILURE);\
		}\
}


#endif /* TOOLS_HPP_ */
