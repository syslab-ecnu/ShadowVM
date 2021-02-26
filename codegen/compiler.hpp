/*
 * jit-compiler.hpp
 *
 *  Created on: 2019年2月21日
 *      Author: imdb
 */

#ifndef JIT_COMPILER_HPP_
#define JIT_COMPILER_HPP_

#include "../com/cpp/com.hpp"


class CodeContext {
public:
	vector<string> include_;
	vector<string> define_;
	vector<string> host_code_;
 	vector<string> device_code_;
	string ExportHostCode ();
	string ExportDeviceCode ();
};

class JitCompiler{
public:
	static void CompileHostFuc(const string & graph_id, const string & code);
	static void CompileDeviceFuc(const string & graph_id, const string & code);
	static TransformFuc GetHostFuc(const string & graph_id, const string & pipeline_id);
	static CUfunction GetDeviceFuc(const string & graph_id, const string & pipeline_id);
	static map<string, void *> host_lib_;
	static map<string,  CUmodule > device_lib_;
	static map<string, TransformFuc> host_fuc_;
	static map<string, CUfunction> device_fuc_;
	static mutex mtx_;
};





/*
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>

#include <stdio.h>
#include <sys/time.h>


using namespace std;

__global__ void add (int * ptr) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	ptr[id] ++;
}
string saxpy = "                                           \n\
 __global__                                         \n\
void saxpy(float a, float *x, float *y, size_t n)   \n\
{                                                               \n\
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;           \n\
  if (tid < n) {                                                \n\
    x[tid] = a * x[tid] + y[tid];                             \n\
  }                                                             \n\
}                                                               \n";

__global__
void test(float a, float *x, float *y, size_t n)
{
 size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
 if (tid < n) {
   x[tid] = a * x[tid] + y[tid];
 }
}

int JIT() {
	struct  timeval begin, end;
	gettimeofday(&begin, nullptr);
	nvrtcProgram prog;
	nvrtcCreateProgram(&prog,         // prog
						   saxpy.c_str(),         // buffer
						   "saxpy.cu",    // name
						   0,             // numHeaders
						   NULL,          // headers
						   NULL);
//	const char *opts[] = {"--gpu-architecture=compute_30",
//							"--fmad=false"};
	const char * opts[] = {"--std=c++11"};
	nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
													  1,     // numOptions
													  opts); // options
	if (compileResult != NVRTC_SUCCESS) {
		cout << "compiling failed" << endl;
		return 0;
	  }
	gettimeofday(&end, nullptr);
	cout << "compiling time:" << (end.tv_sec - begin.tv_sec)*1000
			 + (end.tv_usec - begin.tv_usec)/1000 << " ms"<< endl;


	 gettimeofday(&begin, nullptr);
	 size_t ptxSize;
	 nvrtcGetPTXSize(prog, &ptxSize);
	 char *ptx = new char[ptxSize];
	 nvrtcGetPTX(prog, ptx);
	 nvrtcDestroyProgram(&prog);
	 CUdevice cuDevice;
	 CUcontext context;
	 CUmodule module;
	 CUfunction kernel;
	 cuInit(0);
	 cuDeviceGet(&cuDevice, 0);
	 cuCtxCreate(&context, 0, cuDevice);
	 cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
	 cuModuleGetFunction(&kernel, module, "saxpy");
	 gettimeofday(&end, nullptr);
	 cout << "load kernel time:" << (end.tv_sec - begin.tv_sec)*1000
			 + (end.tv_usec - begin.tv_usec)/1000 << " ms"<< endl;

	 gettimeofday(&begin, nullptr);
	 size_t n = 1024 * 1024;
	 size_t bufferSize = n * sizeof(float);
	 float a = 5.1f;
	 int * hostDataX;
	 int * hostDataY;
	 int * deviceDataX;
	 int * deviceDataY;
	 cudaHostAlloc (&hostDataX, bufferSize, 0);
	 cudaHostAlloc (&hostDataY, bufferSize, 0);
	 cudaMalloc(&deviceDataX, bufferSize) ;
	 cudaMalloc(&deviceDataY, bufferSize) ;
	 cudaMemcpy(deviceDataX, hostDataX, bufferSize, cudaMemcpyHostToDevice);
	 cudaMemcpy(deviceDataX, hostDataX, bufferSize, cudaMemcpyHostToDevice);
	 void *args[] = { &a, &deviceDataX, &deviceDataY, &n };
	 cuLaunchKernel(kernel,
	                   1024, 1, 1,    // grid dim
	                   1024, 1, 1,   // block dim
	                   0, NULL,             // shared mem and stream
	                   args, 0);
	 cudaMemcpy(hostDataX, deviceDataX, bufferSize, cudaMemcpyDeviceToDevice);
	 cudaMemcpy(hostDataY, deviceDataY, bufferSize, cudaMemcpyDeviceToDevice);
	 cudaFree(hostDataX);
	 cudaFree(hostDataY);
	 cudaFree(deviceDataX);
	 cudaFree(deviceDataY);
	 gettimeofday(&end, nullptr);
	 cout << "execute kernel time:" << (end.tv_sec - begin.tv_sec)*1000
			 + (end.tv_usec - begin.tv_usec)/1000 << " ms"<< endl;
	 return 0;
}

void run() {
	printf("hello world\n");
}


*/
#endif /* JIT_COMPILER_HPP_ */
