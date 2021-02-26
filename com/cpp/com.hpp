/*
 * com.h
 *
 *  Created on: 2019年2月22日
 *      Author: imdb
 */

#ifndef COM_H_
#define COM_H_

#include <iostream>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include <mutex>
#include <string>
#include <thread>
#include <regex>
#include <initializer_list>
#include <utility>
#include <condition_variable>

#include <regex.h>
#include <stddef.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdlib.h>
#include <malloc.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <nvrtc.h>
#include <cuda.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include "svm.pb.h"
#include "define.hpp"
#include "cpu-runtime.hpp"
//#include "cuda-runtime.hpp"
//#include "../../object/store.hpp"
using namespace std;
using namespace ShadowVM;



class Pipeline;
class Operator;
class Context {
public:
	static size_t host_bulk_size_;
	static size_t device_bulk_size_;
//	static size_t tuple_per_bulk_;  // SF20
	static size_t tuple_per_bulk_ ;  // SF100
//	static size_t tuple_per_bulk_ ; // yahoo
//	static string home_path_ ";
	static string home_path_ ;
	static string include_path_;
	static size_t cuda_block_num_ ;
	static size_t cuda_thread_num_ ;
	static string data_path_;
//	ObjectStore store_;
};


vector<string> StrSplit(const string &code, string spilter);

typedef thrust::tuple<int> Tuple_1;
typedef thrust::tuple<int, int> Tuple_2;
typedef thrust::tuple<int, int, int> Tuple_3;
typedef thrust::tuple<int, int, int, int> Tuple_4;
typedef thrust::tuple<int, int, int, int, int> Tuple_5;
typedef thrust::tuple<int, int, int, int, int, int> Tuple_6;

struct IsValid
{
  __host__ __device__
  bool operator()(const int x)
  {
    return x != 0;
  }
};

struct IsValid_2
{
  __host__ __device__
  bool operator()(const Tuple_2 x) { return thrust::get<1>(x) != 0; }
};

struct IsValid_3
{
  __host__ __device__
  bool operator()(const Tuple_3 x) { return thrust::get<2>(x) != 0; }
};

struct IsValid_4
{
  __host__ __device__
  bool operator()(const Tuple_4 x) { return thrust::get<3>(x) != 0; }
};

struct IsValid_5
{
  __host__ __device__
  bool operator()(const Tuple_5 x) { return thrust::get<4>(x) != 0; }
};

struct IsValid_6
{
  __host__ __device__
  bool operator()(const Tuple_6 x) { return thrust::get<5>(x) != 0; }
};

struct TupleSum_1 {
	__host__ __device__ Tuple_1 operator() (const Tuple_1 & a, const Tuple_1 & b) {
		return Tuple_1(thrust::get<0>(a)+thrust::get<0>(b));
}};
struct TupleEqual_1 {
	__host__ __device__ bool operator() (const Tuple_1 & a, const Tuple_1 & b) {
		return thrust::get<0>(a) == thrust::get<0>(b);
} };
struct TupleSum_2 {
	__host__ __device__ Tuple_2 operator() (const Tuple_2 & a, const Tuple_2 & b) {
		return Tuple_2(thrust::get<0>(a)+thrust::get<0>(b),
				thrust::get<1>(a)+thrust::get<1>(b));
}};
struct TupleEqual_2 {
	__host__ __device__ bool operator() (const Tuple_2 & a, const Tuple_2 & b) {
		return thrust::get<0>(a)==thrust::get<0>(b) &&
				thrust::get<1>(a)==thrust::get<1>(b);
} };
struct TupleSum_3 {
	__host__ __device__ Tuple_3 operator() (const Tuple_3 & a, const Tuple_3 & b) {
		return Tuple_3(thrust::get<0>(a)+thrust::get<0>(b),
				thrust::get<1>(a)+thrust::get<1>(b),
				thrust::get<2>(a)+thrust::get<2>(b));
}};
struct TupleEqual_3 {
	__host__ __device__ bool operator() (const Tuple_3 & a, const Tuple_3 & b) {
		return thrust::get<0>(a) == thrust::get<0>(b) &&
				thrust::get<1>(a) == thrust::get<1>(b) &&
				thrust::get<2>(a) == thrust::get<2>(b);
} };

inline thrust::zip_iterator<thrust::tuple<int*>> MakeIterator_1(void * ptr1)  {
	return thrust::make_zip_iterator(thrust::make_tuple((int*)ptr1));
}
inline thrust::zip_iterator<thrust::tuple<int*,int*>> MakeIterator_2(void * ptr1, void * ptr2) {
	return thrust::make_zip_iterator(thrust::make_tuple((int*)ptr1, (int*)ptr2));
}
inline thrust::zip_iterator<thrust::tuple<int*,int*,int*>> MakeIterator_3(void * ptr1, void * ptr2, void *ptr3){
	return thrust::make_zip_iterator(thrust::make_tuple((int*)ptr1, (int*)ptr2, (int*)ptr3));
}


size_t GetPageSizeAlign(size_t size) ;

template<typename T>
void BuildHashTable(T * tab, unsigned tab_sz, unsigned key_sz, int * ht, unsigned ht_sz) {
	for (auto i = 0; i < ht_sz; i++)
		ht[i] = -1;
	for (auto i = 0; i < tab_sz; i++)
		InsertHashTable<T>(tab, i, key_sz, ht, ht_sz);
}


template<typename T>
void BuildHashTableForCPU(T * tab, unsigned tab_sz, unsigned key_sz, int * ht, unsigned ht_sz) {
	ht += 2* ht_sz;
	int slot_sz = 32;
	memset(ht, ht_sz * slot_sz, 0);
	for (auto i = 0; i < ht_sz; i++)
		ht[i * slot_sz/sizeof(int)] = -1;
	for (auto i = 0; i < tab_sz; i++)
		InsertHashTableForCPU<T>(tab, i, key_sz, ht, ht_sz);
}


bool InsertHashTable(char * tab, int pos, unsigned key_size,  int * ht, unsigned ht_size);



void GetValueOfItemBySwitch(void * des, void * src, ValType type, size_t size);

void GetValueOfItem(void * des, void * src, ValType type, size_t size);

string GetStringOfItem(void * src, ValType type, size_t size);


extern Context CTX;

typedef int (*TransformFuc)(int, BulkVector, BulkVector, int);




#endif /* COM_H_ */
