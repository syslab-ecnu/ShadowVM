/*
 * cuda-tools.hpp
 *
 *  Created on: 2019年3月13日
 *      Author: imdb
 */

#ifndef CUDA_TOOLS_HPP_
#define CUDA_TOOLS_HPP_
class BulkVector {
public:
 BulkVector() {}
 BulkVector(Bulk * bulk) {AddBulk(bulk);}
 __device__ inline void *GetColumn(KeyIndex key) {return buffer_[key.obj_off_][key.col_off_];}
 __device__ inline void * GetColumn(int obj_off, int col_off) {return buffer_[obj_off][col_off];}
 __device__ inline size_t GetTupleCount(size_t obj_off) {return tuple_ct_[obj_off];}
 __device__  inline size_t GetTupleCount() {return tuple_ct_[0];}
   void AddBulk(Bulk * bulk) {
		for (int i = 0; i< bulk->schema_.column_ct_; i++)
			buffer_[bulk_ct_][i] = bulk->column_buf_[i];
		tuple_ct_[bulk_ct_] = bulk->tuple_count_;
		max_tuple_ct_[bulk_ct_] = bulk->max_tuple_count_;
		bulk_ct_++;
	}
	size_t bulk_ct_ = 0;
	void * buffer_[MaxBulkInputNum][MaxColumnNum];
	size_t tuple_ct_[MaxBulkInputNum];
	size_t max_tuple_ct_[MaxBulkInputNum];
};




__device__ inline int StrCmp(const char* lh, const char * rh, int n) {
	for (int i = 0; i < n; i++) {
		if(lh[i] == '\0' || rh[i] == '\0')
			return 0;
		if (lh[i] != rh[i])
			return lh[i]-rh[i];
	}
	return 0;

	while(*lh != '\0' && *rh != '\0' && (n--)>0) {
		if(*lh > *rh)	return 1;
		if(*lh > *rh)	return -1;
		lh++;
		rh++;
 	}
	if(*lh=='\0' && *rh!='\0')
		return -1;
	if(*rh=='\0' && *lh!='\0')
		return 1;
	return 0;
}


__device__ inline unsigned StrHash(const char * str, int n) {
    const int seed = 31; // 31 131 1313 13131 131313 etc..
    unsigned hash = 0;
    for (int i = 0; *str && i < n; i++)
    	hash = hash * seed + (*str++);
    return (hash & 0x7FFFFFFF);
}

__device__ inline unsigned Hash(const char * str, int n) {
    const int seed = 31; // 31 131 1313 13131 131313 etc..
    unsigned hash = 0;
    for (int i = 0; i < n; i++)
    	hash = hash * seed + (*str++);
    return (hash & 0x7FFFFFFF);
}

__device__ inline void StrCopy(char* des, char * src, int n) {
	while ((n--) > 0)	*(des++) = *(src++);
}


 template <typename T>
__device__ inline int Probe(T * key, unsigned key_size, T * tab,  int * ht, unsigned ht_size) {
 	unsigned index = Hash((char*)key, key_size) % ht_size;
 	for (int i = 0; i < 10; i++) {
 		int pos = ht[index];
		if (pos < 0)
			return -1;
		else if (pos >= 0 && *(tab + pos) == *key)
		   return ht[index];
		index = (++index) % ht_size;
 	}
	//printf("probe failed\n");
	return -1;
}

/*
 template <typename T>
__device__ inline int Probe(T * key, unsigned key_size, T * tab,  int * ht, unsigned ht_size) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int slot_sz = 32;
 	unsigned index = Hash((char*)key, key_size) % ht_size;
 	for (int i = 0; i < 10; i++) {
 		int pos = ht[index*slot_sz/sizeof(int)];
 	//	if (tid ==0)
 	//		printf("pos:%d\n", pos);
		if (pos < 0)
			return -1;
		else if (pos >= 0 && *(T*)((char*)ht + index*slot_sz+sizeof(int)) == *key)
		   return ht[index*slot_sz/sizeof(int)];
		index = (++index) % ht_size;
 	}
	//printf("probe failed\n");
	return -1;
}*/


 template <>
__device__ inline int Probe<char>(char * key, unsigned key_size, char * tab, int * ht, unsigned ht_size) {
 	unsigned index = StrHash(key, key_size) % ht_size;
 	for (int i = 0; i < 10; i++) {
		if (ht[index] < 0)
			return -1;
		else if (ht[index] >= 0 &&  StrCmp(key, tab + ht[index]*key_size, key_size) == 0)
		   return ht[index];
		index = (++index) % ht_size;
 	}
//	printf("probe failed\n");
	return false;
}
/*

template <>
__device__ inline int Probe<char>(char * key, unsigned key_size, char * tab, int * ht, unsigned ht_size) {
	const int slot_sz = 32;
 	unsigned index = StrHash(key, key_size) % ht_size;
 	for (int i = 0; i < 10; i++) {
		if (ht[index*slot_sz/sizeof(int)] < 0)
			return -1;
		else if (ht[index] >= 0 &&  StrCmp(key, (char*)ht + index *slot_sz + sizeof(int), key_size) == 0)
		   return ht[index];
		index = (++index) % ht_size;
 	}
//	printf("probe failed\n");
	return false;
}
*/


__device__ inline int CudaThreadID() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ inline int FireNetThreadID(int loop) {
	int cuda_thread_num = gridDim.x * blockDim.x;
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
 	return loop * cuda_thread_num + thread_id;
}


#endif /* CUDA_TOOLS_HPP_ */
