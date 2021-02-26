/*
 * cpu-tools.hpp
 *
 *  Created on: 2019年3月13日
 *      Author: imdb
 */

#ifndef CPU_TOOLS_HPP_
#define CPU_TOOLS_HPP_

#include <xmmintrin.h>

class BulkVector {
public:
  BulkVector() {}
 BulkVector(Bulk * bulk) {AddBulk(bulk);}
  inline void *GetColumn(KeyIndex key) {return buffer_[key.obj_off_][key.col_off_];}
 inline void * GetColumn(int obj_off, int col_off) {return buffer_[obj_off][col_off];}
 uint64 GetTupleCount(uint64 obj_off) {return tuple_ct_[obj_off];}
 uint64 GetTupleCount() {return tuple_ct_[0];}
   void AddBulk(Bulk * bulk) {
		for (int i = 0; i< bulk->schema_.column_ct_; i++)
			buffer_[bulk_ct_][i] = bulk->column_buf_[i];
		tuple_ct_[bulk_ct_] = bulk->tuple_count_;
		max_tuple_ct_[bulk_ct_] = bulk->max_tuple_count_;
		bulk_ct_++;
	}
   uint64 bulk_ct_ = 0;
	void * buffer_[MaxBulkInputNum][MaxColumnNum];
	uint64 tuple_ct_[MaxBulkInputNum];
	uint64 max_tuple_ct_[MaxBulkInputNum];
};




 inline int StrCmp(const char* lh, const char * rh, int n) {
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


 inline unsigned StrHash(const char * str, int n) {
    const int seed = 31; // 31 131 1313 13131 131313 etc..
    unsigned hash = 0;
    for (int i = 0; *str && i < n; i++)
    	hash = hash * seed + (*str++);
    return (hash & 0x7FFFFFFF);
}

 inline unsigned Hash(const char * str, int n) {
    const int seed = 31; // 31 131 1313 13131 131313 etc..
    unsigned hash = 0;
    for (int i = 0; i < n; i++)
    	hash = hash * seed + (*str++);
    return (hash & 0x7FFFFFFF);
}

 inline void StrCopy(char* des, char * src, int n) {
	while ((n--) > 0)	*(des++) = *(src++);
}

 template<typename T>
inline bool InsertHashTable(T * tab, int pos, unsigned key_size,  int * ht, unsigned ht_size){
 	unsigned index = Hash((char*)(tab + pos), key_size) % ht_size;
	for (int i = 0; i < 10; i++) {
	 	if (ht[index] < 0) {
	 		ht[index] = pos;
	 		return true;
	 	}
		index = (++index) % ht_size;
	}
	return false;
}
template<>
inline bool InsertHashTable<char>(char * tab, int pos, unsigned key_size,  int * ht, unsigned ht_size){
 	unsigned index = StrHash(tab + pos * key_size, key_size) % ht_size;
	for (int i = 0; i < 10; i++) {
	 	if (ht[index] < 0) {
	 		ht[index] = pos;
	 		return true;
	 	}
		index = (++index) % ht_size;
	}
	return false;
}



 template<typename T>
 inline bool InsertHashTableForCPU(T * tab, int pos, unsigned key_size, int * ht, unsigned ht_size){
	int slot_sz = 32;
  	unsigned index = Hash((char*)(tab + pos), key_size) % ht_size;
 	for (int i = 0; i < 10; i++) {
 	 	if (ht[index*slot_sz/sizeof(int)] < 0) {
 	 		ht[index*slot_sz/sizeof(int)] = pos;
 	 		// memcpy((char*)ht+index*slot_sz+sizeof(int), tab+index, key_size);
 	 		 ht[index*slot_sz/sizeof(int)+1] = tab[pos];
 	 		return true;
 	 	}
 		index = (++index) % ht_size;
 	}
 	return false;
  }



 template <typename T>
 inline int Probe(T * key, unsigned key_size, T * tab,  int * ht, unsigned ht_size) {
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

 template <>
  inline int Probe<char>(char * key, unsigned key_size, char * tab, int * ht, unsigned ht_size) {
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




#endif /* CPU_TOOLS_HPP_ */
