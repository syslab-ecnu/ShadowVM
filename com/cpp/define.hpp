/*
 * define.hpp
 *
 *  Created on: 2019年3月8日
 *      Author: imdb
 */

#ifndef DEFINE_HPP_
#define DEFINE_HPP_
//#include <stdio.h>
//#include <assert.h>
//#include "firenet.pb.h"
//using namespace firenet;

typedef unsigned long long uint64;

const uint64 KB = 1024;
const uint64 MB = 1024*1024;
const uint64 GB = 1024*1024*1024;
const uint64 PageSize = 4 * KB;

const uint64 DefaultHostBulkSize = 16 * MB;
const uint64 DefaultDeviceBulkSize = 64 * MB;

const uint64 MaxColumnNum = 24;
const uint64 MaxDim = 32;
const uint64 MaxBulkInputNum = 8;

const uint64 HashRate = 5; //8;

enum FireValType {
  Byte = 0,
  Int16 = 1,
  Int32 = 2,
  Int64 = 3,
  Float16 = 4,
  Float32 = 5,
  Float64 = 6,
  String = 7,
  Tensor = 8,
};

class Schema {
public:
	FireValType column_type_[MaxColumnNum];
	uint64 column_size_[MaxColumnNum];
	uint64 column_ct_ = 0;
	uint64 dim_[MaxDim];
	uint64 dim_ct_ = 0;
   void AddColumn(int type, uint64 size) {
		column_type_[column_ct_] = static_cast<FireValType>(type);
		column_size_[column_ct_] = size;
		column_ct_++;
	}
   uint64 GetColumnType(int offset) const {return column_type_[offset];}
   uint64 GetColumnSize(uint64 offset) const {return column_size_[offset];}
   uint64 GetColumnCount() const {return column_ct_;}
};

class KeyIndex {
public:
   KeyIndex(){}
   KeyIndex(int obj_off, int col_off):obj_off_(obj_off),col_off_(col_off){}
	int obj_off_;
	int col_off_;
};


class Bulk {
public:
   void SetColumnPos(uint64 col, uint64 offset) {column_pos_[col] = offset;}
//	void * GetColumnBuffer(size_t col) {return buffer_ + column_pos_[col];}
   void Init(void * buffer, uint64 capacity) {
		buffer_ = buffer;
		capacity_ = capacity;
 		for (int i = 0; i < schema_.GetColumnCount(); i++)
			column_buf_[i] = (char*)buffer_ + column_pos_[i];
	}
   uint64 id_;
   uint64 capacity_ = 0;
   uint64 tuple_count_= 0;
   uint64 max_tuple_count_= 0;
   uint64 column_pos_[MaxColumnNum];
	void * column_buf_[MaxColumnNum];
	Schema schema_;
	void * buffer_ = nullptr;
};

#endif /* DEFINE_HPP_ */
