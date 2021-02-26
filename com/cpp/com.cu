/*
 * com.cu
 *
 *  Created on: 2019年3月8日
 *      Author: imdb
 */
#include "com.hpp"

Context CTX;

size_t Context::host_bulk_size_ = DefaultHostBulkSize;
size_t Context::device_bulk_size_ = DefaultDeviceBulkSize;
size_t Context::tuple_per_bulk_  = 2000 * 1000;  // SF20
// size_t Context::tuple_per_bulk_  = 3000 * 1000;  // SF100
//	size_t Context::tuple_per_bulk_ = 300 * 1000; // yahoo
//	string Context::home_path_ = "/home/lizf/data/SVM-JIT-bak/";
string Context::home_path_ = "/home/lizf/data/SVM-JIT/";
string Context::include_path_ = "/home/lizf/ShadowVM5/src/com/cpp/";
size_t Context::cuda_block_num_ = 120;
size_t Context::cuda_thread_num_ = 1024;


string Context::data_path_ = "/home/lizf/data/ssb-20/";
//string Context::data_path_ = "/home/lizf/data/ssb-100/";
//string Context::data_path_ = "/home/lizf/data/ysb/10g/3.2M/";


size_t GetPageSizeAlign(size_t size) {
	size_t page_num = size / PageSize;
	if (size % PageSize != 0) page_num++;
 	return PageSize * page_num;
}
void GetValueOfItemBySwitch(void * des, void * src, ValType type, size_t size) {
	switch(type) {
		case ValType::Byte:	  *(char*)des =  *(char*)src; break;
		case ValType::Int16:	  *(short*)des =  atoi((char*)src); break;
		case ValType::Int32:	  *(int*)des =  atoi((char*)src); break;
		case ValType::Int64:	  *(long long*)des =  atol((char*)src); break;
		case ValType::Float32: *(float*)des =  atof((char*)src); break;
		case ValType::Float64: *(double*)des =  atof((char*)src); break;
		case ValType::String:  memcpy(des, src, size); *(char*)(des + size-1) = '\0'; break;
		case ValType::Tensor:  memcpy(des, src, size); break;
	}
}

void GetValueOfItem(void * des, void * src, ValType type, size_t size) {
	if (type == ValType::Byte)	*(char*)des = *(char*)src;
	else if (type == ValType::Int16)	*(short*)des = atoi((char*)src);
	else if (type == ValType::Int32)	*(int*)des = atoi((char*)src);
	else if (type == ValType::Int64)	*(long long*)des = atol((char*)src);
	else if (type == ValType::Float32)	*(float*)des = atof((char*)src);
	else if (type == ValType::Float64)	*(double*)des =  atof((char*)src);
	else if (type == ValType::String) {
		memcpy(des, src, size);
		*(char*)(des + size-1) = '\0';
	} else if ( type == ValType::Tensor)	memcpy(des, src, size);
}

string GetStringOfItem(void * src, ValType type, size_t size) {
	char buffer[64];
	switch(type) {
		case ValType::Byte:	return to_string((long long)*(char*)src);
 		case ValType::Int16:	return to_string((long long)*(short*)src);
		case ValType::Int32:	return to_string((long long)(*(int*)src));
		case ValType::Int64:	return to_string((long long)*(long long*)src);
		case ValType::Float32:	return to_string((double)*(float*)src);
		case ValType::Float64:	return to_string((double)*(double*)src);
		case ValType::String:
			memcpy(buffer, src, size);
			buffer[size] = '\0';
			return string(buffer);
		case ValType::Tensor:
			memcpy(buffer, src, size);
			buffer[size] = '\0';
			return string(buffer);
	}
}

vector<string> StrSplit(const string &code, string spilter) {
	vector<string> ret;
	size_t pos1 = 0;
	size_t pos2 = code.find(spilter);
	pos1 = 0;
	while(pos2 != string::npos) {
		ret.push_back(code.substr(pos1, pos2-pos1));
		pos1 = pos2 + spilter.size();
		pos2 = code.find(spilter, pos1);
	}
	if(pos1 != code.length())
		ret.push_back(code.substr(pos1));
	//cout << ret.size() << endl;
	return ret;
}

