/*
 * template.cpp
 *
 *  Created on: 2019年3月1日
 *      Author: imdb
 */

#include "template.hpp"

string ClassDefineCode::Schema= "\n"\
	"const int MaxColumnNum = 32;\n"\
	"const int MaxDim = 32;\n"\
	"class Schema { \n"\
	"public: \n"\
	"	ValType column_type_[MaxColumnNum]; \n"\
	"	size_t column_size_[MaxColumnNum]; \n"\
	"	size_t column_ct_ = 0; \n"\
	"	size_t dim_[MaxDim]; \n"\
	"	size_t dim_ct_ = 0;  \n"\
	" }; \n";

string ClassDefineCode::Bluk= "\n"\
	"class Bulk { \n"\
	"public:  \n"\
	"	size_t id_; \n"\
	"	size_t capacity_ = 0; \n"\
	"  size_t tuple_count_= 0; \n"\
	"	size_t max_tuple_count_= 0; \n"\
	"	size_t column_pos_[MaxColumnNum]; \n"\
	"	void * column_buf_[MaxColumnNum]; \n"\
	"	Schema schema_; \n"\
	"  void * buffer_ = nullptr; \n"\
	"}; \n";

string ClassDefineCode::ValType = "\n"\
	" enum ValType { \n"\
	" Byte = 0, \n"\
	" Int16 = 1, \n"\
	" Int32 = 2, \n"\
	" Int64 = 3, \n"\
	" Float16 = 4, \n"\
	" Float32 = 5, \n"\
	" Float64 = 6, \n"\
	" String = 7, \n" \
	" Tensor = 8  \n" \
	"};";


string CodeTemplate::LoaderTransform = "\n"\
	"void $(task_id)(int task_id, Bulk * in, Bulk * out) \n"\
	"{   \n "\
	"  char * ptr = (char*)in->buffer_; \n" \
	"   int t = 0; \n"\
	" $(pointer)" \
	" for(int i=0;i<10;i++) { \n"\
	"     ptr += sscanf(ptr,\"$(pattern)\\n\",$(list)); \n"\
	"		$(ptr_move)"\
	" }\n"\
	"}\n";

string CodeTemplate::HostQueryTransform = "\n"\
		"int $(pipe_id)(int task_id, BulkVector in, BulkVector out, uint64 tuple_ct) { \n"\
  		"$(define_variable)"\
		"int j = 0;\n"\
		"int join_index[8];\n"\
		"for(int i = 0; i < tuple_ct; i++) {\n"\
		"$(filter_block)"\
 		"$(join_block)"\
 		"$(select_block)"\
		"$(reduce_block)"\
 		"}\n"\
		"return j;\n"\
		"}\n";



string CodeTemplate::DeviceQueryTransform ="\n"\
"__global__ void $(pipe_id)(int task_id, BulkVector in, BulkVector out, uint64 tuple_ct) { \n"\
"if (blockIdx.x * blockDim.x + threadIdx.x==0) printf(\"device transform: %d,%ld\\n \", task_id,tuple_ct); \n "\
"$(define_variable)"\
"int join_index[8];\n"\
 "int i = blockIdx.x * blockDim.x + threadIdx.x;\n"\
"if(i < tuple_ct) {\n"\
"$(filter_block)"\
"$(join_block)"\
"$(select_block)"\
"$(reduce_block)"\
"}\n"\
 "}\n";


//string CodeTemplate::DeviceQueryTransform ="\n"\
//"__global__ void $(pipe_id)(int task_id, BulkVector in, BulkVector out, size_t tuple_ct) { \n"\
//"$(define_variable)"\
//"int join_index[8];\n"\
//"for (int l = 0;  gridDim.x * blockDim.x * (l) <= tuple_ct ; l++ ) {\n"\
//"int i = FireNetThreadID(l);\n"\
//"if(i < tuple_ct) {\n"\
//"$(filter_block)"\
//"$(join_block)"\
//"$(select_block)"\
//"$(reduce_block)"\
//"}\n"\
// "if(task_id == 1 &&blockIdx.x==0 && threadIdx.x==0) printf(\"task %d:%d,%d\\n\",task_id,l,i);\n"\
//"}\n"\
//"}\n";


//string CodeTemplate::DeviceQueryTransform ="\n"\
//"__global__ void $(pipe_id)(int task_id, BulkVector in, BulkVector out, size_t tuple_ct ) { \n"\
//"if (blockIdx.x * blockDim.x + threadIdx.x==0) printf(\"device transform: %d\\n \", task_id); \n "\
//	"$(define_variable)"\
//	"int join_index[8];\n"\
//	"for (int l = 0; ; l++ ) {\n"\
//	"int i = FireNetThreadID(l);\n"\
//	"if(i < tuple_ct) {\n"\
//	"$(filter_block)"\
//	"$(join_block)"\
//	"$(select_block)"\
//	"$(reduce_block)"\
//	"} else break;\n"\
// 	"}\n"\
//"}\n";

 /*"\n"\
	"extern \"C\" {"\
	"bool $(task_id)(int task_id, Bulk * in, Bulk * out) \n"\
	"{ char * ptr = (char*)in->buffer_; \n" \
	" $(pointer)" \
	" for(int i=0; ptr < (char*)in->buffer_+in->capacity_ && i < out->max_tuple_count_;i++) \n"\
	"     ptr += sscanf(ptr,\"$(pattern)\",$(list)); \n"\
	" return true;\n"\
	"}\n"\
	"}";
*/






