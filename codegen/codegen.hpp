/*
 * codegen.hpp
 *
 *  Created on: 2019年3月6日
 *      Author: imdb
 */

#ifndef CODEGEN_HPP_
#define CODEGEN_HPP_
#include "../com/cpp/com.hpp"
#include "compiler.hpp"

class Hint{
public:
	bool is_reduce_;
};

class CodeGen {
public:
	static string GenBulkPattern(const ObjectInfo & schema, char col_spliter, bool ignore_end = false);
	static string GenBulkPointerDefine(const ObjectInfo & schema, const string & bulk);
	static string GenBulkPointerMove(const ObjectInfo & schema, const string & bulk);
	static string GenBulkPointerList(const ObjectInfo & schema, const string & bulk);
	static string GenColumnPointer(const string & bulk, int id) {return bulk+"_"+to_string(id);}
	static string GenFromTemplate(const string & code_template, const map<string, string> & dic);
	static string GenHostInclude() ;
	static string GenVariableDefine(const map<string, string> & variable) ;
	static KeyIndex GenKeyIndex(const string & exp);
	static string GenExperssion(map<string, string> & variable,
			const vector<ObjectInfo> & schema, const string & exp, bool isReduce = false);
 	static string GenFilterBlock(map<string, string> & variable,
 			const Pipeline & pipe, const vector<string> & filter_exp, bool isGPU);
 	static string GenJoinBlock(map<string, string> & variable,
 			const Pipeline & pipe, const vector<KeyIndex> & lkey,
 			const vector<KeyIndex> & rkey, bool isGPU);
 	static string GenSelectBlock(map<string, string> & variable,
 			const Pipeline & pipe, const vector<string> & select_exp, bool isGPU);
 	static string GenReduceBlock(map<string, string> & variable,
 			const Pipeline & pipe, const vector<string> & metric_exp,
 			const vector<KeyIndex> & reduce_key, bool isGPU);

	//static string GenPointerDefine(const vector<ObjectInfo> & schema);

	static map<ValType, string> ToNativeType;
};





#endif /* CODEGEN_HPP_ */
