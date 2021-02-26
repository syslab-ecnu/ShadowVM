/*
 * test-query-Q1-1.cu
 *
 *  Created on: 2019年4月4日
 *      Author: imdb
 */



 #include "test.hpp"
#include "../com/cpp/com.hpp"

pair<ObjectInfo, Object*> query_date_Q11() {
	auto date = load_date();
	ObjectInfo schema = CreateSchema({TInt32, TInt32, TString}, {4,  4,  40});
	string prediction = "$(0.4)==1993";
	PipelineTest query("query_date");
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, date.first);
	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction}}, schema);
	query.AddOperator(OperatorInfo::Map, {{"select", "$(0.0), $(0.4)"}}, schema);
	query.AddOperator(OperatorInfo::Sink, {}, schema);
	return make_pair(schema, query.Execute(date.second, {},true, false));
}



void test_SSB_Q11(bool isGPU) {
 	auto lineorder = load_lineorder();
	auto date = query_date_Q11();  // date.second->Print();
	//auto part = query_part();  //  part.second->Print();
	//auto supplier = query_supplier();  // supplier.second->Print();
	//lineorder.second->To(SharedMem);
	ObjectInfo result = CreateSchema({
	ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32,
	ValType::Int32, ValType::Int32, ValType::Int32},
		{ 4, 4, 4, 4, 4, 4, 4, 4, 4});
 	string prediction_Q1 = " $(0.8)<25 && $(0.11)>=1 && $(0.11)<=3 ";
	PipelineTest query("test_SSB_Q11");
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, lineorder.first);
	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction_Q1}}, result);
	query.AddOperator(OperatorInfo::JoinWith, {{"lkey", "$(0.5)"}, {"rkey","$(1.0)"}}, result);
   query.AddOperator(OperatorInfo::ReduceBy, {{"dim",""}, {"metric","$(0.9)*$(0.11)/1000"}}, result);
	query.AddOperator(OperatorInfo::Sink, {}, result);
	Object* rc = query.Execute(lineorder.second, {date},true, true, isGPU);
// 	rc->Print();
}
