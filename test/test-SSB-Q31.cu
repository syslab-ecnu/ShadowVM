/*
 * test-SSB-Q31.cu
 *
 *  Created on: 2019年4月4日
 *      Author: imdb
 */





 #include "test.hpp"
#include "../com/cpp/com.hpp"

pair<ObjectInfo, Object*> query_date_Q31() {
	auto date = load_date();
	ObjectInfo schema = CreateSchema({TInt32, TInt32, TString}, {4,  4,  40});
	string prediction = "$(0.4)>1995 && $(0.4)<=1997";
	PipelineTest query("query_date");
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, date.first);
	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction}}, schema);
	query.AddOperator(OperatorInfo::Map, {{"select", "$(0.0), $(0.4)"}}, schema);
	query.AddOperator(OperatorInfo::Sink, {}, schema);
	return make_pair(schema, query.Execute(date.second, {},true, false));
}

pair<ObjectInfo, Object*> query_part_Q31() {
	auto part = load_part();
	//part.second->Print();
	ObjectInfo schema = CreateSchema({TInt32, TString, TString}, {4,  10,  40});
	string prediction = "StrCmp($(0.3),\"MFGR#2221\",8)==0";
	PipelineTest query("query_part");
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, part.first);
	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction}}, schema);
	query.AddOperator(OperatorInfo::Map, {{"select", "$(0.0), $(0.2)"}}, schema);
	query.AddOperator(OperatorInfo::Sink, {}, schema);
	return make_pair(schema, query.Execute(part.second,{}, true, false));
}

pair<ObjectInfo, Object*> query_supplier_Q31() {
	auto supplier = load_supplier();
	ObjectInfo schema = CreateSchema({TInt32, TString, TString}, {4,  15,  40});
	string prediction = "StrCmp($(0.5),\"ASIA\",15)==0";
	PipelineTest query("query_supplier");
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, supplier.first);
	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction}}, schema);
	query.AddOperator(OperatorInfo::Map, {{"select", "$(0.0), $(0.5)"}}, schema);
	query.AddOperator(OperatorInfo::Sink, {}, schema);
	return make_pair(schema, query.Execute(supplier.second,{}, true, false));
}

pair<ObjectInfo, Object*> query_customer_Q31() {
	auto supplier = load_customer();
	ObjectInfo schema = CreateSchema({TInt32, TString, TString}, {4,   15, 40});
	string prediction = "StrCmp($(0.5),\"ASIA\",15)==0";
	PipelineTest query("query_customer");
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, supplier.first);
	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction}}, schema);
	query.AddOperator(OperatorInfo::Map, {{"select", "$(0.0),$(0.4)"}}, schema);
	query.AddOperator(OperatorInfo::Sink, {}, schema);
	return make_pair(schema, query.Execute(supplier.second,{}, true, false));
}


void test_SSB_Q31(bool isGPU) {
 	auto lineorder = load_lineorder();
	auto date = query_date_Q31();   //date.second->Print();
//	auto part = query_part_Q31();  //  part.second->Print();
	auto supplier = query_supplier_Q31(); // supplier.second->Print();
	auto customer = query_customer_Q31();  customer.second->Print();
	//return;
	//lineorder.second->To(SharedMem);
	ObjectInfo result = CreateSchema({
	ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32,
	ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32},
		{ 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,4, 4});

	PipelineTest query("test_SSB_Q31");
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, lineorder.first);
	query.AddOperator(OperatorInfo::JoinWith, {{"lkey", "$(0.4)"}, {"rkey","$(2.0)"}}, result);
	query.AddOperator(OperatorInfo::JoinWith, {{"lkey", "$(0.5)"}, {"rkey","$(1.0)"}}, result);
	query.AddOperator(OperatorInfo::JoinWith, {{"lkey", "$(0.2)"}, {"rkey","$(3.0)"}}, result);
   query.AddOperator(OperatorInfo::ReduceBy, {{"dim","$(1.1), $(2.1)"}, {"metric","$(0.12)/100"}}, result);
	query.AddOperator(OperatorInfo::Sink, {}, result);
	Object* rc = query.Execute(lineorder.second, {date , supplier, customer},true, true, isGPU);
	//rc->Print();
//	PipelineTest query("query");
//	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, lineorder.first);
//	query.AddOperator(OperatorInfo::JoinWith, {{"lkey", "$(0.2)"}, {"rkey","$(1.0)"}}, result);
//   query.AddOperator(OperatorInfo::JoinWith, {{"lkey", "$(0.4)"}, {"rkey","$(2.0)"}}, result);
//	query.AddOperator(OperatorInfo::JoinWith, {{"lkey", "$(0.5)"}, {"rkey","$(3.0)"}}, result);
//   query.AddOperator(OperatorInfo::ReduceBy, {{"dim",  "$(3.1),$(2.1),$(1.1)"}, {"metric","$(0.12)/100"}}, result);
//	query.AddOperator(OperatorInfo::Sink, {}, result);
//	Object* rc = query.Execute(lineorder.second, {customer, supplier, date },true, true, true);
// 	rc->Print();
	return ;

}






