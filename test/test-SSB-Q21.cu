/*
 * test-SSB-Q21.cu
 *
 *  Created on: 2019年4月4日
 *      Author: imdb
 */




 #include "test.hpp"
#include "../com/cpp/com.hpp"




pair<ObjectInfo, Object*> query_date_Q21() {
	auto date = load_date();
	ObjectInfo schema = CreateSchema({TInt32, TInt32, TString}, {4,  4,  HashRate * (4+32)});
	PipelineTest query("query_date");
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, date.first);
	query.AddOperator(OperatorInfo::Map, {{"select", "$(0.0), $(0.4)"}}, schema);
	query.AddOperator(OperatorInfo::Sink, {}, schema);
	return make_pair(schema, query.Execute(date.second, {},true, false));
}


pair<ObjectInfo, Object*> query_part_Q21() {
	auto part = load_part();
	//part.second->Print();
	ObjectInfo schema = CreateSchema({TInt32, TString, TString}, {4,  10,  HashRate * (4+32)});
	string prediction = "StrCmp($(0.3),\"MFGR#12\",8)==0 ";
	PipelineTest query("query_part");
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, part.first);
	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction}}, schema);
	query.AddOperator(OperatorInfo::Map, {{"select", "$(0.0), $(0.2)"}}, schema);
	query.AddOperator(OperatorInfo::Sink, {}, schema);
	return make_pair(schema, query.Execute(part.second,{}, true, true));
}

pair<ObjectInfo, Object*> query_supplier_Q21() {
	auto supplier = load_supplier();
	ObjectInfo schema = CreateSchema({TInt32, TString}, {4,   HashRate * (4+32)});
	string prediction = "StrCmp($(0.5),\"AMERICA\",15)==0";
	PipelineTest query("query_supplier");
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, supplier.first);
	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction}}, schema);
	query.AddOperator(OperatorInfo::Map, {{"select", "$(0.0) "}}, schema);
	query.AddOperator(OperatorInfo::Sink, {}, schema);
	return make_pair(schema, query.Execute(supplier.second,{}, true, false));
}

void test_SSB_Q21(bool isGPU) {
 	auto lineorder = load_lineorder();
	auto date = query_date_Q21();  // date.second->Print();
	auto part = query_part_Q21();  //  part.second->Print();
	auto supplier = query_supplier_Q21(); // supplier.second->Print();
	//lineorder.second->To(SharedMem);
	ObjectInfo result = CreateSchema({
	ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32,
	ValType::Int32, ValType::Int32, ValType::Int32},
		{ 4, 4, 4, 4, 4, 4, 4, 4, 4});
	PipelineTest query("test_SSB_Q21");
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, lineorder.first);
	query.AddOperator(OperatorInfo::JoinWith, {{"lkey", "$(0.5)"}, {"rkey","$(1.0)"}}, result);
	query.AddOperator(OperatorInfo::JoinWith, {{"lkey", "$(0.3)"}, {"rkey","$(2.0)"}}, result);
	query.AddOperator(OperatorInfo::JoinWith, {{"lkey", "$(0.4)"}, {"rkey","$(3.0)"}}, result);
   query.AddOperator(OperatorInfo::ReduceBy, {{"dim","$(1.1), $(2.1)"}, {"metric","$(0.12)/100"}}, result);
	query.AddOperator(OperatorInfo::Sink, {}, result);
	Object* rc = query.Execute(lineorder.second, {date, part, supplier},true, true, isGPU);
//	rc->Print();
	return ;

}


