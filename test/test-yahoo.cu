/*
 * test-yahoo.cu
 *
 *  Created on: 2020年8月1日
 *      Author: lizhif
 */

#include "test.hpp"
#include "../com/cpp/com.hpp"

pair<ObjectInfo, Object*> load_ad() {
	ObjectInfo schema = CreateSchema(
			{TString, TString, TString}, {40, 40, 40});
	PipelineTest load("load_ad");
	load.AddOperator(OperatorInfo::Source,{{"type","file"},{"path",Context::data_path_ +"ad-camp-map/"}},schema);
	load.AddOperator(OperatorInfo::Loader,{{"col_spilter","|"},{"row_spilter","\n"}}, schema);
	load.AddOperator(OperatorInfo::Sink, {},schema);
	return make_pair(schema, load.Execute(nullptr, {}, false, false));
}

pair<ObjectInfo, Object*> load_event() {
	ObjectInfo schema = CreateSchema(
	{TString, TString, TString,TString, TString, TString, TString},
	{40, 40, 40, 10, 10, 10, 10});

	PipelineTest load("load_event");
	load.AddOperator(OperatorInfo::Source,{{"type","file"},{"path",Context::data_path_ +"events/"}},schema);
	load.AddOperator(OperatorInfo::Loader,{{"col_spilter","|"},{"row_spilter","\n"}}, schema);
	load.AddOperator(OperatorInfo::Sink, {},schema);
	return make_pair(schema, load.Execute(nullptr, {}, false, false));
}

void test_yahoo(bool isGPU) {
	auto ad = load_ad();
//	ad.second->Print();
	auto event = load_event();
//	event.second->Print();
	ObjectInfo result = CreateSchema( {TString,TString,TString}, {40,40,40});
	PipelineTest query("yahoo");
	string prediction = "StrCmp($(0.4),\"view\",10)==0";
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, event.first);
	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction}}, result);
	query.AddOperator(OperatorInfo::JoinWith, {{"lkey", "$(0.2)"}, {"rkey","$(1.0)"}}, result);
	query.AddOperator(OperatorInfo::Map, {{"select", "$(1.1),$(1.0), $(0.2)"}}, result);
//	query.AddOperator(OperatorInfo::ReduceBy, {{"dim","$(1.1)"},{}}, result);
	query.AddOperator(OperatorInfo::Sink, {}, result);
//	for (auto i = 0; i < 10; i++) {
		Object* rc = query.Execute(event.second, {ad}, true, true, isGPU);
		rc->Print();
//	}
}

