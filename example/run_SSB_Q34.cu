/*
 * run_SSB_Q34.cu
 */


#include "example.hpp"

void run_SSB_Q34_date(ShadowVMClient & client) {
	run_load_date(client);
	PipelineTest query("q34_date");
	Object* input_obj = ShadowVMService::cached_obj["date"];
	query.pipe_info_.add_pre_pipeline_id("date");
	assert(input_obj != nullptr);
	ObjectInfo schema = CreateSchema({TInt32, TInt32, TString}, {4,  4,  40});
	string prediction = "StrCmp($(0.6),\"Dec1997\",8)==0";
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, input_obj->info_);
	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction}}, schema);
	query.AddOperator(OperatorInfo::Map, {{"select", "$(0.0), $(0.4)"}}, schema);
	query.AddOperator(OperatorInfo::Sink, {}, schema);
	query.SetMode(true, false, false);
	client.RunPipelineCMD(query.pipe_info_);
}

void run_SSB_Q34_part(ShadowVMClient & client) {
	run_load_part(client);
	PipelineTest query("q34_part");
	Object* input_obj = ShadowVMService::cached_obj["part"];
	query.pipe_info_.add_pre_pipeline_id("part");
	assert(input_obj != nullptr);
	ObjectInfo schema = CreateSchema({TInt32, TString, TString}, {4,  10,  40});
	string prediction = "StrCmp($(0.3),\"MFGR#2221\",8)==0";
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, input_obj->info_);
	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction}}, schema);
	query.AddOperator(OperatorInfo::Map, {{"select", "$(0.0), $(0.2)"}}, schema);
	query.AddOperator(OperatorInfo::Sink, {}, schema);
	query.SetMode(true, false, false);
	client.RunPipelineCMD(query.pipe_info_);
}


void run_SSB_Q34_supplier(ShadowVMClient & client) {
	run_load_supplier(client);
	PipelineTest query("q34_supplier");
	Object* input_obj = ShadowVMService::cached_obj["supplier"];
	query.pipe_info_.add_pre_pipeline_id("supplier");
	assert(input_obj != nullptr);
	ObjectInfo schema = CreateSchema({TInt32, TString, TString}, {4,  10,  40});
	string prediction = "StrCmp($(0.3),\"UNITED KI1\",10)==0 || StrCmp($(0.3),\"UNITED KI5\",10)==0";
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, input_obj->info_);
	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction}}, schema);
	query.AddOperator(OperatorInfo::Map, {{"select", "$(0.0), $(0.3)"}}, schema);
	query.AddOperator(OperatorInfo::Sink, {}, schema);
	query.SetMode(true, false, false);
	client.RunPipelineCMD(query.pipe_info_);
}

void run_SSB_Q34_customer(ShadowVMClient & client) {
	run_load_customer(client);
	PipelineTest query("q34_customer");
	Object* input_obj = ShadowVMService::cached_obj["customer"];
	query.pipe_info_.add_pre_pipeline_id("customer");
	assert(input_obj != nullptr);
	ObjectInfo schema = CreateSchema({TInt32, TString, TString}, {4,   10, 40});
	string prediction = "StrCmp($(0.3),\"UNITED KI1\",10)==0 || StrCmp($(0.3),\"UNITED KI5\",10)==0";
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, input_obj->info_);
	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction}}, schema);
	query.AddOperator(OperatorInfo::Map, {{"select", "$(0.0),$(0.3)"}}, schema);
	query.AddOperator(OperatorInfo::Sink, {}, schema);
	query.SetMode(true, false, false);
	client.RunPipelineCMD(query.pipe_info_);
}


void run_SSB_Q34(ShadowVMClient & client) {
	run_SSB_Q34_date(client);
	run_SSB_Q34_supplier(client);
	run_SSB_Q34_customer(client);

	run_load_lineorder(client);

	PipelineTest query("q34_rc");
	query.pipe_info_.add_pre_pipeline_id("lineorder");
	query.pipe_info_.add_pre_pipeline_id("q34_date");
	query.pipe_info_.add_pre_pipeline_id("q34_supplier");
	query.pipe_info_.add_pre_pipeline_id("q34_customer");

	ObjectInfo input = ShadowVMService::cached_obj[query.pipe_info_.pre_pipeline_id(0)]->info_;
	ObjectInfo result = CreateSchema({
	ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32,
	ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32},
		{ 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,4, 4});
 	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, input);
	query.AddOperator(OperatorInfo::JoinWith, {{"lkey", "$(0.4)"}, {"rkey","$(2.0)"}}, result);
	query.AddOperator(OperatorInfo::JoinWith, {{"lkey", "$(0.5)"}, {"rkey","$(1.0)"}}, result);
	query.AddOperator(OperatorInfo::JoinWith, {{"lkey", "$(0.2)"}, {"rkey","$(3.0)"}}, result);
    query.AddOperator(OperatorInfo::ReduceBy, {{"dim","$(1.1), $(2.1)"}, {"metric","$(0.12)/100"}}, result);
	query.AddOperator(OperatorInfo::Sink, {}, result);
	query.SetMode(true, false, true);
	client.RunPipelineCMD(query.pipe_info_);

	Object * rc = ShadowVMService::cached_obj["q34_rc"];
 	rc->Print();

}

