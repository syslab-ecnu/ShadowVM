/*
 * run_SSB_Q43.cu
 */





#include "example.hpp"

void run_SSB_Q43_date(ShadowVMClient & client) {
	run_load_date(client);
	PipelineTest query("q43_date");
	Object* input_obj = ShadowVMService::cached_obj["date"];
	query.pipe_info_.add_pre_pipeline_id("date");
	assert(input_obj != nullptr);
	ObjectInfo schema = CreateSchema({TInt32, TInt32, TString}, {4,  4,  40});
	string prediction = "$(0.4)==1997 || $(0.4)==1998";
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, input_obj->info_);
	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction}}, schema);
	query.AddOperator(OperatorInfo::Map, {{"select", "$(0.0), $(0.4)"}}, schema);
	query.AddOperator(OperatorInfo::Sink, {}, schema);
	query.SetMode(true, false, false);
	client.RunPipelineCMD(query.pipe_info_);
}

void run_SSB_Q43_part(ShadowVMClient & client) {
	run_load_part(client);
	PipelineTest query("q43_part");
	Object* input_obj = ShadowVMService::cached_obj["part"];
	query.pipe_info_.add_pre_pipeline_id("part");
	assert(input_obj != nullptr);
	ObjectInfo schema = CreateSchema({TInt32,  TString}, {4,   40});
	string prediction = "StrCmp($(0.3),\"MFGR#14\",8)==0";
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, input_obj->info_);
	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction}}, schema);
	query.AddOperator(OperatorInfo::Map, {{"select", "$(0.0),$(0.4)"}}, schema);
	query.AddOperator(OperatorInfo::Sink, {}, schema);
	query.SetMode(true, false, false);
	client.RunPipelineCMD(query.pipe_info_);
}


void run_SSB_Q43_supplier(ShadowVMClient & client) {
	run_load_supplier(client);
	PipelineTest query("q43_supplier");
	Object* input_obj = ShadowVMService::cached_obj["supplier"];
	query.pipe_info_.add_pre_pipeline_id("supplier");
	assert(input_obj != nullptr);
	ObjectInfo schema = CreateSchema({TInt32, TString}, {4, 40});
	string prediction = "StrCmp($(0.4),\"UNITED STATES\",15)==0 ";
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, input_obj->info_);
	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction}}, schema);
	query.AddOperator(OperatorInfo::Map, {{"select", "$(0.0), $(0.3)"}}, schema);
	query.AddOperator(OperatorInfo::Sink, {}, schema);
	query.SetMode(true, false, false);
	client.RunPipelineCMD(query.pipe_info_);
}

void run_SSB_Q43_customer(ShadowVMClient & client) {
	run_load_customer(client);
	PipelineTest query("q43_customer");
	Object* input_obj = ShadowVMService::cached_obj["customer"];
	query.pipe_info_.add_pre_pipeline_id("customer");
	assert(input_obj != nullptr);
	ObjectInfo schema = CreateSchema({TInt32, TString, TString}, {4,  15,  40});
	string prediction = "StrCmp($(0.5),\"AMERICA\",15)==0 ";
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, input_obj->info_);
	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction}}, schema);
	query.AddOperator(OperatorInfo::Map, {{"select", "$(0.0),$(0.4)"}}, schema);
	query.AddOperator(OperatorInfo::Sink, {}, schema);
	query.SetMode(true, false, false);
	client.RunPipelineCMD(query.pipe_info_);
}


void run_SSB_Q43(ShadowVMClient & client) {
	run_SSB_Q43_date(client);
	run_SSB_Q43_part(client);
	run_SSB_Q43_supplier(client);
	run_SSB_Q43_customer(client);

	run_load_lineorder(client);

	PipelineTest query("q43_rc");
	query.pipe_info_.add_pre_pipeline_id("lineorder");
	query.pipe_info_.add_pre_pipeline_id("q43_date");
	query.pipe_info_.add_pre_pipeline_id("q43_part");
	query.pipe_info_.add_pre_pipeline_id("q43_supplier");
	query.pipe_info_.add_pre_pipeline_id("q43_customer");


	ObjectInfo input = ShadowVMService::cached_obj[query.pipe_info_.pre_pipeline_id(0)]->info_;
	ObjectInfo result = CreateSchema({
	ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32,
	ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32},
		{ 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,4, 4});
 	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, input);
	query.AddOperator(OperatorInfo::JoinWith, {{"lkey", "$(0.4)"}, {"rkey","$(2.0)"}}, result);
	query.AddOperator(OperatorInfo::JoinWith, {{"lkey", "$(0.5)"}, {"rkey","$(1.0)"}}, result);
	query.AddOperator(OperatorInfo::JoinWith, {{"lkey", "$(0.2)"}, {"rkey","$(3.0)"}}, result);
	query.AddOperator(OperatorInfo::JoinWith, {{"lkey", "$(0.3)"}, {"rkey","$(4.0)"}}, result);
	query.AddOperator(OperatorInfo::ReduceBy, {{"dim","$(1.1), $(4.1))"}, {"metric","($(0.12)-$(0.13))/100"}}, result);
	query.AddOperator(OperatorInfo::Sink, {}, result);
	query.SetMode(true, false, false);
	client.RunPipelineCMD(query.pipe_info_);

	Object * rc = ShadowVMService::cached_obj["q43_rc"];
 	rc->Print();

}
