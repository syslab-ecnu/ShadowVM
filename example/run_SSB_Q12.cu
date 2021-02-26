/*
 * run_SSB_Q12.cu
 *
 */


#include "example.hpp"

void run_SSB_Q12_date(ShadowVMClient & client) {
	run_load_date(client);
	ObjectInfo schema = CreateSchema({TInt32, TInt32, TString}, {4,  4,  40});
	string prediction = "$(0.5)==199401";
	PipelineTest query("q12_date");
	Object* input_obj = ShadowVMService::cached_obj["date"];
	query.pipe_info_.add_pre_pipeline_id("date");
	assert(input_obj != nullptr);
//	cout << input_obj->info_.DebugString() << endl;
//	return ;
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, input_obj->info_);
	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction}}, schema);
	query.AddOperator(OperatorInfo::Map, {{"select", "$(0.0), $(0.4)"}}, schema);
	query.AddOperator(OperatorInfo::Sink, {}, schema);
	query.SetMode(true, false, false);
	client.RunPipelineCMD(query.pipe_info_);
}

void run_SSB_Q12(ShadowVMClient & client) {
	run_SSB_Q12_date(client);
	run_load_lineorder(client);
	ObjectInfo result = CreateSchema({
	ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32,
	ValType::Int32, ValType::Int32, ValType::Int32},
		{ 4, 4, 4, 4, 4, 4, 4, 4, 4});
 	string prediction_Q1 = " $(0.8)>26 && (0.8)<35 && $(0.11)>=4 && $(0.11)<=6 ";
	PipelineTest query("q12_rc");
	query.pipe_info_.add_pre_pipeline_id("lineorder");
	query.pipe_info_.add_pre_pipeline_id("q12_date");

	ObjectInfo input = ShadowVMService::cached_obj[query.pipe_info_.pre_pipeline_id(0)]->info_;

	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, input);
	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction_Q1}}, result);
	query.AddOperator(OperatorInfo::JoinWith, {{"lkey", "$(0.5)"}, {"rkey","$(1.0)"}}, result);
	query.AddOperator(OperatorInfo::ReduceBy, {{"dim",""}, {"metric","$(0.9)*$(0.11)/1000"}}, result);
	query.AddOperator(OperatorInfo::Sink, {}, result);
	query.SetMode(true, false, false);
	client.RunPipelineCMD(query.pipe_info_);

	Object * rc = ShadowVMService::cached_obj["q12_rc"];
	rc->Print();

}
