/*
 * svm_load_SSB.cu
 */


#include "example.hpp"
#include "../test/test.hpp"


void run_load_SSB(ShadowVMClient & client) {
	run_load_lineorder(client);
	run_load_date(client);
	run_load_customer(client);
	run_load_part(client);
}

void run_load_lineorder(ShadowVMClient & client) {
	PipelineTest load("lineorder");
	ObjectInfo schema = GetSSBSchema();
	load.AddOperator(OperatorInfo::Source,{{"type","file"},{"path",Context::data_path_ +"lineorder/"}},schema);
	load.AddOperator(OperatorInfo::Loader,{{"col_spilter","|"},{"row_spilter","\n"}}, schema);
	load.AddOperator(OperatorInfo::Sink, {},schema);
	load.SetMode(false, false, false);
	//cout << load.pipe_info_.DebugString() << endl;
	client.RunPipelineCMD(load.pipe_info_);
}

void run_load_date(ShadowVMClient & client) {
	ObjectInfo schema = CreateSchema(
			{TInt32,TString,TString,TString,TInt32,TInt32,TString,
		TInt32,TInt32,TInt32,TInt32,TInt32,TString,TInt32,TInt32,TInt32,TInt32},
			{4, 18, 9, 9, 4, 4 , 8, 4, 4, 4, 4, 4, 12, 4, 4, 4, 4 });
	PipelineTest load("date");
	load.AddOperator(OperatorInfo::Source,{{"type","file"},{"path",Context::data_path_ +"date/"}},schema);
	load.AddOperator(OperatorInfo::Loader,{{"col_spilter","|"},{"row_spilter","\n"}}, schema);
	load.AddOperator(OperatorInfo::Sink, {},schema);
	load.SetMode(false, false, false);
	client.RunPipelineCMD(load.pipe_info_);
}

void run_load_customer(ShadowVMClient & client){
	ObjectInfo schema = CreateSchema(
			{TInt32, TString,TString,TString,TString,TString,TString,TString},
			{4, 25, 25, 10,  15, 15, 15, 15 });
 	PipelineTest load("customer");
	load.AddOperator(OperatorInfo::Source,{{"type","file"},{"path",Context::data_path_+"customer/"}},schema);
	load.AddOperator(OperatorInfo::Loader,{{"col_spilter","|"},{"row_spilter","\n"}}, schema);
	load.AddOperator(OperatorInfo::Sink, {},schema);
	load.SetMode(false, false, false);
	client.RunPipelineCMD(load.pipe_info_);
}

void run_load_part(ShadowVMClient & client){
	ObjectInfo schema = CreateSchema(
			{TInt32, TString,TString,TString,TString,TString,TString,TInt32,TString},
			{4, 22, 8, 8, 10, 11, 25, 4, 10});
 	PipelineTest load("part");
	load.AddOperator(OperatorInfo::Source,{{"type","file"},{"path",Context::data_path_+"part/"}},schema);
	load.AddOperator(OperatorInfo::Loader,{{"col_spilter","|"},{"row_spilter","\n"}}, schema);
	load.AddOperator(OperatorInfo::Sink, {},schema);
	load.SetMode(false, false, false);
	client.RunPipelineCMD(load.pipe_info_);
}

void run_load_supplier(ShadowVMClient & client){
	ObjectInfo schema = CreateSchema(
			{TInt32, TString,TString,TString,TString,TString,TString},
			{4, 25, 25, 10, 15, 12, 15});
 	PipelineTest load("supplier");
	load.AddOperator(OperatorInfo::Source,{{"type","file"},{"path",Context::data_path_+"supplier/"}},schema);
	load.AddOperator(OperatorInfo::Loader,{{"col_spilter","|"},{"row_spilter","\n"}}, schema);
	load.AddOperator(OperatorInfo::Sink, {},schema);
	load.SetMode(false, false, false);
	client.RunPipelineCMD(load.pipe_info_);
}
