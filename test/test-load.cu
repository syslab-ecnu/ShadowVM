/*
 * load-case.cpp
 *
 *  Created on: 2019年2月27日
 *      Author: imdb
 */

#include "../com/cpp/com.hpp"
#include "../core/core.hpp"
#include "../core/operator.hpp"
#include "../core/pipeline.hpp"
#include "../codegen/compiler.hpp"
#include "test.hpp"

void test_loader() {

}




void test_load_SSB() {
	PipelineInfo pipeinfo;
	pipeinfo.set_id("firenet_load");
 	ObjectInfo object = GetSSBSchema();

	auto source = pipeinfo.add_workflow();
	source->set_type(OperatorInfo::Source);
	(*source->mutable_args())["type"] = "file";
 	(*source->mutable_args())["path"] = Context::data_path_ + "lineorder/";
	*source->mutable_output() = object;

	auto loader = pipeinfo.add_workflow();
	//loader->set_id("firenet_load");
	loader->set_type(OperatorInfo::Loader);
	(*loader->mutable_args())["col_spilter"]=",";
	(*loader->mutable_args())["row_spilter"]="\n";
	//*loader->mutable_output() = object;

	auto sink = pipeinfo.add_workflow();
	sink->set_type(OperatorInfo::Sink);
 	*sink->mutable_output() = object;

	Pipeline pipeline(pipeinfo);
	CodeContext ctx;
	pipeline.GenCode(ctx);
	string code = ctx.ExportHostCode();
	//cout << code << endl;
	JitCompiler::CompileHostFuc(pipeinfo.id() , code);
	pipeline.Open();
	pipeline.Run(false);


	for (auto i = 0;  i < pipeline.source_->object_->NumberOfBulk(); i++) {
		Bulk * bulk = pipeline.sink_->object_->GetBulk(i);
		cout << "bulk "<<bulk->id_ << " has " << bulk->tuple_count_ << " tuples"<< endl;
 	//	for (auto row = bulk->tuple_count_ -1; row > bulk->tuple_count_ - 5; row--) {
	for (auto row = 0 ; row < 5; row++) {
		for (auto c = 0; c < bulk->schema_.GetColumnCount(); c++) {
			cout << GetStringOfItem(bulk->column_buf_[c]+row*pipeline.sink_->object_->column_size_[c],
					pipeline.sink_->object_->column_type_[c], pipeline.sink_->object_->column_size_[c]) << ",";
		}
		cout << endl;
		}
	}
	pipeline.Close();
  }

pair<ObjectInfo, Object*> load_lineorder() {
	ObjectInfo schema = GetSSBSchema();
	PipelineTest load("load_lineorder");
	load.AddOperator(OperatorInfo::Source,{{"type","file"},{"path",Context::data_path_ +"lineorder/"}},schema);
	load.AddOperator(OperatorInfo::Loader,{{"col_spilter","|"},{"row_spilter","\n"}}, schema);
	load.AddOperator(OperatorInfo::Sink, {},schema);
	return make_pair(schema, load.Execute(nullptr, {}, false, false));
 }


pair<ObjectInfo, Object*> load_date() {
	ObjectInfo schema = CreateSchema(
			{TInt32,TString,TString,TString,TInt32,TInt32,TString,
		TInt32,TInt32,TInt32,TInt32,TInt32,TString,TInt32,TInt32,TInt32,TInt32},
			{4, 18, 9, 9, 4, 4 , 8, 4, 4, 4, 4, 4, 12, 4, 4, 4, 4 });
	PipelineTest load("load_date");
	load.AddOperator(OperatorInfo::Source,{{"type","file"},{"path",Context::data_path_ +"date/"}},schema);
	load.AddOperator(OperatorInfo::Loader,{{"col_spilter","|"},{"row_spilter","\n"}}, schema);
	load.AddOperator(OperatorInfo::Sink, {},schema);
	return make_pair(schema, load.Execute(nullptr, {}, false, false));
}
pair<ObjectInfo, Object*> load_customer() {
	ObjectInfo schema = CreateSchema(
			{TInt32, TString,TString,TString,TString,TString,TString,TString},
			{4, 25, 25, 10,  15, 15, 15, 15 });
 	PipelineTest load("load_customer");
	load.AddOperator(OperatorInfo::Source,{{"type","file"},{"path",Context::data_path_+"customer/"}},schema);
	load.AddOperator(OperatorInfo::Loader,{{"col_spilter","|"},{"row_spilter","\n"}}, schema);
	load.AddOperator(OperatorInfo::Sink, {},schema);
	return make_pair(schema, load.Execute(nullptr, {}, false, false));
}
pair<ObjectInfo, Object*> load_part() {
	ObjectInfo schema = CreateSchema(
			{TInt32, TString,TString,TString,TString,TString,TString,TInt32,TString},
			{4, 22, 8, 8, 10, 11, 25, 4, 10});
 	PipelineTest load("load_part");
	load.AddOperator(OperatorInfo::Source,{{"type","file"},{"path",Context::data_path_+"part/"}},schema);
	load.AddOperator(OperatorInfo::Loader,{{"col_spilter","|"},{"row_spilter","\n"}}, schema);
	load.AddOperator(OperatorInfo::Sink, {},schema);
	return make_pair(schema, load.Execute(nullptr, {}, false, false));
}
pair<ObjectInfo, Object*>  load_supplier(){
	ObjectInfo schema = CreateSchema(
			{TInt32, TString,TString,TString,TString,TString,TString},
			{4, 25, 25, 10, 15, 12, 15});
 	PipelineTest load("load_supplier");
	load.AddOperator(OperatorInfo::Source,{{"type","file"},{"path",Context::data_path_+"supplier/"}},schema);
	load.AddOperator(OperatorInfo::Loader,{{"col_spilter","|"},{"row_spilter","\n"}}, schema);
	load.AddOperator(OperatorInfo::Sink, {},schema);
	return make_pair(schema, load.Execute(nullptr, {}, false, false));
}

pair<ObjectInfo, Object*> load_UDF() {
	ObjectInfo schema = CreateSchema({TString},{16*4+1});
	schema.set_storage(StoType::HostMem);
	RamObject * object = new RamObject(schema);
	for (auto i = 0; i<64;i++) {
		Bulk * bulk = object->NewBulk(i);
		bulk->tuple_count_ = bulk->max_tuple_count_;
	}
	return make_pair(schema, object);
}


