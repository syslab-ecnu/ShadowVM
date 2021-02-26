/*
 * test-load.hpp
 *
 *  Created on: 2019年3月11日
 *      Author: imdb
 */

#ifndef TEST_LOAD_HPP_
#define TEST_LOAD_HPP_
#include "../com/cpp/com.hpp"
#include "../core/core.hpp"
#include "../core/operator.hpp"
#include "../core/pipeline.hpp"

void test_loader();

inline ObjectInfo CreateSchema(const vector<ValType> & type, const vector<size_t> & size) {
	ObjectInfo table;
	table.set_storage(HostMem);
	assert(type.size() == size.size());
	for (auto i = 0; i < type.size(); i++) {
		auto column = table.add_column_list();
		column->set_type(type[i]);
		column->set_val_size(size[i]);
	}
	return table;
}

inline ObjectInfo GetSSBSchema(){
	ObjectInfo ssb;
	for (auto i = 1; i <= 17; i++) {
		auto column = ssb.add_column_list();
		if(i == 7) {
			column->set_type(ValType::String);
			column->set_val_size(15);
		} else if(i == 17) {
			column->set_type(ValType::String);
			column->set_val_size(10);
		} else {
			column->set_type(ValType::Int32);
			column->set_val_size(4);
		}
	}
	ssb.set_storage(HostMem);
	return ssb;
}


void test_load_SSB() ;

//ObjectInfo CreateSchema(const vector<ValType> & type, const vector<size_t> & size);

void test_query_SSB_Q2_1();



void test_SSB_Q11(bool isGPU);
void test_SSB_Q12(bool isGPU);
void test_SSB_Q13(bool isGPU);

void test_SSB_Q21(bool isGPU);
void test_SSB_Q22(bool isGPU);
void test_SSB_Q23(bool isGPU);

void test_SSB_Q31(bool isGPU);
void test_SSB_Q32(bool isGPU);
void test_SSB_Q33(bool isGPU);
void test_SSB_Q34(bool isGPU);

void test_SSB_Q41(bool isGPU);
void test_SSB_Q42(bool isGPU);
void test_SSB_Q43(bool isGPU);

void test_UDF();

void test_yahoo(bool isGPU);

class PipelineTest {
public:
	PipelineTest (string id) { pipe_info_.set_id(id);}
	PipelineTest (const PipelineInfo & pipe_info) { pipe_info_ = pipe_info;}


	void AddOperator(OperatorInfo::Type type, map<string, string> args, ObjectInfo & output) {
		auto op = pipe_info_.add_workflow();
		op->set_type(type);
		op->set_id(pipe_info_.id());
		*op->mutable_output() = output;
		for (auto & arg : args)	(*op->mutable_args())[arg.first] = arg.second;
	}
	void SetMode(bool codegen, bool debug, bool isGPU = false) {
		pipe_info_.set_codegen(codegen);
		pipe_info_.set_debug(debug);
		pipe_info_.set_isgpu(isGPU);
	}
	Object * Execute(Object * input_obj, const vector<pair<ObjectInfo, Object *>> & ref_obj,
			bool codegen, bool print, bool isGPU = false) {
 		Pipeline pipe(pipe_info_);
		CodeContext ctx_;
		if (input_obj != nullptr)
			pipe.source_->object_ = input_obj;
		for (auto & obj:ref_obj) {
			pipe.input_info_.push_back(obj.first);
			pipe.ref_object_.push_back(obj.second);
		}
		if (codegen) {
			pipe.GenCode(ctx_);
 				string code = ctx_.ExportHostCode();
				if (print && (!isGPU))	cout << code << endl;
				JitCompiler::CompileHostFuc(pipe.info_.id(), code);
				code =ctx_.ExportDeviceCode();
				if (print && isGPU)	cout << code << endl;
				JitCompiler::CompileDeviceFuc(pipe.info_.id(), code);
		}
	//	cout << "#######################" << endl;
	//	cout << pipe.info_.DebugString() << endl;
		pipe.Open();
		pipe.Run(isGPU);
		return pipe.sink_->object_;
	}

	Object * Run(Object * input_obj, const vector<Object *> & ref_obj,
			bool codegen, bool print, bool isGPU = false) {
 		Pipeline pipe(pipe_info_);
		CodeContext ctx_;
		if (input_obj != nullptr)
			pipe.source_->object_ = input_obj;
		for (auto & obj:ref_obj) {
			pipe.input_info_.push_back(obj->info_);
			pipe.ref_object_.push_back(obj);
		}
		if (codegen) {
			pipe.GenCode(ctx_);
 				string code = ctx_.ExportHostCode();
				if (print && (!isGPU))	cout << code << endl;
				JitCompiler::CompileHostFuc(pipe.info_.id(), code);
				code =ctx_.ExportDeviceCode();
				if (print && isGPU)	cout << code << endl;
				JitCompiler::CompileDeviceFuc(pipe.info_.id(), code);
		}
	//	cout << "#######################" << endl;
	//	cout << pipe.info_.DebugString() << endl;
		pipe.Open();
		pipe.Run(isGPU);
		return pipe.sink_->object_;
	}

 	PipelineInfo pipe_info_;
 };

pair<ObjectInfo, Object*>  load_lineorder();
pair<ObjectInfo, Object*>  load_date();
pair<ObjectInfo, Object*>  load_customer();
pair<ObjectInfo, Object*>  load_part();
pair<ObjectInfo, Object*>  load_supplier();
pair<ObjectInfo, Object*>  load_UDF();

const auto TInt32 =  ValType::Int32 ;
const auto TString = ValType::String;

#endif /* TEST_LOAD_HPP_ */
