/*
 * operator.hpp
 *
 *  Created on: 2019年2月21日
 *      Author: imdb
 */

#ifndef OPERATOR_HPP_
#define OPERATOR_HPP_

#include "../com/cpp/com.hpp"
#include "../object/object.hpp"
#include "../object/store.hpp"

#include "../codegen/compiler.hpp"

#include "pipeline.hpp"
class Operator {
public:
	size_t NumberOfTask() {return object_->NumberOfBulk();}
	virtual void Open(const Pipeline & pipe){}
 	virtual Bulk* Consume(int task_id) {return object_->GetBulk(task_id);}
	virtual Bulk* Produce(int task_id) {return object_->NewBulk(task_id, CTX.tuple_per_bulk_);}
	virtual int Host(int task_id, Bulk * in, Bulk * out) { return 0;}
	virtual int Next(int task_id, Bulk * bulk_in, BulkVector & in, BulkVector & out, bool isGPU) { return 0;}
	virtual int HostReduce(int task_id, const Pipeline & pipe, Bulk * in) { return 0;}
	virtual int DeviceNext(int task_id, BulkVector & in, BulkVector & out) {}
	virtual void Close() {delete object_;}
 	virtual void GenCode(const Pipeline & pipe, CodeContext & ctx, bool isGPU) {}
 	virtual void GenDeviceCode(const Pipeline & pipe, CodeContext & ctx) {}
  	OperatorInfo info_;
	Object * object_ = nullptr;
	vector<string> select_exp_;
	vector<string> filter_exp_;
	vector<string> metric_exp_;
 	vector<KeyIndex> reduce_key_;
	vector<KeyIndex> join_lkey_;
	vector<KeyIndex> join_rkey_;
	bool is_reduce_ = false;
  };

class Source: public Operator {
public:
};

class FileSource: public Source {
public:
 	FileSource(const OperatorInfo & info) {
 		info_ = info;
 		info_.mutable_output()->set_id(info_.id());
 		info_.mutable_output()->set_storage(Disk);
 		info_.mutable_output()->set_path(info_.args().find("path")->second);
 	}
//	void Open(const Pipeline & pipe) {object_ = new FileObject(info_.args().find("path")->second);}
	void Open(const Pipeline & pipe) {object_ = ObjStore.Create(info_.output());}
 	Bulk* Consume(int task_id) {return object_->GetBulk(task_id);}
  };

class ObjectSource: public Source {
public:
	ObjectSource(const OperatorInfo & info) {info_ = info;}
	ObjectSource(const OperatorInfo & info, Object * obj) {info_ = info; object_ = obj;}
	void Open(){ assert(object_ != nullptr); }
};

class Loader: public Operator {
public:
	Loader(const OperatorInfo & info);
	int Host(int task_id,  Bulk * in, Bulk * out);
	int DeviceNext(int task_id, BulkVector & in, BulkVector & out);
 	void GenCode(const Pipeline & pipe, CodeContext & ctx, bool isGPU);
 	char col_spilter_;
	char row_spilter_;
 };

class Query: public Operator {
public:
	Query () {}
	void Open(const Pipeline & pipe);
	int Next(int task_id, Bulk * bulk_in, BulkVector & in, BulkVector & out, bool isGPU);
	int HostReduce(int task_id, const Pipeline & pipe, Bulk * in);
	int DeviceNext(int task_id, BulkVector & in, BulkVector & out);
	void GenCode(const Pipeline & pipe, CodeContext & ctx, bool isGPU);
	void GenKernelCode(const Pipeline & pipe, CodeContext & ctx);

};

class Sink: public Operator {
public:
	Sink(const OperatorInfo & info) {
		info_ = info;
 		info_.mutable_output()->set_id(info_.id());
 		info_.mutable_output()->set_storage(HostMem);
	}
//	void Open(const Pipeline & pipe) {object_ = new RamObject(info_.output());}
	void Open(const Pipeline & pipe) {object_ = ObjStore.Create(info_.output());}
 	vector<string> sort_by_ ;
	vector<string> reduce_exp_;
	vector<string> reduce_by_;
};

#endif /* OPERATOR_HPP_ */
