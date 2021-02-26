/*
 * pipeline.cpp
 *
 *  Created on: 2019年2月25日
 *      Author: imdb
 */

#include "pipeline.hpp"
#include "../codegen/codegen.hpp"
#include "resource.hpp"
Pipeline::Pipeline(const PipelineInfo & info) {
	info_ = info;
	for(int i = 0; i < info_.workflow_size(); i++) {
  		auto op_info = info_.workflow(i);
 		switch (op_info.type()) {
 	 	case OperatorInfo::Source:
 	 		if ((*op_info.mutable_args())["type"][0] == 'f')
 	 			source_ = new FileSource(op_info);
 	 		else
 	 			source_ = new ObjectSource(op_info);
 	 		input_info_.push_back(op_info.output());
  	 //		cout << "this is source" << endl;
 	 		break;
 	 	case OperatorInfo::Loader:
 	 		transform_ = new Loader(op_info);
 	 		is_loader_ = true;
  	 	//	cout << "this is loader" << endl;
 	 		break;
 	 	case OperatorInfo::Sink:
 	 		sink_ = new Sink(op_info);
 	 		output_info_ = op_info.output();
  	// 		cout << "this is sink" << endl;
 	 		break;
		}
	}
	if (is_loader_ == false && transform_ == nullptr) {
		transform_ = new Query();
		transform_->info_.set_id(info_.id());
	}
	vector<string> exp_list;
 	for(int i = 0; i < info_.workflow_size();i++) {
  		auto op_info = info_.workflow(i);
 		switch(op_info.type()) {
 		case OperatorInfo::Map:
			//cout << "[" << op_info.args().find("select")->second << "]" << endl;
			//exp_list = StrSplit(op_info.args().find("select")->second,",") ;
			//transform_->select_exp_ = exp_list;
			//assert(transform_ != nullptr );
 			transform_->select_exp_ = StrSplit(op_info.args().find("select")->second,",") ;
 			break;
		case OperatorInfo::Filter:
			transform_->filter_exp_.push_back(op_info.args().find("predication")->second);
 			break;
		case OperatorInfo::ReduceBy:
			transform_->metric_exp_= StrSplit(op_info.args().find("metric")->second ,",");
			//cout << "metric num:"  << transform_->metric_exp_.size() << endl;
			if (op_info.args().find("dim") != op_info.args().end())
				exp_list = StrSplit(op_info.args().find("dim")->second ,",");
			//cout << "dim num:"  << exp_list.size() << endl;
			for (auto & exp: exp_list)
				transform_->reduce_key_.push_back(CodeGen::GenKeyIndex(exp));
 			transform_->is_reduce_ = true;
  	 		break;
 	 	case OperatorInfo::JoinWith:
 			transform_->join_lkey_.push_back(CodeGen::GenKeyIndex(op_info.args().find("lkey")->second));
 			transform_->join_rkey_.push_back(CodeGen::GenKeyIndex(op_info.args().find("rkey")->second));
 			//input_info_.push_back(op_info.input()[1].pre_object());
 	 		break;
 	 	case OperatorInfo::SortBy:
 	 		//sink_->sort_by_ = Utility::Split(op_info.args().find("key")->second, ",");
 	 		break;
		}
	}
/* 	if (!transform_->is_agg_) {
 		auto mask = sink_->info_.mutable_output()->add_column_list();
 		mask->set_type(ValType::Int32);
 		mask->set_val_size(4);
 		cout << sink_->info_.output().column_list_size()  << endl;
 	}*/
  }

void Pipeline::Open() {
	struct  timeval begin, end;
   gettimeofday(&begin, nullptr);
	source_->Open(*this);
	sink_->Open(*this);
	transform_->Open(*this);
 	for (auto i = 0; i<transform_->join_rkey_.size(); i++) {
		auto obj_off = transform_->join_rkey_[i].obj_off_;
		auto col_off = transform_->join_rkey_[i].col_off_;
		Bulk * bulk = ref_object_[obj_off-1]->GetBulk(0);
		ValType type = input_info_[obj_off].column_list(col_off).type();
		size_t size = input_info_[obj_off].column_list(col_off).val_size();
		auto hash_col_off  = input_info_[obj_off].column_list_size()-1;
		cout << "obj_off:" << obj_off << ", col_off"<< col_off << ", hash_col_off:" <<hash_col_off <<endl;
		cout << "capacity:" << bulk->capacity_ / (bulk->tuple_count_) << "MB" << endl;
 		switch(type) {
		case ValType::Int16 :
			BuildHashTable<short>((short*)bulk->column_buf_[col_off], bulk->tuple_count_, size,
					(int*)bulk->column_buf_[hash_col_off], HashRate*bulk->max_tuple_count_);
	//		BuildHashTableForCPU<short>((short*)bulk->column_buf_[col_off], bulk->tuple_count_, size,
	//						(int*)bulk->column_buf_[hash_col_off], HashRate*bulk->max_tuple_count_);
			break;
		case ValType::Int32 :
			BuildHashTable<int>((int*)bulk->column_buf_[col_off], bulk->tuple_count_, size,
					(int*)bulk->column_buf_[hash_col_off], HashRate*bulk->max_tuple_count_);
	//		BuildHashTableForCPU<int>((int*)bulk->column_buf_[col_off], bulk->tuple_count_, size,
	//			(int*)bulk->column_buf_[hash_col_off], HashRate*bulk->max_tuple_count_);
			break;
		case ValType::Int64 :
			BuildHashTable<long long>((long long*)bulk->column_buf_[col_off], bulk->tuple_count_, size,
					(int*)bulk->column_buf_[hash_col_off], HashRate*bulk->max_tuple_count_);
	//		BuildHashTableForCPU<long long>((long long*)bulk->column_buf_[col_off], bulk->tuple_count_, size,
	//				(int*)bulk->column_buf_[hash_col_off], HashRate*bulk->max_tuple_count_);
			break;
		case ValType::Float32 :
			BuildHashTable<float>((float*)bulk->column_buf_[col_off], bulk->tuple_count_, size,
					(int*)bulk->column_buf_[hash_col_off], HashRate*bulk->max_tuple_count_);
	//		BuildHashTableForCPU<float>((float*)bulk->column_buf_[col_off], bulk->tuple_count_, size,
	//				(int*)bulk->column_buf_[hash_col_off], HashRate*bulk->max_tuple_count_); break;
		case ValType::Float64 :
			BuildHashTable<double>((double*)bulk->column_buf_[col_off], bulk->tuple_count_, size,
					(int*)bulk->column_buf_[hash_col_off], HashRate*bulk->max_tuple_count_);
//			BuildHashTableForCPU<double>((double*)bulk->column_buf_[col_off], bulk->tuple_count_, size,
//					(int*)bulk->column_buf_[hash_col_off], HashRate*bulk->max_tuple_count_); break;
		case ValType::String :
			BuildHashTable<char>((char*)bulk->column_buf_[col_off], bulk->tuple_count_, size,
					(int*)bulk->column_buf_[hash_col_off], HashRate*bulk->max_tuple_count_);
//			BuildHashTableForCPU<char>((char*)bulk->column_buf_[col_off], bulk->tuple_count_, size,
//					(int*)bulk->column_buf_[hash_col_off], HashRate*bulk->max_tuple_count_); break;
		}
	//	cout << "build hash table of " << obj_off<< ":" << col_off << ":"<<hash_col_off << endl;
	}
 	gettimeofday(&end, nullptr);
	cout << "hash table build time:" << (end.tv_sec - begin.tv_sec)*1000  +
			(end.tv_usec - begin.tv_usec) / 1000 << " ms" << endl;
}

void Pipeline::Run(bool isGPU) {
	auto task_num = source_->NumberOfTask();
  	cout << "number of task source:" << source_->NumberOfTask() <<  endl;
 	vector<thread> task_list;
   struct  timeval begin, parepare, end;
   gettimeofday(&begin, nullptr);
   if (isGPU)	{
//	   cudaSetDevice(1);
	   source_->object_->To(SharedMem);
	   sink_->object_->info_.set_storage(SharedMem);

	   if (ResMgr::enable_hybrid_exec_)
	   for (auto i = 0; i<ref_object_.size();i++) {
		   string name = ref_object_[i]->info_.id()+"_host";
		   Object* obj =  ref_object_[i]->Copy(name, SharedMem);
		   host_ref_object_.push_back(obj);
		}

	   for (auto i = 0; i < ref_object_.size(); i++)
		   ref_object_[i]->To(DeviceMem);
	   cout << "ref object num:" << ref_object_.size() << endl;
    }
   for (auto i = 0; i < task_num; i++)
  		   sink_->object_->NewBulk(i, CTX.tuple_per_bulk_);
   gettimeofday(&parepare, nullptr);
   cout << "pipeline prepare data time:" << (parepare.tv_sec - begin.tv_sec)*1000  +
			(parepare.tv_usec - begin.tv_usec) / 1000 << " ms" << endl;
   auto device_task_num = 0;
   int CPU_ct = task_num * ResMgr::CPU_ratio_/10;
   int CUDA_ct = task_num - CPU_ct;

   //cout << "CPU_ct:" << CPU_ct << " ," << " CUDA_ct:" << CUDA_ct << endl;
	for (auto i = 0; i < task_num; i++) {
		if (ResMgr::enable_hybrid_exec_ && isGPU) {
			int ct = 0;

			if (i  % 10 < ResMgr::CPU_ratio_) {
				task_list.push_back(thread(TransformTask, this, i, false));
			} else {
				task_list.push_back(thread(TransformTask, this, i, true));
				device_task_num ++;
			}

		//	bool select_CPU = rand()%10 < ResMgr::CPU_ratio ;
/*			while (true) {
				if ( i % 10 < ResMgr::CPU_ratio || ResMgr::FreeCPUNum() > 0) {
			 //		ResMgr::GetCPU();
					task_list.push_back(thread(TransformTask, this, i, false));
		//			cout << "CPU task " << i << " for : " << ct << endl;
					CPU_ct --;
					break;
				} else if ( ResMgr::FreeStreamNum() > 0) {
					task_list.push_back(thread(TransformTask, this, i, true));
					device_task_num ++;
			//		cout << "CUDA task " << i << " for : " << ct << endl;
					CUDA_ct --;
					break;
				}
				ct ++ ;
				usleep(10);
			}*/
//			cout << "wait " << i << " for : " << ct << endl;
//			if (i % 10 < ResMgr::CPU_ratio || ResMgr::free_CPU_ > 0) {
//				task_list.push_back(thread(TransformTask, this, i, false));
//			} else {
//				task_list.push_back(thread(TransformTask, this, i, true));
//				device_task_num ++;
//			}
		} else  {
	// 	if(!isGPU)
	//	 		ResMgr::GetCPU();
			task_list.push_back(thread(TransformTask, this, i, isGPU));
		}
	}
/*	if (ResMgr::enable_spec_) {
	//	sleep(1);
		for (auto i = 0; i < task_num; i++) {
		//	auto id = task_list[i].get_id();
			task_list[i].detach();
		//	pthread_cancel(task_list[i].native_handle());
		   task_list[i] = thread(TransformTask, this, i, true);

		}
	}*/
	if (ResMgr::enable_hybrid_exec_ && ResMgr::enable_spec_ && isGPU) {
		for (auto i = 0; i < task_num; i++)
			if (task_isGPU_[i] == false) {
				task_list[i].detach();
				task_list[i] = thread(TransformTask, this, i, true);
			}
		//sleep(3);
	}
	for (auto i = 0; i < task_num; i++)
		task_list[i].join();
	cout << "# of CPU tasks:" << task_num - device_task_num   << ", # of GPU tasks:" << device_task_num <<  endl;

//	cuCtxSynchronize();
   gettimeofday(&end, nullptr);
   cout << "pipeline compute time:" << (end.tv_sec - parepare.tv_sec)*1000  +
			(end.tv_usec - parepare.tv_usec) / 1000 << " ms" << endl;
//   cout << "pipeline whole time:" << (end.tv_sec - begin.tv_sec)*1000  +
//			(end.tv_usec - begin.tv_usec) / 1000 << " ms" << endl;

   if (!is_loader_ && task_num > 1) {

	   string fname = ResMgr::trace_path_ + info_.id();
	   if (isGPU)
		   fname += ".CUDA_trace";
	   else
		   fname += ".CPU_trace";
	    fstream fs(fname,ios::in|ios::out|ios::trunc);
		 for (auto i = 0; i < task_num; i++)
			   fs << ResMgr::exec_time[i] << endl;
	    fs.close();
   }
}

void Pipeline::Close() {
	source_->Close();
	transform_->Close();
	sink_->Close();
}

void Pipeline::TransformTask(Pipeline * pipe, size_t id, bool isGPU) {
	pipe->task_isGPU_[id] = isGPU;
//	if (pipe->is_loader_)
//		ResMgr::GetCPU(id);

//	if (!pipe->is_loader_ && id > 5 & isGPU)
//		return;

	struct  timeval begin, parepare, end;
	gettimeofday(&begin, nullptr);


  	if(!isGPU)
  		ResMgr::GetCPU();
  	gettimeofday(&parepare, nullptr);


	double ms = (parepare.tv_sec - begin.tv_sec) * 1000 + (parepare.tv_usec - begin.tv_usec)/1000.0;
  	if(!isGPU)
	ResMgr::wait_time[id] = ms;
	//	if (!isGPU)
//		cout << "CPU task :" << id <<" waittime: " << ms << " ms"<< endl;
//	else
//		cout << "CPU task :" << id <<" runtime: " << ms << " ms"<< endl;


	if (!pipe->is_loader_) {
 		auto bulk_in = pipe->source_->Consume(id);
		auto bulk_out = pipe->sink_->Consume(id);
		BulkVector in(bulk_in);
	//	cout << "ref obj numb:" << pipe->ref_object_.size() << endl;
		if(!ResMgr::enable_hybrid_exec_) {
			for(auto & obj: pipe->ref_object_)
				in.AddBulk(obj->GetBulk(0));
		} else {
			if(isGPU)
			for(auto & obj: pipe->ref_object_)
				in.AddBulk(obj->GetBulk(0));
			if(!isGPU)
			for(auto & obj: pipe->host_ref_object_) {
				in.AddBulk(obj->GetBulk(0));
			//	cout << "obj:path:" << obj->GetBulk(0)->buffer_ << endl;
			}
		}

		BulkVector out(bulk_out);
  		if (!isGPU)
  			bulk_out->tuple_count_ = pipe->transform_->Next(id, bulk_in, in, out, isGPU);
		else
			bulk_out->tuple_count_ = pipe->transform_->Next(id, bulk_in, in, out, isGPU);

  		if (isGPU && pipe->transform_->is_reduce_) {
  			int in_num  = pipe->transform_->reduce_key_.size() + pipe->transform_->metric_exp_.size();
  			vector<int*> column;
  		   for (auto i = 0; i< in_num; i++)
  			   column.push_back((int*)bulk_out->column_buf_[i]);
  		   bulk_out->tuple_count_ = ShrinkTask(bulk_out->tuple_count_, column);
  		  // cout << "task " << id << " reduce " <<  bulk_out->tuple_count_ << " tuples" << endl;
  		}

  		if (pipe->transform_->is_reduce_) {
  		//	if (!isGPU)
  		//		bulk_out->tuple_count_ = ReduceTask(pipe, id, false);
  		//	else
  				//bulk_out->tuple_count_ = GPUReduceTask(pipe, id, isGPU);
  			bulk_out->tuple_count_ = ReduceTask(pipe, id, isGPU);
  		}
	} else {
		pipe->transform_->Host(id,pipe->source_->Consume(id), pipe->sink_->Produce(id) );
	}
 	pipe->sink_->object_->CommitBulk(id);
 	if (!isGPU)
 		ResMgr::ReturnCPU();

 	gettimeofday(&end, nullptr);
	double ms2 = (end.tv_sec - parepare.tv_sec) * 1000 + (end.tv_usec - parepare.tv_usec)/1000.0;
	if (!isGPU) {
		cout << "CPU task :" << id <<" runtime: " << ms2 << " ms"<< endl;
		ResMgr::exec_time[id] = ms2;
	}
	pipe->finished_ct_++;
	pipe->task_stat_[id] = true;
}

int Pipeline::ShrinkTask(int tuple_ct, const vector<int*> & column) {
	int cursor = 0;
	int filter = column.size() - 1;
	for (auto i = 0; i < tuple_ct; i++)
	if (column[filter][i] != 0) {
		for (auto c = 0; c < column.size(); c++)
			column[c][cursor] = column[c][i];
		cursor++;
	}
//	cout << "filtered count " <<cursor << endl;
	return cursor;
}


int Pipeline::ReduceTask(Pipeline * pipe, size_t id, bool isGPU) {
	//cuCtxSynchronize();
	Bulk * in = pipe->sink_->object_->GetBulk(id);
	int tuple_ct = in->tuple_count_;
	auto policy = thrust::seq;
  	int key_num = pipe->transform_->reduce_key_.size();
	int metric_num = pipe->transform_->metric_exp_.size();
	int in_num = key_num + metric_num;
	vector<int*> data;
   for (auto i = 0; i<in->schema_.GetColumnCount();i++)
			data.push_back((int*)in->column_buf_[i]);

//	cout << "key_num:" << key_num << " metric_num:" << metric_num << endl;
	if (key_num == 0 ) {
		if (isGPU)
		for (auto i = 0; i < metric_num; i++) {
			int * metric = (int*) in->column_buf_[i];
			*metric = thrust::reduce(policy, metric, metric+tuple_ct);
		}
		return 1;
	} else if (key_num == 1) {
		auto key = MakeIterator_1(in->column_buf_[0]);
		auto key_out = MakeIterator_1(in->column_buf_[in_num]);
		auto k = thrust::make_zip_iterator(thrust::make_tuple(data[0]));
		auto k_out = thrust::make_zip_iterator(thrust::make_tuple(data[in_num]));
		if (metric_num == 1) {
			auto metric = MakeIterator_1(in->column_buf_[1]);
			auto metric_out = MakeIterator_1(in->column_buf_[1+in_num]);
			thrust::sort_by_key(policy, key, key+tuple_ct, metric);
//			auto new_end = thrust::reduce_by_key(thrust::seq, (int*)in->column_buf_[0],
//					(int*)in->column_buf_[0]+tuple_count, (int*)in->column_buf_[1],
//					(int*)in->column_buf_[2], (int*)in->column_buf_[3]);
			auto new_end = thrust::reduce_by_key(policy, key, key+tuple_ct,
					metric, key_out, metric_out,TupleEqual_1(), TupleSum_1());
			return new_end.first -key_out;
 		} else if (metric_num == 2) {
 			auto metric = MakeIterator_2(in->column_buf_[1], in->column_buf_[2]);
			auto metric_out = MakeIterator_2(in->column_buf_[1+in_num], in->column_buf_[2+in_num]);
			thrust::sort_by_key(policy, key, key+tuple_ct, metric);
			auto new_end = thrust::reduce_by_key(policy, key, key+tuple_ct,
					metric, key_out, metric_out,TupleEqual_1(), TupleSum_2());
			return new_end.first - key_out;
// 			auto km = thrust::make_zip_iterator(thrust::make_tuple(data[0],data[1],data[2]));
// 			auto km_out = thrust::make_zip_iterator(thrust::make_tuple(data[3],data[4],data[5]));
//   		auto m = thrust::make_zip_iterator(thrust::make_tuple(data[1], data[2]));
//	 		auto m_out = thrust::make_zip_iterator(thrust::make_tuple(data[1+in_num], data[2+in_num]));
// 			auto end1 = thrust::copy_if(policy, km, km+tuple_ct, km_out, IsValid_3());
// 			tuple_ct = end1-km_out;
// 			return tuple_ct;
//  			thrust::sort_by_key(policy,k_out, k_out+tuple_ct, m_out);
// 			auto end2 = thrust::reduce_by_key(policy,k_out, k_out+tuple_ct, m_out, k, m, TupleEqual_1(), TupleSum_2());
//  			return  end2.first - k;
 		} else if (metric_num == 3) {
			auto metric = MakeIterator_3(in->column_buf_[1], in->column_buf_[2],  in->column_buf_[3]);
			auto metric_out = MakeIterator_3(in->column_buf_[1+in_num], in->column_buf_[2+in_num],in->column_buf_[3+in_num]);
			thrust::sort_by_key(policy, key, key+tuple_ct, metric);
			auto new_end = thrust::reduce_by_key(policy, key, key+tuple_ct,
					metric, key_out, metric_out,TupleEqual_1(), TupleSum_3());
			return new_end.first - key_out;
		} else assert(false);
	} else if(key_num  == 2) {
		auto key = MakeIterator_2(in->column_buf_[0], in->column_buf_[1]);
		auto key_out = MakeIterator_2(in->column_buf_[0+in_num], in->column_buf_[1+in_num]);
		if (metric_num == 1) {
			auto metric =  MakeIterator_1(in->column_buf_[2]);
 			auto metric_out =MakeIterator_1(in->column_buf_[2+in_num]);
			thrust::sort_by_key(policy, key, key+tuple_ct, metric);
 			auto new_end = thrust::reduce_by_key(policy, key, key+tuple_ct,
					metric, key_out, metric_out,TupleEqual_2(), TupleSum_1());
			return new_end.first -key_out;
		} else if (metric_num == 2) {
			auto metric = MakeIterator_2(in->column_buf_[2], in->column_buf_[3]);
			auto metric_out = MakeIterator_2(in->column_buf_[2+in_num], in->column_buf_[3+in_num]);
			thrust::sort_by_key(policy, key, key+tuple_ct, metric);
			auto new_end = thrust::reduce_by_key(policy, key, key+tuple_ct,
					metric, key_out, metric_out,TupleEqual_2(), TupleSum_2());
			return new_end.first - key_out;
 		} else if (metric_num == 3) {
			auto metric = MakeIterator_3(in->column_buf_[2], in->column_buf_[3], in->column_buf_[4]);
			auto metric_out = MakeIterator_3(in->column_buf_[2+in_num], in->column_buf_[3+in_num],in->column_buf_[4+in_num]);
			thrust::sort_by_key(policy, key, key+tuple_ct, metric);
			auto new_end = thrust::reduce_by_key(policy, key, key+tuple_ct,
					metric, key_out, metric_out,TupleEqual_2(), TupleSum_3());
			return new_end.first - key_out;
		} else assert(false);
	} else if(key_num == 3) {
		auto key = MakeIterator_3(in->column_buf_[0], in->column_buf_[1], in->column_buf_[2]);
		auto key_out = MakeIterator_3(in->column_buf_[0+in_num], in->column_buf_[1+in_num],in->column_buf_[2+in_num]);
		if (metric_num == 1) {
			//cout << key_num << "," << metric_num << endl;
			auto metric =  MakeIterator_1(in->column_buf_[3]);
 			auto metric_out =MakeIterator_1(in->column_buf_[3+in_num]);
			thrust::sort_by_key(policy, key, key+tuple_ct, metric);
 			auto new_end = thrust::reduce_by_key(policy, key, key+tuple_ct,
					metric, key_out, metric_out,TupleEqual_3(), TupleSum_1());
			return new_end.first -key_out;
		} else if (metric_num == 2) {
			auto metric = MakeIterator_2(in->column_buf_[3], in->column_buf_[4]);
			auto metric_out = MakeIterator_2(in->column_buf_[3+in_num], in->column_buf_[4+in_num]);
			thrust::sort_by_key(policy, key, key+tuple_ct, metric);
			auto new_end = thrust::reduce_by_key(policy, key, key+tuple_ct,
					metric, key_out, metric_out,TupleEqual_3(), TupleSum_2());
			return new_end.first - key_out;
 		} else if (metric_num == 3) {
			auto metric = MakeIterator_3(in->column_buf_[3], in->column_buf_[4],  in->column_buf_[5]);
			auto metric_out = MakeIterator_3(in->column_buf_[3+in_num], in->column_buf_[4+in_num],in->column_buf_[5+in_num]);
			thrust::sort_by_key(policy, key, key+tuple_ct, metric);
			auto new_end = thrust::reduce_by_key(policy, key, key+tuple_ct,
					metric, key_out, metric_out,TupleEqual_3(), TupleSum_3());
			return new_end.first - key_out;
		} else assert(false);
	}
}

int Pipeline::GPUReduceTask(Pipeline * pipe, size_t id, bool isGPU) {
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	Bulk * in = pipe->sink_->object_->GetBulk(id);
	vector<thrust::device_ptr<int>> data;
   for (auto i = 0; i<in->schema_.GetColumnCount();i++)
	 		data.push_back(thrust::device_ptr<int>((int*)in->column_buf_[i]));
   int tuple_ct = in->tuple_count_;
  	int key_num = pipe->transform_->reduce_key_.size();
	int metric_num = pipe->transform_->metric_exp_.size();
	int in_num = key_num + metric_num;
 	if (key_num == 1) {
		auto k = thrust::make_zip_iterator(thrust::make_tuple(data[0]));
		auto k_out = thrust::make_zip_iterator(thrust::make_tuple(data[in_num]));
		if (metric_num == 1) {
 			auto m = thrust::make_zip_iterator(thrust::make_tuple(data[1]));
 			auto m_out = thrust::make_zip_iterator(thrust::make_tuple(data[1+in_num]));
 			thrust::sort_by_key(k, k+tuple_ct, m);
 			auto end = thrust::reduce_by_key(k, k+tuple_ct, m, k_out, m_out, TupleEqual_1(), TupleSum_1());
			return end.first - k_out;
 		} else if (metric_num == 2) {
 			auto km = thrust::make_zip_iterator(thrust::make_tuple(data[0],data[1],data[2]));
 			auto km_out = thrust::make_zip_iterator(thrust::make_tuple(data[3],data[4],data[5]));
   		auto m = thrust::make_zip_iterator(thrust::make_tuple(data[1], data[2]));
	 		auto m_out = thrust::make_zip_iterator(thrust::make_tuple(data[1+in_num], data[2+in_num]));
 			auto end1 = thrust::partition(thrust::cuda::par.on(stream), km, km+tuple_ct, IsValid_3());
 			tuple_ct = end1 - km;
 			thrust::sort_by_key(thrust::cuda::par.on(stream),k, k+tuple_ct, m);
 			auto end2 = thrust::reduce_by_key(thrust::cuda::par.on(stream),k, k+tuple_ct, m, k_out, m_out, TupleEqual_1(), TupleSum_2());
         return  end2.first - k_out;

// 			auto end1 = thrust::copy_if(thrust::cuda::par.on(stream), km, km+tuple_ct, km_out, IsValid_3());
// 			tuple_ct = end1-km_out;
//   		   auto m = thrust::make_zip_iterator(thrust::make_tuple(data[1], data[2]));
// 			auto m_out = thrust::make_zip_iterator(thrust::make_tuple(data[1+in_num], data[2+in_num]));
// 			thrust::sort_by_key(thrust::cuda::par.on(stream),k_out, k_out+tuple_ct, m_out);
// 			auto end2 = thrust::reduce_by_key(thrust::cuda::par.on(stream),k_out, k_out+tuple_ct, m_out, k, m, TupleEqual_1(), TupleSum_2());
//  			return  end2.first - k;
 		} else if (metric_num == 3) {
 			auto m = thrust::make_zip_iterator(thrust::make_tuple(data[1], data[2], data[3]));
 	 		auto m_out = thrust::make_zip_iterator(thrust::make_tuple(data[1+in_num], data[2+in_num],data[3+in_num]));
 	 		thrust::sort_by_key(k, k+tuple_ct, m);
			auto end = thrust::reduce_by_key(k, k+tuple_ct, m, k_out, m_out, TupleEqual_1(), TupleSum_3());
			return  end.first - k_out;
		} else assert(false);
	} else if(key_num  == 2) {
		auto k = thrust::make_zip_iterator(thrust::make_tuple(data[0], data[1]));
		auto k_out = thrust::make_zip_iterator(thrust::make_tuple(data[0+in_num], data[1+in_num]));
		if (metric_num == 1) {
			auto m = thrust::make_zip_iterator(thrust::make_tuple(data[2]));
			auto m_out = thrust::make_zip_iterator(thrust::make_tuple(data[2+in_num]));
 			thrust::sort_by_key(k, k+tuple_ct, m);
			auto end = thrust::reduce_by_key(k, k+tuple_ct, m, k_out, m_out, TupleEqual_2(), TupleSum_1());
			return  end.first - k_out;
		} else if (metric_num == 2) {
	 		auto m = thrust::make_zip_iterator(thrust::make_tuple(data[2], data[3]));
	 		auto m_out = thrust::make_zip_iterator(thrust::make_tuple(data[2+in_num], data[3+in_num]));
	 		thrust::sort_by_key(k, k+tuple_ct, m);
			auto end = thrust::reduce_by_key(k, k+tuple_ct, m, k_out, m_out, TupleEqual_2(), TupleSum_2());
			return  end.first - k_out;
 		} else if (metric_num == 3) {
 			auto m = thrust::make_zip_iterator(thrust::make_tuple(data[2], data[3], data[4]));
			auto m_out = thrust::make_zip_iterator(thrust::make_tuple(data[2+in_num], data[3+in_num], data[4+in_num]));
			thrust::sort_by_key(k, k+tuple_ct, m);
			auto end = thrust::reduce_by_key(k, k+tuple_ct, m, k_out, m_out, TupleEqual_2(), TupleSum_3());
			return  end.first - k_out;
		} else assert(false);
	} else if(key_num == 3) {
		auto k = thrust::make_zip_iterator(thrust::make_tuple(data[0], data[1], data[2]));
		auto k_out = thrust::make_zip_iterator(thrust::make_tuple(data[0+in_num], data[1+in_num], data[2+in_num]));
		if (metric_num == 1) {
			auto m = thrust::make_zip_iterator(thrust::make_tuple(data[3]));
			auto m_out = thrust::make_zip_iterator(thrust::make_tuple(data[3+in_num]));
			thrust::sort_by_key(k, k+tuple_ct, m);
			auto end = thrust::reduce_by_key(k, k+tuple_ct, m, k_out, m_out, TupleEqual_3(), TupleSum_1());
			return  end.first - k_out;
		} else if (metric_num == 2) {
			auto m = thrust::make_zip_iterator(thrust::make_tuple(data[3], data[4]));
			auto m_out = thrust::make_zip_iterator(thrust::make_tuple(data[3+in_num], data[4+in_num]));
			thrust::sort_by_key(k, k+tuple_ct, m);
			auto end = thrust::reduce_by_key(k, k+tuple_ct, m, k_out, m_out, TupleEqual_3(), TupleSum_2());
			return  end.first - k_out;
 		} else if (metric_num == 3) {
 			auto m = thrust::make_zip_iterator(thrust::make_tuple(data[3], data[4], data[5]));
			auto m_out = thrust::make_zip_iterator(thrust::make_tuple(data[3+in_num], data[4+in_num], data[5+in_num]));
			thrust::sort_by_key(k, k+tuple_ct, m);
			auto end = thrust::reduce_by_key(k, k+tuple_ct, m, k_out, m_out, TupleEqual_3(), TupleSum_3());
			return  end.first - k_out;
		} else assert(false);

	}
}

void Pipeline::GenCode(CodeContext & ctx) {
	transform_->GenCode(*this, ctx, false);
	transform_->GenCode(*this, ctx, true);
}
