/*
 * operator.cu
 *
 *  Created on: 2019年2月21日
 *      Author: imdb
 */

#include "operator.hpp"
#include "../codegen/codegen.hpp"
#include "../codegen/template.hpp"
#include "resource.hpp"

Loader::Loader(const OperatorInfo & info) {
	info_ = info;
	if(info_.type() == OperatorInfo::Loader) {
		col_spilter_ = *(info_.args().find("col_spilter")->second.c_str());
		row_spilter_ = *(info_.args().find("row_spilter")->second.c_str());
	}
	else
		assert(false);
}


int Loader::Host(int task_id,  Bulk * in, Bulk * out) {
	char item[1024*4];
	int k = 0;
	char * ptr = (char*)in->buffer_;
	int col = 0;
	int row = 0;
	for (auto i = 0; i < in->capacity_ && row <= out->max_tuple_count_ ; i++, ptr++) {
		if (*ptr == col_spilter_) {
			item[k] = '\0';
 //			assert(output->GetColumnBuffer(col) != nullptr);
			void * buffer = out->column_buf_[col];
			ValType column_type = static_cast<ValType>(out->schema_.GetColumnType(col));
			size_t column_size = out->schema_.GetColumnSize(col);
			GetValueOfItemBySwitch(buffer+row*column_size, item,	column_type, column_size);
			col++;
			k = 0;
		} else if (*ptr == row_spilter_) {
			col = 0;
			row++;
			k = 0;
		} else
			item[k++] = *ptr;
	}
	out->tuple_count_  = row;
 }

int Loader::DeviceNext(int task_id,BulkVector & in, BulkVector & out) {
/*	Bulk * in = in_vec.GetMainBulk();
	Bulk * out = out_vec.GetMainBulk();
	printf("hello world : %d\n ", task_id);
   char * ptr = (char*)in->buffer_;
   int t = 0;
   int *out_0 = (int *)out->GetColumnBuffer(0);
	int *out_1 = (int *)out->GetColumnBuffer(1);
	int *out_2 = (int *)out->GetColumnBuffer(2];
	int *out_3 = (int *)out->column_buf_[3];
	int *out_4 = (int *)out->column_buf_[4];
	int *out_5 = (int *)out->column_buf_[5];
	char *out_6 = (char *)out->column_buf_[6];
	int *out_7 = (int *)out->column_buf_[7];
	int *out_8 = (int *)out->column_buf_[8];
	int *out_9 = (int *)out->column_buf_[9];
	int *out_10 = (int *)out->column_buf_[10];
	int *out_11 = (int *)out->column_buf_[11];
	int *out_12 = (int *)out->column_buf_[12];
	int *out_13 = (int *)out->column_buf_[13];
	int *out_14 = (int *)out->column_buf_[14];
	int *out_15 = (int *)out->column_buf_[15];
	char *out_16 = (char *)out->column_buf_[16];
	 for(int i=0; i<10;i++) {
		 int move;
	    int item = sscanf(ptr,"%d,%d,%d,%d,%d,%d,%[^,],%d,%d,%d,%d,%d,%d,%d,%d,%d,%[^,],\n%n",out_0,out_1,out_2,out_3,out_4,out_5,out_6,out_7,out_8,out_9,out_10,out_11,out_12,out_13,out_14,out_15,out_16, &move);
	     ptr+= move;
	     if (task_id == 0 && i > 9900) {
		cout << "move:" << move << " item:"<<item<< endl;
	     }
  out_0++;
	out_1++;
	out_2++;
	out_3++;
	out_4++;
	out_5++;
	out_6+=15;
	out_7++;
	out_8++;
	out_9++;
	out_10++;
	out_11++;
	out_12++;
	out_13++;
	out_14++;
	out_15++;
	out_16+=10;
	 }*/
	//TransformFuc fuc = JitCompiler::GetHostFuc(info_.id(),info_.id());
	//fuc(task_id, in, out);
	return 0;
 }

void Loader::GenCode(const Pipeline & pipe, CodeContext & ctx, bool isGPU) {
	map<string, string> args;
	args["task_id"] = pipe.info_.id();
	args["pointer"] = CodeGen::GenBulkPointerDefine(pipe.output_info_,"out");
	args["pattern"] = CodeGen::GenBulkPattern(pipe.output_info_, col_spilter_);
	args["list"] = CodeGen::GenBulkPointerList(pipe.output_info_,"out");
	args["ptr_move"] = CodeGen::GenBulkPointerMove(pipe.output_info_, "out");
 	string code =  CodeGen::GenFromTemplate(CodeTemplate::LoaderTransform, args);
	ctx.host_code_.push_back(code);
}

void Query::Open(const Pipeline & pipe) {

}

bool use_early_copy = false;


int Query::Next(int task_id, Bulk * bulk_in, BulkVector & in, BulkVector & out, bool isGPU) {
	size_t tuple_ct = in.GetTupleCount();
//	cout << "tuple_ct:" << tuple_ct << endl;
	if (isGPU) {
		struct  timeval begin, parepare, end;

		gettimeofday(&begin, nullptr);
		CUstream stream  = ResMgr::GetStream(task_id);
		gettimeofday(&parepare, nullptr);
		double ms = (parepare.tv_sec - begin.tv_sec) * 1000 + (parepare.tv_usec - begin.tv_usec)/1000.0;
		ResMgr::wait_time[task_id] = ms;
	//	cout << "CUDA task :" << task_id <<" waittime: " << ms << " ms"<< endl;

 		CUmodule module;
 		CUresult err;
		cudaFree(0);
	   CUfunction kernel = JitCompiler::GetDeviceFuc(info_.id(), info_.id());
	   assert(kernel != nullptr);

	   CUdeviceptr buf;
	   if (use_early_copy) {
		   BulkVector new_in = in;
		   assert(cuMemAlloc(&buf, bulk_in->capacity_) == CUDA_SUCCESS);
		 //  cudaMallocManaged(((void**)(&buf)), bulk_in->capacity_);
		 //  memcpy((void*)buf, bulk_in->buffer_, bulk_in->capacity_);
		   assert(cuMemcpyAsync(buf, (CUdeviceptr)(bulk_in->buffer_), bulk_in->capacity_, stream) == CUDA_SUCCESS);

		   for (auto i = 0; i < bulk_in->schema_.GetColumnCount(); i++)
			   new_in.buffer_[0][i] = (void*)buf+bulk_in->column_pos_[i];


		   in = new_in;
	   }

		void *args[] = { &task_id,  &in, &out, &tuple_ct};
		auto block_num = tuple_ct /  CTX.cuda_thread_num_ + 1;
	//	auto block_num = CTX.cuda_block_num_;
		err = cuLaunchKernel(kernel,
					block_num, 1, 1,   // grid dim
				   CTX.cuda_thread_num_, 1, 1,    // block dim
				   0, stream,             // shared mem and stream
				   args,                // arguments
					0);
		if (err != CUDA_SUCCESS)
			cout << "task " << task_id << " error kernel:" << err << endl;
		assert(cuStreamSynchronize(stream) == CUDA_SUCCESS);

		if (use_early_copy) {
			assert(cuMemFree(buf) == CUDA_SUCCESS);
		}

	 	gettimeofday(&end, nullptr);

		double ms2 = (end.tv_sec - parepare.tv_sec) * 1000 + (end.tv_usec - parepare.tv_usec)/1000.0;
		cout << "CUDA task :" << task_id <<" runtime: " << ms2 << " ms"<< endl;
		ResMgr::exec_time[task_id] = ms2;
	//	cuCtxSynchronize();
		ResMgr::ReturnStream(stream);
 		return tuple_ct;

	//	cout << "task: " << task_id << " finish kernel" << endl;
	} else  {
		//cout << "opID:" << info_.id() << endl;
		TransformFuc fuc = JitCompiler::GetHostFuc(info_.id(),info_.id());
		return fuc(task_id, in, out, tuple_ct);
	}
}

int Query::HostReduce(int task_id, const Pipeline & pipe, Bulk * in){
 	int key_num = pipe.transform_->reduce_key_.size();
	int metric_num = pipe.transform_->metric_exp_.size();
	int in_num = key_num+metric_num;
	cout << "key_num:" << key_num << " metric_num:" << metric_num << endl;
 	if (key_num == 1) {
		auto key = MakeIterator_1(in->column_buf_[0]);
		auto key_out = MakeIterator_1(in->column_buf_[in_num]);
		if (metric_num == 1) {
			auto metric = MakeIterator_1(in->column_buf_[1]);
			auto metric_out = MakeIterator_1(in->column_buf_[1+in_num]);
			thrust::sort_by_key(thrust::seq, key, key+in->tuple_count_, metric);
//			auto new_end = thrust::reduce_by_key(thrust::seq, (int*)in->column_buf_[0],
//					(int*)in->column_buf_[0]+in->tuple_count_, (int*)in->column_buf_[1],
//					(int*)in->column_buf_[2], (int*)in->column_buf_[3]);
			auto new_end = thrust::reduce_by_key(thrust::seq, key, key+in->tuple_count_,
					metric, key_out, metric_out,TupleEqual_1(), TupleSum_1());
			return new_end.first -key_out;
 		} else if (metric_num == 2) {
			auto metric = MakeIterator_2(in->column_buf_[1], in->column_buf_[2]);
			auto metric_out = MakeIterator_2(in->column_buf_[1+in_num], in->column_buf_[2+in_num]);
			thrust::sort_by_key(thrust::seq, key, key+in->tuple_count_, metric);
			auto new_end = thrust::reduce_by_key(thrust::seq, key, key+in->tuple_count_,
					metric, key_out, metric_out,TupleEqual_1(), TupleSum_2());
			return new_end.first - key_out;
 		} else if (metric_num == 3) {
			auto metric = MakeIterator_3(in->column_buf_[1], in->column_buf_[2],  in->column_buf_[3]);
			auto metric_out = MakeIterator_3(in->column_buf_[1+in_num], in->column_buf_[2+in_num],in->column_buf_[3+in_num]);
			thrust::sort_by_key(thrust::seq, key, key+in->tuple_count_, metric);
			auto new_end = thrust::reduce_by_key(thrust::seq, key, key+in->tuple_count_,
					metric, key_out, metric_out,TupleEqual_1(), TupleSum_3());
			return new_end.first - key_out;
		} else assert(false);
	} else if(key_num  == 2) {
		auto key = MakeIterator_2(in->column_buf_[0], in->column_buf_[1]);
		auto key_out = MakeIterator_2(in->column_buf_[0+in_num], in->column_buf_[1+in_num]);
		if (metric_num == 1) {
			auto metric =  MakeIterator_1(in->column_buf_[2]);
 			auto metric_out =MakeIterator_1(in->column_buf_[2+in_num]);
			thrust::sort_by_key(thrust::seq, key, key+in->tuple_count_, metric);
 			auto new_end = thrust::reduce_by_key(thrust::seq, key, key+in->tuple_count_,
					metric, key_out, metric_out,TupleEqual_2(), TupleSum_1());
			return new_end.first -key_out;
		} else if (metric_num == 2) {
			auto metric = MakeIterator_2(in->column_buf_[2], in->column_buf_[3]);
			auto metric_out = MakeIterator_2(in->column_buf_[2+in_num], in->column_buf_[3+in_num]);
			thrust::sort_by_key(thrust::seq, key, key+in->tuple_count_, metric);
			auto new_end = thrust::reduce_by_key(thrust::seq, key, key+in->tuple_count_,
					metric, key_out, metric_out,TupleEqual_2(), TupleSum_2());
			return new_end.first - key_out;
 		} else if (metric_num == 3) {
			auto metric = MakeIterator_3(in->column_buf_[2], in->column_buf_[3], in->column_buf_[4]);
			auto metric_out = MakeIterator_3(in->column_buf_[2+in_num], in->column_buf_[3+in_num],in->column_buf_[4+in_num]);
			thrust::sort_by_key(thrust::seq, key, key+in->tuple_count_, metric);
			auto new_end = thrust::reduce_by_key(thrust::seq, key, key+in->tuple_count_,
					metric, key_out, metric_out,TupleEqual_2(), TupleSum_3());
			return new_end.first - key_out;
		} else assert(false);
	} else if(key_num == 3) {
		auto key = MakeIterator_3(in->column_buf_[0], in->column_buf_[1], in->column_buf_[2]);
		auto key_out = MakeIterator_3(in->column_buf_[0+in_num], in->column_buf_[1+in_num],in->column_buf_[2+in_num]);
		if (metric_num == 1) {
			auto metric =  MakeIterator_1(in->column_buf_[3]);
 			auto metric_out =MakeIterator_1(in->column_buf_[3+in_num]);
			thrust::sort_by_key(thrust::seq, key, key+in->tuple_count_, metric);
 			auto new_end = thrust::reduce_by_key(thrust::seq, key, key+in->tuple_count_,
					metric, key_out, metric_out,TupleEqual_3(), TupleSum_1());
			return new_end.first -key_out;
		} else if (metric_num == 2) {
			auto metric = MakeIterator_2(in->column_buf_[3], in->column_buf_[4]);
			auto metric_out = MakeIterator_2(in->column_buf_[3+in_num], in->column_buf_[4+in_num]);
			thrust::sort_by_key(thrust::seq, key, key+in->tuple_count_, metric);
			auto new_end = thrust::reduce_by_key(thrust::seq, key, key+in->tuple_count_,
					metric, key_out, metric_out,TupleEqual_3(), TupleSum_2());
			return new_end.first - key_out;
 		} else if (metric_num == 3) {
			auto metric = MakeIterator_3(in->column_buf_[3], in->column_buf_[4],  in->column_buf_[5]);
			auto metric_out = MakeIterator_3(in->column_buf_[3+in_num], in->column_buf_[4+in_num],in->column_buf_[5+in_num]);
			thrust::sort_by_key(thrust::seq, key, key+in->tuple_count_, metric);
			auto new_end = thrust::reduce_by_key(thrust::seq, key, key+in->tuple_count_,
					metric, key_out, metric_out,TupleEqual_3(), TupleSum_3());
			return new_end.first - key_out;
		} else assert(false);
	} else assert(false);


}

int Query::DeviceNext(int task_id, BulkVector & in, BulkVector & out) {
	return 0;
}

void Query::GenCode(const Pipeline & pipe, CodeContext & ctx, bool isGPU) {
	string code;
	map<string, string> args;
	map<string, string> variable;
	args["pipe_id"] = pipe.info_.id();
	//cout << "pipe_id:" <<info_.id() << endl;
	args["filter_block"] = CodeGen::GenFilterBlock(variable, pipe,filter_exp_, isGPU);
	args["join_block"] = CodeGen::GenJoinBlock(variable, pipe, join_lkey_, join_rkey_, isGPU);
	if (!pipe.transform_->is_reduce_)
		args["select_block"] = CodeGen::GenSelectBlock(variable, pipe, select_exp_, isGPU);
	else
		args["reduce_block"] = CodeGen::GenReduceBlock(variable, pipe, metric_exp_, reduce_key_, isGPU);
	args["define_variable"] = CodeGen::GenVariableDefine(variable);
	if (!isGPU)	{
		code = CodeGen::GenFromTemplate(CodeTemplate::HostQueryTransform, args);
		ctx.host_code_.push_back(code);
	} else {
		code = CodeGen::GenFromTemplate(CodeTemplate::DeviceQueryTransform, args);
		ctx.device_code_.push_back(code);
	}
}

void Query::GenKernelCode(const Pipeline & pipe, CodeContext & ctx){

}

