/*
 * pipeline.hpp
 *
 *  Created on: 2019年2月25日
 *      Author: imdb
 */

#ifndef PIPELINE_HPP_
#define PIPELINE_HPP_
#include "../com/cpp/com.hpp"
#include "../core/operator.hpp"
class Pipeline {
public:
  	Pipeline(const PipelineInfo & info);
 	void Open();
 	void Run(bool GPU = false);
  	void Close();
  	static void TransformTask(Pipeline * pipe, size_t id, bool GPU);
  	static int ShrinkTask(int tuple_ct, const vector<int*> & column);
  	static int ReduceTask(Pipeline * pipe, size_t id, bool GPU);
  	static int GPUReduceTask(Pipeline * pipe, size_t id, bool GPU);
  	void GenCode(CodeContext & ctx);
	bool is_loader_ = false;
 	PipelineInfo info_;
 	Operator * source_ = nullptr;
 	Operator * transform_ = nullptr;
 	Operator * sink_ = nullptr;
	vector<ObjectInfo> input_info_;
	vector<Object*> ref_object_;
	vector<Object*> host_ref_object_;
	ObjectInfo output_info_;
	int finished_ct_ = 0;
	vector<bool> task_isGPU_ = vector<bool>(1000, false);
	vector<bool> task_stat_ = vector<bool>(1000, false);
 };


#endif /* PIPELINE_HPP_ */
