/*
 * service.cu
 *
 *  Created on: 2021年1月29日
 *      Author: lizhif
 */

#include "../com/cpp/com.hpp"
#include "../service/service.hpp"
#include "../core/resource.hpp"

#include "../core/pipeline.hpp"
#include "../codegen/compiler.hpp"
#include "../test/test.hpp"


 void ShadowVMService::Init(){
	 ResMgr::Init();
	 //ResMgr::enable_hybrid_exec_ = true;
	 //ResMgr::CPU_ratio_ = 2;  // 1~10
 }
unordered_map<string, Object* > ShadowVMService::cached_obj;


grpc::Status ShadowVMService::RunPipelineCMD(ServerContext* context,
		const PipelineInfo* pipeinfo, Ret * ret)  {
//	cout << "=> begin " << pipeinfo->id() << endl;
	cudaFree(0);

	PipelineTest pipe(*pipeinfo);
	//cout << pipe.pipe_info_.DebugString() << endl;
	// pipe.Execute(nullptr, {}, false, false);

	Object * input_obj = nullptr;
	vector<Object *> ref_obj;
	int object_ct = pipeinfo->pre_pipeline_id_size() ;
	if (object_ct > 0) {
		if (cached_obj.count(pipeinfo->pre_pipeline_id(0)) == 0)
			return grpc::Status::CANCELLED;
		input_obj = cached_obj[pipeinfo->pre_pipeline_id(0)];
		assert(input_obj != nullptr);
	}
	for (auto i = 1; i < object_ct; i++) {
		auto object_id = pipeinfo->pre_pipeline_id(i);
		if (cached_obj.count(object_id) == 0)
			return grpc::Status::CANCELLED;
		auto object = cached_obj[object_id];
		ref_obj.push_back(object);
	}

	Object* rc = pipe.Run(input_obj, ref_obj,
			pipeinfo->codegen(), pipeinfo->debug(), pipeinfo->isgpu());
	assert(rc != nullptr);
	cached_obj[pipeinfo->id()] = rc;
//	cout << "=> finish " << pipeinfo->id() << endl;

	//cout <<  rc->info_.DebugString() << endl;

 	ret->set_success(true);
	ret->set_info("ShadowVM exec: " + pipeinfo->id());

	return grpc::Status::OK;
}

