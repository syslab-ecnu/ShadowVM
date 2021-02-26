/*
 * service.h
 *
 *  Created on: 2019年2月21日
 *      Author: imdb
 */

#ifndef SERVICE_H_
#define SERVICE_H_
#include <iostream>
#include <thread>
#include <grpcpp/grpcpp.h>
#include "../com/cpp/svm.grpc.pb.h"
#include "../com/cpp/svm.pb.h"

#include "../com/cpp/com.hpp"
#include "../core/resource.hpp"
#include "../core/pipeline.hpp"
#include "../codegen/compiler.hpp"
#include "../test/test.hpp"


using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using ShadowVM::HelloRequest;
using ShadowVM::HelloReply;
using ShadowVM::TestRequest;
using ShadowVM::TestReply;
using ShadowVM::Greeter;
using ShadowVM::ShadowVMAPI;
using ShadowVM::Ret;

class ShadowVMClient {
 public:
	ShadowVMClient() {}
	ShadowVMClient(shared_ptr<Channel> channel)
      : stub_(ShadowVMAPI::NewStub(channel)) {}
 static ShadowVMClient Create(const string & endpoint = "localhost:50051") {
	 ShadowVMClient client(grpc::CreateChannel(
	       endpoint, grpc::InsecureChannelCredentials()));
	 return client;
 }
  void Test(const std::string & test) {
	PipelineInfo pipeline;
    pipeline.set_id("pipeline1");
    ShadowVM::Ret reply;
    ClientContext context;
    grpc::Status status = stub_->RunPipelineCMD(&context, pipeline, &reply);
    if (status.ok())
      cout << reply.info() << endl;
    else
      cout << "RPC failed:"<< status.error_message() << endl;
  }

  void RunPipelineCMD(PipelineInfo & pipeline) {
	 ClientContext context;
	 ShadowVM::Ret reply;
	 grpc::Status status = stub_->RunPipelineCMD(&context, pipeline, &reply);
	 if (status.ok())
	   cout << reply.info() << endl;
	 else
	   cout << "RPC failed:"<< status.error_message() << endl;
  }
 private:
   unique_ptr<ShadowVMAPI::Stub> stub_;
};

class ShadowVMService final : public ShadowVMAPI::Service {
/*	grpc::Status SayHello(ServerContext* context, const HelloRequest* request,
                  HelloReply* reply) override {
		std::string prefix("Hello ");
		reply->set_message(prefix + request->name());
		return grpc::Status::OK;
	}
	grpc::Status SayTest(ServerContext * context, const TestRequest * request,
                  TestReply * reply) override {
		cout << "reviced"  << endl;
		reply->set_message("@@@@"+request->name()+"@@@@");
		return grpc::Status::OK;
	}*/

//	void Execute(const PipelineInfo* pipeline, ShadowVM::Ret * reply);

	grpc::Status RunPipelineCMD(ServerContext* context, const PipelineInfo* pipeline,
			ShadowVM::Ret * ret);
	std::unique_ptr<Server> server;
public:
	static unordered_map<string, Object* > cached_obj;
	void Launch(const string & endpoint = "0.0.0.0:50051") {
	  ServerBuilder builder;
	  builder.AddListeningPort(endpoint, grpc::InsecureServerCredentials());
	  builder.RegisterService(this);
	  server = builder.BuildAndStart();
	  std::cout << "Server listening" << std::endl;
	  Init();
	}

  void Init() ;
  void Wait() {
	  server->Wait();
  }


};


#endif /* SERVICE_H_ */
