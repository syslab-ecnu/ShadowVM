/*
 * jit-compiler.cu
 *
 *  Created on: 2019年2月21日
 *      Author: imdb
 */

#include "compiler.hpp"
#include "codegen.hpp"
#include "template.hpp"


map<string, void *> JitCompiler::host_lib_;
map<string,  CUmodule > JitCompiler::device_lib_;
map<string, TransformFuc> JitCompiler::host_fuc_;
map<string, CUfunction> JitCompiler::device_fuc_;
mutex JitCompiler::mtx_;

void JitCompiler::CompileHostFuc(const string & graph_id, const string & code) {
	struct timeval begin, end;
	gettimeofday(&begin, nullptr);
	char result[1024];
	string source_name = CTX.home_path_+graph_id +".cpp";
	string include_name = " -I " + CTX.include_path_ + "  ";
	string lib_name = CTX.home_path_+graph_id +".so";
	int fd = open(source_name.c_str(), O_CREAT|O_RDWR|O_TRUNC, 0777);
	assert(fd > 0);
	write(fd, code.c_str(), code.size());
	close(fd);
 	string cmd = "clang -g -w -O2 -std=c++11 -fPIC -shared  -o "+ lib_name + include_name  +source_name;
//	string cmd = "nvcc -w -Xcompiler -fPIC -shared -std=c++11 -o "+ lib_name + include_name  +source_name;
 	cout << cmd << endl;
 	FILE* fp = popen(cmd.c_str(), "r");
 	assert(fp != nullptr);
   while (fgets(result, 1024, fp) != NULL) {
	   printf("%s\n", result);
	   assert(false);
	   exit(0);
   }
	pclose(fp);
 	lock_guard<std::mutex> lck (mtx_);
	host_lib_[graph_id] = dlopen(lib_name.c_str(), RTLD_GLOBAL | RTLD_NOW);
	assert(host_lib_[graph_id] != nullptr);
	//cout << "[try to load lib] " << lib_name << "=>" << host_lib_[graph_id] << endl;
	char * error;
   if ((error=dlerror())!=NULL) { printf(">>>%s\n",error); assert(false); }
	gettimeofday(&end, nullptr);
	size_t time = (end.tv_sec - begin.tv_sec)*1000 + (end.tv_usec - begin.tv_usec)/1000;
	cout << "so compile + load time of "<<graph_id << ":" << time << "ms" << endl;
	//system(cmd.c_str());
}
void JitCompiler::CompileDeviceFuc(const string & graph_id, const string & code) {
	struct timeval begin, end;
	gettimeofday(&begin, nullptr);
	char result[1024];
	string source_name = CTX.home_path_+graph_id +".cu";
	string include_name = " -I " + CTX.include_path_ + "  ";
	string lib_name = CTX.home_path_+graph_id +".ptx";
	int fd = open(source_name.c_str(), O_CREAT|O_RDWR|O_TRUNC, 0777);
	assert(fd > 0);
	write(fd, code.c_str(), code.size());
	close(fd);
	string cmd = "nvcc -w -O2 -std=c++11 -ptx -o  "+ lib_name + include_name  +source_name;
	cout << cmd << endl;
	FILE* fp = popen(cmd.c_str(), "r");
	assert(fp != nullptr);
   while (fgets(result, 1024, fp) != NULL) {
	   printf("%s\n", result);
	   assert(false);
	   exit(0);
   }
	pclose(fp);
 	CUmodule module;
 //	cout << "lib_name:" << lib_name << endl;
 	CUresult ret = cuModuleLoad(&module, lib_name.c_str());
 //	cout << "cuModuleLoad: "<< ret << endl;
 	assert (ret == CUDA_SUCCESS);
 	lock_guard<std::mutex> lck (mtx_);
 //	cout << device_lib_.size() << "->"<<graph_id<<":" << module << "->";
 	device_lib_[graph_id] = module;
	gettimeofday(&end, nullptr);
	size_t time = (end.tv_sec - begin.tv_sec)*1000 + (end.tv_usec - begin.tv_usec)/1000;
	 cout << "ptx compile + load time of "<<graph_id << ":" << time << "ms" << endl;
}

TransformFuc JitCompiler::GetHostFuc(const string & graph_id, const string & pipeline_id) {
	//cout <<  graph_id << ":" << pipeline_id << endl;
	lock_guard<std::mutex> lck (mtx_);
	if(host_fuc_.find(pipeline_id) != host_fuc_.end())
		return host_fuc_[pipeline_id];
	host_fuc_[pipeline_id] = (TransformFuc) dlsym(host_lib_[graph_id], pipeline_id.c_str());
	assert(host_fuc_[pipeline_id] != nullptr);
	char * error;
   if ((error=dlerror()) != NULL) printf(">>>%s\n",error);
//	cout << "[try to load fuc] " << pipeline_id<< "=>" << host_fuc_[pipeline_id] << endl;
	return host_fuc_[pipeline_id];
}

CUfunction JitCompiler::GetDeviceFuc(const string & graph_id, const string & pipeline_id) {
 	lock_guard<std::mutex> lck (mtx_);
	if(device_fuc_.find(pipeline_id) != device_fuc_.end())
		return device_fuc_[pipeline_id];
	CUfunction kernel;
	CUmodule module = device_lib_[graph_id];
	assert( cuModuleGetFunction(&kernel, module, pipeline_id.c_str()) == CUDA_SUCCESS);
//	cout <<"module:" << module << " kernel:" << kernel << endl;
	device_fuc_[pipeline_id] = kernel;
 	return kernel;
}

string CodeContext::ExportHostCode() {
	string code = "";
	code += "#include \"define.hpp\" \n";
	code += "#include \"cpu-runtime.hpp\" \n";
 	code += "#include \"cpu-udf.hpp\" \n";
 	code += "extern \"C\" {\n";
	for(auto & it: host_code_) code += it;
	code += "}";
 	return code;
}

string CodeContext::ExportDeviceCode() {
	string code = "";
	code += "#include \"define.hpp\" \n";
 	code += "#include \"cuda-runtime.cu\" \n";
 	code += "#include \"cuda-udf.cu\" \n";
 	code += "extern \"C\" {\n";
	for(auto & it: device_code_) code += it;
	code += "}";
 	return code;
}


