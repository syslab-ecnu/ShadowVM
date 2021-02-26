/*
 * resource.cu
 *
 *  Created on: 2019年3月29日
 *      Author: imdb
 */


#include "resource.hpp"

map<int, CUstream> ResMgr::free_stream_;
set<int> ResMgr::busy_stream_;
int ResMgr::free_CPU_ = CPU_num_;
int ResMgr::busy_CPU_ = 0;
bool ResMgr::enable_hybrid_exec_ = false;
int ResMgr::CPU_ratio_ = 5;

bool ResMgr::enable_spec_ = false;


int ResMgr::CPU_num_= 16; // 16;
int ResMgr::stream_num_ = 8;

mutex ResMgr::mtx_;

sem_t ResMgr::CPU_res_;
sem_t ResMgr::stream_res_;

double ResMgr::wait_time[1000];
double ResMgr::exec_time[1000];
string ResMgr::trace_path_ = "/home/lizf/data/trace/";



void ResMgr::Init() {
	cout << "init resources" << endl;
	cudaFree(0);
	for(auto i  = 0; i < stream_num_; i++) {
		CUstream stream;
		assert(cuStreamCreate(&stream, 0) == CUDA_SUCCESS);
		free_stream_[i] = stream;
		//cout << "stream:" << stream << endl;
	}
	sem_init(&CPU_res_, 0, CPU_num_);
	sem_init(&stream_res_, 0, stream_num_);
	//cout << "end init resources" << endl;
}

ResMgr::~ResMgr() {
	sem_destroy(&CPU_res_);
	sem_destroy(&stream_res_);
}


CUstream ResMgr::GetStream(int offset) {
	sem_wait(&stream_res_);
	for(auto i = 0; i < 1000 / sch_clock_ms_ * 200; i++)  {
		{
			lock_guard<std::mutex> lck (mtx_);
			if (free_stream_.size() >0 )
 			for (auto stream : free_stream_) {
 				free_stream_.erase(stream.first);
 				busy_stream_.insert(stream.first);
			//	if (i > 0) cout << "slow get GPU stream:"<< i << endl;

 	//			cout << "stream:" << free_stream_.size() << "," << busy_stream_.size() << ":" << stream.second << endl;
 				return stream.second;
  			}
		}
	// 	usleep((sch_clock_ms_ + rand() % 100)* 1);

		usleep(10);
	//d  	cout << "GPU wait for : " << i << endl;
	}
	//cout << "can not get a stream" << endl;
	return nullptr;
}

void ResMgr::ReturnStream(CUstream stream) {
	sem_post(&stream_res_);
 	 lock_guard<std::mutex> lck (mtx_);
	 for (auto stream_id:busy_stream_) {
		 free_stream_[stream_id] = stream;
		 busy_stream_.erase(stream_id);
		 return;
	 }
}

bool ResMgr::GetCPU(int offset) {
	sem_wait(&CPU_res_);
	for(auto i = 0; i < 1000 / sch_clock_ms_ * 200; i++) {
		{  lock_guard<std::mutex> lck (mtx_);
			if (busy_CPU_ < CPU_num_ && free_CPU_ > 0) {
				free_CPU_ --;
				busy_CPU_ ++;
			//	if (i > 0) cout << "slow get cpu:"<< i << endl;
				return true;
		}	}
//		usleep((sch_clock_ms_ + rand() % 30 ) * 10);
		usleep(10);
 //	 	cout << "CPU wait for : " << i << endl;
	}
	//cout << "can not get a CPU" << endl;
 	return false;
}

void ResMgr::ReturnCPU() {
	sem_post(&CPU_res_);
	lock_guard<std::mutex> lck (mtx_);
	if (busy_CPU_ == 0 || free_CPU_ == CPU_num_)
		return;
	free_CPU_ ++;
	busy_CPU_ --;
}

int ResMgr::FreeCPUNum() {
	lock_guard<std::mutex> lck (mtx_);
	return free_CPU_;
}

int ResMgr::FreeStreamNum() {
	lock_guard<std::mutex> lck (mtx_);
	return free_stream_.size();
}

