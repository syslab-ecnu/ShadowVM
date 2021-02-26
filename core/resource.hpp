/*
 * resource.hpp
 *
 *  Created on: 2019年3月29日
 *      Author: imdb
 */

#ifndef RESOURCE_HPP_
#define RESOURCE_HPP_
#include "../com/cpp/com.hpp"

#include "semaphore.h"

class ResMgr {
public:
	~ResMgr();
	static void Init();
	static CUstream GetStream(int offset = 0);
	static void ReturnStream(CUstream stream);
	static bool GetCPU(int offset = 0);
	static void ReturnCPU();
	static int FreeCPUNum();
	static int FreeStreamNum();
	static int CPU_num_;
	static int free_CPU_;
	static int busy_CPU_;
	static int stream_num_;
	static map<int, CUstream> free_stream_;
	static set<int> busy_stream_;
	static bool enable_hybrid_exec_;
	static int CPU_ratio_;
	static const int sch_clock_ms_ = 5;

	static bool enable_spec_;

	static sem_t CPU_res_;
	static sem_t stream_res_;

	static mutex mtx_;
	static double wait_time[1000];
	static double exec_time[1000];
	static string trace_path_;
};



#endif /* RESOURCE_HPP_ */
