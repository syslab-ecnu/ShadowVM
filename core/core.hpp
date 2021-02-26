/*
 * runtime.hpp
 *
 *  Created on: 2019年2月21日
 *      Author: imdb
 */

#ifndef RUNTIME_HPP_
#define RUNTIME_HPP_

#include "../com/cpp/com.hpp"
using namespace ShadowVM;
class Core {
public:
	void CompileGraph(const GraphInfo & graph) ;
	void ExecGraph(const GraphInfo & graph) ;
	void CompilePipeline(const PipelineInfo & pipeline) ;
	void ExecPipeline(const PipelineInfo & pipeline) ;
private:
	unordered_map<string, string> pipeline_order;
};


#endif /* RUNTIME_HPP_ */
