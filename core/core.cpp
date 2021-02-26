/*
 * rumtime.cu
 *
 *  Created on: 2019年2月21日
 *      Author: imdb
 */



#include "core.hpp"

void Core::CompileGraph(const GraphInfo & graph) {
	cout << graph.id() << endl;
}
void Core::ExecGraph(const GraphInfo & graph) {
	CompileGraph(graph);
}
void Core::CompilePipeline(const PipelineInfo & pipeline) {

}

void Core::ExecPipeline(const PipelineInfo & pipeline) {
	CompilePipeline(pipeline);
}
