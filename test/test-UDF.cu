/*
 * test-udf.cu
 *
 *  Created on: 2019年4月11日
 *      Author: imdb
 */

 #include "test.hpp"
#include "../com/cpp/com.hpp"
void test_UDF() {
	auto udf = load_UDF();
	ObjectInfo schema = CreateSchema({ValType::Int32,ValType::Int32}, {4,4});
 	PipelineTest query("udf");
// 	string perceptron = "perceptron((char*)($(0.0)))";
// 	string perceptron = "l2Distance($(0.0),$(0.0))";
// 	string perceptron = "nearestNeighbour($(0.0))";
// 	string perceptron = "logisticRegression($(0.0))";
// 	   string perceptron = "correlation($(0.0),$(0.0))";
 //	   string perceptron = "rayleighQuotient($(0.0))";
 	   string perceptron = "crossEntrophy($(0.0),$(0.0))";



	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, udf.first);
 	query.AddOperator(OperatorInfo::Map, {{"select", perceptron}}, schema);
	query.AddOperator(OperatorInfo::Sink, {}, schema);
	auto result =  query.Execute(udf.second, {},true, true, true);
//	result->Print();
}
