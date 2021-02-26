/*
 * example.hpp
 *
 */

#ifndef EXAMPLE_HPP_
#define EXAMPLE_HPP_

#include "../com/cpp/com.hpp"
#include "../service/service.hpp"
#include "../test/test.hpp"

#include "../core/core.hpp"
#include "../core/resource.hpp"
#include "../core/operator.hpp"
#include "../core/pipeline.hpp"
#include "../codegen/compiler.hpp"



void run_load_SSB(ShadowVMClient & client) ;
void run_load_lineorder(ShadowVMClient & client);
void run_load_date(ShadowVMClient & client);
void run_load_customer(ShadowVMClient & client);
void run_load_part(ShadowVMClient & client);
void run_load_supplier(ShadowVMClient & client);

void run_SSB_Q11(ShadowVMClient & client);
void run_SSB_Q12(ShadowVMClient & client);
void run_SSB_Q13(ShadowVMClient & client);

void run_SSB_Q21(ShadowVMClient & client);
void run_SSB_Q22(ShadowVMClient & client);
void run_SSB_Q23(ShadowVMClient & client);

void run_SSB_Q31(ShadowVMClient & client);
void run_SSB_Q32(ShadowVMClient & client);
void run_SSB_Q33(ShadowVMClient & client);
void run_SSB_Q34(ShadowVMClient & client);

void run_SSB_Q41(ShadowVMClient & client);
void run_SSB_Q42(ShadowVMClient & client);
void run_SSB_Q43(ShadowVMClient & client);

void testcase();

#endif /* EXAMPLE_HPP_ */
