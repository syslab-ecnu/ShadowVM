#include "example.hpp"
void testcase() {
 	ShadowVMService service;
	service.Launch();

	ShadowVMClient client = ShadowVMClient::Create();
	//run_load_SSB(client);
	//run_SSB_Q11(client);
	//run_SSB_Q12(client);
	//run_SSB_Q13(client);
	run_SSB_Q21(client);
	//run_SSB_Q22(client);
	//run_SSB_Q23(client);
	//run_SSB_Q31(client);
	//run_SSB_Q32(client);
	//run_SSB_Q33(client);
	//run_SSB_Q34(client);
	//run_SSB_Q41(client);
	//run_SSB_Q42(client);
	//run_SSB_Q43(client);


//	cout << "######" << endl;
//	for (auto kv : service.cached_obj)
//		cout << kv.first << endl;

	service.Wait();
}
