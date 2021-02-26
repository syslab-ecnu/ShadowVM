/*
 * test-query.cu
 *
 *  Created on: 2019年3月11日
 *      Author: imdb
 */





 #include "test.hpp"
#include "../com/cpp/com.hpp"




/*ObjectInfo CreateSchema(const vector<ValType> & type, const vector<size_t> & size) {
	ObjectInfo table;
	table.set_storage(HostMem);
	assert(type.size() == size.size());
	for (auto i = 0; i < type.size(); i++) {
		auto column = table.add_column_list();
		column->set_type(type[i]);
		column->set_val_size(size[i]);
	}
	return table;
}*/

pair<ObjectInfo, Object*> query_date() {
	auto date = load_date();
	ObjectInfo schema = CreateSchema({TInt32, TInt32, TString}, {4,  4,  40});
	string prediction = "$(0.4)>=1992 && $(0.4)<=1997";
	PipelineTest query("query_date");
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, date.first);
	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction}}, schema);
	query.AddOperator(OperatorInfo::Map, {{"select", "$(0.0), $(0.4)"}}, schema);
	query.AddOperator(OperatorInfo::Sink, {}, schema);
	return make_pair(schema, query.Execute(date.second, {},true, false));
}

pair<ObjectInfo, Object*> query_customer() {
	auto customer = load_customer(); customer.second->Print();

}

pair<ObjectInfo, Object*> query_part() {
	auto part = load_part();
	//part.second->Print();
	ObjectInfo schema = CreateSchema({TInt32, TString, TString}, {4,  10,  40});
	string prediction = "StrCmp($(0.3),\"MFGR#2221\",8)==0 ";
	PipelineTest query("query_part");
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, part.first);
	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction}}, schema);
	query.AddOperator(OperatorInfo::Map, {{"select", "$(0.0), $(0.2)"}}, schema);
	query.AddOperator(OperatorInfo::Sink, {}, schema);
	return make_pair(schema, query.Execute(part.second,{}, true, false));
}

pair<ObjectInfo, Object*> query_supplier() {
	auto supplier = load_supplier();
	ObjectInfo schema = CreateSchema({TInt32, TString}, {4,   40});
	string prediction = "StrCmp($(0.5),\"AMERICA\",8)==0";
	PipelineTest query("query_supplier");
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, supplier.first);
	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction}}, schema);
	query.AddOperator(OperatorInfo::Map, {{"select", "$(0.0) "}}, schema);
	query.AddOperator(OperatorInfo::Sink, {}, schema);
	return make_pair(schema, query.Execute(supplier.second,{}, true, false));
}

void test_query_SSB_Q2_1() {
 	auto lineorder = load_lineorder();
	auto date = query_date();  // date.second->Print();
	//auto part = query_part();  //  part.second->Print();
	//auto supplier = query_supplier();  // supplier.second->Print();
	//lineorder.second->To(SharedMem);
	ObjectInfo result = CreateSchema({
	ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32,
	ValType::Int32, ValType::Int32, ValType::Int32},
		{ 4, 4, 4, 4, 4, 4, 4, 4, 4});
//	string prediction = "$(0.5)>=19930000 && $(0.5)<19940000 && $(0.8)<25 && $(0.11)>=1 && $(0.11)<=3 && StrCmp($(0.16),\"AIR\", 10)==0";
	string prediction_Q1 = "$(0.5)>=19930000 && $(0.5)<19940000 && $(0.8)<25 && $(0.11)>=1 && $(0.11)<=3 && StrCmp($(0.16),\"AIR\", 10)==0";
	PipelineTest query("query");
	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, lineorder.first);
	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction_Q1}}, result);
	query.AddOperator(OperatorInfo::JoinWith, {{"lkey", "$(0.5)"}, {"rkey","$(1.0)"}}, result);
	query.AddOperator(OperatorInfo::JoinWith, {{"lkey", "$(0.3)"}, {"rkey","$(2.0)"}}, result);
//	query.AddOperator(OperatorInfo::JoinWith, {{"lkey", "$(0.4)"}, {"rkey","$(3.0)"}}, result);
   query.AddOperator(OperatorInfo::ReduceBy, {{"dim","$(1.1), $(2.1)"}, {"metric","$(0.12)/100"}}, result);
//	query.AddOperator(OperatorInfo::Map, {{"select", "$(0.12), $(1.1), $(2.1)"}}, result);
//   query.AddOperator(OperatorInfo::ReduceBy, {{"dim","$(0.1)"}, {"metric","1, $(0.12)/100"}}, result);
	//query.AddOperator(OperatorInfo::Map, {{"select", "$(0.0), $(0.1), $(0.2)"}}, result);
	query.AddOperator(OperatorInfo::Sink, {}, result);
	Object* rc = query.Execute(lineorder.second, {date},true, true, true);
//	Object* rc = query.Execute(lineorder.second, {},true, true);
//	rc->Print();


	return ;
/*
	select lo_orderdate(5), lo_quantity(7),lo_discount(12), lo_quantity(7) *lo_discount(12)
	from lineorder
	where lo_orderdate > 19930000 and lo_orderdate < 19940000
	and lo_quantity < 25;
	and lo_discount >=1 and lo_discount <= 3
 */


//	ObjectInfo result = CreateSchema({
//		ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32}, {4, 4, 4, 4});
//	string prediction = "$(0.5)>=19930000 && $(0.5)<19940000 && $(0.8)<25 && $(0.11)>=1 && $(0.11)<=3";
//	PipelineTest query("query");
//	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, SSB);
//	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction}}, result);
//	query.AddOperator(OperatorInfo::Map, {{"select", "$(0.5),$(0.8),$(0.11),$(0.8)*$(0.11)"}}, result);
//	query.AddOperator(OperatorInfo::Sink, {}, result);
//	Object* rc = query.Execute(ssb, true);
//	rc->Print();

/*
select sum(lo_extendedprice(9)*lo_discount(11))
from lineorder
where lo_orderdate(5) > 19930000 and lo_orderdate < 19940000
and lo_discount (11) between1 and 3
and lo_quantity (8) < 25
group by lo_discount;
*/
//	ObjectInfo result = CreateSchema({
//	ValType::String, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32,
//	ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32, ValType::Int32},
//		{ 10, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4});
//	string prediction = "$(0.5)>=19930000 && $(0.5)<19940000 && $(0.8)<25 && $(0.11)>=1 && $(0.11)<=3 && StrCmp($(0.16),\"AIR\", 10) == 0 ";
//	PipelineTest query("query");
//	query.AddOperator(OperatorInfo::Source, {{"type", "object"}}, lineorder.first);
//	query.AddOperator(OperatorInfo::Filter, {{"predication", prediction}}, result);
// //  query.AddOperator(OperatorInfo::ReduceBy, {{"dim","$(0.1), $(0.6), $(0.16)"}, {"metric","$(0.0),$(0.3),$(0.9)/100"}}, result);
//	query.AddOperator(OperatorInfo::Map, {{"select", "$(0.16), $(0.9)*$(0.11)"}}, result);
//	query.AddOperator(OperatorInfo::Sink, {}, result);
//	Object* rc = query.Execute(lineorder.second, true, true);
//	rc->Print();

}
