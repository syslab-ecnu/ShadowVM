/*
 * codegen.cpp
 *
 *  Created on: 2019年3月6日
 *      Author: imdb
 */

#include "codegen.hpp"
#include "template.hpp"
#include "../core/pipeline.hpp"

map<ValType, string> CodeGen::ToNativeType = {
		{ValType::Byte, "char"},
		{ValType::Int16, "short"},
		{ValType::Int32, "int"},
		{ValType::Int64, "long long"},
		{ValType::Float32, "float"},
		{ValType::Float64, "double"},
		{ValType::String, "char"},
		{ValType::Tensor, "char"}
};


string CodeGen::GenBulkPattern(const ObjectInfo & schema, char col_spliter, bool ignore_end) {
	string code = "";
	for (auto i = 0; i < schema.column_list_size(); i++) {
		switch(schema.column_list(i).type()) {
			case ValType::Byte: code += "%c"; break;
			case ValType::Int16: case ValType::Int32: code += "%d"; break;
			case ValType::Int64:	code += "%ld"; break;
			case ValType::Float32: code += "%f"; break;
			case ValType::Float64: code += "%lf"; break;
			case ValType::String: code += "%[^"+string(1,col_spliter)+"]"; break;
		}
		if (!(ignore_end && i == schema.column_list_size()-1))
			code += col_spliter;
	}
	return code;
}

string CodeGen::GenBulkPointerDefine(const ObjectInfo & schema, const string & bulk) {
	string code = "";
	string type = "";
	for (auto i = 0; i < schema.column_list_size(); i++) {
		switch(schema.column_list(i).type()) {
			case ValType::Byte:
			case ValType::String: type = "char *"; break;
			case ValType::Int16: type = "short *"; break;
			case ValType::Int32:	type = "int *"; break;
			case ValType::Int64: type = "long long* "; break;
			case ValType::Float32: type = "float *"; break;
			case ValType::Float64: type = "double *"; break;
		}
		code += type + GenColumnPointer(bulk,i)+" = ("+type+")"+bulk+"->column_buf_["+to_string(i)+"];\n";
	}
	return code;
}
string CodeGen::GenBulkPointerMove(const ObjectInfo & schema, const string & bulk) {
	string code = "";
 	for (auto i = 0; i < schema.column_list_size(); i++) {
 		ValType type = schema.column_list(i).type();
 		size_t size  = schema.column_list(i).val_size();
 		if (type == ValType::String || type == ValType::Tensor)
 			code += bulk+"_"+to_string(i)+"+=" + to_string(size) + ";\n";
 		else
 			code += bulk+"_"+to_string(i)+"++;\n";
 	}
	return code;
}
string CodeGen::GenBulkPointerList(const ObjectInfo & schema, const string &  bulk) {
	string code = "";
	for (auto i = 0; i < schema.column_list_size(); i++) {
		code += GenColumnPointer(bulk, i);
		if (i != schema.column_list_size() - 1)
			code += ",";
	}
	return code;
}

string CodeGen::GenFromTemplate(const string & code_template, const map<string, string> & dic) {
	string code = code_template;
	size_t pos1 = 0;
	while((pos1 = code.find("$("), pos1) != string::npos) {
		size_t pos2 = code.find(")", pos1);
		string key = code.substr(pos1+2, pos2-pos1-2);
		//cout << "key:"<<key << endl;
		string value = "";
		if (dic.find(key) != dic.end())
			value = dic.find(key)->second;
		code.replace(pos1,key.size()+3, value);

		// ToDo
		pos1+= value.size();
	}
	return code;
}



string CodeGen::GenFilterBlock(map<string, string> & variable, const Pipeline & pipe, const vector<string> & filter_exp, bool isGPU) {
	string code = "";
	if (filter_exp.size() > 0)
		code += "bool cond = ";
	else
		code += "bool cond = true";
	for (int i = 0; i < filter_exp.size(); i ++) {
 		code +=  GenExperssion(variable, pipe.input_info_, filter_exp[i]);
		if (i != filter_exp.size()-1)	code += "&&";
 	}
	code += ";\n";
	if (isGPU) {
		size_t mask_col = pipe.output_info_.column_list_size() - 1;
		string mask = "out_0_"+to_string(mask_col);
		variable[mask] = "char* "+ mask+ "=(char*)out.GetColumn(0,"+to_string(mask_col)+");\n";
		//return "*((int*)"+mask +"+i) = " + code + ";\n";
		code += "*((int*)"+mask +"+i) = cond;\n";
	}
	return code;
}

string CodeGen::GenJoinBlock(map<string, string> & variable, const Pipeline & pipe,
		const vector<KeyIndex> & lkey, const vector<KeyIndex> & rkey, bool isGPU) {
	assert(lkey.size() == rkey.size());
	string code = "";
	for (int i = 0; i < rkey.size(); i++) {
		string l_obj = to_string(lkey[i].obj_off_);
		string l_col = to_string(lkey[i].col_off_);
		string r_obj = to_string(rkey[i].obj_off_);
		string r_col = to_string(rkey[i].col_off_);
		string hash_col = to_string(pipe.input_info_[rkey[i].obj_off_].column_list_size() - 1);
		string l_input = "in_" + l_obj +"_" + l_col;
		string r_input = "in_" + r_obj +"_" + r_col;
		string r_hash = "in_" + r_obj +"_"+ hash_col;
		ValType valtype = pipe.input_info_[rkey[i].obj_off_].column_list(rkey[i].col_off_).type();
		string type = ToNativeType[valtype];
		variable[l_input] = "char* " + l_input + "=(char*)in.GetColumn("+l_obj+","+l_col+");\n";
		variable[r_input] = "char* " + r_input + "=(char*)in.GetColumn("+r_obj+","+r_col+");\n";
		variable[r_hash] = "char* " + r_hash +   "=(char*)in.GetColumn("+r_obj+","+hash_col+");\n";
		string size = to_string(pipe.input_info_[rkey[i].obj_off_].column_list(rkey[i].col_off_).val_size());
		string index = "join_index["+to_string(i+1)+"]";
		string probe = "Probe<"+type+">(("+type+"*)("+l_input+"+"+size+"*i),"+size + \
				",("+type+"*)("+r_input+"),(int*)"+r_hash+", HashRate*in.max_tuple_ct_["+r_obj+"])";
		string join = index +"="+probe+";\n";
		join += "cond = cond & ("+index+" >=0 ? true:false);\n";
		join = "if(cond) {\n" + join +"}\n";
		code += join;
 	}
	return code;
}

string CodeGen::GenSelectBlock(map<string, string> & variable, const Pipeline & pipe, const vector<string> & exp_list, bool isGPU) {
 	string code = "";
	for (int i = 0; i < exp_list.size(); i ++) {
		ValType valtype = pipe.output_info_.column_list(i).type();
		string proj_type = ToNativeType[pipe.output_info_.column_list(i).type()];
		string proj_size = to_string(pipe.output_info_.column_list(i).val_size());
		string proj_name = "out_0_" + to_string(i);
		variable[proj_name] = "char* "+ proj_name+ "=(char*)out.GetColumn(0,"+to_string(i)+");\n";
		string exec = GenExperssion(variable, pipe.input_info_, exp_list[i]);
		string move = "";
		if (isGPU)
			move = "+i*"+proj_size;
		else
			move = "+j*"+proj_size;
		if (valtype != ValType::String)
			code += "*("+proj_type+"*)(" + proj_name + move+")=" + exec + ";\n";
		else
			code += "StrCopy("+proj_name+move + "," + exec +"," + proj_size +");\n";
   	}
	code = "if(cond){\n" + code + "}\n";
	if (isGPU) {
		size_t mask_col = pipe.output_info_.column_list_size() - 1;
 		string mask = "out_0_"+to_string(mask_col);
		variable[mask] = "char* "+ mask+ "=(char*)out.GetColumn(0,"+to_string(mask_col)+");\n";
		code += "*(int*)(" + mask +"+i*4)=cond;\n";
	} else {
		code += "j+=cond;\n";
	}
	return code;
}

string CodeGen::GenReduceBlock(map<string, string> & variable, const Pipeline & pipe,
		const vector<string> & metric_exp, const vector<KeyIndex> & reduce_key, bool isGPU) {
	string code = "";
 	//cout << "reduce key num:" << reduce_key.size() << " metric num:" << metric_exp.size() << endl;
 	for (int i = 0; i< reduce_key.size();i++) {
 		string exp = "$("+to_string(reduce_key[i].obj_off_)+"."+ to_string(reduce_key[i].col_off_)+")";
 		string exec = GenExperssion(variable, pipe.input_info_, exp, true);
		string proj_type = ToNativeType[pipe.output_info_.column_list(i).type()];
		string proj_size = to_string(pipe.output_info_.column_list(i).val_size());
		string proj_name = "out_0_" + to_string(i);
		variable[proj_name] = "char* "+ proj_name+ "=(char*)out.GetColumn(0,"+to_string(i)+");\n";
		string move = "";
		if (isGPU)
			move = "+i*";
		else
			move = "+j*";
		code += "*("+proj_type+"*)(" + proj_name + move+proj_size+")=" + exec + ";\n";
 	}
 	for (int i = 0; i < metric_exp.size(); i++) {
 		ValType valtype = pipe.output_info_.column_list(reduce_key.size()+i).type();
		string proj_type = ToNativeType[valtype];
		string proj_size = to_string(pipe.output_info_.column_list(reduce_key.size()+i).val_size());
		string proj_name = "out_0_" + to_string(reduce_key.size()+i);
		variable[proj_name] = "char* "+ proj_name+ "=(char*)out.GetColumn(0,"+to_string(i+reduce_key.size())+");\n";
		string exec = GenExperssion(variable, pipe.input_info_, metric_exp[i]);
		if(!isGPU && reduce_key.size() == 0) {
 			variable[proj_name] += "*("+proj_type+"*)"+proj_name + "=0;\n";
			code += "*("+proj_type+"*)(" + proj_name +")+=cond*" + exec + ";\n";
		} else {
			string move = "";
			if (isGPU) {
 				exec = exec + "*cond";
				move = "+i*";
			} else {
				move = "+j*";
			}
			if (valtype != ValType::String)
				code += "*("+proj_type+"*)(" + proj_name + move+proj_size+")=" + exec + ";\n";
			else
				code += "StrCopy("+proj_name+move+proj_size + "," + exec +"," + proj_size +")\n;";
		}
 	}
	if(!isGPU)	{
		code = "if(cond){\n" + code + "\n}\n";
		code += "j+=cond;\n";
	}
	return code;
}

KeyIndex CodeGen::GenKeyIndex(const string & exp) {
	auto pos1 = exp.find("$(");
	auto pos2 = exp.find(".",pos1);
	auto pos3 = exp.find(")", pos2);
	auto obj_off = stoi(exp.substr(pos1 + 2, pos2 - pos1 - 2));
	auto col_off = stoi(exp.substr(pos2 + 1, pos3 - pos1 -1));
	return KeyIndex(obj_off, col_off);
}

string CodeGen::GenExperssion(map<string, string> & variable, const vector<ObjectInfo> & schema, const string & exp, bool isReduce) {
	string code = exp;
	size_t pos1 = 0;
	string value = "";
	while((pos1 = code.find("$("), pos1) != string::npos) {
//		cout << code  <<endl << "*******" << endl;
		size_t pos2 = code.find(".", pos1);
		string obj_off = code.substr(pos1 + 2, pos2 - pos1 - 2);
		size_t pos3 = code.find(")", pos2);
		assert(pos3 != string::npos);
		string col_off = code.substr(pos2 + 1, pos3 - pos2 -1);

		string input = "in_" + obj_off + "_" + col_off;
		variable[input] = "char* "+input + "=(char*)in.GetColumn("+obj_off+","+col_off+");\n";
		ValType valtype = schema[stoi(obj_off)].column_list(stoi(col_off)).type();
		string type = ToNativeType[valtype];
		string size = to_string(schema[stoi(obj_off)].column_list(stoi(col_off)).val_size());
		string pos = "";
		if ( stoi(obj_off) == 0)
			pos = input+"+i*"+size;
		else
			pos = input + "+join_index["+obj_off+"]*"+size;
		if (valtype == ValType::String) {
			if (isReduce)
				value = "Hash("+pos+","+size+")";
			else
				value = "(char*)("+pos+")";
		} else
			value = "(*("+type+"*)("+pos+"))" ;

		code.replace(pos1, pos3 - pos1 + 1, value);
		pos1 += value.size();

	//	cout << "find:" << obj_off << "," << col_off << " @ " <<  pos1 << ":" << pos3 << ":" << code.size() << endl;
	//	cout << code  <<endl << "*******" << endl;
	}
	return code;
}
string CodeGen::GenHostInclude() {
	string code = "";
	code += "#include \"define.hpp\" \n";
	code += "#include \"cpu-runtime.hpp\" \n";
	return code;
}

string CodeGen::GenVariableDefine(const map<string, string> & variable) {
	string code = "";
	for (auto & v : variable)
		code += v.second;
	return code;
}



