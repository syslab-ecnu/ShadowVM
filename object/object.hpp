/*
 * object.hpp
 *
 *  Created on: 2019年2月23日
 *      Author: imdb
 */

#ifndef OBJECT_HPP_
#define OBJECT_HPP_

#include "../com/cpp/com.hpp"

void BulkMove(Bulk * bulk, StoType src, StoType dst, CUstream stream = nullptr);
Bulk* BulkCopy(Bulk * bulk, StoType src, StoType dst);

 class Object {
 public:
	Object(){}
	virtual Bulk * NewBulk(size_t id, size_t tuple_count = MB, size_t min_capacity = KB) {assert(false);}
	virtual Bulk * GetBulk(size_t id);
	void CommitBulk(size_t offset);
	size_t NumberOfBulk() {return bulk_list_.size();}
	void Print();
	void To(StoType newType);
	Object* Copy(string newid, StoType newType);
	bool Dump(string path);
	virtual ~Object() {}
	map<int, Bulk*> bulk_list_;
	size_t total_capacity_ = 0;
	size_t tuple_count_= 0;
	vector<size_t> column_size_;
	vector<ValType> column_type_;
	Schema schema_;
	ObjectInfo info_;
	mutex mtx_;
	bool need_release_;
 };

 class RamObject: public Object {
 public:
	 RamObject(const ObjectInfo & info);
	 ~RamObject();
	 Bulk * NewBulk(size_t id, size_t max_tuple_count = CTX.tuple_per_bulk_, size_t min_capacity = KB);
  };

class FileObject: public Object{
public:
 	FileObject(string path);
	~FileObject();
 	string path_;
	vector<int> fd_list_;
};



#endif /* OBJECT_HPP_ */
