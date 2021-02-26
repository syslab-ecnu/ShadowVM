/*
 * store.hpp
 *
 *  Created on: 2019年4月1日
 *      Author: imdb
 */

#ifndef STORE_HPP_
#define STORE_HPP_

#include "../com/cpp/com.hpp"
#include "object.hpp"

class ObjectStore {
private:
	map<string, Object*> objects;
	int deviceCount;
	int activeDevice;
	vector<cudaDeviceProp> deviceProps;
	mutex mtx;
public:
	ObjectStore();
	~ObjectStore();
	Object* Create(const ObjectInfo & info);
	bool Destroy(string id);
	Object* Get(string id);
};

extern ObjectStore ObjStore;


#endif /* STORE_HPP_ */
