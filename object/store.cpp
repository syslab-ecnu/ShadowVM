/*
 * store.cpp
 *
 *  Created on: 2019年4月1日
 *      Author: imdb
 */

#include "store.hpp"


ObjectStore ObjStore;

ObjectStore::ObjectStore() {
	int count;
	int deviceNum = -1;
	cudaGetDeviceCount(&count);
	deviceCount = count;
	activeDevice = -1;
	if (!count)
		cout << "No CUDA device found." << endl;
	cudaDeviceProp prop;
	for (int i = 0; i < count; ++i) {
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
			deviceProps.push_back(prop);
	}
	if (count) {
		for (auto i = deviceProps.begin(); i != deviceProps.end(); ++i) {
			if (i->major >= 1) {
				deviceNum = i->major;
				break;
			}
		}
		cudaSetDevice(deviceNum);
		activeDevice = deviceNum;
	}
}

ObjectStore::~ObjectStore() {
	return;
	lock_guard<std::mutex> lck(mtx);
	for (auto i = objects.begin(); i != objects.end(); ++i) {
		if (i->second->info_.storage() == Disk)
			delete (FileObject*)(i->second);
		else
			delete (RamObject*)(i->second);
	}
}

Object* ObjectStore::Create(const ObjectInfo &info) {
	lock_guard<std::mutex> lck(mtx);
	auto iter = objects.find(info.id());
//	if (iter != objects.end()) {
//		cout << "[" << info.id() << "]" << endl;
//		assert(false);
//		return iter->second;
//	}
	Object *object = nullptr;
	switch (info.storage()) {
	case SharedMem:
 	case HostMem:
 	case DeviceMem:
		object = (Object*) new RamObject(info);	break;
	case Disk:
//		cout << "[[" << info.path()<<"," << info.id() << "]]" << endl;
		object = (Object*) new FileObject(info.path());	break;
	}
	objects[info.id()] = object;
 	assert(object != nullptr);
	return object;
}

Object* ObjectStore::Get(string id) {
	lock_guard<std::mutex> lck(mtx);
	auto iter = objects.find(id);
	if (iter == objects.end())
		return nullptr;
	return (Object*)iter->second;
}

bool ObjectStore::Destroy(string id) {
	lock_guard<std::mutex> lck(mtx);
	auto iter = objects.find(id);
	if (iter == objects.end())
		return false;
	if (iter->second->info_.storage() == Disk)
		delete (FileObject*)iter->second;
	else
		delete (RamObject*)iter->second;
	objects.erase(iter);
	return true;
}

