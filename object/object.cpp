/*
 * object.cpp
 *
 *  Created on: 2019年2月23日
 *      Author: imdb
 */


#include "object.hpp"

void Object::CommitBulk(size_t id) {
	 lock_guard<std::mutex> lck (mtx_);
	 tuple_count_ += bulk_list_[id]->tuple_count_;
}

Bulk * Object::GetBulk(size_t id) {
	 lock_guard<std::mutex> lck (mtx_);
	if (bulk_list_.find(id) == bulk_list_.end())
		return nullptr;
	else
		return bulk_list_[id];
}

RamObject::RamObject(const ObjectInfo & info) {
	lock_guard<std::mutex> lck (mtx_);
 	info_ = info;
	for(auto i = 0; i<info_.column_list_size();i++) {
		schema_.AddColumn(info_.column_list(i).type(), info_.column_list(i).val_size());
		column_type_.push_back(info_.column_list(i).type());
		column_size_.push_back(info_.column_list(i).val_size());
	}
	cout << "[new ramobject] column num:" << column_size_.size() << endl;
}

RamObject::~RamObject() {
	lock_guard<std::mutex> lck (mtx_);
	if (!need_release_)
		return;
	for (auto i = 0; i < bulk_list_.size(); i++) {
		switch (info_.storage()) {
		case SharedMem:
			cudaHostUnregister(bulk_list_[i]->buffer_);
		case HostMem:
			free(bulk_list_[i]->buffer_);
			break;
		case DeviceMem:
			cudaFree(bulk_list_[i]->buffer_);
			break;
		default:
			break;
		}
	}
}

Bulk * RamObject::NewBulk(size_t id, size_t max_tuple_count, size_t min_capacity) {
	lock_guard<std::mutex> lck (mtx_);
	Bulk * bulk = new Bulk();
		bulk->schema_ = schema_;
		bulk->id_ = id;
	 	bulk->max_tuple_count_ = max_tuple_count;
		for (auto i = 0; i < schema_.GetColumnCount(); i++) {
 			bulk->SetColumnPos(i, bulk->capacity_);
 			assert(bulk->column_pos_[i] % (PageSize)==0);
			bulk->capacity_ += GetPageSizeAlign(max_tuple_count * schema_.GetColumnSize(i));
	   	}
		if (bulk->capacity_ < min_capacity)
			bulk->capacity_ = min_capacity;
		void *mem = nullptr;
		switch (info_.storage()) {
		case HostMem:
			mem = memalign(4*KB, bulk->capacity_);
			break;
		case SharedMem:
			mem = memalign(4*KB, bulk->capacity_);
			checkCudaErrors(cudaHostRegister(mem, bulk->capacity_, cudaHostAllocMapped));
			break;
		case DeviceMem:
			checkCudaErrors(cudaMalloc((void**)&mem, bulk->capacity_));
	//		cout << "[malloc] device mem " << mem  << endl;

			break;
		default:
			throw string("Cannot allocate new bulk to RamObject casted from FileObject.");
			break;
		}
		bulk->Init(mem, bulk->capacity_);
		bulk_list_[id] = bulk;
		total_capacity_ +=  bulk->capacity_;
//		if (info_.storage() == DeviceMem) {
//			for (auto i = 0 ; i < bulk->schema_.GetColumnCount();i++) {
//				cout << bulk->column_pos_[i] << ":" << bulk->column_buf_[i] <<  endl;
//			}
//		}
	  	return bulk;


//
//
//	Bulk * bulk = new Bulk();
//	bulk->schema_ = schema_;
//	bulk->id_ =  id; // bulk_list_.size();
// 	bulk->max_tuple_count_ = max_tuple_count;
//	for (auto i = 0; i < schema_.GetColumnCount(); i++) {
//	 //	bulk->column_pos_.push_back(bulk->capacity_);
//		bulk->SetColumnPos(i, bulk->capacity_);
//		bulk->capacity_ += GetPageSizeAlign(max_tuple_count * schema_.GetColumnSize(i));
//   	}
//	bulk->Init(memalign(4*KB, bulk->capacity_), bulk->capacity_);
///*	bulk->buffer_ = memalign(4*KB, bulk->capacity_);
//	for (auto i = 0; i < schema_.GetColumnCount(); i++) {
//	//	bulk->column_buf_.push_back(bulk->buffer_ + bulk->column_pos_[i] );
//		bulk->column_buf_[i] = bulk->buffer_ + bulk->column_pos_[i];
// 	}*/
//	bulk_list_[bulk->id_] = bulk;
//	total_capacity_ +=  bulk->capacity_;
//  	return bulk;
}




 FileObject::FileObject(string path) {
	lock_guard<std::mutex> lck (mtx_);
	path_ = path;
	struct stat s_buf;
	stat(path_.c_str(),&s_buf);
	if(S_ISDIR(s_buf.st_mode)) {
		DIR * dir = opendir(path_.c_str());
		assert(dir!=nullptr);
		struct dirent * item;
		while((item = readdir(dir)) != NULL) {
			string name = item->d_name;
			if (*(name.c_str()) == '.') continue;
			if (*(path_.c_str() + path_.length()-1)!='/') name='/'+name;
			name = path_ + name;
			int fd = open(name.c_str(), O_RDONLY);
			assert(fd>0);
			struct stat st;
			fstat(fd, &st);
			size_t size = GetPageSizeAlign(st.st_size);
			void * buf = mmap(0, size, PROT_READ, MAP_SHARED|MAP_POPULATE|MAP_NONBLOCK, fd, 0);
			Bulk * bulk = new Bulk();
			bulk->id_ = bulk_list_.size();
			bulk->capacity_ = size;
			bulk->buffer_ = buf;
			bulk_list_[bulk->id_] = bulk;
			fd_list_.push_back(fd);
			total_capacity_ += size;
			cout << "open file:" << name <<":" << size/1024 << "KB" << endl;
		}
	} else {
		string name = path_;
		int fd = open(name.c_str(), O_RDONLY);
		assert(fd>0);
		struct stat st;
		fstat(fd, &st);
		size_t size = GetPageSizeAlign(st.st_size);
		void * buf = mmap(0, size, PROT_READ, MAP_SHARED|MAP_POPULATE|MAP_NONBLOCK, fd, 0);
		Bulk * bulk = new Bulk();
		bulk->id_ = bulk_list_.size();
		bulk->capacity_ = size;
		bulk->buffer_ = buf;
		bulk_list_[bulk->id_] = bulk;
		fd_list_.push_back(fd);
		total_capacity_ += size;
		cout << "open file:" << name <<":" << size/1024 << "KB" << endl;

	}
 }
 FileObject::~FileObject() {
	for (auto i = 0; i < bulk_list_.size(); i++) {
		munmap(bulk_list_[i]->buffer_, bulk_list_[i]->capacity_);
		close(fd_list_[i]);
	}
 }

void Object::Print() {
	for (auto i = 0; i <= 2 && i < NumberOfBulk(); i++) {
		Bulk * bulk = GetBulk(i);
		cout << "bulk "<<bulk->id_ << " has " << bulk->tuple_count_ << " tuples"<< endl;
	//	cout << "address:" << bulk->column_buf_ << endl;
			//	for (auto row = bulk->tuple_count_ -1; row > bulk->tuple_count_ - 5; row--) {
		for (auto row = 0 ; row < 5; row++) {
			for (auto c = 0; c < bulk->schema_.GetColumnCount(); c++) {
				cout << GetStringOfItem(bulk->column_buf_[c]+row* column_size_[c],
						column_type_[c], column_size_[c]) << "|";
			}
			cout << endl;
			}
		}
}


void Object::To(StoType newType) {
	int i;
	void *mem;
	lock_guard<std::mutex> lck (mtx_);
	switch (info_.storage()) {
	case HostMem:
		switch (newType) {
		case SharedMem:
			info_.set_storage(SharedMem);
		//	type = SharedMem;
			for (i = 0; i < bulk_list_.size(); ++i) {
				cudaHostRegister(bulk_list_[i]->buffer_,
					 bulk_list_[i]->capacity_, cudaHostAllocMapped);
				bulk_list_[i]->Init(bulk_list_[i]->buffer_, bulk_list_[i]->capacity_);
 			}
			break;
		case DeviceMem:
			info_.set_storage(DeviceMem);
		//	type = DeviceMem;
			for (i = 0; i < bulk_list_.size(); ++i) {
				mem = nullptr;
				checkCudaErrors(cudaMalloc((void**)&mem, bulk_list_[i]->capacity_));
				assert(mem != nullptr);
				checkCudaErrors(cudaMemcpy(mem, bulk_list_[i]->buffer_,
							bulk_list_[i]->capacity_, cudaMemcpyHostToDevice));
				free(bulk_list_[i]->buffer_);
				bulk_list_[i]->buffer_ = mem;
				bulk_list_[i]->Init(bulk_list_[i]->buffer_, bulk_list_[i]->capacity_);
			}
			break;
		default:
			break;
		}
		break;
	case SharedMem:
		switch (newType) {
		case HostMem:
			info_.set_storage(HostMem);
		//	type = HostMem;
			for (i = 0; i < bulk_list_.size(); ++i)
				cudaHostUnregister(bulk_list_[i]->buffer_);
			break;
		case DeviceMem:
			info_.set_storage(DeviceMem);
			//type = DeviceMem;
			for (i = 0; i < bulk_list_.size(); ++i) {
				mem = nullptr;
				cudaHostUnregister(bulk_list_[i]->buffer_);
				cudaMalloc((void**)&mem, bulk_list_[i]->capacity_);
				assert (mem != nullptr);
				cudaMemcpy(mem, bulk_list_[i]->buffer_,
							bulk_list_[i]->capacity_, cudaMemcpyHostToDevice);
				free(bulk_list_[i]->buffer_);
				bulk_list_[i]->buffer_ = mem;
				bulk_list_[i]->Init(bulk_list_[i]->buffer_, bulk_list_[i]->capacity_);
			}
			break;
		default:
			break;
		}
		break;
	case DeviceMem:
		switch (newType) {
		case HostMem:
			info_.set_storage(HostMem);
			//type = HostMem;
			for (i = 0; i < bulk_list_.size(); ++i) {
				mem = nullptr;
				mem = memalign(4*KB, bulk_list_[i]->capacity_);
				assert(mem != nullptr);
				checkCudaErrors(cudaMemcpy(mem, bulk_list_[i]->buffer_,  bulk_list_[i]->capacity_,
							cudaMemcpyDeviceToHost));
			//	cout << "TO： host Mem " << i << endl;

				cudaFree(bulk_list_[i]->buffer_);
				bulk_list_[i]->buffer_ = mem;
				bulk_list_[i]->Init(bulk_list_[i]->buffer_, bulk_list_[i]->capacity_);
			}
			break;
		case SharedMem:
			info_.set_storage(SharedMem);
		//	type = SharedMem;
			for (i = 0; i < bulk_list_.size(); ++i) {
				mem = nullptr;
				mem = memalign(4*KB, bulk_list_[i]->capacity_);
				assert(mem != nullptr);
				cudaMemcpy(mem, bulk_list_[i]->buffer_, bulk_list_[i]->capacity_,
							cudaMemcpyDeviceToHost);
				cudaFree(bulk_list_[i]->buffer_);
				bulk_list_[i]->buffer_ = mem;
				bulk_list_[i]->Init(bulk_list_[i]->buffer_, bulk_list_[i]->capacity_);
			}
			break;
		default:
			break;
		}
		break;
	default:
		break;
	}
}

bool Object::Dump(string path) {
	char *mem;
	bool ret = false;
	if (path[path.length() - 1] != '/')
		path = path + "/";
	lock_guard<std::mutex> lck (mtx_);
	for (int i = 0; i < bulk_list_.size(); ++i) {
		if (info_.storage() == DeviceMem) {
			mem = new char[bulk_list_[i]->capacity_];
			cudaMemcpy(mem, bulk_list_[i]->buffer_,
						bulk_list_[i]->capacity_, cudaMemcpyDeviceToHost);
		}
		else
			mem = (char*)bulk_list_[i]->buffer_;
		stringstream ss;
		string str;
		ss << i;
		ss >> str;
		string name = path + str + ".osd";
		FILE *fp = fopen(name.c_str(), "wb");
		assert(fp!=nullptr);
		if (fp) {
			fwrite(mem, sizeof(char), bulk_list_[i]->capacity_, fp);
			fclose(fp);
			ret = true;
		}
		if (info_.storage() == DeviceMem)
			delete[] mem;
	}
	return ret;
}

Object* Object::Copy(string newid, StoType newType) {
	RamObject *obj = new RamObject(info_);
	lock_guard<std::mutex> lck (mtx_);
	obj->total_capacity_ = total_capacity_;
	obj->tuple_count_ = tuple_count_;
	obj->need_release_ = true;
//	obj->column_type_.assign(column_type_.begin(), column_type_.end());
//	obj->column_size_.assign(column_size_.begin(), column_size_.end());
//	obj->schema_ = schema_;
//	obj->info_ = info_;
 	obj->info_.set_id(newid);
	for (auto bulk_itr:bulk_list_) {
	//	Bulk *bulk = bulk_list_[i]->copy(type, newType);
		Bulk * bulk = BulkCopy(bulk_itr.second ,info_.storage(), newType);
		obj->bulk_list_[bulk_itr.first] = bulk;
	}
	//cout << "[copy ramobject]" << endl;
	return obj;
}


void BulkMove(Bulk * bulk, StoType src, StoType dst, CUstream stream) {
	void *mem = nullptr;
		switch (src) {
		case HostMem:
			switch (dst) {
			case SharedMem:
				cudaHostRegister(bulk->buffer_, bulk->capacity_, cudaHostAllocMapped);
				bulk->Init(bulk->buffer_,bulk-> capacity_);
				break;
			case DeviceMem:
				cudaMalloc((void**)&mem, bulk->capacity_);
				assert(mem != nullptr);
				if(stream == nullptr)
					cudaMemcpy(mem, bulk->buffer_, bulk->capacity_, cudaMemcpyHostToDevice);
				else
					assert(cuMemcpyHtoDAsync((CUdeviceptr)mem, bulk->buffer_, bulk->capacity_,stream) == CUDA_SUCCESS);;
				free(bulk->buffer_);
				bulk->buffer_ = mem;
				bulk->Init(bulk->buffer_, bulk->capacity_);
				break;
			default:
				break;
			}
			break;
		case SharedMem:
			switch (dst) {
			case HostMem:
				cudaHostUnregister(bulk->buffer_);
				break;
			case DeviceMem:
				cudaHostUnregister(bulk->buffer_);
				cudaMalloc((void**)&mem, bulk->capacity_);
				assert (mem != nullptr);
				if(stream != nullptr)
					cudaMemcpy(mem, bulk->buffer_, bulk->capacity_, cudaMemcpyHostToDevice);
				else
					assert(cuMemcpyHtoDAsync((CUdeviceptr)mem, bulk->buffer_, bulk->capacity_,stream) == CUDA_SUCCESS);;
				free(bulk->buffer_);
				bulk->buffer_ = mem;
				bulk->Init(bulk->buffer_, bulk->capacity_);
				break;
			default:
				break;
			}
			break;
		case DeviceMem:
			switch (dst) {
			case HostMem:
				mem = memalign(4*KB, bulk->capacity_);
				assert(mem != nullptr);
				if (stream != nullptr )
					cudaMemcpy(mem, bulk->buffer_, bulk->capacity_, cudaMemcpyDeviceToHost);
				else
					assert(cuMemcpyDtoHAsync(bulk->buffer_, (CUdeviceptr)mem, bulk->capacity_,stream) == CUDA_SUCCESS);
				cudaFree(bulk->buffer_);
				bulk->buffer_ = mem;
				bulk->Init(bulk->buffer_, bulk->capacity_);
				break;
			case SharedMem:
				mem = memalign(4*KB, bulk->capacity_);
				assert(mem != nullptr);
				if (stream != nullptr)
					cudaMemcpy(mem, bulk->buffer_, bulk->capacity_, cudaMemcpyDeviceToHost);
				else
					assert(cuMemcpyDtoHAsync(bulk->buffer_, (CUdeviceptr)mem, bulk->capacity_,stream) == CUDA_SUCCESS);;
				cudaFree(bulk->buffer_);
				bulk->buffer_ = mem;
				bulk->Init(bulk->buffer_, bulk->capacity_);
				break;
			default:
				break;
			}
			break;
		default:
			break;
		}
}
Bulk* BulkCopy(Bulk * src_bulk, StoType src, StoType dst) {
	void *mem = nullptr;
	Bulk *bulk = new Bulk();
	bulk->id_ = src_bulk->id_;
	bulk->capacity_ = src_bulk->capacity_;
	bulk->tuple_count_= src_bulk->tuple_count_;
	bulk->max_tuple_count_ = src_bulk->max_tuple_count_;
	memcpy(bulk->column_pos_, src_bulk->column_pos_, MaxColumnNum * sizeof(size_t));
	bulk->schema_ = src_bulk->schema_;
	switch (src) {
	case HostMem:
		switch (dst) {
		case HostMem:
			mem = memalign(4*KB, src_bulk->capacity_);
			memcpy(mem, src_bulk->buffer_, src_bulk->capacity_);
 			bulk->Init(mem, src_bulk->capacity_);
//			cout << "copy host memory" << endl;
			break;
		case SharedMem:
			mem = memalign(4*KB, src_bulk->capacity_);
 			cudaHostRegister(mem, src_bulk->capacity_, cudaHostAllocMapped);
			memcpy(mem, src_bulk->buffer_, src_bulk->capacity_);
			bulk->Init(mem, src_bulk->capacity_);
			break;
		case DeviceMem:
			cudaMalloc((void**)&mem, src_bulk->capacity_);
			assert(mem != nullptr);
 			cudaMemcpy(mem, src_bulk->buffer_, src_bulk->capacity_, cudaMemcpyHostToDevice);
			bulk->Init(mem, src_bulk->capacity_);
			break;
		default:
			break;
		}
		break;
	case SharedMem:
		switch (dst) {
		case HostMem:
			mem = memalign(4*KB, src_bulk->capacity_);
			memcpy(mem, src_bulk->buffer_, src_bulk->capacity_);
			bulk->Init(mem, src_bulk->capacity_);
			break;
		case DeviceMem:
			cudaMalloc((void**)&mem, src_bulk->capacity_);
			assert (mem != nullptr);
			cudaMemcpy(mem, src_bulk->buffer_, src_bulk->capacity_, cudaMemcpyHostToDevice);
			bulk->Init(mem, src_bulk->capacity_);
			break;
		default:
			break;
		}
		break;
	case DeviceMem:
		mem = memalign(4*KB, src_bulk->capacity_);
		switch (dst) {
		case SharedMem:
			cudaHostRegister(mem, src_bulk->capacity_, cudaHostAllocMapped);
		case HostMem:
			assert(mem != nullptr);
			cudaMemcpy(mem, src_bulk->buffer_, src_bulk->capacity_, cudaMemcpyDeviceToHost);
			bulk->Init(mem, src_bulk->capacity_);
			break;
		default:
			break;
		}
		break;
	default:
		break;
	}
	return bulk;
}




