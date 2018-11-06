#include <cudnn.h>
#include <cublas_v2.h>
#include <curand.h>
#include <helper_cuda.h>
#include <iostream>
#include <cmath>

#ifndef UTILS
#define UTILS

#define BW (16 * 16)
#define CNMEM_GRANULARITY 512

#define FatalError(s) do {                                             \
	std::stringstream _where, _message;                                \
	_where << __FILE__ << ':' << __LINE__;                             \
	_message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
	std::cerr << _message.str() << "\nAborting...\n";                  \
	cudaDeviceReset();                                                 \
	exit(1);                                                           \
} while(0)

#define checkCUDNN(expression)                               							\
{                                                            							\
	cudnnStatus_t status = (expression);                     							\
	if (status != CUDNN_STATUS_SUCCESS) {                    							\
	  std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": "      \
				<< cudnnGetErrorString(status) << std::endl; 							\
	  std::exit(EXIT_FAILURE);                               							\
	}                                                        							\
}

#define checkCUBLAS(expression)                             							\
{                                                           							\
	cublasStatus_t status = (expression);                   							\
	if (status != CUBLAS_STATUS_SUCCESS) {                  							\
	  std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": "     	\
				<< _cudaGetErrorEnum(status) << std::endl;  							\
	  std::exit(EXIT_FAILURE);                              							\
	}                                                       							\
}

#define checkCURAND(expression)                             							\
{                                                          								\
	curandStatus_t status = (expression);                   							\
	if (status != CURAND_STATUS_SUCCESS) {                  							\
	  std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": "     	\
				<< _cudaGetErrorEnum(status) << std::endl;  							\
	  std::exit(EXIT_FAILURE);                              							\
	}                                                       							\
}

#define checkCNMEM(expression)                               							\
{                                                            							\
	cnmemStatus_t status = (expression);                     							\
	if (status != CNMEM_STATUS_SUCCESS) {                    							\
	  std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": "      \
				<< cnmemGetErrorString(status) << std::endl; 							\
	  std::exit(EXIT_FAILURE);                               							\
	}                                                        							\
}

#define checkCNMEMRet(expression)                               						\
{                                                            							\
	cnmemStatus_t status = (expression);                     							\
	if (status != CNMEM_STATUS_SUCCESS) {                    							\
		if (status == CNMEM_STATUS_OUT_OF_MEMORY) {										\
			return false;																\
		}																				\
	  std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": "      \
				<< cnmemGetErrorString(status) << std::endl; 							\
	  std::exit(EXIT_FAILURE);                               							\
	}                                                      								\
}

#define checkCNMEMSim(expression, req_size, max_consume, free_bytes, action, flag)              \
{                                                            									\
	cnmemStatus_t status = (expression);                     									\
	if (status != CNMEM_STATUS_SUCCESS) {                    									\
		if (status == CNMEM_STATUS_OUT_OF_MEMORY) {												\
			flag = true;																		\
			size_t largest_free_block_size = 0;													\
			cnmemGetLargestFreeBlockSize(largest_free_block_size, NULL);						\
			max_consume = req_size - largest_free_block_size + max_consume;						\
			max_consume = (max_consume > free_bytes) ? free_bytes : max_consume;				\
			action;																				\
		}																						\
		std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": " 			\
		<< cnmemGetErrorString(status) << std::endl; 											\
			std::exit(EXIT_FAILURE);															\
	}                                                      										\
}

struct LayerDimension {
	int N, C, H, W;

	int getTotalSize();
};

template <typename T>
__global__ void fillValue(T *v, int size, int value) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;
	v[i] = value;
}

void outOfMemory();

struct CnmemSpace {
	size_t free_bytes;
	size_t initial_free_bytes;
	bool out_of_memory;

	enum Op {ADD, SUB};

	CnmemSpace(size_t free_bytes);

	void updateSpace(Op op, size_t size);

	bool isAvailable();

	size_t getConsumed();

	void updateMaxConsume(size_t &max_consume);

};

#endif