#include "user_iface.h"
#include "utils.h"
#include <limits>
#include <cnmem.h>

#ifndef LAYER_PARAMS
#define LAYER_PARAMS

enum vDNNConvAlgoPref {PREFER_MEMORY_OPTIMAL, PREFER_PERFORMANCE_OPTIMAL};

enum workspaceStatus_t {WORKSPACE_STATUS_SUCCESS, WORKSPACE_STATUS_OUT_OF_MEMORY};

#define checkWORKSPACE(expression)                               						\
{                                                            							\
	workspaceStatus_t status = (expression);                     						\
	if (status != WORKSPACE_STATUS_SUCCESS) {                    						\
	  std::cerr << "Error in file " << __FILE__ << " on line " << __LINE__ << ": "      \
				<< std::endl; 															\
	  std::exit(EXIT_FAILURE);                               							\
	}                                                        							\
}

struct ConvLayerParams {
	void *W, *b;
	void *dW, *db;
	cudnnTensorDescriptor_t input_tensor, output_tensor, bias_desc;
	cudnnFilterDescriptor_t filter_desc;
	cudnnConvolutionDescriptor_t conv_desc;
	cudnnConvolutionFwdAlgo_t fwd_algo;
	cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
	cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
	size_t fwd_workspace_size, bwd_filter_workspace_size, bwd_data_workspace_size;
	int C_in, C_out, filter_h, filter_w;
	int kernel_size;
	enum ConvDirection {FWD, BWD_FILTER, BWD_DATA};
	UpdateRule update_rule;
	cudnnDataType_t data_type;
	ActivationMode activation_mode;
	cudnnActivationDescriptor_t actv_desc;


	int fwd_req_count, fwd_ret_count;
	int bwd_filter_req_count, bwd_filter_ret_count;
	int bwd_data_req_count, bwd_data_ret_count;
	cudnnConvolutionFwdAlgoPerf_t *fwd_perf;
	cudnnConvolutionBwdFilterAlgoPerf_t *bwd_filter_perf;
	cudnnConvolutionBwdDataAlgoPerf_t *bwd_data_perf;

	void initializeValues(cudnnHandle_t cudnn_handle, ConvDescriptor *user_params, cudnnDataType_t data_type, 
							int batch_size, cudnnTensorFormat_t tensor_format, size_t data_type_size, LayerDimension &output_size, 
							UpdateRule update_rule);

	void allocateSpace(curandGenerator_t curand_gen, cudnnDataType_t data_type, size_t data_type_size, float std_dev, size_t &free_bytes, 
						bool alloc_derivative);

	size_t getWorkspaceSize(size_t &free_bytes, ConvDirection conv_direction, vDNNConvAlgo vdnn_conv_algo);
	workspaceStatus_t getWorkspaceSize(size_t &free_bytes, ConvDirection conv_direction, vDNNConvAlgoPref algo_pref, bool hard_pref, size_t &workspace_size);
	
	void cnmemAllocDerivatives(size_t data_type_size, cudaStream_t stream);
	bool cnmemAllocDerivativesCheck(size_t data_type_size, cudaStream_t stream, size_t &max_consume, size_t free_bytes, bool &out_of_memory);
	void stepParams(cublasHandle_t cublas_handle, double learning_rate);
	void cnmemFreeDerivatives(cudaStream_t stream);
};

struct FCLayerParams {
	void *W, *b;
	void *dW, *db;
	int C_in, C_out;
	int weight_matrix_size;
	UpdateRule update_rule;
	cudnnDataType_t data_type;
	ActivationMode activation_mode;
	cudnnActivationDescriptor_t actv_desc;
	cudnnTensorDescriptor_t output_tensor;

	void initializeValues(FCDescriptor *user_params, int batch_size, cudnnTensorFormat_t tensor_format, cudnnDataType_t data_type, 
							LayerDimension &output_size, UpdateRule update_rule);
	void allocateSpace(curandGenerator_t curand_gen, cudnnDataType_t data_type, size_t data_type_size, 
						float std_dev, size_t &free_bytes, bool alloc_derivative);
	
	void cnmemAllocDerivatives(size_t data_type_size, cudaStream_t stream);
	bool cnmemAllocDerivativesCheck(size_t data_type_size, cudaStream_t stream, size_t &max_consume, size_t free_bytes, bool &out_of_memory);
	void stepParams(cublasHandle_t cublas_handle, double learning_rate);
	void cnmemFreeDerivatives(cudaStream_t stream);
};

struct DropoutLayerParams {
	cudnnDropoutDescriptor_t dropout_desc;
	cudnnTensorDescriptor_t input_tensor;
	void *reserved_space;
	void *state;
	size_t reserved_space_size;
	size_t state_size;

	void initializeValues(cudnnHandle_t cudnn_handle, DropoutDescriptor *user_params, cudnnDataType_t data_type, int batch_size,
										 cudnnTensorFormat_t tensor_format, LayerDimension &output_size);

	void allocateSpace(size_t &free_bytes, cudnnHandle_t cudnn_handle, DropoutDescriptor *user_params, long long seed);
};

struct BatchNormLayerParams {
	cudnnTensorDescriptor_t input_tensor;
	cudnnTensorDescriptor_t sbmv_desc;
	void *scale, *bias;
	void *dscale, *dbias;
	void *running_mean, *running_variance;
	void *result_save_mean, *result_save_inv_var;
	double factor, epsilon;
	cudnnBatchNormMode_t mode;
	int h, w, c;
	int sbmv_size;
	UpdateRule update_rule;
	size_t allocation_size;
	cudnnDataType_t data_type;

	void initializeValues(BatchNormDescriptor *user_params, cudnnDataType_t data_type, cudnnTensorFormat_t tensor_format, 
							int batch_size, LayerDimension &output_size, UpdateRule update_rule);
	void allocateSpace(cudnnDataType_t data_type, size_t data_type_size, size_t &free_bytes, bool alloc_derivative);

	void cnmemAllocDerivatives(size_t data_type_size, cudaStream_t stream);
	bool cnmemAllocDerivativesCheck(size_t data_type_size, cudaStream_t stream, size_t &max_consume, size_t free_bytes, bool &out_of_memory);
	void stepParams(cublasHandle_t cublas_handle, double learning_rate);
	void cnmemFreeDerivatives(cudaStream_t stream);
};

struct PoolingLayerParams {
	cudnnTensorDescriptor_t input_tensor;
	cudnnTensorDescriptor_t output_tensor;

	cudnnPoolingDescriptor_t pool_desc;

	void initializeValues(PoolingDescriptor *user_params, cudnnDataType_t data_type, cudnnTensorFormat_t tensor_format, 
							int batch_size, LayerDimension &output_size);
	void allocateSpace(size_t &free_bytes);
};

struct ActivationLayerParams {
	cudnnActivationDescriptor_t actv_desc;
	cudnnTensorDescriptor_t input_tensor;

	void initializeValues(ActivationDescriptor *user_params, cudnnDataType_t data_type, cudnnTensorFormat_t tensor_format, 
							int batch_size, LayerDimension &output_size);
	void allocateSpace(size_t &free_bytes);
};

struct SoftmaxLayerParams {
	cudnnTensorDescriptor_t input_tensor;
	cudnnSoftmaxAlgorithm_t algo;
	cudnnSoftmaxMode_t mode;
	
	void initializeValues(SoftmaxDescriptor *user_params, cudnnDataType_t data_type, cudnnTensorFormat_t tensor_format, 
							int batch_size, LayerDimension &output_size);
	void allocateSpace(size_t &free_bytes);
};

#endif
