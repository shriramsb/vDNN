#include <iostream>
#include <vector>
#include <string>

#include <cudnn.h>
#include <cublas_v2.h>
#include <curand.h>
#include <helper_cuda.h>
#include "user_iface.h"
#include "layer_params.h"
#include "utils.h"

// ---------------------- vDNN start ----------------------
#include <cnmem.h>
// ---------------------- vDNN emd ------------------------

#ifndef NEURAL_NET
#define NEURAL_NET
class NeuralNet {
public:
	void **layer_input, **dlayer_input, **params;
	int *layer_input_size;
	int *y, *pred_y;
	float *loss;
	float softmax_eps;
	void *one_vec;
	float init_std_dev;

	std::vector<LayerOp> layer_type;
	int num_layers;
	cudnnHandle_t cudnn_handle;
	cublasHandle_t cublas_handle;
	curandGenerator_t curand_gen;

	cudnnDataType_t data_type;
	size_t data_type_size;
	cudnnTensorFormat_t tensor_format;
	int batch_size;

	size_t init_free_bytes, free_bytes, total_bytes;
	size_t workspace_size;
	void *workspace;

	int input_channels, input_h, input_w;
	int num_classes;

	float *h_loss;
	int *h_pred_y;
	
	// vDNN
	vDNNType vdnn_type;
	vDNNConvAlgo vdnn_conv_algo;
	cudaStream_t stream_compute, stream_memory;

	bool pre_alloc_conv_derivative, pre_alloc_fc_derivative, pre_alloc_batch_norm_derivative;

	void **h_layer_input;
	bool *to_offload, *prefetched;

	enum OffloadType {OFFLOAD_ALL, OFFLOAD_NONE, OFFLOAD_CONV, OFFLOAD_ALTERNATE_CONV};

	NeuralNet(std::vector<LayerSpecifier> &layers, DataType data_type, int batch_size, TensorFormat tensor_format, 
				long long dropout_seed, float softmax_eps, float init_std_dev, vDNNType vdnn_type, vDNNConvAlgo vdnn_conv_algo, 
				UpdateRule update_rule);

	void getLoss(void *X, int *y, double learning_rate, std::vector<float> &fwd_vdnn_lag, std::vector<float> &bwd_vdnn_lag, bool train = true, int *correct_count = NULL, float *loss = NULL);
	void getLoss(void *X, int *y, double learning_rate, bool train = true, int *correct_count = NULL, float *loss = NULL);

	void compareOutputCorrect(int *correct_count, int *y);

	float computeLoss();

	int findPrefetchLayer(int cur_layer);

	bool simulateNeuralNetworkMemory(vDNNConvAlgoPref algo_pref, bool hard, size_t &exp_max_consume, size_t &max_consume);
	bool simulateCNMEMMemory(size_t &max_consume);
	void vDNNOptimize(size_t &exp_max_consume, size_t &max_consume);
	void setOffload(OffloadType offload_type);
	void resetPrefetched();

	// data of time
	cudaEvent_t start_compute, stop_compute;
	void getComputationTime(void *X, int *y, double learning_rate, std::vector<float> &fwd_computation_time, std::vector<float> &bwd_computation_time);
	cudaEvent_t start_transfer, stop_transfer;
	void getTransferTime(void *X, int *y, double learning_rate, std::vector<float> &fwd_transfer_time, std::vector<float> &bwd_transfer_time);
};

#endif