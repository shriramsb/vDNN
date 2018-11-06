#include "neural_net.h"
#include <time.h>
#include <cstdio>
#include <string>

template <typename T>
__global__ void softmaxLossBackProp(int *y, T *SO, T *dSO, int batch_size, int output_size, float eps) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= batch_size)
		return;
	int cur_class = static_cast<int>(y[i]);
	dSO[i * output_size + cur_class] = -1 / (SO[i * output_size + cur_class] * batch_size + eps);
}

template <typename T>
__global__ void computeSoftmaxLoss(T *O, int *y, float *loss, int batch_size, int num_classes, float eps) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= batch_size)
		return;

	loss[i] = -logf(O[i * num_classes + y[i]] + eps);
}

template <typename T>
__global__ void inferClass(T *O, int *pred_y, int batch_size, int num_classes) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= batch_size)
		return;

	T max = O[i * num_classes];
	int index = 0;
	for (int j = 1; j < num_classes; j++) {
		if (O[i * num_classes + j] > max) {
			max = O[i * num_classes + j];
			index = j;
		}
	}
	pred_y[i] = index;
}



float NeuralNet::computeLoss() {
	if (layer_type[num_layers - 1] == SOFTMAX) {
		if (data_type == CUDNN_DATA_FLOAT)
			computeSoftmaxLoss<float><<<ceil(1.0 * batch_size / BW), BW>>>((float *)layer_input[num_layers], this->y, loss, batch_size, num_classes, softmax_eps);
		else if (data_type == CUDNN_DATA_DOUBLE)
			computeSoftmaxLoss<double><<<ceil(1.0 * batch_size / BW), BW>>>((double *)layer_input[num_layers], this->y, loss, batch_size, num_classes, softmax_eps);
	}
	checkCudaErrors(cudaMemcpy(h_loss, loss, batch_size * sizeof(float), cudaMemcpyDeviceToHost));
	float total_loss = 0.0;
	for (int i = 0; i < batch_size; i++)
		total_loss += h_loss[i];
	return total_loss / batch_size;
}

void NeuralNet::compareOutputCorrect(int *correct_count, int *y) {
	*correct_count = 0;

	if (data_type == CUDNN_DATA_FLOAT) {
		float *typecast_O = (float *)layer_input[num_layers - 1];
		inferClass<float><<<ceil(1.0 * batch_size / BW), BW>>>(typecast_O, pred_y, batch_size, num_classes);
		checkCudaErrors(cudaMemcpy(h_pred_y, pred_y, batch_size * sizeof(int), cudaMemcpyDeviceToHost));
		for (int i = 0; i < batch_size; i++) {
			if (h_pred_y[i] == y[i])
				*correct_count = *correct_count + 1;
		}
	}
	else if (data_type == CUDNN_DATA_DOUBLE) {
		double *typecast_O = (double *)layer_input[num_layers - 1];
		inferClass<double><<<ceil(1.0 * batch_size / BW), BW>>>(typecast_O, pred_y, batch_size, num_classes);
		checkCudaErrors(cudaMemcpy(h_pred_y, pred_y, batch_size * sizeof(int), cudaMemcpyDeviceToHost));
		for (int i = 0; i < batch_size; i++) {
			if (h_pred_y[i] == y[i])
				*correct_count = *correct_count + 1;
		}
	}
}




NeuralNet::NeuralNet(std::vector<LayerSpecifier> &layers, DataType data_type, int batch_size, TensorFormat tensor_format, 
						long long dropout_seed, float softmax_eps, float init_std_dev, vDNNType vdnn_type, vDNNConvAlgo vdnn_conv_algo, 
						UpdateRule update_rule) {
	
	// ---------------------- vDNN start ----------------------
	checkCudaErrors(cudaStreamCreate(&stream_compute));
	checkCudaErrors(cudaStreamCreate(&stream_memory));
	this->vdnn_type = vdnn_type;
	this->vdnn_conv_algo = vdnn_conv_algo;
	// ---------------------- vDNN end ------------------------

	// create handle
	checkCUDNN(cudnnCreate(&cudnn_handle));
	checkCUDNN(cudnnSetStream(cudnn_handle, stream_compute));

	checkCUBLAS(cublasCreate(&cublas_handle));
	checkCUBLAS(cublasSetStream(cublas_handle, stream_compute));

	checkCURAND(curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT));
	checkCURAND(curandSetStream(curand_gen, stream_compute));

	checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));
	init_free_bytes = free_bytes;
	std::cout << "Free bytes at start: " << free_bytes << std::endl;

	pre_alloc_conv_derivative = false;
	pre_alloc_fc_derivative = false;
	pre_alloc_batch_norm_derivative = true;

	if (vdnn_type == vDNN_NONE) {
		pre_alloc_conv_derivative = true;
		pre_alloc_fc_derivative = true;
		pre_alloc_batch_norm_derivative = true;
	}

	if (data_type == DATA_FLOAT) {
		this->data_type = CUDNN_DATA_FLOAT;
		data_type_size = sizeof(float);
	}

	else if (data_type == DATA_DOUBLE) {
		this->data_type = CUDNN_DATA_DOUBLE;
		data_type_size = sizeof(double);
	}

	if (tensor_format == TENSOR_NCHW)
		this->tensor_format = CUDNN_TENSOR_NCHW;
	else if (tensor_format == TENSOR_NHWC)
		this->tensor_format = CUDNN_TENSOR_NHWC;

	this->batch_size = batch_size;
	this->softmax_eps = softmax_eps;
	this->init_std_dev = init_std_dev;

	num_layers = layers.size();
	// allocation of space for input to each layer
	layer_input = (void **)malloc((num_layers + 1) * sizeof(void *));
	layer_input_size = (int *)malloc((num_layers + 1) * sizeof(int));
	dlayer_input = (void **)malloc((num_layers + 1) * sizeof(void *));
	params = (void **)malloc(num_layers * sizeof(void *));

	LayerDimension prev_output_size;
	LayerDimension current_output_size;
	for (int i = 0; i < num_layers; i++) {
		layer_type.push_back(layers[i].type);
		if (layers[i].type == CONV) {
			ConvDescriptor *user_params = (ConvDescriptor *)layers[i].params;
			params[i] = malloc(sizeof(ConvLayerParams));
			((ConvLayerParams *)params[i])->initializeValues(cudnn_handle, user_params, this->data_type, batch_size, this->tensor_format, 
																data_type_size, current_output_size, update_rule);
		}
		else if (layers[i].type == FULLY_CONNECTED) {
			FCDescriptor *user_params = (FCDescriptor *)layers[i].params;
			params[i] = malloc(sizeof(FCLayerParams));
			((FCLayerParams *)params[i])->initializeValues(user_params, batch_size, this->tensor_format, this->data_type, 
															current_output_size, update_rule);
		}
		else if (layers[i].type == DROPOUT) {
			DropoutDescriptor *user_params = (DropoutDescriptor *)layers[i].params;
			params[i] = malloc(sizeof(DropoutLayerParams));
			((DropoutLayerParams *)params[i])->initializeValues(cudnn_handle, user_params, this->data_type, batch_size, 
																this->tensor_format, current_output_size);
			
		}

		else if (layers[i].type == BATCHNORM) {
			BatchNormDescriptor *user_params = (BatchNormDescriptor *)layers[i].params;
			params[i] = malloc(sizeof(BatchNormLayerParams));
			((BatchNormLayerParams *)params[i])->initializeValues(user_params, this->data_type, this->tensor_format, batch_size, 
																	current_output_size, update_rule);
			
		}
		else if (layers[i].type == POOLING) {
			PoolingDescriptor *user_params = (PoolingDescriptor *)layers[i].params;
			params[i] = malloc(sizeof(BatchNormLayerParams));

			((PoolingLayerParams *)params[i])->initializeValues(user_params, this->data_type, this->tensor_format, 
																batch_size, current_output_size);
		}

		else if (layers[i].type == ACTV) {
			ActivationDescriptor *user_params = (ActivationDescriptor *)layers[i].params;
			params[i] = malloc(sizeof(ActivationLayerParams));
			((ActivationLayerParams *)params[i])->initializeValues(user_params, this->data_type, this->tensor_format, 
																	batch_size, current_output_size);
		}

		else if (layers[i].type == SOFTMAX) {
			SoftmaxDescriptor *user_params = (SoftmaxDescriptor *)layers[i].params;
			params[i] = malloc(sizeof(SoftmaxLayerParams));
			((SoftmaxLayerParams *)params[i])->initializeValues(user_params, this->data_type, this->tensor_format, 
																batch_size, current_output_size);
			// std::cout << current_output_size.N << ' ' << current_output_size.C << current_output_size.H << current_output_size.W << std::endl;
		}
		if (i == 0) {
			prev_output_size = current_output_size;
		}
		// incomplete - have to check flatten and check exact dimension
		// else if (current_output_size.getTotalSize() != prev_output_size.getTotalSize()) {
		// 	std::cout << "Layer " << i << " output and next layer's input size mismatch\n";
		// 	exit(0);
		// }
	}

	// ---------------------- vDNN start ----------------------

	// allocate space in host memory for layers to be transferred
	h_layer_input = (void **)malloc(num_layers * sizeof(void *));
	to_offload = (bool *)malloc(num_layers * sizeof(bool));
	prefetched = (bool *)malloc(num_layers * sizeof(bool));

	// ---------------------- vDNN end ------------------------
	checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));
	std::cout << "Free bytes just before allocate space: " << free_bytes << std::endl;
	// allocate space for parameters
	// Exception BatchNorm - looks like it will take lots of space if only FC layers - space taken = size of one input
	for (int i = 0; i < num_layers; i++) {
		size_t input_size;
		if (layers[i].type == CONV) {
			ConvDescriptor *user_params = (ConvDescriptor *)layers[i].params;
			((ConvLayerParams *)params[i])->allocateSpace(curand_gen, this->data_type, data_type_size, init_std_dev, 
															free_bytes, pre_alloc_conv_derivative);

			input_size = batch_size * user_params->input_channels * user_params->input_h * user_params->input_w;
			if (i == 0) {
				input_channels = user_params->input_channels;
				input_h = user_params->input_h;
				input_w = user_params->input_w;
			}
		}
		else if (layers[i].type == FULLY_CONNECTED) {
			FCDescriptor *user_params = (FCDescriptor *)layers[i].params;
			((FCLayerParams *)params[i])->allocateSpace(curand_gen, this->data_type, data_type_size, init_std_dev, 
														free_bytes, pre_alloc_fc_derivative);
			input_size = batch_size * user_params->input_channels;
			if (i == 0) {
				input_channels = user_params->input_channels;
				input_h = 1;
				input_w = 1;
			}
		}
		else if (layers[i].type == DROPOUT) {
			DropoutDescriptor *user_params = (DropoutDescriptor *)layers[i].params;
			((DropoutLayerParams *)params[i])->allocateSpace(free_bytes, cudnn_handle, user_params, dropout_seed);
			input_size = batch_size * user_params->channels * user_params->h * user_params->w;
			if (i == 0) {
				input_channels = user_params->channels;
				input_h = user_params->h;
				input_w = user_params->w;
			}
		}
		else if (layers[i].type == BATCHNORM) {
			BatchNormDescriptor *user_params = (BatchNormDescriptor *)layers[i].params;
			((BatchNormLayerParams *)params[i])->allocateSpace(this->data_type, data_type_size, 
																free_bytes, pre_alloc_batch_norm_derivative);
			input_size = batch_size * user_params->channels * user_params->h * user_params->w;
			if (i == 0) {
				input_channels = user_params->channels;
				input_h = user_params->h;
				input_w = user_params->w;
			}
		}
		else if (layers[i].type == POOLING) {
			PoolingDescriptor *user_params = (PoolingDescriptor *)layers[i].params;
			((PoolingLayerParams *)params[i])->allocateSpace(free_bytes);
			input_size = batch_size * user_params->input_channels * user_params->input_h * user_params->input_w;
			if (i == 0) {
				input_channels = user_params->input_channels;
				input_h = user_params->input_h;
				input_w = user_params->input_w;
			}
		}
		else if (layers[i].type == ACTV) {
			ActivationDescriptor *user_params = (ActivationDescriptor *)layers[i].params;
			((ActivationLayerParams *)params[i])->allocateSpace(free_bytes);
			input_size = batch_size * user_params->channels * user_params->h * user_params->w;
			if (i == 0) {
				input_channels = user_params->channels;
				input_h = user_params->h;
				input_w = user_params->w;
			}
		}
		else if (layers[i].type == SOFTMAX) {
			SoftmaxDescriptor *user_params = (SoftmaxDescriptor *)layers[i].params;
			((SoftmaxLayerParams *)params[i])->allocateSpace(free_bytes);
			input_size = batch_size * user_params->channels * user_params->h * user_params->w;

			// assuming this is last layer, allocate for next layer as well
			// checkCudaErrors(cudaMalloc(&layer_input[i + 1], input_size * data_type_size));
			// checkCudaErrors(cudaMalloc(&dlayer_input[i + 1], input_size * data_type_size));
			layer_input_size[i + 1] = input_size;
			if (i == 0) {
				input_channels = user_params->channels;
				input_h = user_params->h;
				input_w = user_params->w;
			}
			if (i == num_layers - 1) {
				num_classes = user_params->channels;
			}
		}

		// do not allocate memory initially
		// checkCudaErrors(cudaMalloc(&layer_input[i], input_size * data_type_size));
		// checkCudaErrors(cudaMalloc(&dlayer_input[i], input_size * data_type_size));
		
		// ---------------------- vDNN start ----------------------
		layer_input_size[i] = input_size;
		// ---------------------- vDNN end ------------------------
	}
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));
	std::cout << "Free bytes just after allocate space: " << free_bytes << std::endl;
	// very small - could be allocated initially itself
	checkCudaErrors(cudaMalloc((void **)&y, batch_size * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&pred_y, batch_size * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&loss, batch_size * sizeof(float)));
	checkCudaErrors(cudaMalloc(&one_vec, batch_size * data_type_size));

	if (this->data_type == CUDNN_DATA_FLOAT)
		fillValue<float><<<ceil(1.0 * batch_size / BW), BW>>>((float *)one_vec, batch_size, 1);
	else
		fillValue<double><<<ceil(1.0 * batch_size / BW), BW>>>((double *)one_vec, batch_size, 1);
	
	checkCudaErrors(cudaMallocHost((void **)&h_loss, batch_size * sizeof(float)));
	checkCudaErrors(cudaMallocHost((void **)&h_pred_y, batch_size * sizeof(int)));
	
	// do not allocate workspace initially
	// allocate space for workspace and also keep track of algo
	// size_t cur_workspace_size;
	// workspace_size = 0;
	// for (int i = 0; i < num_layers; i++) {
	// 	if (layers[i].type == CONV) {
	// 		((ConvLayerParams *)params[i])->getWorkspaceSize(cur_workspace_size, free_bytes);
	// 		if (cur_workspace_size > workspace_size)
	// 			workspace_size = cur_workspace_size;
	// 	}
	// }

	// checkCudaErrors(cudaMalloc(&workspace, workspace_size));
	// free_bytes = free_bytes - workspace_size;
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));

	// leave 600 MB and use the rest
	std::cout << "Free bytes: " << free_bytes << std::endl;
	free_bytes -= 1024 * 1024 * 600;
	// ---------------------- vDNN start ----------------------
	size_t exp_max_consume, max_consume;
	vDNNOptimize(exp_max_consume, max_consume);
	std::cout << "actual_max_consume: " << max_consume << std::endl;
	std::cout << "exp_max_consume: " << exp_max_consume << std::endl;
	std::cout << "diff_max_consume(MB): " << (max_consume - exp_max_consume) / (1.0 * 1024 * 1024) << std::endl;
	std::cout << "exp_free_bytes(MB): " << (free_bytes + 1024 * 1024 * 600 - exp_max_consume) / (1.0 * 1024 * 1024) << std::endl;
	std::cout << "exp_total_consume(MB): " << (init_free_bytes - (free_bytes + 600 * 1024 * 1024 - exp_max_consume)) / (1.0 * 1024 * 1024) << std::endl;
	std::cout << "actual_total_consume(MB): " << (init_free_bytes - (free_bytes + 600 * 1024 * 1024 - max_consume)) / (1.0 * 1024 * 1024) << std::endl;

	// ---------------------- vDNN end ------------------------


	// ---------------------- vDNN start ----------------------

	free_bytes = max_consume;

	cnmemDevice_t cnmem_device;
	size_t cnmem_stream_memory_size = free_bytes;

	cnmem_device.device = 0;
	cnmem_device.size = cnmem_stream_memory_size;
	cnmem_device.numStreams = 0;
	cnmem_device.streams = NULL;
	cnmem_device.streamSizes = NULL;

	// do not allow call to cudaMalloc
	checkCNMEM(cnmemInit(1, &cnmem_device, CNMEM_FLAGS_CANNOT_GROW));
	// ---------------------- vDNN end ------------------------

	// ---------------------- vDNN start ----------------------
	for (int i = 0; i < num_layers; i++) {
		std::cerr << "to_offload[i] " << to_offload[i] << std::endl;
	}

	for (int i = 0; i < num_layers; i++) {
		// allocate pinned memory in host
		if (to_offload[i])
			checkCudaErrors(cudaMallocHost(&h_layer_input[i], layer_input_size[i] * data_type_size));
	}
	// ---------------------- vDNN end ------------------------
	checkCudaErrors(cudaDeviceSynchronize());
	size_t temp_free_bytes;
	checkCudaErrors(cudaMemGetInfo(&temp_free_bytes, &total_bytes));
	std::cout << "Free bytes just before end of NeuralNet: " << temp_free_bytes << std::endl;
	// {
	// 	int n;
	// 	std::cout << "waiting..\n";
	// 	std::cin >> n;
	// }

	// data of time
	checkCudaErrors(cudaEventCreate(&start_compute));
	checkCudaErrors(cudaEventCreate(&stop_compute));

	checkCudaErrors(cudaEventCreate(&start_transfer));
	checkCudaErrors(cudaEventCreate(&stop_transfer));
}

bool NeuralNet::simulateNeuralNetworkMemory(vDNNConvAlgoPref algo_pref, bool hard, size_t &exp_max_consume, size_t &max_consume) {
	CnmemSpace space_tracker(free_bytes);
	max_consume = 0;
	// forward pass
	// allocate space for 1st input
	std::cerr << "Initial Used space(MB): " << space_tracker.getConsumed() << std::endl;
	space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[0] * data_type_size);
	space_tracker.updateMaxConsume(max_consume);
	std::cerr << "Used space after allocating input(MB): " << space_tracker.getConsumed() << std::endl;
	
	std::cerr << "Forward pass" << std::endl;
	for (int i = 0; i < num_layers; i++) {
		if (layer_type[i] == SOFTMAX)
			break;

		std::cerr << "Processing layer " << i << std::endl;

		std::cerr << "Initial Used space(MB): " << space_tracker.getConsumed() << std::endl;
		space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[i + 1] * data_type_size);
		std::cerr << "Used space after output allocation(MB): " << space_tracker.getConsumed() << std::endl;
		space_tracker.updateMaxConsume(max_consume);

		if (layer_type[i] == CONV) {
			ConvLayerParams *cur_params = (ConvLayerParams *)params[i];
			size_t cur_workspace_size;
			checkWORKSPACE(cur_params->getWorkspaceSize(space_tracker.free_bytes, ConvLayerParams::FWD, algo_pref, hard, cur_workspace_size));
			space_tracker.updateSpace(CnmemSpace::SUB, cur_workspace_size);
			space_tracker.updateMaxConsume(max_consume);

			if (!space_tracker.isAvailable())
				return false;
			std::cerr << "Used space after workspace allocation(MB): " << space_tracker.getConsumed() << std::endl;

			// current layer computation over, deallocate workspace
			space_tracker.updateSpace(CnmemSpace::ADD, cur_workspace_size);
			std::cerr << "Used space after workspace deallocation(MB): " << space_tracker.getConsumed() << std::endl;
		}



		if (!space_tracker.isAvailable())
			return false;
		// deallocate layer input
		if (to_offload[i]) {
			std::cerr << "deallocating input to " << i << std::endl;
			space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i] * data_type_size);
			std::cerr << "Used space after deallocating input(MB): " << space_tracker.getConsumed() << std::endl;
		}
	}

	std::cerr << "Backward pass" << std::endl;
	if (batch_size * num_classes * data_type_size != layer_input_size[num_layers] * data_type_size) {
		std::cout << "Panic!! Using wrong size\n";
		exit(0);
	}	
	// backward pass
	space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[num_layers] * data_type_size);
	std::cerr << "Used space after allocating final derivative(MB): " << space_tracker.getConsumed() << std::endl;
	space_tracker.updateMaxConsume(max_consume);
	// std::cerr << "max_consume: " << max_consume << std::endl;
	for (int i = num_layers - 1; i >= 0; i--) {
		// allocate space for previous layer derivative
		std::cerr << "Processing layer " << i << std::endl;
		std::cerr << "Used space initial(MB): " << space_tracker.getConsumed() << std::endl;
		if (i > 0) {
			if (layer_type[i] == SOFTMAX)
				continue;
			else {
				space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[i] * data_type_size);
				std::cerr << "Used space after allocating prev. derivative(MB): " << space_tracker.getConsumed() << std::endl;
				space_tracker.updateMaxConsume(max_consume);
			}
			// std::cerr << "max_consume: " << max_consume << std::endl;
		}

		int layer_to_prefetch = findPrefetchLayer(i);
		// if layer to be prefetched, allocate space for that layer
		if (layer_to_prefetch != -1) {
			std::cerr << "Prefetch layer " << layer_to_prefetch << std::endl;
			space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[layer_to_prefetch] * data_type_size);
			std::cerr << "Used space after allocating prefetch(MB): " << space_tracker.getConsumed() << std::endl;
			space_tracker.updateMaxConsume(max_consume);
		}

		if (layer_type[i] == CONV) {
			ConvLayerParams *cur_params = (ConvLayerParams *)params[i];
			size_t cur_filter_workspace_size;
			checkWORKSPACE(cur_params->getWorkspaceSize(space_tracker.free_bytes, ConvLayerParams::BWD_FILTER, algo_pref, hard, cur_filter_workspace_size));
			size_t cur_data_workspace_size = 0;
			if (i > 0)
				checkWORKSPACE(cur_params->getWorkspaceSize(space_tracker.free_bytes, ConvLayerParams::BWD_DATA, algo_pref, hard, cur_data_workspace_size));

			size_t cur_workspace_size = (cur_filter_workspace_size > cur_data_workspace_size) ? cur_filter_workspace_size :cur_data_workspace_size;

			space_tracker.updateSpace(CnmemSpace::SUB, cur_workspace_size);
			std::cerr << "Used space after allocating workspace(MB): " << space_tracker.getConsumed() << std::endl;
			space_tracker.updateMaxConsume(max_consume);
			
			if (!pre_alloc_conv_derivative) {
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->kernel_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->C_out * data_type_size);
				space_tracker.updateMaxConsume(max_consume);
				std::cerr << "Used space after allocating weight derv.(MB): " << space_tracker.getConsumed() << std::endl;
			}

			// std::cerr << "max_consume: " << max_consume << std::endl;
			if (!space_tracker.isAvailable())
				return false;

			// current layer computation over, deallocate workspace
			space_tracker.updateSpace(CnmemSpace::ADD, cur_workspace_size);
			std::cerr << "Used space after deallocating workspace(MB): " << space_tracker.getConsumed() << std::endl;

			if (!pre_alloc_conv_derivative) {
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->kernel_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->C_out * data_type_size);
				std::cerr << "Used space after deallocating weight derv.(MB): " << space_tracker.getConsumed() << std::endl;
			}
		}
		else if (layer_type[i] == FULLY_CONNECTED) {
			FCLayerParams *cur_params = (FCLayerParams *)params[i];

			if (!pre_alloc_fc_derivative) {
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->weight_matrix_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->C_out * data_type_size);
				space_tracker.updateMaxConsume(max_consume);
				std::cerr << "Used space after allocating weight derv.(MB): " << space_tracker.getConsumed() << std::endl;
			}

			if (!space_tracker.isAvailable())
				return false;

			if (!pre_alloc_fc_derivative) {
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->weight_matrix_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->C_out * data_type_size);
				std::cerr << "Used space after deallocating weight derv.(MB): " << space_tracker.getConsumed() << std::endl;
			}
		}
		else if (layer_type[i] == BATCHNORM) {
			BatchNormLayerParams *cur_params = (BatchNormLayerParams *)params[i];
			if (!pre_alloc_batch_norm_derivative) {
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->allocation_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->allocation_size * data_type_size);
				space_tracker.updateMaxConsume(max_consume);
				std::cerr << "Used space after allocating weight derv.(MB): " << space_tracker.getConsumed() << std::endl;
			}

			if (!space_tracker.isAvailable())
				return false;

			if (!pre_alloc_batch_norm_derivative) {
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->allocation_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->allocation_size * data_type_size);
				std::cerr << "Used space after deallocating weight derv.(MB): " << space_tracker.getConsumed() << std::endl;
			}
		}

		if (!space_tracker.isAvailable())
			return false;
		// deallocate layer output and derivative
		space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i + 1] * data_type_size);
		space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i + 1] * data_type_size);
		std::cerr << "Used space after deallocating output, derivative(MB): " << space_tracker.getConsumed() << std::endl;
		// if 1st layer, deallocate input layer also
		if (i == 0) {
			space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i] * data_type_size);
			std::cerr << "Used space after deallocating input(MB): " << space_tracker.getConsumed() << std::endl;
		}
	}
	if (space_tracker.getConsumed() > 0)
		std::cerr << "Panic!! more free bytes\n";
	if (space_tracker.getConsumed() != 0)
		std::cerr << "Panic!! bytes not freed properly\n";
	// return true;

	exp_max_consume = max_consume;
	// check with cnmem once
	bool ret_val = simulateCNMEMMemory(max_consume);
	return ret_val;
}

bool NeuralNet::simulateCNMEMMemory(size_t &max_consume) {

	size_t init_max_consume = max_consume;
	cnmemDevice_t cnmem_device;

	size_t t;
	checkCudaErrors(cudaMemGetInfo(&free_bytes, &t));
	std::cout << "free_bytes: " << free_bytes << std::endl;
	free_bytes -= 100 * 1024 * 1024;
	cnmem_device.device = 0;
	cnmem_device.numStreams = 0;
	cnmem_device.streams = NULL;
	cnmem_device.streamSizes = NULL;

	std::string cnmem_memory_state_filename;
	if (vdnn_type == vDNN_ALL) {
		if (vdnn_conv_algo == vDNN_PERFORMANCE_OPTIMAL) {
			cnmem_memory_state_filename = "cnmem_all_p.dat";
		}
		else if (vdnn_conv_algo == vDNN_MEMORY_OPTIMAL) {
			cnmem_memory_state_filename = "cnmem_all_m.dat";
		}
	}
	else if (vdnn_type == vDNN_CONV) {
		if (vdnn_conv_algo == vDNN_PERFORMANCE_OPTIMAL) {
			cnmem_memory_state_filename = "cnmem_conv_p.dat";
		}
		else if (vdnn_conv_algo == vDNN_MEMORY_OPTIMAL) {
			cnmem_memory_state_filename = "cnmem_conv_m.dat";
		}
	}
	else if (vdnn_type == vDNN_DYN) {
		cnmem_memory_state_filename = "cnmem_dyn.dat";
	}
	else {
		cnmem_memory_state_filename = "cnmem_unknown.dat";
	}
	FILE *cnmem_memory_state_fptr = fopen(cnmem_memory_state_filename.c_str(), "w");

	size_t run_count = 0;
	bool out_of_memory = false;
	while (true) {
		run_count++;
		if (max_consume >= free_bytes)
			break;
		out_of_memory = false;
		cnmem_device.size = max_consume;
		std::cerr << run_count << ' ' << max_consume << std::endl;
		if (max_consume > free_bytes)
			std::cerr << "panic!! max_consume > free_bytes\n";
		checkCNMEM(cnmemInit(1, &cnmem_device, CNMEM_FLAGS_CANNOT_GROW));
		
		resetPrefetched();
		fprintf(cnmem_memory_state_fptr, "//////////////////////////////////////////////////////////////////\n");
		fprintf(cnmem_memory_state_fptr, "run_count: %lu\n", run_count);
		fprintf(cnmem_memory_state_fptr, "max_consume: %lu\n", max_consume);
		fprintf(cnmem_memory_state_fptr, "//////////////////////////////////////////////////////////////////\n");

		fprintf(cnmem_memory_state_fptr, "initial state\n");
		cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);

		checkCNMEMSim(cnmemMalloc(&layer_input[0], layer_input_size[0] * data_type_size, NULL), 
						layer_input_size[0] * data_type_size, max_consume, free_bytes, checkCNMEM(cnmemFinalize()); continue, out_of_memory);

		fprintf(cnmem_memory_state_fptr, "after alloc. layer_input[%d] - size: %lu\n", 0, layer_input_size[0] * data_type_size);
		cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);

		// forward propagate
		for (int i = 0; i < num_layers; i++) {
			size_t cur_workspace_size;
			void *cur_workspace;

			checkCNMEMSim(cnmemMalloc(&layer_input[i + 1], layer_input_size[i + 1] * data_type_size, NULL), 
							layer_input_size[i + 1] * data_type_size, max_consume, free_bytes, break, out_of_memory);

			fprintf(cnmem_memory_state_fptr, "after alloc. layer_input[%d] - size: %lu\n", i + 1, layer_input_size[i + 1] * data_type_size);
			cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);

			if (layer_type[i] == CONV) {
				// std::cout << "conv\n";
				ConvLayerParams *cur_params = (ConvLayerParams *)params[i];

				cur_workspace_size = cur_params->fwd_workspace_size;
				checkCNMEMSim(cnmemMalloc(&cur_workspace, cur_workspace_size, NULL), 
								cur_workspace_size, max_consume, free_bytes, break, out_of_memory);

				fprintf(cnmem_memory_state_fptr, "after alloc. conv. workspace - size: %lu\n", cur_workspace_size);
				cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);

			}		
			
			if (layer_type[i] == CONV) {
				checkCNMEMSim(cnmemFree(cur_workspace, NULL), 
								cur_workspace_size, max_consume, free_bytes, break, out_of_memory);

				fprintf(cnmem_memory_state_fptr, "after free conv. workspace - size: %lu\n", cur_workspace_size);
				cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
			}

			if (to_offload[i]) {
				checkCNMEMSim(cnmemFree(layer_input[i], NULL), 
								layer_input_size[i] * data_type_size, max_consume, free_bytes, break, out_of_memory);
				
				fprintf(cnmem_memory_state_fptr, "after free layer_input[%d] - size: %lu\n", i, layer_input_size[i] * data_type_size);
				cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
			}

			if (layer_type[i + 1] == ACTV or layer_type[i + 1] == SOFTMAX) {
				i = i + 1;
			}
		}

		if (out_of_memory) {
			checkCNMEM(cnmemFinalize());
			if (max_consume < free_bytes)
				continue;
			else
				break;
		}

		checkCNMEMSim(cnmemMalloc(&dlayer_input[num_layers], batch_size * num_classes * data_type_size, NULL), 
						layer_input_size[num_layers] * data_type_size, max_consume, free_bytes, checkCNMEM(cnmemFinalize()); continue, out_of_memory);

		fprintf(cnmem_memory_state_fptr, "after alloc. dlayer_input[%d] - size: %lu\n", num_layers, layer_input_size[num_layers] * data_type_size);
		cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);

		for (int i = num_layers - 1; i >= 0; i--) {
			// ---------------------- vDNN start ----------------------
			size_t cur_filter_workspace_size, cur_data_workspace_size, cur_workspace_size;
			void *cur_workspace;

			if (i > 0) {
				if (layer_type[i] == ACTV or layer_type[i] == SOFTMAX) {
					dlayer_input[i] = dlayer_input[i + 1];
				}
				else {

					int layer_to_prefetch = findPrefetchLayer(i);
					if (layer_to_prefetch != -1) {
						checkCNMEMSim(cnmemMalloc(&layer_input[layer_to_prefetch], layer_input_size[layer_to_prefetch] * data_type_size, NULL), 
										layer_input_size[layer_to_prefetch] * data_type_size, max_consume, free_bytes, break, out_of_memory);

						fprintf(cnmem_memory_state_fptr, "after alloc. prefetch layer_input[%d] - size: %lu\n", layer_to_prefetch, layer_input_size[layer_to_prefetch] * data_type_size);
						cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);

					}

					checkCNMEMSim(cnmemMalloc(&dlayer_input[i], layer_input_size[i] * data_type_size, NULL), 
									layer_input_size[i] * data_type_size, max_consume, free_bytes, break, out_of_memory);

					fprintf(cnmem_memory_state_fptr, "after alloc. dlayer_input[%d] - size: %lu\n", i, layer_input_size[i] * data_type_size);
					cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
				}
			}

			if (layer_type[i] == CONV) {
				// std::cout << "here\n";
				ConvLayerParams *cur_params = (ConvLayerParams *)params[i];

				// allocate space for derivative
				if (!pre_alloc_conv_derivative) {
					if (!cur_params->cnmemAllocDerivativesCheck(data_type_size, NULL, max_consume, free_bytes, out_of_memory))
						break;

					fprintf(cnmem_memory_state_fptr, "after alloc. dW - size: %lu\n", cur_params->kernel_size * data_type_size);
					fprintf(cnmem_memory_state_fptr, "after alloc. db - size: %lu\n", cur_params->C_out * data_type_size);
					cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
				}

				cur_filter_workspace_size = cur_params->bwd_filter_workspace_size;
				if (i > 0)
					cur_data_workspace_size = cur_params->bwd_data_workspace_size;
				else
					cur_data_workspace_size = 0;
				cur_workspace_size = (cur_filter_workspace_size > cur_data_workspace_size) ? cur_filter_workspace_size : cur_data_workspace_size;
				checkCNMEMSim(cnmemMalloc(&cur_workspace, cur_workspace_size, NULL), 
								cur_workspace_size, max_consume, free_bytes, break, out_of_memory);

				fprintf(cnmem_memory_state_fptr, "after alloc. conv. workspace - size: %lu\n", cur_workspace_size);
				cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);

			}

			else if (layer_type[i] == FULLY_CONNECTED) {
				FCLayerParams *cur_params = (FCLayerParams *)params[i];

				if (!pre_alloc_fc_derivative) {
					if (!cur_params->cnmemAllocDerivativesCheck(data_type_size, NULL, max_consume, free_bytes, out_of_memory))
						break;

					fprintf(cnmem_memory_state_fptr, "after alloc. dW - size: %lu\n", cur_params->weight_matrix_size * data_type_size);
					fprintf(cnmem_memory_state_fptr, "after alloc. db - size: %lu\n", cur_params->C_out * data_type_size);
					cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
				}
			}

			else if (layer_type[i] == BATCHNORM) {
				BatchNormLayerParams *cur_params = (BatchNormLayerParams *)params[i];

				if (!pre_alloc_batch_norm_derivative) {
					if (!cur_params->cnmemAllocDerivativesCheck(data_type_size, NULL, max_consume, free_bytes, out_of_memory))
						break;

					fprintf(cnmem_memory_state_fptr, "after alloc. dscale - size: %lu\n", cur_params->allocation_size * data_type_size);
					fprintf(cnmem_memory_state_fptr, "after alloc. dbias - size: %lu\n", cur_params->allocation_size * data_type_size);
					cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
				}
			}

			else if (layer_type[i] == SOFTMAX) {
				// std::cout << "compute here\n";
				SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[i];
				continue;
			}

			if (layer_type[i] == CONV) {
				checkCNMEMSim(cnmemFree(cur_workspace, NULL), 
								cur_workspace_size, max_consume, free_bytes, break, out_of_memory);
				fprintf(cnmem_memory_state_fptr, "after free conv. workspace - size: %lu\n", cur_workspace_size);
				cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);

				if (!pre_alloc_conv_derivative) {
					ConvLayerParams *cur_params = (ConvLayerParams *)params[i];
					cur_params->cnmemFreeDerivatives(NULL);
					fprintf(cnmem_memory_state_fptr, "after free dP - size: %lu\n", (long unsigned)0);
					cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
				}
			}
			else if (layer_type[i] == FULLY_CONNECTED) {
				if (!pre_alloc_fc_derivative) {
					FCLayerParams *cur_params = (FCLayerParams *)params[i];
					cur_params->cnmemFreeDerivatives(NULL);
					fprintf(cnmem_memory_state_fptr, "after free dP - size: %lu\n", (long unsigned)0);
					cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
				}
			}
			else if (layer_type[i] == BATCHNORM) {
				if (!pre_alloc_batch_norm_derivative) {
					BatchNormLayerParams *cur_params = (BatchNormLayerParams *)params[i];
					cur_params->cnmemFreeDerivatives(NULL);
					fprintf(cnmem_memory_state_fptr, "after free dP - size: %lu\n", (long unsigned)0);
					cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
				}
			}

			checkCNMEMSim(cnmemFree(layer_input[i + 1], NULL), 
							layer_input_size[i + 1] * data_type_size, max_consume, free_bytes, break, out_of_memory);
			fprintf(cnmem_memory_state_fptr, "after free layer_input[%d] - size: %lu\n", i + 1, layer_input_size[i + 1] * data_type_size);
			cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
			checkCNMEMSim(cnmemFree(dlayer_input[i + 1], NULL), 
							layer_input_size[i + 1] * data_type_size, max_consume, free_bytes, break, out_of_memory);
			fprintf(cnmem_memory_state_fptr, "after free dlayer_input[%d] - size: %lu\n", i + 1, layer_input_size[i + 1] * data_type_size);
			cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
			if (i == 0) {
				checkCNMEMSim(cnmemFree(layer_input[i], NULL), 
								layer_input_size[i] * data_type_size, max_consume, free_bytes, break, out_of_memory);
				fprintf(cnmem_memory_state_fptr, "after free layer_input[%d] - size: %lu\n", i, layer_input_size[i] * data_type_size);
				cnmemPrintMemoryStateTogether(cnmem_memory_state_fptr, NULL);
			}	
		}
		checkCNMEM(cnmemFinalize());
		if (out_of_memory) {
			if (max_consume < free_bytes)
				continue;
			else
				break;
		}
		break;
	}
	free_bytes += 100 * 1024 * 1024;
	if (max_consume < free_bytes) {
		double exp_size = (init_max_consume + init_free_bytes - free_bytes) / (1.0 * 1024 * 1024);
		double act_size = (max_consume + init_free_bytes - free_bytes) / (1.0 * 1024 * 1024);
		fprintf(cnmem_memory_state_fptr, "expected_memory_consume: %f MB\n", exp_size);
		fprintf(cnmem_memory_state_fptr, "actual_memory_consume: %f MB\n", act_size);
	}
	else {
		fprintf(cnmem_memory_state_fptr, "out of memory\n");
	}

	fclose(cnmem_memory_state_fptr);
	if (max_consume < free_bytes)
		return true;
	else
		return false;
}

void NeuralNet::vDNNOptimize(size_t &exp_max_consume, size_t &max_consume) {

	bool hard = true, soft = false;

	// if type is vDNN_ALL or vDNN_CONV, check if sufficient space is available
	if (vdnn_type == vDNN_ALL) {
		setOffload(OFFLOAD_ALL);
		resetPrefetched();
		if (vdnn_conv_algo == vDNN_PERFORMANCE_OPTIMAL) {
			if (!simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume))
				outOfMemory();
		}
		else if (vdnn_conv_algo == vDNN_MEMORY_OPTIMAL) {
			if (!simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume))
				outOfMemory();
		}

		return;
	}
	else if (vdnn_type == vDNN_CONV) {
		setOffload(OFFLOAD_CONV);
		resetPrefetched();
		if (vdnn_conv_algo == vDNN_PERFORMANCE_OPTIMAL) {
			if (!simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume))
				outOfMemory();
		}
		else if (vdnn_conv_algo == vDNN_MEMORY_OPTIMAL) {
			if (!simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume))
				outOfMemory();
		}

		return;
	}
	else if (vdnn_type == vDNN_NONE) {
		setOffload(OFFLOAD_NONE);
		resetPrefetched();
		if (vdnn_conv_algo == vDNN_PERFORMANCE_OPTIMAL) {
			if (!simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume))
				outOfMemory();
		}
		else if (vdnn_conv_algo == vDNN_MEMORY_OPTIMAL) {
			if (!simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume))
				outOfMemory();
		}

		return;	
	}
	else if (vdnn_type == vDNN_ALTERNATE_CONV) {
		setOffload(OFFLOAD_ALTERNATE_CONV);
		resetPrefetched();
		if (vdnn_conv_algo == vDNN_PERFORMANCE_OPTIMAL) {
			if (!simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume))
				outOfMemory();
		}
		else if (vdnn_conv_algo == vDNN_MEMORY_OPTIMAL) {
			if (!simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume))
				outOfMemory();
		}
		return;
	}

	if (vdnn_type == vDNN_DYN) {
	
		// check for trainability
		std::cerr << "vDNN_DYN\n";
		setOffload(NeuralNet::OFFLOAD_ALL);
		resetPrefetched();
		if(!simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume))
			outOfMemory();
	
		// check if work with fastest algo and no offload, if so, select it and return
		setOffload(NeuralNet::OFFLOAD_NONE);
		resetPrefetched();
		if (simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume)) {
			std::cerr << "Choosing PERF_OPT, NO OFFLOAD\n";
			return;
		}
	
		// check if conv offload and fastest algo works, then check if all offload and fastest algo works
		setOffload(NeuralNet::OFFLOAD_CONV);
		resetPrefetched();
		if (simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume)) {
			std::cerr << "Choosing PERF_OPT, CONV OFFLOAD\n";
			return;
		}
	
		setOffload(NeuralNet::OFFLOAD_ALL);
		resetPrefetched();
		if (simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, hard, exp_max_consume, max_consume)) {
			std::cerr << "Choosing PERF_OPT, ALL OFFLOAD\n";
			return;
		}
	
		// optimize using greedy algo memory usage while improving performance
		setOffload(NeuralNet::OFFLOAD_CONV);
		resetPrefetched();
		if (simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, soft, exp_max_consume, max_consume)) {
			std::cerr << "Choosing GREEDY, CONV OFFLOAD\n";
			return;
		}
	
		setOffload(NeuralNet::OFFLOAD_ALL);
		resetPrefetched();
		if (simulateNeuralNetworkMemory(PREFER_PERFORMANCE_OPTIMAL, soft, exp_max_consume, max_consume)) {
			std::cerr << "Choosing GREEDY, ALL OFFLOAD\n";
			return;
		}
	
		setOffload(NeuralNet::OFFLOAD_CONV);
		resetPrefetched();
		if (simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume)) {
			std::cerr << "Choosing MEM_OPT, CONV OFFLOAD\n";
			return;
		}

		setOffload(NeuralNet::OFFLOAD_ALL);
		resetPrefetched();
		if(simulateNeuralNetworkMemory(PREFER_MEMORY_OPTIMAL, hard, exp_max_consume, max_consume)) {
			std::cerr << "Choosing MEM_OPT, ALL OFFLOAD\n";
			return;
		}
	}
	exit(0);

}

void NeuralNet::setOffload(NeuralNet::OffloadType offload_type) {
	if (offload_type == OFFLOAD_NONE) {
		for (int i = 0; i < num_layers; i++)
			to_offload[i] = false;
	}
	else if (offload_type == OFFLOAD_CONV) {
		for (int i = 0; i < num_layers; i++) {
			if (layer_type[i] == CONV)
				to_offload[i] = true;
			else
				to_offload[i] = false;
		}
		// set last non SOFTMAX/ACTV layer to no_offload
		for (int i = num_layers - 1; i >= 0; i--) {
			if (layer_type[i] == SOFTMAX or layer_type[i] == ACTV)
				;
			else {
				to_offload[i] = false;
				break;
			}
		}
	}
	else if (offload_type == OFFLOAD_ALL) {
		for (int i = 0; i < num_layers; i++) {
			// Only SOFTMAX, CONV, POOL, FULLY_CONNECTED used so far
			if (layer_type[i] == ACTV or layer_type[i] == SOFTMAX or layer_type[i] == FULLY_CONNECTED)
				to_offload[i] = false;
			else
				to_offload[i] = true;
		}
		// set last non SOFTMAX/ACTV layer to no_offload
		for (int i = num_layers - 1; i >= 0; i--) {
			if (layer_type[i] == SOFTMAX or layer_type[i] == ACTV)
				;
			else {
				to_offload[i] = false;
				break;
			}
		}
	}
	else if (offload_type == OFFLOAD_ALTERNATE_CONV) {
		for (int i = 0; i < num_layers; i++) {
			if (layer_type[i] == CONV)
				to_offload[i] = true;
			else
				to_offload[i] = false;
		}
		// set last non SOFTMAX/ACTV layer to no_offload
		for (int i = num_layers - 1; i >= 0; i--) {
			if (layer_type[i] == SOFTMAX or layer_type[i] == ACTV)
				;
			else {
				to_offload[i] = false;
				break;
			}
		}
		bool toggle = true;
		for (int i = 0; i < num_layers; i++) {
			if (layer_type[i] == CONV) {
				if (toggle == false)
					to_offload[i] = false;
				toggle = !toggle;

			}
		}	
	}
}

void NeuralNet::resetPrefetched() {
	for (int i = 0; i < num_layers; i++)
		prefetched[i] = false;
}

void NeuralNet::getLoss(void *X, int *y, double learning_rate, bool train, int *correct_count, float *loss) {
	std::vector<float> t1, t2;
	this->getLoss(X, y, learning_rate, t1, t2, train, correct_count, loss);
}

void NeuralNet::getLoss(void *X, int *y, double learning_rate, std::vector<float> &fwd_vdnn_lag, std::vector<float> &bwd_vdnn_lag, bool train, int *correct_count, float *scalar_loss) {

	CnmemSpace space_tracker(free_bytes);
	// std::cout << "here\n";
	// std::cout << "Free bytes: " << free_bytes << std::endl;
	for (int i = 0; i < num_layers; i++)
		prefetched[i] = false;

	checkCNMEM(cnmemMalloc(&layer_input[0], layer_input_size[0] * data_type_size, NULL));
	space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[0] * data_type_size);
	checkCudaErrors(cudaMemcpy(layer_input[0], X, batch_size * input_channels * input_h * input_w * data_type_size, cudaMemcpyHostToDevice));
	if (train == true) {
		checkCudaErrors(cudaMemcpy(this->y, y, batch_size * data_type_size, cudaMemcpyHostToDevice));
	}
	float alpha = 1.0, beta = 0.0;
	float Salpha = 1.0, Sbeta = 0.0;
	double Dalpha = 1.0, Dbeta = 0.0;

	// forward propagate
	for (int i = 0; i < num_layers; i++) {
		if (train == false && i == num_layers - 1)
			break;
		// ---------------------- vDNN start ----------------------
		size_t cur_workspace_size;
		void *cur_workspace;

		// offload if required
		if (i > 0 && to_offload[i] && train == true)
			checkCudaErrors(cudaMemcpyAsync(h_layer_input[i], layer_input[i], 
											layer_input_size[i] * data_type_size, cudaMemcpyDeviceToHost, stream_memory));

		checkCNMEM(cnmemMalloc(&layer_input[i + 1], layer_input_size[i + 1] * data_type_size, NULL));
		space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[i + 1] * data_type_size);
		// std::cout << "Free bytes: " << free_bytes << std::endl;
		// ---------------------- vDNN end ------------------------
		// std::cout << "here" << i << std::endl;
		if (layer_type[i] == CONV) {
			// std::cout << "conv\n";
			ConvLayerParams *cur_params = (ConvLayerParams *)params[i];

			cur_workspace_size = cur_params->fwd_workspace_size;
			checkCNMEM(cnmemMalloc(&cur_workspace, cur_workspace_size, NULL));			
			// computation
			checkCUDNN(cudnnConvolutionForward(cudnn_handle, &alpha, 
												cur_params->input_tensor, layer_input[i],
												cur_params->filter_desc, cur_params->W,
												cur_params->conv_desc, cur_params->fwd_algo,
												cur_workspace, cur_workspace_size,
												&beta,
												cur_params->output_tensor, layer_input[i + 1]));
			checkCUDNN(cudnnAddTensor(cudnn_handle, &alpha, 
										cur_params->bias_desc, cur_params->b, 
										&alpha,
										cur_params->output_tensor, layer_input[i + 1]));

			// if activation required
			if (cur_params->activation_mode != ACTIVATION_NONE) {
				checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc,
												&alpha,
												cur_params->output_tensor, layer_input[i + 1],
												&beta,
												cur_params->output_tensor, layer_input[i + 1]));
			}

			space_tracker.updateSpace(CnmemSpace::SUB, cur_workspace_size);
			// std::cout << "Free bytes: " << free_bytes << std::endl;
			
		}

		else if (layer_type[i] == FULLY_CONNECTED) {
			// std::cout << "FC\n";
			FCLayerParams *cur_params = (FCLayerParams *)params[i];
			// std::cout << "FChere" << i << std::endl;

			if (data_type == CUDNN_DATA_FLOAT) {
				checkCUBLAS(cublasSgemm(cublas_handle,
										CUBLAS_OP_N, CUBLAS_OP_N,
										cur_params->C_out, batch_size, cur_params->C_in,
										&Salpha,
										(float *)cur_params->W, cur_params->C_out,
										(float *)layer_input[i], cur_params->C_in,
										&Sbeta,
										(float *)layer_input[i + 1], cur_params->C_out));
				checkCUBLAS(cublasSgemm(cublas_handle,
										CUBLAS_OP_N, CUBLAS_OP_N,
										cur_params->C_out, batch_size, 1,
										&Salpha,
										(float *)cur_params->b, cur_params->C_out,
										(float *)one_vec, 1,
										&Salpha,
										(float *)layer_input[i + 1], cur_params->C_out));
			}
			else if (data_type == CUDNN_DATA_DOUBLE) {
				checkCUBLAS(cublasDgemm(cublas_handle,
										CUBLAS_OP_N, CUBLAS_OP_N,
										cur_params->C_out, batch_size, cur_params->C_in,
										&Dalpha,
										(double *)cur_params->W, cur_params->C_out,
										(double *)layer_input[i], cur_params->C_in,
										&Dbeta,
										(double *)layer_input[i + 1], cur_params->C_out));
				checkCUBLAS(cublasDgemm(cublas_handle,
										CUBLAS_OP_N, CUBLAS_OP_N,
										cur_params->C_out, batch_size, 1,
										&Dalpha,
										(double *)cur_params->b, cur_params->C_out,
										(double *)one_vec, 1,
										&Dalpha,
										(double *)layer_input[i + 1], cur_params->C_out));
			}
			if (cur_params->activation_mode != ACTIVATION_NONE) {
				checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc,
												&alpha,
												cur_params->output_tensor, layer_input[i + 1],
												&beta,
												cur_params->output_tensor, layer_input[i + 1]));
			}
			// std::cout << "FChere" << i << std::endl;
		}
		else if (layer_type[i] == DROPOUT) {
			// std::cout << "Dropout\n";
			DropoutLayerParams *cur_params = (DropoutLayerParams *)params[i];
			checkCUDNN(cudnnDropoutForward(cudnn_handle, cur_params->dropout_desc,
											cur_params->input_tensor, layer_input[i],
											cur_params->input_tensor, layer_input[i + 1],
											cur_params->reserved_space,
											cur_params->reserved_space_size));
		}
		else if (layer_type[i] == BATCHNORM) {
			// std::cout << "Batchnorm\n";
			BatchNormLayerParams *cur_params = (BatchNormLayerParams *)params[i];

			if (train == true) {
				checkCUDNN(cudnnBatchNormalizationForwardTraining(cudnn_handle, cur_params->mode,
																	&alpha, &beta,
																	cur_params->input_tensor, layer_input[i],
																	cur_params->input_tensor, layer_input[i + 1],
																	cur_params->sbmv_desc,
																	cur_params->scale, cur_params->bias,
																	cur_params->factor,
																	cur_params->running_mean, cur_params->running_variance,
																	cur_params->epsilon,
																	cur_params->result_save_mean, cur_params->result_save_inv_var));

			}
			else {
				checkCUDNN(cudnnBatchNormalizationForwardInference(cudnn_handle, cur_params->mode,
																	&alpha, &beta,
																	cur_params->input_tensor, layer_input[i],
																	cur_params->input_tensor, layer_input[i + 1],
																	cur_params->sbmv_desc,
																	cur_params->scale, cur_params->bias,
																	cur_params->running_mean, cur_params->running_variance,
																	cur_params->epsilon));
			}
		}
		else if (layer_type[i] == POOLING) {
			// std::cout << "Pooling\n";
			PoolingLayerParams *cur_params = (PoolingLayerParams *)params[i];
			checkCUDNN(cudnnPoolingForward(cudnn_handle, cur_params->pool_desc,
											&alpha,
											cur_params->input_tensor, layer_input[i],
											&beta,
											cur_params->output_tensor, layer_input[i + 1]));
		}
		else if (layer_type[i] == ACTV) {
			// std::cout << "Actv\n";
			std::cout << "Panic!! ACTV wrong place\n";
			exit(0);
			ActivationLayerParams *cur_params = (ActivationLayerParams *)params[i];
			checkCUDNN(cudnnActivationForward(cudnn_handle, cur_params->actv_desc,
												&alpha,
												cur_params->input_tensor, layer_input[i],
												&beta,
												cur_params->input_tensor, layer_input[i + 1]));
		}
		else if (layer_type[i] == SOFTMAX) {
			// std::cout << "Softmax\n";
			std::cout << "Panic!! SOFTMAX wrong place\n";
			exit(0);
			if (train == true) {
				SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[i];
				checkCUDNN(cudnnSoftmaxForward(cudnn_handle, cur_params->algo, cur_params->mode,
												&alpha,
												cur_params->input_tensor, layer_input[i],
												&beta,
												cur_params->input_tensor, layer_input[i + 1]));
			}
		}

		// ---------------------- vDNN start ----------------------
		// synchronization
		// checkCudaErrors(cudaDeviceSynchronize());

		// if next layer is ACTV or SOFTMAX, complete that and come to synchronization
		// the case in above if for ACTV and SOFTMAX never occurs
		if (layer_type[i + 1] == SOFTMAX) {
			i++;
			if (train == true) {
				layer_input[i + 1] = layer_input[i];
				SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[i];
				checkCUDNN(cudnnSoftmaxForward(cudnn_handle, cur_params->algo, cur_params->mode,
												&alpha,
												cur_params->input_tensor, layer_input[i],
												&beta,
												cur_params->input_tensor, layer_input[i + 1]));
			}
			i--;
		}

		struct timespec start_time, end_time;
		checkCudaErrors(cudaStreamSynchronize(stream_compute));
		
		if (train)
			clock_gettime(CLOCK_MONOTONIC, &start_time);
		
		checkCudaErrors(cudaStreamSynchronize(stream_memory));
		

		if (train) {
			clock_gettime(CLOCK_MONOTONIC, &end_time);
			float lag = (end_time.tv_sec - start_time.tv_sec) * 1e3 + (end_time.tv_nsec - start_time.tv_nsec) * 1e-6;
			fwd_vdnn_lag.push_back(lag);
		}
		// std::cout << "EndSynchere" << i << std::endl;
		if (layer_type[i] == CONV) {
			checkCNMEM(cnmemFree(cur_workspace, NULL));
			space_tracker.updateSpace(CnmemSpace::ADD, cur_workspace_size);
		}

		if (to_offload[i] && train == true) {
			checkCNMEM(cnmemFree(layer_input[i], NULL));
			space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i] * data_type_size);
		}
		if (train == false) {
			checkCNMEM(cnmemFree(layer_input[i], NULL));
			space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i] * data_type_size);
		}

		if (layer_type[i + 1] == ACTV or layer_type[i + 1] == SOFTMAX) {
			i = i + 1;
		}
		// std::cout << "EndSynchere" << i << std::endl;

		// ---------------------- vDNN end ------------------------
	}

	// std::cout << "here" << std::endl;
	if (train == false) {
		compareOutputCorrect(correct_count, y);
		checkCNMEM(cnmemFree(layer_input[num_layers - 1], NULL));
		space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[num_layers - 1] * data_type_size);
		return;
	}
	*scalar_loss = computeLoss();

	// ---------------------- vDNN start ----------------------
	checkCNMEM(cnmemMalloc(&dlayer_input[num_layers], batch_size * num_classes * data_type_size, NULL));
	space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[num_layers] * data_type_size);
	// std::cout << "Free bytes: " << free_bytes << std::endl;
	// ---------------------- vDNN end ------------------------
	if (layer_type[num_layers - 1] == SOFTMAX) {
		// SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[num_layers - 1];
		if (data_type == CUDNN_DATA_FLOAT) {
			checkCudaErrors(cudaMemset(dlayer_input[num_layers], 0, batch_size * num_classes * sizeof(float)));
			softmaxLossBackProp<float><<<ceil(1.0 * batch_size / BW), BW>>>(this->y, (float *)layer_input[num_layers], 
																			(float *)dlayer_input[num_layers], batch_size, num_classes, softmax_eps);
		}
		else if (data_type == CUDNN_DATA_DOUBLE) {
			checkCudaErrors(cudaMemset(dlayer_input[num_layers], 0, batch_size * num_classes * sizeof(double)));
			softmaxLossBackProp<double><<<ceil(1.0 * batch_size / BW), BW>>>(this->y, (double *)layer_input[num_layers], 
																			(double *)dlayer_input[num_layers], batch_size, num_classes, softmax_eps);
		}
	}
	for (int i = num_layers - 1; i >= 0; i--) {
		// ---------------------- vDNN start ----------------------
		size_t cur_filter_workspace_size, cur_data_workspace_size, cur_workspace_size;
		void *cur_workspace;

		if (i > 0) {
			if (layer_type[i] == ACTV or layer_type[i] == SOFTMAX) {
				dlayer_input[i] = dlayer_input[i + 1];
			}
			else {
				int layer_to_prefetch = findPrefetchLayer(i);
				if (layer_to_prefetch != -1) {
					checkCNMEM(cnmemMalloc(&layer_input[layer_to_prefetch], layer_input_size[layer_to_prefetch] * data_type_size, NULL));
					space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[layer_to_prefetch] * data_type_size);
					// std::cout << "Free bytes: " << free_bytes << std::endl;
					if (layer_to_prefetch != 0) {
						checkCudaErrors(cudaMemcpyAsync(layer_input[layer_to_prefetch], h_layer_input[layer_to_prefetch], 
														layer_input_size[layer_to_prefetch] * data_type_size, cudaMemcpyHostToDevice, stream_memory));
					}
					else {
						// std::cout << "transfer here\n";
						checkCudaErrors(cudaMemcpyAsync(layer_input[layer_to_prefetch], X, 
														layer_input_size[layer_to_prefetch] * data_type_size, cudaMemcpyHostToDevice, stream_memory));
						// std::cout << "transfer here\n";
					}
				}
				checkCNMEM(cnmemMalloc(&dlayer_input[i], layer_input_size[i] * data_type_size, NULL));
				space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[i] * data_type_size);
			}
			// std::cout << "Free bytes: " << free_bytes << std::endl;
		}
		// ---------------------- vDNN end ------------------------

		if (layer_type[i] == CONV) {
			// std::cout << "here\n";
			ConvLayerParams *cur_params = (ConvLayerParams *)params[i];

			if (cur_params->activation_mode != ACTIVATION_NONE) {
				checkCUDNN(cudnnActivationBackward(cudnn_handle, cur_params->actv_desc, &alpha,
												cur_params->output_tensor, layer_input[i + 1],
												cur_params->output_tensor, dlayer_input[i + 1],
												cur_params->output_tensor, layer_input[i + 1],
												&beta,
												cur_params->output_tensor, dlayer_input[i + 1]));
			}

			// allocate space for derivative
			if (!pre_alloc_conv_derivative) {
				cur_params->cnmemAllocDerivatives(data_type_size, NULL);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->kernel_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->C_out * data_type_size);
			}

			cur_filter_workspace_size = cur_params->bwd_filter_workspace_size;
			if (i > 0)
				cur_data_workspace_size = cur_params->bwd_data_workspace_size;
			else
				cur_data_workspace_size = 0;
			// std::cout << "bwd cur_workspace_size: " << cur_workspace_size << std::endl;
			cur_workspace_size = (cur_filter_workspace_size > cur_data_workspace_size) ? cur_filter_workspace_size : cur_data_workspace_size;
			checkCNMEM(cnmemMalloc(&cur_workspace, cur_workspace_size, NULL));

			checkCUDNN(cudnnConvolutionBackwardBias(cudnn_handle, &alpha,
													cur_params->output_tensor, dlayer_input[i + 1],
													&beta,
													cur_params->bias_desc, cur_params->db));

			// std::cout << "neural_net: backward conv i:" << i << std::endl;

			checkCUDNN(cudnnConvolutionBackwardFilter(cudnn_handle, &alpha,
														cur_params->input_tensor, layer_input[i],
														cur_params->output_tensor, dlayer_input[i + 1],
														cur_params->conv_desc, cur_params->bwd_filter_algo,
														cur_workspace, cur_workspace_size,
														&beta, 
														cur_params->filter_desc,
														cur_params->dW));
			if (i > 0)
				checkCUDNN(cudnnConvolutionBackwardData(cudnn_handle, &alpha,
														cur_params->filter_desc, cur_params->W,
														cur_params->output_tensor, dlayer_input[i + 1],
														cur_params->conv_desc, cur_params->bwd_data_algo,
														cur_workspace, cur_workspace_size,
														&beta,
														cur_params->input_tensor, dlayer_input[i]));

			space_tracker.updateSpace(CnmemSpace::SUB, cur_workspace_size);
			// std::cout << "Free bytes: " << free_bytes << std::endl;
			// std::cout << "here\n";
			cur_params->stepParams(cublas_handle, learning_rate);
		}

		else if (layer_type[i] == FULLY_CONNECTED) {
			FCLayerParams *cur_params = (FCLayerParams *)params[i];

			if (cur_params->activation_mode != ACTIVATION_NONE) {
				checkCUDNN(cudnnActivationBackward(cudnn_handle, cur_params->actv_desc, &alpha,
												cur_params->output_tensor, layer_input[i + 1],
												cur_params->output_tensor, dlayer_input[i + 1],
												cur_params->output_tensor, layer_input[i + 1],
												&beta,
												cur_params->output_tensor, dlayer_input[i + 1]));
			}

			if (!pre_alloc_fc_derivative) {
				cur_params->cnmemAllocDerivatives(data_type_size, NULL);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->weight_matrix_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->C_out * data_type_size);
			}

			if (data_type == CUDNN_DATA_FLOAT) {
				// bias backward
				checkCUBLAS(cublasSgemm(cublas_handle,
										CUBLAS_OP_N, CUBLAS_OP_N,
										cur_params->C_out, 1, batch_size,
										&Salpha,
										(float *)dlayer_input[i + 1], cur_params->C_out,
										(float *)one_vec, batch_size,
										&Sbeta,
										(float *)cur_params->db, cur_params->C_out));

				// weight backward
				checkCUBLAS(cublasSgemm(cublas_handle,
										CUBLAS_OP_N, CUBLAS_OP_T,
										cur_params->C_out, cur_params->C_in, batch_size,
										&Salpha,
										(float *)dlayer_input[i + 1], cur_params->C_out,
										(float *)layer_input[i], cur_params->C_in,
										&Sbeta,
										(float *)cur_params->dW, cur_params->C_out));

				// data backward
				if (i > 0)
					checkCUBLAS(cublasSgemm(cublas_handle,
											CUBLAS_OP_T, CUBLAS_OP_N,
											cur_params->C_in, batch_size, cur_params->C_out,
											&Salpha,
											(float *)cur_params->W, cur_params->C_out,
											(float *)dlayer_input[i + 1], cur_params->C_out,
											&Sbeta,
											(float *)dlayer_input[i], cur_params->C_in));
			}

			else if (data_type == CUDNN_DATA_DOUBLE) {
				// bias backward
				checkCUBLAS(cublasDgemm(cublas_handle,
										CUBLAS_OP_N, CUBLAS_OP_N,
										cur_params->C_out, 1, batch_size,
										&Dalpha,
										(double *)dlayer_input[i + 1], cur_params->C_out,
										(double *)one_vec, batch_size,
										&Dbeta,
										(double *)cur_params->db, cur_params->C_out));

				// weight backward
				checkCUBLAS(cublasDgemm(cublas_handle,
										CUBLAS_OP_N, CUBLAS_OP_T,
										cur_params->C_out, cur_params->C_in, batch_size,
										&Dalpha,
										(double *)dlayer_input[i + 1], cur_params->C_out,
										(double *)layer_input[i], cur_params->C_in,
										&Dbeta,
										(double *)cur_params->dW, cur_params->C_out));

				// data backward
				if (i > 0)
					checkCUBLAS(cublasDgemm(cublas_handle,
											CUBLAS_OP_T, CUBLAS_OP_N,
											cur_params->C_in, batch_size, cur_params->C_out,
											&Dalpha,
											(double *)cur_params->W, cur_params->C_out,
											(double *)dlayer_input[i + 1], cur_params->C_out,
											&Dbeta,
											(double *)dlayer_input[i], cur_params->C_in));
			}
			cur_params->stepParams(cublas_handle, learning_rate);
		}

		else if (layer_type[i] == DROPOUT) {
			DropoutLayerParams *cur_params = (DropoutLayerParams *)params[i];
			checkCUDNN(cudnnDropoutBackward(cudnn_handle, cur_params->dropout_desc,
											cur_params->input_tensor, dlayer_input[i + 1],
											cur_params->input_tensor, dlayer_input[i],
											cur_params->reserved_space, cur_params->reserved_space_size));
		}

		else if (layer_type[i] == BATCHNORM) {
			BatchNormLayerParams *cur_params = (BatchNormLayerParams *)params[i];

			if (!pre_alloc_batch_norm_derivative) {
				cur_params->cnmemAllocDerivatives(data_type_size, NULL);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->allocation_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::SUB, cur_params->allocation_size * data_type_size);
			}

			checkCUDNN(cudnnBatchNormalizationBackward(cudnn_handle, cur_params->mode,
														&alpha, &beta,
														&alpha, &beta,
														cur_params->input_tensor, layer_input[i],
														cur_params->input_tensor, dlayer_input[i + 1],
														cur_params->input_tensor, dlayer_input[i],
														cur_params->sbmv_desc, cur_params->scale,
														cur_params->dscale, cur_params->dbias,
														cur_params->epsilon,
														cur_params->result_save_mean, cur_params->result_save_inv_var));

			cur_params->stepParams(cublas_handle, learning_rate);
		}

		else if (layer_type[i] == POOLING) {
			PoolingLayerParams *cur_params = (PoolingLayerParams *)params[i];
			checkCUDNN(cudnnPoolingBackward(cudnn_handle, cur_params->pool_desc, &alpha, 
											cur_params->output_tensor, layer_input[i + 1],
											cur_params->output_tensor, dlayer_input[i + 1],
											cur_params->input_tensor, layer_input[i],
											&beta,
											cur_params->input_tensor, dlayer_input[i]));
		}

		else if (layer_type[i] == ACTV) {
			ActivationLayerParams *cur_params = (ActivationLayerParams *)params[i];
			checkCUDNN(cudnnActivationBackward(cudnn_handle, cur_params->actv_desc, &alpha,
												cur_params->input_tensor, layer_input[i + 1],
												cur_params->input_tensor, dlayer_input[i + 1],
												cur_params->input_tensor, layer_input[i],
												&beta,
												cur_params->input_tensor, dlayer_input[i]));
			continue;
		}

		else if (layer_type[i] == SOFTMAX) {
			// std::cout << "compute here\n";
			SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[i];
			checkCUDNN(cudnnSoftmaxBackward(cudnn_handle, cur_params->algo, cur_params->mode, &alpha,
											cur_params->input_tensor, layer_input[i + 1],
											cur_params->input_tensor, dlayer_input[i + 1],
											&beta,
											cur_params->input_tensor, dlayer_input[i]));
			// std::cout << "compute here\n";
			continue;
		}

		// ---------------------- vDNN start ----------------------
		
		// checkCudaErrors(cudaDeviceSynchronize());
		struct timespec start_time, end_time;
		checkCudaErrors(cudaStreamSynchronize(stream_compute));

		if (train)
			clock_gettime(CLOCK_MONOTONIC, &start_time);

		checkCudaErrors(cudaStreamSynchronize(stream_memory));
		if (train) {
			clock_gettime(CLOCK_MONOTONIC, &end_time);
			float lag = (end_time.tv_sec - start_time.tv_sec) * 1e3 + (end_time.tv_nsec - start_time.tv_nsec) * 1e-6;
			bwd_vdnn_lag.insert(bwd_vdnn_lag.begin(), lag);
		}

		if (layer_type[i] == CONV) {
			checkCNMEM(cnmemFree(cur_workspace, NULL));
			space_tracker.updateSpace(CnmemSpace::ADD, cur_workspace_size);
			if (!pre_alloc_conv_derivative) {
				ConvLayerParams *cur_params = (ConvLayerParams *)params[i];
				cur_params->cnmemFreeDerivatives(NULL);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->kernel_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->C_out * data_type_size);
			}
		}
		else if (layer_type[i] == FULLY_CONNECTED) {
			if (!pre_alloc_fc_derivative) {
				FCLayerParams *cur_params = (FCLayerParams *)params[i];
				cur_params->cnmemFreeDerivatives(NULL);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->weight_matrix_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->C_out * data_type_size);
			}
		}
		else if (layer_type[i] == BATCHNORM) {
			if (train == true and !pre_alloc_batch_norm_derivative) {
				BatchNormLayerParams *cur_params = (BatchNormLayerParams *)params[i];
				cur_params->cnmemFreeDerivatives(NULL);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->allocation_size * data_type_size);
				space_tracker.updateSpace(CnmemSpace::ADD, cur_params->allocation_size * data_type_size);
			}
		}

		checkCNMEM(cnmemFree(layer_input[i + 1], NULL));
		space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i + 1] * data_type_size);
		checkCNMEM(cnmemFree(dlayer_input[i + 1], NULL));
		space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i + 1] * data_type_size);
		if (i == 0) {
			checkCNMEM(cnmemFree(layer_input[i], NULL));
			space_tracker.updateSpace(CnmemSpace::ADD, layer_input_size[i] * data_type_size);
		}	
		// ---------------------- vDNN end ------------------------
	}
	if (space_tracker.getConsumed() != 0) {
		std::cout << "Panic!! Space not updated properly\n";
	}

	// exit(0);
}


int NeuralNet::findPrefetchLayer(int cur_layer) {
	for (int i = cur_layer - 1; i >= 0; i--) {
		if (to_offload[i] && !prefetched[i]) {
			prefetched[i] = true;
			return i;
		}
		else if (layer_type[i] == CONV) {
			return -1;
		}
	}
	return -1;
}
