#include "layer_params.h"

void ConvLayerParams::initializeValues(cudnnHandle_t cudnn_handle, ConvDescriptor *user_params, cudnnDataType_t data_type, 
									int batch_size, cudnnTensorFormat_t tensor_format, size_t data_type_size, LayerDimension &output_size, 
									UpdateRule update_rule) {
	// create tensor, filter, conv descriptor
	checkCUDNN(cudnnCreateTensorDescriptor(&input_tensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&output_tensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&bias_desc));
	checkCUDNN(cudnnCreateFilterDescriptor(&filter_desc));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

	C_in = user_params->input_channels;
	C_out = user_params->output_channels;
	filter_h = user_params->kernel_h;
	filter_w = user_params->kernel_w;
	kernel_size = C_out * C_in * filter_h * filter_w;
	this->data_type = data_type;
	this->activation_mode = user_params->activation_mode;

	checkCUDNN(cudnnSetTensor4dDescriptor(input_tensor, tensor_format, data_type, 
										batch_size, user_params->input_channels, user_params->input_h, user_params->input_w));


	checkCUDNN(cudnnSetFilter4dDescriptor(filter_desc, data_type, tensor_format, 
										user_params->output_channels, user_params->input_channels, user_params->kernel_h, user_params->kernel_w));

	int dilation_h = 1, dilation_w = 1;
	checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc, user_params->pad_h, user_params->pad_w, 
												user_params->stride_y, user_params->stride_x,
												dilation_h, dilation_w, 
												CUDNN_CROSS_CORRELATION, data_type));

	int output_batch_size, output_channels, output_h, output_w;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc, input_tensor, filter_desc,
													&output_batch_size, &output_channels, &output_h, &output_w));

	checkCUDNN(cudnnSetTensor4dDescriptor(output_tensor, tensor_format, data_type, 
										output_batch_size, output_channels, output_h, output_w));
	checkCUDNN(cudnnSetTensor4dDescriptor(bias_desc, tensor_format, data_type, 
										1, output_channels, 1, 1));

	fwd_req_count = 10;
	fwd_perf = (cudnnConvolutionFwdAlgoPerf_t *)malloc(fwd_req_count * sizeof(cudnnConvolutionFwdAlgoPerf_t));
	checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn_handle, 
													input_tensor, filter_desc, conv_desc, output_tensor, 
													fwd_req_count, &fwd_ret_count, fwd_perf));

	// std::cout << "Printing forward conv algo perf\n";
	// std::cout << "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM: " << CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM << std::endl;
	// for (int i = 0; i < fwd_ret_count; i++) {
	// 	std::cout << i << std::endl;
	// 	std::cout << "algo: " << fwd_perf[i].algo << std::endl;
	// 	std::cout << "status: " << cudnnGetErrorString(fwd_perf[i].status) << std::endl;
	// 	std::cout << "time(ms): " << fwd_perf[i].time << std::endl;
	// 	std::cout << "memory(MB): " << fwd_perf[i].memory * 1.0 / 1024 / 1024 << std::endl;
	// 	std::cout << "mathType: " << fwd_perf[i].mathType << std::endl;
	// 	std::cout << std::endl;
	// }

	bwd_filter_req_count = 10;
	bwd_filter_perf = (cudnnConvolutionBwdFilterAlgoPerf_t *)malloc(bwd_filter_req_count * sizeof(cudnnConvolutionBwdFilterAlgoPerf_t));
	checkCUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(cudnn_handle, 
															input_tensor, output_tensor, conv_desc, filter_desc, 
															bwd_filter_req_count, &bwd_filter_ret_count, bwd_filter_perf));

	// std::cout << "Printing bwdfilter conv algo perf\n";
	// std::cout << "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 " << CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 << std::endl;
	// for (int i = 0; i < bwd_filter_ret_count; i++) {
	// 	std::cout << i << std::endl;
	// 	std::cout << "algo: " << bwd_filter_perf[i].algo << std::endl;
	// 	std::cout << "status: " << cudnnGetErrorString(bwd_filter_perf[i].status) << std::endl;
	// 	std::cout << "time(ms): " << bwd_filter_perf[i].time << std::endl;
	// 	std::cout << "memory(MB): " << bwd_filter_perf[i].memory * 1.0 / 1024 / 1024 << std::endl;
	// 	std::cout << "mathType: " << bwd_filter_perf[i].mathType << std::endl;
	// 	std::cout << std::endl;
	// }
	bwd_data_req_count = 10;
	bwd_data_perf = (cudnnConvolutionBwdDataAlgoPerf_t *)malloc(bwd_data_req_count * sizeof(cudnnConvolutionBwdDataAlgoPerf_t));
	checkCUDNN(cudnnFindConvolutionBackwardDataAlgorithm(cudnn_handle,
															filter_desc, output_tensor, conv_desc, input_tensor, 
															bwd_data_req_count, &bwd_data_ret_count, bwd_data_perf));

	// std::cout << "Printing bwddata conv algo perf\n";
	// for (int i = 0; i < bwd_data_ret_count; i++) {
	// 	std::cout << i << std::endl;
	// 	std::cout << "algo: " << bwd_data_perf[i].algo << std::endl;
	// 	std::cout << "status: " << cudnnGetErrorString(bwd_data_perf[i].status) << std::endl;
	// 	std::cout << "time(ms): " << bwd_data_perf[i].time << std::endl;
	// 	std::cout << "memory(MB): " << bwd_data_perf[i].memory * 1.0 / 1024 / 1024 << std::endl;
	// 	std::cout << "mathType: " << bwd_data_perf[i].mathType << std::endl;
	// 	std::cout << std::endl;
	// }

	this->update_rule = update_rule;

	cudnnActivationMode_t mode;
	if (activation_mode == SIGMOID)
		mode = CUDNN_ACTIVATION_SIGMOID;
	else if (activation_mode == RELU)
		mode = CUDNN_ACTIVATION_RELU;
	else if (activation_mode == TANH)
		mode = CUDNN_ACTIVATION_TANH;
	else if (activation_mode == CLIPPED_RELU)
		mode = CUDNN_ACTIVATION_CLIPPED_RELU;
	else if (activation_mode == ELU)
		mode = CUDNN_ACTIVATION_ELU;

	if (activation_mode != ACTIVATION_NONE) {
		checkCUDNN(cudnnCreateActivationDescriptor(&actv_desc));
		checkCUDNN(cudnnSetActivationDescriptor(actv_desc, mode, CUDNN_PROPAGATE_NAN, user_params->actv_coef));
	}

	output_size.N = output_batch_size, output_size.C = output_channels, output_size.H = output_h, output_size.W = output_w;
	
}

void ConvLayerParams::allocateSpace(curandGenerator_t curand_gen, cudnnDataType_t data_type, size_t data_type_size, 
									float std_dev, size_t &free_bytes, bool alloc_derivative) {

	if (kernel_size % 2 != 0)
		kernel_size += 1;
	checkCudaErrors(cudaMalloc(&W, kernel_size * data_type_size));
	checkCudaErrors(cudaMalloc(&b, C_out * data_type_size));
	
	if (alloc_derivative) {
		checkCudaErrors(cudaMalloc(&dW, kernel_size * data_type_size));
		checkCudaErrors(cudaMalloc(&db, C_out * data_type_size));
	}

	if (data_type == CUDNN_DATA_FLOAT) {
		checkCURAND(curandGenerateNormal(curand_gen, (float *)W, kernel_size, 0, std_dev));
		fillValue<float><<<ceil(1.0 * C_out / BW), BW>>>((float *)b, C_out, 0);
	}
	else {
		checkCURAND(curandGenerateNormalDouble(curand_gen, (double *)W, kernel_size, 0, std_dev));
		fillValue<double><<<ceil(1.0 * C_out / BW), BW>>>((double *)b, C_out, 0);
	}

	free_bytes = free_bytes - 2 * (kernel_size + C_out) * data_type_size;

}

void ConvLayerParams::cnmemAllocDerivatives(size_t data_type_size, cudaStream_t stream) {
	checkCNMEM(cnmemMalloc(&dW, kernel_size * data_type_size, stream));
	checkCNMEM(cnmemMalloc(&db, C_out * data_type_size, stream));
}

bool ConvLayerParams::cnmemAllocDerivativesCheck(size_t data_type_size, cudaStream_t stream, 
													size_t &max_consume, size_t free_bytes, bool &out_of_memory) {
	checkCNMEMSim(cnmemMalloc(&dW, kernel_size * data_type_size, stream), 
					kernel_size * data_type_size, max_consume, free_bytes, return false, out_of_memory);
	checkCNMEMSim(cnmemMalloc(&db, C_out * data_type_size, stream), 
					C_out * data_type_size, max_consume, free_bytes, return false, out_of_memory);

	return true;
}

void ConvLayerParams::stepParams(cublasHandle_t cublas_handle, double learning_rate) {
	float Salpha = -learning_rate;
	double Dalpha = -learning_rate;

	if (update_rule == SGD) {
		if (data_type == CUDNN_DATA_FLOAT) {
			checkCUBLAS(cublasSaxpy(cublas_handle, kernel_size,
									&Salpha,
									(float *)dW, 1,
									(float *)W, 1));

			checkCUBLAS(cublasSaxpy(cublas_handle, C_out,
									&Salpha,
									(float *)db, 1,
									(float *)b, 1));
		}
		else if (data_type == CUDNN_DATA_DOUBLE) {
			checkCUBLAS(cublasDaxpy(cublas_handle, kernel_size,
									&Dalpha,
									(double *)dW, 1,
									(double *)W, 1));

			checkCUBLAS(cublasDaxpy(cublas_handle, C_out,
									&Dalpha,
									(double *)db, 1,
									(double *)b, 1));
		}
	}
}

void ConvLayerParams::cnmemFreeDerivatives(cudaStream_t stream) {
	checkCNMEM(cnmemFree(dW, stream));
	checkCNMEM(cnmemFree(db, stream));
}

size_t ConvLayerParams::getWorkspaceSize(size_t &free_bytes, ConvLayerParams::ConvDirection conv_direction, vDNNConvAlgo vdnn_conv_algo) {
	if (vdnn_conv_algo == vDNN_PERFORMANCE_OPTIMAL) {
		if (conv_direction == FWD) {
			if (fwd_perf[0].memory > free_bytes)
				outOfMemory();
			fwd_algo = fwd_perf[0].algo;
			return fwd_perf[0].memory;
		}
		else if (conv_direction == BWD_FILTER) {
			if (bwd_filter_perf[0].memory > free_bytes)
				outOfMemory();
			bwd_filter_algo = bwd_filter_perf[0].algo;
			return bwd_filter_perf[0].memory;
		}
		else if (conv_direction == BWD_DATA) {
			if (bwd_data_perf[0].memory > free_bytes)
				outOfMemory();
			bwd_data_algo = bwd_data_perf[0].algo;
			return bwd_data_perf[0].memory;
		}
	}
	else if (vdnn_conv_algo == vDNN_MEMORY_OPTIMAL) {
		if (conv_direction == FWD) {
			for (int i = 0; i < fwd_ret_count; i++) {
				if (fwd_perf[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM && fwd_perf[i].status == CUDNN_STATUS_SUCCESS &&
						fwd_perf[i].memory < free_bytes) {
					fwd_algo = fwd_perf[i].algo;
					return fwd_perf[i].memory;
				}
			}
		}
		else if (conv_direction == BWD_FILTER) {
			for (int i = 0; i < bwd_filter_ret_count; i++) {
				if (bwd_filter_perf[i].algo == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 && bwd_filter_perf[i].status == CUDNN_STATUS_SUCCESS &&
						bwd_filter_perf[i].memory < free_bytes) {
					bwd_filter_algo = bwd_filter_perf[i].algo;
					// std::cout << "Free bytes " << free_bytes << std::endl;
					// std::cout << "bwd_filter_perf[i].memory " << bwd_filter_perf[i].memory << std::endl;
					return bwd_filter_perf[i].memory;
				}
			}
		}
		else if (conv_direction == BWD_DATA) {
			for (int i = 0; i < bwd_data_ret_count; i++) {
				if (bwd_data_perf[i].algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 && bwd_data_perf[i].status == CUDNN_STATUS_SUCCESS &&
						bwd_data_perf[i].memory < free_bytes) {
					bwd_data_algo = bwd_data_perf[i].algo;
					return bwd_data_perf[i].memory;
				}
			}
		}
		std::cout << "Error in getWorkspaceSize" << std::endl;
		exit(0);
	}
	return 0;
}

workspaceStatus_t ConvLayerParams::getWorkspaceSize(size_t &free_bytes, ConvLayerParams::ConvDirection conv_direction, vDNNConvAlgoPref algo_pref, 
										bool hard_pref, size_t &workspace_size) {
	if (hard_pref) {
		if (algo_pref == PREFER_PERFORMANCE_OPTIMAL) {
			if (conv_direction == FWD) {
				if (fwd_perf[0].memory > free_bytes && fwd_perf[0].status == CUDNN_STATUS_SUCCESS)
					return WORKSPACE_STATUS_OUT_OF_MEMORY;
				fwd_algo = fwd_perf[0].algo;
				fwd_workspace_size = fwd_perf[0].memory;
				workspace_size = fwd_workspace_size;
				return WORKSPACE_STATUS_SUCCESS;
			}
			else if (conv_direction == BWD_FILTER) {
				if (bwd_filter_perf[0].memory > free_bytes && bwd_filter_perf[0].status == CUDNN_STATUS_SUCCESS)
					return WORKSPACE_STATUS_OUT_OF_MEMORY;
				bwd_filter_algo = bwd_filter_perf[0].algo;
				bwd_filter_workspace_size = bwd_filter_perf[0].memory;
				workspace_size = bwd_filter_workspace_size;
				return WORKSPACE_STATUS_SUCCESS;
			}
			else if (conv_direction == BWD_DATA) {
				if (bwd_data_perf[0].memory > free_bytes && bwd_data_perf[0].status == CUDNN_STATUS_SUCCESS)
					return WORKSPACE_STATUS_OUT_OF_MEMORY;
				bwd_data_algo = bwd_data_perf[0].algo;
				bwd_data_workspace_size = bwd_data_perf[0].memory;
				workspace_size = bwd_data_workspace_size;
				return WORKSPACE_STATUS_SUCCESS;
			}
		}
		else if (algo_pref == PREFER_MEMORY_OPTIMAL) {
			if (conv_direction == FWD) {
				for (int i = 0; i < fwd_ret_count; i++) {
					if (fwd_perf[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
						if (fwd_perf[i].memory < free_bytes && fwd_perf[i].status == CUDNN_STATUS_SUCCESS) {
							fwd_algo = fwd_perf[i].algo;
							fwd_workspace_size = fwd_perf[i].memory;
							workspace_size = fwd_workspace_size;
							return WORKSPACE_STATUS_SUCCESS;
						}
						else
							return WORKSPACE_STATUS_OUT_OF_MEMORY;
				}
			}
			else if (conv_direction == BWD_FILTER) {
				for (int i = 0; i < bwd_filter_ret_count; i++) {
					if (bwd_filter_perf[i].algo == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1)
						if (bwd_filter_perf[i].memory < free_bytes && bwd_filter_perf[i].status == CUDNN_STATUS_SUCCESS) {
							bwd_filter_algo = bwd_filter_perf[i].algo;
							// std::cout << "Free bytes " << free_bytes << std::endl;
							// std::cout << "bwd_filter_perf[i].memory " << bwd_filter_perf[i].memory << std::endl;
							bwd_filter_workspace_size = bwd_filter_perf[i].memory;
							workspace_size = bwd_filter_workspace_size;
							return WORKSPACE_STATUS_SUCCESS;
						}
						else
							return WORKSPACE_STATUS_OUT_OF_MEMORY;
				}
			}
			else if (conv_direction == BWD_DATA) {
				for (int i = 0; i < bwd_data_ret_count; i++) {
					if (bwd_data_perf[i].algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_1)
						if (bwd_data_perf[i].memory < free_bytes && bwd_data_perf[i].status == CUDNN_STATUS_SUCCESS) {
							bwd_data_algo = bwd_data_perf[i].algo;
							bwd_data_workspace_size = bwd_data_perf[i].memory;
							workspace_size = bwd_data_workspace_size;
							return WORKSPACE_STATUS_SUCCESS;
						}
						else
							return WORKSPACE_STATUS_OUT_OF_MEMORY;
				}
			}
		}
	}
	else {
		// only performance optimal is possible
		if (algo_pref == PREFER_PERFORMANCE_OPTIMAL) {
			if (conv_direction == FWD) {
				for (int i = 0; i < fwd_ret_count; i++) {
					if (fwd_perf[i].memory < free_bytes && fwd_perf[i].status == CUDNN_STATUS_SUCCESS) {
						fwd_algo = fwd_perf[i].algo;
						fwd_workspace_size = fwd_perf[i].memory;
						workspace_size = fwd_workspace_size;
						return WORKSPACE_STATUS_SUCCESS;
					}
				}
			}
			else if (conv_direction == BWD_FILTER) {
				for (int i = 0; i < bwd_filter_ret_count; i++) {
					if (bwd_filter_perf[i].memory < free_bytes && bwd_filter_perf[i].status == CUDNN_STATUS_SUCCESS) {
						bwd_filter_algo = bwd_filter_perf[i].algo;
						// std::cout << "Free bytes " << free_bytes << std::endl;
						// std::cout << "bwd_filter_perf[i].memory " << bwd_filter_perf[i].memory << std::endl;
						bwd_filter_workspace_size = bwd_filter_perf[i].memory;
						workspace_size = bwd_filter_workspace_size;
						return WORKSPACE_STATUS_SUCCESS;
					}
				}
			}
			else if (conv_direction == BWD_DATA) {
				for (int i = 0; i < bwd_data_ret_count; i++) {
					if (bwd_data_perf[i].memory < free_bytes && bwd_data_perf[i].status == CUDNN_STATUS_SUCCESS) {
						bwd_data_algo = bwd_data_perf[i].algo;
						bwd_data_workspace_size = bwd_data_perf[i].memory;
						workspace_size = bwd_data_workspace_size;
						return WORKSPACE_STATUS_SUCCESS;
					}
				}
			}
		}
	}
	return WORKSPACE_STATUS_OUT_OF_MEMORY;
}

void FCLayerParams::initializeValues(FCDescriptor *user_params, int batch_size, cudnnTensorFormat_t tensor_format, cudnnDataType_t data_type, 
										LayerDimension &output_size, UpdateRule update_rule) {
	C_in = user_params->input_channels;
	C_out = user_params->output_channels;
	weight_matrix_size = C_in * C_out;
	this->data_type = data_type;
	this->activation_mode = user_params->activation_mode;

	this->update_rule = update_rule;

	cudnnActivationMode_t mode;
	if (activation_mode == SIGMOID)
		mode = CUDNN_ACTIVATION_SIGMOID;
	else if (activation_mode == RELU)
		mode = CUDNN_ACTIVATION_RELU;
	else if (activation_mode == TANH)
		mode = CUDNN_ACTIVATION_TANH;
	else if (activation_mode == CLIPPED_RELU)
		mode = CUDNN_ACTIVATION_CLIPPED_RELU;
	else if (activation_mode == ELU)
		mode = CUDNN_ACTIVATION_ELU;

	if (activation_mode != ACTIVATION_NONE) {
		checkCUDNN(cudnnCreateActivationDescriptor(&actv_desc));
		checkCUDNN(cudnnSetActivationDescriptor(actv_desc, mode, CUDNN_PROPAGATE_NAN, user_params->actv_coef));
		checkCUDNN(cudnnCreateTensorDescriptor(&output_tensor));
		checkCUDNN(cudnnSetTensor4dDescriptor(output_tensor, tensor_format, data_type, 
										batch_size, user_params->output_channels, 1, 1));
	}

	output_size.N = batch_size, output_size.C = C_out, output_size.H = output_size.W = 1;
}

void FCLayerParams::allocateSpace(curandGenerator_t curand_gen, cudnnDataType_t data_type, size_t data_type_size, 
									float std_dev, size_t &free_bytes, bool alloc_derivative) {
	int wt_alloc_size = weight_matrix_size;
	if (wt_alloc_size % 2 != 0)
		wt_alloc_size += 1;
	checkCudaErrors(cudaMalloc(&W, wt_alloc_size * data_type_size));
	checkCudaErrors(cudaMalloc(&b, C_out * data_type_size));
	if (alloc_derivative) {
		checkCudaErrors(cudaMalloc(&dW, wt_alloc_size * data_type_size));
		checkCudaErrors(cudaMalloc(&db, C_out * data_type_size));
	}

	if (data_type == CUDNN_DATA_FLOAT) {
		checkCURAND(curandGenerateNormal(curand_gen, (float *)W, wt_alloc_size, 0, std_dev));
		fillValue<float><<<ceil(1.0 * C_out / BW), BW>>>((float *)b, C_out, 0);
	}
	else if (data_type == CUDNN_DATA_DOUBLE) {
		checkCURAND(curandGenerateNormalDouble(curand_gen, (double *)W, wt_alloc_size, 0, std_dev));
		fillValue<double><<<ceil(1.0 * C_out / BW), BW>>>((double *)b, C_out, 0);
	}
	free_bytes = free_bytes - 2 * (C_in * C_out + C_out) * data_type_size;
}

void FCLayerParams::cnmemAllocDerivatives(size_t data_type_size, cudaStream_t stream) {
	checkCNMEM(cnmemMalloc(&dW, weight_matrix_size * data_type_size, stream));
	checkCNMEM(cnmemMalloc(&db, C_out * data_type_size, stream));
}

bool FCLayerParams::cnmemAllocDerivativesCheck(size_t data_type_size, cudaStream_t stream, 
												size_t &max_consume, size_t free_bytes, bool &out_of_memory) {
	checkCNMEMSim(cnmemMalloc(&dW, weight_matrix_size * data_type_size, stream), 
					weight_matrix_size * data_type_size, max_consume, free_bytes, return false, out_of_memory);
	checkCNMEMSim(cnmemMalloc(&db, C_out * data_type_size, stream), 
					C_out * data_type_size, max_consume, free_bytes, return false, out_of_memory);
	return true;
}

void FCLayerParams::stepParams(cublasHandle_t cublas_handle, double learning_rate) {
	float Salpha = -learning_rate;
	double Dalpha = -learning_rate;
	
	// {
	// 	float *db_h = (float *)malloc(C_out * sizeof(float));
	// 	checkCudaErrors(cudaMemcpy(db_h, db, C_out * sizeof(float), cudaMemcpyDeviceToHost));
	// 	for (int i = 0; i < C_out; i++) {
	// 		std::cout << db_h[i] << ' ';
	// 	}
	// 	std::cout << "\n";
	// 	int n;
	// 	std::cin >> n;
	// }


	if (update_rule == SGD) {
		if (data_type == CUDNN_DATA_FLOAT) {
			checkCUBLAS(cublasSaxpy(cublas_handle, weight_matrix_size,
									&Salpha,
									(float *)dW, 1,
									(float *)W, 1));
	
			checkCUBLAS(cublasSaxpy(cublas_handle, C_out,
									&Salpha,
									(float *)db, 1,
									(float *)b, 1));
		}
		else if (data_type == CUDNN_DATA_DOUBLE) {
			checkCUBLAS(cublasDaxpy(cublas_handle, weight_matrix_size,
									&Dalpha,
									(double *)dW, 1,
									(double *)W, 1));

			checkCUBLAS(cublasDaxpy(cublas_handle, C_out,
									&Dalpha,
									(double *)db, 1,
									(double *)b, 1));
		}
	}
	// {
	// 	float *db_h = (float *)malloc(C_out * sizeof(float));
	// 	checkCudaErrors(cudaMemcpy(db_h, b, C_out * sizeof(float), cudaMemcpyDeviceToHost));
	// 	for (int i = 0; i < C_out; i++) {
	// 		std::cout << db_h[i] << ' ';
	// 	}
	// 	std::cout << "\n";
	// 	int n;
	// 	std::cin >> n;
	// }
}

void FCLayerParams::cnmemFreeDerivatives(cudaStream_t stream) {
	checkCNMEM(cnmemFree(dW, stream));
	checkCNMEM(cnmemFree(db, stream));
}

void DropoutLayerParams::initializeValues(cudnnHandle_t cudnn_handle, DropoutDescriptor *user_params, cudnnDataType_t data_type, int batch_size,
										 cudnnTensorFormat_t tensor_format, LayerDimension &output_size) {
	checkCUDNN(cudnnCreateDropoutDescriptor(&dropout_desc));
	checkCUDNN(cudnnCreateTensorDescriptor(&input_tensor));

	checkCUDNN(cudnnSetTensor4dDescriptor(input_tensor, tensor_format, data_type, 
										batch_size, user_params->channels, user_params->h, user_params->w));

	checkCUDNN(cudnnDropoutGetStatesSize(cudnn_handle, &state_size));

	checkCUDNN(cudnnDropoutGetReserveSpaceSize(input_tensor, &reserved_space_size));
	
	output_size.N = batch_size, output_size.C = user_params->channels, output_size.H = user_params->h, output_size.W = user_params->w;

}

void DropoutLayerParams::allocateSpace(size_t &free_bytes, cudnnHandle_t cudnn_handle, DropoutDescriptor *user_params, long long seed) {
	checkCudaErrors(cudaMalloc(&state, state_size));
	checkCudaErrors(cudaMalloc(&reserved_space, reserved_space_size));
	checkCUDNN(cudnnSetDropoutDescriptor(dropout_desc, cudnn_handle, user_params->dropout_value, state, state_size, seed));

	free_bytes = free_bytes - (state_size + reserved_space_size);
}

void BatchNormLayerParams::initializeValues(BatchNormDescriptor *user_params, cudnnDataType_t data_type, cudnnTensorFormat_t tensor_format, 
							int batch_size, LayerDimension &output_size, UpdateRule update_rule) {
	checkCUDNN(cudnnCreateTensorDescriptor(&input_tensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&sbmv_desc));
	c = user_params->channels, h = user_params->h, w = user_params->w;
	if (user_params->mode == BATCHNORM_PER_ACTIVATION) {
		mode = CUDNN_BATCHNORM_PER_ACTIVATION;
		checkCUDNN(cudnnSetTensor4dDescriptor(sbmv_desc, tensor_format, data_type,
												1, user_params->channels, user_params->h, user_params->w));
		sbmv_size = c * h * w;
	}
	else if (user_params->mode == BATCHNORM_SPATIAL) {
		mode = CUDNN_BATCHNORM_SPATIAL;
		checkCUDNN(cudnnSetTensor4dDescriptor(sbmv_desc, tensor_format, data_type,
												1, user_params->channels, 1, 1));
		sbmv_size = c;
	}

	checkCUDNN(cudnnSetTensor4dDescriptor(input_tensor, tensor_format, data_type, 
										batch_size, user_params->channels, user_params->h, user_params->w));
	
	factor = user_params->factor;
	epsilon = user_params->epsilon;

	this->update_rule = update_rule;
	this->data_type = data_type;

	if (mode == CUDNN_BATCHNORM_PER_ACTIVATION)
		allocation_size = c * h * w;
	else
		allocation_size = c;

	output_size.N = batch_size, output_size.C = user_params->channels, output_size.H = user_params->h, output_size.W = user_params->w;
}

void BatchNormLayerParams::allocateSpace(cudnnDataType_t data_type, size_t data_type_size, size_t &free_bytes, bool alloc_derivative) {

	size_t allocation_size_bytes = allocation_size * data_type_size;

	checkCudaErrors(cudaMalloc(&scale, allocation_size_bytes));
	checkCudaErrors(cudaMalloc(&bias, allocation_size_bytes));
	if (alloc_derivative) {
		checkCudaErrors(cudaMalloc(&dscale, allocation_size_bytes));
		checkCudaErrors(cudaMalloc(&dbias, allocation_size_bytes));
	}

	checkCudaErrors(cudaMalloc(&running_mean, allocation_size_bytes));
	checkCudaErrors(cudaMalloc(&running_variance, allocation_size_bytes));

	checkCudaErrors(cudaMalloc(&result_save_mean, allocation_size_bytes));
	checkCudaErrors(cudaMalloc(&result_save_inv_var, allocation_size_bytes));

	if (data_type == CUDNN_DATA_FLOAT) {
		fillValue<float><<<ceil(1.0 * allocation_size / BW), BW>>>((float *)scale, allocation_size, 1);
		fillValue<float><<<ceil(1.0 * allocation_size / BW), BW>>>((float *)bias, allocation_size, 1);
	}
	else if (data_type == CUDNN_DATA_DOUBLE) {
		fillValue<double><<<ceil(1.0 * allocation_size / BW), BW>>>((double *)scale, allocation_size, 1);
		fillValue<double><<<ceil(1.0 * allocation_size / BW), BW>>>((double *)bias, allocation_size, 1);
	}
	free_bytes = free_bytes - 6 * allocation_size_bytes;

}

void BatchNormLayerParams::cnmemAllocDerivatives(size_t data_type_size, cudaStream_t stream) {
	checkCNMEM(cnmemMalloc(&dscale, allocation_size * data_type_size, stream));
	checkCNMEM(cnmemMalloc(&dbias, allocation_size * data_type_size, stream));
}

bool BatchNormLayerParams::cnmemAllocDerivativesCheck(size_t data_type_size, cudaStream_t stream, 
														size_t &max_consume, size_t free_bytes, bool &out_of_memory) {
	checkCNMEMSim(cnmemMalloc(&dscale, allocation_size * data_type_size, stream), 
					allocation_size * data_type_size, max_consume, free_bytes, return false, out_of_memory);
	checkCNMEMSim(cnmemMalloc(&dbias, allocation_size * data_type_size, stream), 
					allocation_size * data_type_size, max_consume, free_bytes, return false, out_of_memory);
	return true;
}

void BatchNormLayerParams::stepParams(cublasHandle_t cublas_handle, double learning_rate) {
	float Salpha = -learning_rate;
	double Dalpha = -learning_rate;

	if (update_rule == SGD) {
		if (data_type == CUDNN_DATA_FLOAT) {
			checkCUBLAS(cublasSaxpy(cublas_handle, sbmv_size,
									&Salpha,
									(float *)dscale, 1,
									(float *)scale, 1));
			checkCUBLAS(cublasSaxpy(cublas_handle, sbmv_size,
									&Salpha,
									(float *)dbias, 1,
									(float *)bias, 1));
		}
		else if (data_type == CUDNN_DATA_DOUBLE) {
			checkCUBLAS(cublasDaxpy(cublas_handle, sbmv_size,
									&Dalpha,
									(double *)dscale, 1,
									(double *)scale, 1));
			checkCUBLAS(cublasDaxpy(cublas_handle, sbmv_size,
									&Dalpha,
									(double *)dbias, 1,
									(double *)bias, 1));
		}
	}
}

void BatchNormLayerParams::cnmemFreeDerivatives(cudaStream_t stream) {
	checkCNMEM(cnmemFree(dscale, stream));
	checkCNMEM(cnmemFree(dbias, stream));
}

void PoolingLayerParams::initializeValues(PoolingDescriptor *user_params, cudnnDataType_t data_type, cudnnTensorFormat_t tensor_format, 
							int batch_size, LayerDimension &output_size) {
	checkCUDNN(cudnnCreateTensorDescriptor(&input_tensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&output_tensor));

	checkCUDNN(cudnnSetTensor4dDescriptor(input_tensor, tensor_format, data_type, 
										batch_size, user_params->input_channels, user_params->input_h, user_params->input_w));
	

	checkCUDNN(cudnnCreatePoolingDescriptor(&pool_desc));

	cudnnPoolingMode_t mode;
	if (user_params->mode == POOLING_MAX)
		mode = CUDNN_POOLING_MAX;
	else if (user_params->mode == POOLING_AVERAGE_COUNT_INCLUDE_PADDING)
		mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
	else if (user_params->mode == POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)
		mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

	checkCUDNN(cudnnSetPooling2dDescriptor(pool_desc, mode, CUDNN_PROPAGATE_NAN,
											user_params->kernel_h, user_params->kernel_w,
											user_params->pad_h, user_params->pad_w,
											user_params->stride_y, user_params->stride_x));


	int output_batch_size, output_channels, output_h, output_w;
	checkCUDNN(cudnnGetPooling2dForwardOutputDim(pool_desc, input_tensor, 
											&output_batch_size, &output_channels, &output_h, &output_w));

	checkCUDNN(cudnnSetTensor4dDescriptor(output_tensor, tensor_format, data_type, 
										output_batch_size, output_channels, output_h, output_w));

	output_size.N = output_batch_size, output_size.C = output_channels, output_size.H = output_h, output_size.W = output_w;

}

void PoolingLayerParams::allocateSpace(size_t &free_bytes) {

}

void ActivationLayerParams::initializeValues(ActivationDescriptor *user_params, cudnnDataType_t data_type,
											cudnnTensorFormat_t tensor_format, int batch_size, LayerDimension &output_size) {
	checkCUDNN(cudnnCreateTensorDescriptor(&input_tensor));

	checkCUDNN(cudnnSetTensor4dDescriptor(input_tensor, tensor_format, data_type, 
										batch_size, user_params->channels, user_params->h, user_params->w));

	cudnnActivationMode_t mode;
	if (user_params->mode == SIGMOID)
		mode = CUDNN_ACTIVATION_SIGMOID;
	else if (user_params->mode == RELU)
		mode = CUDNN_ACTIVATION_RELU;
	else if (user_params->mode == TANH)
		mode = CUDNN_ACTIVATION_TANH;
	else if (user_params->mode == CLIPPED_RELU)
		mode = CUDNN_ACTIVATION_CLIPPED_RELU;
	else if (user_params->mode == ELU)
		mode = CUDNN_ACTIVATION_ELU;

	checkCUDNN(cudnnCreateActivationDescriptor(&actv_desc));
	checkCUDNN(cudnnSetActivationDescriptor(actv_desc, mode, CUDNN_PROPAGATE_NAN, user_params->coef));

	output_size.N = batch_size, output_size.C = user_params->channels, output_size.H = user_params->h, output_size.W = user_params->w;
}

void ActivationLayerParams::allocateSpace(size_t &free_bytes) {
	
}

void SoftmaxLayerParams::initializeValues(SoftmaxDescriptor *user_params, cudnnDataType_t data_type,
											cudnnTensorFormat_t tensor_format, int batch_size, LayerDimension &output_size) {
	if (user_params->algo == SOFTMAX_FAST)
		algo = CUDNN_SOFTMAX_FAST;
	else if (user_params->algo == SOFTMAX_ACCURATE)
		algo = CUDNN_SOFTMAX_ACCURATE;

	if (user_params->mode == SOFTMAX_MODE_INSTANCE)
		mode = CUDNN_SOFTMAX_MODE_INSTANCE;
	else if (user_params->mode == SOFTMAX_MODE_CHANNEL) {
		mode = CUDNN_SOFTMAX_MODE_CHANNEL;
	}

	checkCUDNN(cudnnCreateTensorDescriptor(&input_tensor));
	checkCUDNN(cudnnSetTensor4dDescriptor(input_tensor, tensor_format, data_type, 
										batch_size, user_params->channels, user_params->h, user_params->w));

	output_size.N = batch_size, output_size.C = user_params->channels, output_size.H = user_params->h, output_size.W = user_params->w;	
}

void SoftmaxLayerParams::allocateSpace(size_t &free_bytes) {

}