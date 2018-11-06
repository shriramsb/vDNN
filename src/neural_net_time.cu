#include "neural_net.h"

void NeuralNet::getComputationTime(void *X, int *y, double learning_rate, 
									std::vector<float> &fwd_computation_time, std::vector<float> &bwd_computation_time) {
	for (int i = 0; i < num_layers; i++)
		prefetched[i] = false;

	// checkCNMEM(cnmemMalloc(&layer_input[0], layer_input_size[0] * data_type_size, NULL));
	// checkCudaErrors(cudaMemcpy(layer_input[0], X, batch_size * input_channels * input_h * input_w * data_type_size, cudaMemcpyHostToDevice));
	// checkCudaErrors(cudaMemcpy(this->y, y, batch_size * data_type_size, cudaMemcpyHostToDevice));
	
	float alpha = 1.0, beta = 0.0;
	float Salpha = 1.0, Sbeta = 0.0;
	double Dalpha = 1.0, Dbeta = 0.0;

	// forward propagate
	for (int i = 0; i < num_layers; i++) {
		size_t cur_workspace_size;
		void *cur_workspace;

		checkCNMEM(cnmemMalloc(&layer_input[i], layer_input_size[i] * data_type_size, NULL));
		checkCNMEM(cnmemMalloc(&layer_input[i + 1], layer_input_size[i + 1] * data_type_size, NULL));
		if (layer_type[i] == CONV) {
			ConvLayerParams *cur_params = (ConvLayerParams *)params[i];

			cur_workspace_size = cur_params->fwd_workspace_size;
			checkCNMEM(cnmemMalloc(&cur_workspace, cur_workspace_size, NULL));
		}

		checkCudaErrors(cudaEventRecord(start_compute, stream_compute));
		if (layer_type[i] == CONV) {
			ConvLayerParams *cur_params = (ConvLayerParams *)params[i];

			cur_workspace_size = cur_params->fwd_workspace_size;			
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
		}
		else if (layer_type[i] == DROPOUT) {
			DropoutLayerParams *cur_params = (DropoutLayerParams *)params[i];
			checkCUDNN(cudnnDropoutForward(cudnn_handle, cur_params->dropout_desc,
											cur_params->input_tensor, layer_input[i],
											cur_params->input_tensor, layer_input[i + 1],
											cur_params->reserved_space,
											cur_params->reserved_space_size));
		}
		else if (layer_type[i] == BATCHNORM) {
			BatchNormLayerParams *cur_params = (BatchNormLayerParams *)params[i];

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
		else if (layer_type[i] == POOLING) {
			PoolingLayerParams *cur_params = (PoolingLayerParams *)params[i];
			checkCUDNN(cudnnPoolingForward(cudnn_handle, cur_params->pool_desc,
											&alpha,
											cur_params->input_tensor, layer_input[i],
											&beta,
											cur_params->output_tensor, layer_input[i + 1]));
		}
		else if (layer_type[i] == ACTV) {
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
			SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[i];
			checkCUDNN(cudnnSoftmaxForward(cudnn_handle, cur_params->algo, cur_params->mode,
											&alpha,
											cur_params->input_tensor, layer_input[i],
											&beta,
											cur_params->input_tensor, layer_input[i + 1]));
		}

		// ---------------------- vDNN start ----------------------
		// synchronization
		// checkCudaErrors(cudaDeviceSynchronize());

		// if next layer is ACTV or SOFTMAX, complete that and come to synchronization
		// the case in above if for ACTV and SOFTMAX never occurs
		if (layer_type[i + 1] == SOFTMAX) {
			i++;
			layer_input[i + 1] = layer_input[i];
			SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[i];
			checkCUDNN(cudnnSoftmaxForward(cudnn_handle, cur_params->algo, cur_params->mode,
											&alpha,
											cur_params->input_tensor, layer_input[i],
											&beta,
											cur_params->input_tensor, layer_input[i + 1]));
			i--;
		}

		// sync with stream_compute guaranteed
		checkCudaErrors(cudaEventRecord(stop_compute, stream_compute));
		checkCudaErrors(cudaEventSynchronize(stop_compute));
		float compute_time = 0;
		checkCudaErrors(cudaEventElapsedTime(&compute_time, start_compute, stop_compute));
		
		fwd_computation_time.push_back(compute_time);
		
		if (layer_type[i] == CONV) {
			checkCNMEM(cnmemFree(cur_workspace, NULL));
		}

		checkCNMEM(cnmemFree(layer_input[i], NULL));
		checkCNMEM(cnmemFree(layer_input[i + 1], NULL));

		if (layer_type[i + 1] == ACTV or layer_type[i + 1] == SOFTMAX) {
			i = i + 1;
		}

		// ---------------------- vDNN end ------------------------
	}

	// time for loss compute ignored
	// *scalar_loss = computeLoss();

	// time for softmax backward ignored
	// ---------------------- vDNN start ----------------------
	// checkCNMEM(cnmemMalloc(&dlayer_input[num_layers], batch_size * num_classes * data_type_size, NULL));
	// space_tracker.updateSpace(CnmemSpace::SUB, layer_input_size[num_layers] * data_type_size);
	// // std::cout << "Free bytes: " << free_bytes << std::endl;
	// // ---------------------- vDNN end ------------------------
	// if (layer_type[num_layers - 1] == SOFTMAX) {
	// 	// SoftmaxLayerParams *cur_params = (SoftmaxLayerParams *)params[num_layers - 1];
	// 	if (data_type == CUDNN_DATA_FLOAT) {
	// 		checkCudaErrors(cudaMemset(dlayer_input[num_layers], 0, batch_size * num_classes * sizeof(float)));
	// 		softmaxLossBackProp<float><<<ceil(1.0 * batch_size / BW), BW>>>(this->y, (float *)layer_input[num_layers], 
	// 																		(float *)dlayer_input[num_layers], batch_size, num_classes, softmax_eps);
	// 	}
	// 	else if (data_type == CUDNN_DATA_DOUBLE) {
	// 		checkCudaErrors(cudaMemset(dlayer_input[num_layers], 0, batch_size * num_classes * sizeof(double)));
	// 		softmaxLossBackProp<double><<<ceil(1.0 * batch_size / BW), BW>>>(this->y, (double *)layer_input[num_layers], 
	// 																		(double *)dlayer_input[num_layers], batch_size, num_classes, softmax_eps);
	// 	}
	// }

	for (int i = num_layers - 1; i >= 0; i--) {
		// ---------------------- vDNN start ----------------------
		size_t cur_filter_workspace_size, cur_data_workspace_size, cur_workspace_size;
		void *cur_workspace;

		checkCNMEM(cnmemMalloc(&layer_input[i + 1], layer_input_size[i + 1] * data_type_size, NULL));
		checkCNMEM(cnmemMalloc(&layer_input[i], layer_input_size[i] * data_type_size, NULL));
		checkCNMEM(cnmemMalloc(&dlayer_input[i + 1], layer_input_size[i] * data_type_size, NULL));

		if (i > 0) {
			if (layer_type[i] == ACTV or layer_type[i] == SOFTMAX) {
				dlayer_input[i] = dlayer_input[i + 1];
			}
			else {
				checkCNMEM(cnmemMalloc(&dlayer_input[i], layer_input_size[i] * data_type_size, NULL));
			}
		}
		// ---------------------- vDNN end ------------------------

		if (layer_type[i] == CONV) {
			ConvLayerParams *cur_params = (ConvLayerParams *)params[i];

			// allocate space for derivative
			if (!pre_alloc_conv_derivative) {
				cur_params->cnmemAllocDerivatives(data_type_size, NULL);
			}

			cur_filter_workspace_size = cur_params->bwd_filter_workspace_size;
			if (i > 0)
				cur_data_workspace_size = cur_params->bwd_data_workspace_size;
			else
				cur_data_workspace_size = 0;
			// std::cout << "bwd cur_workspace_size: " << cur_workspace_size << std::endl;
			cur_workspace_size = (cur_filter_workspace_size > cur_data_workspace_size) ? cur_filter_workspace_size : cur_data_workspace_size;
			checkCNMEM(cnmemMalloc(&cur_workspace, cur_workspace_size, NULL));

		}

		else if (layer_type[i] == FULLY_CONNECTED) {
			FCLayerParams *cur_params = (FCLayerParams *)params[i];

			if (!pre_alloc_fc_derivative) {
				cur_params->cnmemAllocDerivatives(data_type_size, NULL);
			}
		}

		else if (layer_type[i] == BATCHNORM) {
			BatchNormLayerParams *cur_params = (BatchNormLayerParams *)params[i];

			if (!pre_alloc_batch_norm_derivative) {
				cur_params->cnmemAllocDerivatives(data_type_size, NULL);
			}
		}


		if (!(i + 1 < num_layers && layer_type[i + 1] == SOFTMAX))
			checkCudaErrors(cudaEventRecord(start_compute, stream_compute));

		if (layer_type[i] == CONV) {
			ConvLayerParams *cur_params = (ConvLayerParams *)params[i];

			if (cur_params->activation_mode != ACTIVATION_NONE) {
				checkCUDNN(cudnnActivationBackward(cudnn_handle, cur_params->actv_desc, &alpha,
												cur_params->output_tensor, layer_input[i + 1],
												cur_params->output_tensor, dlayer_input[i + 1],
												cur_params->output_tensor, layer_input[i + 1],
												&beta,
												cur_params->output_tensor, dlayer_input[i + 1]));
			}

			cur_filter_workspace_size = cur_params->bwd_filter_workspace_size;
			if (i > 0)
				cur_data_workspace_size = cur_params->bwd_data_workspace_size;
			else
				cur_data_workspace_size = 0;
			// std::cout << "bwd cur_workspace_size: " << cur_workspace_size << std::endl;
			cur_workspace_size = (cur_filter_workspace_size > cur_data_workspace_size) ? cur_filter_workspace_size : cur_data_workspace_size;

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

		checkCudaErrors(cudaEventRecord(stop_compute, stream_compute));
		checkCudaErrors(cudaEventSynchronize(stop_compute));
		float compute_time;
		checkCudaErrors(cudaEventElapsedTime(&compute_time, start_compute, stop_compute));

		bwd_computation_time.insert(bwd_computation_time.begin(), compute_time);

		if (layer_type[i] == CONV) {
			checkCNMEM(cnmemFree(cur_workspace, NULL));
			if (!pre_alloc_conv_derivative) {
				ConvLayerParams *cur_params = (ConvLayerParams *)params[i];
				cur_params->cnmemFreeDerivatives(NULL);
			}
		}
		else if (layer_type[i] == FULLY_CONNECTED) {
			if (!pre_alloc_fc_derivative) {
				FCLayerParams *cur_params = (FCLayerParams *)params[i];
				cur_params->cnmemFreeDerivatives(NULL);
			}
		}
		else if (layer_type[i] == BATCHNORM) {
			if (!pre_alloc_batch_norm_derivative) {
				BatchNormLayerParams *cur_params = (BatchNormLayerParams *)params[i];
				cur_params->cnmemFreeDerivatives(NULL);
			}
		}

		checkCNMEM(cnmemFree(layer_input[i + 1], NULL));
		checkCNMEM(cnmemFree(dlayer_input[i + 1], NULL));
		checkCNMEM(cnmemFree(layer_input[i], NULL));
		if (i > 0 && layer_type[i] != SOFTMAX)
			checkCNMEM(cnmemFree(dlayer_input[i], NULL));
	}
}


void NeuralNet::getTransferTime(void *X, int *y, double learning_rate, std::vector<float> &fwd_transfer_time, std::vector<float> &bwd_transfer_time) {
	for (int i = 0; i < num_layers; i++) {
		if (layer_type[i] == SOFTMAX)
			continue;

		void *device_data;
		void *host_data;

		checkCNMEM(cnmemMalloc(&device_data, layer_input_size[i] * data_type_size, NULL));
		checkCudaErrors(cudaMallocHost(&host_data, layer_input_size[i] * data_type_size));

		checkCudaErrors(cudaEventRecord(start_transfer, stream_memory));

		checkCudaErrors(cudaMemcpyAsync(host_data, device_data, layer_input_size[i] * data_type_size, cudaMemcpyDeviceToHost, stream_memory));

		checkCudaErrors(cudaEventRecord(stop_transfer, stream_memory));
		checkCudaErrors(cudaEventSynchronize(stop_transfer));
		float transfer_time;
		checkCudaErrors(cudaEventElapsedTime(&transfer_time, start_transfer, stop_transfer));
		fwd_transfer_time.push_back(transfer_time);

		checkCudaErrors(cudaEventRecord(start_transfer, stream_memory));

		checkCudaErrors(cudaMemcpyAsync(device_data, host_data, layer_input_size[i] * data_type_size, cudaMemcpyHostToDevice, stream_memory));

		checkCudaErrors(cudaEventRecord(stop_transfer, stream_memory));
		checkCudaErrors(cudaEventSynchronize(stop_transfer));
		checkCudaErrors(cudaEventElapsedTime(&transfer_time, start_transfer, stop_transfer));
		bwd_transfer_time.push_back(transfer_time);
	}
}