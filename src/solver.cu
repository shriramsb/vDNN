#include "solver.h"

Solver::Solver(NeuralNet *model, void *X_train, int *y_train, void *X_val, int *y_val, int num_epoch, UpdateRule update_rule, 
					double learning_rate, double learning_rate_decay, int num_train, int num_val) {
	this->model = model;
	this->X_train = X_train, this->X_val = X_val;
	this->y_train = y_train, this->y_val = y_val;
	this->num_epoch = num_epoch;
	this->update_rule = update_rule;
	this->learning_rate = learning_rate, this->learning_rate_decay = learning_rate_decay;

	this->num_train = num_train, this->num_val = num_val;
	this->num_features = model->input_channels * model->input_h * model->input_w;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	

}

float Solver::step(int start_X, int start_y) {
	std::vector<float> t1, t2;
	return this->step(start_X, start_y, t1, t2);
}

float Solver::step(int start_X, int start_y, std::vector<float> &fwd_vdnn_lag, std::vector<float> &bwd_vdnn_lag) {
	float temp_loss;
	// std::cout << "start_X: " << start_X << std::endl;
	if (model->data_type == CUDNN_DATA_FLOAT)
		model->getLoss(&(((float *)X_train)[start_X]), &y_train[start_y], learning_rate, fwd_vdnn_lag, bwd_vdnn_lag, true, NULL, &temp_loss);
	else if (model->data_type == CUDNN_DATA_DOUBLE)
		model->getLoss(&(((double *)X_train)[start_X]), &y_train[start_y], learning_rate, fwd_vdnn_lag, bwd_vdnn_lag, true, NULL, &temp_loss);

	// float Salpha = -learning_rate;
	// double Dalpha = -learning_rate;
	// if (update_rule == SGD) {
	// 	for (int i = 0; i < model->num_layers; i++) {
	// 		if (model->layer_type[i] == CONV) {
	// 			ConvLayerParams *cur_params = (ConvLayerParams *)model->params[i];
	// 			int kernel_size = cur_params->C_in * cur_params->C_out * cur_params->filter_h * cur_params->filter_w;
	// 			if (model->data_type == CUDNN_DATA_FLOAT) {
	// 				checkCUBLAS(cublasSaxpy(model->cublas_handle, kernel_size,
	// 										&Salpha,
	// 										(float *)cur_params->dW, 1,
	// 										(float *)cur_params->W, 1));

	// 				checkCUBLAS(cublasSaxpy(model->cublas_handle, cur_params->C_out,
	// 										&Salpha,
	// 										(float *)cur_params->db, 1,
	// 										(float *)cur_params->b, 1));
	// 			}
	// 			else if (model->data_type == CUDNN_DATA_DOUBLE) {
	// 				checkCUBLAS(cublasDaxpy(model->cublas_handle, kernel_size,
	// 										&Dalpha,
	// 										(double *)cur_params->dW, 1,
	// 										(double *)cur_params->W, 1));

	// 				checkCUBLAS(cublasDaxpy(model->cublas_handle, cur_params->C_out,
	// 										&Dalpha,
	// 										(double *)cur_params->db, 1,
	// 										(double *)cur_params->b, 1));
	// 			}

	// 		}

	// 		else if (model->layer_type[i] == FULLY_CONNECTED) {
	// 			FCLayerParams *cur_params = (FCLayerParams *)model->params[i];
	// 			if (model->data_type == CUDNN_DATA_FLOAT) {
	// 				checkCUBLAS(cublasSaxpy(model->cublas_handle, cur_params->C_in * cur_params->C_out,
	// 										&Salpha,
	// 										(float *)cur_params->dW, 1,
	// 										(float *)cur_params->W, 1));

	// 				checkCUBLAS(cublasSaxpy(model->cublas_handle, cur_params->C_out,
	// 										&Salpha,
	// 										(float *)cur_params->db, 1,
	// 										(float *)cur_params->b, 1));
	// 			}
	// 			else if (model->data_type == CUDNN_DATA_DOUBLE) {
	// 				checkCUBLAS(cublasDaxpy(model->cublas_handle, cur_params->C_in * cur_params->C_out,
	// 										&Dalpha,
	// 										(double *)cur_params->dW, 1,
	// 										(double *)cur_params->W, 1));

	// 				checkCUBLAS(cublasDaxpy(model->cublas_handle, cur_params->C_out,
	// 										&Dalpha,
	// 										(double *)cur_params->db, 1,
	// 										(double *)cur_params->b, 1));
	// 			}
	// 		}

	// 		else if (model->layer_type[i] == BATCHNORM) {
	// 			BatchNormLayerParams *cur_params = (BatchNormLayerParams *)model->params[i];
	// 			if (model->data_type == CUDNN_DATA_FLOAT) {
	// 				checkCUBLAS(cublasSaxpy(model->cublas_handle, cur_params->sbmv_size,
	// 										&Salpha,
	// 										(float *)cur_params->dscale, 1,
	// 										(float *)cur_params->scale, 1));
	// 				checkCUBLAS(cublasSaxpy(model->cublas_handle, cur_params->sbmv_size,
	// 										&Salpha,
	// 										(float *)cur_params->dbias, 1,
	// 										(float *)cur_params->bias, 1));

	// 			}
	// 			else if (model->data_type == CUDNN_DATA_DOUBLE) {
	// 				checkCUBLAS(cublasDaxpy(model->cublas_handle, cur_params->sbmv_size,
	// 										&Dalpha,
	// 										(double *)cur_params->dscale, 1,
	// 										(double *)cur_params->scale, 1));
	// 				checkCUBLAS(cublasDaxpy(model->cublas_handle, cur_params->sbmv_size,
	// 										&Dalpha,
	// 										(double *)cur_params->dbias, 1,
	// 										(double *)cur_params->bias, 1));

	// 			}
	// 		}
	// 	}
	// }
	checkCudaErrors(cudaDeviceSynchronize());
	return temp_loss;

}

void Solver::train(std::vector<float> &loss, std::vector<int> &val_acc) {

	int batch_size = model->batch_size;
	int num_train_batches = num_train / model->batch_size;
	int num_val_batches = num_val / model->batch_size;
	for (int i = 0; i < num_epoch; i++) {
		for (int j = 0; j < num_train_batches; j++) {
			int start_sample = j * num_features * batch_size;

			float milli = 0;
			checkCudaErrors(cudaEventRecord(start, model->stream_compute));

			float temp_loss = step(start_sample, j * batch_size);

			checkCudaErrors(cudaEventRecord(stop, model->stream_compute));
			checkCudaErrors(cudaEventSynchronize(stop));
			checkCudaErrors(cudaEventElapsedTime(&milli, start, stop));
			std::cout << "One forward, backward pass time(ms): " << milli << std::endl;
			
			loss.push_back(temp_loss);
			std::cout << "loss: " << temp_loss << std::endl;
		}
		int correct_count = 0;
		for (int j = 0; j < num_val_batches; j++) {
			
			int start_sample = j * num_features * batch_size;
			int temp_correct_count;
			if (model->data_type == CUDNN_DATA_FLOAT)
				model->getLoss(&(((float *)X_val)[start_sample]), &y_val[j * batch_size], learning_rate, false, &temp_correct_count, NULL);
			else if (model->data_type == CUDNN_DATA_DOUBLE)
				model->getLoss(&(((double *)X_val)[start_sample]), &y_val[j * batch_size], learning_rate, false, &temp_correct_count, NULL);
			correct_count += temp_correct_count;
		}
		val_acc.push_back(correct_count);
		std::cout << "val_acc: " << val_acc[i] << std::endl;
		// learning_rate *= learning_rate_decay;
		// std::cout << "learning_rate: " << learning_rate << std::endl;
	}
	learning_rate *= learning_rate_decay;
	
}

void Solver::checkAccuracy(void *X, int *y, int num_samples, int *num_correct) {
	int batch_size = model->batch_size;
	int num_iter = num_samples / batch_size;
	*num_correct = 0;
	for (int i = 0; i < num_iter; i++) {
		int start_sample = i * num_features * batch_size;
		int temp_correct_count;
		if (model->data_type == CUDNN_DATA_FLOAT)
				model->getLoss(&(((float *)X)[start_sample]), &y[i * batch_size], learning_rate, false, &temp_correct_count, NULL);
			else if (model->data_type == CUDNN_DATA_DOUBLE)
				model->getLoss(&(((double *)X)[start_sample]), &y[i * batch_size], learning_rate, false, &temp_correct_count, NULL);
		*num_correct = *num_correct + temp_correct_count;
	}
}

void Solver::getTrainTime(std::vector<float> &loss, std::vector<float> &time, int num_epoch, 
							std::vector<std::vector<float> > &fwd_vdnn_lag, std::vector<std::vector<float> > &bwd_vdnn_lag) {
	int batch_size = model->batch_size;
	int num_train_batches = num_train / model->batch_size;
	for (int i = 0; i < num_epoch; i++) {
		for (int j = 0; j < num_train_batches; j++) {
			int start_sample = j * num_features * batch_size;

			checkCudaErrors(cudaEventRecord(start));
			float milli;

			std::vector<float> cur_fwd_vdnn_lag, cur_bwd_vdnn_lag; 
			float temp_loss = step(start_sample, j * batch_size, cur_fwd_vdnn_lag, cur_bwd_vdnn_lag);

			checkCudaErrors(cudaEventRecord(stop));
			checkCudaErrors(cudaEventSynchronize(stop));
			checkCudaErrors(cudaEventElapsedTime(&milli, start, stop));
			// std::cout << "One forward, backward pass time(ms): " << milli << std::endl;
			
			fwd_vdnn_lag.push_back(cur_fwd_vdnn_lag);
			bwd_vdnn_lag.push_back(cur_bwd_vdnn_lag);
			
			loss.push_back(temp_loss);
			time.push_back(milli);
			// std::cout << "loss: " << temp_loss << std::endl;
			// for (int i = 0; i < cur_fwd_vdnn_lag.size(); i++) {
			// 	std::cout << "fwd_lag " << i << ":" << cur_fwd_vdnn_lag[i] << std::endl; 
			// }
			// for (int i = 0; i < cur_bwd_vdnn_lag.size(); i++) {
			// 	std::cout << "bwd_lag " << i << ":" << cur_bwd_vdnn_lag[i] << std::endl; 
			// }
		}
	}
	learning_rate *= learning_rate_decay;
}

void Solver::getComputationTime(long num_epoch, std::vector<std::vector<float> > &fwd_computation_time, std::vector<std::vector<float> > &bwd_computation_time) {
	int batch_size = model->batch_size;
	int num_train_batches = num_train / model->batch_size;
	for (int i = 0; i < num_epoch; i++) {
		for (int j = 0; j < num_train_batches; j++) {
			int start_sample = j * num_features * batch_size;

			float milli;

			std::vector<float> cur_fwd_computation_time, cur_bwd_computation_time; 
			stepComputationTime(start_sample, j * batch_size, cur_fwd_computation_time, cur_bwd_computation_time);
			
			fwd_computation_time.push_back(cur_fwd_computation_time);
			bwd_computation_time.push_back(cur_bwd_computation_time);
			
		}
		learning_rate *= learning_rate_decay;	
	}
}

void Solver::getTransferTime(long num_epoch, std::vector<std::vector<float> > &fwd_transfer_time, std::vector<std::vector<float> > &bwd_transfer_time) {
	int batch_size = model->batch_size;
	int num_train_batches = num_train / model->batch_size;
	for (int i = 0; i < num_epoch; i++) {
		for (int j = 0; j < num_train_batches; j++) {
			int start_sample = j * num_features * batch_size;

			float milli;

			std::vector<float> cur_fwd_transfer_time, cur_bwd_transfer_time; 
			stepTransferTime(start_sample, j * batch_size, cur_fwd_transfer_time, cur_bwd_transfer_time);
			
			fwd_transfer_time.push_back(cur_fwd_transfer_time);
			bwd_transfer_time.push_back(cur_bwd_transfer_time);
			
		}
		learning_rate *= learning_rate_decay;	
	}
}

void Solver::stepComputationTime(int start_X, int start_y, std::vector<float> &fwd_computation_time, std::vector<float> &bwd_computation_time) {
	if (model->data_type == CUDNN_DATA_FLOAT)
		model->getComputationTime(&(((float *)X_train)[start_X]), &y_train[start_y], learning_rate, fwd_computation_time, bwd_computation_time);
	else if (model->data_type == CUDNN_DATA_DOUBLE)
		model->getComputationTime(&(((double *)X_train)[start_X]), &y_train[start_y], learning_rate, fwd_computation_time, bwd_computation_time);
}

void Solver::stepTransferTime(int start_X, int start_y, std::vector<float> &fwd_transfer_time, std::vector<float> &bwd_transfer_time) {
	if (model->data_type == CUDNN_DATA_FLOAT)
		model->getTransferTime(&(((float *)X_train)[start_X]), &y_train[start_y], learning_rate, fwd_transfer_time, bwd_transfer_time);
	else if (model->data_type == CUDNN_DATA_DOUBLE)
		model->getTransferTime(&(((double *)X_train)[start_X]), &y_train[start_y], learning_rate, fwd_transfer_time, bwd_transfer_time);
}
