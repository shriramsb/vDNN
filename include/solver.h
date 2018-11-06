#include "neural_net.h"

class Solver {
public:
	NeuralNet *model;
	void *X_train, *X_val;
	int *y_train, *y_val;
	int num_epoch;
	UpdateRule update_rule;
	double learning_rate, learning_rate_decay;
	int num_train, num_val;
	int num_train_batches;
	int num_features;
	cudaEvent_t start, stop;

	Solver(NeuralNet *model, void *X_train, int *y_train, void *X_val, int *y_val, int num_epoch, UpdateRule update_rule, 
			double learning_rate, double learning_rate_decay, int num_train, int num_val);
	void train(std::vector<float> &loss, std::vector<int> &val_acc);
	float step(int start_X, int start_y, std::vector<float> &fwd_vdnn_lag, std::vector<float> &bwd_vdnn_lag);
	float step(int start_X, int start_y);
	void checkAccuracy(void *X, int *y, int num_samples, int *num_correct);

	void getTrainTime(std::vector<float> &loss, std::vector<float> &time, int num_epoch, 
						std::vector<std::vector<float> > &fwd_vdnn_lag, std::vector<std::vector<float> > &bwd_vdnn_lag);

	void getComputationTime(long num_epoch, std::vector<std::vector<float> > &fwd_computation_time, std::vector<std::vector<float> > &bwd_computation_time);
	void stepComputationTime(int start_X, int start_y, std::vector<float> &fwd_computation_time, std::vector<float> &bwd_computation_time);

	void getTransferTime(long num_epoch, std::vector<std::vector<float> > &fwd_transfer_time, std::vector<std::vector<float> > &bwd_transfer_time);
	void stepTransferTime(int start_X, int start_y, std::vector<float> &fwd_transfer_time, std::vector<float> &bwd_transfer_time);

};