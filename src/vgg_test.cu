#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

#include "solver.h"

using namespace std;

typedef unsigned char uchar;

int num_train = 128, num_test = 500;

int reverseInt(int n) {
	int bytes = 4;
	unsigned char ch[bytes];
	for (int i = 0; i < bytes; i++) {
		ch[i] = (n >> i * 8) & 255;
	}
	int p = 0;
	for (int i = 0; i < bytes; i++) {
		p += (int) ch[i] << (bytes - i - 1) * 8;
	}
	return p;
}

void readMNIST(vector<vector<uchar> > &train_images, vector<vector<uchar> > &test_images, vector<uchar> &train_labels, vector<uchar> &test_labels) {
	string filename_train_images = "data/train-images.idx3-ubyte";
	string filename_train_labels = "data/train-labels.idx1-ubyte";

	string filename_test_images = "data/t10k-images.idx3-ubyte";
	string filename_test_labels = "data/t10k-labels.idx1-ubyte";

	// read train/test images
	for (int i = 0; i < 2; i++) {
		string filename;
		if (i == 0)
			filename = filename_train_images;
		else
			filename = filename_test_images;

		ifstream f(filename.c_str(), ios::binary);
		if (!f.is_open())
			printf("Cannot read MNIST from %s\n", filename.c_str());

		// read metadata
		int magic_number = 0, n_images = 0, n_rows = 0, n_cols = 0;
		f.read((char *) &magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		f.read((char *) &n_images, sizeof(n_images));
		n_images = reverseInt(n_images);
		f.read((char *) &n_rows, sizeof(n_rows));
		n_rows = reverseInt(n_rows);
		f.read((char *) &n_cols, sizeof(n_cols));
		n_cols = reverseInt(n_cols);

		for (int k = 0; k < n_images; k++) {
			vector<uchar> temp;
			temp.reserve(n_rows * n_cols);
			for (int j = 0; j < n_rows * n_cols; j++) {
				uchar t = 0;
				f.read((char *)&t, sizeof(t));
				temp.push_back(t);
			}
			if (i == 0)
				train_images.push_back(temp);
			else
				test_images.push_back(temp);
		}
		f.close();

	}

	// read train/test labels
	for (int i = 0; i < 2; i++) {
		string filename;
		if (i == 0)
			filename = filename_train_labels;
		else
			filename = filename_test_labels;

		ifstream f(filename.c_str(), ios::binary);
		if (!f.is_open())
			printf("Cannot read MNIST from %s\n", filename.c_str());

		// read metadata
		int magic_number = 0, n_labels = 0;
		f.read((char *) &magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);
		f.read((char *) &n_labels, sizeof(n_labels));
		n_labels = reverseInt(n_labels);

		for (int k = 0; k < n_labels; k++) {
			uchar t = 0;
			f.read((char *)&t, sizeof(t));
			if (i == 0)
				train_labels.push_back(t);
			else
				test_labels.push_back(t);
		}

		f.close();

	}
}

void printTimes(vector<float> &time, string filename);
void printvDNNLag(vector<vector<float> > &fwd_vdnn_lag, vector<vector<float> > &bwd_vdnn_lag, string filename);
void printComputationTransferTimes(vector<vector<float> > &fwd_times, vector<vector<float> >&bwd_times, bool computation, string filename);

int main(int argc, char *argv[]) {

	
	float *f_train_images, *f_test_images;
	int *f_train_labels, *f_test_labels;
	int rows = 224, cols = 224, channels = 3;
	int input_size = rows * cols * channels;
	// f_train_images = (float *)malloc(num_train * input_size * sizeof(float));
	// f_train_labels = (int *)malloc(num_train * sizeof(int));
	checkCudaErrors(cudaMallocHost(&f_train_images, num_train * input_size * sizeof(float)));
	checkCudaErrors(cudaMallocHost(&f_train_labels, num_train * sizeof(int)));
	f_test_images = (float *)malloc(num_test * input_size * sizeof(float));
	f_test_labels = (int *)malloc(num_test * sizeof(int));

	float *mean_image;
	mean_image = (float *)malloc(input_size * sizeof(float));

	for (int i = 0; i < input_size; i++) {
		mean_image[i] = 0;
		for (int k = 0; k < num_train; k++) {
			mean_image[i] += f_train_images[k * input_size + i];
		}
		mean_image[i] /= num_train;
	}


	for (int i = 0; i < num_train; i++) {
		for (int j = 0; j < input_size; j++) {
			f_train_images[i * input_size + j] -= mean_image[j];
		}
	}

	for (int i = 0; i < num_test; i++) {
		for (int j = 0; j < input_size; j++) {
			f_test_images[i * input_size + j] -= mean_image[j];
		}

	}

	
	// VGG
	vector<LayerSpecifier> layer_specifier;
	{
		ConvDescriptor part0_conv0;
		part0_conv0.initializeValues(3, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part0_conv0;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part0_conv1;
		part0_conv1.initializeValues(64, 64, 3, 3, 224, 224, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part0_conv1;
		layer_specifier.push_back(temp);
	}
	{
		PoolingDescriptor pool0;
		pool0.initializeValues(64, 2, 2, 224, 224, 0, 0, 2, 2, POOLING_MAX);
		LayerSpecifier temp;
		temp.initPointer(POOLING);
		*((PoolingDescriptor *)temp.params) = pool0;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part1_conv0;
		part1_conv0.initializeValues(64, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part1_conv0;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part1_conv1;
		part1_conv1.initializeValues(128, 128, 3, 3, 112, 112, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part1_conv1;
		layer_specifier.push_back(temp);
	}
	{
		PoolingDescriptor pool1;
		pool1.initializeValues(128, 2, 2, 112, 112, 0, 0, 2, 2, POOLING_MAX);
		LayerSpecifier temp;
		temp.initPointer(POOLING);
		*((PoolingDescriptor *)temp.params) = pool1;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part2_conv0;
		part2_conv0.initializeValues(128, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part2_conv0;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part2_conv1;
		part2_conv1.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part2_conv1;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part2_conv2;
		part2_conv2.initializeValues(256, 256, 3, 3, 56, 56, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part2_conv2;
		layer_specifier.push_back(temp);
	}
	{
		PoolingDescriptor pool2;
		pool2.initializeValues(256, 2, 2, 56, 56, 0, 0, 2, 2, POOLING_MAX);
		LayerSpecifier temp;
		temp.initPointer(POOLING);
		*((PoolingDescriptor *)temp.params) = pool2;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part3_conv0;
		part3_conv0.initializeValues(256, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part3_conv0;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part3_conv1;
		part3_conv1.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part3_conv1;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part3_conv2;
		part3_conv2.initializeValues(512, 512, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part3_conv2;
		layer_specifier.push_back(temp);
	}
	{
		PoolingDescriptor pool3;
		pool3.initializeValues(512, 2, 2, 28, 28, 0, 0, 2, 2, POOLING_MAX);
		LayerSpecifier temp;
		temp.initPointer(POOLING);
		*((PoolingDescriptor *)temp.params) = pool3;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part4_conv0;
		part4_conv0.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part4_conv0;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part4_conv1;
		part4_conv1.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part4_conv1;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor part4_conv2;
		part4_conv2.initializeValues(512, 512, 3, 3, 14, 14, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = part4_conv2;
		layer_specifier.push_back(temp);
	}
	{
		PoolingDescriptor pool3;
		pool3.initializeValues(512, 2, 2, 14, 14, 0, 0, 2, 2, POOLING_MAX);
		LayerSpecifier temp;
		temp.initPointer(POOLING);
		*((PoolingDescriptor *)temp.params) = pool3;
		layer_specifier.push_back(temp);
	}
	
	{
		FCDescriptor part5_fc0;
		part5_fc0.initializeValues(7 * 7 * 512, 4096, RELU);
		LayerSpecifier temp;
		temp.initPointer(FULLY_CONNECTED);
		*((FCDescriptor *)temp.params) = part5_fc0;
		layer_specifier.push_back(temp);
	}
	{
		FCDescriptor part5_fc1;
		part5_fc1.initializeValues(4096, 4096, RELU);
		LayerSpecifier temp;
		temp.initPointer(FULLY_CONNECTED);
		*((FCDescriptor *)temp.params) = part5_fc1;
		layer_specifier.push_back(temp);
	}
	{
		FCDescriptor part5_fc2;
		part5_fc2.initializeValues(4096, 1000);
		LayerSpecifier temp;
		temp.initPointer(FULLY_CONNECTED);
		*((FCDescriptor *)temp.params) = part5_fc2;
		layer_specifier.push_back(temp);
	}
	{
		SoftmaxDescriptor s_max;
		s_max.initializeValues(SOFTMAX_ACCURATE, SOFTMAX_MODE_INSTANCE, 1000, 1, 1);
		LayerSpecifier temp;
		temp.initPointer(SOFTMAX);
		*((SoftmaxDescriptor *)temp.params) = s_max;
		layer_specifier.push_back(temp);
	}

	vDNNConvAlgo vdnn_conv_algo = vDNN_PERFORMANCE_OPTIMAL;
	vDNNType vdnn_type = vDNN_DYN;
	string filename("vdnn_dyn");
	if (argc == 3) {
		filename.assign("vdnn");
		// argv[1] - layers to offload, argv[2] - conv algo to use
		if (strcmp(argv[1], "dyn") == 0) {
			vdnn_type = vDNN_DYN;
			filename.append("_dyn");
		}
		else if (strcmp(argv[1], "conv") == 0) {
			vdnn_type = vDNN_CONV;
			filename.append("_conv");
		}
		else if (strcmp(argv[1], "all") == 0) {
			vdnn_type = vDNN_ALL;
			filename.append("_all");
		}
		else if (strcmp(argv[1], "alternate_conv") == 0) {
			vdnn_type = vDNN_ALTERNATE_CONV;
			filename.append("_alternate_conv");
		}
		else {
			printf("invalid argument.. using vdnn dynamic\n");
			filename.assign("vdnn_dyn");
		}
		if ((strcmp(argv[1], "conv") == 0 or strcmp(argv[1], "all") == 0 or strcmp(argv[1], "alternate_conv") == 0)) {
			if (strcmp(argv[2], "p") == 0) {
				vdnn_conv_algo = vDNN_PERFORMANCE_OPTIMAL;
				filename.append("_p");
			}
			else if (strcmp(argv[2], "m") == 0) {
				vdnn_conv_algo = vDNN_MEMORY_OPTIMAL;
				filename.append("_m");
			}
			else {
				printf("invalid argument.. using vdnn dynamic\n");
				filename.assign("vdnn_dyn");
			}
		}
	}

	int batch_size = 64;
	long long dropout_seed = 1;
	float softmax_eps = 1e-8;
	float init_std_dev = 0.1;
	NeuralNet net(layer_specifier, DATA_FLOAT, batch_size, TENSOR_NCHW, dropout_seed, softmax_eps, init_std_dev, vdnn_type, vdnn_conv_algo, SGD);

	int num_epoch = 1000;
	double learning_rate = 1e-3;
	double learning_rate_decay = 0.9;
	
	Solver solver(&net, (void *)f_train_images, f_train_labels, (void *)f_train_images, f_train_labels, num_epoch, SGD, learning_rate, learning_rate_decay, num_train, num_train);
	vector<float> loss;
	vector<float> time;
	vector<vector<float> > fwd_vdnn_lag, bwd_vdnn_lag;
	solver.getTrainTime(loss, time, 100, fwd_vdnn_lag, bwd_vdnn_lag);
	printTimes(time, filename);
	printvDNNLag(fwd_vdnn_lag, bwd_vdnn_lag, filename);

	vector<vector<float> > fwd_computation_time, bwd_computation_time;
	solver.getComputationTime(1, fwd_computation_time, bwd_computation_time);

	vector<vector<float> > fwd_transfer_time, bwd_transfer_time;
	solver.getTransferTime(1, fwd_transfer_time, bwd_transfer_time);

	printComputationTransferTimes(fwd_computation_time, bwd_computation_time, true, filename);
	printComputationTransferTimes(fwd_transfer_time, bwd_transfer_time, false, filename);

}

void printTimes(vector<float> &time, string filename) {
	float mean_time = 0.0;
	float std_dev = 0.0;
	int N = time.size();
	for (int i = 0; i < N; i++) {
		mean_time += time[i];
	}
	mean_time /= N;
	for (int i = 0; i < N; i++) {
		std_dev += pow(time[i] - mean_time, 2);
	}
	std_dev /= N;
	std_dev = pow(std_dev, 0.5);
	cout << "Average time: " << mean_time << endl;
	cout << "Standard deviation: " << std_dev << endl;

	filename.append(".dat");
	fstream f;
	f.open(filename.c_str(), ios_base::out);

	for (int i = 0; i < N; i++) {
		f << time[i] << endl;
	}
	f << "mean_time: " << mean_time << endl;
	f << "standard_deviation: " << std_dev << endl;
	f.close();

	filename.append(".bin");
	fstream f_bin;
	f_bin.open(filename.c_str(), ios_base::out);
	f_bin.write((char *)&N, sizeof(N));
	for (int i = 0; i < N; i++) {
		f_bin.write((char *)&time[i], sizeof(time[i]));
	}
	f_bin.close();

}

void printvDNNLag(vector<vector<float> > &fwd_vdnn_lag, vector<vector<float> > &bwd_vdnn_lag, string filename) {
	filename.append("_lag.dat");
	
	fstream f;
	f.open(filename.c_str(), ios_base::out);

	int N = fwd_vdnn_lag.size();
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < fwd_vdnn_lag[i].size(); j++) {
			f << "fwd" << j << ": " << fwd_vdnn_lag[i][j] << endl;
		}
		for (int j = 0; j < bwd_vdnn_lag[i].size(); j++) {
			f << "bwd" << j << ": " << bwd_vdnn_lag[i][j] << endl;
		}
		f << endl;
	}
	f.close();
}

void printComputationTransferTimes(vector<vector<float> > &fwd_times, vector<vector<float> >&bwd_times, bool computation, string filename) {
	if (computation)
		filename.append("_compute_time.dat");
	else
		filename.append("_transfer_time.dat");

	fstream f;
	f.open(filename.c_str(), ios_base::out);

	int N = fwd_times.size();
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < fwd_times[i].size(); j++) {
			f << "fwd" << j << ": " << fwd_times[i][j] << endl;
		}
		for (int j = 0; j < bwd_times[i].size(); j++) {
			f << "bwd" << j << ": " << bwd_times[i][j] << endl;
		}
		f << endl;
	}
	f.close();	
}