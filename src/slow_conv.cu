#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

#include "solver.h"

using namespace std;

typedef unsigned char uchar;

int num_train = 1000, num_test = 500;

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

int main(int argc, char *argv[]) {

	// int num_train = 100 * batch_size, num_val = batch_size;
	// void *X_train = malloc(num_train * input_channels * sizeof(float));
	// int *y_train = (int *)malloc(num_train * sizeof(int));
	// void *X_val = malloc(num_val * input_channels * sizeof(float));
	// int *y_val = (int *)malloc(num_val * sizeof(int));
	// for (int i = 0; i < num_train; i++) {
	// 	for (int j = 0; j < input_channels; j++)
	// 		((float *)X_train)[i * input_channels + j] = (rand() % 1000) * 1.0 / 1000;
	// 	y_train[i] = 0;
	// }

	// for (int i = 0; i < num_val; i++) {
	// 	for (int j = 0; j < input_channels; j++)
	// 		((float *)X_val)[i * input_channels + j] = (rand() % 1000) * 1.0 / 1000;
	// 	y_val[i] = rand() % 2;
	// }
	
	// int rows = 28, cols = 28, channels = 1;
	// vector<vector<uchar> > train_images, test_images;
	// vector<uchar> train_labels, test_labels;
	// readMNIST(train_images, test_images, train_labels, test_labels);
	// float *f_train_images, *f_train_labels, *f_test_images, *f_test_labels;
	float *f_train_images, *f_test_images;
	int *f_train_labels, *f_test_labels;
	int rows = 156, cols = 156, channels = 3;
	int input_size = rows * cols * channels;
	f_train_images = (float *)malloc(num_train * input_size * sizeof(float));
	f_train_labels = (int *)malloc(num_train * sizeof(int));
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

	// int input_channels = rows * cols * channels * 3, hidden_channels1 = 50, hidden_channels2 = 100, output_channels = 10;
	// vector<LayerSpecifier> layer_specifier;
	// ConvDescriptor layer0;
	// LayerSpecifier temp;
	// layer0.initializeValues(1, 3, 3, 3, rows, cols, 1, 1, 1, 1);
	// temp.initPointer(CONV);
	// *((ConvDescriptor *)temp.params) = layer0;
	// layer_specifier.push_back(temp);
	// ActivationDescriptor layer0_actv;
	// layer0_actv.initializeValues(RELU, 3, rows, cols);
	// temp.initPointer(ACTV);
	// *((ActivationDescriptor *)temp.params) = layer0_actv;
	// layer_specifier.push_back(temp);

	// BatchNormDescriptor layer0_bn;

	// for (int i = 0; i < 200; i++) {
	// 	layer0_bn.initializeValues(BATCHNORM_SPATIAL, 1e-5, 0.1, 3, rows, cols);
	// 	temp.initPointer(BATCHNORM);
	// 	*((BatchNormDescriptor *)temp.params) = layer0_bn;
	// 	layer_specifier.push_back(temp);

	// 	layer0.initializeValues(3, 3, 3, 3, rows, cols, 1, 1, 1, 1);
	// 	temp.initPointer(CONV);
	// 	*((ConvDescriptor *)temp.params) = layer0;
	// 	layer_specifier.push_back(temp);
	// 	layer0_actv.initializeValues(RELU, 3, rows, cols);
	// 	temp.initPointer(ACTV);
	// 	*((ActivationDescriptor *)temp.params) = layer0_actv;
	// 	layer_specifier.push_back(temp);
	// }

	// PoolingDescriptor layer0_pool;
	// layer0_pool.initializeValues(3, 2, 2, rows, cols, 0, 0, 2, 2, POOLING_MAX);
	// temp.initPointer(POOLING);
	// *((PoolingDescriptor *)temp.params) = layer0_pool;
	// layer_specifier.push_back(temp);

	// layer0_bn.initializeValues(BATCHNORM_SPATIAL, 1e-5, 0.1, 3, rows / 2, cols / 2);
	// temp.initPointer(BATCHNORM);
	// *((BatchNormDescriptor *)temp.params) = layer0_bn;
	// layer_specifier.push_back(temp);

	// // DropoutDescriptor layer0_dropout;
	// // layer0_dropout.initializeValues(0.2, 3, rows / 2, cols / 2);
	// // temp.initPointer(DROPOUT);
	// // *((DropoutDescriptor *)temp.params) = layer0_dropout;
	// // layer_specifier.push_back(temp);

	// layer0.initializeValues(3, 3, 3, 3, rows / 2, cols / 2, 1, 1, 1, 1);
	// temp.initPointer(CONV);
	// *((ConvDescriptor *)temp.params) = layer0;
	// layer_specifier.push_back(temp);
	// layer0_actv.initializeValues(RELU, 3, rows / 2, cols / 2);
	// temp.initPointer(ACTV);
	// *((ActivationDescriptor *)temp.params) = layer0_actv;
	// layer_specifier.push_back(temp);

	// layer0_bn.initializeValues(BATCHNORM_SPATIAL, 1e-5, 0.1, 3, rows / 2, cols / 2);
	// temp.initPointer(BATCHNORM);
	// *((BatchNormDescriptor *)temp.params) = layer0_bn;
	// layer_specifier.push_back(temp);

	// FCDescriptor layer1;
	// layer1.initializeValues(input_channels, hidden_channels1);
	// temp.initPointer(FULLY_CONNECTED);
	// *((FCDescriptor *)(temp.params)) = layer1;
	// layer_specifier.push_back(temp);

	// temp.initPointer(ACTV);
	// ActivationDescriptor layer1_actv;
	// layer1_actv.initializeValues(RELU, hidden_channels1, 1, 1);
	// *((ActivationDescriptor *)temp.params) = layer1_actv;
	// layer_specifier.push_back(temp);

	// layer0_bn.initializeValues(BATCHNORM_PER_ACTIVATION, 1e-5, 0.1, hidden_channels1, 1, 1);
	// temp.initPointer(BATCHNORM);
	// *((BatchNormDescriptor *)temp.params) = layer0_bn;
	// layer_specifier.push_back(temp);

	// temp.initPointer(FULLY_CONNECTED);
	// FCDescriptor layer2;
	// layer2.initializeValues(hidden_channels1, output_channels);
	// *((FCDescriptor *)temp.params) = layer2;
	// layer_specifier.push_back(temp);

	// // temp.initPointer(FULLY_CONNECTED);
	// // FCDescriptor layer3;
	// // layer3.initializeValues(hidden_channels2, output_channels);
	// // *((FCDescriptor *)temp.params) = layer3;
	// // layer_specifier.push_back(temp);

	// temp.initPointer(SOFTMAX);
	// SoftmaxDescriptor smax;
	// smax.initializeValues(SOFTMAX_ACCURATE, SOFTMAX_MODE_INSTANCE, output_channels, 1, 1);
	// *((SoftmaxDescriptor *)(temp.params)) = smax;
	// layer_specifier.push_back(temp);
	
	// AlexNet
	vector<LayerSpecifier> layer_specifier;
	{
		ConvDescriptor layer0;
		layer0.initializeValues(3, 32, 1, 1, 156, 156, 0, 0, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = layer0;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor layer0;
		layer0.initializeValues(32, 32, 1, 1, 156, 156, 0, 0, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = layer0;
		layer_specifier.push_back(temp);
	}
	{
		ConvDescriptor layer0;
		layer0.initializeValues(32, 3, 1, 1, 156, 156, 0, 0, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = layer0;
		layer_specifier.push_back(temp);
	}
	{
		FCDescriptor layer8;
		layer8.initializeValues(156 * 156 * 3, 10);
		LayerSpecifier temp;
		temp.initPointer(FULLY_CONNECTED);
		*((FCDescriptor *)temp.params) = layer8;
		layer_specifier.push_back(temp);
	}
	{
		SoftmaxDescriptor layer11;
		layer11.initializeValues(SOFTMAX_ACCURATE, SOFTMAX_MODE_INSTANCE, 10, 1, 1);
		LayerSpecifier temp;
		temp.initPointer(SOFTMAX);
		*((SoftmaxDescriptor *)temp.params) = layer11;
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
		else {
			printf("invalid argument.. using vdnn dynamic\n");
			filename.assign("vdnn_dyn");
		}
		if ((strcmp(argv[1], "conv") == 0 or strcmp(argv[1], "all") == 0)) {
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

	int batch_size = 128;
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
	pow(std_dev, 0.5);
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