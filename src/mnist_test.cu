#include <iostream>
#include <cstdlib>
#include <string>

#include "solver.h"

using namespace std;

typedef unsigned char uchar;

int num_train = 60000, num_test = 10000;

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

int main() {

	int rows = 28, cols = 28, channels = 1;
	float *f_train_images, *f_test_images;
	int *f_train_labels, *f_test_labels;
	// int rows = 28, cols = 28, channels = 1;
	int input_size = rows * cols * channels;
	f_train_images = (float *)malloc(num_train * input_size * sizeof(float));
	f_train_labels = (int *)malloc(num_train * sizeof(int));
	f_test_images = (float *)malloc(num_test * input_size * sizeof(float));
	f_test_labels = (int *)malloc(num_test * sizeof(int));

	{
		vector<vector<uchar> > train_images, test_images;
		vector<uchar> train_labels, test_labels;
		readMNIST(train_images, test_images, train_labels, test_labels);
	
		for (int k = 0; k < num_train; k++) {
			for (int j = 0; j < rows * cols; j++) {
				f_train_images[k * input_size + j] = (float)train_images[k][j];
			}
			f_train_labels[k] = (int)train_labels[k];
		}
	
		for (int k = 0; k < num_test; k++) {
			for (int j = 0; j < rows * cols; j++) {
				f_test_images[k * input_size + j] = (float)test_images[k][j];
			}
			f_test_labels[k] = (int)test_labels[k];
		}
	}

	

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
	
	vector<LayerSpecifier> layer_specifier;
	{
		ConvDescriptor layer0;
		layer0.initializeValues(1, 3, 3, 3, 28, 28, 1, 1, 1, 1, RELU);
		LayerSpecifier temp;
		temp.initPointer(CONV);
		*((ConvDescriptor *)temp.params) = layer0;
		layer_specifier.push_back(temp);
	}
	{
		FCDescriptor layer1;
		layer1.initializeValues(3 * 28 * 28, 50, RELU);
		LayerSpecifier temp;
		temp.initPointer(FULLY_CONNECTED);
		*((FCDescriptor *)temp.params) = layer1;
		layer_specifier.push_back(temp);
	}
	{
		FCDescriptor layer2;
		layer2.initializeValues(50, 10);
		LayerSpecifier temp;
		temp.initPointer(FULLY_CONNECTED);
		*((FCDescriptor *)temp.params) = layer2;
		layer_specifier.push_back(temp);
	}
	{
		SoftmaxDescriptor layer2_smax;
		layer2_smax.initializeValues(SOFTMAX_ACCURATE, SOFTMAX_MODE_INSTANCE, 10, 1, 1);
		LayerSpecifier temp;
		temp.initPointer(SOFTMAX);
		*((SoftmaxDescriptor *)temp.params) = layer2_smax;
		layer_specifier.push_back(temp);
	}

	int batch_size = 128;
	long long dropout_seed = 1;
	float softmax_eps = 1e-8;
	float init_std_dev = 0.01;
	NeuralNet net(layer_specifier, DATA_FLOAT, batch_size, TENSOR_NCHW, dropout_seed, softmax_eps, init_std_dev, vDNN_ALL, vDNN_MEMORY_OPTIMAL, SGD);

	int num_epoch = 1000;
	double learning_rate = 1e-4;
	double learning_rate_decay = 0.9;
	
	Solver solver(&net, (void *)f_train_images, f_train_labels, (void *)f_train_images, f_train_labels, num_epoch, SGD, learning_rate, learning_rate_decay, num_train, num_train);
	vector<float> loss;
	vector<int> val_acc;
	solver.train(loss, val_acc);
	int num_correct;
	solver.checkAccuracy(f_train_images, f_train_labels, num_train, &num_correct);
	cout << num_correct << endl;



}