#include "utils.h"

#ifndef USER_IFACE
#define USER_IFACE

enum LayerOp {CONV, FULLY_CONNECTED, BATCHNORM, DROPOUT, POOLING, ACTV, SOFTMAX, CROSS_ENTROPY, SVM};
enum SoftmaxAlgorithm {SOFTMAX_FAST, SOFTMAX_ACCURATE};
enum SoftmaxMode {SOFTMAX_MODE_INSTANCE, SOFTMAX_MODE_CHANNEL};
enum DataType {DATA_FLOAT, DATA_DOUBLE};
enum TensorFormat {TENSOR_NCHW, TENSOR_NHWC};
enum BatchNormMode {BATCHNORM_PER_ACTIVATION, BATCHNORM_SPATIAL};
enum PoolingMode {POOLING_MAX, POOLING_AVERAGE_COUNT_INCLUDE_PADDING, POOLING_AVERAGE_COUNT_EXCLUDE_PADDING};
enum ActivationMode {SIGMOID, RELU, TANH, CLIPPED_RELU, ELU, ACTIVATION_NONE};
enum UpdateRule {SGD};
enum vDNNType {vDNN_ALL, vDNN_CONV, vDNN_NONE, vDNN_DYN, vDNN_ALTERNATE_CONV};
enum vDNNConvAlgo {vDNN_PERFORMANCE_OPTIMAL, vDNN_MEMORY_OPTIMAL};

struct ConvDescriptor {
	int input_channels, output_channels, kernel_h, kernel_w;	// define kernel parameters
	int input_h, input_w;										// output width can be inferred
	int pad_h, pad_w, stride_y, stride_x;
	ActivationMode activation_mode;
	double actv_coef;

	void initializeValues(int input_channels, int output_channels, int kernel_h, int kernel_w, int input_h, int input_w, 
						int pad_h, int pad_w, int stride_x, int stride_y, ActivationMode activation_mode = ACTIVATION_NONE, double actv_coef = 1.0);
	
};

struct PoolingDescriptor {
	int input_channels, kernel_h, kernel_w;
	int input_h, input_w;
	int pad_h, pad_w, stride_y, stride_x;
	PoolingMode mode;

	void initializeValues(int input_channels, int kernel_h, int kernel_w,
						int input_h, int input_w, int pad_h, int pad_w, int stride_x, int stride_y, PoolingMode mode);
};

struct DropoutDescriptor {
	double dropout_value;
	int channels, h, w;

	void initializeValues(double dropout_value, int channels, int h, int w);
};

struct FCDescriptor {
	int input_channels, output_channels;
	ActivationMode activation_mode;
	double actv_coef;

	void initializeValues(int input_channels, int output_channels, ActivationMode activation_mode = ACTIVATION_NONE, double actv_coef = 1.0);

};

struct BatchNormDescriptor {
	BatchNormMode mode;
	double epsilon, factor;
	int channels, h, w;

	void initializeValues(BatchNormMode mode, double epsilon, double factor, int channels, int h, int w);
};

struct ActivationDescriptor {
	ActivationMode mode;
	int channels, h, w;
	double coef;
	void initializeValues(ActivationMode mode, int channels, int h, int w, double coef = 1.0);
};

struct SoftmaxDescriptor {
	int channels, h, w;
	SoftmaxAlgorithm algo;
	SoftmaxMode mode;

	void initializeValues(SoftmaxAlgorithm algo, SoftmaxMode mode, int channels, int h, int w);
};

struct LayerSpecifier {
	LayerOp type;
	void *params;

	void initPointer(LayerOp type);
	
	void freePointer();

};

#endif