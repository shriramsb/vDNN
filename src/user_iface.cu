#include "user_iface.h"

void ConvDescriptor::initializeValues(int input_channels, int output_channels, int kernel_h, int kernel_w, int input_h, int input_w, 
									int pad_h, int pad_w, int stride_x, int stride_y, ActivationMode activation_mode, double actv_coef) {
	this->input_channels = input_channels, this->output_channels = output_channels, this->kernel_h = kernel_h, this->kernel_w = kernel_w;
	this->input_h = input_h, this->input_w = input_w;
	this->pad_h = pad_h, this->pad_w = pad_w, this->stride_y = stride_y, this->stride_x = stride_x;
	this->activation_mode = activation_mode;
	this->actv_coef = actv_coef;
}

void PoolingDescriptor::initializeValues(int input_channels, int kernel_h, int kernel_w,
									int input_h, int input_w, int pad_h, int pad_w, int stride_x, int stride_y, PoolingMode mode) {
	this->input_channels = input_channels, this->kernel_h = kernel_h, this->kernel_w = kernel_w;
	this->input_h = input_h, this->input_w = input_w;
	this->pad_h = pad_h, this->pad_w = pad_w, this->stride_y = stride_y, this->stride_x = stride_x;
	this->mode = mode;
}

void DropoutDescriptor::initializeValues(double dropout_value, int channels, int h, int w) {
	this->dropout_value = dropout_value;
	this->channels = channels;
	this->h = h;
	this->w = w;
}

void FCDescriptor::initializeValues(int input_channels, int output_channels, ActivationMode activation_mode, double actv_coef) {
	this->input_channels = input_channels;
	this->output_channels = output_channels;
	this->activation_mode = activation_mode;
	this->actv_coef = actv_coef;
}


void BatchNormDescriptor::initializeValues(BatchNormMode mode, double epsilon, double factor, int channels, int h, int w) {
	this->mode = mode;
	this->epsilon = epsilon, this->factor = factor;
	this->channels = channels, this->h = h, this->w = w;
}

void ActivationDescriptor::initializeValues(ActivationMode mode, int channels, int h, int w, double coef) {
	this->mode = mode;
	this->channels = channels;
	this->h = h;
	this->w = w;
	this->coef = coef;
}

void SoftmaxDescriptor::initializeValues(SoftmaxAlgorithm algo, SoftmaxMode mode, int channels, int h, int w) {
	this->algo = algo;
	this->mode = mode;
	this->channels = channels;
	this->h = h;
	this->w = w;
}

void LayerSpecifier::initPointer(LayerOp type) {
	this->type = type;
	if (type == CONV)
		params = malloc(sizeof(ConvDescriptor));
	else if (type == FULLY_CONNECTED)
		params = malloc(sizeof(FCDescriptor));
	else if (type == BATCHNORM)
		params = malloc(sizeof(BatchNormDescriptor));
	else if (type == DROPOUT)
		params = malloc(sizeof(DropoutDescriptor));
	else if (type == POOLING)
		params = malloc(sizeof(PoolingDescriptor));
	else if (type == ACTV)
		params = malloc(sizeof(ActivationDescriptor));
	else if (type == SOFTMAX)
		params = malloc(sizeof(SoftmaxDescriptor));
}

void LayerSpecifier::freePointer() {
	free(params);
}
