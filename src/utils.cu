#include "utils.h"

int LayerDimension::getTotalSize() {
	return N * C * H * W;
}

void outOfMemory() {
	std::cout << "Out of Memory\n";
	exit(0);
}

CnmemSpace::CnmemSpace(size_t free_bytes) {
	this->free_bytes = free_bytes;
	this->initial_free_bytes = free_bytes;
	this->out_of_memory = false;
} 

void CnmemSpace::updateSpace(CnmemSpace::Op op, size_t size) {

	if (op == ADD)
		free_bytes += ceil(1.0 * size / CNMEM_GRANULARITY) * CNMEM_GRANULARITY;
	else if (op == SUB) {
		size_t required_space = ceil(1.0 * size / CNMEM_GRANULARITY) * CNMEM_GRANULARITY;
		if (required_space > free_bytes)
			this->out_of_memory = true;
		free_bytes -= required_space;
	}
}

bool CnmemSpace::isAvailable() {
	return !out_of_memory;
}

size_t CnmemSpace::getConsumed() {
	return (initial_free_bytes - free_bytes);
}

void CnmemSpace::updateMaxConsume(size_t &max_consume) {
	max_consume = max_consume > (initial_free_bytes - free_bytes) ? max_consume : (initial_free_bytes - free_bytes);
}