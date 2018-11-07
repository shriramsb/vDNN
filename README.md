# vDNN
My implementation of the paper titled **vDNN: Virtualized Deep Neural Networks for Scalable, Memory-Efficient Neural Network Design** (https://arxiv.org/abs/1602.08124). Supports only linear networks currently.

### Instructions to set up
Run cmake and make inside ./cnmem/ as well as in ./. Look at vgg_test.cu for an example of specifying and training neural network. New program has to added in ./CMakeLists.txt. Look how vgg_test.cu has been added for an example. Essential API function declarations for training are in include/user_iface.h and include/solver.h. 

cnmem/ is a software-side memory manager by Nvidia (https://github.com/NVIDIA/cnmem). It has been modified a little use heurisitcs other than best-fit.
