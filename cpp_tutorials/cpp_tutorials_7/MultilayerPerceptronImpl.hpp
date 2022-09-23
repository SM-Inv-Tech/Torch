#ifndef TORCH_MULTILAYERPERCEPTRONIMPL_HPP
#define TORCH_MULTILAYERPERCEPTRONIMPL_HPP

#include "torch/torch.h"

#include "LinearBnReluImpl.hpp"

class MultilayerPerceptronImpl : public torch::nn::Module
{
public:
    MultilayerPerceptronImpl(int input_features, int out_features);

    torch::Tensor forward(torch::Tensor x);

private:
    int mid_features[3] = {128, 64, 32};

    LinearBnReLu fc_1 = nullptr;
    LinearBnReLu fc_2 = nullptr;
    LinearBnReLu fc_3 = nullptr;

    torch::nn::Linear fc = nullptr;
};

TORCH_MODULE(MultilayerPerceptron);


#endif //TORCH_MULTILAYERPERCEPTRONIMPL_HPP
