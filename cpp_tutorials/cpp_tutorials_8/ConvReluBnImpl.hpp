////////////////////////////////////////////////////////////////////////
/// \file      ConvReluBnImpl.hpp
/// \copyright Copyright (c) 2022 标准模型投资有限公司.
///            All rights reserved.
///            Licensed under the MIT License.
/// \author    金小海
/// \date      2022年09月23日, Fri, 09:54
/// \version   1.0
/// \brief
#ifndef TORCH_CONVRELUBNIMPL_HPP
#define TORCH_CONVRELUBNIMPL_HPP

#include "torch/torch.h"

#include "options.hpp"

class ConvReluBnImpl : public torch::nn::Module
{
public:
    ConvReluBnImpl(int input_channel = 3, int output_channel = 64, int kernel_size = 3, int stride = 1);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv = nullptr;
    torch::nn::BatchNorm2d bn = nullptr;
};

TORCH_MODULE(ConvReluBn);


#endif //TORCH_CONVRELUBNIMPL_HPP
