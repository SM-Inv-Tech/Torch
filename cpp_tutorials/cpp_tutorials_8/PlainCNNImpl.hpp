////////////////////////////////////////////////////////////////////////
/// \file      PlaneCNN.hpp
/// \copyright Copyright (c) 2022 标准模型投资有限公司.
///            All rights reserved.
///            Licensed under the MIT License.
/// \author    金小海
/// \date      2022年09月23日, Fri, 10:02
/// \version   1.0
/// \brief
#ifndef TORCH_PLAINCNNIMPL_HPP
#define TORCH_PLAINCNNIMPL_HPP

#include "torch/torch.h"

#include "ConvReluBnImpl.hpp"

class PlainCNNImpl : public torch::nn::Module
{
public:
    PlainCNNImpl(int in_channels, int out_channels);

    torch::Tensor forward(torch::Tensor x);

private:
    int mid_channels[3] = {32, 64, 128};
    ConvReluBn conv1 = nullptr;
    ConvReluBn down1 = nullptr;
    ConvReluBn conv2 = nullptr;
    ConvReluBn down2 = nullptr;
    ConvReluBn conv3 = nullptr;
    ConvReluBn down3 = nullptr;
    torch::nn::Conv2d out_conv = nullptr;
};

TORCH_MODULE(PlainCNN);

#endif //TORCH_PLAINCNNIMPL_HPP
