////////////////////////////////////////////////////////////////////////
/// \file      VGGImpl.hpp
/// \copyright Copyright (c) 2022 标准模型投资有限公司.
///            All rights reserved.
///            Licensed under the MIT License.
/// \author    金小海
/// \date      2022年09月23日, Fri, 16:14
/// \version   1.0
/// \brief
#ifndef TORCH_VGGIMPL_HPP
#define TORCH_VGGIMPL_HPP

#include <vector>

#include "torch/torch.h"


class VGGImpl : public torch::nn::Module
{
public:
    VGGImpl(std::vector<int> &cfg, int num_classes = 1000, bool batch_norm = false);

public:
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential features = nullptr;
    torch::nn::AdaptiveAvgPool2d avgPool = nullptr;
    torch::nn::Sequential classifier;
};


#endif //TORCH_VGGIMPL_HPP