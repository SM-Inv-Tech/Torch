////////////////////////////////////////////////////////////////////////
/// \file      LinearBnReluImpl.hpp
/// \copyright Copyright (c) 2022 标准模型投资有限公司.
///            All rights reserved.
///            Licensed under the MIT License.
/// \author    金小海
/// \date      2022年09月23日, Fri, 08:24
/// \version   1.0
/// \brief
#ifndef TORCH_LINEARBNRELUIMPL_HPP
#define TORCH_LINEARBNRELUIMPL_HPP

#include "torch/torch.h"

class LinearBnReLuImpl : public torch::nn::Module
{
public:
    LinearBnReLuImpl(int input_size, int output_size);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear fc = nullptr;
    torch::nn::BatchNorm1d bn = nullptr;
};

TORCH_MODULE(LinearBnReLu);


#endif //TORCH_LINEARBNRELUIMPL_HPP
