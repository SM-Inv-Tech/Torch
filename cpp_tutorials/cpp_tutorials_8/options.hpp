////////////////////////////////////////////////////////////////////////
/// \file      options_func.hpp
/// \copyright Copyright (c) 2022 标准模型投资有限公司.
///            All rights reserved.
///            Licensed under the MIT License.
/// \author    金小海
/// \date      2022年09月23日, Fri, 09:56
/// \version   1.0
/// \brief
#ifndef TORCH_OPTIONS_HPP
#define TORCH_OPTIONS_HPP

#include "torch/torch.h"

////////////////////////////////////////////////////////////////////////
/// \brief
/// \param in_planes
/// \param out_planes
/// \param kernel_size
/// \param stride
/// \param padding
/// \param with_bias
/// \return
inline
torch::nn::Conv2dOptions conv_options(int in_planes, int out_planes, int kernel_size, int stride = 1, int padding = 0, bool with_bias = false)
{
    torch::nn::Conv2dOptions options = torch::nn::Conv2dOptions(in_planes, out_planes, kernel_size);
    options.stride(stride);
    options.padding(padding);
    options.bias(with_bias);
    return options;
}

#endif //TORCH_OPTIONS_HPP
