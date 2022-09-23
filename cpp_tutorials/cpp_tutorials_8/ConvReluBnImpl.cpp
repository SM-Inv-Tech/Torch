//
// Created by xiaohai on 9/23/22.
//

#include "ConvReluBnImpl.hpp"


////////////////////////////////////////////////////////////////////////
/// \brief
/// \param input_channel
/// \param output_channel
/// \param kernel_size
/// \param stride
ConvReluBnImpl::ConvReluBnImpl(int input_channel, int output_channel, int kernel_size, int stride)
{
    conv = register_module("conv", torch::nn::Conv2d(conv_options(input_channel, output_channel, kernel_size, stride, kernel_size / 2)));
    bn = register_module("bn", torch::nn::BatchNorm2d(output_channel));
}


////////////////////////////////////////////////////////////////////////
/// \brief
/// \param x
/// \return
torch::Tensor ConvReluBnImpl::forward(torch::Tensor x)
{
    x = conv->forward(x);
    x = torch::relu(x);
    x = bn(x);

    return x;
}
