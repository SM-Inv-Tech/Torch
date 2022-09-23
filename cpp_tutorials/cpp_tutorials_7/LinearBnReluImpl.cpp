#include "LinearBnReluImpl.hpp"

////////////////////////////////////////////////////////////////////////
/// \brief
/// \param input_size
/// \param output_size
LinearBnReLuImpl::LinearBnReLuImpl(int input_size, int output_size)
{
    fc = torch::nn::Linear(torch::nn::LinearOptions(input_size, output_size));
    bn = torch::nn::BatchNorm1d(output_size);

    register_module("fc", fc);
    register_module("bn", bn);
}


////////////////////////////////////////////////////////////////////////
/// \brief
/// \param x
/// \return
torch::Tensor LinearBnReLuImpl::forward(torch::Tensor x)
{
    x = fc->forward(x);
    x = torch::relu(x);
    x = bn(x);

    return x;
}
