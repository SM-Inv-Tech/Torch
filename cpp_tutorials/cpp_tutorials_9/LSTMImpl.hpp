////////////////////////////////////////////////////////////////////////
/// \file      LSTMImpl.hpp
/// \copyright Copyright (c) 2022 标准模型投资有限公司.
///            All rights reserved.
///            Licensed under the MIT License.
/// \author    金小海
/// \date      2022年09月23日, Fri, 10:21
/// \version   1.0
/// \brief
#ifndef TORCH_LSTMIMPL_HPP
#define TORCH_LSTMIMPL_HPP

#include "torch/torch.h"

#include "options.hpp"

class LSTMImpl : public torch::nn::Module
{
public:
    LSTMImpl(int in_features, int hidden_layer_size, int out_size, int num_layers, bool batch_first);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::LSTM lstm = nullptr;
    torch::nn::Linear fc = nullptr;
};

TORCH_MODULE(LSTM);


#endif //TORCH_LSTMIMPL_HPP
