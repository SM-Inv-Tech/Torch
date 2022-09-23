////////////////////////////////////////////////////////////////////////
/// \file      options.hpp
/// \copyright Copyright (c) 2022 标准模型投资有限公司.
///            All rights reserved.
///            Licensed under the MIT License.
/// \author    金小海
/// \date      2022年09月23日, Fri, 10:20
/// \version   1.0
/// \brief
#ifndef TORCH_OPTIONS_HPP
#define TORCH_OPTIONS_HPP

#include "torch/torch.h"

////////////////////////////////////////////////////////////////////////
/// \brief
/// \param in_features
/// \param hidden_layer_size
/// \param num_layers
/// \param batch_first
/// \param bidirectional
/// \return
inline
torch::nn::LSTMOptions lstmOption(int in_features, int hidden_layer_size, int num_layers, bool batch_first = false, bool bidirectional = false)
{
    torch::nn::LSTMOptions option = torch::nn::LSTMOptions(in_features, hidden_layer_size);
    option.num_layers(num_layers).batch_first(batch_first).bidirectional(bidirectional);
    return option;
}

#endif //TORCH_OPTIONS_HPP
