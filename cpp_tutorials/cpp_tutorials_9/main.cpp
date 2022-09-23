////////////////////////////////////////////////////////////////////////
/// \file      main.cpp
/// \copyright Copyright (c) 2022 标准模型投资有限公司.
///            All rights reserved.
///            Licensed under the MIT License.
/// \author    金小海
/// \date      2022年09月23日, Fri, 10:19
/// \version   1.0
/// \brief
#include <iostream>

#include "torch/torch.h"

#include "LSTMImpl.hpp"

int main(int argc, char **argv)
{
    auto model = LSTM(3, 100, 2, 1, true);
    auto data = torch::Tensor(torch::linspace(1, 15, 15)).unsqueeze(1).repeat({1, 3}).unsqueeze(0).repeat({4, 1, 1});//[4,15,3]
    auto target = torch::full({4, 2}, 16).to(torch::kFloat);
    auto optimizer = torch::optim::Adam(model->parameters(), 0.003);
    for (int i = 0; i < 130; i++)
    {
        optimizer.zero_grad();

        auto out = model->forward(data.to(torch::kFloat));
        auto loss = torch::mse_loss(out, target);
        loss.backward();

        optimizer.step();

        std::cout << out << std::endl;
    }

    return 0;
}