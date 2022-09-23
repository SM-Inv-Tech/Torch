////////////////////////////////////////////////////////////////////////
/// \file      main.cpp
/// \copyright Copyright (c) 2022 标准模型投资有限公司.
///            All rights reserved.
///            Licensed under the MIT License.
/// \author    金小海
/// \date      2022年09月23日, Fri, 09:54
/// \version   1.0
/// \brief
#include <iostream>

#include "torch/torch.h"
#include "PlainCNNImpl.hpp"

int main(int argc, char **argv)
{
    auto model = PlainCNN(3, 1);
    auto data = torch::randint(255, {1, 3, 224, 224});
    torch::optim::Adam optimizer_model(model->parameters(), 0.0003);
    auto target = torch::zeros({1, 1, 26, 26});
    for (int i = 0; i < 30; i++)
    {
        optimizer_model.zero_grad();

        auto out = model->forward(data);
        auto loss = torch::mse_loss(out, target);
        loss.backward();

        optimizer_model.step();

        std::cout << out[0][0][0];
    }

    return 0;
}