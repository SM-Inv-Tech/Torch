////////////////////////////////////////////////////////////////////////        
/// \file      main.cpp                                          
/// \copyright Copyright (c) 2022 标准模型投资有限公司. 
///            All rights reserved.                   
///            Licensed under the MIT License.                                  
/// \author    金小海                                                        
/// \date      2022年09月23日, Fri, 09:17                                                   
/// \version   1.0                                                              
/// \brief     
#include <iostream>

#include "torch/torch.h"
#include "MultilayerPerceptronImpl.hpp"

int main(int argc, char **argv)
{
    auto model = MultilayerPerceptron(10, 1);
    auto data = torch::rand({2, 10});
    auto target = torch::ones({2, 1});
    torch::optim::Adam optimizer_mlp(model->parameters(), 0.0005);
    for (int i = 0; i < 400; i++)
    {
        optimizer_mlp.zero_grad();

        auto out = model->forward(data);

        auto loss = torch::mse_loss(out, target);
        loss.backward();

        optimizer_mlp.step();

        // std::cout << out << std::endl;
    }

    // 预测
    auto y_pred = model->forward(data);
    std::cout << target << std::endl;
    std::cout << y_pred << std::endl;

    return 0;
}