////////////////////////////////////////////////////////////////////////        
/// \file      main.cpp                                          
/// \copyright Copyright (c) 2022 标准模型投资有限公司. 
///            All rights reserved.                   
///            Licensed under the MIT License.                                  
/// \author    金小海                                                        
/// \date      2022年09月23日, Fri, 14:05                                                   
/// \version   1.0                                                              
/// \brief     
#include <iostream>

#include "ImageDataset.hpp"

int main(int argc, char **argv)
{
    int batch_size = 2;
    auto data_set = ImageDataset("../dataset/hymenoptera_data/train", ".jpg").map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(data_set), batch_size);
    for (auto &batch: *data_loader)
    {
        auto data = batch.data;
        auto target = batch.target;
        std::cout << target << std::endl;
    }

    return 0;
}