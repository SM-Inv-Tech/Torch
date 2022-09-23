////////////////////////////////////////////////////////////////////////
/// \file      main.cpp
/// \copyright Copyright (c) 2022 标准模型投资有限公司.
///            All rights reserved.
///            Licensed under the MIT License.
/// \author    金小海
/// \date      2022年09月23日, Fri, 08:03
/// \version   1.0
/// \brief     张量间操作
#include <iostream>

#include "torch/torch.h"

int main(int argc, char **argv)
{
    /* 拼接和堆叠 */
    torch::Tensor tensor = torch::arange(2, 10, 1);
    tensor = tensor.reshape({2, 4});
    std::cout << tensor << std::endl;

    /* 拼接（维度1表示列拼接） */
    auto tensor_2 = torch::cat({tensor, tensor}, 1);
    std::cout << tensor_2 << std::endl;

    /* 堆叠 */
    auto tensor_3 = torch::stack({tensor, tensor}, 1);
    std::cout << tensor_3 << std::endl;

    /* 四则运算操作同理，像对应元素乘除直接用*和/即可，也可以用.mul和.div。矩阵乘法用.mm，加入批次就是.bmm。 */
    auto tensor_4 = tensor / tensor;
    std::cout << tensor_4 << std::endl;

    auto tensor_5 = tensor * tensor;
    std::cout << tensor_5 << std::endl;

    auto tensor_6 = torch::mm(tensor, tensor.reshape({4, 2}));
    std::cout << tensor_6 << std::endl;

    return 0;
}