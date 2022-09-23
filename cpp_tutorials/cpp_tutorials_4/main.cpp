////////////////////////////////////////////////////////////////////////
/// \file      main.cpp
/// \copyright Copyright (c) 2022 标准模型投资有限公司.
///            All rights reserved.
///            Licensed under the MIT License.
/// \author    金小海
/// \date      2022年09月22日, Thu, 10:54
/// \version   1.0
/// \brief     张量变形
#include <iostream>

#include "torch/torch.h"

int main(int argc, char **argv)
{
    /* 张量的变形操作 */

    torch::Tensor tensor = torch::arange(0, 16, 1);
    std::cout << tensor.sizes() << std::endl;
    std::cout << tensor << std::endl;

    /* view函数 (改变形状) */
    tensor = tensor.view({2, 8});
    std::cout << tensor.sizes() << std::endl;
    std::cout << tensor << std::endl;

    tensor = tensor.view({2, -1});
    std::cout << tensor.sizes() << std::endl;
    std::cout << tensor << std::endl;

    /* transpose函数（转置） */
    tensor = tensor.transpose(0, 1);
    std::cout << tensor.sizes() << std::endl;
    std::cout << tensor << std::endl;

    /* reshape函数 */
    tensor = tensor.reshape({4, 4});
    std::cout << tensor.sizes() << std::endl;
    std::cout << tensor << std::endl;

    /* permute函数(将tensor的维度转换，参数代表原来的维度) */
    tensor = tensor.reshape({2, 2, 4});
    std::cout << tensor.sizes() << std::endl;
    tensor = tensor.permute({0, 2, 1});
    std::cout << tensor.sizes() << std::endl;
    std::cout << tensor << std::endl;

    return 0;
}