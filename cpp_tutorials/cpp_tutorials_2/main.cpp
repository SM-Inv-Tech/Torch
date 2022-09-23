////////////////////////////////////////////////////////////////////////
/// \file      main.cpp
/// \copyright Copyright (c) 2022 标准模型投资有限公司.
///            All rights reserved.
///            Licensed under the MIT License.
/// \author    金小海
/// \date      2022年09月22日, Thu, 09:26
/// \version   1.0
/// \brief     张量初始化
#include <iostream>

#include "torch/torch.h"

int main(int argc, char **argv)
{
    /* 固定尺寸和值的初始化 */

    /* 产生全为0的张量 */
    torch::Tensor tensor = torch::zeros({3, 4});
    std::cout << tensor << std::endl;

    /* 产生全为1的张量 */
    tensor = torch::ones({3, 4});
    std::cout << tensor << std::endl;

    /* 产生单位张量 */
    tensor = torch::eye(4);
    std::cout << tensor << std::endl;

    /* 产生指定值和指定维度的张量 */
    tensor = torch::full({3, 4}, 10);
    std::cout << tensor << std::endl;

    /* 产生单行向量 */
    tensor = torch::tensor({3, 4, 5});
    std::cout << tensor << std::endl;

    /* 固定尺寸，随机值的初始化 */

    /* 产生0到1之间的均匀分布 */
    tensor = torch::rand({3, 4});
    std::cout << tensor << std::endl;

    /* 产生正态分布 */
    tensor = torch::randn({3, 4});
    std::cout << tensor << std::endl;

    /* 产生指定范围的整型随机分布,[min, max) */
    tensor = torch::randint(2, 6, {4, 4});
    std::cout << tensor << std::endl;

    return 0;
}