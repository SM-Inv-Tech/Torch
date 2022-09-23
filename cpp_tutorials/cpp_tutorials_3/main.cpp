////////////////////////////////////////////////////////////////////////
/// \file      main.cpp
/// \copyright Copyright (c) 2022 标准模型投资有限公司.
///            All rights reserved.
///            Licensed under the MIT License.
/// \author    金小海
/// \date      2022年09月22日, Thu, 09:44
/// \version   1.0
/// \brief     张量初始化
#include <iostream>
#include <vector>

#include "torch/torch.h"

int main(int argc, char **argv)
{
    /* 从C++的其它数据类型转换而来 */

    int arr[10] = {3, 4, 5, 6};
    std::vector<float> vec{7, 8, 9};

    /* 使用from_blob进行转化 */
    torch::Tensor tensor = torch::from_blob(arr, {2, 2}, torch::kInt);
    std::cout << tensor << std::endl;

    tensor = torch::from_blob(vec.data(), {3, 1}, torch::kFloat);
    std::cout << tensor << std::endl;


    /* 根据已有张量初始化 */

    tensor = torch::zeros({3, 4});
    /* 浅拷贝 */
    auto tensor_1 = torch::Tensor(tensor);

    /* zeros_like初始化 */
    tensor = torch::zeros_like(tensor);
    std::cout << tensor << std::endl;

    /* ones_like初始化 */
    tensor = torch::ones_like(tensor);
    std::cout << tensor << std::endl;

    /* rand_like初始化 */
    tensor = torch::rand_like(tensor, torch::kFloat);
    std::cout << tensor << std::endl;

    /* 深拷贝 */
    auto tensor_2 = tensor.clone();
    std::cout << tensor_2 << std::endl;

    return 0;
}
