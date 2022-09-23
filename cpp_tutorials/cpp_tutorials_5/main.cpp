////////////////////////////////////////////////////////////////////////
/// \file      main.cpp
/// \copyright Copyright (c) 2022 标准模型投资有限公司.
///            All rights reserved.
///            Licensed under the MIT License.
/// \author    金小海
/// \date      2022年09月22日, Thu, 11:16
/// \version   1.0
/// \brief     张量截取
#include <iostream>

#include "torch/torch.h"

int main(int argc, char **argv)
{
    /* 通过索引访问 */
    torch::Tensor tensor = torch::rand({10, 3, 28, 28});
    /* 第0张照片 */
    std::cout << tensor[0].sizes() << std::endl;
    /* 第0张照片的第0个通道 */
    std::cout << tensor[0][0].sizes() << std::endl;
    /* 第0张照片的第0个通道的第0行像素 dim为1 */
    std::cout << tensor[0][0][0].sizes() << std::endl;
    /* 第0张照片的第0个通道的第0行的第0个像素 dim为0 */
    std::cout << tensor[0][0][0][0].sizes() << std::endl;

    /* index_select函数选取(tensor变量指定选取的索引) */
    /* 选择第0维的1,2,5"三行" */
    std::cout << tensor.index_select(0, torch::tensor({1, 2, 5})).sizes() << std::endl;

    /* index_select函数选取(tensor变量指定选取的索引) */
    /* 选择第1维的第0和第2行组成新张量 */
    std::cout << tensor.index_select(1, torch::tensor({0, 2})).sizes() << std::endl;

    /* index_select函数选取(使用范围函数)*/
    /* 选择十张图片每个通道的前8列 */
    std::cout << tensor.index_select(2, torch::arange(0, 8)).sizes() << std::endl;

    /* narrow函数选取(ndim, start, length) */
    std::cout << tensor.narrow(1, 0, 2).sizes() << std::endl;

    /* select函数选取 */
    std::cout << tensor.select(3, 2).sizes() << std::endl;

    return 0;
}