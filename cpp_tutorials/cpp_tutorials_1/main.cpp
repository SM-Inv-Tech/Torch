#include "torch/torch.h"

int main(int argc, char **argv)
{
    /* 测试GPU加速是否可用 */
    bool is_gpu = torch::cuda::is_available();

    /* 创建张量 */
    torch::Tensor tensor = torch::tensor({1, 2});
    if (is_gpu)
    {
        tensor.to(torch::kCUDA);
    }

    return 0;
}