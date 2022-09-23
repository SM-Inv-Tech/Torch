#include "MultilayerPerceptronImpl.hpp"


////////////////////////////////////////////////////////////////////////
/// \brief
/// \param input_features
/// \param out_features
MultilayerPerceptronImpl::MultilayerPerceptronImpl(int input_features, int out_features)
{
    fc_1 = LinearBnReLu(input_features, mid_features[0]);
    fc_2 = LinearBnReLu(mid_features[0], mid_features[1]);
    fc_3 = LinearBnReLu(mid_features[1], mid_features[2]);
    fc = torch::nn::Linear(mid_features[2], out_features);

    register_module("fc_1", fc_1);
    register_module("fc_2", fc_2);
    register_module("fc_3", fc_3);
    register_module("fc", fc);
}


////////////////////////////////////////////////////////////////////////
/// \brief
/// \param x
/// \return
torch::Tensor MultilayerPerceptronImpl::forward(torch::Tensor x)
{
    x = fc_1->forward(x);
    x = fc_2->forward(x);
    x = fc_3->forward(x);
    x = fc->forward(x);

    return x;
}
