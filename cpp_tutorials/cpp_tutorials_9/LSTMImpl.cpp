#include "LSTMImpl.hpp"


////////////////////////////////////////////////////////////////////////
/// \brief
/// \param in_features
/// \param hidden_layer_size
/// \param out_size
/// \param num_layers
/// \param batch_first
LSTMImpl::LSTMImpl(int in_features, int hidden_layer_size, int out_size, int num_layers, bool batch_first)
{
    lstm = torch::nn::LSTM(lstmOption(in_features, hidden_layer_size, num_layers, batch_first));
    fc = torch::nn::Linear(hidden_layer_size, out_size);

    register_module("lstm", lstm);
    register_module("ln", fc);
}


////////////////////////////////////////////////////////////////////////
/// \brief
/// \param x
/// \return
torch::Tensor LSTMImpl::forward(torch::Tensor x)
{
    auto out = lstm->forward(x);
    auto predictions = fc->forward(std::get<0>(out));

    return predictions.select(1, -1);
}
