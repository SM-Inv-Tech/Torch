////////////////////////////////////////////////////////////////////////
/// \file      ImageDataset.hpp
/// \copyright Copyright (c) 2022 标准模型投资有限公司.
///            All rights reserved.
///            Licensed under the MIT License.
/// \author    金小海
/// \date      2022年09月23日, Fri, 13:45
/// \version   1.0
/// \brief
#ifndef TORCH_IMAGEDATASET_HPP
#define TORCH_IMAGEDATASET_HPP

#include "torch/torch.h"

#include "opencv2/opencv.hpp"

#include <vector>

class ImageDataset : public torch::data::datasets::Dataset<ImageDataset>
{
public:
    ImageDataset(std::string path, std::string type);

public:
    torch::data::Example<> get(std::size_t index) override;

    torch::optional<size_t> size() const override;

private:
    void load_data(const std::string& file_path);

private:
    std::string image_path;
    std::string image_type;
    int num_classes = -1;

    std::vector<std::string> images;
    std::vector<int> labels;
};


#endif //TORCH_IMAGEDATASET_HPP
