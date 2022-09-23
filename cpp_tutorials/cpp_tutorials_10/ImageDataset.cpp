#include <filesystem>

#include "ImageDataset.hpp"

////////////////////////////////////////////////////////////////////////
/// \brief
/// \param path
/// \param type
ImageDataset::ImageDataset(std::string path, std::string type) : image_path(path), image_type(type)
{
    load_data(image_path);

}


////////////////////////////////////////////////////////////////////////
/// \brief
/// \return
torch::data::Example<> ImageDataset::get(std::size_t index)
{
    /* 图片 */
    std::string path = images[index];
    cv::Mat image = cv::imread(path);
    cv::resize(image, image, cv::Size(224, 224));

    /* 标签 */
    int label = labels[index];

    /* 制作图片和标签 */
    torch::Tensor img_tensor = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kByte).permute({2, 0, 1}); // Channels x Height x Width
    torch::Tensor label_tensor = torch::full({1}, label);

    return {img_tensor.clone(), label_tensor.clone()};
}


////////////////////////////////////////////////////////////////////////
/// \brief
/// \return
torch::optional<size_t> ImageDataset::size() const
{
    return images.size();
}


////////////////////////////////////////////////////////////////////////
/// \brief
void ImageDataset::load_data(const std::string& file_path)
{
    std::filesystem::path url(file_path);
    if (!std::filesystem::exists(url))
    {
        return;
    }

    std::filesystem::recursive_directory_iterator end;
    for (std::filesystem::recursive_directory_iterator begin(url); begin != end; ++begin)
    {
        if (begin->is_regular_file() && begin->path().filename().extension() == ".jpg")
        {
            images.push_back(begin->path().string());
            labels.push_back(num_classes);
        }
        else if (begin->is_directory())
        {
            ++num_classes;
            load_data(begin->path().string());
        }
    }
}

