#include <torch/extension.h>
#include <vector>
#include <cmath>
#include <cstdlib>

// this is not used as there is no actual PIL->Tensor conv
// torch::Tensor to_image(torch::Tensor tensor) {
    // return tensor.to(torch::kFloat32); 
// }

torch::Tensor to_dtype(torch::Tensor tensor, torch::Dtype dtype, bool scale) {
    tensor = tensor.to(dtype);
    if (scale && dtype == torch::kFloat32) {
        auto data = tensor.data_ptr<float>();
        int size = tensor.numel();
        for (int i = 0; i < size; i++) {
            data[i] = data[i] / 255.0f;
        }
    }
    return tensor;
}

torch::Tensor normalize(torch::Tensor tensor, std::vector<double> mean, std::vector<double> std) {
    TORCH_CHECK(tensor.dim() == 3, "Expected tensor of image ( C, H, W)");
    
    int c = tensor.size(0);
    int h = tensor.size(1);
    int w = tensor.size(2);
    
    float* pixels = tensor.data_ptr<float>();
    
    // equiv to tensor.sub_().div_()
    for (int ch = 0; ch < c; ch++) {
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                int index = ch * h * w + i * w + j;
                pixels[index] = (pixels[index] - mean[ch]) / std[ch];
            }
        }
    }
    return tensor;
}

torch::Tensor pad(torch::Tensor tensor, int padding, double fill) {
    TORCH_CHECK(tensor.dim() == 3, "Expected tensor of image ( C, H, W)");
    TORCH_CHECK(padding >= 0, "Padding must be positive");
    
    int c = tensor.size(0);
    int h = tensor.size(1);
    int w = tensor.size(2);
    
    int new_h = h + 2 * padding; // 2 for top and bottom
    int new_w = w + 2 * padding; // same but left and right
    
    auto result = torch::full({c, new_h, new_w}, fill, tensor.options());
    float* pixels_old = tensor.data_ptr<float>();
    float* pixels = result.data_ptr<float>();
    
    for (int ch = 0; ch < c; ch++) {
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                int index_old = ch * h * w + i * w + j;
                int index = ch * new_h * new_w + (i + padding) * new_w + (j + padding);
                pixels[index] = pixels_old[index_old];
            }
        }
    }
    return result;
}

torch::Tensor random_crop(torch::Tensor tensor, int size) {
    int c = tensor.size(0);
    int h = tensor.size(1);
    int w = tensor.size(2);
    
    TORCH_CHECK(h >= size && w >= size, "Tensor dimensions must be >= crop size");
    
    int top = rand() % (h - size + 1); // where to start from
    int left = rand() % (w - size + 1); // same but for left-side of the tensor/image
    
    auto result = torch::empty({c, size, size}, tensor.options());
    float* pixels_old = tensor.data_ptr<float>();
    float* pixels = result.data_ptr<float>();
    
    for (int ch = 0; ch < c; ch++) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                int index_old = ch * h * w + (top + i) * w + (left + j);
                int index = ch * size * size + i * size + j;
                pixels[index] = pixels_old[index_old];
            }
        }
    }
    return result;
}

torch::Tensor random_horizontal_flip(torch::Tensor tensor, double p) {
    TORCH_CHECK(p >= 0.0 && p <= 1.0, "Probability must be in [0.0, 1.0]");
    
    double r = (double)rand() / RAND_MAX;
    if (r >= p) {
        return tensor;
    }
    
    int c = tensor.size(0);
    int h = tensor.size(1);
    int w = tensor.size(2);
    
    auto result = torch::empty_like(tensor);
    float* pixels_old = tensor.data_ptr<float>();
    float* pixels = result.data_ptr<float>();
    
    for (int ch = 0; ch < c; ch++) {
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                int index_old = ch * h * w + i * w + j;
                int flip_w = w - 1 - j;
                int index = ch * h * w + i * w + flip_w;
                pixels[index] = pixels_old[index_old];
            }
        }
    }
    return result;
}

torch::Tensor random_erasing(torch::Tensor tensor, double p, 
                             double scale_min, double scale_max, double value) {
    TORCH_CHECK(p >= 0.0 && p <= 1.0, "Probability must be in [0.0, 1.0]");
    TORCH_CHECK(scale_min <= scale_max, "scale_min must be <= scale_max");
    TORCH_CHECK(scale_min >= 0 && scale_max <= 1, "Scale should be between 0 and 1");
    
    double r = (double)rand() / RAND_MAX;
    if (r >= p) {
        return tensor;
    }
    
    int c = tensor.size(0);
    int h = tensor.size(1);
    int w = tensor.size(2);
    int area = h * w;
    
    double scale = scale_min + (scale_max - scale_min) * rand() / RAND_MAX;
    int erase_area = (int)(area * scale);
    int erase_h = (int)sqrt(erase_area);
    int erase_w = erase_h;
    
    if (erase_h > h) {
        erase_h = h;
    }
    if (erase_w > w) {
        erase_w = w;
    }
    
    int top = rand() % (h - erase_h + 1);
    int left = rand() % (w - erase_w + 1);
    
    auto result = tensor.clone();
    float* pixels = result.data_ptr<float>();
    
    for (int ch = 0; ch < c; ch++) {
        for (int i = top; i < top + erase_h; i++) {
            for (int j = left; j < left + erase_w; j++) {
                int index = ch * h * w + i * w + j;
                pixels[index] = value;
            }
        }
    }
    return result;
}

std::tuple<torch::Tensor, torch::Tensor> cutmix(torch::Tensor images, torch::Tensor labels, int num_classes) {
    int batch = images.size(0);
    int c = images.size(1);
    int h = images.size(2);
    int w = images.size(3);
    
    double lambda = (double)rand() / RAND_MAX;
    
    // shuffle indices for the batch
    std::vector<int> indices(batch);
    for (int i = 0; i < batch; i++) indices[i] = i;
    for (int i = batch - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        std::swap(indices[i], indices[j]);
    }
    
    double cut_ratio = sqrt(1.0 - lambda);
    int cut_h = (int)(h * cut_ratio);
    int cut_w = (int)(w * cut_ratio);
    int cx = rand() % w;
    int cy = rand() % h;
    int x1 = std::max(0, cx - cut_w / 2);
    int x2 = std::min(w, cx + cut_w / 2);
    int y1 = std::max(0, cy - cut_h / 2);
    int y2 = std::min(h, cy + cut_h / 2);
    
    auto result_images = images.clone();
    float* pixels = result_images.data_ptr<float>();
    
    for (int b = 0; b < batch; b++) {
        int shuffled_b = indices[b];
        for (int ch = 0; ch < c; ch++) {
            for (int i = y1; i < y2; i++) {
                for (int j = x1; j < x2; j++) {
                    int index = b * c * h * w + ch * h * w + i * w + j;
                    int index_old = shuffled_b * c * h * w + ch * h * w + i * w + j;
                    pixels[index] = pixels[index_old];
                }
            }
        }
    }
    
    lambda = 1.0 - ((x2 - x1) * (y2 - y1)) / (double)(h * w);
    auto result_labels = torch::zeros({batch, num_classes}, images.options());
    float* label_pixels = result_labels.data_ptr<float>();
    long* original_labels = labels.data_ptr<long>();
    
    for (int b = 0; b < batch; b++) {
        int label1 = original_labels[b];
        int label2 = original_labels[indices[b]];
        label_pixels[b * num_classes + label1] += lambda;
        label_pixels[b * num_classes + label2] += (1.0 - lambda);
    }
    
    return std::make_tuple(result_images, result_labels);
}

std::tuple<torch::Tensor, torch::Tensor> mixup(torch::Tensor images, torch::Tensor labels, int num_classes) {
    int batch = images.size(0);
    int total_size = images.numel();
    
    double lambda = (double)rand() / RAND_MAX;
    
    std::vector<int> indices(batch);
    for (int i = 0; i < batch; i++) indices[i] = i;
    for (int i = batch - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        std::swap(indices[i], indices[j]);
    }
    
    auto result_images = torch::empty_like(images);
    float* pixels = result_images.data_ptr<float>();
    float* pixels_old = images.data_ptr<float>();
    
    int size_per_batch = total_size / batch;
    
    for (int b = 0; b < batch; b++) {
        int shuffled_b = indices[b];
        for (int i = 0; i < size_per_batch; i++) {
            int index_old = b * size_per_batch + i;
            int index = shuffled_b * size_per_batch + i;
            pixels[index_old] = pixels_old[index_old] * lambda + pixels_old[index] * (1.0 - lambda);
        }
    }
    
    auto result_labels = torch::zeros({batch, num_classes}, images.options());
    float* label_pixels = result_labels.data_ptr<float>();
    long* original_labels = labels.data_ptr<long>();
    
    for (int b = 0; b < batch; b++) {
        int label1 = original_labels[b];
        int label2 = original_labels[indices[b]];
        label_pixels[b * num_classes + label1] += lambda;
        label_pixels[b * num_classes + label2] += (1.0 - lambda);
    }
    
    return std::make_tuple(result_images, result_labels);
}

std::tuple<torch::Tensor, torch::Tensor> random_choice_mix(torch::Tensor images, torch::Tensor labels, int num_classes) {
    int choice = rand() % 3;
    
    if (choice == 0) {
        return cutmix(images, labels, num_classes);
    } else if (choice == 1) {
        return mixup(images, labels, num_classes);
    } else {
        int batch = labels.size(0);
        auto result_labels = torch::zeros({batch, num_classes}, torch::kFloat32);
        float* label_pixels = result_labels.data_ptr<float>();
        long* original_labels = labels.data_ptr<long>();
        
        for (int b = 0; b < batch; b++) {
            label_pixels[b * num_classes + original_labels[b]] = 1.0;
        }
        
        return std::make_tuple(images, result_labels);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    srand(time(NULL));
    
    // m.def("to_image", &to_image);
    m.def("to_dtype", &to_dtype);
    m.def("normalize", &normalize);
    m.def("pad", &pad);
    m.def("random_crop", &random_crop);
    m.def("random_horizontal_flip", &random_horizontal_flip);
    m.def("random_erasing", &random_erasing);
    m.def("cutmix", &cutmix);
    m.def("mixup", &mixup);
    m.def("random_choice_mix", &random_choice_mix);
}
