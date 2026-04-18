#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "preprocess.hpp"

namespace py = pybind11;

py::array_t<float> preprocess(py::array_t<uint8_t> input_image, int target_h, int target_w) {
    // 1. Get buffer info from NumPy array
    auto buf = input_image.request();
    int h = buf.shape[0];
    int w = buf.shape[1];
    
    // 2. Wrap the raw NumPy pointer into an OpenCV Mat (Zero Copy)
    // We assume the input is BGR (standard for cv2.imread or most streams)
    cv::Mat img(h, w, CV_8UC3, buf.ptr);
    
    // 3. Perform Optimized Preprocessing
    cv::Mat resized, rgb, float_img;
    
    // OpenCV resize uses SIMD instructions (AVX/NEON) internally
    cv::resize(img, resized, cv::Size(target_w, target_h), 0, 0, cv::INTER_LINEAR);
    
    // Convert BGR -> RGB (TFLite models usually expect RGB)
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    
    // Convert to Float and Scale to [0, 1]
    rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
    
    // 4. Create the output NumPy array with shape (1, H, W, 3) for TFLite
    std::vector<ssize_t> shape = {1, (ssize_t)target_h, (ssize_t)target_w, 3};
    auto result = py::array_t<float>(shape);
    
    // Copy the data from OpenCV Mat to the NumPy buffer
    // OpenCV data is contiguous by default after these operations
    std::memcpy(result.request().ptr, float_img.data, target_h * target_w * 3 * sizeof(float));
    
    return result;
}
