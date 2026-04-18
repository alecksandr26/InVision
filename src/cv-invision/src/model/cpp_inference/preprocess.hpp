#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <pybind11/numpy.h>

namespace py = pybind11;

// Only one function needed now; OpenCV handles its own memory management
py::array_t<float> preprocess(py::array_t<uint8_t> input_image, int target_h, int target_w);

#endif /* PREPROCESS_H */
