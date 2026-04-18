#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

struct Box {
    float x1, y1, x2, y2;
};

// Return type changed to py::array_t<float>
py::array_t<float> postprocess(py::array_t<float> output,
                               int orig_h, int orig_w,
                               float conf_thres, float iou_thres);

#endif /* POSTPROCESS_H */
