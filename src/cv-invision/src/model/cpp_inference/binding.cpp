#include <pybind11/pybind11.h>
#include "preprocess.hpp"
#include "postprocess.hpp"


PYBIND11_MODULE(cpp_inference, m) {
    m.def("postprocess", &postprocess, "Fast C++ post-processing");
    m.def("preprocess", &preprocess, "Fast C++ preprocessing");
}
