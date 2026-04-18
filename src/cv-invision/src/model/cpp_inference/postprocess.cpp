#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <cmath>

namespace py = pybind11;
using namespace py::literals;  // enables "_a" literal

struct Box {
    float x1, y1, x2, y2;
};

static float iou(const Box& a, const Box& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);
    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter_area / (area_a + area_b - inter_area);
}

static std::vector<int> nms(const std::vector<Box>& boxes,
                            const std::vector<float>& scores,
                            float iou_threshold) {
    std::vector<int> indices(boxes.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(),
              [&](int i, int j) { return scores[i] > scores[j]; });
    std::vector<int> keep;
    while (!indices.empty()) {
        int idx = indices[0];
        keep.push_back(idx);
        std::vector<int> new_indices;
        for (size_t i = 1; i < indices.size(); ++i) {
            if (iou(boxes[idx], boxes[indices[i]]) <= iou_threshold)
                new_indices.push_back(indices[i]);
        }
        indices.swap(new_indices);
    }
    return keep;
}

std::vector<py::dict> postprocess(py::array_t<float> output,
                                  int orig_h, int orig_w,
                                  float conf_thres, float iou_thres) {
    auto buf = output.request();
    float* out = static_cast<float*>(buf.ptr);
    
    // Assume shape [1, 14, 8400] or [1, 8400, 14]? 
    // Python code does output_data[0].T -> shape becomes (8400, 14)
    // So we handle both: if ndim==3 and shape[1]==14 and shape[2]==8400, treat as [1,14,8400]
    // else if ndim==2 and shape[0]==8400 and shape[1]==14, treat directly.
    int num_dets, num_classes;
    float* data_ptr = out;
    
    if (buf.ndim == 3 && buf.shape[1] == 14 && buf.shape[2] == 8400) {
        // Shape [1, 14, 8400]
        num_dets = buf.shape[2];
        num_classes = buf.shape[1] - 4; // 10
        // data layout: for each detection i, class scores start at out[ i*14 + 4 ]
        // we'll iterate accordingly
        data_ptr = out; // already correct
    } else if (buf.ndim == 2 && buf.shape[0] == 8400 && buf.shape[1] == 14) {
        // Shape [8400, 14] after transpose
        num_dets = buf.shape[0];
        num_classes = buf.shape[1] - 4; // 10
        data_ptr = out;
    } else {
        throw std::runtime_error("Unsupported output tensor shape");
    }
    
    std::vector<float> scores;
    std::vector<Box> boxes;
    std::vector<int> class_ids;
    
    for (int i = 0; i < num_dets; ++i) {
        float max_score = 0.0f;
        int cls = -1;
        // class scores start at offset 4
        for (int c = 0; c < num_classes; ++c) {
            float score = data_ptr[i * (4 + num_classes) + 4 + c];
            if (score > max_score) {
                max_score = score;
                cls = c;
            }
        }
        if (max_score > conf_thres) {
            float xc = data_ptr[i * (4 + num_classes) + 0];
            float yc = data_ptr[i * (4 + num_classes) + 1];
            float w  = data_ptr[i * (4 + num_classes) + 2];
            float h  = data_ptr[i * (4 + num_classes) + 3];
            float x1 = (xc - w * 0.5f) * orig_w;
            float y1 = (yc - h * 0.5f) * orig_h;
            float x2 = (xc + w * 0.5f) * orig_w;
            float y2 = (yc + h * 0.5f) * orig_h;
            boxes.push_back({x1, y1, x2, y2});
            scores.push_back(max_score);
            class_ids.push_back(cls);
        }
    }
    
    auto keep = nms(boxes, scores, iou_thres);
    std::vector<py::dict> detections;
    for (int idx : keep) {
        const Box& b = boxes[idx];
        detections.push_back(py::dict(
            "box"_a = py::make_tuple((int)b.x1, (int)b.y1, (int)b.x2, (int)b.y2),
            "confidence"_a = scores[idx],
            "class_id"_a = class_ids[idx]
        ));
    }
    return detections;
}

PYBIND11_MODULE(postprocess_cpp, m) {
    m.def("postprocess", &postprocess, "Fast C++ post-processing");
}
