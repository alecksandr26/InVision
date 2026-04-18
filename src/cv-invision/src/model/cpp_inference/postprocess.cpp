#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>
#include <omp.h>
#include "postprocess.hpp"

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

static std::vector<int> nms(const std::vector<Box>& boxes, const std::vector<float>& scores, float iou_threshold) {
    std::vector<int> indices(boxes.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&](int i, int j) { return scores[i] > scores[j]; });

    std::vector<int> keep;
    std::vector<bool> suppressed(boxes.size(), false);
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (suppressed[idx]) continue;
        keep.push_back(idx);
        for (size_t j = i + 1; j < indices.size(); ++j) {
            int next_idx = indices[j];
            if (!suppressed[next_idx] && iou(boxes[idx], boxes[next_idx]) > iou_threshold) {
                suppressed[next_idx] = true;
            }
        }
    }
    return keep;
}

py::array_t<float> postprocess(py::array_t<float> output, int orig_h, int orig_w, float conf_thres, float iou_thres) {
    auto buf = output.request();
    float* data_ptr = static_cast<float*>(buf.ptr);

    // Expecting shape (8400, 14) after Python transpose
    int num_dets = buf.shape[0];
    int num_classes = buf.shape[1] - 4;

    std::vector<float> scores;
    std::vector<Box> boxes;
    std::vector<int> class_ids;

    #pragma omp parallel
    {
        std::vector<float> l_scores; std::vector<Box> l_boxes; std::vector<int> l_ids;
        #pragma omp for nowait
        for (int i = 0; i < num_dets; ++i) {
            float max_s = 0.0f; int cls = -1;
            float* row = data_ptr + (i * (4 + num_classes));
            for (int c = 0; c < num_classes; ++c) {
                if (row[4 + c] > max_s) { max_s = row[4 + c]; cls = c; }
            }
            if (max_s > conf_thres) {
                float x1 = (row[0] - row[2] * 0.5f) * orig_w;
                float y1 = (row[1] - row[3] * 0.5f) * orig_h;
                float x2 = (row[0] + row[2] * 0.5f) * orig_w;
                float y2 = (row[1] + row[3] * 0.5f) * orig_h;
                l_boxes.push_back({x1, y1, x2, y2});
                l_scores.push_back(max_s);
                l_ids.push_back(cls);
            }
        }
        #pragma omp critical
        {
            boxes.insert(boxes.end(), l_boxes.begin(), l_boxes.end());
            scores.insert(scores.end(), l_scores.begin(), l_scores.end());
            class_ids.insert(class_ids.end(), l_ids.begin(), l_ids.end());
        }
    }

    auto keep = nms(boxes, scores, iou_thres);
    auto result = py::array_t<float>({(py::ssize_t)keep.size(), (py::ssize_t)6});
    float* res_ptr = static_cast<float*>(result.request().ptr);

    for (size_t i = 0; i < keep.size(); ++i) {
        int idx = keep[i];
        res_ptr[i * 6 + 0] = boxes[idx].x1;
        res_ptr[i * 6 + 1] = boxes[idx].y1;
        res_ptr[i * 6 + 2] = boxes[idx].x2;
        res_ptr[i * 6 + 3] = boxes[idx].y2;
        res_ptr[i * 6 + 4] = scores[idx];
        res_ptr[i * 6 + 5] = (float)class_ids[idx];
    }
    return result;
}
