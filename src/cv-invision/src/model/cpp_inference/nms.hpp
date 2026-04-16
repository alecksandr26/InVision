#pragma once
#include <vector>
#include <algorithm>
#include <cmath>

struct Box {
  float x1, y1, x2, y2;
};

inline float iou(const Box& a, const Box& b) {
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

inline std::vector<int> nms(const std::vector<Box>& boxes,
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
