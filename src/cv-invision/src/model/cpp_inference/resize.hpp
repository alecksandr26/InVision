#pragma once
#include <vector>
#include <cstdint>

inline std::vector<float> resize_rgb(const uint8_t* src, int src_w, int src_h,
                                     int dst_w, int dst_h) {
  std::vector<float> dst(dst_w * dst_h * 3);
  float x_ratio = static_cast<float>(src_w - 1) / dst_w;
  float y_ratio = static_cast<float>(src_h - 1) / dst_h;
  for (int y = 0; y < dst_h; ++y) {
    float src_y = y * y_ratio;
    int y0 = static_cast<int>(src_y);
    int y1 = std::min(y0 + 1, src_h - 1);
    float dy = src_y - y0;
    for (int x = 0; x < dst_w; ++x) {
      float src_x = x * x_ratio;
      int x0 = static_cast<int>(src_x);
      int x1 = std::min(x0 + 1, src_w - 1);
      float dx = src_x - x0;
      // Interpolate each channel (RGB assumed already)
      for (int c = 0; c < 3; ++c) {
	float v00 = src[(y0 * src_w + x0) * 3 + c];
	float v01 = src[(y0 * src_w + x1) * 3 + c];
	float v10 = src[(y1 * src_w + x0) * 3 + c];
	float v11 = src[(y1 * src_w + x1) * 3 + c];
	float top = v00 * (1 - dx) + v01 * dx;
	float bottom = v10 * (1 - dx) + v11 * dx;
	dst[(y * dst_w + x) * 3 + c] = (top * (1 - dy) + bottom * dy) / 255.0f;
      }
    }
  }
  return dst;
}
