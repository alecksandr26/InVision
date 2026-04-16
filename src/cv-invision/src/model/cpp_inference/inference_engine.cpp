#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "nms.hpp"
#include "resize.hpp"

namespace py = pybind11;

using namespace pybind11::literals;


class InferenceEngine {
public:
  InferenceEngine(const std::string& model_path, int num_threads,
		  float conf_thres, float iou_thres)
    : conf_thres_(conf_thres), iou_thres_(iou_thres) {
    // Load TFLite model
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model_) throw std::runtime_error("Failed to load model");
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model_, resolver)(&interpreter_);
    if (!interpreter_) throw std::runtime_error("Failed to build interpreter");
    interpreter_->SetNumThreads(num_threads);
    interpreter_->AllocateTensors();

    // Get input details
    input_ = interpreter_->typed_input_tensor<float>(0);
    auto* dims = interpreter_->tensor(interpreter_->inputs()[0])->dims;
    input_h_ = dims->data[1];
    input_w_ = dims->data[2];
    // Assume output shape: [1, 14, 8400] (YOLO style)
    output_ = interpreter_->typed_output_tensor<float>(0);
  }

  std::vector<py::dict> run_inference(py::array_t<uint8_t> image) {
    // 1. Preprocess
    auto buf = image.request();
    if (buf.ndim != 3) throw std::runtime_error("Image must be 3D (H,W,C)");
    int orig_h = buf.shape[0];
    int orig_w = buf.shape[1];
    uint8_t* img_data = static_cast<uint8_t*>(buf.ptr);
    // Convert BGR->RGB if needed (your camera gives BGR? we assume RGB input)
    // We'll keep it as is – user must provide RGB. Or we add conversion.
    auto input_tensor = resize_rgb(img_data, orig_w, orig_h, input_w_, input_h_);
    std::memcpy(input_, input_tensor.data(), input_tensor.size() * sizeof(float));

    // 2. Run inference
    interpreter_->Invoke();

    // 3. Postprocess
    // output shape: [1, 14, 8400] – YOLO variant
    float* out = output_;
    int num_dets = 8400; // hardcoded for this model, can be read from tensor shape
    std::vector<float> scores;
    std::vector<Box> boxes;
    std::vector<int> class_ids;

    for (int i = 0; i < num_dets; ++i) {
      float max_score = 0.0f;
      int cls = -1;
      for (int c = 0; c < 13; ++c) { // assuming 13 classes (or 14 total)
	float score = out[i * 14 + 4 + c]; // [xc, yc, w, h, class_scores...]
	if (score > max_score) { max_score = score; cls = c; }
      }
      if (max_score > conf_thres_) {
	float xc = out[i * 14 + 0];
	float yc = out[i * 14 + 1];
	float w = out[i * 14 + 2];
	float h = out[i * 14 + 3];
	// denormalize
	float x1 = (xc - w * 0.5f) * orig_w;
	float y1 = (yc - h * 0.5f) * orig_h;
	float x2 = (xc + w * 0.5f) * orig_w;
	float y2 = (yc + h * 0.5f) * orig_h;
	boxes.push_back({x1, y1, x2, y2});
	scores.push_back(max_score);
	class_ids.push_back(cls);
      }
    }

    // NMS
    auto keep = nms(boxes, scores, iou_thres_);

    // Build Python dicts
    std::vector<py::dict> detections;
    for (int idx : keep) {
      const auto& b = boxes[idx];
      detections.push_back(py::dict(
				    "box"_a = py::make_tuple((int)b.x1, (int)b.y1, (int)b.x2, (int)b.y2),
				    "confidence"_a = scores[idx],
				    "class_id"_a = class_ids[idx]
				    ));
    }
    return detections;
  }

private:
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  float* input_;
  float* output_;
  int input_h_, input_w_;
  float conf_thres_, iou_thres_;
};

// Pybind11 module
PYBIND11_MODULE(inference_engine_cpp, m) {
  py::class_<InferenceEngine>(m, "InferenceEngine")
    .def(py::init<const std::string&, int, float, float>(),
	 py::arg("model_path"), py::arg("num_threads")=4,
	 py::arg("conf_thres")=0.25, py::arg("iou_thres")=0.45)
    .def("run_inference", &InferenceEngine::run_inference);
}
