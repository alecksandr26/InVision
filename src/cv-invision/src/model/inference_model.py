import numpy as np
import cv2
import os

from .const import UNIFIED_CLASSES
from ..utils.log import get_logger

logger = get_logger(__name__)

# Try to import the C++ post-processing module
try:
    from .postprocess_cpp import postprocess as cpp_postprocess
    CPP_POSTPROCESS_AVAILABLE = True
    logger.info("✓ C++ post-processing module loaded")
except ImportError as e:
    CPP_POSTPROCESS_AVAILABLE = False
    logger.warning(f"⚠ C++ post-processing not available, using Python fallback: {e}")

# =============================================================================
# MOCK CLASSES (unchanged – they consume the same dict format)
# =============================================================================

class MockTensor:
    __slots__ = ('_array', 'shape')
    def __init__(self, data):
        self._array = np.array(data, dtype=np.float32)
        self.shape = self._array.shape
    def cpu(self): return self
    def numpy(self): return self._array
    def tolist(self): return self._array.tolist()
    def __getitem__(self, idx): return self._array[idx]
    def __len__(self): return len(self._array)
    def __repr__(self): return f"MockTensor({self._array.tolist()})"

class MockBoxes:
    __slots__ = ('_detections', '_count', 'xyxy', 'conf', 'cls')
    def __init__(self, detections):
        self._detections = detections
        self._count = len(detections)
        if self._count:
            boxes = [d["box"] for d in detections]
            confs = [d["confidence"] for d in detections]
            clss  = [d["class_id"] for d in detections]
            self.xyxy = MockTensor(boxes)
            self.conf = MockTensor(confs)
            self.cls  = MockTensor(clss)
        else:
            self.xyxy = MockTensor(np.empty((0, 4)))
            self.conf = MockTensor([])
            self.cls  = MockTensor([])
    def __len__(self): return self._count
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return MockBoxes([self._detections[idx]])
        return self
    def __iter__(self):
        for i in range(self._count):
            yield self[i]

class MockResults:
    __slots__ = ('boxes', 'names')
    def __init__(self, detections, names_dict):
        self.boxes = MockBoxes(detections)
        self.names = names_dict
    def __repr__(self):
        nboxes = len(self.boxes)
        if nboxes == 0:
            return "InVision Results: No objects detected."
        lines = [f"InVision Results ({nboxes} objects):"]
        for i, box in enumerate(self.boxes):
            label = self.names.get(int(box.cls[0]), "Unknown")
            coords = box.xyxy[0].astype(int).tolist()
            lines.append(f"  [{i}] {label:12} | Conf: {box.conf[0]:.2%} | Box: {coords}")
        return "\n".join(lines)


# =============================================================================
# TFLITE INFERENCE MODEL (Python TFLite + C++ post-processing)
# =============================================================================

class TFLiteInferenceModel:
    def __init__(self, model_path, num_threads=4, conf_thres=0.25, iou_thres=0.45):
        logger.info(f"🔄 Initializing TFLite interpreter with {num_threads} threads")
        
        # We always use Python TFLite inference (fallback), but post‑processing may be C++
        self._cpp_mode = False   # No full C++ engine
        
        self._init_fallback(model_path, num_threads, conf_thres, iou_thres)
        
        logger.info("🐍 Using Python TFLite inference engine")
        if CPP_POSTPROCESS_AVAILABLE:
            logger.info("⚡ Using C++ post-processing (fast)")
        else:
            logger.info("🐍 Using Python post-processing (slow)")

    def _init_fallback(self, model_path, num_threads, conf_thres, iou_thres):
        from tensorflow import lite as tflite
        logger.info(f"🔧 Forced using tensorflow.lite backend (x86/ARM)")
        logger.info(f"🧵 Initializing TFLite interpreter with {num_threads} threads")
        
        self.interpreter = tflite.Interpreter(
            model_path=str(model_path),
            num_threads=num_threads,
        )
        self.interpreter.allocate_tensors()
        in_details = self.interpreter.get_input_details()[0]
        out_details = self.interpreter.get_output_details()[0]
        self._input_idx = in_details['index']
        self._output_idx = out_details['index']
        self.input_h, self.input_w = in_details['shape'][1], in_details['shape'][2]

        self._set_tensor = self.interpreter.set_tensor
        self._invoke = self.interpreter.invoke
        self._get_tensor = self.interpreter.get_tensor

        self.names = {i: name for i, name in enumerate(UNIFIED_CLASSES)}
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def _preprocess(self, frame):
        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.expand_dims(img.astype(np.float32) / 255.0, axis=0)

    def _postprocess_python(self, output_data, orig_h, orig_w):
        """Original Python post‑processing (fallback)"""
        pred = output_data[0].T
        conf_th = self.conf_thres
        scores = np.max(pred[:, 4:], axis=1)
        mask = scores > conf_th
        if not np.any(mask):
            return []
        pred = pred[mask]
        scores = scores[mask]
        class_ids = np.argmax(pred[:, 4:], axis=1)
        boxes_norm = pred[:, :4]
        x1 = (boxes_norm[:, 0] - boxes_norm[:, 2] * 0.5) * orig_w
        y1 = (boxes_norm[:, 1] - boxes_norm[:, 3] * 0.5) * orig_h
        w_px = boxes_norm[:, 2] * orig_w
        h_px = boxes_norm[:, 3] * orig_h
        cv_boxes = [[x1[i], y1[i], w_px[i], h_px[i]] for i in range(len(x1))]
        indices = cv2.dnn.NMSBoxes(cv_boxes, scores.tolist(), conf_th, self.iou_thres)
        if indices is None or len(indices) == 0:
            return []
        final = []
        for idx in indices.flatten():
            bx, by, bw, bh = cv_boxes[idx]
            final.append({
                "box": [int(bx), int(by), int(bx + bw), int(by + bh)],
                "confidence": float(scores[idx]),
                "class_id": int(class_ids[idx])
            })
        return final

    def __call__(self, frame, verbose=False):
        # 1. Preprocess
        input_data = self._preprocess(frame)
        # 2. Run TFLite inference
        self._set_tensor(self._input_idx, input_data)
        self._invoke()
        output_data = self._get_tensor(self._output_idx)   # numpy array

        # 3. Post-process – use C++ if available, else Python
        if CPP_POSTPROCESS_AVAILABLE:
            # C++ function expects (output_array, orig_h, orig_w, conf_thres, iou_thres)
            detections = cpp_postprocess(
                output_data, frame.shape[0], frame.shape[1],
                self.conf_thres, self.iou_thres
            )
        else:
            detections = self._postprocess_python(output_data, frame.shape[0], frame.shape[1])

        results = [MockResults(detections, self.names)]

        if verbose:
            print(results[0])
        return results
