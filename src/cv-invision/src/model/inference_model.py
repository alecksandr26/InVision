import numpy as np
import cv2
import os

from .const import UNIFIED_CLASSES
from ..utils.log import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Environment variables for runtime configuration
# ----------------------------------------------------------------------
# Force disable C++ modules (for debugging or fallback)
FORCE_PYTHON_PREPROCESS = os.environ.get("INVISION_FORCE_PYTHON_PREPROCESS", "").lower() in ("1", "true", "yes")
FORCE_PYTHON_POSTPROCESS = os.environ.get("INVISION_FORCE_PYTHON_POSTPROCESS", "").lower() in ("1", "true", "yes")

# Override TFLite interpreter threads (default: auto from detection_model)
TFLITE_NUM_THREADS = os.environ.get("INVISION_TFLITE_NUM_THREADS")
if TFLITE_NUM_THREADS is not None:
    TFLITE_NUM_THREADS = int(TFLITE_NUM_THREADS)

# Disable XNNPACK delegate (may help on some x86 systems)
TFLITE_DISABLE_XNNPACK = os.environ.get("INVISION_TFLITE_DISABLE_XNNPACK", "").lower() in ("1", "true", "yes")

# Try to import C++ preprocessing
if not FORCE_PYTHON_PREPROCESS:
    try:
        from .cpp_inference import preprocess as cpp_preprocess
        CPP_PREPROCESS_AVAILABLE = True
        logger.info("✓ C++ preprocessing module loaded")
    except ImportError as e:
        CPP_PREPROCESS_AVAILABLE = False
        logger.warning(f"⚠ C++ preprocessing not available: {e}")
else:
    CPP_PREPROCESS_AVAILABLE = False
    logger.info("🐍 C++ preprocessing disabled by environment variable")

# Try to import the C++ post-processing module
if not FORCE_PYTHON_POSTPROCESS:
    try:
        from .cpp_inference import postprocess as cpp_postprocess
        CPP_POSTPROCESS_AVAILABLE = True
        logger.info("✓ C++ post-processing module loaded")
    except ImportError as e:
        CPP_POSTPROCESS_AVAILABLE = False
        logger.warning(f"⚠ C++ post-processing not available: {e}")
else:
    CPP_POSTPROCESS_AVAILABLE = False
    logger.info("🐍 C++ post-processing disabled by environment variable")

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
        
        
        


    def _init_fallback(self, model_path, num_threads, conf_thres, iou_thres):
        # from tensorflow import lite as tflite
        from ai_edge_litert import interpreter as tflite

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

    def _preprocess_python(self, frame):
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
        # 1. Pre-processing: Use the new C++ OpenCV logic
        if CPP_PREPROCESS_AVAILABLE:
            # The C++ function now returns the full (1, 640, 640, 3) float32 tensor
            input_data = cpp_preprocess(frame, self.input_h, self.input_w)
        else:
            input_data = self._preprocess_python(frame)
            
        # 2. Process - run TFLite inference
        self._set_tensor(self._input_idx, input_data)
        self._invoke()
        # Inside TFLiteInferenceModel.__call__
        output_data = self._get_tensor(self._output_idx)

        if CPP_POSTPROCESS_AVAILABLE:
            # 1. Transpose from (1, 14, 8400) to (8400, 14) for C++ speed
            out_to_pass = np.ascontiguousarray(output_data[0].T)
            
            # 2. Call C++ (now returns a numpy array)
            cpp_arr = cpp_postprocess(
                out_to_pass, frame.shape[0], frame.shape[1],
                self.conf_thres, self.iou_thres
            )
            
            # 3. Safe reconstruction of dictionaries
            detections = []
            for row in cpp_arr:
                detections.append({
                    "box": [int(row[0]), int(row[1]), int(row[2]), int(row[3])],
                    "confidence": float(row[4]),
                    "class_id": int(row[5])
                })
        else:
            detections = self._postprocess_python(output_data, frame.shape[0], frame.shape[1])

        results = [MockResults(detections, self.names)]

        if verbose:
            print(results[0])
        return results
