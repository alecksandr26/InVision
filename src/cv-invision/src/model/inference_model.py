import numpy as np
import cv2
from .const import UNIFIED_CLASSES
from ..utils.log import get_logger

logger = get_logger(__name__)

# =============================================================================
# OPTIMIZED MOCK CLASSES (__slots__ reduces memory & attribute lookup)
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
            # Build lists in one pass (avoid repeated list comprehensions)
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
# EXTREMELY OPTIMIZED TFLITE INFERENCE ENGINE
# =============================================================================

class TFLiteInferenceModel:
    __slots__ = (
        'interpreter', 'input_h', 'input_w', 'names', 'conf_thres', 'iou_thres',
        '_set_tensor', '_invoke', '_get_tensor', '_input_idx', '_output_idx'
    )
    def __init__(self, model_path, num_threads=4, conf_thres=0.25, iou_thres=0.45):
        from tensorflow import lite as tflite

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

        # Cache interpreter methods to avoid attribute lookups inside __call__
        self._set_tensor = self.interpreter.set_tensor
        self._invoke = self.interpreter.invoke
        self._get_tensor = self.interpreter.get_tensor

        self.names = {i: name for i, name in enumerate(UNIFIED_CLASSES)}
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def _preprocess(self, frame):
        # Direct resize to cached dimensions
        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalise and expand dims in one expression (avoids extra .astype call)
        return np.expand_dims(img.astype(np.float32) / 255.0, axis=0)

    def _postprocess(self, output_data, orig_h, orig_w):
        # output_data shape: (1, 14, 8400)  ->  take [0] and transpose
        pred = output_data[0].T                     # (8400, 14)
        conf_th = self.conf_thres

        # Fast confidence filtering
        # Note: predictions[:, 4:] -> classes start at index 4 (for YOLO)
        scores = np.max(pred[:, 4:], axis=1)        # (8400,)
        mask = scores > conf_th
        if not np.any(mask):
            return []

        pred = pred[mask]
        scores = scores[mask]

        # Class IDs (int)
        class_ids = np.argmax(pred[:, 4:], axis=1)  # (N,)

        # Normalized boxes [xc, yc, w, h]
        boxes_norm = pred[:, :4]                    # (N, 4)

        # Vectorised pixel‑scale conversion (identical arithmetic)
        x1 = (boxes_norm[:, 0] - boxes_norm[:, 2] * 0.5) * orig_w
        y1 = (boxes_norm[:, 1] - boxes_norm[:, 3] * 0.5) * orig_h
        w_px = boxes_norm[:, 2] * orig_w
        h_px = boxes_norm[:, 3] * orig_h

        # Build list of [x, y, w, h] for OpenCV NMS (list of lists of floats)
        # Using list comprehension directly is slightly faster than np.column_stack + tolist
        cv_boxes = [[x1[i], y1[i], w_px[i], h_px[i]] for i in range(len(x1))]

        # NMS – OpenCV’s C++ implementation is fastest; pass Python lists
        indices = cv2.dnn.NMSBoxes(cv_boxes, scores.tolist(), conf_th, self.iou_thres)

        if indices is None or len(indices) == 0:
            return []

        # Build final detections (exactly the same integer conversion)
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
        # Preprocess
        input_data = self._preprocess(frame)
        self._set_tensor(self._input_idx, input_data)
        self._invoke()
        output_data = self._get_tensor(self._output_idx)

        # Postprocess – pass original height, width
        detections = self._postprocess(output_data, frame.shape[0], frame.shape[1])
        results = [MockResults(detections, self.names)]

        if verbose:
            print(results[0])
        return results
