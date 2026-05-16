import cv2
from abc import ABC, abstractmethod
from src.utils.log import get_logger

logger = get_logger(__name__)


class BaseSource(ABC):
    """Abstract frame source — same interface for all inputs."""
    @abstractmethod
    def read(self) -> tuple[bool, any]:
        ...
    @abstractmethod
    def release(self):
        ...
    @property
    @abstractmethod
    def fps(self) -> float:
        ...
    @property
    @abstractmethod
    def total_frames(self) -> int:
        ...


class VideoFileSource(BaseSource):
    def __init__(self, path: str):
        self.path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video: '{path}'")
        logger.info(f"📹 VideoFileSource: '{path}'")
        logger.info(f"⚙️  FPS: {self.fps} | Frames: {self.total_frames}")

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()

    @property
    def fps(self):
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def total_frames(self):
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))


class CameraSource(BaseSource):
    def __init__(self, device_id: int = 0):
        self._cap = cv2.VideoCapture(device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera: {device_id}")
        logger.info(f"📷 CameraSource: device {device_id}")

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()

    @property
    def fps(self):
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def total_frames(self):
        return -1  


class PiCameraSource(BaseSource):
    """Reads frames natively from Raspberry Pi cameras."""
    def __init__(self):
        try:
            from picamera2 import Picamera2
        except ImportError:
            raise ImportError("picamera2 is required for PiCameraSource. Install it first.")
        
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(main={"format": "RGB888", "size": (640, 480)})
        self.picam2.configure(config)
        self.picam2.start()
        logger.info("📷 PiCameraSource: native raspi camera initialized")

    def read(self):
        try:
            frame = self.picam2.capture_array()
            # Convert to BGR immediately so YOLO and OpenCV downstream work flawlessly
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame_bgr
        except Exception as e:
            logger.error(f"PiCamera read failed: {e}")
            return False, None

    def release(self):
        self.picam2.stop()
        self.picam2.close()

    @property
    def fps(self):
        return 30.0

    @property
    def total_frames(self):
        return -1


class StreamSource(BaseSource):
    def __init__(self, url: str):
        self._cap = cv2.VideoCapture(url)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open stream: '{url}'")
        logger.info(f"📡 StreamSource: '{url}'")

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()

    @property
    def fps(self):
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def total_frames(self):
        return -1
