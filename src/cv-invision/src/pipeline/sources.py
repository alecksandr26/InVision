import cv2
from abc import ABC, abstractmethod
from src.utils.log import get_logger

logger = get_logger(__name__)


class BaseSource(ABC):
    """Abstract frame source — same interface for all inputs."""

    @abstractmethod
    def read(self) -> tuple[bool, any]:
        """Returns (success, frame)"""
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
    """Reads frames from a video file."""

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
    """Reads frames from a camera (Pi camera or USB webcam)."""

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
        return -1  # ← live camera has no total frames


class StreamSource(BaseSource):
    """Reads frames from an RTSP or HTTP stream."""

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
        return -1  # ← stream has no total frames
