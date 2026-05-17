import cv2
from abc import ABC, abstractmethod
from cv_invision.utils.log import get_logger

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



import cv2

class PiCameraSource(BaseSource):
    """Reads frames natively from Raspberry Pi cameras with true maximum FOV and soft pink colors."""
    def __init__(self):
        try:
            from picamera2 import Picamera2
        except ImportError:
            raise ImportError("picamera2 is required for PiCameraSource. Install it first.")
        
        self.picam2 = Picamera2()
        
        # 1. Generate standard video configuration targeting your high-res uncropped matrix
        # Using 3280x2464 forces the hardware to capture the entire wide-angle (0,0) sensor canvas
        config = self.picam2.create_video_configuration(
            main={"format": "RGB888", "size": (3280, 2464)}
        )
        
        # 2. Apply and kick off hardware streams
        self.picam2.configure(config)
        self.picam2.start()
        logger.info("📷 PiCameraSource: Native uncropped wide-angle canvas initialized (3280x2464 -> 640x640 processing pipeline)")

    def read(self):
        try:
            # Captures full resolution frame straight from the ISP
            frame = self.picam2.capture_array()
            
            if frame is None:
                return False, None
            
            # 3. Clean resize directly down to your square 640x640 resolution
            # This completely bypasses the internal hardware crop and gives you huge FOV!
            frame_resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
                
            return True, frame_resized
        except Exception as e:
            logger.error(f"PiCamera read failed: {e}")
            return False, None

    def release(self):
        try:
            self.picam2.stop()
            self.picam2.close()
        except Exception:
            pass

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
