import threading
import queue
from abc import ABC, abstractmethod
from src.utils.log import get_logger

logger = get_logger(__name__)

class BaseStage(ABC):
    def __init__(self, name: str, in_queue: queue.Queue = None, out_queue: queue.Queue = None, drop_policy: str = "drop_oldest"):
        self.name        = name
        self.in_queue    = in_queue
        self.out_queue   = out_queue
        self.drop_policy = drop_policy
        self._stop_event = threading.Event()
        self._thread     = threading.Thread(target=self._run, name=name, daemon=True)

    def start(self):
        logger.info(f"▶️  Stage '{self.name}' starting...")
        self._thread.start()

    def stop(self):
        logger.info(f"⏹️  Stage '{self.name}' stopping...")
        self._stop_event.set()

    def join(self):
        self._thread.join()

    def _put(self, item):
        if self.out_queue is None:
            return

        if self.drop_policy == "drop_oldest":
            try:
                self.out_queue.put_nowait(item)
            except queue.Full:
                try:
                    self.out_queue.get_nowait()
                except queue.Empty:
                    pass
                self.out_queue.put_nowait(item)
                logger.debug(f"  [{self.name}] dropped oldest frame")

        elif self.drop_policy == "drop_newest":
            try:
                self.out_queue.put_nowait(item)
            except queue.Full:
                logger.debug(f"  [{self.name}] dropped newest frame")

        elif self.drop_policy == "block":
            # Fix: Loop and check stop event to prevent shutdown deadlocks
            while not self._stop_event.is_set():
                try:
                    self.out_queue.put(item, timeout=0.2)
                    break
                except queue.Full:
                    continue

    def _run(self):
        logger.info(f"✅ Stage '{self.name}' running...")
        while not self._stop_event.is_set():
            try:
                item = self.in_queue.get(timeout=1.0) if self.in_queue else None

                if item is None:
                    logger.info(f"🛑 Stage '{self.name}' received sentinel — shutting down")
                    self._put(None) 
                    break

                result = self.process(item)

                if result is not None:
                    self._put(result)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Stage '{self.name}' error: {e}")
                raise

        logger.info(f"✅ Stage '{self.name}' done.")

    @abstractmethod
    def process(self, item):
        ...
