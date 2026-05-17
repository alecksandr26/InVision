import threading
import queue
from abc import ABC, abstractmethod
from cv_invision.utils.log import get_logger

logger = get_logger(__name__)

class BaseStage(ABC):
    def __init__(self, name: str, in_queue: queue.Queue = None, out_queue: queue.Queue = None, drop_policy: str = "drop_oldest"):
        self.name        = name
        self.in_queue    = in_queue
        self.out_queue   = out_queue
        self.drop_policy = drop_policy
        self._stop_event = threading.Event()
        self._thread     = threading.Thread(target=self._run, name=name, daemon=True)
        
        # New: Registry for external callback functions
        self._callbacks = []

    def register_callback(self, callback_fn):
        """Allows central_control to inject processing hooks dynamically."""
        if callable(callback_fn):
            self._callbacks.append(callback_fn)
            logger.info(f"⚓ Callback registered to stage: {self.name}")
        else:
            raise ValueError("Provided callback must be executable.")

    def start(self):
        logger.info(f"▶️  Stage '{self.name}' starting...")
        self._thread.start()

    def stop(self):
        logger.info(f"⏹️  Stage '{self.name}' stopping...")
        self._stop_event.set()

    def join(self):
        self._thread.join()

    def _put(self, item):
        # ... (Keep your original drop_policy logic perfectly intact here) ...
        if self.out_queue is None: return
        if self.drop_policy == "drop_oldest":
            try: self.out_queue.put_nowait(item)
            except queue.Full:
                try: self.out_queue.get_nowait()
                except queue.Empty: pass
                self.out_queue.put_nowait(item)
        elif self.drop_policy == "drop_newest":
            try: self.out_queue.put_nowait(item)
            except queue.Full: pass
        elif self.drop_policy == "block":
            while not self._stop_event.is_set():
                try:
                    self.out_queue.put(item, timeout=0.2)
                    break
                except queue.Full: continue

    def _run(self):
        logger.info(f"✅ Stage '{self.name}' running...")
        while not self._stop_event.is_set():
            try:
                item = self.in_queue.get(timeout=1.0) if self.in_queue else None

                if item is None:
                    logger.info(f"🛑 Stage '{self.name}' received sentinel — shutting down")
                    self._put(None) 
                    break

                # Execute original stage processing logic (YOLO or DeepSort)
                result = self.process(item)

                if result is not None:
                    # New: Run any registered central control hooks before pushing downstream
                    for callback in self._callbacks:
                        try:
                            callback(result)
                        except Exception as e:
                            logger.error(f"❌ Error in stage [{self.name}] callback: {e}")
                    
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
