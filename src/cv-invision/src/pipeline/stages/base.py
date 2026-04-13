import threading
import queue
from abc import ABC, abstractmethod
from src.utils.log import get_logger

logger = get_logger(__name__)


class BaseStage(ABC):
    """
    Base class for all pipeline stages.
    Each stage runs in its own thread, reads from
    an input queue and writes to an output queue.

    Subclasses must implement:
        process(item) → processed item or None to drop
    """

    def __init__(
        self,
        name        : str,
        in_queue    : queue.Queue = None,
        out_queue   : queue.Queue = None,
        drop_policy : str         = "drop_oldest",
    ):
        self.name        = name
        self.in_queue    = in_queue
        self.out_queue   = out_queue
        self.drop_policy = drop_policy
        self._stop_event = threading.Event()
        self._thread     = threading.Thread(
            target = self._run,
            name   = name,
            daemon = True   # ← dies when main thread dies
        )

    def start(self):
        logger.info(f"▶️  Stage '{self.name}' starting...")
        self._thread.start()

    def stop(self):
        logger.info(f"⏹️  Stage '{self.name}' stopping...")
        self._stop_event.set()

    def join(self):
        self._thread.join()

    def _put(self, item):
        """
        Puts item into output queue respecting drop policy.
        """
        if self.out_queue is None:
            return

        if self.drop_policy == "drop_oldest":
            # if full drop oldest to make room
            if self.out_queue.full():
                try:
                    self.out_queue.get_nowait()
                    logger.debug(f"  [{self.name}] dropped oldest frame")
                except queue.Empty:
                    pass
            self.out_queue.put_nowait(item)

        elif self.drop_policy == "drop_newest":
            try:
                self.out_queue.put_nowait(item)
            except queue.Full:
                logger.debug(f"  [{self.name}] dropped newest frame")

        elif self.drop_policy == "block":
            self.out_queue.put(item, timeout=1.0)

    def _run(self):
        """Main thread loop — reads from in_queue, calls process(), writes to out_queue."""
        logger.info(f"✅ Stage '{self.name}' running...")

        while not self._stop_event.is_set():
            try:
                # get item from input queue
                item = self.in_queue.get(timeout=1.0) if self.in_queue else None

                # sentinel = shutdown signal
                if item is None:
                    logger.info(f"🛑 Stage '{self.name}' received sentinel — shutting down")
                    self._put(None)  # pass sentinel downstream
                    break

                # process item
                result = self.process(item)

                # push result downstream
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
        """
        Process a single item from the input queue.

        Args:
            item: Whatever the previous stage put in the queue

        Returns:
            Processed result to pass downstream, or None to drop the frame
        """
        ...
