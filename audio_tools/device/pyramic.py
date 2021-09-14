import mmap
import threading
import time

import numpy as np
import numpy.random as npr

import audio_tools.interface as ati


class PyramicStream(ati.PacketStream):
    """
    Pyramic 48-element microphone stream [1].

    Instantiating this object requires root privileges.

    [1] https://github.com/sahandKashani/Pyramic_v2.git
    """

    def __init__(self, pci: int, rma: int, pkt_size: int):
        """
        Parameters
        ----------
        pci: int
            Pyramic Control Interface (PCI) descriptor.
            Expected: byte offset from start of /dev/mem.
        rma: int
            Reserved Memory Area (RMA, 8MB) descriptor.
            Corresponds to the location where audio samples are DMA-ed by the Pyramic.
            Expected: byte offset from start of /dev/mem.
        pkt_size: int
            Number of 48-element microphone samples per buffer.
            Upper-bounded to 29127 samples.
            (Reason: 29127 < 8 [MByte/RMA] / 3 [buffer/RMA] / (2*48 [byte/sample]))

        Notes
        -----
        Description of the mmap-ed HW registers to read/write in the PCI can be found in
        ./Pyramic_v2/hw/hdl/mic_48/hdl/control_interface.vhd
        """
        super().__init__(dtype=np.dtype("<i2"), sample_rate=48000)
        assert 0 < pkt_size <= 29127, "Parameter[pkt_size] is out of bounds."

        self._pci = pci
        self._rma = rma
        self._pkt_size = pkt_size

        self._thread = None
        self._pkt_q = queue.Queue()
        self._active = threading.Event()  # thread start/stop synchronization

    def start(self):
        if self.active():
            pass
        else:
            self._active.set()
            self._thread = PyramicStream._BufferSampler(self)
            self._thread.start()

    def stop(self):
        self._active.clear()

    def get(self) -> np.ndarray:
        return self._pkt_q.get()

    def clear(self):
        try:
            while True:
                self._pkt_q.get(block=False)
                self._pkt_q.task_done()
        except queue.Empty:
            pass

    def __len__(self) -> int:
        return self._pkt_q.qsize()

    def active(self) -> bool:
        return self._active.is_set()

    class _BufferSampler(threading.Thread):
        def __init__(self, pyramic: "PyramicStream"):
            super().__init__()
            self._pyramic = pyramic

        def run(self):
            try:
                self._synchronize()
                self._acquire_audio()
            except Exception:
                pass
            finally:
                self._cleanup()

        def _synchronize(self):
            pass

        def _acquire_audio(self):
            pass

        def _cleanup(self):
            # reset state to that after PyramicStream.__init__().
            self._pyramic._active.clear()
            self._client._thread = None
