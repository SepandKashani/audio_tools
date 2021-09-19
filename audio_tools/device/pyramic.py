import io
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
        assert 0 < pkt_size <= 29127, "Parameter[pkt_size] is out of bounds."
        super().__init__(
            dtype=np.dtype([("id", ">u1"), ("data", ">i2", (pkt_size, 48))]),
            sample_rate=48000,
        )

        self._pci_offset = pci
        self._rma_offset = rma

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

            # PCI/RMA frontend
            self._mem = None
            self._pci = None
            self._rma = None
            self._data = dict[str, np.ndarray]()

        def run(self):
            try:
                self._setup()
                self._acquire()
                self._sample_audio()
            except Exception:
                pass
            finally:
                self._cleanup()

        def _setup(self):
            self._mem = io.open("/dev/mem", mode="r+b")

            q, r = divmod(self._pyramic._pci_offset, mmap.ALLOCATIONGRANULARITY)
            self._pci = mmap.mmap(
                self._mem.fileno(),
                length=r + 8 * 4,  # 4-byte words (8x)
                access=mmap.ACCESS_READ | mmap.ACCESS_WRITE,
                offset=q * mmap.ALLOCATIONGRANULARITY,
            )
            fields = [
                "REG_COMMAND",
                "REG_STATUS",
                "REG_BUF_ACTIVE",
                "REG_USED_BYTES",
                "REG_DMA_BUF_LENGTH_BYTES",
                "DMA_BUF_0_ADDR",
                "DMA_BUF_1_ADDR",
                "DMA_BUF_2_ADDR",
            ]
            for i, k in enumerate(fields):
                self._data[k] = np.ndarray(
                    shape=(1,),
                    dtype=np.dtype(dict(big=">", little="<")[sys.byteorder] + "u4"),
                    buffer=self._pci,
                    offset=r + 4 * i,
                )

            q, r = divmod(self._pyramic._rma_offset, mmap.ALLOCATIONGRANULARITY)
            self._rma = mmap.mmap(
                self._mem.fileno(),
                length=r + 8 * (2 ** 20),  # 8MB region
                access=mmap.ACCESS_READ,
                offset=q * mmap.ALLOCATIONGRANULARITY,
            )
            self._data["BUF"] = np.ndarray(
                shape=(3,),  # triple-buffered design
                dtype=self._pyramic.dtype["data"],
                buffer=self._rma,
                offset=r,
                order="C",
            )

        def _acquire(self):
            addr = lambda i: self._pyramic._rma_offset + self._pyramic.dtype["data"].itemsize * i
            self._data["REG_DMA_BUF_LENGTH_BYTES"][0] = self._pyramic.dtype["data"].itemsize
            self._data["DMA_BUF_0_ADDR"][0] = addr(0)
            self._data["DMA_BUF_1_ADDR"][0] = addr(1)
            self._data["DMA_BUF_2_ADDR"][0] = addr(2)
            self._data["REG_COMMAND"][0] = 0x1  # START
            self._mem.flush()

        def _sample_audio(self):
            reg = self._data["REG_BUF_ACTIVE"]
            hw_acquire = lambda: reg[0] > 0x0
            wr_buffers = lambda: {0x0: (None, None), 0x1: (0, 2), 0x2: (1, 0), 0x4: (2, 1)}[reg[0]]

            pkt_size, *_ = self._pyramic.dtype["data"].shape  # smpl/pkt
            pkt_rate = self._pyramic.sample_rate / pkt_size  # pkt/s
            period = 0.1 / pkt_rate
            while not hw_acquire():  # sampling not yet begun.
                time.sleep(period)

            pkt = np.zeros((1,), dtype=self._pyramic.dtype)
            wb_prev, _ = wr_buffers()
            while self._pyramic.active() and hw_acquire():
                wb, rb = wr_buffers()
                if wb == None:  # acquisition was stopped (somehow)
                    self._pyramic._active.clear()
                elif wb == wb_prev:  # no new packet yet -> stall
                    time.sleep(period)
                else:  # finished acquiring a packet -> provide to user
                    pkt["id"] += 1
                    pkt["data"][:] = self._data["BUF"][rb]
                    self._pyramic._pkt_q.put(pkt.copy())
                    wb_prev = wb

        def _cleanup(self):
            # reset state to that after PyramicStream.__init__().
            self._pyramic._active.clear()
            self._client._thread = None

            self._data["REG_COMMAND"][0] = 0x2  # STOP
            self._mem.flush()

            self._data = dict()
            self._rma.close()
            self._pci.close()
            self._mem.close()
