import time

import numpy as np

import audio_tools.interface as ati


class PureToneStream(ati.PacketStream):
    """
    Simulated multi-channel pure-tone stream.
    """

    def __init__(self, dtype: np.dtype, sample_rate: int, fc: float):
        """
        Parameters
        ----------
        dtype: np.dtype
            Packet data format. Must follow the recommended packet format described in
            PacketStream.__init__().
        sample_rate: int
            Audio stream sample rate. [unit: sample/s]
            (This is not the packet rate!)
        fc: float
            Tone frequency. [Hz]
        """
        super().__init__(dtype=dtype, sample_rate=sample_rate)
        self._fc = fc

    def start(self):
        rng = np.random.default_rng()
        N_sample, N_channel = self.dtype["data"].shape
        self._t = (2 * np.pi * self._fc / self.sample_rate) * np.arange(N_sample)
        self._phase = rng.uniform(0, 2 * np.pi, size=(N_channel,))
        self._id = 0
        self._active = True
        self._time = time.time()  # last sample time

    def stop(self):
        self._active = False

    def get(self) -> np.ndarray:
        pkt_size, *_ = self.dtype["data"].shape  # smpl/pkt
        pkt_rate = self.sample_rate / pkt_size  # pkt/s
        t_interval = time.time() - self._time
        t_wait = 1 / pkt_rate - t_interval
        if t_wait > 0:  # enforce rate limiting
            time.sleep(t_wait)
        self._time = time.time()

        pkt = np.zeros((1,), dtype=self.dtype)

        pkt["id"] = self._id
        self._id += 1

        pkt["data"] = np.cos(self._t.reshape((-1, 1)) + self._phase)
        self._phase = np.fmod(  # phase gain of last packet + 1 step
            self._phase + self._t[-1] + self._t[1],
            2 * np.pi,
        )

        return pkt

    def clear(self):
        pass

    def __len__(self) -> int:
        return 1  # infinite stream, any value works.
