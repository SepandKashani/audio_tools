import abc

import numpy as np


class PacketStream:
    """
    Audio packet-stream interface.

    Manages packet acquisition (start/stop) and FIFO packet extraction (get/clear).
    """

    def __init__(self, dtype: np.dtype, sample_rate: int):
        """
        Parameters
        ----------
        dtype: np.dtype
            Packet data format.
        sample_rate: int
            Audio stream sample rate. [unit: sample/s]
            (This is not the packet rate!)

        Notes
        -----
        * dtype must have a 'data' field specified as follows:
            ('data', SAMPLE_DTYPE_SPEC: str, (N_sample: int, N_channel: int))
          Ex: ('data', '>f4', (10, 2)) means:
            * each packet contains 10 temporal samples, each 2 channels wide.
            * each audio sample is encoded in 32-bit big-endian floating point format.
        * dtyes should add sequencing info to packets. Recommended packet format:
            [('id', '>u1'), ('data', SAMPLE_DTYPE_SPEC: str, (N_sample: int, N_channel: int))]
          Reason: Field[id] lets one track if a packet was lost/missed by packet consumers.
        """
        self._dtype = dtype
        self._sample_rate = sample_rate
        self._active = False

    @abc.abstractmethod
    def start(self):
        """
        Start packet acquisition.

        This method must set PacketStream._active to True.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self):
        """
        Stop packet acquisition.

        This method must set PacketStream._active to False.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get(self) -> np.ndarray:
        """
        Pop oldest packet from the stream.

        Blocks until packet is available.
        Behaviour is unspecified if PacketStream.active() returns False.

        Returns
        -------
        pkt: np.ndarray[dtype]
            (1,) audio data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def clear(self):
        """
        Drop buffered packets.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Returns
        -------
        N: int
            Number of buffered packets available.
        """
        raise NotImplementedError

    @property
    def dtype(self) -> np.dtype:
        """
        Returns
        -------
        dtype: np.dtype
            Packet data format.

            The return-value is only considered valid after calling PacketStream.start().
        """
        return self._dtype

    @property
    def sample_rate(self) -> int:
        """
        Returns
        -------
        sample_rate: int
            Audio stream sample rate. [unit: sample/s]

            The return-value is only considered valid after calling PacketStream.start().
        """
        return self._sample_rate

    def active(self) -> bool:
        """
        Returns
        -------
        acquiring: bool
            True if packets are being acquired (i.e. PacketStream.start() called), otherwise False.
        """
        return self._active
