import queue

import numpy as np

import audio_tools.interface as ati


class PacketServer(ati.PacketStream):
    """
    Route packets from a PacketStream to different parties through a TCP interface.

    This class is designed to dispatch available packets to each destination as soon as possible.
    """

    def __init__(self, stream: ati.PacketStream, port: int):
        """
        Parameters
        ----------
        stream: PacketStream
            Packet source.
        port: int
            TCP listening port.
        """
        super().__init__(dtype=None, sample_rate=None)  # not yet known
        # TODO
        self._stream = stream
        self._port = port

    def start(self):
        # TODO
        raise NotImplementedError

    def stop(self):
        # TODO
        raise NotImplementedError

    def get(self) -> np.ndarray:
        raise NotImplementedError("Operation not supported.")

    def clear(self):
        raise NotImplementedError("Operation not supported.")

    def __len__(self) -> int:
        raise NotImplementedError("Operation not supported.")


class PacketClient(ati.PacketStream):
    """
    Get packets from a PacketServer.
    """

    def __init__(self, host: str, port: int):
        """
        Parameters
        ----------
        host: str
            PacketServer hostname/IP-address.
        port: int
            PacketServer port.
        """
        super().__init__(dtype=None, sample_rate=None)  # not yet known
        # TODO
        self._queue = queue.Queue()

    def start(self):
        # TODO
        raise NotImplementedError

    def stop(self):
        # TODO
        raise NotImplementedError

    def get(self) -> np.ndarray:
        return self._queue.get()

    def clear(self):
        try:
            while True:
                self._queue.get(block=False)
                self._queue.task_done()
        except queue.Empty:
            pass

    def __len__(self) -> int:
        return self._queue.qsize()
