import io
import json
import queue
import socket
import threading

import numpy as np

import audio_tools.interface as ati


class PacketServer(ati.PacketStream):
    """
    Route packets from a PacketStream to several PacketClients through a TCP interface.

    This class is designed to dispatch available packets to each destination as soon as possible.

    Architecture
    ------------
    PacketServer is multi-threaded to efficiently connect the local PacketStream to remote
    PacketClients via:
    * A NetworkListener process which monitors all incoming connection requests;
    * A LocalStreamer process which fan-outs local packets to NetworkStreamer processes; and
    * NetworkStreamer processes (one per PacketClient) which transmit local packets to remote consumers.

    When receiving a new connection request, the NetworkListener spawns a new NetworkStreamer to
    handle the connection. Shared ressources are accessed through a lock when required.

    Note
    ----
    * PacketServer.stop() is not instantaneous due to internal timeouts to avoid thread deadlocks.
    * Some NetworkStreamers may not end after calling PacketServer.stop(). This can happen if the
      remote connection was not made through a PacketClient. This is a non-issue: the OS will
      force-close such zombie connections after a while.
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
        self._stream = stream
        self._port = port

        self._thread: dict[int, threading.Thread] = {}
        self._pkt_q: dict[int, queue.Queue] = {}
        self._q_lck = threading.Lock()  # thread synchronization to modify [_thread | _pkt_q]
        self._active = threading.Event()  # thread start/stop notification

    def start(self):
        if self.active():
            pass
        else:
            if not self._stream.active():  # initiate acquisition
                self._stream.start()
            self._dtype = self._stream.dtype
            self._sample_rate = self._stream.sample_rate
            self._active.set()

            self._thread = {
                0: PacketServer._LocalStreamer(self),
                1: PacketServer._NetworkListener(self),
            }
            for t in self._thread.values():
                t.start()

    def stop(self):
        self._stream.stop()  # stop acquisition
        self._stream.clear()

        self._active.clear()  # gracefully terminate threads
        for t in self._thread.values():
            t.join()
        self._thread = dict()

        for q in self._pkt_q.values():  # and dispose of any lingering state.
            while not q.empty():
                q.get()
                q.task_done()
        self._pkt_q = dict()

    def get(self) -> np.ndarray:
        raise NotImplementedError("Operation not supported.")

    def clear(self):
        raise NotImplementedError("Operation not supported.")

    def __len__(self) -> int:
        raise NotImplementedError("Operation not supported.")

    def active(self) -> bool:
        return self._active.is_set()

    class _LocalStreamer(threading.Thread):
        pass

    class _NetworkStreamer(threading.Thread):
        def __init__(self, srvr: "PacketServer", skt: socket.socket, cid: int):
            super().__init__()
            self._srvr = srvr
            self._skt = skt
            self._cid = cid

        def run(self):
            try:
                self._send_header()
                self._stream_audio()
            except Exception:
                pass
            finally:
                self._cleanup()

        def _send_header(self):
            # Transmit all metadata required for PacketClient to decode audio packets. The header is
            # JSON-encoded (utf-8), prepended with a >u4 describing its length in bytes.
            metadata = json.dumps(
                {
                    "sample_rate": self._srvr.sample_rate,
                    "dtype_descr": self._srvr.dtype.descr,
                }
            ).encode()
            header = len(metadata).to_bytes(4, byteorder="big", signed=False) + metadata
            self._skt.sendall(header)

        def _stream_audio(self):
            q = self._srvr._pkt_q[self._cid]
            pkt_size, *_ = self._srvr.dtype["data"].shape  # smpl/pkt
            pkt_rate = self._srvr.sample_rate / pkt_size  # pkt/s

            skt_alive = True
            while self._srvr.active() and skt_alive:
                try:
                    pkt = q.get(timeout=10 / pkt_rate)
                    q.task_done()
                    self._skt.sendall(bytes(pkt))
                except queue.Empty:
                    # No packet was in the queue despite waiting significantly longer than the
                    # packet arrival rate. Delay reason unknown, but the thread should still run
                    # because PacketClient.get() is designed to block until a packet is available.
                    # The thread should only be taken down if:
                    # * PacketServer.stop() is called;
                    # * the socket is broken/closed. (Ex: PacketClient.stop() called.)
                    pass
                except OSError:
                    # Something went wrong with sendall(). Whatever the cause, the socket is no
                    # longer in a useable state -> end connection.
                    skt_alive = False

        def _cleanup(self):  # cleanup connection + shared resources
            with self._srvr._q_lck:
                self._srvr._thread.pop(self._cid, None)

                q = self._srvr._pkt_q.pop(self._cid, None)
                if q is not None:
                    try:
                        while True:
                            q.get(block=False)
                            q.task_done()
                    except queue.Empty:
                        pass
            self._skt.close()

    class _NetworkListener(threading.Thread):
        pass


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
