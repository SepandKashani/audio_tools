import io
import json
import itertools
import queue
import socket
import threading
import time

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

        self._thread = {}  # dict[int, threading.Thread]
        self._pkt_q = {}  # dict[int, queue.Queue]
        self._cid = itertools.count(start=2)
        self._q_lck = threading.Lock()  # thread synchronization to modify [_thread, _pkt_q, _cid]
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

            # create/launch non-NetworkStreamer threads
            self._thread.update(
                {
                    0: PacketServer._LocalStreamer(self),
                    1: PacketServer._NetworkListener(self),
                }
            )
            for t_id in {0, 1}:
                self._thread[t_id].start()

    def stop(self):
        self._stream.stop()  # stop acquisition
        self._stream.clear()

        self._active.clear()  # gracefully terminate threads
        pkt_size, *_ = self.dtype["data"].shape  # smpl/pkt
        pkt_rate = self.sample_rate / pkt_size  # pkt/s
        for t in list(self._thread.values()):  # guarantees static dict-view.
            # The only threads which may hang are NetworkStreamers. (See Notes above.)
            # Wait a reasonable amount of time for most lingering packets to be sent.
            t.join(timeout=30 / pkt_rate)
        with self._q_lck:  # Discard all state related to dead threads (if any).
            for t_id in [_[0] for _ in self._thread.items() if not _[1].is_alive()]:
                self._thread.pop(t_id, None)
                self._pkt_q.pop(t_id, None)

    def get(self) -> np.ndarray:
        raise NotImplementedError("Operation not supported.")

    def clear(self):
        raise NotImplementedError("Operation not supported.")

    def __len__(self) -> int:
        raise NotImplementedError("Operation not supported.")

    def active(self) -> bool:
        return self._active.is_set()

    class _LocalStreamer(threading.Thread):
        def __init__(self, srvr: "PacketServer"):
            super().__init__()
            self._srvr = srvr

        def run(self):
            pkt_size, *_ = self._srvr.dtype["data"].shape  # smpl/pkt
            pkt_rate = self._srvr.sample_rate / pkt_size  # pkt/s
            stream = self._srvr._stream

            while self._srvr.active():
                if len(stream) > 0:  # required to avoid PacketStream.get()'s blocking behaviour.
                    pkt = stream.get()
                    for q in self._srvr._pkt_q.values():
                        # Note: if no client available, then stream packets are 'lost'. This is
                        # intended behaviour.
                        q.put(pkt)
                else:
                    # No data available at source, but new packet imminent -> stall briefly.
                    time.sleep(0.5 / pkt_rate)

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
        def __init__(self, srvr: "PacketServer"):
            super().__init__()
            self._srvr = srvr

        def run(self):
            with socket.socket() as skt:
                skt.bind(("", self._srvr._port))
                skt.listen(5)

                # Activate non-blocking mode. (Required to avoid socket.accept() freeze.)
                # A conservative timeout period is chosen to avoid wasting resources here.
                pkt_size, *_ = self._srvr.dtype["data"].shape  # smpl/pkt
                pkt_rate = self._srvr.sample_rate / pkt_size  # pkt/s
                skt.settimeout(20 / pkt_rate)

                while self._srvr.active():
                    try:
                        conn, addr = skt.accept()
                        with self._srvr._q_lck:
                            cid = next(self._srvr._cid)
                            t = PacketServer._NetworkStreamer(self._srvr, conn, cid)

                            self._srvr._pkt_q[cid] = queue.Queue()
                            self._srvr._thread[cid] = t
                            t.start()
                    except OSError:
                        pass  # skt.accept() just timed-out -> not a problem.


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
        self._host = host
        self._port = port

        self._thread = None
        self._pkt_q = queue.Queue()
        self._active = threading.Event()  # thread start/stop synchronization

    def start(self):
        if self.active():
            pass
        else:
            conn = socket.socket()
            conn.connect((self._host, self._port))

            self._active.set()
            self._thread = PacketClient._NetworkStreamer(self, conn)
            self._thread.start()

        # PacketStream interface constraint: stall until dtype/sample_rate known
        while any([self.dtype is None, self.sample_rate is None]):
            time.sleep(5e-3)

    def stop(self):
        self._active.clear()  # gracefully terminate acquisition

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

    class _NetworkStreamer(threading.Thread):
        def __init__(self, client: "PacketClient", skt: socket.socket):
            super().__init__()
            self._client = client
            self._skt = skt

        def run(self):
            try:
                self._decode_header()
                self._stream_audio()
            except Exception:
                pass
            finally:
                self._cleanup()

        def _decode_header(self):
            data = b""
            N_left = lambda: 4 - len(data)
            while N_left() > 0:
                data += self._skt.recv(N_left())
            N_byte = int.from_bytes(data, byteorder="big", signed=False)

            data = b""
            N_left = lambda: N_byte - len(data)
            while N_left() > 0:
                data += self._skt.recv(N_left())
            metadata = json.loads(data.decode())

            # Set stream properties
            dtype = np.dtype(list(map(tuple, metadata["dtype_descr"])))
            self._client._dtype = dtype
            self._client._sample_rate = metadata["sample_rate"]

        def _stream_audio(self):
            skt_alive = True
            while self._client.active() and skt_alive:
                pkt = self._channel2pkt()
                if pkt is not None:
                    self._client._pkt_q.put(pkt.copy())
                else:
                    # connection closed -> stop acquisition
                    skt_alive = False

        def _channel2pkt(self) -> np.ndarray:
            # Obtain a full packet from the channel, or None if infeasible.
            data = b""
            N_left = lambda: self._client.dtype.itemsize - len(data)
            while True:
                d = self._skt.recv(N_left())
                data += d
                if N_left() == 0:
                    pkt = np.frombuffer(data, dtype=self._client.dtype)
                    return pkt
                elif len(d) == 0:
                    return None

        def _cleanup(self):
            self._skt.close()

            # reset state to that after PacketClient.__init__().
            self._client._active.clear()
            self._client._dtype = None
            self._client._sample_rate = None
            self._client._thread = None
