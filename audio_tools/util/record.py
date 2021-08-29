import argparse
import pathlib as plib

import numpy as np
import scipy.io.wavfile as siw

import audio_tools.interface as ati
import audio_tools.network as atn


def record(stream: ati.PacketStream, duration: float) -> tuple[int, np.ndarray]:
    """
    Record a fixed-size audio stream from a Packet source.

    Parameters
    ----------
    stream: PacketStream
        Packet source.
    duration: float
        Acquisition time. [s]

        This parameter specifies a minimum acquisition time: at least `duration` seconds of audio
        will be recorded.

    Returns
    -------
    sample_rate: int
        Audio sample rate. [unit: sample/s]
    data: np.ndarray[uint/int/float]
        (N_sample, N_channel) audio stream.
    """
    if not stream.active():
        stream.start()

    # How many packets to capture
    N_sample = int(np.ceil(duration * stream.sample_rate))
    pkt_size = stream.dtype["data"].shape[0]
    N_pkt = int(np.ceil(N_sample / pkt_size))

    pkts = np.zeros((N_pkt,), dtype=stream.dtype)
    for i in range(N_pkt):
        pkts[i] = stream.get()

    data = np.concatenate(pkts["data"], axis=0)  # Extract audio samples only
    return stream.sample_rate, data


def _parse_args(args: list[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a fixed-size audio stream from a PacketServer."
    )
    parser.add_argument("ip", type=str, help="Source IP address")
    parser.add_argument("port", type=int, help="Source port")
    parser.add_argument("length", type=float, help="Acquisition time [unit: second]")
    parser.add_argument("file", type=str, help=".WAV file to store audio stream")

    args = parser.parse_args(args)
    args.file = plib.Path(args.file).expanduser().resolve()
    assert args.length > 0, "Non-negative length expected."
    return args


if __name__ == "__main__":
    args = _parse_args()

    cl = atn.PacketClient(host=args.host, port=args.port)
    fs, data = record(stream=cl, duration=args.length)
    cl.stop()
    cl.clear()

    args.file.parent.mkdir(parents=True, exist_ok=True)
    siw.write(filename=str(args.file), rate=fs, data=data)
