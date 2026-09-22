"""Decoder parallelism must not change training pixels or timestamps."""
import tempfile
from pathlib import Path
import unittest

import av
import numpy as np

from rldx.utils.video_utils import get_all_frames, get_frames_by_indices


class DecoderThreadTest(unittest.TestCase):
    def test_thread_count_preserves_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "clip.mkv")
            with av.open(path, "w") as container:
                stream = container.add_stream("ffv1", rate=10)
                stream.width, stream.height = 32, 32
                stream.pix_fmt = "bgr0"
                for i in range(12):
                    pixels = np.full((32, 32, 3), i * 17, dtype=np.uint8)
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            baseline, timestamps = get_all_frames(
                path, "torchcodec", {"num_ffmpeg_threads": 0}
            )
            actual, actual_timestamps = get_all_frames(path, "torchcodec")
            np.testing.assert_array_equal(actual, baseline)
            np.testing.assert_array_equal(actual_timestamps, timestamps)
            indices = [0, 5, 11]
            for kwargs in ({}, {"num_ffmpeg_threads": 2}):
                np.testing.assert_array_equal(
                    get_frames_by_indices(path, indices, "torchcodec", kwargs),
                    baseline[indices],
                )


if __name__ == "__main__":
    unittest.main()
