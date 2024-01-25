import time

import cv2
import numpy as np
from pycudwt import Wavelets


class EmbedDwt(object):
    def __init__(self, watermarks=[], wmLen=8, scales=[0, 36, 0], block=4):
        self._watermarks = watermarks
        self._wmLen = wmLen
        self._scales = scales
        self._block = block

        # Create a wavelets instance - this is just to "warmup" the GPU by loading cuda libraries.
        # Note: calling it in just this instance of EmbedDwtSvd will warmup for all future instances
        # of EmbedDwtSvd in same Python process!
        Wavelets(
            np.random.randint(low=0, high=255, size=(1024, 1024), dtype=np.uint8),
            "haar",
            1,
        )

    def encode(self, bgr):
        (row, col, channels) = bgr.shape
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        for channel in range(3):
            if self._scales[channel] <= 0:
                continue

            # send image to GPU
            wv = Wavelets(yuv[: row // 4 * 4, : col // 4 * 4, channel], "haar", 1)

            # perform the discrete wavelets transform
            wv.forward()  # wv.coeffs = [A, [H1, V1, D1]]

            # encode our coefficients with bit sequence
            encoded_approx_matrix = self.encode_frame(
                wv.coeffs[0], self._scales[channel]
            )

            # load the encoded coefficients back into the wavelets instance in GPU memory
            # and perform inverse discrete wavelets transform
            wv.set_coeff(encoded_approx_matrix, 0, 0)
            wv.inverse()

            # load the inverse wavelets transform back into the image
            yuv[: row // 4 * 4, : col // 4 * 4, channel] = np.array(wv.image).astype(
                np.uint8
            )

        bgr_encoded = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return bgr_encoded

    def decode(self, bgr):
        (row, col, _) = bgr.shape
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
        wv = Wavelets(yuv[: row // 4 * 4, : col // 4 * 4, 1], "haar", 1)
        wv.forward()
        scores = self.decode_frame(wv.coeffs[0], self._scales[1], [])
        return scores

    def decode_frame(self, frame, scale, scores):
        frame_long = frame.flatten()
        raw = (frame_long % scale) / scale

        num_tiles = int(
            np.floor(len(frame_long) / self._wmLen)
        )  # throwing away last partial iter, if any
        guessed_bits = np.zeros((self._wmLen,), dtype=np.float32)
        for window in range(num_tiles):
            window_raw = raw[window * self._wmLen : (window + 1) * self._wmLen]
            window_bits = window_raw > 0.5
            guessed_bits += window_bits

        guessed_bits /= num_tiles
        guessed_bits_binary = (guessed_bits > 0.5).astype(np.int8)
        return guessed_bits_binary

    def encode_frame(self, frame, scale):
        """
        frame is a matrix (M, N)

        we get K (watermark bits size) blocks (self._block x self._block)

        For i-th block, we encode watermark[i] bit into it
        """
        (row, col) = frame.shape
        frame_long = frame.flatten()

        num_tiles = int(np.ceil(len(frame_long) / self._wmLen))
        wmBits_tiled = np.tile(np.array(self._watermarks).astype(np.int32), num_tiles)[
            : len(frame_long)
        ]

        frame_long = (frame_long // scale + 0.25 + 0.5 * wmBits_tiled) * scale
        return frame_long.reshape((row, col))
