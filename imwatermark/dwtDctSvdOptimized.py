import pprint

import cv2
import numpy as np
import pywt
from pycudwt import Wavelets
from scipy.fftpack import dct, idct

pp = pprint.PrettyPrinter(indent=2)


class EmbedDwtDctSvdOptimized(object):
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

        for channel in range(2):
            if self._scales[channel] <= 0:
                continue

            # discrete wavelets decomposition
            wv = Wavelets(yuv[: row // 4 * 4, : col // 4 * 4, channel], "haar", 1)
            wv.forward()  # wv.coeffs = [A, [H1, V1, D1]]

            # encode the frame with out information
            encoded_approx_matrix = self.encode_frame(
                wv.coeffs[0].astype(np.float64), self._scales[channel]
            )

            # perform the inverse DWT to get our image back
            wv.set_coeff(encoded_approx_matrix, 0, 0)
            wv.inverse()
            yuv[: row // 4 * 4, : col // 4 * 4, channel] = np.array(wv.image)

        bgr_encoded = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return bgr_encoded

    def decode(self, bgr):
        (row, col, channels) = bgr.shape

        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        scores = [[] for i in range(self._wmLen)]
        for channel in range(2):
            if self._scales[channel] <= 0:
                continue

            ca1, (h1, v1, d1) = pywt.dwt2(
                yuv[: row // 4 * 4, : col // 4 * 4, channel], "haar"
            )

            scores = self.decode_frame(ca1, self._scales[channel], scores)

        avgScores = list(map(lambda l: np.array(l).mean(), scores))

        bits = np.array(avgScores) * 255 > 127
        return bits

    def decode_frame(self, frame, scale, scores):
        (row, col) = frame.shape
        num = 0
        for i in range(row // self._block):
            for j in range(col // self._block):
                block = frame[
                    i * self._block : i * self._block + self._block,
                    j * self._block : j * self._block + self._block,
                ]

                score = self.infer_dct_svd(block, scale)
                wmBit = num % self._wmLen
                scores[wmBit].append(score)
                num = num + 1

        return scores

    def infer_dct_svd(self, block, scale):
        u, s, v = np.linalg.svd(cv2.dct(block))
        score = int((s[0] % scale) > scale * 0.5)
        return score

    def encode_frame(self, frame, scale):
        """
        frame is a matrix (M, N)

        we get K (watermark bits size) blocks (self._block x self._block)

        For i-th block, we encode watermark[i] bit into it
        """
        (row, col) = frame.shape
        num_rows = row // self._block
        num_cols = col // self._block

        # generate our sequence of watermark bits to encode
        incrementing_num = np.arange(num_rows * num_cols, dtype=np.int16).reshape(
            (num_rows, num_cols)
        )
        wmBits = np.array(self._watermarks)[(incrementing_num % self._wmLen).flatten()]

        # reshape our frame into smaller blocks that we will operate on
        block_m = self._block  # rows in block
        block_n = self._block  # cols in block
        blocks = frame.reshape(num_rows, self._block, num_cols, self._block).swapaxes(
            1, 2
        )

        # decompose each block first by DCT, then into singluar values
        U, S, Vh = np.linalg.svd(dct(blocks, norm="ortho"))
        S[:, :, 0] = (
            S[:, :, 0] // scale + 0.25 + 0.5 * wmBits.reshape(num_rows, num_cols)
        ) * scale

        # put our singular values into matrices
        singular_values_as_matrices = np.zeros((num_rows, num_cols, block_m, block_n))
        diag_indices = np.arange(4)
        singular_values_as_matrices[..., diag_indices, diag_indices] = S

        # reconstruct our blocks from the singular values and IDCT
        reconstructed_blocks = U @ singular_values_as_matrices @ Vh
        diffused_blocks = idct(reconstructed_blocks, norm="ortho")

        # reshape our frame back to normal, and we're done!
        encoded_frame = diffused_blocks.swapaxes(1, 2).reshape(
            num_rows * block_m, num_cols * block_n
        )
        return encoded_frame
