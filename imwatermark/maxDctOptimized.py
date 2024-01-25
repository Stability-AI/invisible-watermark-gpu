import cv2
import numpy as np
from pycudwt import Wavelets


class EmbedMaxDctOptimized(object):
    def __init__(self, watermarks=[], wmLen=8, scales=[0, 36, 36], block=4):
        self._watermarks = watermarks
        self._wmLen = wmLen
        self._scales = scales
        self._block = block

        # Create a wavelets instance - this is just to "warmup" the GPU by loading cuda libraries.
        # Note: calling it in just this instance of EmbedMaxDct will warmup for all future instances
        # of EmbedMaxDct in same Python process!
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
        (row, col, channels) = bgr.shape

        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        scores = [[] for i in range(self._wmLen)]
        for channel in range(2):
            if self._scales[channel] <= 0:
                continue

            wv = Wavelets(yuv[: row // 4 * 4, : col // 4 * 4, channel], "haar", 1)
            wv.forward()

            scores = self.decode_frame(wv.coeffs[0], self._scales[channel], scores)

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

                score = self.infer_dct_matrix(block, scale)
                wmBit = num % self._wmLen
                scores[wmBit].append(score)
                num = num + 1

        return scores

    def infer_dct_matrix(self, block, scale):
        pos = np.argmax(abs(block.flatten()[1:])) + 1
        i, j = pos // self._block, pos % self._block

        val = block[i][j]
        if val < 0:
            val = abs(val)

        if (val % scale) > 0.5 * scale:
            return 1
        else:
            return 0

    def encode_frame(self, frame, scale):
        """
        frame is a matrix (M, N)

        we get K (watermark bits size) blocks (self._block x self._block)

        For i-th block, we encode watermark[i] bit into it
        """
        (row, col) = frame.shape
        num_rows = row // self._block
        num_cols = col // self._block
        num_cells = num_rows * num_cols

        # generate our sequence of watermark bits to encode
        incrementing_num = np.arange(num_rows * num_cols, dtype=np.int16).reshape(
            (num_rows, num_cols)
        )
        wmBits = np.array(self._watermarks)[(incrementing_num % self._wmLen).flatten()]

        block_m = self._block  # rows in block
        block_n = self._block  # cols in block
        blocksize = block_m * block_n

        # find the non-upper left maxiumum coefficient (i, j) index per block
        blocks = frame.reshape(num_rows, self._block, num_cols, self._block).swapaxes(
            1, 2
        )
        max_block_val_idx = (
            np.argmax(np.abs(blocks.reshape(num_cells, blocksize)[:, 1:]), axis=1) + 1
        )
        max_block_i = max_block_val_idx // self._block
        max_block_j = max_block_val_idx % self._block

        # extract the maximum coefficient value per block
        block_idx = np.arange(num_cells)
        block_idx_to_block = blocks.reshape(num_cells, self._block, self._block)
        max_block_vals = block_idx_to_block[
            block_idx, max_block_i.flatten(), max_block_j.flatten()
        ]

        # perform the bit stuffing operation using the indices & values
        block_idx_to_block[block_idx, max_block_i.flatten(), max_block_j.flatten()] = (
            np.sign(max_block_vals)
            * (
                np.abs(max_block_vals) // scale
                + 0.25
                + 0.5 * wmBits[incrementing_num.flatten()].flatten()
            )
            * scale
        )

        # reshape the blocks back into the original frame shape
        encoded_frame = (
            block_idx_to_block.reshape(num_rows, num_cols, self._block, self._block)
            .transpose(0, 2, 1, 3)
            .reshape(row, col)
        )
        return encoded_frame
