import time

import cv2
import numpy as np
import jax.numpy as jnp
from pycudwt import Wavelets


class EmbedDwtSvd(object):
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

        # there appears to be a warmup cost for the first time 
        # we call jnp.linalg.svd as well
        _, _, _ = jnp.linalg.svd(np.random.rand(512, 512))

    def encode(self, bgr):
        (row, col, channels) = bgr.shape
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        for channel in range(3):
            if self._scales[channel] <= 0:
                continue

            print(f"channel: {channel}")

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

        # from scipy import linalg
        # starttime = time.time(); U, s, Vh = linalg.svd(np.random.rand(512, 512), check_finite=False); print(f"scipy SVD took: {(time.time() - starttime) * 1000:.2f} ms")
        # starttime = time.time(); jnp.linalg.svd(frame); print(f"jnp SVD took: {(time.time() - starttime) * 1000:.2f} ms")

        ### test reconstruction
        # U, S_jax, Vh = jnp.linalg.svd(frame, full_matrices=True)
        # S = np.array(S_jax)
        # smat = np.diag(S)
        # original_frame = np.dot(U, np.dot(smat, Vh))
        # print(np.allclose(frame, original_frame, atol=1e-02))

        starttime = time.time()
        
        svdtime = time.time()
        # U, S, Vh = np.linalg.svd(frame)  # 160ms
        U, S_jax, Vh = jnp.linalg.svd(frame)   # 25ms (!)
        S = np.array(S_jax)
        num_singular_values = len(S)
        print(f"jnp SVD took: {(time.time() - svdtime) * 1000:.2f} ms")

        # create a tiled version of the watermark bits that extends
        # the same length as our singular values
        # we will repeat the watermarking and then average at the end
        tilingtime = time.time()
        num_tiles = int(np.ceil(num_singular_values / self._wmLen))
        bits_length = int(num_singular_values)
        wmBits_tiled = np.tile(np.array(self._watermarks).astype(np.int32), num_tiles)[:bits_length]
        print(f"tiling took: {(time.time() - tilingtime) * 1000:.2f} ms")
        
        # encode bits, one per singular value
        enctime = time.time()
        S = (S // scale + 0.25 + 0.5 * wmBits_tiled) * scale
        print(f"encoding time took: {(time.time() - enctime) * 1000:.2f} ms")

        # compute full matrix again
        inversetime = time.time()
        smat = np.diag(S)
        encoded_frame = np.dot(U, np.dot(smat, Vh))
        print(f"inversetime took: {(time.time() - inversetime) * 1000:.2f} ms")

        print(f"encode_frame took: {(time.time() - starttime) * 1000:.2f} ms")
        # import ipdb; ipdb.set_trace()

        # now let's try to decode it
        # decodestarttime = time.time()

        # S_jax_d = jnp.linalg.svd(encoded_frame, compute_uv=False)   # 25ms (!)
        # Sd = np.array(S_jax_d)
        # num_singular_values = len(Sd)

        # bits = (Sd % scale) > (0.5 * scale)
        # num_tiles = int(np.ceil(num_singular_values / self._wmLen))
        # bit_array = np.zeros((num_tiles * self._wmLen), dtype=np.int32)
        # bit_array[:num_singular_values] = bits
        # bit_array = bit_array.reshape((num_tiles, self._wmLen))
        # mean_bits = bit_array.mean(axis=0)

        # rounded_bits = np.round(mean_bits)
        # correct = (rounded_bits == self._watermarks)

        # import ipdb; ipdb.set_trace()

        # correct = (rounded_bits == wmBits_tiled)
        # print(f"bit error rate %: {1. - correct.sum() / len(correct)}")

        # import ipdb; ipdb.set_trace()

        # correct = (rounded_bits == wmBits_tiled)
        # print(f"bit error rate %: {1. - correct.sum() / len(correct)}")

        # compute average over frames of length self._wmLen

        # print(f"decode_frame took: {(time.time() - decodestarttime) * 1000:.2f} ms")
        # import ipdb; ipdb.set_trace()

        # plt.clf()
        # plt.hist(frame_diffs, bins=50, range=(-5, 5))
        # plt.savefig("./histogram.png")
        
        return encoded_frame  #, np.array(S_jax)