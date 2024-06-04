import numpy as np
import copy
import cv2
import pywt
import math
import pprint
import glob
import os
import time
import csv

pp = pprint.PrettyPrinter(indent=2)

class EmbedDwtDctSvdCorner(object):
    def __init__(self, watermarks=[], wmLen=8, scales=[0,36,0], block=4):
        self._watermarks = watermarks
        self._wmLen = wmLen
        self._scales = scales
        self._block = block

    def encode(self, bgr, algorithm='dwtDctSvd'):
        (row, col, channels) = bgr.shape

        # Calculate the region to watermark (1/16 x 1/16 of the image)
        region_row = row // 16
        region_col = col // 16

        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        for channel in range(2):
            if self._scales[channel] <= 0:
                continue

            ca1, (h1, v1, d1) = pywt.dwt2(yuv[:region_row, :region_col, channel], 'haar')
            self.encode_frame(ca1, self._scales[channel])

            yuv[:region_row, :region_col, channel] = pywt.idwt2((ca1, (h1, v1, d1)), 'haar')

        bgr_encoded = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return bgr_encoded

    def decode(self, bgr, algorithm='dwtDctSvd'):
        (row, col, channels) = bgr.shape

        # Calculate the region to decode (1/16 x 1/16 of the image)
        region_row = row // 16
        region_col = col // 16

        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        scores = [[] for i in range(self._wmLen)]
        for channel in range(2):
            if self._scales[channel] <= 0:
                continue

            ca1, (h1, v1, d1) = pywt.dwt2(yuv[:region_row, :region_col, channel], 'haar')

            scores = self.decode_frame(ca1, self._scales[channel], scores)

        avgScores = list(map(lambda l: np.array(l).mean(), scores))

        bits = (np.array(avgScores) * 255 > 127).astype(int)
        return bits

    def decode_frame(self, frame, scale, scores):
        (row, col) = frame.shape
        num = 0

        for i in range(row // self._block):
            for j in range(col // self._block):
                block = frame[i * self._block: i * self._block + self._block,
                              j * self._block: j * self._block + self._block]

                score = self.infer_dct_svd(block, scale)
                wmBit = num % self._wmLen
                scores[wmBit].append(score)
                num = num + 1

        return scores

    def diffuse_dct_svd(self, block, wmBit, scale):
        u, s, v = np.linalg.svd(cv2.dct(block))

        s[0] = (s[0] // scale + 0.25 + 0.5 * wmBit) * scale
        return cv2.idct(np.dot(u, np.dot(np.diag(s), v)))

    def infer_dct_svd(self, block, scale):
        u, s, v = np.linalg.svd(cv2.dct(block))

        score = int((s[0] % scale) > scale * 0.5)
        return score

    def encode_frame(self, frame, scale):
        (row, col) = frame.shape
        num = 0
        for i in range(row // self._block):
            for j in range(col // self._block):
                block = frame[i * self._block: i * self._block + self._block,
                              j * self._block: j * self._block + self._block]
                wmBit = self._watermarks[(num % self._wmLen)]

                diffusedBlock = self.diffuse_dct_svd(block, wmBit, scale)
                frame[i * self._block: i * self._block + self._block,
                      j * self._block: j * self._block + self._block] = diffusedBlock

                num = num + 1