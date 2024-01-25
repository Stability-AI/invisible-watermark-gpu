import base64
import logging
import struct
import time
import uuid

import jax.numpy as jnp
import numpy as np

from .dwt import EmbedDwt
from .dwtDctSvd import EmbedDwtDctSvd
from .dwtDctSvdOptimized import EmbedDwtDctSvdOptimized
from .dwtSvd import EmbedDwtSvd
from .maxDct import EmbedMaxDct
from .maxDctOptimized import EmbedMaxDctOptimized
from .rivaGan import RivaWatermark

logger = logging.getLogger(__name__)


class WatermarkEncoder(object):
    def __init__(self, content=b""):
        seq = np.array([n for n in content], dtype=np.uint8)
        self._watermarks = list(np.unpackbits(seq))
        self._wmLen = len(self._watermarks)
        self._wmType = "bytes"

    @classmethod
    def loadModel(cls):
        RivaWatermark.loadModel()

    def warmup_gpu(self):
        """
        Call this function before doing multiple encodes!

        This function is to perform `cuLibraryLoadData` operations which
        can take a couple of seconds, but are a one-time cost to perform.

        Currently only supports ['dwtDct'] method of encoding.
        """
        start = time.time()
        EmbedMaxDct(self._watermarks, wmLen=self._wmLen)
        EmbedDwtDctSvdOptimized(self._watermarks, wmLen=self._wmLen)
        _, _, _ = jnp.linalg.svd(np.random.rand(512, 512))
        elapsed_ms = (time.time() - start) * 1000.0
        logger.info(f"GPU warmup completed in {elapsed_ms:.2f} ms")

    def set_by_ipv4(self, addr):
        bits = []
        ips = addr.split(".")
        for ip in ips:
            bits += list(np.unpackbits(np.array([ip % 255], dtype=np.uint8)))
        self._watermarks = bits
        self._wmLen = len(self._watermarks)
        self._wmType = "ipv4"
        assert self._wmLen == 32

    def set_by_uuid(self, uid):
        u = uuid.UUID(uid)
        self._wmType = "uuid"
        seq = np.array([n for n in u.bytes], dtype=np.uint8)
        self._watermarks = list(np.unpackbits(seq))
        self._wmLen = len(self._watermarks)

    def set_by_bytes(self, content):
        self._wmType = "bytes"
        seq = np.array([n for n in content], dtype=np.uint8)
        self._watermarks = list(np.unpackbits(seq))
        self._wmLen = len(self._watermarks)

    def set_by_b16(self, b16):
        content = base64.b16decode(b16)
        self.set_by_bytes(content)
        self._wmType = "b16"

    def set_by_bits(self, bits=[]):
        self._watermarks = [int(bit) % 2 for bit in bits]
        self._wmLen = len(self._watermarks)
        self._wmType = "bits"

    def set_watermark(self, wmType="bytes", content=""):
        if wmType == "ipv4":
            self.set_by_ipv4(content)
        elif wmType == "uuid":
            self.set_by_uuid(content)
        elif wmType == "bits":
            self.set_by_bits(content)
        elif wmType == "bytes":
            self.set_by_bytes(content)
        elif wmType == "b16":
            self.set_by_b16(content)
        else:
            raise NameError("%s is not supported" % wmType)

    def get_length(self):
        return self._wmLen

    def encode(self, cv2Image, method="dwtDct", **configs):
        (r, c, channels) = cv2Image.shape
        if r * c < 256 * 256:
            raise ValueError("image too small, should be larger than 256x256")

        if method == "dwtDct":
            embed = EmbedMaxDct(self._watermarks, wmLen=self._wmLen, **configs)
            return embed.encode(cv2Image)
        elif method == "dwtDctOptimized":
            embed = EmbedMaxDctOptimized(self._watermarks, wmLen=self._wmLen, **configs)
            return embed.encode(cv2Image)
        elif method == "dwt":
            embed = EmbedDwt(self._watermarks, wmLen=self._wmLen, **configs)
            return embed.encode(cv2Image)
        elif method == "dwtSvd":
            embed = EmbedDwtSvd(self._watermarks, wmLen=self._wmLen, **configs)
            return embed.encode(cv2Image)
        elif method == "dwtDctSvd":
            embed = EmbedDwtDctSvd(self._watermarks, wmLen=self._wmLen, **configs)
            return embed.encode(cv2Image)
        elif method == "dwtDctSvdOptimized":
            embed = EmbedDwtDctSvdOptimized(
                self._watermarks, wmLen=self._wmLen, **configs
            )
            return embed.encode(cv2Image)
        elif method == "rivaGan":
            embed = RivaWatermark(self._watermarks, self._wmLen)
            return embed.encode(cv2Image)
        else:
            raise NameError("%s is not supported" % method)


class WatermarkDecoder(object):
    def __init__(self, wm_type="bytes", length=0):
        self._wmType = wm_type
        if wm_type == "ipv4":
            self._wmLen = 32
        elif wm_type == "uuid":
            self._wmLen = 128
        elif wm_type == "bytes":
            self._wmLen = length
        elif wm_type == "bits":
            self._wmLen = length
        elif wm_type == "b16":
            self._wmLen = length
        else:
            raise NameError("%s is unsupported" % wm_type)

    def reconstruct_ipv4(self, bits):
        ips = [str(ip) for ip in list(np.packbits(bits))]
        return ".".join(ips)

    def reconstruct_uuid(self, bits):
        nums = np.packbits(bits)
        bstr = b""
        for i in range(16):
            bstr += struct.pack(">B", nums[i])

        return str(uuid.UUID(bytes=bstr))

    def reconstruct_bits(self, bits):
        # return ''.join([str(b) for b in bits])
        return bits

    def reconstruct_b16(self, bits):
        bstr = self.reconstruct_bytes(bits)
        return base64.b16encode(bstr)

    def reconstruct_bytes(self, bits):
        nums = np.packbits(bits)
        bstr = b""
        for i in range(self._wmLen // 8):
            bstr += struct.pack(">B", nums[i])
        return bstr

    def reconstruct(self, bits):
        if len(bits) != self._wmLen:
            raise ValueError("bits are not matched with watermark length")

        if self._wmType == "ipv4":
            return self.reconstruct_ipv4(bits)
        elif self._wmType == "uuid":
            return self.reconstruct_uuid(bits)
        elif self._wmType == "bits":
            return self.reconstruct_bits(bits)
        elif self._wmType == "b16":
            return self.reconstruct_b16(bits)
        else:
            return self.reconstruct_bytes(bits)

    def decode(self, cv2Image, method="dwtDct", **configs):
        (r, c, channels) = cv2Image.shape
        if r * c < 256 * 256:
            raise ValueError("image too small, should be larger than 256x256")

        bits = []
        if method == "dwtDct":
            embed = EmbedMaxDct(watermarks=[], wmLen=self._wmLen, **configs)
            bits = embed.decode(cv2Image)
        elif method == "dwtDctOptimized":
            embed = EmbedMaxDctOptimized(watermarks=[], wmLen=self._wmLen, **configs)
            return embed.decode(cv2Image)
        elif method == "dwt":
            embed = EmbedDwt(watermarks=[], wmLen=self._wmLen, **configs)
            return embed.decode(cv2Image)
        elif method == "dwtSvd":
            embed = EmbedDwtSvd(watermarks=[], wmLen=self._wmLen, **configs)
            bits = embed.decode(cv2Image)
        elif method == "dwtDctSvd":
            embed = EmbedDwtDctSvd(watermarks=[], wmLen=self._wmLen, **configs)
            bits = embed.decode(cv2Image)
        elif method == "dwtDctSvdOptimized":
            embed = EmbedDwtDctSvdOptimized(watermarks=[], wmLen=self._wmLen, **configs)
            bits = embed.decode(cv2Image)
        elif method == "rivaGan":
            embed = RivaWatermark(watermarks=[], wmLen=self._wmLen, **configs)
            bits = embed.decode(cv2Image)
        else:
            raise NameError("%s is not supported" % method)
        return self.reconstruct(bits)
