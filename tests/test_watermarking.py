import time

from pycudwt import Wavelets
import numpy as np
from PIL import Image

from imwatermark import WatermarkEncoder, WatermarkDecoder

np.random.seed(42)

METHOD_DWT_DCT_SVD = 'dwtDctSvdOptimized'
WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]

# warmup GPU
Wavelets(np.random.randint(low=0, high=255, size=(1024, 1024), dtype=np.uint8), "haar", 1)
WatermarkEncoder().warmup_gpu()

def apply_watermark(image: Image.Image, method: str, bits: list) -> Image.Image:
    encoder = WatermarkEncoder()
    encoder.set_watermark("bits", bits)
    watermarked = encoder.encode(np.array(image)[:, :, ::-1], method)
    return Image.fromarray(watermarked[:, :, ::-1])

class TestEncodeDecode:
    def test_encode_decode(self):
        """
        Ensure we can encode and decode a watermark without any errors.
        """
        image = Image.open("test_vectors/original.jpg")
        image = image.crop((0, 0, 1024, 1024))

        start = time.time()
        watermarked = apply_watermark(image, METHOD_DWT_DCT_SVD, WATERMARK_BITS)
        watermarking_ms = (time.time() - start) * 1000.
        print(f"Watermarking took {watermarking_ms:.2f} ms")

        enc_length = 48
        decoder = WatermarkDecoder('bits', enc_length)

        start = time.time()
        watermarked_bgr = np.array(watermarked)[:, :, ::-1]
        recovered_watermark = [int(b) for b in decoder.decode(watermarked_bgr, 'dwt', scales=[0, 6, 0])]
        decoding_ms = (time.time() - start) * 1000.
        print(f"Decoding took {decoding_ms:.2f} ms")

        bit_error_rate = (recovered_watermark == np.array(WATERMARK_BITS)).mean()
        print(f"Bit error rate: {bit_error_rate}")
        assert bit_error_rate < 0.1, "Bit error rate was too high!"
