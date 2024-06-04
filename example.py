import cv2
import numpy as np
from imwatermark import WatermarkDecoder
from imwatermark import WatermarkEncoder
from PIL import Image
import time

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDctSvd')
        img = Image.fromarray(img[:, :, ::-1])
    return img

wm = "StableDiffusionV3"
wm_encoder = WatermarkEncoder()
wm_encoder.set_watermark('bytes', wm.encode('utf-8'))
img = cv2.imread('./test_vectors/original.jpg')
### measure time

start = time.time()
img = put_watermark(img, wm_encoder)
end = time.time()
print("Encoding Time: ", end - start)
img.save('example_test_wm.png')


def testit(img_path):
    bgr = cv2.imread(img_path)
    decoder = WatermarkDecoder('bytes', 136)
    # watermark = decoder.decode(bgr, 'dwtDctSvd')
    # watermark = decoder.decode(bgr, 'dwtDct')
    start = time.time()
    watermark = decoder.decode(bgr, 'dwtDctSvd')
    end = time.time()
    print("Decoding Time: ", end - start)
    try:
        dec = watermark.decode('utf-8')
    except:
        dec = "hello"
    print(dec)

testit('example_test_wm.png')


