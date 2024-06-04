algorithm = "dwtDctSvddwtDctSvd"


import numpy as np
import cv2
from imwatermark import WatermarkEncoder

# bgr = cv2.imread('/weka2/home-rahiment/watermark/invisible-watermark/invisible-watermark-gpu/test_vectors/original.jpg')
# WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
# WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
# encoder = WatermarkEncoder()
# encoder.set_watermark("bits", WATERMARK_BITS)
# bgr_encoded = encoder.encode(bgr, 'dwtDctdwtDct')

# cv2.imwrite('test_wm.png', bgr_encoded)


### for all images inside a pth run the above
import glob
import os
WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
# generated_imgs = '/weka2/Rahim/Watermarking/_samples/sdv3_api_8b_beta/fireworks/07_fireworks_wm/v0/clip_scores/ema_False/default/000000000'
# wm_path = '/weka2/Rahim/Watermarking/_samples/watermarked/invisible-watermark/fireworks/'

generated_imgs = '/weka2/Rahim/Watermarking/_samples/sdv3_api_8b_beta/parti_1600/07_fireworks_wm/v0/clip_scores/ema_False/default/000000000'
wm_path = '/weka2/Rahim/Watermarking/_samples/watermarked/invisible-watermark/parti_1600'


encoder = WatermarkEncoder()
encoder.set_watermark("bits", WATERMARK_BITS)

### for images in the glob but sorted
for img_path in sorted(glob.glob(generated_imgs + '/*.png')):
    for algorithm in ['dwtDct']:
    # for algorithm in ['dwtDct', 'dwtDctSvd']:
        bgr = cv2.imread(img_path)
        bgr_encoded = encoder.encode(bgr, algorithm)
        filename_save = os.path.join(wm_path, algorithm)
        if not os.path.exists(filename_save):
            os.makedirs(filename_save)
        print(filename_save + '/' + os.path.basename(img_path))
        cv2.imwrite(filename_save + '/' + os.path.basename(img_path), bgr_encoded)
