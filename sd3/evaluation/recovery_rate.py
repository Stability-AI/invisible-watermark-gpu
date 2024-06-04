import detect
import os
from detect import GetWatermarkMatch
import numpy as np
import cv2
import glob
import pandas as pd


WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
# bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]



wm_path = '/weka2/Rahim/Watermarking/_samples/watermarked/invisible-watermark/fireworks/'
# wm_path = '/weka2/Rahim/Watermarking/_samples/watermarked/invisible-watermark/parti_1600'
# wm_path = '/weka2/Rahim/Watermarking/_samples/watermarked/invisible-watermark/fireworks_james/'

jpeg_quality = [100, 95, 90, 80, 50, 10]
df = pd.DataFrame(columns=jpeg_quality, index=['dwtDct', 'dwtDctSvd'])

for algorithm in ['dwtDctSvd']:
    for quality in jpeg_quality:
        recovery_rates = []
        for img_path in sorted(glob.glob(wm_path + '/' + algorithm + '/*' + '.png')):
            bgr = cv2.imread(img_path)
            if quality < 100:
                ### compress using gpeg with 0.95 quality
                new_img_path = img_path.replace('.png', f'_{quality}.jpeg')
                cv2.imwrite(new_img_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                bgr = cv2.imread(new_img_path)
            get_watermark_match = GetWatermarkMatch(WATERMARK_BITS, algorithm=algorithm)
            recovered_bits = get_watermark_match(bgr)
            print(recovered_bits)
            recovery_rates.append(recovered_bits)
        result = np.mean(recovery_rates)
        df.loc[algorithm, quality] = result
print(df)
# df.to_csv('/weka2/home-rahiment/watermark/invisible-watermark/invisible-watermark-gpu/sd3/evaluation/recovery_rates_f30.csv')
# df.to_csv('/weka2/home-rahiment/watermark/invisible-watermark/invisible-watermark-gpu/sd3/evaluation/recovery_rates_parti1600.csv')




