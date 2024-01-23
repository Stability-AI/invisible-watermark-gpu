import os
import time
import glob
from collections import defaultdict, Counter

from tqdm import tqdm

from PIL import Image
import numpy as np
from scipy.spatial import distance
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
from skimage import img_as_ubyte
import matplotlib.pyplot as plt

from imwatermark import WatermarkEncoder, WatermarkDecoder


METHODS_AVAILIABLE = ['dwtSvd', 'rivaGan', 'dwtDct']
WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]

# get image paths for LAION hr dataset
print("Finding images...")
FIRST_N = 10
img_paths = [path for path in glob.glob("../laion1024/*.jpg")]
img_paths = img_paths[:FIRST_N]

# warmup 
encoder = WatermarkEncoder()
encoder.set_watermark("bits", WATERMARK_BITS)
encoder.warmup_gpu()
encoder.loadModel()

# store results
encode_timings = defaultdict(list)
ssim_scores = defaultdict(list)
decode_timings = defaultdict(list)
all_zeros = Counter()
hamming_scores = defaultdict(list)
psnr_scores = defaultdict(list)

for method in METHODS_AVAILIABLE:
    # ensure folder exists
    folder = f"../laion1024_results/{method}/"
    if not os.path.exists(folder):
        os.makedirs(folder)

for img_path in tqdm(img_paths):
    for method in METHODS_AVAILIABLE:
        image_rgb = Image.open(img_path)
        slug = "_".join(img_path.split("/")[2:])

        starttime = time.time()
        if 'riva' in method.lower():
            watermark_bits = WATERMARK_BITS[:32]
        else:
            watermark_bits = WATERMARK_BITS

        start = time.time()
        encoder.set_watermark("bits", watermark_bits)
        print(f"encoder.set_watermark took: {(time.time() - start) * 1000:.2f} ms")

        start = time.time()
        image_bgr = np.array(image_rgb)[:, :, ::-1]
        print(f"np.array BGR took: {(time.time() - start) * 1000:.2f} ms")

        start = time.time()
        watermarked_bgr = encoder.encode(image_bgr, method)
        print(f"encoder.encode took: {(time.time() - start) * 1000:.2f} ms")

        start = time.time()
        watermarked_rgb = Image.fromarray(watermarked_bgr[:, :, ::-1])
        print(f"Image.fromarray took: {(time.time() - start) * 1000:.2f} ms")

        encode_timings[method].append((time.time() - starttime) * 1000)
        ### END ENCODE ###

        # convert the images to grayscale
        # compute the Structural Similarity Index (SSIM) between the two
        grayA = np.array(image_rgb.convert('L'))
        grayB = np.array(watermarked_rgb.convert('L'))
        (score, diff) = ssim(grayA, grayB, full=True, data_range=256)
        diff = (diff * 255).astype("uint8")
        ssim_scores[method].append(score)

        psnr = peak_signal_noise_ratio(np.array(image_rgb), np.array(watermarked_rgb), data_range=256)
        psnr_scores[method].append(psnr)

        # Plotting
        plt.clf()

        # Assuming the images are square and have the same dimensions
        img_height, img_width = grayA.shape
        dpi = 100  # Set the dots per inch
        fig_width, fig_height = 4 * img_width / dpi, img_height / dpi  # Width for 4 images, height for 1

        # Create a figure with the appropriate size
        fig, axes = plt.subplots(1, 4, figsize=(fig_width, fig_height), dpi=dpi)

        # Display original image
        # axes[0].imshow(grayA, cmap='gray')
        axes[0].imshow(np.array(image_rgb), cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Display altered image
        # axes[1].imshow(grayB, cmap='gray')
        axes[1].imshow(np.array(watermarked_rgb), cmap='gray')
        axes[1].set_title('Watermarked Image')
        axes[1].axis('off')

        # Display SSIM difference image
        axes[2].imshow(img_as_ubyte(diff), cmap='gray')
        axes[2].set_title(f'SSIM Diff (ssim={score:.4f})')
        axes[2].axis('off')

        axes[3].imshow(np.array(grayA) - np.array(grayB), cmap='gray')
        axes[3].set_title(f'Difference Image (psnr={psnr:.4f})')
        axes[3].axis('off')

        plt.title(f"Method: {method}, Image: {slug}, SSIM: {score:.3f}, PSNR: {psnr:.3f}")
        plt.savefig(f"../laion1024_results/{method}/compare_{slug}", dpi=dpi)

        ### DECODE ###
        enc_length = encoder.get_length()
        starttime = time.time()
        decoder = WatermarkDecoder('bits', enc_length)
        # import ipdb; ipdb.set_trace()
        recovered_watermark = [int(bit) for bit in decoder.decode(watermarked_bgr, method)]
        decode_timings[method].append((time.time() - starttime) * 1000)
        ### END DECODE ###

        if sum(recovered_watermark) == 0:
            print("All zeros!")
            all_zeros[method] += 1
        else:
            hamming_dist = distance.hamming(recovered_watermark, watermark_bits)
            print(f"Bit error rate for {method} on '{img_path}': {hamming_dist}")
            hamming_scores[method].append(hamming_dist)

for method in METHODS_AVAILIABLE:
    avg_hamming_score_among_non_all_zeros = np.sum(hamming_scores[method]) / (len(img_paths) - all_zeros[method])
    print(f"Average bit error rate for {method}: {avg_hamming_score_among_non_all_zeros:.4f}")
    print(f"Average encode ms: {np.mean(encode_timings[method]):.4f} +/- {np.std(encode_timings[method]):.4f}")
    print(f"Average decode ms: {np.mean(decode_timings[method]):.4f}")

    pct_all_zeros = all_zeros[method] / len(img_paths) * 100.
    print(f"Percent decodes with all zeros: {pct_all_zeros:.2f}%")

    ms_per_pixel = np.mean(encode_timings[method]) / (1024**2)
    print(f"Encode ms per pixel: {ms_per_pixel:.6f}")
    
    print("\n")
