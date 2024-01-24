import os
import time
import glob
import random
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


WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
BIT_ERROR_RATE_THRESH = 0.1

# get image paths for LAION hr dataset
print("Finding images...")
FIRST_N = 5
img_paths = [path for path in glob.glob("../laion1024/*.jpg")]
img_paths = img_paths[:FIRST_N]
random.shuffle(img_paths)

# warmup 
encoder = WatermarkEncoder()
encoder.set_watermark("bits", WATERMARK_BITS)
encoder.warmup_gpu()
encoder.loadModel()

METHODS_AVAILIABLE = [
    # method, scale
    ('dwt', 6),
    ('dwt', 8),
    # ('dwt', 12),
    # ('dwt', 16),

    # ('dwtDct', 36),

    # ('dwtDctSvd', 36),

    # ('rivaGan', -1),
]


with open("results.txt", "w") as f:
    # store results
    encode_timings = defaultdict(list)
    ssim_scores = defaultdict(list)
    decode_timings = defaultdict(list)
    all_zeros = Counter()
    hamming_scores = defaultdict(list)
    psnr_scores = defaultdict(list)

    for method_string, SCALE in METHODS_AVAILIABLE:
        method = f"{method_string}_scale={SCALE}"

        # ensure folder exists
        folder = f"../laion1024_results/{method}/"
        if not os.path.exists(folder):
            os.makedirs(folder)

    for img_path in tqdm(img_paths):
        for method_string, SCALE in METHODS_AVAILIABLE:
            method = f"{method_string}_scale={SCALE}"

            image_rgb = Image.open(img_path)
            slug = "_".join(img_path.split("/")[2:])
            print(f"method: {method}, slug: {slug}")

            starttime = time.time()
            if 'riva' in method.lower():
                watermark_bits = WATERMARK_BITS[:32]
                scales = None
            else:
                watermark_bits = WATERMARK_BITS
                scales = [0, SCALE , 0]

            encoder.set_watermark("bits", watermark_bits)

            start = time.time()
            image_bgr = np.array(image_rgb)[:, :, ::-1]
            print(f"np.array BGR took: {(time.time() - start) * 1000:.2f} ms")

            start = time.time()
            watermarked_bgr = encoder.encode(image_bgr, method_string, scales=scales)
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
            dpi = 256  # Set the dots per inch
            fig_width, fig_height = 4 * img_width / dpi, img_height / dpi  # Width for 4 images, height for 1

            # Create a figure with the appropriate size
            fig, axes = plt.subplots(1, 4, figsize=(fig_width, fig_height), dpi=dpi)
            fig.suptitle(f"Method: {method}, Image: {slug}, SSIM: {score:.3f}, PSNR: {psnr:.3f}")

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

            plt.savefig(f"../laion1024_results/{method}/compare_{slug}", dpi=dpi)

            ### DECODE ###
            enc_length = encoder.get_length()
            starttime = time.time()
            decoder = WatermarkDecoder('bits', enc_length)
            # import ipdb; ipdb.set_trace()
            recovered_watermark = [int(bit) for bit in decoder.decode(watermarked_bgr, method_string, scales=scales)]
            decode_timings[method].append((time.time() - starttime) * 1000)
            ### END DECODE ###
            
            hamming_dist = 1.0
            if sum(recovered_watermark) == 0:
                print("All zeros!")
                all_zeros[method] += 1
            
            hamming_dist = distance.hamming(recovered_watermark, watermark_bits)
            print(f"Bit error rate for {method} on '{img_path}': {hamming_dist}")
            hamming_scores[method].append(hamming_dist)
    
    # write and compute some statistics
    perfect_decodes = {}
    avg_hamming_dist = {}
    percent_passing_below_error_threshold = {}

    for method_string, SCALE in METHODS_AVAILIABLE:
        method = f"{method_string}_scale={SCALE}"
        f.write(f"\n==== RESULTS, method: {method} ====\n")

        avg_hamming_dist = np.sum(hamming_scores[method]) / (len(img_paths))
        f.write(f"Average bit error rate for {method}: {avg_hamming_dist:.2f}\n")
        f.write(f"Average encode ms: {np.mean(encode_timings[method]):.2f} +/- {np.std(encode_timings[method]):.2f}\n")
        f.write(f"Average decode ms: {np.mean(decode_timings[method]):.2f}\n")

        pct_all_zeros = all_zeros[method] / len(img_paths) * 100.
        f.write(f"Percent decodes with all zeros: {pct_all_zeros:.2f}%\n")

        ms_per_pixel = np.mean(encode_timings[method]) / (1024**2)
        f.write(f"Encode ms per pixel: {ms_per_pixel:.6f}\n")

        perc_perfect_decodes = 0 
        if hamming_scores[method]:
            perc_perfect_decodes = hamming_scores[method].count(0) / len(decode_timings[method]) * 100.
            perfect_decodes[method] = perc_perfect_decodes
        
        f.write(f"Percent perfect decodes: {perc_perfect_decodes:.2f} %\n")

        perc_below_thresh = sum([1 for h in hamming_scores[method] if h < BIT_ERROR_RATE_THRESH]) / len(hamming_scores[method]) * 100.
        percent_passing_below_error_threshold[method] = perc_below_thresh
        f.write(f"Percent decodes below thresh ({BIT_ERROR_RATE_THRESH}): {perc_below_thresh:.2f} %\n")

        f.write(f"Average SSIM: {np.mean(ssim_scores[method]):.2f}\n")
        f.write(f"Average PSNR: {np.mean(psnr_scores[method]):.2f}\n")
        
        f.write("\n")

    ###############################
    ####### ENCODE MS vs PSNR
    plt.clf()
    plt.figure(figsize=(8, 6))

    methods = [f"{method_string}_scale={SCALE}" for method_string, SCALE in METHODS_AVAILIABLE]
    encode_ms = [np.mean(encode_timings[m]) for m in methods]
    psnrs = [np.mean(psnr_scores[m]) for m in methods]
    ssims = [np.mean(ssim_scores[m]) for m in methods]

    offset = -0.3

    # method_labels = [f"{method_string}_scale={SCALE}" for method_string, SCALE in METHODS_AVAILIABLE]
    x = encode_ms
    y = psnrs

    # Create the plot
    plt.scatter(x, y, color='blue')  # Plotting the points

    # Adding labels to each point
    for i in range(len(x)):
        plt.text(x[i] + offset, y[i] + offset, methods[i], fontsize=9, ha='left', va='top')

    # Optional: Setting the limit for better visualization
    # plt.xlim(0, max(x) + 1)
    # plt.ylim(0, max(y) + 1)

    plt.xlabel('Encode (ms)')
    plt.ylabel('Peak Signal to Noise Ratio (PSNR)')
    plt.title('Watermarking Tradeoffs')
    plt.grid(True)
    plt.savefig("plot_encode_vs_psnr.png")

    ###############################
    ####### ENCODE MS vs PERFECT DECODES
    plt.clf()
    plt.figure(figsize=(8, 6))

    methods = [f"{method_string}_scale={SCALE}" for method_string, SCALE in METHODS_AVAILIABLE]
    encode_ms = [np.mean(encode_timings[m]) for m in methods]
    perfect_decodes = [perfect_decodes[m] for m in methods]
    x = encode_ms
    y = perfect_decodes

    # Create the plot
    plt.scatter(x, y, color='blue')  # Plotting the points

    # Adding labels to each point
    for i in range(len(x)):
        plt.text(x[i] + offset, y[i] + offset, methods[i], fontsize=9, ha='left', va='top')

    plt.xlabel('Encode (ms)')
    plt.ylabel('Percent of time watermark is recovered perfectly')
    plt.title('Encode time (ms) vs Perfect recovery rate')
    plt.grid(True)
    plt.savefig("plot_encode_vs_encoding_perfect.png")

    ###############################
    ####### ENCODE MS vs BELOW THRESH
    plt.clf()
    plt.figure(figsize=(8, 6))

    methods = [f"{method_string}_scale={SCALE}" for method_string, SCALE in METHODS_AVAILIABLE]
    encode_ms = [np.mean(encode_timings[m]) for m in methods]
    passing = [percent_passing_below_error_threshold[m] for m in methods]
    x = encode_ms
    y = passing

    # Create the plot
    plt.scatter(x, y, color='blue')  # Plotting the points

    # Adding labels to each point
    for i in range(len(x)):
        plt.text(x[i] + offset, y[i] + offset, methods[i], fontsize=9, ha='left', va='top')

    plt.xlabel('Encode (ms)')
    plt.ylabel('Percent of time watermark is recovered')
    plt.title(f'Encode time vs Recovery rate at threshold BER={BIT_ERROR_RATE_THRESH}')
    plt.grid(True)
    plt.savefig("plot_encode_vs_encoding_threshold.png")
