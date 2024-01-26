import glob
import os
import random
import time
from collections import Counter, defaultdict

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import distance
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from imwatermark import WatermarkDecoder, WatermarkEncoder

WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]  # [:32]
BIT_ERROR_RATE_THRESH = 0.2
TEST_ATTACKS = True


def compute_and_add_ber(
    pil_rgb_image: Image,
    method_string: str,
    method_dict_to_save_to: defaultdict,
    scales: list,
    enc_length: int,
    watermark_actual_bits,
):
    bgr = np.array(pil_rgb_image)[:, :, ::-1]
    decoder = WatermarkDecoder("bits", enc_length)
    wm = [int(bit) for bit in decoder.decode(bgr, method_string, scales=scales)]
    ber = distance.hamming(wm, watermark_actual_bits)
    method_dict_to_save_to[method].append(ber)


# get image paths, these are simply 1024x1024 color
# images from the LAION high resolution dataset
print("Finding images...")
FIRST_N = 500
img_paths = [path for path in glob.glob("../laion1024/*.jpg")]
random.shuffle(img_paths)
img_paths = img_paths[:FIRST_N]

# warmup
encoder = WatermarkEncoder()
encoder.set_watermark("bits", WATERMARK_BITS)
encoder.warmup_gpu()
encoder.loadModel()

# FOR TESTING
METHODS_AVAILIABLE = [
    # starting with: fastest and cheapest
    ("dwtDctOptimized", 16),
    ("dwt", 24),
    ("dwtDctSvdOptimized", 24),
    ("rivaGan", -1),
]

# (method, scale)
# METHODS_AVAILIABLE = [
#     # starting with: fastest and cheapest
#     ('dwtDctOptimized', 16),
#     ('dwt', 24),
#     # tier 2, more complex, more robust
#     ('dwtDctSvdOptimized', 24),
#     # slowest, most powerful
#     ('rivaGan', -1),
# ]

with open("results.txt", "w") as f:
    # store results
    encode_timings = defaultdict(list)
    ssim_scores = defaultdict(list)
    decode_timings = defaultdict(list)
    all_zeros = Counter()
    hamming_scores = defaultdict(list)
    psnr_scores = defaultdict(list)

    # attacks
    attack_jpg_ber = defaultdict(list)
    attack_horizonal_flip_ber = defaultdict(list)
    attack_vertical_flip_ber = defaultdict(list)
    attack_random_crop_ber = defaultdict(list)
    attack_random_noise_ber = defaultdict(list)
    attack_rotate_ber = defaultdict(list)

    for method_string, scale in METHODS_AVAILIABLE:
        method = f"{method_string}_scale={scale}"

        # ensure folder exists
        folder = f"../laion1024_results/{method}/"
        if not os.path.exists(folder):
            os.makedirs(folder)

    for img_path in tqdm(img_paths):
        for method_string, scale in METHODS_AVAILIABLE:
            method = f"{method_string}_scale={scale}"

            # load the image from disk
            if not os.path.exists(img_path):
                continue

            image_rgb = Image.open(img_path)
            slug = "_".join(img_path.split("/")[2:])
            print(f"method: {method}, slug: {slug}")

            # if any special setup needs to be done for this method, do it here
            starttime = time.time()
            watermark_bits = WATERMARK_BITS
            scales = [0, scale, 0]

            if "riva" in method.lower():
                watermark_bits = WATERMARK_BITS[:32]
                scales = None

            encoder.set_watermark("bits", watermark_bits)
            enc_length = encoder.get_length()

            # make sure our image is actually in color
            start = time.time()
            try:
                image_bgr = np.array(image_rgb)[:, :, ::-1]
                print(f"np.array BGR took: {(time.time() - start) * 1000:.2f} ms")
            except IndexError:
                # greyscale image, we don't want it here
                os.remove(img_path)
                continue

            # encode the image
            start = time.time()
            watermarked_bgr = encoder.encode(image_bgr, method_string, scales=scales)
            print(f"encoder.encode took: {(time.time() - start) * 1000:.2f} ms")

            # convert back to PIL image
            start = time.time()
            watermarked_rgb = Image.fromarray(watermarked_bgr[:, :, ::-1])
            print(f"Image.fromarray took: {(time.time() - start) * 1000:.2f} ms")

            encode_timings[method].append((time.time() - starttime) * 1000)
            ### END ENCODE ###

            # convert the images to grayscale
            # compute the Structural Similarity Index (SSIM) between the two as well as PSNR
            grayA = np.array(image_rgb.convert("L"))
            grayB = np.array(watermarked_rgb.convert("L"))
            (score, diff) = ssim(grayA, grayB, full=True, data_range=256)
            diff = (diff * 255).astype("uint8")
            ssim_scores[method].append(score)
            psnr = peak_signal_noise_ratio(
                np.array(image_rgb), np.array(watermarked_rgb), data_range=256
            )
            psnr_scores[method].append(psnr)

            # Plotting differnces visually
            plt.clf()
            img_height, img_width = grayA.shape
            dpi = 256  # Set the dots per inch
            fig_width, fig_height = (
                4 * img_width / dpi,
                img_height / dpi,
            )  # Width for 4 images, height for 1
            fig, axes = plt.subplots(1, 4, figsize=(fig_width, fig_height), dpi=dpi)
            fig.suptitle(
                f"Method: {method}, Image: {slug}, SSIM: {score:.3f}, PSNR: {psnr:.3f}"
            )

            # Display original image
            # axes[0].imshow(grayA, cmap='gray')
            axes[0].imshow(np.array(image_rgb), cmap="gray")
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            # Display altered image
            # axes[1].imshow(grayB, cmap='gray')
            axes[1].imshow(np.array(watermarked_rgb), cmap="gray")
            axes[1].set_title("Watermarked Image")
            axes[1].axis("off")

            # Display SSIM difference image
            axes[2].imshow(img_as_ubyte(diff), cmap="gray")
            axes[2].set_title(f"SSIM Diff (ssim={score:.4f})")
            axes[2].axis("off")

            axes[3].imshow(np.array(grayA) - np.array(grayB), cmap="gray")
            axes[3].set_title(f"Difference Image (psnr={psnr:.4f})")
            axes[3].axis("off")

            plt.savefig(f"../laion1024_results/{method}/compare_{slug}", dpi=dpi)

            # save watermarked image to disk and load it back. this seems
            # to make the problem harder, but it's representative of what
            # will actually happen when users download the images!
            tmp_savepath = f"../laion1024_results/{method}/watermarked_{slug}".replace(
                ".jpg", ".png"
            )
            watermarked_rgb.save(tmp_savepath)
            to_decode_bgr = np.array(Image.open(tmp_savepath))[:, :, ::-1]

            # now decode the image and check if we got it right
            starttime = time.time()
            decoder = WatermarkDecoder("bits", enc_length)
            recovered_watermark = [
                int(bit)
                for bit in decoder.decode(to_decode_bgr, method_string, scales=scales)
            ]
            decode_timings[method].append((time.time() - starttime) * 1000)

            # compute bit error rate
            hamming_dist = 1.0
            if sum(recovered_watermark) == 0:
                print("All zeros!")
                all_zeros[method] += 1

            hamming_dist = distance.hamming(recovered_watermark, watermark_bits)
            print(f"Bit error rate for {method} on '{img_path}': {hamming_dist}")
            hamming_scores[method].append(hamming_dist)

            # now save to JPG on disk & try to decode
            if TEST_ATTACKS:
                watermarked_rgb_copy = watermarked_rgb.copy()

                # jpg attack
                tmp_savepath_jpg = (
                    f"../laion1024_results/{method}/watermarked_{slug}.jpg"
                )
                watermarked_rgb_copy.save(tmp_savepath_jpg)
                jpg_img_rgb = Image.open(tmp_savepath)
                compute_and_add_ber(
                    jpg_img_rgb,
                    method_string,
                    attack_jpg_ber,
                    scales,
                    enc_length,
                    watermark_bits,
                )
                os.remove(tmp_savepath_jpg)

                # horizontal flip attack
                horizontal_flip = watermarked_rgb.copy().transpose(
                    Image.FLIP_LEFT_RIGHT
                )
                compute_and_add_ber(
                    horizontal_flip,
                    method_string,
                    attack_horizonal_flip_ber,
                    scales,
                    enc_length,
                    watermark_bits,
                )

                # vertical flip attack
                vertical_flip = watermarked_rgb.copy().transpose(Image.FLIP_TOP_BOTTOM)
                compute_and_add_ber(
                    vertical_flip,
                    method_string,
                    attack_vertical_flip_ber,
                    scales,
                    enc_length,
                    watermark_bits,
                )

                # crop attack
                crop_size = (768, 768)
                original_width, original_height = watermarked_rgb.size
                if crop_size[0] > original_width or crop_size[1] > original_height:
                    raise ValueError(
                        "Crop size should be smaller than the original image size"
                    )
                max_x = original_width - crop_size[0]
                max_y = original_height - crop_size[1]
                start_x = random.randint(0, max_x)
                start_y = random.randint(0, max_y)
                cropped_image = watermarked_rgb.copy().crop(
                    (start_x, start_y, start_x + crop_size[0], start_y + crop_size[1])
                )
                compute_and_add_ber(
                    cropped_image,
                    method_string,
                    attack_random_crop_ber,
                    scales,
                    enc_length,
                    watermark_bits,
                )

                # noise attack
                image_array = np.array(watermarked_rgb.copy())
                noise_scale = 16
                noise = np.random.randint(
                    -noise_scale, noise_scale, image_array.shape, dtype="int32"
                )
                noisy_image_array = np.clip(image_array + noise, 0, 255)
                noisy_image = Image.fromarray(noisy_image_array.astype("uint8"))
                compute_and_add_ber(
                    noisy_image,
                    method_string,
                    attack_random_noise_ber,
                    scales,
                    enc_length,
                    watermark_bits,
                )

                # rotate attack
                rotated_image = watermarked_rgb.copy().rotate(-90)
                compute_and_add_ber(
                    rotated_image,
                    method_string,
                    attack_rotate_ber,
                    scales,
                    enc_length,
                    watermark_bits,
                )

    # write and compute some statistics
    perfect_decodes = {}
    avg_hamming_dist = {}
    percent_passing_below_error_threshold = {}

    for method_string, scale in METHODS_AVAILIABLE:
        method = f"{method_string}_scale={scale}"
        f.write(f"\n==== RESULTS, method: {method} ====\n")

        # latency
        f.write(
            f"Average encode ms: {np.mean(encode_timings[method]):.2f} ± {np.std(encode_timings[method]):.2f}\n"
        )
        f.write(f"Average decode ms: {np.mean(decode_timings[method]):.2f}\n")
        ms_per_pixel = np.mean(encode_timings[method]) / (1024**2)
        f.write(f"Encode ms per pixel: {ms_per_pixel:.6f}\n")

        # bit error rate
        avg_hamming_dist = np.sum(hamming_scores[method]) / (len(img_paths))
        f.write(f"Average bit error rate for {method}: {avg_hamming_dist:.2f}\n")
        pct_all_zeros = all_zeros[method] / len(img_paths) * 100.0
        f.write(f"Percent decodes with all zeros: {pct_all_zeros:.2f}%\n")

        perc_perfect_decodes = 0
        if hamming_scores[method]:
            perc_perfect_decodes = (
                hamming_scores[method].count(0) / len(decode_timings[method]) * 100.0
            )
            perfect_decodes[method] = perc_perfect_decodes

        f.write(f"Percent perfect decodes: {perc_perfect_decodes:.2f} %\n")

        perc_below_thresh = (
            sum([1 for h in hamming_scores[method] if h < BIT_ERROR_RATE_THRESH])
            / len(hamming_scores[method])
            * 100.0
        )
        percent_passing_below_error_threshold[method] = perc_below_thresh
        f.write(
            f"Percent decodes below thresh ({BIT_ERROR_RATE_THRESH}): {perc_below_thresh:.2f} %\n"
        )

        # quality metrics
        f.write(f"Average SSIM: {np.mean(ssim_scores[method]):.2f}\n")
        f.write(f"Average PSNR: {np.mean(psnr_scores[method]):.2f}\n")

        # attack metrics
        f.write(f"Mean BER for attacks on {method}\n")
        f.write(f"\tJPG convert: {np.mean(attack_jpg_ber[method]):.2f}\n")
        f.write(
            f"\tFlip (horizontal): {np.mean(attack_horizonal_flip_ber[method]):.2f}\n"
        )
        f.write(f"\tFlip (vertical): {np.mean(attack_vertical_flip_ber[method]):.2f}\n")
        f.write(f"\tRandom crop: {np.mean(attack_random_crop_ber[method]):.2f}\n")
        f.write(f"\tRandom noise: {np.mean(attack_random_noise_ber[method]):.2f}\n")
        f.write(f"\t90° Rotate: {np.mean(attack_rotate_ber[method]):.2f}\n")

        f.write("\n")

    # now make some plots to summarize these metrics
    offset = -0.3
    methods = [
        f"{method_string}_scale={SCALE}" for method_string, SCALE in METHODS_AVAILIABLE
    ]
    encode_ms = [np.mean(encode_timings[m]) for m in methods]
    psnrs = [np.mean(psnr_scores[m]) for m in methods]
    perfect_decodes = [perfect_decodes[m] for m in methods]
    passing = [percent_passing_below_error_threshold[m] for m in methods]
    mean_ber = [np.mean(hamming_scores[m]) for m in methods]

    ###############################
    ####### ENCODE MS vs PSNR
    plt.clf()
    plt.figure(figsize=(8, 6))
    plt.scatter(encode_ms, psnrs, color="blue")  # Plotting the points

    # Adding labels to each point
    for i in range(len(encode_ms)):
        plt.text(
            encode_ms[i] + offset * 0.4,
            psnrs[i] + offset * 0.4,
            methods[i],
            fontsize=9,
            ha="left",
            va="top",
        )

    plt.xlabel("Encode (ms)")
    plt.ylabel("Peak Signal to Noise Ratio (PSNR)")
    plt.title("Encoding time (ms) vs PSNR")
    plt.grid(True)
    plt.savefig("plot_encode_vs_psnr.png")

    ###############################
    ####### ENCODE MS vs PERFECT DECODES
    plt.clf()
    plt.figure(figsize=(8, 6))
    plt.scatter(encode_ms, perfect_decodes, color="blue")  # Plotting the points

    # Adding labels to each point
    for i in range(len(encode_ms)):
        plt.text(
            encode_ms[i] + offset,
            perfect_decodes[i] + offset,
            methods[i],
            fontsize=9,
            ha="left",
            va="top",
        )

    plt.xlabel("Encode (ms)")
    plt.ylabel("Percent of time watermark is recovered perfectly")
    plt.title("Encode time (ms) vs Perfect recovery rate")
    plt.grid(True)
    plt.savefig("plot_encode_vs_encoding_perfect.png")

    ###############################
    ####### ENCODE MS vs BELOW THRESH
    plt.clf()
    plt.figure(figsize=(8, 6))
    plt.scatter(encode_ms, passing, color="blue")  # Plotting the points

    # Adding labels to each point
    for i in range(len(encode_ms)):
        plt.text(
            encode_ms[i] + offset,
            passing[i] + offset,
            methods[i],
            fontsize=9,
            ha="left",
            va="top",
        )

    plt.xlabel("Encode (ms)")
    plt.ylabel("Percent of time watermark is recovered")
    plt.title(f"Encode time vs Recovery rate at threshold BER={BIT_ERROR_RATE_THRESH}")
    plt.grid(True)
    plt.savefig("plot_encode_vs_encoding_threshold.png")

    ###############################
    ####### ENCODE MS vs BELOW THRESH
    plt.clf()
    plt.figure(figsize=(8, 6))
    plt.scatter(psnrs, passing, color="blue")  # Plotting the points

    # Adding labels to each point
    for i in range(len(psnrs)):
        plt.text(
            psnrs[i] + offset,
            passing[i] + offset,
            methods[i],
            fontsize=9,
            ha="left",
            va="top",
        )

    plt.xlabel("PSNR (Peak signal to noise ratio)")
    plt.ylabel("Percent of time watermark is recovered")
    plt.title(f"Encode time vs Recovery rate at threshold BER={BIT_ERROR_RATE_THRESH}")
    plt.grid(True)
    plt.savefig("plot_psnr_vs_encoding_threshold.png")

    ###############################
    ####### ENCODE MS vs BER
    plt.clf()
    plt.figure(figsize=(8, 6))
    plt.scatter(psnrs, mean_ber, color="blue")  # Plotting the points

    # Adding labels to each point
    for i in range(len(psnrs)):
        plt.text(
            psnrs[i] + offset,
            mean_ber[i] + offset,
            methods[i],
            fontsize=9,
            ha="left",
            va="top",
        )

    plt.xlabel("PSNR (Peak signal to noise ratio)")
    plt.ylabel("Mean Bit Error Rate (BER)")
    plt.title(f"Encode time vs BER")
    plt.grid(True)
    plt.savefig("plot_psnr_vs_ber.png")
