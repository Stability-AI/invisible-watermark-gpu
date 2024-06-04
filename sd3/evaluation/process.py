from imwatermark import WatermarkDecoder, WatermarkEncoder
import cv2
import csv
import collections
import glob
import os
import time 

# A fixed 48-bit message that was choosen at random
# WATERMARK_MESSAGE = 0xB3EC907BB19E
WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
# bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]

encoder = WatermarkEncoder()
encoder.set_watermark('bits', WATERMARK_BITS)

decoder = WatermarkDecoder('bits', len(WATERMARK_BITS))

summaries = collections.defaultdict(list)

generated_imgs = '/weka2/Rahim/Watermarking/_samples/sdv3_api_8b_beta/fireworks/07_fireworks_wm/v0/clip_scores/ema_False/default/000000000'
wm_path = '/weka2/Rahim/Watermarking/_samples/watermarked/invisible-watermark/fireworks/'
# wm_path = '/weka2/Rahim/Watermarking/_samples/watermarked/invisible-watermark/fireworks_SvdCorner/'
# Open the CSV file for writing results
with open('results3.csv', 'w') as csvfile:
    fieldnames = ['image_index', 'algorithm', 'jpeg_quality', 'match_count', 'is_pass', 'read_time', 'write_time', 'encode_time', 'decode_time']
    csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    csv_writer.writeheader()

    # jpeg_quality = [100, 95, 90, 80, 50, 10]
    jpeg_quality = [100]

    # for algorithm in ['dwtDctSvdOptimized']:
    # for algorithm in ['dwtDctSvd']:
    for algorithm in ['dwtSvdCorner']:    
        for img_path in sorted(glob.glob(generated_imgs + '/*.png')):
            for quality in jpeg_quality:
                # Time to read the original image
                start_read_time = time.perf_counter()
                bgr = cv2.imread(img_path)
                read_time = time.perf_counter() - start_read_time

                # Time to encode the image
                start_encode_time = time.perf_counter()
                bgr_encoded = encoder.encode(bgr, algorithm)
                encode_time = time.perf_counter() - start_encode_time

                # Prepare directory and path for saving the encoded image
                filename_save = os.path.join(wm_path, algorithm)
                if not os.path.exists(filename_save):
                    os.makedirs(filename_save)
                encoded_img_path = os.path.join(filename_save, os.path.basename(img_path))
                ### add a some text to the image
                cv2.putText(bgr_encoded, f'{algorithm}_{quality}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imwrite(encoded_img_path, bgr_encoded)

                # Time to save the encoded image
                start_write_time = time.perf_counter()
                if quality < 100:
                    new_img_path = encoded_img_path.replace('.png', f'_{quality}.jpeg')
                    cv2.imwrite(new_img_path, bgr_encoded, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                else:
                    new_img_path = encoded_img_path
                    cv2.imwrite(new_img_path, bgr_encoded)
                write_time = time.perf_counter() - start_write_time

                # Time to decode
                start_decode_time = time.perf_counter()
                bgr_encoded_loaded = cv2.imread(new_img_path)
                watermark_test = decoder.decode(bgr_encoded_loaded, algorithm)
                decode_time = time.perf_counter() - start_decode_time

                # Count matches
                match_count = 0
                for test_bit, ref_bit in zip([int(b) for b in watermark_test], WATERMARK_BITS):
                    if test_bit == ref_bit:
                        match_count += 1

                is_pass = match_count >= 35
                image_idx = os.path.basename(img_path).split('.')[0]

                # Collect and log the time data
                out_dict = {
                    'image_index': image_idx,
                    'algorithm': algorithm,
                    'jpeg_quality': quality,
                    'match_count': match_count,
                    'is_pass': is_pass,
                    'read_time': f'{read_time:.4f}',
                    'write_time': f'{write_time:.4f}',
                    'encode_time': f'{encode_time:.4f}',
                    'decode_time': f'{decode_time:.4f}'
                }
                csv_writer.writerow(out_dict)
                print(f"match_count:{match_count}, Read: {read_time:.4f}s, Write: {write_time:.4f}s, Encode: {encode_time:.4f}s, Decode: {decode_time:.4f}s for {algorithm} at quality {quality}")

               
import csv
from collections import defaultdict
import statistics

# Initialize dictionaries to store data
summaries = defaultdict(list)
timing_summaries = defaultdict(lambda: {'read_times': [], 'write_times': [], 'encode_times': [], 'decode_times': []})

# Read the CSV file and gather data
with open('results3.csv', 'r') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        key = (row['algorithm'], row['jpeg_quality'])
        # Append pass/fail status
        summaries[key].append(row['is_pass'] == 'True')
        # Collect timing data
        timing_summaries[key]['read_times'].append(float(row['read_time']))
        timing_summaries[key]['write_times'].append(float(row['write_time']))
        timing_summaries[key]['encode_times'].append(float(row['encode_time']))
        timing_summaries[key]['decode_times'].append(float(row['decode_time']))

# Calculate and print pass counts and average timings
for key, passes in summaries.items():
    read_avg = statistics.mean(timing_summaries[key]['read_times'])
    write_avg = statistics.mean(timing_summaries[key]['write_times'])
    encode_avg = statistics.mean(timing_summaries[key]['encode_times'])
    decode_avg = statistics.mean(timing_summaries[key]['decode_times'])

    # Calculate standard deviations if applicable
    read_std = statistics.stdev(timing_summaries[key]['read_times']) if len(timing_summaries[key]['read_times']) > 1 else 0
    write_std = statistics.stdev(timing_summaries[key]['write_times']) if len(timing_summaries[key]['write_times']) > 1 else 0
    encode_std = statistics.stdev(timing_summaries[key]['encode_times']) if len(timing_summaries[key]['encode_times']) > 1 else 0
    decode_std = statistics.stdev(timing_summaries[key]['decode_times']) if len(timing_summaries[key]['decode_times']) > 1 else 0

    pass_count = passes.count(True)
    print(f"{key}: Pass Count = {pass_count}, "
          f"Read Avg = {read_avg:.4f}s (±{read_std:.4f}), "
          f"Write Avg = {write_avg:.4f}s (±{write_std:.4f}), "
          f"Encode Avg = {encode_avg:.4f}s (±{encode_std:.4f}), "
          f"Decode Avg = {decode_avg:.4f}s (±{decode_std:.4f})")



                

"""
#######################
('dwtDct', '100') 24
('dwtDct', '95') 15
('dwtDct', '90') 5
('dwtDct', '80') 3
('dwtDct', '50') 1
('dwtDct', '10') 0
#######################
('dwtDctSvd', '100') 30
('dwtDctSvd', '95') 30
('dwtDctSvd', '90') 30
('dwtDctSvd', '80') 30
('dwtDctSvd', '50') 24
('dwtDctSvd', '10') 0
#######################
"""
