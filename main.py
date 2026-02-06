# main.py
# Astro Pi Mission Space Lab – ISS speed experiment
# Captures images, calculates ISS speed using OpenCV features
# Fully compliant with template

from time import time, sleep
from pathlib import Path
import math
import cv2
import numpy as np
from picamera import PiCamera
from sense_hat import SenseHat

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
MAX_RUNTIME_SECONDS = 595
MAX_IMAGES_TARGET = 42
IMAGE_INTERVAL_SECONDS = 8  # seconds
CAMERA_RESOLUTION = (1280, 960)
CAMERA_FOV_DEGREES = 62.0
FIXED_ALTITUDE_M = 408000.0  # ISS altitude in meters (allowed constant)

Path("data/images").mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------

def get_ground_scale_m_per_pixel():
    """Returns ground scale in meters per pixel"""
    fov_rad = math.radians(CAMERA_FOV_DEGREES)
    angle_per_pixel = fov_rad / CAMERA_RESOLUTION[0]
    return FIXED_ALTITUDE_M * angle_per_pixel

def compute_pixel_shift(img1_gray, img2_gray):
    """Compute median pixel displacement between two images using ORB features"""
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)
    if des1 is None or des2 is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 10:
        return None

    displacements = []
    for m in matches[:50]:
        x1, y1 = kp1[m.queryIdx].pt
        x2, y2 = kp2[m.trainIdx].pt
        displacements.append(math.hypot(x2 - x1, y2 - y1))

    return float(np.median(displacements))

def robust_median(values):
    """Compute median with outlier rejection"""
    if not values:
        return None
    arr = np.array(values)
    med = np.median(arr)
    dev = np.abs(arr - med)
    mad = np.median(dev)
    if mad == 0:
        return float(med)
    return float(np.median(arr[dev <= 2 * mad]))

# -------------------------------------------------------------------------
# MAIN PROGRAM
# -------------------------------------------------------------------------

def main():
    camera = PiCamera()
    camera.resolution = CAMERA_RESOLUTION
    camera.framerate = 1

    try:
        sense = SenseHat()
    except:
        sense = None

    # Capture images
    images = []
    start_time = time()
    for i in range(MAX_IMAGES_TARGET):
        if time() - start_time >= MAX_RUNTIME_SECONDS:
            break
        filename = f"data/images/image_{i:02d}.jpg"
        camera.capture(filename)
        images.append((time(), filename))
        sleep(IMAGE_INTERVAL_SECONDS)  # replay-safe timing

    camera.close()

    # Process consecutive image pairs
    speeds = []
    scale = get_ground_scale_m_per_pixel()

    for i in range(len(images) - 1):
        t1, f1 = images[i]
        t2, f2 = images[i + 1]
        dt = t2 - t1
        if dt <= 0:
            continue

        img1 = cv2.imread(f1)
        img2 = cv2.imread(f2)
        if img1 is None or img2 is None:
            continue

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        shift = compute_pixel_shift(gray1, gray2)
        if shift is None:
            continue

        distance_m = shift * scale
        speed_kms = (distance_m / dt) / 1000
        speeds.append(speed_kms)

    final_speed = robust_median(speeds)
    if final_speed is None:
        final_speed = 0.0  # fallback if no valid images

    # Write result.txt exactly one number, one line
    with open("result.txt", "w") as f:
        f.write(f"{final_speed:.6f}\n")

    # Optional: display on Sense HAT
    if sense:
        sense.show_message(f"{final_speed:.2f} km/s", scroll_speed=0.05)

# -------------------------------------------------------------------------
# FAILSAFE WRAPPER
# -------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except Exception:
        with open("result.txt", "w") as f:
            f.write("0.000000\n")
