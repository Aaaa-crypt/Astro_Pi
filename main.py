from time import time, sleep
import math
import cv2
import numpy as np
from pathlib import Path

# hardware imports

try:
    from picamzero import Camera
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False

try:
    from sense_hat import SenseHat
    SENSEHAT_AVAILABLE = True
except ImportError:
    SENSEHAT_AVAILABLE = False

# configuration

NUM_IMAGES = 42
CAPTURE_INTERVAL = 8
MAX_RUNTIME = 595

ORB_FEATURES = 2000
MIN_MATCHES = 10
MIN_KEYPOINTS = 10
MATCH_KEEP_RATIO = 0.3
MAD_THRESHOLD = 2.0

# helper functins

def get_ground_scale_m_per_pixel():
    """
    Calculate meters per pixel using small-angle approximation.
    """
    ISS_ALTITUDE_M = 408000.0
    CAMERA_FOV_DEG = 62.0
    IMAGE_WIDTH = 4056

    fov_rad = math.radians(CAMERA_FOV_DEG)
    angle_per_pixel = fov_rad / IMAGE_WIDTH
    return ISS_ALTITUDE_M * angle_per_pixel


def find_pixel_shift(image1_path, image2_path):
    """
    Compute median pixel displacement between two images using ORB.
    """
    img1 = cv2.imread(str(image1_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(image2_path), cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return None

    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if (
        des1 is None or des2 is None or
        len(kp1) < MIN_KEYPOINTS or
        len(kp2) < MIN_KEYPOINTS
    ):
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    if len(matches) < MIN_MATCHES:
        return None

    matches = sorted(matches, key=lambda m: m.distance)
    matches = matches[:max(int(len(matches) * MATCH_KEEP_RATIO), MIN_MATCHES)]

    displacements = []
    for m in matches:
        x1, y1 = kp1[m.queryIdx].pt
        x2, y2 = kp2[m.trainIdx].pt
        displacements.append(math.hypot(x2 - x1, y2 - y1))

    return float(np.median(displacements))


def calculate_robust_median(values):
    """
    MAD-filtered mean of values.
    """
    if not values:
        return None

    arr = np.array(values)
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))

    if mad == 0:
        return float(median)

    filtered = arr[np.abs(arr - median) <= MAD_THRESHOLD * mad]
    return float(np.mean(filtered))

# main program

def main():
    print("=" * 60)
    print("Astro Pi ISS Speed Estimation")
    print("=" * 60)

    if not CAMERA_AVAILABLE:
        print("[ERROR] Camera not available")
        with open("result.txt", "w") as f:
            f.write("0.0000")
        return

    camera = Camera()
    print("[OK] Camera initialized")

    sense = None
    if SENSEHAT_AVAILABLE:
        try:
            sense = SenseHat()
            sense.clear()
        except:
            pass

    speed_estimates = []
    prev_image = None
    prev_time = None

    scale_m_per_pixel = get_ground_scale_m_per_pixel()
    start_time = time()

    for i in range(NUM_IMAGES):
        if time() - start_time >= MAX_RUNTIME:
            break

        image_path = f"image_{i:02d}.jpg"
        capture_time = time()
        camera.take_photo(image_path)
        print(f"[{i+1:02d}] Captured {image_path}")

        if prev_image is not None:
            dt = capture_time - prev_time
            if dt > 0:
                shift = find_pixel_shift(prev_image, image_path)
                if shift is not None:
                    distance_m = shift * scale_m_per_pixel
                    speed_kms = (distance_m / dt) / 1000.0
                    speed_estimates.append(speed_kms)
                    print(f"    Δ={shift:.1f}px → {speed_kms:.4f} km/s")

            # deletes some of the images
            if i > 1:
                try:
                    Path(prev_image).unlink()
                except:
                    pass

        prev_image = image_path
        prev_time = capture_time

        if i < NUM_IMAGES - 1:
            sleep(CAPTURE_INTERVAL)

    if not speed_estimates:
        final_speed = 0.0
    else:
        final_speed = calculate_robust_median(speed_estimates)

    with open("result.txt", "w") as f:
        f.write(f"{final_speed:.4f}")

    if sense:
        try:
            sense.show_message(f"{final_speed:.2f} km/s", scroll_speed=0.05)
        except:
            pass

    print(f"Final speed: {final_speed:.4f} km/s")
    print("Program complete")

# entry point

if __name__ == "__main__":
    try:
        main()
    except Exception:
        with open("result.txt", "w") as f:
            f.write("0.0000")
