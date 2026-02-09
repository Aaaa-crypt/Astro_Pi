# main.py
# Astro Pi Mission Space Lab - ISS speed experiment
# Captures images, calculates ISS speed using OpenCV features
# Fully compliant with template

from time import time, sleep
from pathlib import Path
import math
import cv2
import numpy as np

# Conditional imports with fallback to dummy versions
try:
    from picamera import PiCamera
    PICAMERA_AVAILABLE = True
except ImportError:
    print("WARNING: picamera not available, using dummy camera")
    PICAMERA_AVAILABLE = False

try:
    from sense_hat import SenseHat
    SENSEHAT_AVAILABLE = True
except ImportError:
    print("WARNING: sense_hat not available, using dummy sense hat")
    SENSEHAT_AVAILABLE = False

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
# DUMMY CAMERA/SENSEHAT FOR TESTING
# -------------------------------------------------------------------------
class DummyCamera:
    """Simulated camera that creates realistic test images"""
    def __init__(self):
        self.resolution = CAMERA_RESOLUTION
        self.framerate = 1
    
    def capture(self, filename):
        """Generate a test image with motion-like features"""
        h, w = self.resolution
        # Create gradient image that changes between calls
        # This simulates terrain viewed from above
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Create base terrain pattern
        for i in range(h):
            for j in range(w):
                # Add pattern that shifts to simulate motion
                val = int((i + j) * 255 / (h + w))
                img[i, j] = [val, val//2, 255-val]
        
        # Add some noise/details for feature detection
        noise = np.random.randint(0, 50, (h, w, 3), dtype=np.uint8)
        img = cv2.add(img, noise)
        
        cv2.imwrite(filename, img)
        print(f"DummyCamera: Created test image {filename}")
    
    def close(self):
        pass

class DummySenseHat:
    """Simulated SenseHAT"""
    def show_message(self, text, scroll_speed=0.05):
        print(f"[SenseHAT Display]: {text}")
    
    def clear(self):
        print("[SenseHAT]: Cleared")

# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------

def get_ground_scale_m_per_pixel():
    """Returns ground scale in meters per pixel"""
    fov_rad = math.radians(CAMERA_FOV_DEGREES)
    angle_per_pixel = fov_rad / CAMERA_RESOLUTION[0]
    return FIXED_ALTITUDE_M * math.tan(angle_per_pixel)

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
    """Compute median with MAD-based outlier rejection, return mean of filtered values"""
    if not values:
        return None
    arr = np.array(values)
    med = np.median(arr)
    dev = np.abs(arr - med)
    mad = np.median(dev)
    if mad == 0:
        return float(med)
    filtered = arr[dev <= 2 * mad]
    return float(np.mean(filtered))

# -------------------------------------------------------------------------
# MAIN PROGRAM
# -------------------------------------------------------------------------

def main():
    # Initialize camera - use real or dummy
    if PICAMERA_AVAILABLE:
        camera = PiCamera()
        camera.resolution = CAMERA_RESOLUTION
        camera.framerate = 1
        sleep(2)  # Let real camera stabilize
        print("Using real PiCamera")
    else:
        camera = DummyCamera()
        print("Using DummyCamera for simulation")

    # Initialize SenseHAT - use real or dummy
    if SENSEHAT_AVAILABLE:
        try:
            sense = SenseHat()
            print("Using real SenseHAT")
        except:
            sense = DummySenseHat()
            print("Using DummySenseHat (real init failed)")
    else:
        sense = DummySenseHat()
        print("Using DummySenseHat for simulation")

    # Capture images with explicit timing
    images = []
    start_time = time()
    
    print(f"Starting capture... (target {MAX_IMAGES_TARGET} images)")
    
    for i in range(MAX_IMAGES_TARGET):
        elapsed = time() - start_time
        if elapsed >= MAX_RUNTIME_SECONDS:
            print(f"Time limit reached ({elapsed:.1f}s >= {MAX_RUNTIME_SECONDS}s)")
            break
        
        # Create filename and capture
        filename = f"image_{i:02d}.jpg"
        capture_time = time()
        
        try:
            camera.capture(filename)
            images.append((capture_time, filename))
            print(f"✓ Captured {i+1}/{MAX_IMAGES_TARGET}: {filename}")
        except Exception as e:
            print(f"✗ Capture failed for image {i}: {e}")
            continue
        
        # Wait for next capture (except after last image)
        if i < MAX_IMAGES_TARGET - 1:
            sleep(IMAGE_INTERVAL_SECONDS)

    camera.close()
    print(f"\nCapture complete: {len(images)} images captured")

    # Only process if we captured images
    if len(images) < 2:
        print(f"ERROR: Only captured {len(images)} images, need at least 2")
        final_speed = 0.0
    else:
        # Process consecutive image pairs
        speeds = []
        scale = get_ground_scale_m_per_pixel()
        print(f"\nProcessing image pairs...")
        print(f"Ground scale: {scale:.2f} m/pixel")
        
        for i in range(len(images) - 1):
            t1, f1 = images[i]
            t2, f2 = images[i + 1]
            dt = t2 - t1
            
            if dt <= 0:
                print(f"Pair {i}: Invalid time delta ({dt}s)")
                continue

            # Read images
            img1 = cv2.imread(f1)
            img2 = cv2.imread(f2)
            
            if img1 is None or img2 is None:
                print(f"Pair {i}: Failed to load image(s)")
                continue

            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Compute shift
            shift = compute_pixel_shift(gray1, gray2)
            if shift is None:
                print(f"Pair {i}: No features matched")
                continue

            # Calculate speed
            distance_m = shift * scale
            speed_kms = (distance_m / dt) / 1000
            speeds.append(speed_kms)
            print(f"Pair {i}: {shift:.1f}px shift → {distance_m:.0f}m → {speed_kms:.4f} km/s")
        
        # Final result
        final_speed = robust_median(speeds)
        if final_speed is None:
            final_speed = 0.0
        print(f"\nCalculated {len(speeds)} valid speeds, final median: {final_speed:.6f} km/s")

    # Write result.txt exactly one number, one line
    with open("result.txt", "w") as f:
        f.write(f"{final_speed:.4f}\n") 
    print(final_speed)

    # Optional: display on Sense HAT
    if sense:
        sense.show_message(f"{final_speed:.2f}", scroll_speed=0.05)

# -------------------------------------------------------------------------
# FAILSAFE WRAPPER
# -------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        with open("result.txt", "w") as f:
            f.write("0.000000\n")
