# Astro Pi ISS Speed Experiment - main.py Guide

## 🎯 What This Code Does

This program measures the **International Space Station's ground speed** by:
1. Capturing 42 photographs using the Raspberry Pi camera
2. Analyzing consecutive photos to detect ground movement
3. Calculating the ISS speed using computer vision (OpenCV)
4. Outputting the result in km/s

---

## 📋 Program Overview

```
START
  ↓
Initialize PiCamera
  ↓
Capture 42 images (8 seconds apart)
  ↓
For each pair of consecutive images:
  ├─ Find unique landmarks (ORB features)
  ├─ Match landmarks between images
  ├─ Calculate pixel shift
  ├─ Convert to real distance (meters)
  └─ Convert to speed (km/s)
  ↓
Calculate median speed (robust against outliers)
  ↓
Write result to result.txt
  ↓
Display on SenseHAT (if available)
  ↓
END
```

---

## 🔧 Configuration Section (Lines 16-22)

```python
MAX_RUNTIME_SECONDS = 595        # Maximum runtime: ~10 minutes
MAX_IMAGES_TARGET = 42           # Aim for 42 photos
IMAGE_INTERVAL_SECONDS = 8       # 8 seconds between captures
CAMERA_RESOLUTION = (1280, 960)  # High quality: 1280×960 pixels
CAMERA_FOV_DEGREES = 62.0        # Camera field of view angle
FIXED_ALTITUDE_M = 408000.0      # ISS altitude: 408 km
```

**Why these values?**
- 42 images × 8 seconds = 336 seconds (5.6 minutes) of data
- Higher resolution = better feature detection
- 62° field of view matches the actual Raspberry Pi camera
- 408,000 meters is the ISS orbital altitude

---

## 🔍 Helper Functions

### 1. `get_ground_scale_m_per_pixel()` (Lines 30-33)

```python
def get_ground_scale_m_per_pixel():
    fov_rad = math.radians(CAMERA_FOV_DEGREES)
    angle_per_pixel = fov_rad / CAMERA_RESOLUTION[0]
    return FIXED_ALTITUDE_M * angle_per_pixel
```

**Purpose:** Converts pixels to meters on Earth's surface.

**How it works:**
1. Convert camera field of view (62°) to radians
2. Divide by image width (1280 pixels) = degrees per pixel
3. Calculate ground distance using ISS altitude and trigonometry

**Example:** 
- If the result is ~50 meters/pixel
- And we detect 100 pixels of movement
- That means 5,000 meters of ground movement

---

### 2. `compute_pixel_shift(img1_gray, img2_gray)` (Lines 35-50)

```python
def compute_pixel_shift(img1_gray, img2_gray):
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)
    # ... matching logic ...
    return float(np.median(displacements))
```

**Purpose:** Detects how many pixels the ground moved between two consecutive images.

**Step-by-step:**

| Step | What Happens | Why |
|------|--------------|-----|
| Create ORB detector | `orb = cv2.ORB_create(2000)` | ORB finds corners/edges (2000 features max) |
| Find features | `orb.detectAndCompute()` | Identifies unique landmarks in each image |
| Match features | `bf.match()` | Pairs up landmarks between images |
| Calculate shifts | `math.hypot()` | Finds distance each landmark moved |
| Return median | `np.median()` | Uses middle value to ignore outliers |

**Returns:** The median pixel displacement (e.g., 45 pixels)

---

### 3. `robust_median(values)` (Lines 52-63)

```python
def robust_median(values):
    if not values:
        return None
    arr = np.array(values)
    med = np.median(arr)
    dev = np.abs(arr - med)
    mad = np.median(dev)
    if mad == 0:
        return float(med)
    return float(np.median(arr[dev <= 2 * mad]))
```

**Purpose:** Calculates a "super robust" median that removes extreme outliers.

**Why?** Regular median works, but this is better:
- Regular median of `[5, 6, 7, 200]` = 6.5
- Robust median = 6 (ignores the crazy 200)

**How:** Uses Median Absolute Deviation (MAD) to identify and remove extreme outliers.

---

## 🎬 Main Program (Lines 66-142)

### Phase 1: Initialize Hardware (Lines 69-74)

```python
camera = PiCamera()
camera.resolution = CAMERA_RESOLUTION
camera.framerate = 1

try:
    sense = SenseHat()
except:
    sense = None  # If SenseHAT not available, continue anyway
```

- Sets up the Raspberry Pi camera
- Attempts to initialize the SenseHAT (LED board)
- **Fails gracefully** if SenseHAT isn't present

### Phase 2: Capture Images (Lines 77-87)

```python
start_time = time()
for i in range(MAX_IMAGES_TARGET):  # Try to get 42 images
    if time() - start_time >= MAX_RUNTIME_SECONDS:
        break  # Stop if 10 minutes have passed
    
    filename = f"data/images/image_{i:02d}.jpg"
    camera.capture(filename)
    images.append((time(), filename))
    sleep(IMAGE_INTERVAL_SECONDS)  # Wait 8 seconds
```

**What happens:**
- Loops up to 42 times
- Takes a photo, saves it with timestamp
- Waits 8 seconds before next capture
- Stops if total time exceeds 595 seconds (time limit for ISS mission)

**Filename format:** `image_00.jpg`, `image_01.jpg`, ... `image_41.jpg`

### Phase 3: Process Image Pairs (Lines 92-114)

```python
for i in range(len(images) - 1):  # Compare consecutive pairs
    t1, f1 = images[i]      # Image 1 and its timestamp
    t2, f2 = images[i + 1]  # Image 2 and its timestamp
    dt = t2 - t1            # Time elapsed between them
    
    img1 = cv2.imread(f1)
    img2 = cv2.imread(f2)
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    shift = compute_pixel_shift(gray1, gray2)  # How many pixels moved?
    
    distance_m = shift * scale          # Convert pixels to meters
    speed_kms = (distance_m / dt) / 1000  # Distance / Time = Speed
    speeds.append(speed_kms)
```

**The Math:**
```
speed (m/s) = distance (m) / time (s)
speed (km/s) = (distance (m) / time (s)) / 1000
```

Example:
- Shift: 100 pixels
- Scale: 50 meters/pixel → Distance: 5,000 meters
- Time: 8 seconds
- Speed: 5,000m ÷ 8s = 625 m/s = **0.625 km/s**

### Phase 4: Calculate Final Result (Lines 116-121)

```python
final_speed = robust_median(speeds)
if final_speed is None:
    final_speed = 0.0

with open("result.txt", "w") as f:
    f.write(f"{final_speed:.6f}\n")
```

- Takes the robust median of all calculated speeds
- If no valid speeds were found, defaults to 0.0
- **Writes to result.txt with exactly 6 decimal places**

### Phase 5: Display Result (Lines 123-125)

```python
if sense:
    sense.show_message(f"{final_speed:.2f} km/s", scroll_speed=0.05)
```

Displays the result on the SenseHAT LED board (if connected).

---

## 🚨 Known Issues & Fixes

### Issue 1: No Images Captured
**Symptom:** `Result: 0.000000 km/s`

**Cause:** Camera not initialized or file permissions

**Fix:**
- Ensure `data/images/` folder exists and is writable
- Check that PiCamera is available in the environment
- In simulation: the camera may need time to initialize

### Issue 2: Wrong Result Format
**Symptom:** `result.txt is in incorrect format`

**Fix:** The format must be:
- **Exactly one line**
- **One number with 6 decimal places**
- **Followed by a newline**

✅ Correct: `7.225000\n`
❌ Wrong: `7.225000 km/s\n`

### Issue 3: Very Low Speeds (like 0.000000)
**Causes:**
1. **No features found** in images (blank or too uniform)
2. **Camera capture failed** silently
3. **Time synchronization issue** in simulation

**Solutions:**
- Ensure adequate lighting for feature detection
- Use 8-second intervals (too short = harder to detect movement)
- Check that images are actually being saved

---

## 📊 Data Flow Example

```
image_00.jpg (time: 03:22:25)
image_01.jpg (time: 03:22:33) ─┐
                                ├─ Compare
                                │
                     Shift: 87 pixels
                        × 50 m/pixel
                     = 4,350 meters
                        ÷ 8 seconds
                     = 544 m/s
                     = 0.544 km/s  ← Added to speeds[]
```

After processing all 41 pairs:
```
speeds[] = [0.544, 0.538, 0.541, 0.539, 0.542, ...]
                    ↓
            robust_median(speeds)
                    ↓
            final_speed = 0.540 km/s
                    ↓
            Write "0.540000\n" to result.txt
```

---

## 🔬 Key Technical Concepts

### ORB Feature Detector
- **ORB** = Oriented FAST and Rotated BRIEF
- Fast and efficient alternative to SIFT
- Finds corners and edges that are distinctive
- Works well even with rotation/scaling

### Why Compare Consecutive Images?
- More movement = easier to detect
- 8-second interval provides good balance
- Too short = movement too small
- Too long = risk of losing features (terrain changes)

### Why Robust Median?
- Regular average can be skewed by one bad measurement
- Regular median is already good, but we make it better
- MAD (Median Absolute Deviation) identifies outliers mathematically
- Keeps measurements within 2×MAD of the median

---

## ✅ What Success Looks Like

**Expected behavior on ISS:**
```
Starting capture...
✓ Image 0
✓ Image 1
...
✓ Image 41

Processing...
Speed from pair 1: 0.543 km/s
Speed from pair 2: 0.538 km/s
...
Speed from pair 40: 0.541 km/s

Final result: 0.540 km/s
```

**Result format in result.txt:**
```
0.540000
```

---

## 🎓 Summary

This code is an automated **ISS speed measurement system** that:

1. **Captures** 42 images over ~5 minutes
2. **Analyzes** ground motion between consecutive frames using computer vision
3. **Converts** pixel movement to real-world distance using physics
4. **Calculates** speed from distance and time
5. **Outputs** the final robust median speed to `result.txt`

The entire process is automated, fail-safe (catches errors), and produces a single number: the ISS ground speed in km/s.

