# 🧪 Western Blot: Analysis & Mathematics

This document provides the internal logic and mathematical foundations for the Western Blot module. It outlines the transformation from raw pixels to loading-corrected densitometry.

## 1. Lane Detection: 1D Vertical Projection

To handle the 2D image reliably, we first reduce it to a series of vertical 1-D signals. 

### Why Median instead of Mean?
We use the **Median** of the pixel rows across the lane width (`statistic="median"`) as our default 1D summary. The median is far more robust to single-pixel artifacts (like dust or specks) which can skew the mean.

```python
# From peak_analysis.py:
def extract_lane_profile(image, x_start, x_end, ..., statistic="median"):
    # ...
    lane_strip = image[y_start:y_end, strip_left:strip_right]
    if statistic == "median":
        profile = np.median(lane_strip, axis=1)
    # ...
```

## 2. Background Subtraction: The "Rolling Ball"

Karcytics uses a morphological "Rolling Ball" algorithm (similar to ImageJ) for baseline estimation.

### Justification: Rolling Ball vs. Linear
- **Linear Baseline**: Most common in older software. It draws a straight line between the local minima outside a peak. *Problem*: If the background is splotchy or has a gradient (very common in blots), a linear line will either cut through the protein signal or miss the increasing background "hump," leading to over- or under-quantification.
- **Rolling Ball**: A virtual circle of radius $r$ is rolled over the curve. The path traced by the ball is the background. *Benefit*: It accurately tracks non-linear background splotchiness and is geometrically consistent across all lanes, removing "human bias" from the baseline drawing.

```python
# From peak_analysis.py:
def rolling_ball_baseline(profile, radius=50, mode="floor"):
    size = 2 * radius + 1
    if mode == "ceiling":
        baseline = maximum_filter1d(profile, size=size)
    else:
        baseline = minimum_filter1d(profile, size=size)
    
    # Smooth to finalize the baseline path
    baseline = uniform_filter1d(baseline, size=radius)
    return baseline
```

## 3. Signal-to-Noise Ratio (SNR)

Karcytics estimates noise using the **Median Absolute Deviation (MAD)** of the baseline-corrected signal.

$$Noise \approx 1.4826 \times MAD(Signal - Baseline)$$

A peak is only accepted if its height $H$ relative to the baseline $B$ meets the SNR threshold:
$$H / Noise > SNR_{threshold}$$

## 4. Densitometry: Area Under the Curve (AUC)

The final intensity ($I$) is the **definite integral** of the peak $P(y)$ minus the baseline $B(y)$. This accounts for both the height and the breadth of the band.

$$I = \int_{y_{start}}^{y_{end}} (P(y) - B(y)) \,dy$$

In our code, this translates to the sum of the corrected signal within the detected peak's IP (Inter-Peak) bounds.

## 5. Normalization Formulas

### Ponceau Loading Correction
A loading factor $F$ is calculated per lane:
$$F = \frac{Intensity_{Ponceau\_Lane}}{\text{Mean}(Intensity_{All\_Ponceau\_Lanes})}$$

The loading-corrected WB value is then:
$$\text{Corrected} = \text{Intensity}_{WB} / F$$

---

### Fold-Change Normalization
To compute relative change, we divide every sample by the control lane's value:
$$\text{Fold Change} = \frac{\text{Corrected}}{\text{Corrected}_{Control\_Lane}}$$
This ensures the control lane always equals **1.0**.
