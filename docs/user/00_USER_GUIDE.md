# 📖 Western Blot: User Guide & Instructions

This guide provides step-by-step instructions for performing high-precision densitometry using the Karcytics Western Blot module.

## Step 1: Image Setup
- **Load Image**: Open your Blot (or Ponceau) stain.
- **Invert**: If the background is light, use "Invert" so your bands appear as bright signals.
- **Rotate**: Use the slider until your lanes are perfectly vertical.
- **Crop**: Drag the handles to focus on the active part of the gel. 

## Step 2: Lane Detection
- **Auto-Detect**: Karcytics analyzes the vertical projection to find your lanes.
- **Manual Adjustment**: Drag existing boundaries to center them on the bands.
- **Right-Click Context Menu**:
    - **Right-click inside a lane** → A menu appears with:
        - **✂️ Split Lane Here**: Inserts a new boundary at the click position, splitting the lane into two.
        - **🚫 Insert Gap**: Creates a 3-way split — the center ~20px is auto-marked as "Exclude" (for empty wells). The gap boundaries are draggable, so you can resize it to match the actual empty region.
    - **Right-click near a boundary** → A menu appears with:
        - **🔗 Merge Lanes**: Removes the boundary, combining the two adjacent lanes into one.
    - Gap/Excluded lanes are visually hatched so they're immediately distinguishable from sample lanes.

## Step 3: Band Detection
- **SNR Threshold**: Adjust the "Min SNR" slider. A value of **3.0** is standard for distinct peaks.
- **Manual Band Clicking**:
    - **Left-Click**: Adds a new band (snaps to the nearest peak).
    - **Shift + Click-Drag**: Manually defines the integration area. Perfect for smears.
    - **Right-Click on Band (▲)**: Instantly deletes the band.

## Step 4: Normalization (Optional)
- **Ponceau S**: If you have a Ponceau stain, load it first. 
- **Total Lane**: Recommended. Sums all Ponceau signal for the most robust loading factor.
- **Reference Band**: Match a single housekeeping protein (e.g., Actin).

## Step 5: Results & Export
- **Fold Change**: Select your "Control Lane" (e.g., Lane 1). All other samples will be scaled relative to it ($Control = 1.0$).
- **Export**: Click the "Export" buttons to save your tables as CSV/Excel or the chart as a high-resolution PNG for publication.

---

## 🖱️ Quick Reference: Mouse & Keyboard

| Action | Result |
| :--- | :--- |
| **Middle-Click + Drag** | Pan around the image. |
| **Mouse Wheel** | Zoom in/out (centered on cursor). |
| **Left-Click (Bands Step)** | Add/toggle a band marker. |
| **Shift + Drag (Bands Step)** | Manually integrate an area. |
| **Right-Click (Bands Step)** | Delete a band marker. |
| **Right-Click (Lanes Step)** | Context menu: Split / Insert Gap / Merge. |
| **Ctrl + Z** | Undo the last action. |
