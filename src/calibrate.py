"""
calibrate.py — Interactively set the capture region and tune death detection

Run with:  python3 -m src.calibrate

This tool helps you:
1. Draw a rectangle around the game window to set the capture region
2. Verify death detection is working

CONCEPT: Why calibration matters
The RL agent's observation is only as good as what it captures. If the
capture region includes the taskbar or other windows, the agent wastes
capacity learning about irrelevant pixels. Tight framing = better learning.
"""

import mss
import numpy as np
import cv2
import json
import os

REGION_FILE = "region.json"


def select_region():
    """
    Capture the full screen, let the user drag a rectangle over the game window.
    Saves the result to region.json.
    """
    print("Capturing full screen...")
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # primary monitor
        raw = sct.grab(monitor)
        screen = np.array(raw)[:, :, :3]

    print("\nA window will open showing your screen.")
    print("Click and drag to select the Geometry Dash window, then press ENTER.")
    print("Press C to cancel.\n")

    region_selected = {}

    def on_select(event, x, y, flags, param):
        nonlocal region_selected, screen, display
        if event == cv2.EVENT_LBUTTONDOWN:
            param["start"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            param["end"] = (x, y)
            x1, y1 = param["start"]
            x2, y2 = param["end"]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Select Game Region", display)
            region_selected["left"] = min(x1, x2) + monitor["left"]
            region_selected["top"] = min(y1, y2) + monitor["top"]
            region_selected["width"] = abs(x2 - x1)
            region_selected["height"] = abs(y2 - y1)

    display = screen.copy()
    display = cv2.resize(display, (1280, 720))
    scale_x = screen.shape[1] / 1280
    scale_y = screen.shape[0] / 720

    cv2.namedWindow("Select Game Region")
    params = {}
    cv2.setMouseCallback("Select Game Region", on_select, params)
    cv2.imshow("Select Game Region", display)

    while True:
        key = cv2.waitKey(0)
        if key == 13:  # ENTER
            break
        elif key == ord("c"):
            print("Cancelled.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    if not region_selected:
        print("No region selected.")
        return

    # Scale coordinates back to actual screen pixels
    region_selected["left"] = int(region_selected["left"])
    region_selected["top"] = int(region_selected["top"])
    region_selected["width"] = int(region_selected["width"] * scale_x)
    region_selected["height"] = int(region_selected["height"] * scale_y)

    with open(REGION_FILE, "w") as f:
        json.dump(region_selected, f, indent=2)

    print(f"Region saved to {REGION_FILE}:")
    print(json.dumps(region_selected, indent=2))


def verify_capture():
    """Show a live preview of what the agent will see."""
    if not os.path.exists(REGION_FILE):
        print(f"No {REGION_FILE} found. Run select_region() first.")
        return

    with open(REGION_FILE) as f:
        region = json.load(f)

    from .capture import ScreenCapture
    cap = ScreenCapture(region=region)

    print("Showing live agent view (84x84 grayscale). Press Q to quit.")
    while True:
        frame = cap.grab()
        display = cv2.resize(frame, (336, 336), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Agent View (84x84 upscaled)", display)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def load_region() -> dict | None:
    """Load saved region from disk, or None if not calibrated yet."""
    if os.path.exists(REGION_FILE):
        with open(REGION_FILE) as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    print("=== Geometry Dash Capture Calibration ===\n")
    print("1. Select game region")
    print("2. Verify agent view")
    choice = input("\nChoice (1/2): ").strip()
    if choice == "1":
        select_region()
    elif choice == "2":
        verify_capture()
