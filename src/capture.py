"""
capture.py — Screen capture utilities

CONCEPT: Observations
An RL agent can only learn from what it can "see". In Atari games, researchers
fed raw pixels to neural networks and the agent learned to play from scratch.
We do the same here: grab a region of the screen, shrink it to a fixed size,
and convert to grayscale. This becomes our "observation" — the input to our
neural network.

Why grayscale? Color adds 3x the data but rarely adds useful information for
detecting obstacles. Less data = faster training.

Why resize to 84x84? It's a convention from DeepMind's Atari paper (2013).
Small enough to train quickly, large enough to see obstacles.
"""

import mss
import numpy as np
import cv2


# The screen region we capture. Adjust these to frame the Geometry Dash window.
# You can run `python3 -m src.calibrate` to set this interactively.
DEFAULT_REGION = {
    "left": 0,
    "top": 0,
    "width": 1280,
    "height": 720,
}

# Standard observation size used in Atari RL research
OBS_WIDTH = 84
OBS_HEIGHT = 84


class ScreenCapture:
    """Captures a region of the screen and preprocesses it for the RL agent."""

    def __init__(self, region: dict = None):
        self.region = region or DEFAULT_REGION
        self.sct = mss.mss()

    def grab(self) -> np.ndarray:
        """
        Capture one frame from the screen.

        Returns:
            np.ndarray of shape (OBS_HEIGHT, OBS_WIDTH) with dtype uint8.
            Values range 0-255 (pixel brightness).
        """
        raw = self.sct.grab(self.region)
        # mss returns BGRA; convert to BGR then grayscale
        frame = np.array(raw)[:, :, :3]  # drop alpha channel
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (OBS_WIDTH, OBS_HEIGHT), interpolation=cv2.INTER_AREA)
        return resized

    def grab_color(self) -> np.ndarray:
        """Grab a full-color frame (useful for debugging/visualization)."""
        raw = self.sct.grab(self.region)
        return np.array(raw)[:, :, :3]

    def update_region(self, region: dict):
        self.region = region
