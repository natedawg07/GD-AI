"""
detector.py — Detecting game events from the screen

CONCEPT: Reward Signal
The reward signal is THE most important design decision in RL.
The agent has no idea what "winning" means — it only knows numbers.
A positive reward says "that was good", negative says "that was bad".

Two events we need to detect:
  1. Player death  → end the episode (done=True), give a penalty
  2. Progress      → give a small reward each frame for staying alive

This is called "reward shaping" — engineering the reward to guide learning.

CONCEPT: State Machines
We use a state machine to track sequences of events over time.
No single frame is enough evidence — we need to observe patterns.

  NORMAL  → watching for candidate events
  PENDING → saw enough death candidates, waiting for freeze confirmation

CONCEPT: Portal vs Death classification
Both portals and deaths produce ROI spikes. We separate them with two rules:

  Death signature:
    - ROI spikes AND full frame spikes (explosion envelops the screen)
    - Followed by game freeze (full_diff → ~0)
    - Requires 2+ consecutive candidate frames

  Portal signature:
    - ROI spikes BUT full frame stays quiet (localized light effect)
    - Persists for 3+ consecutive frames (anomalies are only 1 frame)
    - If confirmed portal: cancel any pending death state and reset

The key insight: anomalous single frames near portals never reach the 3-frame
portal threshold AND never reach the 2-frame death candidate threshold at the
same time, so they fall through both filters harmlessly.
"""

import numpy as np
from enum import Enum, auto


# --- ROI: where the player is in the 84x84 observation ---
# GD player sits in the left-center of the screen.
# Columns 5-35 (left third), rows 25-60 (middle band).
ROI_COL_START = 5
ROI_COL_END   = 35
ROI_ROW_START = 25
ROI_ROW_END   = 60

# Mean absolute pixel change in the ROI to flag a frame as noteworthy
ROI_DIFF_THRESHOLD = 8.0

# Minimum full-frame diff required for a death candidate.
# Custom explosion: full≈29-35. Portals: full≈9-16. Gap is clear.
CANDIDATE_FULL_THRESHOLD = 18.0

# Consecutive death candidate frames before entering PENDING
CONSECUTIVE_DEATH_FRAMES = 2

# Consecutive "roi only" frames to confirm a portal (and cancel pending death)
# Anomalies near portals are always single frames; real portals sustain 3+
CONSECUTIVE_PORTAL_FRAMES = 3

# In PENDING: full_diff below this = game froze = death confirmed
# Death freeze ≈ 1.70, portal slowdown bottomed at 2.59
FREEZE_THRESHOLD = 2.0

# Frames to wait in PENDING before giving up (false alarm timeout)
PENDING_TIMEOUT = 6

# Frames to suppress after reset() — covers spawn animation (~3s at 0.05s/step)
POST_RESET_COOLDOWN = 60

# Frames to suppress after confirmed death — covers death animation
POST_DEATH_COOLDOWN = 20


class _State(Enum):
    NORMAL  = auto()  # watching for candidates
    PENDING = auto()  # saw death candidates, waiting for freeze confirmation


class GameStateDetector:
    """
    Two-stage death detector with portal cancellation.

    Death path:   2+ candidate frames (roi+full spike) → PENDING → freeze → DEAD
    Portal path:  3+ roi-only frames → cancel any PENDING state → back to NORMAL
    """

    def __init__(
        self,
        roi_diff_threshold: float = ROI_DIFF_THRESHOLD,
        candidate_full_threshold: float = CANDIDATE_FULL_THRESHOLD,
        consecutive_death_frames: int = CONSECUTIVE_DEATH_FRAMES,
        consecutive_portal_frames: int = CONSECUTIVE_PORTAL_FRAMES,
        freeze_threshold: float = FREEZE_THRESHOLD,
        pending_timeout: int = PENDING_TIMEOUT,
        post_reset_cooldown: int = POST_RESET_COOLDOWN,
        post_death_cooldown: int = POST_DEATH_COOLDOWN,
    ):
        self._roi_diff_threshold       = roi_diff_threshold
        self._candidate_full_threshold = candidate_full_threshold
        self._consecutive_death_frames = consecutive_death_frames
        self._consecutive_portal_frames = consecutive_portal_frames
        self._freeze_threshold         = freeze_threshold
        self._pending_timeout          = pending_timeout
        self._post_reset_cooldown      = post_reset_cooldown
        self._post_death_cooldown      = post_death_cooldown

        self._state: _State = _State.NORMAL
        self._prev_frame: np.ndarray | None = None
        self._death_candidates: int = 0
        self._portal_candidates: int = 0
        self._pending_frames: int = 0
        self._frames_since_reset: int = 0
        self._frames_since_death: int = post_death_cooldown

    def is_dead(self, frame: np.ndarray) -> bool:
        """
        Run the state machine on one new frame.

        Returns:
            True exactly once when death is confirmed.
        """
        self._frames_since_reset += 1
        self._frames_since_death += 1

        in_spawn_cooldown = self._frames_since_reset < self._post_reset_cooldown
        in_death_cooldown = self._frames_since_death < self._post_death_cooldown

        if in_spawn_cooldown or in_death_cooldown or self._prev_frame is None:
            self._prev_frame = frame.copy()
            self._state = _State.NORMAL
            self._death_candidates = 0
            self._portal_candidates = 0
            return False

        # Compute diffs
        roi_curr  = frame[ROI_ROW_START:ROI_ROW_END, ROI_COL_START:ROI_COL_END]
        roi_prev  = self._prev_frame[ROI_ROW_START:ROI_ROW_END, ROI_COL_START:ROI_COL_END]
        roi_diff  = np.abs(roi_curr.astype(np.int16) - roi_prev.astype(np.int16)).mean()
        full_diff = np.abs(frame.astype(np.int16) - self._prev_frame.astype(np.int16)).mean()

        self._prev_frame = frame.copy()

        roi_active  = roi_diff  > self._roi_diff_threshold
        full_active = full_diff > self._candidate_full_threshold

        # --- Portal check (runs every frame) ---
        # ROI spikes but full stays quiet = localized portal light
        if roi_active and not full_active:
            self._portal_candidates += 1
            self._death_candidates = 0  # portal frames can't count toward death
            if self._portal_candidates >= self._consecutive_portal_frames:
                # Confirmed portal — cancel any pending death detection
                self._state = _State.NORMAL
                self._portal_candidates = 0
                self._death_candidates = 0
        else:
            self._portal_candidates = 0

        # --- Death state machine ---
        if self._state is _State.NORMAL:
            if roi_active and full_active:
                self._death_candidates += 1
                if self._death_candidates >= self._consecutive_death_frames:
                    self._state = _State.PENDING
                    self._pending_frames = 0
                    self._death_candidates = 0
            elif not (roi_active and not full_active):
                # Only reset death candidates on truly quiet frames,
                # not while portal candidates are accumulating
                self._death_candidates = 0

        elif self._state is _State.PENDING:
            self._pending_frames += 1

            if full_diff < self._freeze_threshold:
                # Game froze → death confirmed
                self._state = _State.NORMAL
                self._frames_since_death = 0
                self._death_candidates = 0
                self._portal_candidates = 0
                return True

            if self._pending_frames >= self._pending_timeout:
                # Timed out without freeze → false alarm
                self._state = _State.NORMAL
                self._death_candidates = 0

        return False

    def reset(self):
        """Call at the start of each episode."""
        self._state = _State.NORMAL
        self._prev_frame = None
        self._death_candidates = 0
        self._portal_candidates = 0
        self._pending_frames = 0
        self._frames_since_reset = 0
        self._frames_since_death = self._post_death_cooldown

    def get_progress_reward(self) -> float:
        """Small positive reward for each frame survived."""
        return 0.01

    def get_death_penalty(self) -> float:
        """Negative reward on death."""
        return -1.0


def tune_thresholds():
    """
    Run with:
        python3 -c "from src.detector import tune_thresholds; tune_thresholds()"

    Die a few times and hit portals to verify labeling.
    """
    from .capture import ScreenCapture
    from .calibrate import load_region
    import time

    region = load_region()
    cap = ScreenCapture(region=region)
    detector = GameStateDetector()
    prev = None

    print(f"ROI_DIFF={ROI_DIFF_THRESHOLD}  FULL_MIN={CANDIDATE_FULL_THRESHOLD}  "
          f"FREEZE={FREEZE_THRESHOLD}")
    print(f"Death: {CONSECUTIVE_DEATH_FRAMES}+ candidate frames  |  "
          f"Portal: {CONSECUTIVE_PORTAL_FRAMES}+ roi-only frames\n")
    print("[frame]  roi    full   state    flag")

    frame_n = 0
    try:
        while True:
            frame = cap.grab()

            if prev is not None:
                roi_curr  = frame[ROI_ROW_START:ROI_ROW_END, ROI_COL_START:ROI_COL_END]
                roi_prev  = prev[ROI_ROW_START:ROI_ROW_END, ROI_COL_START:ROI_COL_END]
                roi_diff  = np.abs(roi_curr.astype(np.int16) - roi_prev.astype(np.int16)).mean()
                full_diff = np.abs(frame.astype(np.int16) - prev.astype(np.int16)).mean()

                roi_active  = roi_diff  > ROI_DIFF_THRESHOLD
                full_active = full_diff > CANDIDATE_FULL_THRESHOLD

                if roi_active and full_active:
                    flag = "<-- death candidate"
                elif roi_active and not full_active:
                    flag = f"<-- portal ({detector._portal_candidates+1}/{CONSECUTIVE_PORTAL_FRAMES})"
                else:
                    flag = ""

                dead = detector.is_dead(frame)
                state = detector._state.name
                if dead:
                    flag = "*** DEATH CONFIRMED ***"

                print(f"[{frame_n:5d}]  roi={roi_diff:6.2f}  full={full_diff:6.2f}"
                      f"  [{state:7s}]  {flag}")
            else:
                detector.is_dead(frame)

            prev = frame.copy()
            frame_n += 1
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nDone.")
