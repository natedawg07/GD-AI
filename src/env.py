"""
env.py — The Geometry Dash Gymnasium Environment

CONCEPT: The RL Loop
This is the heart of the project. The training loop looks like this:

    obs = env.reset()
    while not done:
        action = agent.predict(obs)   # agent picks an action
        obs, reward, done, info = env.step(action)  # game responds
        agent.learn(obs, reward, done)  # agent updates its policy

Each iteration of this loop is called a "timestep".
A full run from level start to death is called an "episode".
The agent trains across thousands of episodes.

CONCEPT: Observation Space vs Action Space
- observation_space: tells the RL library what shape/type the input is.
  We use Box(0, 255, shape=(N_STACK, 84, 84)) — a stack of N grayscale frames.
- action_space: tells the library what actions are valid.
  We use Discrete(2) — two possible actions (0=nothing, 1=jump).

CONCEPT: Frame Stacking
A single frame is a snapshot. But the agent can't tell if the player is
moving up or down from one frame alone — it needs to see motion.
By stacking the last 4 frames together, the agent gets an implicit sense
of velocity and direction. This is a classic trick from the Atari DQN paper.
"""

import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

from .capture import ScreenCapture, OBS_WIDTH, OBS_HEIGHT
from .actions import execute_action
from .detector import GameStateDetector


# Number of frames to stack as a single observation
N_STACK = 4

# Seconds to wait after pressing restart before capturing frames
RESTART_DELAY = 2.0

# How many seconds to pause between steps (controls action frequency)
# At 60fps, stepping every 0.05s = acting every 3 frames
STEP_DELAY = 0.05


class GeometryDashEnv(gym.Env):
    """
    A Gymnasium environment that wraps the live Geometry Dash game.

    The agent sees a stack of 4 grayscale 84x84 frames and can either
    do nothing or jump each step.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, capture_region: dict = None, render_mode: str = None):
        super().__init__()

        # --- Spaces ---
        # observation_space: 4 stacked grayscale frames, pixel values 0-255
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(N_STACK, OBS_HEIGHT, OBS_WIDTH),
            dtype=np.uint8,
        )
        # action_space: 0 = do nothing, 1 = jump
        self.action_space = spaces.Discrete(2)

        # --- Components ---
        self.capture = ScreenCapture(region=capture_region)
        self.detector = GameStateDetector()

        # Frame buffer: holds last N_STACK frames
        # deque with maxlen automatically drops the oldest frame when full
        self._frames: deque = deque(maxlen=N_STACK)

        self.render_mode = render_mode
        self._episode_steps = 0
        self._total_reward = 0.0
        self._episode_count = 0

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """
        Start a new episode.

        In practice this means: the player died, we press restart,
        wait for the level to load, then grab the initial observation.
        """
        super().reset(seed=seed)

        self.detector.reset()
        self._episode_steps = 0
        self._total_reward = 0.0

        # Press R to restart the level (works in GD practice mode / after death)
        self._press_restart()
        time.sleep(RESTART_DELAY)

        # Fill the frame buffer with the first frame (repeat it N_STACK times)
        first_frame = self.capture.grab()
        for _ in range(N_STACK):
            self._frames.append(first_frame)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        """
        Apply one action and return the result.

        Args:
            action: int, 0 or 1

        Returns:
            observation: stacked frames (N_STACK, H, W)
            reward: float
            terminated: bool — True if the episode ended naturally (death)
            truncated: bool — True if we cut the episode short (time limit)
            info: dict — extra debug information
        """
        # 1. Execute the action in the game
        execute_action(action)
        time.sleep(STEP_DELAY)

        # 2. Capture the new frame
        frame = self.capture.grab()
        self._frames.append(frame)

        # 3. Check if the player died
        terminated = self.detector.is_dead(frame)

        # 4. Calculate reward
        if terminated:
            reward = self.detector.get_death_penalty()
            survived = self._episode_steps * STEP_DELAY
            self._episode_count += 1
            print(f"[Episode {self._episode_count}] Death detected — survived {survived:.1f}s ({self._episode_steps} steps)")
        else:
            reward = self.detector.get_progress_reward()

        self._episode_steps += 1
        self._total_reward += reward

        obs = self._get_obs()
        truncated = False  # we'll add a max-step limit later
        info = {
            "episode_steps": self._episode_steps,
            "total_reward": self._total_reward,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Optional: display what the agent sees (useful for debugging)."""
        import cv2
        obs = self._get_obs()
        # Show the most recent frame enlarged
        display = cv2.resize(obs[-1], (336, 336), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Agent View", display)
        cv2.waitKey(1)

    def close(self):
        import cv2
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Stack the frame buffer into a single observation array."""
        return np.array(self._frames, dtype=np.uint8)

    def _press_restart(self):
        """
        Press the key that restarts the level in Geometry Dash.
        After death, pressing Space or clicking restarts.
        We use Space here; adjust if needed.
        """
        from pynput.keyboard import Key, Controller
        kb = Controller()
        kb.press(Key.space)
        time.sleep(0.05)
        kb.release(Key.space)
