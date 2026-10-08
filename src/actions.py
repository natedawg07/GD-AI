"""
actions.py — Sending inputs to the game

CONCEPT: Action Space
In RL, the "action space" defines what the agent is allowed to do.
Geometry Dash has the simplest possible action space: press or don't press.

    Action 0 = do nothing
    Action 1 = jump (click / spacebar)

This is called a "Discrete(2)" action space in Gym terminology.
Compare this to continuous action spaces (e.g., a steering angle from -1 to 1)
— discrete is easier to learn from because there are fewer choices.

The agent outputs a probability distribution over actions each step:
    [P(do nothing), P(jump)]
and samples from it. Over time, training pushes these probabilities toward
actions that lead to more reward.
"""

import time
from pynput.keyboard import Key, Controller as KeyboardController


keyboard = KeyboardController()

# How long to hold the jump key (seconds). Geometry Dash registers a tap.
JUMP_DURATION = 0.05


def jump():
    """Press and release space — makes the player jump."""
    keyboard.press(Key.space)
    time.sleep(JUMP_DURATION)
    keyboard.release(Key.space)


def do_nothing():
    """Take no action this step."""
    pass


# Map integer actions (from the RL agent) to functions
ACTION_MAP = {
    0: do_nothing,
    1: jump,
}


def execute_action(action: int):
    """
    Execute an action by integer index.

    Args:
        action: 0 (do nothing) or 1 (jump)
    """
    if action not in ACTION_MAP:
        raise ValueError(f"Unknown action: {action}. Must be 0 or 1.")
    ACTION_MAP[action]()
