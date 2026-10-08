# Geometry Dash RL Agent

A reinforcement learning agent that learns to play Geometry Dash from screen pixels.

## Setup

```bash
pip3 install gymnasium stable-baselines3 mss pillow opencv-python pynput tensorboard numpy torch torchvision
```

## Usage

### Step 1: Calibrate the capture region
Open Geometry Dash, then:
```bash
python3 -m src.calibrate
# Choose 1, drag a box around the game window, press ENTER
# Choose 2 to verify the agent view looks correct
```

### Step 2: Train
```bash
python3 train.py train --steps 1000000
```

Monitor training in TensorBoard:
```bash
tensorboard --logdir logs/
```

### Step 3: Watch the agent play
```bash
python3 train.py run --model checkpoints/gd_ppo_final
```

## Project Structure

```
src/
  capture.py    — Screen capture and preprocessing
  actions.py    — Sending inputs to the game
  detector.py   — Death detection and reward calculation
  env.py        — Gymnasium environment (the main glue)
  calibrate.py  — Interactive region selection tool
train.py        — Training and evaluation script
region.json     — Your saved capture region (created by calibrate)
checkpoints/    — Saved model weights
logs/           — TensorBoard training logs
```

## Key Concepts

| Concept | File | Description |
|---|---|---|
| Observation space | `env.py` | 4 stacked grayscale 84x84 frames |
| Action space | `actions.py` | Discrete(2): nothing or jump |
| Reward | `detector.py` | +0.01/step alive, -1.0 on death |
| Algorithm | `train.py` | PPO with CnnPolicy (SB3) |
| Frame stacking | `env.py` | Last 4 frames give sense of motion |
