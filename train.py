"""
train.py — Main training script

CONCEPT: PPO (Proximal Policy Optimization)
PPO is the RL algorithm we use. Here's the intuition:

The "policy" is a neural network that maps observations → action probabilities.
At each step:
  - The policy looks at the stacked frames and outputs [P(nothing), P(jump)]
  - An action is sampled from this distribution
  - The reward signal tells us if that was a good choice

PPO updates the policy to make good actions more likely. The "proximal" part
means it takes small, careful steps — it won't change the policy too drastically
in one update. This makes training stable.

Alternatives:
  - DQN: estimates Q-values (expected future reward per action). Good for simple
    envs but less stable than PPO for visual inputs.
  - A3C: asynchronous version, runs multiple envs in parallel. More complex.
  - PPO: best default choice for beginners. Stable, well-understood, works well
    with image inputs via CNN.

CONCEPT: CnnPolicy
stable-baselines3's CnnPolicy automatically builds a convolutional neural
network (CNN) to process image observations. CNNs are designed for images:
they learn spatial patterns (edges, shapes, positions) by sliding small
filters across the image. The output of the CNN feeds into a fully-connected
network that outputs action probabilities.

You don't need to define the network architecture manually — SB3 handles it.
"""

import os
import json
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from src.env import GeometryDashEnv
from src.calibrate import load_region

# Where to save model checkpoints and logs
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
REGION_FILE = "region.json"


def make_env(region: dict = None):
    """Factory function to create a single environment instance."""
    def _init():
        return GeometryDashEnv(capture_region=region)
    return _init


def train(total_timesteps: int = 1_000_000, resume: str = None):
    """
    Train a PPO agent to play Geometry Dash.

    Args:
        total_timesteps: how many environment steps to train for
        resume: path to a saved model to continue training from
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Load the capture region (must have run calibration first)
    region = load_region()
    if region is None:
        print("No capture region found. Run calibration first:")
        print("  python3 -m src.calibrate")
        return

    print(f"Using capture region: {region}")

    # CONCEPT: VecEnv (Vectorized Environments)
    # SB3 wraps your environment in a "VecEnv" which can run multiple
    # environments in parallel. Here we use DummyVecEnv (1 env, sequential).
    # Later we could use SubprocVecEnv to run N envs in parallel for faster
    # data collection.
    env = DummyVecEnv([make_env(region)])

    # CONCEPT: Hyperparameters
    # These control how PPO learns. Defaults work well to start.
    # - learning_rate: how big each gradient update step is
    # - n_steps: how many steps to collect before each update
    # - batch_size: how many samples to process per gradient update
    # - n_epochs: how many times to reuse each batch of data
    # - gamma: discount factor — how much to value future vs immediate reward
    #   (0 = only care about now, 1 = care equally about all future rewards)
    # - ent_coef: entropy bonus — encourages exploration (trying new actions)
    ppo_kwargs = dict(
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        ent_coef=0.01,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )

    if resume:
        print(f"Resuming from {resume}")
        model = PPO.load(resume, env=env, **ppo_kwargs)
    else:
        # CnnPolicy: SB3 will auto-build a CNN for image observations
        model = PPO("CnnPolicy", env, **ppo_kwargs)

    # CONCEPT: Callbacks
    # Callbacks run at intervals during training. CheckpointCallback saves
    # the model periodically so you don't lose progress if training crashes.
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="gd_ppo",
    )

    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print("Open TensorBoard to monitor: tensorboard --logdir logs/\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        reset_num_timesteps=resume is None,
    )

    model.save(os.path.join(CHECKPOINT_DIR, "gd_ppo_final"))
    print("\nTraining complete. Model saved to checkpoints/gd_ppo_final")


def run_agent(model_path: str):
    """Load a trained model and watch it play."""
    region = load_region()
    env = GeometryDashEnv(capture_region=region, render_mode="human")
    model = PPO.load(model_path)

    print(f"Running agent from {model_path}. Press Ctrl+C to stop.")
    obs, _ = env.reset()
    episode = 0

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            if terminated or truncated:
                episode += 1
                print(f"Episode {episode} ended | Steps: {info['episode_steps']} | Reward: {info['total_reward']:.2f}")
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or run Geometry Dash RL agent")
    parser.add_argument("mode", choices=["train", "run"], help="train or run")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Training timesteps")
    parser.add_argument("--resume", type=str, default=None, help="Path to model to resume training")
    parser.add_argument("--model", type=str, default=None, help="Path to model for run mode")
    args = parser.parse_args()

    if args.mode == "train":
        train(total_timesteps=args.steps, resume=args.resume)
    elif args.mode == "run":
        if not args.model:
            print("Specify --model path for run mode")
        else:
            run_agent(args.model)
