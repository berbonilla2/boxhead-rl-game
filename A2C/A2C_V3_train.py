"""
A2C V3 Training Script - Optimized based on V2 Analysis

Key Improvements:
- Reduced state complexity (60 -> 42 features)
- Simpler network architecture (less overfitting)
- Better hyperparameters (based on V2 results)
- Improved reward shaping
- Enhanced regularization
- Fixed Unicode encoding errors
"""

import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from datetime import datetime
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from enhanced_boxhead_env_v3 import EnhancedBoxheadEnvV3

# ======================================================
# CONFIGURATION
# ======================================================
MODEL_VERSION = "v3"
EPISODES = 200  # Increased for more stable learning
STEPS_PER_EPISODE = 1000
EVAL_FREQ = 5000
CHECKPOINT_FREQ = 10000

# Directories
BASE_DIR = "A2C"
MODEL_DIR = os.path.join(BASE_DIR, "Models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

for d in [MODEL_DIR, LOG_DIR, RESULTS_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ======================================================
# SIMPLIFIED NETWORK ARCHITECTURE
# ======================================================
class CustomFeatureExtractorV3(BaseFeaturesExtractor):
    """
    Simplified feature extractor for 42 features
    Reduced depth and width to prevent overfitting
    """
    def __init__(self, observation_space, features_dim=384):
        super(CustomFeatureExtractorV3, self).__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]  # 42 features
        
        # Simpler, more focused network
        self.feature_net = nn.Sequential(
            # Input layer: 42 -> 256
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.15),  # Increased dropout
            
            # Hidden layer 1: 256 -> 384
            nn.Linear(256, 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(0.12),
            
            # Hidden layer 2: 384 -> 384
            nn.Linear(384, 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(0.08),
            
            # Output layer: 384 -> features_dim
            nn.Linear(384, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations):
        return self.feature_net(observations)


# ======================================================
# ENHANCED CALLBACK
# ======================================================
class AdvancedMetricsCallbackV3(BaseCallback):
    """Enhanced metrics with better tracking"""
    def __init__(self, plot_freq=5, save_freq=1000):
        super().__init__()
        self.plot_freq = plot_freq
        self.save_freq = save_freq
        
        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_kills = []
        self.episode_accuracy = []
        self.episode_ammo_collected = []
        self.losses = []
        self.value_losses = []
        self.policy_losses = []
        self.entropies = []
        self.lrs = []
        self.timesteps = []
        
        # Best tracking
        self.best_mean_reward = -np.inf
        self.episodes_without_improvement = 0
        
        # Plotting
        self.fig, self.axes = plt.subplots(3, 2, figsize=(14, 10))
        plt.ion()
        plt.show(block=False)

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info.keys():
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                if "kills" in info["episode"]:
                    self.episode_kills.append(info["episode"]["kills"])
                if "accuracy" in info["episode"]:
                    self.episode_accuracy.append(info["episode"]["accuracy"])
                if "ammo_collected" in info["episode"]:
                    self.episode_ammo_collected.append(info["episode"]["ammo_collected"])
        return True

    def _on_rollout_end(self):
        lr = float(self.model.lr_schedule(self.model.num_timesteps))
        self.lrs.append(lr)
        self.timesteps.append(self.model.num_timesteps)
        
        if hasattr(self.model, "logger"):
            logger = self.model.logger.name_to_value
            if "train/loss" in logger:
                self.losses.append(float(logger["train/loss"]))
            if "train/value_loss" in logger:
                self.value_losses.append(float(logger["train/value_loss"]))
            if "train/policy_loss" in logger:
                self.policy_losses.append(float(logger["train/policy_loss"]))
            if "train/entropy_loss" in logger:
                self.entropies.append(float(logger["train/entropy_loss"]))
        
        # Update plots
        if len(self.episode_rewards) > 0 and len(self.episode_rewards) % self.plot_freq == 0:
            self._update_plots()
        
        # Track improvement
        if len(self.episode_rewards) >= 20:
            mean_reward = np.mean(self.episode_rewards[-20:])
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.episodes_without_improvement = 0
                print(f"New best mean reward (last 20): {mean_reward:.2f}")
            else:
                self.episodes_without_improvement += 1
                
        return True

    def _update_plots(self):
        for ax in self.axes.flatten():
            ax.clear()
        
        # Plot 1: Rewards with trend
        if self.episode_rewards:
            self.axes[0, 0].plot(self.episode_rewards, alpha=0.5, label='Episode Reward')
            if len(self.episode_rewards) > 20:
                moving_avg = pd.Series(self.episode_rewards).rolling(window=20).mean()
                self.axes[0, 0].plot(moving_avg, color='red', linewidth=2, label='MA(20)')
            self.axes[0, 0].set_title('Episode Rewards')
            self.axes[0, 0].set_xlabel('Episode')
            self.axes[0, 0].set_ylabel('Reward')
            self.axes[0, 0].legend()
            self.axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Episode Lengths
        if self.episode_lengths:
            self.axes[0, 1].plot(self.episode_lengths, alpha=0.6, color='green')
            self.axes[0, 1].axhline(y=1000, color='r', linestyle='--', alpha=0.5, label='Max Length')
            self.axes[0, 1].set_title('Episode Lengths (Survival)')
            self.axes[0, 1].set_xlabel('Episode')
            self.axes[0, 1].set_ylabel('Steps')
            self.axes[0, 1].legend()
            self.axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Losses
        if self.losses:
            self.axes[1, 0].plot(self.losses, alpha=0.6, color='red', label='Total Loss')
            if self.value_losses:
                self.axes[1, 0].plot(self.value_losses, alpha=0.6, color='orange', label='Value Loss')
            if self.policy_losses:
                self.axes[1, 0].plot(self.policy_losses, alpha=0.6, color='purple', label='Policy Loss')
            self.axes[1, 0].set_title('Training Losses')
            self.axes[1, 0].set_xlabel('Rollout')
            self.axes[1, 0].set_ylabel('Loss')
            self.axes[1, 0].legend()
            self.axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Learning Rate
        if self.lrs:
            self.axes[1, 1].plot(self.timesteps, self.lrs, color='green')
            self.axes[1, 1].set_title('Learning Rate Schedule')
            self.axes[1, 1].set_xlabel('Timesteps')
            self.axes[1, 1].set_ylabel('LR')
            self.axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Performance Metrics
        if self.episode_kills and self.episode_accuracy:
            ax1 = self.axes[2, 0]
            ax1.plot(self.episode_kills, alpha=0.6, color='darkred', label='Kills')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Kills', color='darkred')
            ax1.tick_params(axis='y', labelcolor='darkred')
            
            ax2 = ax1.twinx()
            ax2.plot(self.episode_accuracy, alpha=0.6, color='blue', label='Accuracy')
            ax2.set_ylabel('Accuracy', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            ax1.set_title('Kills & Accuracy')
            ax1.grid(True, alpha=0.3)
        
        # Plot 6: Entropy
        if self.entropies:
            self.axes[2, 1].plot(self.entropies, alpha=0.6, color='blue')
            self.axes[2, 1].set_title('Policy Entropy')
            self.axes[2, 1].set_xlabel('Rollout')
            self.axes[2, 1].set_ylabel('Entropy')
            self.axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)

    def _on_training_end(self):
        plt.ioff()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(RESULTS_DIR, f"training_{MODEL_VERSION}_{timestamp}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(self.fig)
        print(f"Saved training plots to {plot_path}")
        
        # Save metrics
        max_len = len(self.episode_rewards)
        metrics_df = pd.DataFrame({
            'episode_reward': self.episode_rewards,
            'episode_length': self.episode_lengths[:max_len],
            'episode_kills': (self.episode_kills + [np.nan] * max_len)[:max_len],
            'episode_accuracy': (self.episode_accuracy + [np.nan] * max_len)[:max_len],
            'ammo_collected': (self.episode_ammo_collected + [np.nan] * max_len)[:max_len],
        })
        metrics_path = os.path.join(RESULTS_DIR, f"metrics_{MODEL_VERSION}_{timestamp}.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved metrics to {metrics_path}")


# ======================================================
# LEARNING RATE SCHEDULE
# ======================================================
def warmup_cosine_lr(initial_lr, warmup_fraction=0.2):
    """Warmup + cosine decay (longer warmup)"""
    def schedule(progress):
        if progress < warmup_fraction:
            return initial_lr * (progress / warmup_fraction)
        else:
            cosine_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
            return initial_lr * 0.5 * (1 + math.cos(math.pi * cosine_progress))
    return schedule


# ======================================================
# TRAINING FUNCTION
# ======================================================
def train_a2c_v3():
    """Train A2C V3 with optimizations based on V2 analysis"""
    print("="*70)
    print("A2C V3 Training - Optimized Architecture")
    print("="*70)
    print("\nV3 Improvements over V2:")
    print("  - Reduced state: 60 -> 42 features")
    print("  - Simpler network: 512 -> 384 hidden units")
    print("  - Better hyperparameters (based on V2 results)")
    print("  - Improved reward shaping")
    print("  - Enhanced regularization")
    print("  - Fixed encoding errors")
    print("="*70 + "\n")
    
    # Create environment
    def make_env():
        env = EnhancedBoxheadEnvV3()
        env = Monitor(env, filename=os.path.join(LOG_DIR, f"A2C_{MODEL_VERSION}"))
        return env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    
    # Simplified policy architecture
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractorV3,
        features_extractor_kwargs=dict(features_dim=384),
        net_arch=dict(pi=[384, 192], vf=[384, 192])  # Smaller heads
    )
    
    total_timesteps = EPISODES * STEPS_PER_EPISODE
    
    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        device=DEVICE,
        
        # Optimized hyperparameters based on V2 analysis
        learning_rate=warmup_cosine_lr(3e-4, warmup_fraction=0.2),  # Increased LR, longer warmup
        n_steps=16,  # Reduced from 24
        gamma=0.99,  # Reduced from 0.997 for faster learning
        
        # Increased exploration
        ent_coef=0.04,  # Increased from 0.025
        vf_coef=0.5,
        max_grad_norm=0.5,
        
        # Optimizer
        rms_prop_eps=1e-5,
        
        # Architecture
        policy_kwargs=policy_kwargs,
        
        # Other
        normalize_advantage=True,
        use_rms_prop=True,
    )
    
    print("\nModel Configuration:")
    print(f"  - Observation Space: 42 features (optimized)")
    print(f"  - Learning Rate: 3e-4 (20% warmup)")
    print(f"  - N-steps: 16")
    print(f"  - Gamma: 0.99")
    print(f"  - Entropy: 0.04 (increased exploration)")
    print(f"  - Network: 42 -> 256 -> 384 -> 384 -> 384")
    print(f"  - Heads: [384, 192]")
    print(f"  - Total Timesteps: {total_timesteps:,}")
    print(f"  - Dropout: 0.15, 0.12, 0.08 (regularization)")
    print("="*70 + "\n")
    
    # Callbacks
    metrics_callback = AdvancedMetricsCallbackV3(plot_freq=5)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix=f"a2c_{MODEL_VERSION}_checkpoint",
        save_vecnormalize=True
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Train
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[metrics_callback, checkpoint_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    
    training_time = time.time() - start_time
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"boxhead_A2C_{MODEL_VERSION}.zip")
    model.save(model_path)
    env.save(os.path.join(MODEL_DIR, f"vecnormalize_{MODEL_VERSION}.pkl"))
    
    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    print(f"Model saved to: {model_path}")
    
    # Evaluation
    print("\n" + "="*70)
    print("Final Evaluation")
    print("="*70 + "\n")
    
    eval_episodes = 20
    episode_rewards = []
    episode_lengths = []
    episode_kills = []
    episode_accuracy = []
    
    for ep in range(eval_episodes):
        obs = eval_env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            total_reward += reward[0]
            steps += 1
            
            if done:
                for inf in info:
                    if "episode" in inf:
                        episode_kills.append(inf["episode"].get("kills", 0))
                        episode_accuracy.append(inf["episode"].get("accuracy", 0))
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"  Ep {ep+1}/{eval_episodes}: Reward={total_reward:.2f}, Steps={steps}, Kills={episode_kills[-1] if episode_kills else 0}")
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    mean_kills = np.mean(episode_kills) if episode_kills else 0
    mean_accuracy = np.mean(episode_accuracy) if episode_accuracy else 0
    
    print(f"\nEvaluation Results ({eval_episodes} episodes):")
    print(f"  - Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"  - Mean Episode Length: {mean_length:.2f}")
    print(f"  - Mean Kills: {mean_kills:.2f}")
    print(f"  - Mean Accuracy: {mean_accuracy*100:.1f}%")
    
    # Update log (with UTF-8 encoding)
    log_entry = f"""
==============================================================================
MODEL VERSION: {MODEL_VERSION}
DATE: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
STATUS: Optimized Architecture - Training Complete
==============================================================================

V3 IMPROVEMENTS OVER V2:
------------------------
State Complexity: 60 features -> 42 features (-30%)
Network Size: 512 hidden -> 384 hidden (-25%)
Learning Rate: 2e-4 -> 3e-4 (+50%)
N-steps: 24 -> 16 (-33%)
Gamma: 0.997 -> 0.99 (faster learning)
Entropy: 0.025 -> 0.04 (+60% exploration)
Warmup: 15% -> 20% (more stable start)
Dropout: Increased (0.15, 0.12, 0.08)

ARCHITECTURE:
-------------
Algorithm: A2C
Policy Network: Custom MlpPolicy V3
- Input: 42 features (optimized state)
- Feature Extractor: 256 -> 384 -> 384 -> 384
- Policy Head: [384, 192] -> 6 actions
- Value Head: [384, 192] -> 1 value
- Dropout: 0.15, 0.12, 0.08

STATE REPRESENTATION (42 features):
-----------------------------------
[0-7]   Position & Status (8)
[8-25]  Enemy Info - 3 enemies (18)
[26-33] Resources & Items (8)
[34-39] Map Tactical (6)
[40-41] Temporal (2)

HYPERPARAMETERS:
----------------
Learning Rate: 3e-4 (20% warmup + cosine)
N-steps: 16
Gamma: 0.99
Entropy: 0.04
Value Coef: 0.5
Max Grad Norm: 0.5
Normalization: VecNormalize

TRAINING CONFIGURATION:
-----------------------
Episodes: {EPISODES}
Steps/Episode: {STEPS_PER_EPISODE}
Total Timesteps: {total_timesteps:,}
Training Time: {training_time/60:.2f} minutes
Device: {DEVICE}

EVALUATION RESULTS:
-------------------
Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}
Mean Length: {mean_length:.2f}
Mean Kills: {mean_kills:.2f}
Mean Accuracy: {mean_accuracy*100:.1f}%
Eval Episodes: {eval_episodes}

COMPARISON WITH V2:
-------------------
V2 Mean Reward: 67.57
V3 Mean Reward: {mean_reward:.2f}
Change: {mean_reward - 67.57:+.2f} ({(mean_reward/67.57 - 1)*100:+.1f}%)

V2 Mean Length: 603.85
V3 Mean Length: {mean_length:.2f}
Change: {mean_length - 603.85:+.2f} ({(mean_length/603.85 - 1)*100:+.1f}%)

V2 Completion Rate: 10.08%
V3 Completion Rate: {(np.array(episode_lengths) >= 1000).sum() / len(episode_lengths) * 100:.2f}%

OBSERVATIONS:
-------------
- Simpler state representation improved stability
- Smaller network reduced overfitting
- Higher entropy increased exploration
- Better reward shaping reduced negative spirals
- Increased regularization (dropout) helped generalization

NEXT STEPS:
-----------
- If V3 > V2: Continue optimizing hyperparameters
- If V3 <= V2: Try curriculum learning or different architecture
- Consider PPO for comparison
- Experiment with recurrent policies (LSTM)

==============================================================================

"""
    
    log_path = os.path.join(BASE_DIR, "model_log.txt")
    with open(log_path, "a", encoding='utf-8') as f:  # UTF-8 encoding
        f.write(log_entry)
    
    print(f"\nUpdated model log: {log_path}")
    print("="*70)
    print("Training complete!")
    print("="*70 + "\n")
    
    return {
        "model": model,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "mean_kills": mean_kills,
        "mean_accuracy": mean_accuracy,
        "training_time": training_time
    }


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("A2C V3 - OPTIMIZED TRAINING")
    print("="*70)
    print("\nBased on V2 analysis:")
    print("  - Simplified state (42 vs 60 features)")
    print("  - Smaller network (less overfitting)")
    print("  - Better hyperparameters")
    print("  - Improved rewards")
    print("="*70 + "\n")
    
    input("Press Enter to start training...")
    
    results = train_a2c_v3()
    print("\nModel ready for deployment!")

