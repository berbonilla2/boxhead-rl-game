"""
A2C V4 Training Script - Optimized for Consistency and Higher Performance

Based on V3 Analysis:
- V3 Results: Mean 310.93 ± 151.86, Completion 50.66%
- V4 Goals: Mean 400+, Std < 100, Completion 60-70%

Key V4 Improvements:
1. Reduced entropy (0.04 -> 0.028) for more exploitation
2. Increased n-steps (16 -> 20) for better credit assignment
3. Higher value coefficient (0.5 -> 0.7) for stability
4. Smoother LR schedule (25% warmup)
5. Lower gradient clipping (0.5 -> 0.3) for stability
6. Enhanced reward normalization
7. Better environment balance
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
from enhanced_boxhead_env_v4 import EnhancedBoxheadEnvV4

# ======================================================
# CONFIGURATION
# ======================================================
MODEL_VERSION = "v4"
EPISODES = 250  # Increased for more stable convergence
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
# OPTIMIZED NETWORK WITH SKIP CONNECTIONS
# ======================================================
class CustomFeatureExtractorV4(BaseFeaturesExtractor):
    """
    Improved feature extractor with residual-like connections
    for better gradient flow and stability
    """
    def __init__(self, observation_space, features_dim=384):
        super(CustomFeatureExtractorV4, self).__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]  # 42 features
        
        # Layer 1
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.drop1 = nn.Dropout(0.18)  # Increased dropout
        
        # Layer 2
        self.fc2 = nn.Linear(256, 384)
        self.ln2 = nn.LayerNorm(384)
        self.drop2 = nn.Dropout(0.15)
        
        # Layer 3 with skip connection
        self.fc3 = nn.Linear(384, 384)
        self.ln3 = nn.LayerNorm(384)
        self.drop3 = nn.Dropout(0.1)
        
        # Output layer
        self.fc4 = nn.Linear(384, features_dim)
        self.ln4 = nn.LayerNorm(features_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, observations):
        # Layer 1
        x = self.fc1(observations)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.drop1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.drop2(x)
        
        # Layer 3 with residual
        identity = x
        x = self.fc3(x)
        x = self.ln3(x)
        x = self.relu(x)
        x = self.drop3(x)
        x = x + identity  # Skip connection
        
        # Output
        x = self.fc4(x)
        x = self.ln4(x)
        x = self.relu(x)
        
        return x


# ======================================================
# ENHANCED CALLBACK (SAME GRAPHS AS BEFORE)
# ======================================================
class AdvancedMetricsCallbackV4(BaseCallback):
    """Enhanced metrics tracking - keeping same graph structure"""
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
        self.episode_health = []
        self.losses = []
        self.value_losses = []
        self.policy_losses = []
        self.entropies = []
        self.lrs = []
        self.timesteps = []
        
        # Best tracking
        self.best_mean_reward = -np.inf
        self.best_completion_rate = 0
        
        # Plotting - SAME 3x2 GRID AS V3
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
                if "health_remaining" in info["episode"]:
                    self.episode_health.append(info["episode"]["health_remaining"])
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
            if len(self.episode_lengths) >= 20:
                completion = (np.array(self.episode_lengths[-20:]) >= 1000).sum() / 20
                if completion > self.best_completion_rate:
                    self.best_completion_rate = completion
                    print(f"New best completion rate (last 20): {completion*100:.1f}%")
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                print(f"New best mean reward (last 20): {mean_reward:.2f}")
                
        return True

    def _update_plots(self):
        """Same graph layout as V3 for comparison"""
        for ax in self.axes.flatten():
            ax.clear()
        
        # Plot 1: Rewards with trend (SAME AS V3)
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
        
        # Plot 2: Episode Lengths (SAME AS V3)
        if self.episode_lengths:
            self.axes[0, 1].plot(self.episode_lengths, alpha=0.6, color='green')
            self.axes[0, 1].axhline(y=1000, color='r', linestyle='--', alpha=0.5, label='Max Length')
            self.axes[0, 1].set_title('Episode Lengths (Survival)')
            self.axes[0, 1].set_xlabel('Episode')
            self.axes[0, 1].set_ylabel('Steps')
            self.axes[0, 1].legend()
            self.axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Losses (SAME AS V3)
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
        
        # Plot 4: Learning Rate (SAME AS V3)
        if self.lrs:
            self.axes[1, 1].plot(self.timesteps, self.lrs, color='green')
            self.axes[1, 1].set_title('Learning Rate Schedule')
            self.axes[1, 1].set_xlabel('Timesteps')
            self.axes[1, 1].set_ylabel('LR')
            self.axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Kills & Accuracy (SAME AS V3)
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
        
        # Plot 6: Entropy (SAME AS V3)
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
            'health_remaining': (self.episode_health + [np.nan] * max_len)[:max_len],
        })
        metrics_path = os.path.join(RESULTS_DIR, f"metrics_{MODEL_VERSION}_{timestamp}.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved metrics to {metrics_path}")


# ======================================================
# OPTIMIZED LEARNING RATE SCHEDULE
# ======================================================
def warmup_cosine_lr_v4(initial_lr, warmup_fraction=0.25):
    """Longer warmup + smoother cosine decay"""
    def schedule(progress):
        if progress < warmup_fraction:
            # Linear warmup
            return initial_lr * (progress / warmup_fraction)
        else:
            # Cosine annealing
            cosine_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
            return initial_lr * 0.5 * (1 + math.cos(math.pi * cosine_progress))
    return schedule


# ======================================================
# TRAINING FUNCTION
# ======================================================
def train_a2c_v4():
    """Train A2C V4 - Optimized for consistency"""
    print("="*70)
    print("A2C V4 Training - Optimized for Consistency & High Performance")
    print("="*70)
    print("\nV4 Improvements over V3:")
    print("  - Lower entropy: 0.04 -> 0.028 (more exploitation)")
    print("  - Higher n-steps: 16 -> 20 (better credit assignment)")
    print("  - Higher vf_coef: 0.5 -> 0.7 (value stability)")
    print("  - Longer warmup: 20% -> 25% (smoother start)")
    print("  - Lower grad clip: 0.5 -> 0.3 (more stable)")
    print("  - Skip connections in network")
    print("  - Smoother reward signals")
    print("  - Better environment balance")
    print("="*70 + "\n")
    
    # Create environment
    def make_env():
        env = EnhancedBoxheadEnvV4()
        env = Monitor(env, filename=os.path.join(LOG_DIR, f"A2C_{MODEL_VERSION}"))
        return env
    
    env = DummyVecEnv([make_env])
    # More aggressive normalization
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=8.0, clip_reward=8.0)
    
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=8.0, clip_reward=8.0)
    
    # Network with skip connections
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractorV4,
        features_extractor_kwargs=dict(features_dim=384),
        net_arch=dict(pi=[384, 192], vf=[384, 192])
    )
    
    total_timesteps = EPISODES * STEPS_PER_EPISODE
    
    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        device=DEVICE,
        
        # Optimized hyperparameters for V4
        learning_rate=warmup_cosine_lr_v4(3e-4, warmup_fraction=0.25),
        n_steps=20,  # Increased from 16
        gamma=0.99,
        
        # Reduced entropy for more exploitation
        ent_coef=0.028,  # Reduced from 0.04
        vf_coef=0.7,  # Increased from 0.5
        max_grad_norm=0.3,  # Reduced from 0.5
        
        # Optimizer
        rms_prop_eps=1e-5,
        
        # Architecture
        policy_kwargs=policy_kwargs,
        
        # Other
        normalize_advantage=True,
        use_rms_prop=True,
    )
    
    print("\nModel Configuration:")
    print(f"  - Observation Space: 42 features")
    print(f"  - Learning Rate: 3e-4 (25% warmup + cosine)")
    print(f"  - N-steps: 20 (+25% from V3)")
    print(f"  - Gamma: 0.99")
    print(f"  - Entropy: 0.028 (-30% from V3, more exploitation)")
    print(f"  - Value Coef: 0.7 (+40% from V3, more stability)")
    print(f"  - Grad Clip: 0.3 (-40% from V3, smoother updates)")
    print(f"  - Network: 42 -> 256 -> 384 -> 384(skip) -> 384")
    print(f"  - Heads: [384, 192]")
    print(f"  - Total Timesteps: {total_timesteps:,}")
    print(f"  - Dropout: 0.18, 0.15, 0.1 (increased)")
    print(f"  - Skip connections: Enabled")
    print("="*70 + "\n")
    
    # Callbacks
    metrics_callback = AdvancedMetricsCallbackV4(plot_freq=5)
    
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
    episode_health = []
    
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
                        episode_health.append(inf["episode"].get("health_remaining", 0))
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"  Ep {ep+1}/{eval_episodes}: Reward={total_reward:.2f}, Steps={steps}, Kills={episode_kills[-1] if episode_kills else 0}")
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    mean_kills = np.mean(episode_kills) if episode_kills else 0
    mean_accuracy = np.mean(episode_accuracy) if episode_accuracy else 0
    mean_health = np.mean(episode_health) if episode_health else 0
    completion_rate = (np.array(episode_lengths) >= 1000).sum() / len(episode_lengths) * 100
    
    print(f"\nEvaluation Results ({eval_episodes} episodes):")
    print(f"  - Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"  - Mean Episode Length: {mean_length:.2f}")
    print(f"  - Completion Rate: {completion_rate:.1f}%")
    print(f"  - Mean Kills: {mean_kills:.2f}")
    print(f"  - Mean Accuracy: {mean_accuracy*100:.1f}%")
    print(f"  - Mean Health Remaining: {mean_health:.2f}")
    
    # Update log with UTF-8 encoding
    log_entry = f"""
==============================================================================
MODEL VERSION: {MODEL_VERSION}
DATE: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
STATUS: Optimized for Consistency - Training Complete
==============================================================================

V4 IMPROVEMENTS OVER V3:
------------------------
Entropy: 0.04 -> 0.028 (-30%, more exploitation)
N-steps: 16 -> 20 (+25%, better credit assignment)
Value Coef: 0.5 -> 0.7 (+40%, more value stability)
Warmup: 20% -> 25% (+25%, smoother start)
Gradient Clip: 0.5 -> 0.3 (-40%, more stable updates)
Dropout: Increased to 0.18, 0.15, 0.1
Architecture: Added skip connections
Reward Normalization: More aggressive (clip 8.0 vs 10.0)
Environment: Smoother rewards, better balance

ARCHITECTURE:
-------------
Algorithm: A2C
Policy Network: Custom MlpPolicy V4 with Skip Connections
- Input: 42 features (same as V3)
- Feature Extractor: 256 -> 384 -> 384(skip) -> 384
- Policy Head: [384, 192] -> 6 actions
- Value Head: [384, 192] -> 1 value
- Skip Connections: Enabled (layer 3)
- Dropout: 0.18, 0.15, 0.1

HYPERPARAMETERS:
----------------
Learning Rate: 3e-4 (25% warmup + cosine)
N-steps: 20
Gamma: 0.99
Entropy: 0.028
Value Coef: 0.7
Max Grad Norm: 0.3
Normalization: VecNormalize (clip 8.0)

TRAINING CONFIGURATION:
-----------------------
Episodes: {EPISODES}
Steps/Episode: 1000
Total Timesteps: {total_timesteps:,}
Training Time: {training_time/60:.2f} minutes
Device: {DEVICE}

EVALUATION RESULTS:
-------------------
Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}
Mean Length: {mean_length:.2f}
Completion Rate: {completion_rate:.1f}%
Mean Kills: {mean_kills:.2f}
Mean Accuracy: {mean_accuracy*100:.1f}%
Mean Health: {mean_health:.2f}
Eval Episodes: {eval_episodes}

COMPARISON WITH V3:
-------------------
Metric                  V3          V4          Change
Mean Reward           310.93      {mean_reward:.2f}      {mean_reward - 310.93:+.2f} ({(mean_reward/310.93 - 1)*100:+.1f}%)
Std Dev               151.86      {std_reward:.2f}      {std_reward - 151.86:+.2f} ({(std_reward/151.86 - 1)*100:+.1f}%)
Mean Length           872.41      {mean_length:.2f}      {mean_length - 872.41:+.2f} ({(mean_length/872.41 - 1)*100:+.1f}%)
Completion Rate       50.66%      {completion_rate:.1f}%      {completion_rate - 50.66:+.1f}pp

ANALYSIS:
---------
Target: Mean 400+, Std < 100, Completion 60-70%
Achieved: Mean {mean_reward:.2f}, Std {std_reward:.2f}, Completion {completion_rate:.1f}%

Success Metrics:
{'✓ SUCCESS' if mean_reward > 400 else '- Target not reached'}: Mean reward {mean_reward:.2f} (target 400+)
{'✓ SUCCESS' if std_reward < 100 else '- Target not reached'}: Std dev {std_reward:.2f} (target < 100)
{'✓ SUCCESS' if completion_rate >= 60 else '- Target not reached'}: Completion {completion_rate:.1f}% (target 60-70%)

OBSERVATIONS:
-------------
- Skip connections {'improved' if mean_reward > 310.93 else 'did not improve'} performance
- Lower entropy {'increased' if mean_reward > 310.93 else 'decreased'} exploitation effectiveness
- Higher n-steps {'improved' if mean_length > 872.41 else 'did not improve'} credit assignment
- Value regularization {'stabilized' if std_reward < 151.86 else 'did not stabilize'} training

==============================================================================

"""
    
    log_path = os.path.join(BASE_DIR, "model_log.txt")
    with open(log_path, "a", encoding='utf-8') as f:
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
        "completion_rate": completion_rate,
        "mean_kills": mean_kills,
        "mean_accuracy": mean_accuracy,
        "training_time": training_time
    }


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("A2C V4 - CONSISTENCY & PERFORMANCE OPTIMIZATION")
    print("="*70)
    print("\nBased on V3 analysis (Mean 310.93, Std 151.86):")
    print("  Goals: Mean 400+, Std < 100, Completion 60-70%")
    print("\nKey changes:")
    print("  - Lower entropy for exploitation")
    print("  - Higher n-steps for credit assignment")
    print("  - Skip connections for stability")
    print("  - Smoother rewards and learning")
    print("="*70 + "\n")
    
    input("Press Enter to start training...")
    
    results = train_a2c_v4()
    print("\nModel ready!")

