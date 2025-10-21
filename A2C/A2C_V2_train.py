"""
A2C V2 Training Script for Boxhead Game
Enhanced with Comprehensive State Representation

States included:
• Position: Current coordinates of the agent on the map
• Enemy Information: Positions and health of nearby zombies
• Agent Status: Current health level
• Resources: Ammo count and currently equipped weapon
• Items: Locations of nearby pickups (ammo, weapons)
• Map Layout: Static features such as walls, chokepoints, and open areas
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
from enhanced_boxhead_env_v2 import EnhancedBoxheadEnvV2

# ======================================================
# CONFIGURATION
# ======================================================
MODEL_VERSION = "v2"  # Version with comprehensive state representation
EPISODES = 150  # Increased for better learning
STEPS_PER_EPISODE = 1000
EVAL_FREQ = 5000  # Evaluate every 5000 steps
CHECKPOINT_FREQ = 10000  # Save checkpoint every 10000 steps

# Directories
BASE_DIR = "A2C"
MODEL_DIR = os.path.join(BASE_DIR, "Models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

for d in [MODEL_DIR, LOG_DIR, RESULTS_DIR, CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {DEVICE}")

# ======================================================
# CUSTOM NETWORK ARCHITECTURE FOR 60 FEATURES
# ======================================================
class CustomFeatureExtractorV2(BaseFeaturesExtractor):
    """
    Custom feature extractor optimized for comprehensive state representation (60 features)
    Deeper network to handle complex state interactions
    """
    def __init__(self, observation_space, features_dim=512):
        super(CustomFeatureExtractorV2, self).__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]  # 60 features
        
        # Extra deep network for rich state representation
        self.feature_net = nn.Sequential(
            # Input layer: 60 -> 256
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Hidden layer 1: 256 -> 512
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Hidden layer 2: 512 -> 512
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.08),
            
            # Hidden layer 3: 512 -> 512
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.05),
            
            # Output layer: 512 -> features_dim
            nn.Linear(512, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations):
        return self.feature_net(observations)


# ======================================================
# ADVANCED CALLBACK
# ======================================================
class AdvancedMetricsCallback(BaseCallback):
    """
    Enhanced callback for tracking and visualizing training metrics
    Now includes accuracy tracking
    """
    def __init__(self, plot_freq=5, save_freq=1000):
        super().__init__()
        self.plot_freq = plot_freq
        self.save_freq = save_freq
        
        # Metrics storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_kills = []
        self.episode_damage = []
        self.episode_accuracy = []
        self.losses = []
        self.value_losses = []
        self.policy_losses = []
        self.entropies = []
        self.lrs = []
        self.timesteps = []
        
        # Best model tracking
        self.best_mean_reward = -np.inf
        
        # Setup plotting
        self.fig, self.axes = plt.subplots(3, 2, figsize=(14, 10))
        plt.ion()
        plt.show(block=False)

    def _on_step(self):
        # Collect episode data
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info.keys():
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                if "kills" in info["episode"]:
                    self.episode_kills.append(info["episode"]["kills"])
                if "damage_taken" in info["episode"]:
                    self.episode_damage.append(info["episode"]["damage_taken"])
                if "accuracy" in info["episode"]:
                    self.episode_accuracy.append(info["episode"]["accuracy"])
        
        return True

    def _on_rollout_end(self):
        # Collect training metrics
        lr = float(self.model.lr_schedule(self.model.num_timesteps))
        self.lrs.append(lr)
        self.timesteps.append(self.model.num_timesteps)
        
        if hasattr(self.model, "logger"):
            logger = self.model.logger.name_to_value
            
            # Collect losses
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
        
        # Check if best model
        if len(self.episode_rewards) >= 10:
            mean_reward = np.mean(self.episode_rewards[-10:])
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                print(f"🏆 New best mean reward: {mean_reward:.2f}")
                
        return True

    def _update_plots(self):
        """Update all training plots"""
        for ax in self.axes.flatten():
            ax.clear()
        
        # Plot 1: Episode Rewards
        if self.episode_rewards:
            self.axes[0, 0].plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
            if len(self.episode_rewards) > 10:
                moving_avg = pd.Series(self.episode_rewards).rolling(window=10).mean()
                self.axes[0, 0].plot(moving_avg, color='red', label='Moving Avg (10)')
            self.axes[0, 0].set_title('Episode Rewards')
            self.axes[0, 0].set_xlabel('Episode')
            self.axes[0, 0].set_ylabel('Reward')
            self.axes[0, 0].legend()
            self.axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Episode Lengths
        if self.episode_lengths:
            self.axes[0, 1].plot(self.episode_lengths, alpha=0.6, color='green')
            self.axes[0, 1].set_title('Episode Lengths (Survival Time)')
            self.axes[0, 1].set_xlabel('Episode')
            self.axes[0, 1].set_ylabel('Steps')
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
            self.axes[1, 1].set_ylabel('Learning Rate')
            self.axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Kills and Accuracy
        if self.episode_kills:
            ax1 = self.axes[2, 0]
            ax1.plot(self.episode_kills, alpha=0.6, color='darkred', label='Kills')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Kills', color='darkred')
            ax1.tick_params(axis='y', labelcolor='darkred')
            ax1.grid(True, alpha=0.3)
            
            if self.episode_accuracy:
                ax2 = ax1.twinx()
                ax2.plot(self.episode_accuracy, alpha=0.6, color='blue', label='Accuracy')
                ax2.set_ylabel('Accuracy', color='blue')
                ax2.tick_params(axis='y', labelcolor='blue')
            
            self.axes[2, 0].set_title('Kills & Accuracy per Episode')
        
        # Plot 6: Entropy
        if self.entropies:
            self.axes[2, 1].plot(self.entropies, alpha=0.6, color='blue')
            self.axes[2, 1].set_title('Policy Entropy (Exploration)')
            self.axes[2, 1].set_xlabel('Rollout')
            self.axes[2, 1].set_ylabel('Entropy')
            self.axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)

    def _on_training_end(self):
        """Save final plots and metrics"""
        plt.ioff()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(RESULTS_DIR, f"training_{MODEL_VERSION}_{timestamp}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(self.fig)
        print(f"📊 Saved training plots to {plot_path}")
        
        # Save metrics to CSV
        max_len = max(len(self.episode_rewards), len(self.episode_kills), len(self.episode_accuracy))
        metrics_df = pd.DataFrame({
            'episode_reward': self.episode_rewards + [np.nan] * (max_len - len(self.episode_rewards)),
            'episode_length': self.episode_lengths + [np.nan] * (max_len - len(self.episode_lengths)),
            'episode_kills': self.episode_kills + [np.nan] * (max_len - len(self.episode_kills)),
            'episode_accuracy': self.episode_accuracy + [np.nan] * (max_len - len(self.episode_accuracy)),
        })
        metrics_path = os.path.join(RESULTS_DIR, f"metrics_{MODEL_VERSION}_{timestamp}.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"📁 Saved metrics to {metrics_path}")


# ======================================================
# LEARNING RATE SCHEDULES
# ======================================================
def warmup_cosine_lr(initial_lr, warmup_fraction=0.1):
    """
    Learning rate schedule with warmup followed by cosine annealing
    """
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
def train_a2c_v2():
    """
    Train A2C V2 with comprehensive state representation
    60 features: Position, Enemy Info, Agent Status, Resources, Items, Map Layout
    """
    print(f"\n{'='*70}")
    print(f"🚀 Starting A2C V2 Training - Comprehensive State Representation")
    print(f"{'='*70}\n")
    
    print("📋 State Representation (60 features):")
    print("  • Position: Agent coordinates (3 features)")
    print("  • Enemy Information: Top 5 enemies with position, health, type (30 features)")
    print("  • Agent Status: Health, damage, kills, accuracy (4 features)")
    print("  • Resources: Ammo, weapons (5 features)")
    print("  • Items: Nearest pickups (6 features)")
    print("  • Map Layout: Walls, chokepoints, open areas (8 features)")
    print("  • Action History: Last 4 actions (4 features)")
    print(f"\n{'='*70}\n")
    
    # Create environment
    def make_env():
        env = EnhancedBoxheadEnvV2()
        env = Monitor(env, filename=os.path.join(LOG_DIR, f"A2C_{MODEL_VERSION}"))
        return env
    
    # Create vectorized environment with normalization
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    
    # Define policy kwargs with custom architecture for 60 features
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractorV2,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128])  # Larger networks for complex state
    )
    
    # Optimized hyperparameters
    total_timesteps = EPISODES * STEPS_PER_EPISODE
    
    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        device=DEVICE,
        
        # Learning parameters
        learning_rate=warmup_cosine_lr(2e-4, warmup_fraction=0.15),  # Lower LR, longer warmup
        n_steps=24,  # Increased for better credit assignment with complex state
        gamma=0.997,  # Slightly higher for long-term planning
        
        # Regularization
        ent_coef=0.025,  # Increased for more exploration with complex state
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping
        
        # Optimizer settings
        rms_prop_eps=1e-5,
        
        # Architecture
        policy_kwargs=policy_kwargs,
        
        # Other
        normalize_advantage=True,
        use_rms_prop=True,
    )
    
    print("\n📋 Model Configuration:")
    print(f"  - Observation Space: 60 features (comprehensive state)")
    print(f"  - Learning Rate: 2e-4 (with 15% warmup + cosine decay)")
    print(f"  - N-steps: 24")
    print(f"  - Gamma: 0.997")
    print(f"  - Entropy Coefficient: 0.025")
    print(f"  - Network Architecture: (60 → 256 → 512 → 512 → 512 → 512)")
    print(f"  - Policy/Value Heads: [512, 256, 128]")
    print(f"  - Total Timesteps: {total_timesteps:,}")
    print(f"  - Observation Normalization: Enabled")
    print(f"  - Reward Normalization: Enabled")
    print(f"\n{'='*70}\n")
    
    # Setup callbacks
    metrics_callback = AdvancedMetricsCallback(plot_freq=5, save_freq=1000)
    
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
    
    # Train the model
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[metrics_callback, checkpoint_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    
    training_time = time.time() - start_time
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"boxhead_A2C_{MODEL_VERSION}.zip")
    model.save(model_path)
    
    # Save normalization statistics
    env.save(os.path.join(MODEL_DIR, f"vecnormalize_{MODEL_VERSION}.pkl"))
    
    print(f"\n✅ Training completed in {training_time/60:.2f} minutes")
    print(f"💾 Model saved to: {model_path}")
    
    # ======================================================
    # EVALUATION
    # ======================================================
    print(f"\n{'='*70}")
    print("🏁 Running Final Evaluation...")
    print(f"{'='*70}\n")
    
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
        print(f"  Episode {ep+1}/{eval_episodes}: Reward={total_reward:.2f}, Steps={steps}, Kills={episode_kills[-1] if episode_kills else 0}")
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    mean_kills = np.mean(episode_kills) if episode_kills else 0
    mean_accuracy = np.mean(episode_accuracy) if episode_accuracy else 0
    
    print(f"\n📊 Evaluation Results ({eval_episodes} episodes):")
    print(f"  - Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  - Mean Episode Length: {mean_length:.2f}")
    print(f"  - Mean Kills: {mean_kills:.2f}")
    print(f"  - Mean Accuracy: {mean_accuracy*100:.1f}%")
    
    # ======================================================
    # UPDATE MODEL LOG
    # ======================================================
    log_entry = f"""
==============================================================================
MODEL VERSION: {MODEL_VERSION}
DATE: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
STATUS: Comprehensive State Representation
==============================================================================

ARCHITECTURE:
-------------
Algorithm: Advantage Actor-Critic (A2C)
Policy Network: Custom MlpPolicy with V2 feature extractor
- Input Layer: 60 features (comprehensive state representation)
- Feature Extractor: [256 → 512 → 512 → 512] with LayerNorm and Dropout
- Policy Head: [512, 256, 128] → 6 actions
- Value Head: [512, 256, 128] → 1 value

COMPREHENSIVE STATE REPRESENTATION (60 features):
-------------------------------------------------
[0-2]   POSITION: Agent coordinates and facing direction
[3-32]  ENEMY INFORMATION: Top 5 enemies (position, health, type, speed)
[33-36] AGENT STATUS: Health, damage taken, kills, accuracy
[37-41] RESOURCES: Ammo count, current weapon, available weapons
[42-47] ITEMS: Nearest ammo and weapon pickups
[48-55] MAP LAYOUT: Wall distances, chokepoints, open areas
[56-59] ACTION HISTORY: Last 4 actions

NEW FEATURES IN V2:
-------------------
✅ Weapon System: Pistol, Shotgun, Rifle (different damage, range, ammo cost)
✅ Ammo Management: Ammo pickups and consumption
✅ Item Pickups: Weapons and ammo spawning in environment
✅ Map Layout: Walls, chokepoints, open areas for strategic positioning
✅ Multiple Enemy Tracking: Top 5 nearest enemies (vs 2 in previous version)
✅ Combat Stats: Shot accuracy tracking
✅ Strategic Positioning: Reward for optimal distance and map usage

HYPERPARAMETERS:
----------------
Learning Rate: 2e-4 (15% warmup + cosine annealing)
Discount Factor (gamma): 0.997
N-steps: 24
Entropy Coefficient: 0.025
Value Function Coefficient: 0.5
Max Gradient Norm: 0.5
RMSprop epsilon: 1e-5
Normalization: Observations and Rewards (VecNormalize)

TRAINING CONFIGURATION:
-----------------------
Total Episodes: {EPISODES}
Steps per Episode: {STEPS_PER_EPISODE}
Total Timesteps: {total_timesteps:,}
Training Time: {training_time/60:.2f} minutes
Device: {DEVICE}

ENHANCED REWARD STRUCTURE:
--------------------------
- Survival reward: +0.1 per step
- Health-based reward: +0.2 * (health/100)
- Optimal distance reward: +0.4 (80-150 pixels from enemy)
- Kill rewards: +8.0 (demon), +5.0 (zombie)
- Hit reward: +1.0
- Ammo pickup: +3.0
- Weapon pickup: +5.0
- Wall collision penalty: -0.05
- Too close penalty: -0.3
- Low ammo penalty: -0.2
- Miss penalty: -0.05
- No ammo penalty: -0.1
- Collision penalty: -0.8
- Death penalty: -50
- Survival bonus: +25 (full episode)

EVALUATION RESULTS:
-------------------
Mean Eval Reward: {mean_reward:.2f} ± {std_reward:.2f}
Mean Episode Length: {mean_length:.2f}
Mean Kills per Episode: {mean_kills:.2f}
Mean Accuracy: {mean_accuracy*100:.1f}%
Episodes Evaluated: {eval_episodes}

IMPROVEMENTS FROM V1:
---------------------
1. ✅ Comprehensive state (60 features vs 25)
2. ✅ Weapon system with 3 weapon types
3. ✅ Ammo management mechanics
4. ✅ Item pickup system
5. ✅ Map layout with strategic features
6. ✅ Multi-enemy tracking (5 vs 2)
7. ✅ Combat accuracy tracking
8. ✅ Deeper network (512 vs 256 hidden)
9. ✅ Larger policy/value heads [512,256,128]
10. ✅ Fine-tuned hyperparameters for complex state

OBSERVATIONS:
-------------
- 60-feature state provides rich tactical information
- Weapon system adds strategic depth
- Ammo management creates resource planning challenge
- Map features enable positional strategies
- Accuracy tracking helps optimize shooting behavior
- Deeper network handles complex state interactions

NEXT STEPS:
-----------
- Analyze weapon usage patterns
- Evaluate ammo management efficiency
- Study map utilization (chokepoints vs open areas)
- Compare performance with v1 (25 features)
- Experiment with curriculum learning
- Try hierarchical policies (macro/micro actions)

==============================================================================

"""
    
    # Append to log file
    log_path = os.path.join(BASE_DIR, "model_log.txt")
    with open(log_path, "a") as f:
        f.write(log_entry)
    
    print(f"\n📝 Updated model log: {log_path}")
    print(f"\n{'='*70}")
    print("✨ Training and evaluation complete!")
    print(f"{'='*70}\n")
    
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
    print("A2C V2 - COMPREHENSIVE STATE REPRESENTATION TRAINING")
    print("="*70)
    print("\nState Components:")
    print("  ✓ Position: Agent location and facing direction")
    print("  ✓ Enemy Info: 5 nearest enemies (position, health, type)")
    print("  ✓ Agent Status: Health, damage, kills, accuracy")
    print("  ✓ Resources: Ammo count, weapon inventory")
    print("  ✓ Items: Pickup locations (ammo, weapons)")
    print("  ✓ Map Layout: Walls, chokepoints, open areas")
    print("="*70 + "\n")
    
    input("Press Enter to start training (or Ctrl+C to cancel)...")
    
    results = train_a2c_v2()
    print("\n🎉 All done! Model ready for deployment.")

