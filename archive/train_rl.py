import os
import time
import math
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from game import Player, Enemy, Demon, distance, random_spawn_position

# ======================================================
# CONFIGURATION
# ======================================================
ALGORITHMS = ["DQN", "PPO", "A2C"]
EPISODES = 50
STEPS_PER_EPISODE = 800
LOG_DIR = "./logs"
MODEL_DIR = "./models"
RESULTS_DIR = "./results"
for d in [LOG_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {DEVICE}")

WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480
PLAYER_SPEED = 2.0
ZOMBIE_SPEED = 0.7
DEMON_SPEED = 0.9
ENEMY_SPAWN_RATE = 0.004


# ======================================================
# ENVIRONMENT
# ======================================================
class BoxheadEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BoxheadEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(6)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player = Player(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
        self.zombies = [Enemy(*random_spawn_position(), (255, 0, 0), ZOMBIE_SPEED) for _ in range(4)]
        self.demons = [Demon(*random_spawn_position()) for _ in range(2)]
        self.steps = 0
        self.total_reward = 0
        self.player.health = 100
        return self._get_obs(), {}

    def _get_obs(self):
        if self.zombies or self.demons:
            enemies = self.zombies + self.demons
            nearest = min(enemies, key=lambda e: distance((self.player.x, self.player.y), (e.x, e.y)))
            dx = (nearest.x - self.player.x) / WINDOW_WIDTH
            dy = (nearest.y - self.player.y) / WINDOW_HEIGHT
            dist = distance((self.player.x, self.player.y), (nearest.x, nearest.y))
        else:
            dx = dy = dist = 0

        obs = np.array([
            self.player.x / WINDOW_WIDTH,
            self.player.y / WINDOW_HEIGHT,
            self.player.health / 100,
            dx, dy,
            dist / (math.hypot(WINDOW_WIDTH, WINDOW_HEIGHT)),
            0, 0, 0
        ], dtype=np.float32)
        return obs

    def step(self, action):
        self.steps += 1
        reward = 0.0

        # === Movement ===
        if action == 1:
            self.player.y -= PLAYER_SPEED
        elif action == 2:
            self.player.y += PLAYER_SPEED
        elif action == 3:
            self.player.x -= PLAYER_SPEED
        elif action == 4:
            self.player.x += PLAYER_SPEED

        # === Boundaries ===
        self.player.x = np.clip(self.player.x, 0, WINDOW_WIDTH)
        self.player.y = np.clip(self.player.y, 0, WINDOW_HEIGHT)

        # === Nearest enemy distance ===
        if self.zombies or self.demons:
            enemies = self.zombies + self.demons
            nearest = min(enemies, key=lambda e: distance((self.player.x, self.player.y), (e.x, e.y)))
            dist = distance((self.player.x, self.player.y), (nearest.x, nearest.y))
        else:
            dist = 999

        # === Reward shaping ===
        reward += 0.05                           # survival reward
        reward += 0.1 * (self.player.health / 100)
        reward += (1.0 - min(dist / 500, 1.0)) * 0.2
        if action == 5:
            reward -= 0.01  # discourage spam shooting

        # === Health decay ===
        self.player.health -= 0.1
        if self.player.health <= 0:
            reward -= 30

        self.total_reward += reward
        done = self.player.health <= 0 or self.steps >= STEPS_PER_EPISODE
        truncated = False
        info = {}
        if done:
            info["episode"] = {"r": self.total_reward, "l": self.steps}
        return self._get_obs(), reward, done, truncated, info

    def render(self): pass


# ======================================================
# CALLBACK
# ======================================================
class MetricsCallback(BaseCallback):
    def __init__(self, algo_name, plot_freq=10):
        super().__init__()
        self.algo_name = algo_name
        self.plot_freq = plot_freq
        self.episode_rewards, self.losses, self.lrs = [], [], []
        self.fig, self.ax = plt.subplots(3, 1, figsize=(8, 8))
        plt.ion(); plt.show(block=False)

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info.keys():
                self.episode_rewards.append(info["episode"]["r"])
        return True

    def _on_rollout_end(self):
        lr = float(self.model.lr_schedule(self.model.num_timesteps))
        self.lrs.append(lr)
        if hasattr(self.model, "logger"):
            loss = self.model.logger.name_to_value.get("train/loss")
            if loss is not None:
                self.losses.append(float(loss))
        if len(self.episode_rewards) > 0 and len(self.episode_rewards) % self.plot_freq == 0:
            self.ax[0].cla(); self.ax[1].cla(); self.ax[2].cla()
            self.ax[0].plot(self.episode_rewards, color="blue", label="Reward")
            self.ax[1].plot(self.losses, color="red", label="Loss")
            self.ax[2].plot(self.lrs, color="green", label="LR")
            for a in self.ax: a.legend(); a.grid(True)
            plt.tight_layout(); plt.pause(0.01)
        return True

    def _on_training_end(self):
        plt.ioff()
        plt.savefig(os.path.join(RESULTS_DIR, f"{self.algo_name}_training.png"))
        plt.close(self.fig)
        print(f"📊 Saved {self.algo_name}_training.png")


# ======================================================
# COSINE LR
# ======================================================
def cosine_lr(initial_lr):
    return lambda progress: initial_lr * (0.5 * (1 + math.cos(math.pi * progress)))


# ======================================================
# TRAIN + EVAL
# ======================================================
def train_and_eval(algo_name):
    print(f"\n🚀 Training {algo_name}")
    env = DummyVecEnv([lambda: Monitor(BoxheadEnv(), filename=os.path.join(LOG_DIR, f"{algo_name}.monitor.csv"))])
    callback = MetricsCallback(algo_name)
    total_steps = EPISODES * STEPS_PER_EPISODE

    if algo_name == "DQN":
        model = DQN("MlpPolicy", env, verbose=1, device=DEVICE,
                    learning_rate=cosine_lr(5e-4),
                    buffer_size=50000, batch_size=64,
                    train_freq=4, exploration_fraction=0.7)
    elif algo_name == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, device=DEVICE,
                    n_steps=512, batch_size=64,
                    ent_coef=0.01, learning_rate=cosine_lr(3e-4))
    elif algo_name == "A2C":
        model = A2C("MlpPolicy", env, verbose=1, device=DEVICE,
                    learning_rate=cosine_lr(7e-4), n_steps=8)

    start = time.time()
    model.learn(total_timesteps=total_steps, callback=callback)
    model.save(os.path.join(MODEL_DIR, f"boxhead_{algo_name}.zip"))
    print(f"✅ {algo_name} finished in {time.time() - start:.2f}s")

    # === Evaluation ===
    eval_env = DummyVecEnv([lambda: Monitor(BoxheadEnv())])
    episode_rewards = []
    for _ in range(10):
        obs = eval_env.reset()
        done, total_r = False, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)   # ✅ FIXED HERE
            total_r += reward[0]
        episode_rewards.append(total_r)
    mean_r, std_r = np.mean(episode_rewards), np.std(episode_rewards)
    print(f"🏁 {algo_name} Mean Eval Reward: {mean_r:.2f} ± {std_r:.2f}")

    return {
        "algo": algo_name,
        "rewards": callback.episode_rewards,
        "losses": callback.losses,
        "lrs": callback.lrs,
        "mean_eval_reward": mean_r,
        "std_eval_reward": std_r
    }


# ======================================================
# MAIN LOOP
# ======================================================
results = []
for algo in ALGORITHMS:
    results.append(train_and_eval(algo))

# === Save comparison plots ===
plt.figure(figsize=(10, 6))
for r in results: plt.plot(r["rewards"], label=f"{r['algo']} reward")
plt.legend(); plt.title("Reward Comparison"); plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "compare_rewards.png"))

plt.figure(figsize=(10, 6))
for r in results: plt.plot(r["losses"], label=f"{r['algo']} loss")
plt.legend(); plt.title("Loss Comparison"); plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "compare_losses.png"))

plt.figure(figsize=(10, 6))
for r in results: plt.plot(r["lrs"], label=f"{r['algo']} learning rate")
plt.legend(); plt.title("Learning Rate Comparison"); plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "compare_lr.png"))

# === Summary CSV ===
df = pd.DataFrame([{
    "Algorithm": r["algo"],
    "Mean Eval Reward": r["mean_eval_reward"],
    "Std Reward": r["std_eval_reward"],
    "Episodes": len(r["rewards"]),
} for r in results])
df.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
print("📁 Saved results/summary.csv and all comparison plots!")
