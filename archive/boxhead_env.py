import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import random
import pygame
import time

# ========== CONFIGURATION ==========
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480
FPS = 60

PLAYER_SPEED = 4.0
ZOMBIE_SPEED = 1.2
DEMON_SPEED = 1.4
ENEMY_SPAWN_RATE = 0.003
SPAWN_MIN_DISTANCE = 50

# Reward tuning
REWARD_STEP = 0.05
REWARD_KILL_ZOMBIE = 3.0
REWARD_KILL_DEMON = 5.0
REWARD_PICKUP = 2.0
PENALTY_HIT = -2.0
PENALTY_DEATH = -10.0

MAX_STEPS = 1000  # per episode

# Colors (only needed for visualization)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
PURPLE = (180, 0, 180)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


# ===== Helper functions =====
def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def random_spawn_position():
    """Spawn outside 50px radius from center."""
    while True:
        x, y = random.randint(0, WINDOW_WIDTH), random.randint(0, WINDOW_HEIGHT)
        if distance((x, y), (WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2)) > SPAWN_MIN_DISTANCE:
            return x, y


# ===== Entity classes =====
class Player:
    def __init__(self):
        self.x, self.y = WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2
        self.radius = 12
        self.health = 100
        self.direction = (0, -1)
        self.last_shot = 0
        self.shot_delay = 0.25

    def move(self, action):
        dx, dy = 0, 0
        if action == 1:  # Up
            dy = -PLAYER_SPEED
            self.direction = (0, -1)
        elif action == 2:  # Down
            dy = PLAYER_SPEED
            self.direction = (0, 1)
        elif action == 3:  # Left
            dx = -PLAYER_SPEED
            self.direction = (-1, 0)
        elif action == 4:  # Right
            dx = PLAYER_SPEED
            self.direction = (1, 0)
        self.x = np.clip(self.x + dx, 0, WINDOW_WIDTH)
        self.y = np.clip(self.y + dy, 0, WINDOW_HEIGHT)

    def shoot(self, bullets):
        now = time.time()
        if now - self.last_shot >= self.shot_delay:
            dx, dy = self.direction
            bullets.append(Bullet(self.x + dx * 15, self.y + dy * 15, 8, self.direction, "player"))
            self.last_shot = now


class Bullet:
    def __init__(self, x, y, speed, direction, owner):
        self.x, self.y = x, y
        self.speed = speed
        self.direction = direction
        self.owner = owner
        self.radius = 4

    def update(self):
        self.x += self.direction[0] * self.speed
        self.y += self.direction[1] * self.speed


class Enemy:
    def __init__(self, x, y, color, speed, radius=10, health=30):
        self.x, self.y = x, y
        self.color = color
        self.speed = speed
        self.radius = radius
        self.health = health

    def move_toward(self, tx, ty):
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        if dist > 0:
            dx, dy = dx / dist, dy / dist
            self.x += dx * self.speed
            self.y += dy * self.speed


class Demon(Enemy):
    def __init__(self, x, y):
        super().__init__(x, y, PURPLE, DEMON_SPEED, radius=14, health=50)
        self.last_shot = time.time()

    def shoot(self, bullets, player):
        now = time.time()
        if now - self.last_shot >= 5:
            for _ in range(3):
                dx, dy = player.x - self.x, player.y - self.y
                dist = math.hypot(dx, dy)
                if dist == 0:
                    continue
                dx, dy = dx / dist, dy / dist
                spread = random.uniform(-0.2, 0.2)
                bullets.append(Bullet(self.x, self.y, 5, (dx + spread, dy + spread), "demon"))
            self.last_shot = now


class Package:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.radius = 8


# ===== RL Environment =====
class BoxheadEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        super(BoxheadEnv, self).__init__()

        self.render_mode = render_mode
        self.window = None

        # Observation: position, health, nearest enemy/package info
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)  # 0:idle, 1:up, 2:down, 3:left, 4:right, 5:shoot

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player = Player()
        self.zombies = [Enemy(*random_spawn_position(), RED, ZOMBIE_SPEED) for _ in range(3)]
        self.demons = [Demon(*random_spawn_position()) for _ in range(1)]
        self.packages = [Package(*random_spawn_position()) for _ in range(1)]
        self.bullets = []
        self.step_count = 0
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        # nearest zombie, demon, package distances
        def nearest(entities):
            if not entities:
                return (0, 0)
            nearest_e = min(entities, key=lambda e: distance((self.player.x, self.player.y), (e.x, e.y)))
            return (
                (nearest_e.x - self.player.x) / WINDOW_WIDTH,
                (nearest_e.y - self.player.y) / WINDOW_HEIGHT,
            )

        z_dx, z_dy = nearest(self.zombies)
        d_dx, d_dy = nearest(self.demons)
        p_dx, p_dy = nearest(self.packages)

        obs = np.array(
            [
                self.player.x / WINDOW_WIDTH,
                self.player.y / WINDOW_HEIGHT,
                self.player.health / 100,
                z_dx,
                z_dy,
                d_dx,
                d_dy,
                p_dx,
                p_dy,
            ],
            dtype=np.float32,
        )
        return obs

    def step(self, action):
        reward = 0
        self.step_count += 1
        self.player.move(action)
        if action == 5:
            self.player.shoot(self.bullets)

        # Update bullets
        for b in self.bullets[:]:
            b.update()
            if not (0 <= b.x <= WINDOW_WIDTH and 0 <= b.y <= WINDOW_HEIGHT):
                self.bullets.remove(b)
                continue
            if b.owner == "player":
                for e in self.zombies + self.demons:
                    if distance((b.x, b.y), (e.x, e.y)) <= e.radius:
                        e.health -= 25
                        reward += REWARD_KILL_DEMON if isinstance(e, Demon) else REWARD_KILL_ZOMBIE
                        self.bullets.remove(b)
                        break
            elif b.owner == "demon":
                if distance((b.x, b.y), (self.player.x, self.player.y)) <= self.player.radius:
                    self.player.health -= 15
                    reward += PENALTY_HIT
                    self.bullets.remove(b)

        # Enemy movement
        for z in self.zombies[:]:
            z.move_toward(self.player.x, self.player.y)
            if z.health <= 0:
                self.zombies.remove(z)
            elif distance((z.x, z.y), (self.player.x, self.player.y)) <= z.radius + self.player.radius:
                self.player.health -= 0.4
                reward += PENALTY_HIT

        for d in self.demons[:]:
            d.move_toward(self.player.x, self.player.y)
            d.shoot(self.bullets, self.player)
            if d.health <= 0:
                self.demons.remove(d)
            elif distance((d.x, d.y), (self.player.x, self.player.y)) <= d.radius + self.player.radius:
                self.player.health -= 0.6
                reward += PENALTY_HIT

        # Pickup check
        for p in self.packages[:]:
            if distance((p.x, p.y), (self.player.x, self.player.y)) <= p.radius + self.player.radius:
                reward += REWARD_PICKUP
                self.player.health = 100
                self.packages.remove(p)

        # Spawn new enemies occasionally
        if random.random() < ENEMY_SPAWN_RATE:
            if random.random() < 0.8:
                self.zombies.append(Enemy(*random_spawn_position(), RED, ZOMBIE_SPEED))
            else:
                self.demons.append(Demon(*random_spawn_position()))

        reward += REWARD_STEP
        terminated = self.player.health <= 0
        truncated = self.step_count >= MAX_STEPS
        done = terminated or truncated

        if terminated:
            reward += PENALTY_DEATH

        obs = self._get_obs()
        info = {}
        return obs, reward, done, False, info

    def render(self):
        pass  # disabled for speed

    def close(self):
        pygame.quit()
