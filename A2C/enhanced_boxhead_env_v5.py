"""
Enhanced Boxhead Environment V5 - Final Optimized Version
Based on V4 analysis: Maintain peak performance, reduce variance

V4 Peak Results: Mean 482.74, Completion 76.67% (episodes 1-150)
V5 Goals: Mean 480-500, Std < 120, Completion 75-80%, MAINTAIN peak

Key improvements:
- Even smoother reward signals for variance reduction
- Tighter normalization
- Optimized for consistency
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from collections import deque


class EnhancedBoxheadEnvV5(gym.Env):
    """
    V5: Final optimized environment with focus on maintaining peak performance
    
    State: 42 features (proven to work well)
    Focus: Consistency and variance reduction
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, window_width=640, window_height=480, difficulty=1.0):
        super(EnhancedBoxheadEnvV5, self).__init__()
        
        self.window_width = window_width
        self.window_height = window_height
        self.difficulty = difficulty
        
        # Optimized speeds for consistency
        self.player_speed = 2.8
        self.zombie_speed = 0.62 * difficulty  # Slightly slower for consistency
        self.demon_speed = 0.82 * difficulty
        self.enemy_spawn_rate = 0.0023 * difficulty  # Reduced for stability
        self.item_spawn_rate = 0.0035  # More items for resource availability
        self.steps_per_episode = 1000
        
        # Weapon system
        self.weapons = {
            'pistol': {'damage': 34, 'ammo_cost': 1, 'range': 195},  # Buffed for efficiency
            'shotgun': {'damage': 68, 'ammo_cost': 2, 'range': 135},
        }
        
        self._initialize_map_layout()
        
        # Same observation space as V3/V4 (42 features worked well)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(42,), dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(6)
        self.action_history = deque(maxlen=2)
        
        self.reset()

    def _initialize_map_layout(self):
        """Simple map layout"""
        self.walls = [
            {'x': 0, 'y': 0, 'width': self.window_width, 'height': 15},
            {'x': 0, 'y': self.window_height - 15, 'width': self.window_width, 'height': 15},
            {'x': 0, 'y': 0, 'width': 15, 'height': self.window_height},
            {'x': self.window_width - 15, 'y': 0, 'width': 15, 'height': self.window_height},
        ]
        
        self.safe_zones = [
            {'x': self.window_width/2, 'y': self.window_height/2, 'radius': 80},
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Position
        self.player_x = self.window_width / 2
        self.player_y = self.window_height / 2
        self.player_direction = (0, -1)
        
        # Status
        self.player_health = 100.0
        
        # Resources
        self.ammo = 75  # Start with even more
        self.current_weapon = 'pistol'
        self.has_shotgun = False
        
        # Enemies - conservative start
        self.zombies = self._spawn_enemies(2, 'zombie')
        self.demons = self._spawn_enemies(1, 'demon')
        
        # Items
        self.ammo_pickups = self._spawn_items(3, 'ammo')
        self.weapon_pickups = self._spawn_items(1, 'weapon')
        
        # Stats
        self.steps = 0
        self.total_reward = 0
        self.kills = 0
        self.damage_taken = 0
        self.shots_fired = 0
        self.shots_hit = 0
        self.ammo_collected = 0
        self.damage_dealt = 0
        
        # Action history
        self.action_history.clear()
        for _ in range(2):
            self.action_history.append(0)
        
        return self._get_obs(), {}

    def _spawn_enemies(self, count, enemy_type):
        """Spawn enemies at safe distances"""
        enemies = []
        for _ in range(count):
            attempts = 0
            while attempts < 15:
                x = np.random.randint(70, self.window_width - 70)
                y = np.random.randint(70, self.window_height - 70)
                dist = math.hypot(x - self.window_width/2, y - self.window_height/2)
                
                if dist > 150:  # Even further
                    if enemy_type == 'zombie':
                        enemies.append({
                            'x': x, 'y': y, 'health': 30, 
                            'speed': self.zombie_speed, 'type': 'zombie'
                        })
                    else:
                        enemies.append({
                            'x': x, 'y': y, 'health': 50,
                            'speed': self.demon_speed, 'type': 'demon'
                        })
                    break
                attempts += 1
        return enemies

    def _spawn_items(self, count, item_type):
        """Spawn items"""
        items = []
        for _ in range(count):
            x = np.random.randint(70, self.window_width - 70)
            y = np.random.randint(70, self.window_height - 70)
            
            if item_type == 'ammo':
                items.append({'x': x, 'y': y, 'amount': 32, 'type': 'ammo'})  # More ammo
            else:
                items.append({'x': x, 'y': y, 'weapon': 'shotgun', 'type': 'weapon'})
        return items

    def _is_inside_wall(self, x, y):
        """Check wall collision"""
        return x < 20 or x > self.window_width - 20 or y < 20 or y > self.window_height - 20

    def _get_obs(self):
        """Same observation structure as V3/V4 (42 features)"""
        obs = np.zeros(42, dtype=np.float32)
        
        # [0-7] POSITION & STATUS
        obs[0] = (self.player_x / self.window_width) * 2 - 1
        obs[1] = (self.player_y / self.window_height) * 2 - 1
        obs[2] = self.player_direction[0]
        obs[3] = self.player_direction[1]
        obs[4] = (self.player_health / 100.0) * 2 - 1
        obs[5] = np.tanh(self.kills / 5.0)
        obs[6] = (self.shots_hit / max(self.shots_fired, 1)) * 2 - 1 if self.shots_fired > 0 else 0
        obs[7] = (len(self.zombies) + len(self.demons)) / 10.0 * 2 - 1
        
        # [8-25] ENEMY INFO: 3 nearest
        all_enemies = self.zombies + self.demons
        if all_enemies:
            enemies_with_dist = []
            for e in all_enemies:
                dist = math.hypot(e['x'] - self.player_x, e['y'] - self.player_y)
                enemies_with_dist.append((e, dist))
            enemies_with_dist.sort(key=lambda x: x[1])
            
            for i in range(min(3, len(enemies_with_dist))):
                enemy, dist = enemies_with_dist[i]
                base_idx = 8 + i * 6
                max_dist = math.hypot(self.window_width, self.window_height)
                
                obs[base_idx] = (enemy['x'] - self.player_x) / self.window_width
                obs[base_idx + 1] = (enemy['y'] - self.player_y) / self.window_height
                obs[base_idx + 2] = (dist / max_dist) * 2 - 1
                obs[base_idx + 3] = 1.0 if enemy['type'] == 'zombie' else -1.0
                obs[base_idx + 4] = (enemy['health'] / 50.0) * 2 - 1
                threat = (1.0 - dist / max_dist) * (1.5 if enemy['type'] == 'demon' else 1.0)
                obs[base_idx + 5] = np.tanh(threat)
        
        # [26-33] RESOURCES & ITEMS
        obs[26] = (self.ammo / 100.0) * 2 - 1
        obs[27] = 1.0 if self.current_weapon == 'pistol' else -1.0
        obs[28] = 1.0 if self.current_weapon == 'shotgun' else -1.0
        obs[29] = 1.0 if self.has_shotgun else -1.0
        
        if self.ammo_pickups:
            nearest = min(self.ammo_pickups, 
                         key=lambda p: math.hypot(p['x'] - self.player_x, p['y'] - self.player_y))
            dist = math.hypot(nearest['x'] - self.player_x, nearest['y'] - self.player_y)
            obs[30] = (nearest['x'] - self.player_x) / self.window_width
            obs[31] = (nearest['y'] - self.player_y) / self.window_height
            obs[32] = (dist / math.hypot(self.window_width, self.window_height)) * 2 - 1
        
        obs[33] = 1.0 if self.weapon_pickups else -1.0
        
        # [34-39] MAP TACTICAL
        obs[34] = (self.player_x / self.window_width) * 2 - 1
        obs[35] = ((self.window_width - self.player_x) / self.window_width) * 2 - 1
        obs[36] = (self.player_y / self.window_height) * 2 - 1
        obs[37] = ((self.window_height - self.player_y) / self.window_height) * 2 - 1
        
        center_dist = math.hypot(self.player_x - self.window_width/2, 
                                self.player_y - self.window_height/2)
        obs[38] = (center_dist / (math.hypot(self.window_width, self.window_height) / 2)) * 2 - 1
        
        wall_danger = 1.0 if self._is_inside_wall(self.player_x + 30, self.player_y) or \
                            self._is_inside_wall(self.player_x - 30, self.player_y) or \
                            self._is_inside_wall(self.player_x, self.player_y + 30) or \
                            self._is_inside_wall(self.player_x, self.player_y - 30) else 0.0
        obs[39] = wall_danger * 2 - 1
        
        # [40-41] TEMPORAL
        for i, action in enumerate(self.action_history):
            obs[40 + i] = (action / 2.5) * 2 - 1
        
        return obs

    def step(self, action):
        self.steps += 1
        reward = 0.0
        
        self.action_history.append(action)
        
        # Movement
        old_x, old_y = self.player_x, self.player_y
        if action == 1:
            self.player_y -= self.player_speed
            self.player_direction = (0, -1)
        elif action == 2:
            self.player_y += self.player_speed
            self.player_direction = (0, 1)
        elif action == 3:
            self.player_x -= self.player_speed
            self.player_direction = (-1, 0)
        elif action == 4:
            self.player_x += self.player_speed
            self.player_direction = (1, 0)
        
        # Wall collision
        if self._is_inside_wall(self.player_x, self.player_y):
            self.player_x, self.player_y = old_x, old_y
            reward -= 0.005  # Minimal penalty
        
        self.player_x = np.clip(self.player_x, 20, self.window_width - 20)
        self.player_y = np.clip(self.player_y, 20, self.window_height - 20)
        
        # Enemy AI
        all_enemies = self.zombies + self.demons
        for enemy in all_enemies:
            dx = self.player_x - enemy['x']
            dy = self.player_y - enemy['y']
            dist = math.hypot(dx, dy)
            if dist > 0:
                dx /= dist
                dy /= dist
                enemy['x'] += dx * enemy['speed']
                enemy['y'] += dy * enemy['speed']
            
            # Collision
            if dist < 18:
                damage = 0.32 if enemy['type'] == 'zombie' else 0.52  # Further reduced
                self.player_health -= damage
                self.damage_taken += damage
                reward -= 0.35  # Further reduced penalty
        
        # Shooting
        if action == 5:
            weapon = self.weapons[self.current_weapon]
            if self.ammo >= weapon['ammo_cost']:
                self.ammo -= weapon['ammo_cost']
                self.shots_fired += 1
                
                hit_enemy = None
                min_dist = weapon['range']
                
                for enemy in all_enemies:
                    dist = math.hypot(enemy['x'] - self.player_x, enemy['y'] - self.player_y)
                    if dist > 0:
                        to_enemy = ((enemy['x'] - self.player_x) / dist, 
                                   (enemy['y'] - self.player_y) / dist)
                        dot = to_enemy[0] * self.player_direction[0] + to_enemy[1] * self.player_direction[1]
                        
                        if dot > 0.5 and dist < min_dist:  # Easier to hit
                            min_dist = dist
                            hit_enemy = enemy
                
                if hit_enemy:
                    damage = weapon['damage']
                    hit_enemy['health'] -= damage
                    self.damage_dealt += damage
                    self.shots_hit += 1
                    reward += 0.5  # Consistent, moderate hit reward
                    
                    if hit_enemy['health'] <= 0:
                        kill_reward = 4.5 if hit_enemy['type'] == 'demon' else 3.0  # Moderate kills
                        reward += kill_reward
                        self.kills += 1
                        if hit_enemy in self.zombies:
                            self.zombies.remove(hit_enemy)
                        else:
                            self.demons.remove(hit_enemy)
                else:
                    reward -= 0.005  # Minimal miss penalty
            else:
                reward -= 0.02
        
        # Item pickups
        for ammo in self.ammo_pickups[:]:
            if math.hypot(ammo['x'] - self.player_x, ammo['y'] - self.player_y) < 25:
                self.ammo = min(100, self.ammo + ammo['amount'])
                reward += 1.8  # Moderate reward
                self.ammo_collected += 1
                self.ammo_pickups.remove(ammo)
        
        for weapon in self.weapon_pickups[:]:
            if math.hypot(weapon['x'] - self.player_x, weapon['y'] - self.player_y) < 25:
                if not self.has_shotgun:
                    self.has_shotgun = True
                    self.current_weapon = 'shotgun'
                    reward += 3.0
                self.weapon_pickups.remove(weapon)
        
        # Spawn
        if np.random.random() < self.enemy_spawn_rate:
            if np.random.random() < 0.75:
                self.zombies.extend(self._spawn_enemies(1, 'zombie'))
            else:
                self.demons.extend(self._spawn_enemies(1, 'demon'))
        
        if np.random.random() < self.item_spawn_rate:
            if np.random.random() < 0.78:
                self.ammo_pickups.extend(self._spawn_items(1, 'ammo'))
            else:
                if not self.has_shotgun:
                    self.weapon_pickups.extend(self._spawn_items(1, 'weapon'))
        
        # === ULTRA-SMOOTH REWARD SHAPING FOR VARIANCE REDUCTION ===
        # Base survival
        reward += 0.16
        
        # Health reward (very smooth)
        health_ratio = self.player_health / 100.0
        reward += 0.28 * health_ratio
        
        # Bonus for high health (encourage survival)
        if health_ratio > 0.85:
            reward += 0.12
        elif health_ratio > 0.7:
            reward += 0.06
        
        # Ultra-smooth distance management
        if all_enemies:
            nearest_dist = min(math.hypot(e['x'] - self.player_x, e['y'] - self.player_y) 
                             for e in all_enemies)
            
            # Very smooth distance reward
            if 115 <= nearest_dist <= 195:
                reward += 0.42  # Optimal zone
            elif 95 <= nearest_dist < 115:
                reward += 0.28  # Good zone
            elif 195 < nearest_dist <= 230:
                reward += 0.28  # Good zone
            elif 75 <= nearest_dist < 95:
                reward += 0.14  # Acceptable
            elif nearest_dist < 65:
                reward -= 0.10  # Small penalty
            else:
                reward += 0.05  # Baseline
        
        # Smooth ammo management
        ammo_ratio = self.ammo / 100.0
        if ammo_ratio > 0.5:
            reward += 0.08
        elif ammo_ratio > 0.3:
            reward += 0.04
        elif ammo_ratio < 0.12:
            reward -= 0.06
        
        # Movement reward
        movement = math.hypot(self.player_x - old_x, self.player_y - old_y)
        if movement > 0:
            reward += 0.02
        
        # Health decay (minimal)
        self.player_health -= 0.035
        
        # Terminal
        terminated = self.player_health <= 0
        truncated = self.steps >= self.steps_per_episode
        done = terminated or truncated
        
        if terminated:
            reward -= 22  # Further reduced death penalty
        
        if truncated:
            # Big completion bonus
            reward += 40  # Increased
            # Performance bonuses
            reward += self.kills * 2.8
            reward += (self.player_health / 100.0) * 12  # Health bonus
            reward += (self.shots_hit / max(self.shots_fired, 1)) * 8  # Accuracy bonus
        
        self.total_reward += reward
        
        info = {}
        if done:
            info["episode"] = {
                "r": self.total_reward,
                "l": self.steps,
                "kills": self.kills,
                "damage_taken": self.damage_taken,
                "damage_dealt": self.damage_dealt,
                "accuracy": self.shots_hit / max(self.shots_fired, 1),
                "ammo_collected": self.ammo_collected,
                "health_remaining": self.player_health
            }
        
        return self._get_obs(), reward, done, truncated, info

    def render(self):
        pass

    def close(self):
        pass

