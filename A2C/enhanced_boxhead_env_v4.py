"""
Enhanced Boxhead Environment V4 - Optimized for Consistency
Based on V3 analysis: Focus on reducing variance and improving stability

V3 Results: Mean 310.93 ± 151.86, Completion 50.66%
V4 Goals: Mean 400+, Std < 100, Completion 60-70%

Key improvements:
- Smoother reward signals
- Better normalization
- More stable dynamics
- Curriculum-ready structure
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from collections import deque


class EnhancedBoxheadEnvV4(gym.Env):
    """
    V4: Optimized for consistency and higher performance
    
    State: 42 features (same as V3 - worked well)
    Focus: Stability over complexity
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, window_width=640, window_height=480, difficulty=1.0):
        super(EnhancedBoxheadEnvV4, self).__init__()
        
        self.window_width = window_width
        self.window_height = window_height
        self.difficulty = difficulty  # For curriculum learning
        
        # Balanced speeds
        self.player_speed = 2.8  # Slightly increased from 2.5
        self.zombie_speed = 0.65 * difficulty  # Slightly reduced
        self.demon_speed = 0.85 * difficulty
        self.enemy_spawn_rate = 0.0025 * difficulty
        self.item_spawn_rate = 0.003  # Increased
        self.steps_per_episode = 1000
        
        # Weapon system
        self.weapons = {
            'pistol': {'damage': 32, 'ammo_cost': 1, 'range': 190},  # Buffed slightly
            'shotgun': {'damage': 65, 'ammo_cost': 2, 'range': 130},
        }
        
        self._initialize_map_layout()
        
        # Same observation space as V3 (42 features worked well)
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
        self.ammo = 70  # Start with more
        self.current_weapon = 'pistol'
        self.has_shotgun = False
        
        # Enemies - balanced start
        self.zombies = self._spawn_enemies(2, 'zombie')  # Reduced from 3
        self.demons = self._spawn_enemies(1, 'demon')
        
        # Items
        self.ammo_pickups = self._spawn_items(3, 'ammo')  # Increased
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
                
                if dist > 140:  # Even further from player
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
                items.append({'x': x, 'y': y, 'amount': 30, 'type': 'ammo'})  # More ammo
            else:
                items.append({'x': x, 'y': y, 'weapon': 'shotgun', 'type': 'weapon'})
        return items

    def _is_inside_wall(self, x, y):
        """Check wall collision"""
        return x < 20 or x > self.window_width - 20 or y < 20 or y > self.window_height - 20

    def _get_obs(self):
        """Same observation structure as V3 (42 features)"""
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
            reward -= 0.01  # Minimal penalty
        
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
                damage = 0.35 if enemy['type'] == 'zombie' else 0.55  # Slightly reduced
                self.player_health -= damage
                self.damage_taken += damage
                reward -= 0.4  # Reduced penalty
        
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
                        
                        if dot > 0.55 and dist < min_dist:  # Easier to hit
                            min_dist = dist
                            hit_enemy = enemy
                
                if hit_enemy:
                    damage = weapon['damage']
                    hit_enemy['health'] -= damage
                    self.damage_dealt += damage
                    self.shots_hit += 1
                    reward += 0.6  # Consistent hit reward
                    
                    if hit_enemy['health'] <= 0:
                        kill_reward = 5.0 if hit_enemy['type'] == 'demon' else 3.5  # Slightly reduced
                        reward += kill_reward
                        self.kills += 1
                        if hit_enemy in self.zombies:
                            self.zombies.remove(hit_enemy)
                        else:
                            self.demons.remove(hit_enemy)
                else:
                    reward -= 0.01  # Minimal miss penalty
            else:
                reward -= 0.03
        
        # Item pickups
        for ammo in self.ammo_pickups[:]:
            if math.hypot(ammo['x'] - self.player_x, ammo['y'] - self.player_y) < 25:
                self.ammo = min(100, self.ammo + ammo['amount'])
                reward += 2.0  # Reduced but still positive
                self.ammo_collected += 1
                self.ammo_pickups.remove(ammo)
        
        for weapon in self.weapon_pickups[:]:
            if math.hypot(weapon['x'] - self.player_x, weapon['y'] - self.player_y) < 25:
                if not self.has_shotgun:
                    self.has_shotgun = True
                    self.current_weapon = 'shotgun'
                    reward += 3.5
                self.weapon_pickups.remove(weapon)
        
        # Spawn
        if np.random.random() < self.enemy_spawn_rate:
            if np.random.random() < 0.75:
                self.zombies.extend(self._spawn_enemies(1, 'zombie'))
            else:
                self.demons.extend(self._spawn_enemies(1, 'demon'))
        
        if np.random.random() < self.item_spawn_rate:
            if np.random.random() < 0.75:
                self.ammo_pickups.extend(self._spawn_items(1, 'ammo'))
            else:
                if not self.has_shotgun:
                    self.weapon_pickups.extend(self._spawn_items(1, 'weapon'))
        
        # === SMOOTHED REWARD SHAPING ===
        # Base survival
        reward += 0.14
        
        # Health reward (smoother, more important)
        health_ratio = self.player_health / 100.0
        reward += 0.25 * health_ratio
        
        # Bonus for high health
        if health_ratio > 0.8:
            reward += 0.1
        
        # Distance management (smoother)
        if all_enemies:
            nearest_dist = min(math.hypot(e['x'] - self.player_x, e['y'] - self.player_y) 
                             for e in all_enemies)
            
            # Smooth distance reward
            if 110 <= nearest_dist <= 190:
                reward += 0.4
            elif 90 <= nearest_dist < 110 or 190 < nearest_dist <= 220:
                reward += 0.2  # Partial reward
            elif nearest_dist < 70:
                reward -= 0.12  # Smaller penalty
            else:
                reward += 0.06
        
        # Ammo management (smoother)
        ammo_ratio = self.ammo / 100.0
        if ammo_ratio > 0.4:
            reward += 0.06
        elif ammo_ratio < 0.15:
            reward -= 0.08
        
        # Movement reward (encourage active play)
        movement = math.hypot(self.player_x - old_x, self.player_y - old_y)
        if movement > 0:
            reward += 0.02
        
        # Health decay (reduced)
        self.player_health -= 0.04
        
        # Terminal
        terminated = self.player_health <= 0
        truncated = self.steps >= self.steps_per_episode
        done = terminated or truncated
        
        if terminated:
            reward -= 25  # Further reduced death penalty
        
        if truncated:
            # Big completion bonus
            reward += 35
            # Bonus for performance
            reward += self.kills * 2.5
            reward += (self.player_health / 100.0) * 10  # Health bonus
        
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

