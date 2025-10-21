"""
Enhanced Boxhead Environment V3 - Optimized based on V2 learnings
Reduced complexity, better reward shaping, improved state representation

Key improvements from V2:
- Reduced from 60 to 42 features (removed redundancy)
- Simplified to 3 enemy tracking (vs 5)
- Better reward structure (less negative spirals)
- Clearer state representation
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from collections import deque


class EnhancedBoxheadEnvV3(gym.Env):
    """
    Optimized environment based on V2 analysis
    
    State: 42 features (optimized from 60)
    - Position & Status: 8 features
    - Enemy Info (3 enemies): 18 features  
    - Resources & Items: 8 features
    - Map Tactical Info: 6 features
    - Temporal: 2 features (reduced from 4)
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, window_width=640, window_height=480):
        super(EnhancedBoxheadEnvV3, self).__init__()
        
        self.window_width = window_width
        self.window_height = window_height
        self.player_speed = 2.5  # Slightly increased for better maneuverability
        self.zombie_speed = 0.7
        self.demon_speed = 0.9
        self.enemy_spawn_rate = 0.003  # Slightly reduced
        self.item_spawn_rate = 0.0025
        self.steps_per_episode = 1000
        
        # Simplified weapon system
        self.weapons = {
            'pistol': {'damage': 30, 'ammo_cost': 1, 'range': 180},  # Buffed
            'shotgun': {'damage': 60, 'ammo_cost': 2, 'range': 120},  # Buffed
        }
        
        # Simplified map layout
        self._initialize_map_layout()
        
        # Optimized observation space: 42 features
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(42,), dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(6)
        
        # Reduced action history
        self.action_history = deque(maxlen=2)
        
        self.reset()

    def _initialize_map_layout(self):
        """Simplified map with essential features"""
        # Simpler wall structure
        self.walls = [
            # Outer boundaries only
            {'x': 0, 'y': 0, 'width': self.window_width, 'height': 15},
            {'x': 0, 'y': self.window_height - 15, 'width': self.window_width, 'height': 15},
            {'x': 0, 'y': 0, 'width': 15, 'height': self.window_height},
            {'x': self.window_width - 15, 'y': 0, 'width': 15, 'height': self.window_height},
        ]
        
        # Key tactical zones
        self.safe_zones = [
            {'x': self.window_width/2, 'y': self.window_height/2, 'radius': 80},  # Center
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Position
        self.player_x = self.window_width / 2
        self.player_y = self.window_height / 2
        self.player_direction = (0, -1)
        
        # Status
        self.player_health = 100.0
        
        # Resources - simplified
        self.ammo = 60  # Start with more ammo
        self.current_weapon = 'pistol'
        self.has_shotgun = False
        
        # Enemies - reduced to 3 starting
        self.zombies = self._spawn_enemies(3, 'zombie')
        self.demons = self._spawn_enemies(1, 'demon')
        
        # Items
        self.ammo_pickups = self._spawn_items(2, 'ammo')
        self.weapon_pickups = self._spawn_items(1, 'weapon')
        
        # Stats
        self.steps = 0
        self.total_reward = 0
        self.kills = 0
        self.damage_taken = 0
        self.shots_fired = 0
        self.shots_hit = 0
        self.ammo_collected = 0
        
        # Action history
        self.action_history.clear()
        for _ in range(2):
            self.action_history.append(0)
        
        return self._get_obs(), {}

    def _spawn_enemies(self, count, enemy_type):
        """Spawn enemies at random safe positions"""
        enemies = []
        for _ in range(count):
            attempts = 0
            while attempts < 15:
                x = np.random.randint(60, self.window_width - 60)
                y = np.random.randint(60, self.window_height - 60)
                dist = math.hypot(x - self.window_width/2, y - self.window_height/2)
                
                if dist > 120:  # Further from player
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
            x = np.random.randint(60, self.window_width - 60)
            y = np.random.randint(60, self.window_height - 60)
            
            if item_type == 'ammo':
                items.append({'x': x, 'y': y, 'amount': 25, 'type': 'ammo'})
            else:
                items.append({'x': x, 'y': y, 'weapon': 'shotgun', 'type': 'weapon'})
        return items

    def _is_inside_wall(self, x, y):
        """Check wall collision"""
        return x < 20 or x > self.window_width - 20 or y < 20 or y > self.window_height - 20

    def _get_obs(self):
        """
        Optimized state (42 features):
        
        [0-7]   POSITION & STATUS: Location, health, combat stats
        [8-25]  ENEMY INFO: 3 nearest enemies (6 features each)
        [26-33] RESOURCES & ITEMS: Ammo, weapons, pickups
        [34-39] MAP TACTICAL: Walls, safe zones, positioning
        [40-41] TEMPORAL: Last 2 actions
        """
        obs = np.zeros(42, dtype=np.float32)
        
        # [0-7] POSITION & STATUS
        obs[0] = (self.player_x / self.window_width) * 2 - 1
        obs[1] = (self.player_y / self.window_height) * 2 - 1
        obs[2] = self.player_direction[0]
        obs[3] = self.player_direction[1]
        obs[4] = (self.player_health / 100.0) * 2 - 1
        obs[5] = np.tanh(self.kills / 5.0)  # Normalized kills
        obs[6] = (self.shots_hit / max(self.shots_fired, 1)) * 2 - 1 if self.shots_fired > 0 else 0
        obs[7] = (len(self.zombies) + len(self.demons)) / 10.0 * 2 - 1  # Enemy count
        
        # [8-25] ENEMY INFO: 3 nearest enemies
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
                # Threat level based on distance and type
                threat = (1.0 - dist / max_dist) * (1.5 if enemy['type'] == 'demon' else 1.0)
                obs[base_idx + 5] = np.tanh(threat)
        
        # [26-33] RESOURCES & ITEMS
        obs[26] = (self.ammo / 100.0) * 2 - 1
        obs[27] = 1.0 if self.current_weapon == 'pistol' else -1.0
        obs[28] = 1.0 if self.current_weapon == 'shotgun' else -1.0
        obs[29] = 1.0 if self.has_shotgun else -1.0
        
        # Nearest ammo
        if self.ammo_pickups:
            nearest = min(self.ammo_pickups, 
                         key=lambda p: math.hypot(p['x'] - self.player_x, p['y'] - self.player_y))
            dist = math.hypot(nearest['x'] - self.player_x, nearest['y'] - self.player_y)
            obs[30] = (nearest['x'] - self.player_x) / self.window_width
            obs[31] = (nearest['y'] - self.player_y) / self.window_height
            obs[32] = (dist / math.hypot(self.window_width, self.window_height)) * 2 - 1
        
        # Has weapon pickup nearby
        obs[33] = 1.0 if self.weapon_pickups else -1.0
        
        # [34-39] MAP TACTICAL
        obs[34] = (self.player_x / self.window_width) * 2 - 1  # Left wall
        obs[35] = ((self.window_width - self.player_x) / self.window_width) * 2 - 1  # Right
        obs[36] = (self.player_y / self.window_height) * 2 - 1  # Top
        obs[37] = ((self.window_height - self.player_y) / self.window_height) * 2 - 1  # Bottom
        
        # Distance to center (safe zone)
        center_dist = math.hypot(self.player_x - self.window_width/2, 
                                self.player_y - self.window_height/2)
        obs[38] = (center_dist / (math.hypot(self.window_width, self.window_height) / 2)) * 2 - 1
        
        # Danger level (proximity to walls and enemies)
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
            reward -= 0.02  # Reduced penalty
        
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
                damage = 0.4 if enemy['type'] == 'zombie' else 0.6  # Reduced damage
                self.player_health -= damage
                self.damage_taken += damage
                reward -= 0.5  # Reduced penalty
        
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
                        
                        if dot > 0.6 and dist < min_dist:  # Easier to hit
                            min_dist = dist
                            hit_enemy = enemy
                
                if hit_enemy:
                    hit_enemy['health'] -= weapon['damage']
                    self.shots_hit += 1
                    reward += 0.8  # Reduced hit reward
                    
                    if hit_enemy['health'] <= 0:
                        kill_reward = 6.0 if hit_enemy['type'] == 'demon' else 4.0  # Reduced
                        reward += kill_reward
                        self.kills += 1
                        if hit_enemy in self.zombies:
                            self.zombies.remove(hit_enemy)
                        else:
                            self.demons.remove(hit_enemy)
                else:
                    reward -= 0.02  # Small miss penalty
            else:
                reward -= 0.05  # No ammo
        
        # Item pickups
        for ammo in self.ammo_pickups[:]:
            if math.hypot(ammo['x'] - self.player_x, ammo['y'] - self.player_y) < 25:
                self.ammo = min(100, self.ammo + ammo['amount'])
                reward += 2.5
                self.ammo_collected += 1
                self.ammo_pickups.remove(ammo)
        
        for weapon in self.weapon_pickups[:]:
            if math.hypot(weapon['x'] - self.player_x, weapon['y'] - self.player_y) < 25:
                if not self.has_shotgun:
                    self.has_shotgun = True
                    self.current_weapon = 'shotgun'
                    reward += 4.0
                self.weapon_pickups.remove(weapon)
        
        # Spawn
        if np.random.random() < self.enemy_spawn_rate:
            if np.random.random() < 0.75:
                self.zombies.extend(self._spawn_enemies(1, 'zombie'))
            else:
                self.demons.extend(self._spawn_enemies(1, 'demon'))
        
        if np.random.random() < self.item_spawn_rate:
            if np.random.random() < 0.7:
                self.ammo_pickups.extend(self._spawn_items(1, 'ammo'))
            else:
                if not self.has_shotgun:  # Only spawn if not collected
                    self.weapon_pickups.extend(self._spawn_items(1, 'weapon'))
        
        # === IMPROVED REWARD SHAPING ===
        # Base survival
        reward += 0.12
        
        # Health reward (smoother)
        reward += 0.18 * (self.player_health / 100.0)
        
        # Distance management
        if all_enemies:
            nearest_dist = min(math.hypot(e['x'] - self.player_x, e['y'] - self.player_y) 
                             for e in all_enemies)
            # Reward optimal distance (100-180 pixels)
            if 100 <= nearest_dist <= 180:
                reward += 0.35
            elif nearest_dist < 60:
                reward -= 0.15  # Reduced penalty
            else:
                reward += 0.08
        
        # Ammo incentive (smoother)
        if self.ammo > 30:
            reward += 0.08
        elif self.ammo < 10:
            reward -= 0.1
        
        # Movement reward
        movement = math.hypot(self.player_x - old_x, self.player_y - old_y)
        if movement > 0:
            reward += 0.03
        
        # Health decay (reduced)
        self.player_health -= 0.05
        
        # Terminal
        terminated = self.player_health <= 0
        truncated = self.steps >= self.steps_per_episode
        done = terminated or truncated
        
        if terminated:
            reward -= 30  # Reduced death penalty
        
        if truncated:
            reward += 30  # Increased completion bonus
            reward += self.kills * 2  # Bonus for kills
        
        self.total_reward += reward
        
        info = {}
        if done:
            info["episode"] = {
                "r": self.total_reward,
                "l": self.steps,
                "kills": self.kills,
                "damage_taken": self.damage_taken,
                "accuracy": self.shots_hit / max(self.shots_fired, 1),
                "ammo_collected": self.ammo_collected
            }
        
        return self._get_obs(), reward, done, truncated, info

    def render(self):
        pass

    def close(self):
        pass

