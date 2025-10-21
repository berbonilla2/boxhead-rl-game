"""
Enhanced Boxhead Environment V2 with Comprehensive State Representation
Includes: Position, Enemy Info, Agent Status, Resources, Items, Map Layout
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from collections import deque


class EnhancedBoxheadEnvV2(gym.Env):
    """
    Enhanced version of Boxhead environment with comprehensive state representation:
    
    States included:
    • Position: Current coordinates of the agent on the map
    • Enemy Information: Positions and health of nearby zombies/demons
    • Agent Status: Current health level
    • Resources: Ammo count and currently equipped weapon
    • Items: Locations of nearby pickups (ammo, weapons)
    • Map Layout: Static features such as walls, chokepoints, and open areas
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, window_width=640, window_height=480):
        super(EnhancedBoxheadEnvV2, self).__init__()
        
        self.window_width = window_width
        self.window_height = window_height
        self.player_speed = 2.0
        self.zombie_speed = 0.7
        self.demon_speed = 0.9
        self.enemy_spawn_rate = 0.004
        self.item_spawn_rate = 0.002
        self.steps_per_episode = 1000
        
        # Weapon system
        self.weapons = {
            'pistol': {'damage': 25, 'ammo_cost': 1, 'range': 150},
            'shotgun': {'damage': 50, 'ammo_cost': 2, 'range': 100},
            'rifle': {'damage': 35, 'ammo_cost': 1, 'range': 200}
        }
        
        # Map layout: Define static features (walls, chokepoints, open areas)
        self._initialize_map_layout()
        
        # Enhanced observation space with comprehensive state representation
        # Total features: 50+
        # Position(3) + Enemy Info(30) + Agent Status(4) + Resources(5) + 
        # Items(6) + Map Features(8) + Action History(4) = 60 features
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(60,), dtype=np.float32
        )
        
        # 6 discrete actions
        self.action_space = spaces.Discrete(6)
        
        # Action history buffer
        self.action_history = deque(maxlen=4)
        
        self.reset()

    def _initialize_map_layout(self):
        """
        Initialize static map features:
        - Walls (boundaries)
        - Chokepoints (narrow passages)
        - Open areas (safe zones)
        """
        # Define wall positions (as rectangles: x, y, width, height)
        self.walls = [
            # Outer boundaries
            {'x': 0, 'y': 0, 'width': self.window_width, 'height': 20},  # Top
            {'x': 0, 'y': self.window_height - 20, 'width': self.window_width, 'height': 20},  # Bottom
            {'x': 0, 'y': 0, 'width': 20, 'height': self.window_height},  # Left
            {'x': self.window_width - 20, 'y': 0, 'width': 20, 'height': self.window_height},  # Right
            
            # Internal obstacles (create chokepoints)
            {'x': 200, 'y': 150, 'width': 40, 'height': 180},  # Left pillar
            {'x': 400, 'y': 150, 'width': 40, 'height': 180},  # Right pillar
        ]
        
        # Define chokepoints (narrow passages between obstacles)
        self.chokepoints = [
            {'x': 240, 'y': 240, 'radius': 50},  # Center passage
            {'x': 120, 'y': 240, 'radius': 40},  # Left passage
            {'x': 520, 'y': 240, 'radius': 40},  # Right passage
        ]
        
        # Define open/safe areas (larger spaces)
        self.open_areas = [
            {'x': 320, 'y': 80, 'radius': 60},   # Top center
            {'x': 320, 'y': 400, 'radius': 60},  # Bottom center
            {'x': 100, 'y': 240, 'radius': 50},  # Left
            {'x': 540, 'y': 240, 'radius': 50},  # Right
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # === POSITION: Agent coordinates ===
        self.player_x = self.window_width / 2
        self.player_y = self.window_height / 2
        self.player_direction = (0, -1)
        
        # === AGENT STATUS: Health ===
        self.player_health = 100.0
        
        # === RESOURCES: Ammo and weapon ===
        self.ammo = 50
        self.current_weapon = 'pistol'  # Start with pistol
        self.available_weapons = ['pistol']  # Unlocked weapons
        
        # === ENEMY INFORMATION: Nearby zombies and demons ===
        self.zombies = self._spawn_enemies(5, 'zombie')
        self.demons = self._spawn_enemies(2, 'demon')
        
        # === ITEMS: Pickup locations ===
        self.ammo_pickups = self._spawn_items(3, 'ammo')
        self.weapon_pickups = self._spawn_items(2, 'weapon')
        
        # Game state
        self.steps = 0
        self.total_reward = 0
        self.kills = 0
        self.damage_taken = 0
        self.shots_fired = 0
        self.shots_hit = 0
        
        # Clear action history
        self.action_history.clear()
        for _ in range(4):
            self.action_history.append(0)
        
        return self._get_obs(), {}

    def _spawn_enemies(self, count, enemy_type):
        """Spawn enemies at random positions away from center"""
        enemies = []
        for _ in range(count):
            attempts = 0
            while attempts < 20:  # Max attempts to find valid spawn
                x = np.random.randint(50, self.window_width - 50)
                y = np.random.randint(50, self.window_height - 50)
                dist = math.hypot(x - self.window_width/2, y - self.window_height/2)
                
                # Check not too close to player and not inside walls
                if dist > 100 and not self._is_inside_wall(x, y):
                    if enemy_type == 'zombie':
                        enemies.append({
                            'x': x, 'y': y, 'health': 30, 
                            'speed': self.zombie_speed, 'type': 'zombie'
                        })
                    else:  # demon
                        enemies.append({
                            'x': x, 'y': y, 'health': 50,
                            'speed': self.demon_speed, 'type': 'demon'
                        })
                    break
                attempts += 1
        return enemies

    def _spawn_items(self, count, item_type):
        """Spawn items at random positions"""
        items = []
        for _ in range(count):
            attempts = 0
            while attempts < 20:
                x = np.random.randint(50, self.window_width - 50)
                y = np.random.randint(50, self.window_height - 50)
                
                if not self._is_inside_wall(x, y):
                    if item_type == 'ammo':
                        items.append({'x': x, 'y': y, 'amount': 20, 'type': 'ammo'})
                    else:  # weapon
                        weapon_types = ['shotgun', 'rifle']
                        items.append({'x': x, 'y': y, 'weapon': np.random.choice(weapon_types), 'type': 'weapon'})
                    break
                attempts += 1
        return items

    def _is_inside_wall(self, x, y):
        """Check if position is inside a wall"""
        for wall in self.walls:
            if (wall['x'] <= x <= wall['x'] + wall['width'] and
                wall['y'] <= y <= wall['y'] + wall['height']):
                return True
        return False

    def _get_nearest_chokepoint_distance(self, x, y):
        """Get distance to nearest chokepoint"""
        if not self.chokepoints:
            return 1.0
        min_dist = min(math.hypot(cp['x'] - x, cp['y'] - y) for cp in self.chokepoints)
        return min_dist / math.hypot(self.window_width, self.window_height)

    def _get_nearest_open_area_distance(self, x, y):
        """Get distance to nearest open/safe area"""
        if not self.open_areas:
            return 1.0
        min_dist = min(math.hypot(oa['x'] - x, oa['y'] - y) for oa in self.open_areas)
        return min_dist / math.hypot(self.window_width, self.window_height)

    def _get_obs(self):
        """
        Comprehensive state representation (60 features):
        
        [0-2]   POSITION: Agent coordinates
        [3-32]  ENEMY INFORMATION: Nearby zombies and demons (top 5 enemies)
        [33-36] AGENT STATUS: Health and damage info
        [37-41] RESOURCES: Ammo, weapon info
        [42-47] ITEMS: Nearby pickups
        [48-55] MAP LAYOUT: Walls, chokepoints, open areas
        [56-59] ACTION HISTORY: Last 4 actions
        """
        obs = np.zeros(60, dtype=np.float32)
        
        # ========== [0-2] POSITION: Agent coordinates ==========
        obs[0] = (self.player_x / self.window_width) * 2 - 1  # [-1, 1]
        obs[1] = (self.player_y / self.window_height) * 2 - 1
        obs[2] = self.player_direction[0]  # Current facing direction
        
        # ========== [3-32] ENEMY INFORMATION: Top 5 nearby enemies ==========
        all_enemies = self.zombies + self.demons
        if all_enemies:
            # Sort by distance
            enemies_with_dist = []
            for e in all_enemies:
                dist = math.hypot(e['x'] - self.player_x, e['y'] - self.player_y)
                enemies_with_dist.append((e, dist))
            enemies_with_dist.sort(key=lambda x: x[1])
            
            # Track up to 5 nearest enemies (6 features each)
            for i in range(min(5, len(enemies_with_dist))):
                enemy, dist = enemies_with_dist[i]
                base_idx = 3 + i * 6
                max_dist = math.hypot(self.window_width, self.window_height)
                
                obs[base_idx] = (enemy['x'] - self.player_x) / self.window_width  # Delta X
                obs[base_idx + 1] = (enemy['y'] - self.player_y) / self.window_height  # Delta Y
                obs[base_idx + 2] = (dist / max_dist) * 2 - 1  # Distance
                obs[base_idx + 3] = 1.0 if enemy['type'] == 'zombie' else -1.0  # Type
                obs[base_idx + 4] = (enemy['health'] / 50.0) * 2 - 1  # Health (max 50 for demon)
                obs[base_idx + 5] = (enemy['speed'] / 2.0) * 2 - 1  # Speed
        
        # ========== [33-36] AGENT STATUS: Health and combat stats ==========
        obs[33] = (self.player_health / 100.0) * 2 - 1  # Current health
        obs[34] = (self.damage_taken / 100.0) * 2 - 1  # Total damage taken
        obs[35] = np.tanh(self.kills / 10.0)  # Kills (soft normalized)
        obs[36] = (self.shots_hit / max(self.shots_fired, 1)) * 2 - 1 if self.shots_fired > 0 else 0  # Accuracy
        
        # ========== [37-41] RESOURCES: Ammo and weapons ==========
        obs[37] = (self.ammo / 100.0) * 2 - 1  # Current ammo (capped at 100)
        obs[38] = 1.0 if self.current_weapon == 'pistol' else -1.0
        obs[39] = 1.0 if self.current_weapon == 'shotgun' else -1.0
        obs[40] = 1.0 if self.current_weapon == 'rifle' else -1.0
        obs[41] = len(self.available_weapons) / 3.0 * 2 - 1  # Weapons unlocked
        
        # ========== [42-47] ITEMS: Nearby pickups ==========
        # Nearest ammo pickup
        if self.ammo_pickups:
            nearest_ammo = min(self.ammo_pickups, 
                             key=lambda p: math.hypot(p['x'] - self.player_x, p['y'] - self.player_y))
            dist = math.hypot(nearest_ammo['x'] - self.player_x, nearest_ammo['y'] - self.player_y)
            obs[42] = (nearest_ammo['x'] - self.player_x) / self.window_width
            obs[43] = (nearest_ammo['y'] - self.player_y) / self.window_height
            obs[44] = (dist / math.hypot(self.window_width, self.window_height)) * 2 - 1
        
        # Nearest weapon pickup
        if self.weapon_pickups:
            nearest_weapon = min(self.weapon_pickups,
                                key=lambda p: math.hypot(p['x'] - self.player_x, p['y'] - self.player_y))
            dist = math.hypot(nearest_weapon['x'] - self.player_x, nearest_weapon['y'] - self.player_y)
            obs[45] = (nearest_weapon['x'] - self.player_x) / self.window_width
            obs[46] = (nearest_weapon['y'] - self.player_y) / self.window_height
            obs[47] = (dist / math.hypot(self.window_width, self.window_height)) * 2 - 1
        
        # ========== [48-55] MAP LAYOUT: Static features ==========
        # Distance to walls (4 directions)
        obs[48] = (self.player_x / self.window_width) * 2 - 1  # Distance to left wall
        obs[49] = ((self.window_width - self.player_x) / self.window_width) * 2 - 1  # Right
        obs[50] = (self.player_y / self.window_height) * 2 - 1  # Top
        obs[51] = ((self.window_height - self.player_y) / self.window_height) * 2 - 1  # Bottom
        
        # Nearest chokepoint distance
        obs[52] = self._get_nearest_chokepoint_distance(self.player_x, self.player_y) * 2 - 1
        
        # Nearest open area distance
        obs[53] = self._get_nearest_open_area_distance(self.player_x, self.player_y) * 2 - 1
        
        # Is player in chokepoint? (binary)
        in_chokepoint = any(math.hypot(cp['x'] - self.player_x, cp['y'] - self.player_y) < cp['radius'] 
                           for cp in self.chokepoints)
        obs[54] = 1.0 if in_chokepoint else -1.0
        
        # Is player in open area? (binary)
        in_open_area = any(math.hypot(oa['x'] - self.player_x, oa['y'] - self.player_y) < oa['radius']
                          for oa in self.open_areas)
        obs[55] = 1.0 if in_open_area else -1.0
        
        # ========== [56-59] ACTION HISTORY: Last 4 actions ==========
        for i, action in enumerate(self.action_history):
            obs[56 + i] = (action / 2.5) * 2 - 1
        
        return obs

    def step(self, action):
        self.steps += 1
        reward = 0.0
        
        # Store action in history
        self.action_history.append(action)
        
        # === Movement ===
        old_x, old_y = self.player_x, self.player_y
        if action == 1:  # Up
            self.player_y -= self.player_speed
            self.player_direction = (0, -1)
        elif action == 2:  # Down
            self.player_y += self.player_speed
            self.player_direction = (0, 1)
        elif action == 3:  # Left
            self.player_x -= self.player_speed
            self.player_direction = (-1, 0)
        elif action == 4:  # Right
            self.player_x += self.player_speed
            self.player_direction = (1, 0)
        
        # Check wall collisions
        if self._is_inside_wall(self.player_x, self.player_y):
            self.player_x, self.player_y = old_x, old_y  # Revert movement
            reward -= 0.05  # Small penalty for hitting wall
        
        # Clip to boundaries
        self.player_x = np.clip(self.player_x, 20, self.window_width - 20)
        self.player_y = np.clip(self.player_y, 20, self.window_height - 20)
        
        # === Enemy AI ===
        all_enemies = self.zombies + self.demons
        if all_enemies:
            for enemy in all_enemies:
                # Move towards player
                dx = self.player_x - enemy['x']
                dy = self.player_y - enemy['y']
                dist = math.hypot(dx, dy)
                if dist > 0:
                    dx /= dist
                    dy /= dist
                    enemy['x'] += dx * enemy['speed']
                    enemy['y'] += dy * enemy['speed']
                
                # Check collision with player
                if dist < 15:  # Collision radius
                    damage = 0.5 if enemy['type'] == 'zombie' else 0.7
                    self.player_health -= damage
                    self.damage_taken += damage
                    reward -= 0.8  # Collision penalty
        
        # === Shooting mechanics ===
        if action == 5 and self.ammo > 0:
            weapon = self.weapons[self.current_weapon]
            if self.ammo >= weapon['ammo_cost']:
                self.ammo -= weapon['ammo_cost']
                self.shots_fired += 1
                
                # Find enemies in shooting direction and range
                hit_enemy = None
                min_dist = weapon['range']
                
                for enemy in all_enemies:
                    dist = math.hypot(enemy['x'] - self.player_x, enemy['y'] - self.player_y)
                    # Check if enemy is in shooting direction
                    to_enemy = (enemy['x'] - self.player_x, enemy['y'] - self.player_y)
                    if dist > 0:
                        to_enemy = (to_enemy[0] / dist, to_enemy[1] / dist)
                        dot = to_enemy[0] * self.player_direction[0] + to_enemy[1] * self.player_direction[1]
                        
                        if dot > 0.7 and dist < min_dist:  # In front and in range
                            min_dist = dist
                            hit_enemy = enemy
                
                if hit_enemy:
                    hit_enemy['health'] -= weapon['damage']
                    self.shots_hit += 1
                    reward += 1.0  # Hit reward
                    
                    if hit_enemy['health'] <= 0:
                        kill_reward = 8.0 if hit_enemy['type'] == 'demon' else 5.0
                        reward += kill_reward
                        self.kills += 1
                        if hit_enemy in self.zombies:
                            self.zombies.remove(hit_enemy)
                        else:
                            self.demons.remove(hit_enemy)
                else:
                    reward -= 0.05  # Miss penalty
            else:
                reward -= 0.1  # No ammo penalty
        
        # === Item pickups ===
        # Ammo pickups
        for ammo in self.ammo_pickups[:]:
            if math.hypot(ammo['x'] - self.player_x, ammo['y'] - self.player_y) < 20:
                self.ammo = min(100, self.ammo + ammo['amount'])
                reward += 3.0
                self.ammo_pickups.remove(ammo)
        
        # Weapon pickups
        for weapon in self.weapon_pickups[:]:
            if math.hypot(weapon['x'] - self.player_x, weapon['y'] - self.player_y) < 20:
                if weapon['weapon'] not in self.available_weapons:
                    self.available_weapons.append(weapon['weapon'])
                    self.current_weapon = weapon['weapon']  # Auto-equip
                    reward += 5.0
                self.weapon_pickups.remove(weapon)
        
        # === Spawn new content ===
        if np.random.random() < self.enemy_spawn_rate:
            if np.random.random() < 0.7:
                self.zombies.extend(self._spawn_enemies(1, 'zombie'))
            else:
                self.demons.extend(self._spawn_enemies(1, 'demon'))
        
        if np.random.random() < self.item_spawn_rate:
            if np.random.random() < 0.6:
                self.ammo_pickups.extend(self._spawn_items(1, 'ammo'))
            else:
                self.weapon_pickups.extend(self._spawn_items(1, 'weapon'))
        
        # === Reward shaping ===
        # Survival reward
        reward += 0.1
        
        # Health-based reward
        reward += 0.2 * (self.player_health / 100.0)
        
        # Strategic positioning rewards
        if all_enemies:
            nearest_dist = min(math.hypot(e['x'] - self.player_x, e['y'] - self.player_y) 
                             for e in all_enemies)
            # Reward for maintaining optimal distance
            if 80 <= nearest_dist <= 150:
                reward += 0.4
            elif nearest_dist < 40:
                reward -= 0.3  # Too close
        
        # Ammo management reward
        if self.ammo > 20:
            reward += 0.1
        elif self.ammo < 5:
            reward -= 0.2  # Low ammo penalty
        
        # === Health decay ===
        self.player_health -= 0.06
        
        # === Terminal conditions ===
        terminated = self.player_health <= 0
        truncated = self.steps >= self.steps_per_episode
        done = terminated or truncated
        
        if terminated:
            reward -= 50  # Death penalty
        
        if truncated:
            reward += 25  # Survival bonus
        
        self.total_reward += reward
        
        info = {}
        if done:
            info["episode"] = {
                "r": self.total_reward, 
                "l": self.steps,
                "kills": self.kills,
                "damage_taken": self.damage_taken,
                "accuracy": self.shots_hit / max(self.shots_fired, 1)
            }
        
        return self._get_obs(), reward, done, truncated, info

    def render(self):
        pass  # Rendering disabled for training speed

    def close(self):
        pass

