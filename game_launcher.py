#!/usr/bin/env python3
"""
Boxhead Game Launcher - Choose Version and Play with GUI
Supports all A2C versions (V1-V5) with visual gameplay
"""

import pygame
import numpy as np
import math
import random
import time
import os
import sys
from collections import deque
import pickle

# Add the A2C directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'A2C'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'archive'))

# Try to import stable-baselines3
try:
    from stable_baselines3 import A2C
    from stable_baselines3.common.vec_env import VecNormalize
    SB3_AVAILABLE = True
except ImportError:
    print("Warning: stable-baselines3 not available. Only manual play mode will work.")
    SB3_AVAILABLE = False

# ==============================
# 🎨 VISUAL CONFIGURATION
# ==============================
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480
FPS = 60

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
PURPLE = (180, 0, 180)
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
PINK = (255, 192, 203)
LASER_COLOR = (255, 255, 255, 60)

# ==============================
# 🎮 GAME ENTITIES
# ==============================
class Player:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.radius = 12
        self.speed = 2.8
        self.health = 100
        self.max_health = 100
        self.direction = (0, -1)
        self.last_shot = 0
        self.shot_delay = 0.25
        self.ammo = 75
        self.current_weapon = 'pistol'
        self.has_shotgun = False

    def move(self, action):
        """Move based on action (0-5)"""
        dx, dy = 0, 0
        if action == 1:  # Up
            dy = -self.speed
            self.direction = (0, -1)
        elif action == 2:  # Down
            dy = self.speed
            self.direction = (0, 1)
        elif action == 3:  # Left
            dx = -self.speed
            self.direction = (-1, 0)
        elif action == 4:  # Right
            dx = self.speed
            self.direction = (1, 0)
        
        self.x = np.clip(self.x + dx, 15, WINDOW_WIDTH - 15)
        self.y = np.clip(self.y + dy, 15, WINDOW_HEIGHT - 15)

    def shoot(self, bullets):
        """Shoot if action is 5"""
        now = time.time()
        if now - self.last_shot >= self.shot_delay and self.ammo > 0:
            dx, dy = self.direction
            damage = 34 if self.current_weapon == 'pistol' else 68
            range_val = 195 if self.current_weapon == 'pistol' else 135
            bullets.append(Bullet(self.x + dx * 15, self.y + dy * 15, 8, self.direction, "player", damage, range_val))
            self.last_shot = now
            self.ammo -= 1 if self.current_weapon == 'pistol' else 2

    def draw(self, surface):
        # Body
        pygame.draw.circle(surface, BLUE, (int(self.x), int(self.y)), self.radius)
        
        # Direction indicator
        fx_tip, fy_tip = self.x + self.direction[0] * 15, self.y + self.direction[1] * 15
        pygame.draw.line(surface, YELLOW, (self.x, self.y), (fx_tip, fy_tip), 3)
        
        # Health bar
        health_width = 30 * (self.health / self.max_health)
        pygame.draw.rect(surface, RED, (self.x - 15, self.y - 25, 30, 4))
        pygame.draw.rect(surface, GREEN, (self.x - 15, self.y - 25, health_width, 4))
        
        # Ammo counter
        font = pygame.font.Font(None, 24)
        ammo_text = font.render(f"Ammo: {self.ammo}", True, WHITE)
        surface.blit(ammo_text, (10, 10))
        
        # Weapon indicator
        weapon_text = font.render(f"Weapon: {self.current_weapon}", True, WHITE)
        surface.blit(weapon_text, (10, 35))

class Bullet:
    def __init__(self, x, y, speed, direction, owner, damage=25, range_val=200):
        self.x, self.y = x, y
        self.speed = speed
        self.direction = direction
        self.owner = owner
        self.damage = damage
        self.range_val = range_val
        self.radius = 4
        self.distance_traveled = 0

    def update(self):
        self.x += self.direction[0] * self.speed
        self.y += self.direction[1] * self.speed
        self.distance_traveled += self.speed

    def draw(self, surface):
        color = WHITE if self.owner == "player" else ORANGE
        pygame.draw.circle(surface, color, (int(self.x), int(self.y)), self.radius)

class Enemy:
    def __init__(self, x, y, enemy_type):
        self.x, self.y = x, y
        self.enemy_type = enemy_type
        if enemy_type == 'zombie':
            self.color = RED
            self.speed = 0.62
            self.radius = 10
            self.health = 30
            self.max_health = 30
        else:  # demon
            self.color = PURPLE
            self.speed = 0.82
            self.radius = 14
            self.health = 50
            self.max_health = 50
        self.last_shot = time.time()

    def move_toward(self, target_x, target_y):
        dx, dy = target_x - self.x, target_y - self.y
        dist = math.hypot(dx, dy)
        if dist > 0:
            dx, dy = dx / dist, dy / dist
            self.x += dx * self.speed
            self.y += dy * self.speed

    def shoot(self, bullets, player):
        if self.enemy_type == 'demon':
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

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)
        # Health bar
        health_width = (self.radius * 2) * (self.health / self.max_health)
        pygame.draw.rect(surface, RED, (self.x - self.radius, self.y - self.radius - 8, self.radius * 2, 3))
        pygame.draw.rect(surface, GREEN, (self.x - self.radius, self.y - self.radius - 8, health_width, 3))

class Item:
    def __init__(self, x, y, item_type):
        self.x, self.y = x, y
        self.item_type = item_type
        self.radius = 8
        if item_type == 'ammo':
            self.color = YELLOW
        elif item_type == 'weapon':
            self.color = CYAN
        elif item_type == 'health':
            self.color = GREEN

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)

# ==============================
# 🎮 GAME LAUNCHER CLASS
# ==============================
class BoxheadGameLauncher:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Boxhead RL - Version Launcher")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Game state
        self.mode = "menu"  # "menu", "playing", "paused"
        self.selected_version = None
        self.model = None
        self.vec_normalize = None
        
        # Game entities
        self.player = None
        self.zombies = []
        self.demons = []
        self.bullets = []
        self.items = []
        
        # Game stats
        self.score = 0
        self.step_count = 0
        self.max_steps = 10000  # Increased step limit
        self.game_over = False
        
        # Debug settings
        self.debug_mode = True  # Set to False to reduce output
        self.debug_frequency = 30  # Print debug info every N steps
        self.use_random_actions = False  # Toggle for testing
        self.disable_vecnormalize = False  # Toggle VecNormalize
        
        # Agent action delay settings
        self.last_agent_action_time = time.time()
        self.agent_action_delay = .5  # 0.5 seconds delay between agent actions
        self.delayed_agent_actions = 0  # Counter for delayed agent actions
        self.last_agent_action = 0  # Last action taken by agent
        
        # Available versions
        self.versions = {
            "V1 (Archive)": {
                "model_path": "archive/models/boxhead_A2C.zip",
                "env_type": "v1",
                "description": "Original 9-feature environment (RECOMMENDED)"
            },
            "V2": {
                "model_path": "A2C/Models/boxhead_A2C_v2.zip",
                "vecnorm_path": "A2C/Models/vecnormalize_v2.pkl",
                "env_type": "v2",
                "description": "60-feature environment (over-complex)"
            },
            "V3": {
                "model_path": "A2C/Models/boxhead_A2C_v3.zip",
                "vecnorm_path": "A2C/Models/vecnormalize_v3.pkl",
                "env_type": "v3",
                "description": "42-feature environment (optimized)"
            },
            "V4": {
                "model_path": "A2C/Models/boxhead_A2C_v4.zip",
                "vecnorm_path": "A2C/Models/vecnormalize_v4.pkl",
                "env_type": "v4",
                "description": "Skip connections + optimizations"
            },
            "V5": {
                "model_path": "A2C/Models/boxhead_A2C_v5.zip",
                "vecnorm_path": "A2C/Models/vecnormalize_v5.pkl",
                "env_type": "v5",
                "description": "Final optimized version (best)"
            }
        }

    def show_menu(self):
        """Display version selection menu"""
        self.screen.fill(BLACK)
        
        # Title
        title = self.font.render("Boxhead RL - Choose Version", True, WHITE)
        title_rect = title.get_rect(center=(WINDOW_WIDTH//2, 50))
        self.screen.blit(title, title_rect)
        
        # Version options
        y_offset = 120
        for i, (version_name, info) in enumerate(self.versions.items()):
            color = YELLOW if i == 0 else WHITE
            version_text = self.small_font.render(f"{i+1}. {version_name}", True, color)
            self.screen.blit(version_text, (50, y_offset))
            
            desc_text = self.small_font.render(f"   {info['description']}", True, GRAY)
            self.screen.blit(desc_text, (50, y_offset + 25))
            
            y_offset += 60
        
        # Instructions
        inst_text = self.small_font.render("Press 1-5 to select version, ESC to quit", True, WHITE)
        self.screen.blit(inst_text, (50, y_offset + 20))
        
        pygame.display.flip()

    def load_model(self, version_name):
        """Load the selected model"""
        if not SB3_AVAILABLE:
            print("Error: stable-baselines3 not available. Cannot load models.")
            return False
            
        version_info = self.versions[version_name]
        
        try:
            # Load model
            self.model = A2C.load(version_info["model_path"])
            print(f"Loaded model: {version_info['model_path']}")
            
            # Load vector normalization if available
            if "vecnorm_path" in version_info and os.path.exists(version_info["vecnorm_path"]):
                with open(version_info["vecnorm_path"], 'rb') as f:
                    self.vec_normalize = pickle.load(f)
                print(f"Loaded VecNormalize: {version_info['vecnorm_path']}")
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def reset_game(self):
        """Reset game state"""
        self.player = Player(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
        self.zombies = [Enemy(*self.random_spawn_position(), 'zombie') for _ in range(2)]
        self.demons = [Enemy(*self.random_spawn_position(), 'demon') for _ in range(1)]
        self.bullets = []
        self.items = [Item(*self.random_spawn_position(), 'ammo') for _ in range(2)]
        self.score = 0
        self.step_count = 0
        self.game_over = False
        # Reset input delay timer
        self.last_input_time = time.time()

    def random_spawn_position(self):
        """Spawn outside 50px radius from center"""
        while True:
            x = random.randint(20, WINDOW_WIDTH - 20)
            y = random.randint(20, WINDOW_HEIGHT - 20)
            if math.hypot(x - WINDOW_WIDTH/2, y - WINDOW_HEIGHT/2) > 50:
                return x, y

    def get_observation(self):
        """Get current observation based on version"""
        if not self.player:
            # Return appropriate size based on selected version
            if self.selected_version == "V1 (Archive)":
                return np.zeros(9, dtype=np.float32)
            elif self.selected_version == "V2":
                return np.zeros(60, dtype=np.float32)
            else:  # V3, V4, V5
                return np.zeros(42, dtype=np.float32)
        
        # V1: 9 features (original simple state)
        if self.selected_version == "V1 (Archive)":
            def nearest_enemy(enemies):
                if not enemies:
                    return (0, 0, 1.0)  # dx, dy, distance
                nearest = min(enemies, key=lambda e: math.hypot(e.x - self.player.x, e.y - self.player.y))
                dx = (nearest.x - self.player.x) / WINDOW_WIDTH
                dy = (nearest.y - self.player.y) / WINDOW_HEIGHT
                dist = math.hypot(nearest.x - self.player.x, nearest.y - self.player.y) / 500
                return (dx, dy, min(dist, 1.0))
            
            z_dx, z_dy, z_dist = nearest_enemy(self.zombies)
            d_dx, d_dy, d_dist = nearest_enemy(self.demons)
            
            obs = np.array([
                self.player.x / WINDOW_WIDTH,           # 0: Player X
                self.player.y / WINDOW_HEIGHT,          # 1: Player Y  
                self.player.health / 100,               # 2: Player health
                z_dx,                                   # 3: Delta X to nearest zombie
                z_dy,                                   # 4: Delta Y to nearest zombie
                z_dist,                                 # 5: Distance to nearest zombie
                d_dx,                                   # 6: Delta X to nearest demon
                d_dy,                                   # 7: Delta Y to nearest demon
                d_dist,                                 # 8: Distance to nearest demon
            ], dtype=np.float32)
            return obs
        
        # V2: 60 features (over-complex)
        elif self.selected_version == "V2":
            # Basic features
            obs = [
                self.player.x / WINDOW_WIDTH,
                self.player.y / WINDOW_HEIGHT,
                self.player.health / 100,
                self.player.ammo / 100,
                1.0 if self.player.current_weapon == 'pistol' else 0.0,
                1.0 if self.player.has_shotgun else 0.0,
            ]
            
            # Enemy features (up to 5 zombies, 3 demons)
            for i in range(5):
                if i < len(self.zombies):
                    z = self.zombies[i]
                    dx = (z.x - self.player.x) / WINDOW_WIDTH
                    dy = (z.y - self.player.y) / WINDOW_HEIGHT
                    dist = math.hypot(z.x - self.player.x, z.y - self.player.y) / 500
                    health = z.health / z.max_health
                    obs.extend([dx, dy, dist, health])
                else:
                    obs.extend([0, 0, 1, 0])  # No enemy
            
            for i in range(3):
                if i < len(self.demons):
                    d = self.demons[i]
                    dx = (d.x - self.player.x) / WINDOW_WIDTH
                    dy = (d.y - self.player.y) / WINDOW_HEIGHT
                    dist = math.hypot(d.x - self.player.x, d.y - self.player.y) / 500
                    health = d.health / d.max_health
                    obs.extend([dx, dy, dist, health])
                else:
                    obs.extend([0, 0, 1, 0])  # No enemy
            
            # Item features (up to 5 items)
            for i in range(5):
                if i < len(self.items):
                    item = self.items[i]
                    dx = (item.x - self.player.x) / WINDOW_WIDTH
                    dy = (item.y - self.player.y) / WINDOW_HEIGHT
                    dist = math.hypot(item.x - self.player.x, item.y - self.player.y) / 500
                    item_type = 1.0 if item.item_type == 'ammo' else (2.0 if item.item_type == 'health' else 3.0)
                    obs.extend([dx, dy, dist, item_type])
                else:
                    obs.extend([0, 0, 1, 0])  # No item
            
            # Pad to exactly 60 features
            while len(obs) < 60:
                obs.append(0.0)
            
            return np.array(obs[:60], dtype=np.float32)
        
        # V3, V4, V5: 42 features (optimized)
        else:
            # Basic features
            obs = [
                self.player.x / WINDOW_WIDTH,
                self.player.y / WINDOW_HEIGHT,
                self.player.health / 100,
                self.player.ammo / 100,
                1.0 if self.player.current_weapon == 'pistol' else 0.0,
                1.0 if self.player.has_shotgun else 0.0,
            ]
            
            # Nearest enemy features
            def nearest_enemy(enemies):
                if not enemies:
                    return (0, 0, 1.0, 0)  # dx, dy, distance, health
                nearest = min(enemies, key=lambda e: math.hypot(e.x - self.player.x, e.y - self.player.y))
                dx = (nearest.x - self.player.x) / WINDOW_WIDTH
                dy = (nearest.y - self.player.y) / WINDOW_HEIGHT
                dist = math.hypot(nearest.x - self.player.x, nearest.y - self.player.y) / 500
                health = nearest.health / nearest.max_health
                return (dx, dy, min(dist, 1.0), health)
            
            z_dx, z_dy, z_dist, z_health = nearest_enemy(self.zombies)
            d_dx, d_dy, d_dist, d_health = nearest_enemy(self.demons)
            
            obs.extend([z_dx, z_dy, z_dist, z_health, d_dx, d_dy, d_dist, d_health])
            
            # Additional features for V3-V5
            # Enemy counts
            obs.extend([len(self.zombies) / 10, len(self.demons) / 5])
            
            # Item features
            ammo_items = sum(1 for item in self.items if item.item_type == 'ammo')
            health_items = sum(1 for item in self.items if item.item_type == 'health')
            weapon_items = sum(1 for item in self.items if item.item_type == 'weapon')
            obs.extend([ammo_items / 5, health_items / 3, weapon_items / 2])
            
            # Bullet features
            player_bullets = sum(1 for b in self.bullets if b.owner == 'player')
            enemy_bullets = sum(1 for b in self.bullets if b.owner == 'demon')
            obs.extend([player_bullets / 10, enemy_bullets / 5])
            
            # Distance to walls
            obs.extend([
                self.player.x / WINDOW_WIDTH,  # Distance to left wall
                (WINDOW_WIDTH - self.player.x) / WINDOW_WIDTH,  # Distance to right wall
                self.player.y / WINDOW_HEIGHT,  # Distance to top wall
                (WINDOW_HEIGHT - self.player.y) / WINDOW_HEIGHT,  # Distance to bottom wall
            ])
            
            # Pad to exactly 42 features
            while len(obs) < 42:
                obs.append(0.0)
            
            return np.array(obs[:42], dtype=np.float32)

    def get_action_from_model(self):
        """Get action from loaded model"""
        if not self.model:
            return 0
        
        current_time = time.time()
        
        # Check if enough time has passed since last agent action
        if self.agent_action_delay > 0 and current_time - self.last_agent_action_time < self.agent_action_delay:
            # Return the last action if not enough time has passed
            self.delayed_agent_actions += 1
            if self.debug_mode and self.step_count % 30 == 0:
                print(f"Agent action delayed: {current_time - self.last_agent_action_time:.2f}s < {self.agent_action_delay}s")
            return getattr(self, 'last_agent_action', 0)
        
        # Update last agent action time
        self.last_agent_action_time = current_time
        
        obs = self.get_observation()
        
        # Debug: Print observation details every 60 steps
        if self.debug_mode and self.step_count % 60 == 0:
            print(f"\n--- DEBUG INFO (Step {self.step_count}) ---")
            print(f"Observation shape: {obs.shape}")
            print(f"Observation sample: {obs[:10] if len(obs) >= 10 else obs}")
            print(f"Player pos: ({self.player.x:.1f}, {self.player.y:.1f})")
            print(f"Nearest zombie: {self.zombies[0].x:.1f}, {self.zombies[0].y:.1f}" if self.zombies else "No zombies")
            print(f"Nearest demon: {self.demons[0].x:.1f}, {self.demons[0].y:.1f}" if self.demons else "No demons")
        
        # Apply normalization if available (but allow disabling it)
        if self.vec_normalize and not self.disable_vecnormalize:
            try:
                obs = self.vec_normalize.normalize_obs(obs.reshape(1, -1))[0]
                if self.debug_mode and self.step_count % 60 == 0:
                    print(f"Applied VecNormalize")
            except Exception as e:
                if self.debug_mode and self.step_count % 60 == 0:
                    print(f"VecNormalize failed: {e}")
                    print(f"Disabling VecNormalize and using raw observations")
                self.disable_vecnormalize = True
        
        # Get action from model or use random actions for testing
        if self.use_random_actions:
            import random
            action = random.randint(0, 5)
            if self.debug_mode and self.step_count % 30 == 0:
                print(f"Using RANDOM action: {action}")
        else:
            # Try different prediction modes
            try:
                # Special handling for V3/V5 which tend to get stuck
                if self.selected_version in ["V3", "V5"]:
                    # V3/V5: Try non-deterministic first (they seem to work better this way)
                    action, _ = self.model.predict(obs, deterministic=False)
                    
                    # If still getting 0, try with noise
                    if action == 0 and self.step_count > 5:
                        obs_noisy = obs + np.random.normal(0, 0.05, obs.shape).astype(np.float32)
                        action, _ = self.model.predict(obs_noisy, deterministic=False)
                        if self.debug_mode and self.step_count % 60 == 0:
                            print(f"V3/V5: Tried noisy observation")
                    
                    # If still 0, try deterministic
                    if action == 0 and self.step_count > 10:
                        action, _ = self.model.predict(obs, deterministic=True)
                        if self.debug_mode and self.step_count % 60 == 0:
                            print(f"V3/V5: Switched to deterministic")
                    
                    # If still 0, force some movement
                    if action == 0 and self.step_count > 15:
                        import random
                        action = random.choice([1, 2, 3, 4])  # Force movement, no idle
                        if self.debug_mode and self.step_count % 60 == 0:
                            print(f"V3/V5: Forced movement action: {action}")
                else:
                    # V1, V2, V4: Try deterministic first
                    action, _ = self.model.predict(obs, deterministic=True)
                    
                    # If we keep getting the same action (0), try non-deterministic
                    if action == 0 and self.step_count > 10:
                        action, _ = self.model.predict(obs, deterministic=False)
                        if self.debug_mode and self.step_count % 60 == 0:
                            print(f"Switched to non-deterministic prediction")
                    
                    # If still getting 0, try with different observation
                    if action == 0 and self.step_count > 20:
                        # Add small noise to observation
                        obs_noisy = obs + np.random.normal(0, 0.01, obs.shape).astype(np.float32)
                        action, _ = self.model.predict(obs_noisy, deterministic=False)
                        if self.debug_mode and self.step_count % 60 == 0:
                            print(f"Tried noisy observation")
                        
            except Exception as e:
                if self.debug_mode and self.step_count % 60 == 0:
                    print(f"Model prediction failed: {e}")
                action = 0
        
        # Convert numpy array to scalar if needed
        if isinstance(action, np.ndarray):
            if action.size == 1:
                action = int(action.item())
            elif action.size > 1:
                action = int(action[0])
            else:
                action = 0
        elif hasattr(action, 'item'):
            action = action.item()
        
        # Store the last action for delay purposes
        self.last_agent_action = action
        
        # Debug: Print action being taken
        if self.debug_mode:
            action_names = {
                0: "IDLE",
                1: "UP",
                2: "DOWN", 
                3: "LEFT",
                4: "RIGHT",
                5: "SHOOT"
            }
            print(f"Agent Action: {action} ({action_names.get(action, 'UNKNOWN')})")
        
        return action

    def update_game(self):
        """Update game state"""
        if self.game_over:
            return
        
        # Get action from model
        action = self.get_action_from_model()
        
        # Debug: Print game state every N steps
        if self.debug_mode and self.step_count % self.debug_frequency == 0:
            print(f"\n--- Step {self.step_count} ---")
            print(f"Player: pos=({self.player.x:.1f}, {self.player.y:.1f}), health={self.player.health:.1f}, ammo={self.player.ammo}")
            print(f"Enemies: {len(self.zombies)} zombies, {len(self.demons)} demons")
            print(f"Items: {len(self.items)} items, Bullets: {len(self.bullets)}")
            print(f"Score: {self.score}")
        
        # Update player
        self.player.move(action)
        if action == 5:  # Shoot
            self.player.shoot(self.bullets)
        
        # Update bullets
        for bullet in self.bullets[:]:
            bullet.update()
            
            # Remove bullets that are out of bounds or traveled too far
            if (not (0 <= bullet.x <= WINDOW_WIDTH and 0 <= bullet.y <= WINDOW_HEIGHT) or
                bullet.distance_traveled > bullet.range_val):
                self.bullets.remove(bullet)
                continue
            
            # Player bullets hit enemies
            if bullet.owner == "player":
                for enemy in self.zombies + self.demons:
                    if math.hypot(bullet.x - enemy.x, bullet.y - enemy.y) <= enemy.radius:
                        enemy.health -= bullet.damage
                        if bullet in self.bullets:
                            self.bullets.remove(bullet)
                        if enemy.health <= 0:
                            self.score += 10 if enemy.enemy_type == 'zombie' else 20
                        break
            
            # Demon bullets hit player
            elif bullet.owner == "demon":
                if math.hypot(bullet.x - self.player.x, bullet.y - self.player.y) <= self.player.radius:
                    self.player.health -= 15
                    self.bullets.remove(bullet)
        
        # Update enemies
        for zombie in self.zombies[:]:
            zombie.move_toward(self.player.x, self.player.y)
            if zombie.health <= 0:
                self.zombies.remove(zombie)
            elif math.hypot(zombie.x - self.player.x, zombie.y - self.player.y) <= zombie.radius + self.player.radius:
                self.player.health -= 0.4
        
        for demon in self.demons[:]:
            demon.move_toward(self.player.x, self.player.y)
            demon.shoot(self.bullets, self.player)
            if demon.health <= 0:
                self.demons.remove(demon)
            elif math.hypot(demon.x - self.player.x, demon.y - self.player.y) <= demon.radius + self.player.radius:
                self.player.health -= 0.6
        
        # Update items
        for item in self.items[:]:
            if math.hypot(item.x - self.player.x, item.y - self.player.y) <= item.radius + self.player.radius:
                if item.item_type == 'ammo':
                    self.player.ammo = min(100, self.player.ammo + 20)
                elif item.item_type == 'weapon':
                    self.player.has_shotgun = True
                    self.player.current_weapon = 'shotgun'
                elif item.item_type == 'health':
                    self.player.health = min(100, self.player.health + 30)
                self.items.remove(item)
        
        # Spawn new enemies occasionally
        if random.random() < 0.002:
            if random.random() < 0.8:
                self.zombies.append(Enemy(*self.random_spawn_position(), 'zombie'))
            else:
                self.demons.append(Enemy(*self.random_spawn_position(), 'demon'))
        
        # Spawn new items
        if random.random() < 0.003:
            item_type = random.choice(['ammo', 'health'])
            self.items.append(Item(*self.random_spawn_position(), item_type))
        
        # Check game over conditions
        self.step_count += 1
        if self.player.health <= 0:  # Only end when health is 0
            self.game_over = True

    def draw_game(self):
        """Draw the game"""
        self.screen.fill(BLACK)
        
        # Background grid
        for x in range(0, WINDOW_WIDTH, 32):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, WINDOW_HEIGHT))
        for y in range(0, WINDOW_HEIGHT, 32):
            pygame.draw.line(self.screen, GRAY, (0, y), (WINDOW_WIDTH, y))
        
        # Draw entities
        if self.player:
            self.player.draw(self.screen)
        
        for bullet in self.bullets:
            bullet.draw(self.screen)
        
        for zombie in self.zombies:
            zombie.draw(self.screen)
        
        for demon in self.demons:
            demon.draw(self.screen)
        
        for item in self.items:
            item.draw(self.screen)
        
        # Draw UI
        score_text = self.small_font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (WINDOW_WIDTH - 150, 10))
        
        step_text = self.small_font.render(f"Steps: {self.step_count}/{self.max_steps}", True, WHITE)
        self.screen.blit(step_text, (WINDOW_WIDTH - 150, 35))
        
        # Debug info
        debug_text = self.small_font.render(f"Debug: {'ON' if self.debug_mode else 'OFF'}", True, WHITE)
        self.screen.blit(debug_text, (WINDOW_WIDTH - 150, 60))
        
        freq_text = self.small_font.render(f"Freq: {self.debug_frequency}", True, WHITE)
        self.screen.blit(freq_text, (WINDOW_WIDTH - 150, 85))
        
        # Controls
        controls_text = self.small_font.render("D: Debug, F: Freq, T: Random, V: VecNorm, I: Agent Delay, R: Restart, ESC: Menu", True, WHITE)
        self.screen.blit(controls_text, (10, WINDOW_HEIGHT - 30))
        
        # Show current mode
        mode_text = self.small_font.render(f"Mode: {'RANDOM' if self.use_random_actions else 'AI'}", True, WHITE)
        self.screen.blit(mode_text, (WINDOW_WIDTH - 150, 110))
        
        # Show VecNormalize status
        vecnorm_text = self.small_font.render(f"VecNorm: {'OFF' if self.disable_vecnormalize else 'ON'}", True, WHITE)
        self.screen.blit(vecnorm_text, (WINDOW_WIDTH - 150, 135))
        
        # Show agent action delay status
        delay_text = self.small_font.render(f"Agent Delay: {self.agent_action_delay}s", True, WHITE)
        self.screen.blit(delay_text, (WINDOW_WIDTH - 150, 160))
        
        # Show delayed agent actions counter
        delayed_text = self.small_font.render(f"Delayed: {self.delayed_agent_actions}", True, WHITE)
        self.screen.blit(delayed_text, (WINDOW_WIDTH - 150, 185))
        
        # Show if agent action was recently taken
        current_time = time.time()
        if current_time - self.last_agent_action_time < 0.1:  # Show for 0.1 seconds after action
            delay_indicator = self.small_font.render("AGENT ACTED", True, GREEN)
            self.screen.blit(delay_indicator, (WINDOW_WIDTH - 150, 210))
        
        if self.game_over:
            game_over_text = self.font.render("GAME OVER", True, RED)
            game_over_rect = game_over_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
            self.screen.blit(game_over_text, game_over_rect)
            
            restart_text = self.small_font.render("Press R to restart, ESC for menu", True, WHITE)
            restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 40))
            self.screen.blit(restart_text, restart_rect)
        
        pygame.display.flip()

    def run(self):
        """Main game loop"""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    
                    if self.mode == "menu":
                        if event.key == pygame.K_1:
                            self.selected_version = "V1 (Archive)"
                        elif event.key == pygame.K_2:
                            self.selected_version = "V2"
                        elif event.key == pygame.K_3:
                            self.selected_version = "V3"
                        elif event.key == pygame.K_4:
                            self.selected_version = "V4"
                        elif event.key == pygame.K_5:
                            self.selected_version = "V5"
                        elif event.key == pygame.K_ESCAPE:
                            running = False
                        
                        if self.selected_version:
                            if self.load_model(self.selected_version):
                                self.mode = "playing"
                                self.reset_game()
                            else:
                                self.selected_version = None
                    
                    elif self.mode == "playing":
                        if event.key == pygame.K_ESCAPE:
                            self.mode = "menu"
                            self.selected_version = None
                            self.model = None
                            self.vec_normalize = None
                        elif event.key == pygame.K_r and self.game_over:
                            self.reset_game()
                        elif event.key == pygame.K_d:
                            # Toggle debug mode
                            self.debug_mode = not self.debug_mode
                            print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                        elif event.key == pygame.K_f:
                            # Toggle debug frequency
                            self.debug_frequency = 10 if self.debug_frequency == 30 else 30
                            print(f"Debug frequency: every {self.debug_frequency} steps")
                        elif event.key == pygame.K_t and not self.game_over:
                            # Toggle random actions (only when not game over)
                            self.use_random_actions = not self.use_random_actions
                            print(f"Random actions: {'ON' if self.use_random_actions else 'OFF'}")
                        elif event.key == pygame.K_v and not self.game_over:
                            # Toggle VecNormalize (only when not game over)
                            self.disable_vecnormalize = not self.disable_vecnormalize
                            print(f"VecNormalize: {'DISABLED' if self.disable_vecnormalize else 'ENABLED'}")
                        elif event.key == pygame.K_i and not self.game_over:
                            # Toggle agent action delay (only when not game over)
                            self.agent_action_delay = 0.0 if self.agent_action_delay > 0 else 0.5
                            print(f"Agent action delay: {self.agent_action_delay}s")
            
            if self.mode == "menu":
                self.show_menu()
            elif self.mode == "playing":
                self.update_game()
                self.draw_game()
            
            self.clock.tick(FPS)
        
        pygame.quit()

# ==============================
# 🚀 MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    print("Boxhead RL Game Launcher")
    print("=" * 40)
    print("Available versions:")
    print("1. V1 (Archive) - Original 9-feature environment")
    print("2. V2 - 60-feature environment (over-complex)")
    print("3. V3 - 42-feature environment (optimized)")
    print("4. V4 - Skip connections + optimizations")
    print("5. V5 - Final optimized version (best)")
    print("=" * 40)
    print("Starting game launcher...")
    
    launcher = BoxheadGameLauncher()
    launcher.run()
