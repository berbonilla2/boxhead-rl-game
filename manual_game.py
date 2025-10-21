#!/usr/bin/env python3
"""
Boxhead Manual Play - Simple GUI version without RL models
Play the game manually with keyboard controls
"""

import pygame
import math
import random
import time

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
LASER_COLOR = (255, 255, 255, 60)

# ==============================
# 🎮 GAME ENTITIES
# ==============================
class Player:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.radius = 12
        self.speed = 4.0
        self.health = 100
        self.max_health = 100
        self.direction = (0, -1)
        self.last_shot = 0
        self.shot_delay = 0.25
        self.ammo = 100
        self.current_weapon = 'pistol'
        self.has_shotgun = False

    def move(self, keys):
        moved = False
        dx, dy = 0, 0
        
        # Allow diagonal movement
        if keys[pygame.K_w]:
            dy -= self.speed
            if not keys[pygame.K_s]:  # Only set direction if not moving down
                self.direction = (0, -1)
            moved = True
        if keys[pygame.K_s]:
            dy += self.speed
            if not keys[pygame.K_w]:  # Only set direction if not moving up
                self.direction = (0, 1)
            moved = True
        if keys[pygame.K_a]:
            dx -= self.speed
            if not keys[pygame.K_d]:  # Only set direction if not moving right
                self.direction = (-1, 0)
            moved = True
        if keys[pygame.K_d]:
            dx += self.speed
            if not keys[pygame.K_a]:  # Only set direction if not moving left
                self.direction = (1, 0)
            moved = True
        
        # Normalize diagonal movement
        if dx != 0 and dy != 0:
            length = math.sqrt(dx*dx + dy*dy)
            dx = (dx / length) * self.speed
            dy = (dy / length) * self.speed
        
        self.x += dx
        self.y += dy

        # Clamp position inside screen boundaries
        self.x = max(self.radius, min(self.x, WINDOW_WIDTH - self.radius))
        self.y = max(self.radius, min(self.y, WINDOW_HEIGHT - self.radius))

        return moved

    def shoot(self, bullets):
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
            self.speed = 1.2
            self.radius = 10
            self.health = 30
            self.max_health = 30
        else:  # demon
            self.color = PURPLE
            self.speed = 1.4
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
# 🎮 MAIN GAME CLASS
# ==============================
class BoxheadGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Boxhead - Manual Play")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Game entities
        self.player = Player(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
        self.zombies = []
        self.demons = []
        self.bullets = []
        self.items = []
        
        # Game stats
        self.score = 0
        self.step_count = 0
        self.max_steps = 1000
        self.game_over = False
        
        # Initialize enemies and items
        self.reset_enemies()
        self.reset_items()

    def random_spawn_position(self):
        """Spawn outside 50px radius from center"""
        while True:
            x = random.randint(20, WINDOW_WIDTH - 20)
            y = random.randint(20, WINDOW_HEIGHT - 20)
            if math.hypot(x - WINDOW_WIDTH/2, y - WINDOW_HEIGHT/2) > 50:
                return x, y

    def reset_enemies(self):
        """Reset enemies"""
        self.zombies = [Enemy(*self.random_spawn_position(), 'zombie') for _ in range(4)]
        self.demons = [Enemy(*self.random_spawn_position(), 'demon') for _ in range(2)]

    def reset_items(self):
        """Reset items"""
        self.items = [
            Item(*self.random_spawn_position(), 'ammo') for _ in range(3)
        ] + [
            Item(*self.random_spawn_position(), 'health') for _ in range(1)
        ]

    def update_game(self):
        """Update game state"""
        if self.game_over:
            return
        
        keys = pygame.key.get_pressed()
        self.player.move(keys)
        
        # Shooting (continuous while held)
        if keys[pygame.K_SPACE]:
            self.player.shoot(self.bullets)
        
        # Weapon switching
        if keys[pygame.K_1]:
            self.player.current_weapon = 'pistol'
        elif keys[pygame.K_2] and self.player.has_shotgun:
            self.player.current_weapon = 'shotgun'
        
        # Mouse shooting (alternative)
        mouse_buttons = pygame.mouse.get_pressed()
        if mouse_buttons[0]:  # Left mouse button
            # Update direction to mouse
            mouse_x, mouse_y = pygame.mouse.get_pos()
            dx = mouse_x - self.player.x
            dy = mouse_y - self.player.y
            if dx != 0 or dy != 0:
                length = math.sqrt(dx*dx + dy*dy)
                self.player.direction = (dx/length, dy/length)
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
        if random.random() < 0.003:
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
        if self.player.health <= 0 or self.step_count >= self.max_steps:
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
        
        # Instructions
        if not self.game_over:
            inst_text = self.small_font.render("WASD: Move, SPACE/Click: Shoot, 1/2: Weapons", True, WHITE)
            self.screen.blit(inst_text, (10, WINDOW_HEIGHT - 30))
        
        if self.game_over:
            game_over_text = self.font.render("GAME OVER", True, RED)
            game_over_rect = game_over_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
            self.screen.blit(game_over_text, game_over_rect)
            
            restart_text = self.small_font.render("Press R to restart, ESC to quit", True, WHITE)
            restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 40))
            self.screen.blit(restart_text, restart_rect)
        
        pygame.display.flip()

    def reset_game(self):
        """Reset the game"""
        self.player = Player(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
        self.reset_enemies()
        self.reset_items()
        self.bullets = []
        self.score = 0
        self.step_count = 0
        self.game_over = False

    def run(self):
        """Main game loop"""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r and self.game_over:
                        self.reset_game()
            
            self.update_game()
            self.draw_game()
            self.clock.tick(FPS)
        
        pygame.quit()

# ==============================
# 🚀 MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    print("Boxhead Manual Play")
    print("=" * 30)
    print("Controls:")
    print("WASD - Move")
    print("SPACE - Shoot")
    print("1/2 - Switch weapons")
    print("R - Restart (when game over)")
    print("ESC - Quit")
    print("=" * 30)
    print("Starting game...")
    
    game = BoxheadGame()
    game.run()
