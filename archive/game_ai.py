import pygame
import math
import random
import torch
import numpy as np
from stable_baselines3 import DQN, PPO, A2C
import time
from game import Player, Enemy, Demon, Barrel, Package, distance, random_spawn_position

# =====================================================
# CONFIGURATION
# =====================================================
MODEL_DIR = "models"
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480
FPS = 60
PLAYER_SPEED = 2.0
ZOMBIE_SPEED = 0.7
DEMON_SPEED = 0.9
ENEMY_SPAWN_RATE = 0.003

# =====================================================
# INIT
# =====================================================
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Boxhead AI Selector")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 26, bold=True)
small_font = pygame.font.SysFont("Arial", 20)

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# MODEL SELECTION UI
# =====================================================
def draw_menu(selected_index):
    screen.fill((10, 10, 10))
    title_text = font.render("Select AI Model", True, (255, 255, 255))
    screen.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, 80))
    models = ["DQN", "PPO", "A2C"]

    for i, name in enumerate(models):
        color = (255, 255, 0) if i == selected_index else (200, 200, 200)
        text = font.render(name, True, color)
        screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, 160 + i * 60))

    hint = small_font.render("↑ ↓ to choose, Enter to start, Esc to quit", True, (180, 180, 180))
    screen.blit(hint, (WINDOW_WIDTH // 2 - hint.get_width() // 2, 400))
    pygame.display.flip()

def select_model_ui():
    selected_index = 0
    models = ["DQN", "PPO", "A2C"]
    running = True
    while running:
        draw_menu(selected_index)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_index = (selected_index - 1) % len(models)
                elif event.key == pygame.K_DOWN:
                    selected_index = (selected_index + 1) % len(models)
                elif event.key == pygame.K_RETURN:
                    return models[selected_index]
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit(); exit()
        clock.tick(15)

# =====================================================
# GAME SETUP (After Model Selection)
# =====================================================
ALGO_NAME = select_model_ui()
MODEL_PATH = f"{MODEL_DIR}/boxhead_{ALGO_NAME}.zip"

model_class = {"DQN": DQN, "PPO": PPO, "A2C": A2C}[ALGO_NAME]
try:
    model = model_class.load(MODEL_PATH, device=device)
    print(f"✅ Loaded {MODEL_PATH} on {device}")
except Exception as e:
    print(f"❌ Failed to load model {MODEL_PATH}: {e}")
    pygame.quit()
    exit()

pygame.display.set_caption(f"Boxhead AI Player ({ALGO_NAME})")

# =====================================================
# GAME VARIABLES
# =====================================================
player = Player(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
player.speed = PLAYER_SPEED
bullets = []
zombies = [Enemy(*random_spawn_position(), (255, 0, 0), ZOMBIE_SPEED) for _ in range(5)]
demons = [Demon(*random_spawn_position()) for _ in range(2)]
barrels = [Barrel(*random_spawn_position()) for _ in range(3)]
packages = [Package(*random_spawn_position()) for _ in range(2)]
score = 0
running = True

# =====================================================
# HELPER
# =====================================================
def get_state():
    if zombies or demons:
        enemies = zombies + demons
        nearest = min(enemies, key=lambda e: distance((player.x, player.y), (e.x, e.y)))
        dx = (nearest.x - player.x) / WINDOW_WIDTH
        dy = (nearest.y - player.y) / WINDOW_HEIGHT
    else:
        dx = dy = 0

    base_obs = np.array([
        player.x / WINDOW_WIDTH,
        player.y / WINDOW_HEIGHT,
        player.health / 100,
        dx, dy
    ], dtype=np.float32)
    pad = np.zeros(9 - base_obs.shape[0], dtype=np.float32)
    return np.concatenate([base_obs, pad])

def ai_action_to_movement(action):
    if action == 1:
        player.y -= player.speed
        player.direction = (0, -1)
    elif action == 2:
        player.y += player.speed
        player.direction = (0, 1)
    elif action == 3:
        player.x -= player.speed
        player.direction = (-1, 0)
    elif action == 4:
        player.x += player.speed
        player.direction = (1, 0)
    elif action == 5:
        player.shoot(bullets)

    # Clamp inside window
    player.x = max(player.radius, min(WINDOW_WIDTH - player.radius, player.x))
    player.y = max(player.radius, min(WINDOW_HEIGHT - player.radius, player.y))

# =====================================================
# MAIN GAME LOOP
# =====================================================
while running:
    dt = clock.tick(FPS)
    screen.fill((0, 0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        ):
            running = False

    # --- AI decision ---
    obs = np.expand_dims(get_state(), axis=0)
    action, _ = model.predict(obs, deterministic=True)
    ai_action_to_movement(action)

    # --- Update bullets ---
    for b in bullets[:]:
        b.update()
        if not (0 <= b.x <= WINDOW_WIDTH and 0 <= b.y <= WINDOW_HEIGHT):
            bullets.remove(b)
            continue
        if b.owner == "player":
            for e in zombies + demons:
                if distance((b.x, b.y), (e.x, e.y)) <= e.radius:
                    e.health -= 25
                    score += 10
                    if b in bullets: bullets.remove(b)
                    break
            for br in barrels[:]:
                if distance((b.x, b.y), (br.x, br.y)) <= br.radius:
                    br.explode(zombies + demons)
                    barrels.remove(br)
                    score += 5
                    if b in bullets: bullets.remove(b)
                    break
        elif b.owner == "demon":
            if distance((b.x, b.y), (player.x, player.y)) <= player.radius:
                player.health -= 15
                bullets.remove(b)

    # --- Enemy behavior ---
    for z in zombies[:]:
        z.move_toward(player.x, player.y)
        if z.health <= 0:
            zombies.remove(z); score += 15
        elif distance((z.x, z.y), (player.x, player.y)) <= z.radius + player.radius:
            player.health -= 0.4

    for d in demons[:]:
        d.move_toward(player.x, player.y)
        d.shoot(bullets, player)
        if d.health <= 0:
            demons.remove(d); score += 25
        elif distance((d.x, d.y), (player.x, player.y)) <= d.radius + player.radius:
            player.health -= 0.6

    if random.random() < ENEMY_SPAWN_RATE:
        if random.random() < 0.8:
            zombies.append(Enemy(*random_spawn_position(), (255, 0, 0), ZOMBIE_SPEED))
        else:
            demons.append(Demon(*random_spawn_position()))

    for p in packages[:]:
        if distance((p.x, p.y), (player.x, player.y)) <= p.radius + player.radius:
            player.health = 100
            packages.remove(p)
            score += 20

    # --- Draw everything ---
    for x in range(0, WINDOW_WIDTH, 32):
        pygame.draw.line(screen, (40, 40, 40), (x, 0), (x, WINDOW_HEIGHT))
    for y in range(0, WINDOW_HEIGHT, 32):
        pygame.draw.line(screen, (40, 40, 40), (0, y), (WINDOW_WIDTH, y))

    player.draw(screen)
    for b in bullets: b.draw(screen)
    for z in zombies: z.draw(screen)
    for d in demons: d.draw(screen)
    for br in barrels: br.draw(screen)
    for p in packages: p.draw(screen)

    # --- HUD ---
    health_text = small_font.render(f"Health: {int(player.health)}", True, (255, 255, 255))
    score_text = small_font.render(f"Score: {int(score)}", True, (255, 255, 0))
    ai_text = small_font.render(f"AI: {ALGO_NAME}", True, (150, 255, 150))
    fps_text = small_font.render(f"FPS: {int(clock.get_fps())}", True, (200, 200, 200))
    screen.blit(health_text, (10, 10))
    screen.blit(ai_text, (10, 30))
    screen.blit(score_text, (WINDOW_WIDTH - 150, 10))
    screen.blit(fps_text, (WINDOW_WIDTH - 100, 30))

    pygame.display.flip()

    if player.health <= 0:
        print(f"💀 Game Over — Final Score: {score}")
        running = False

pygame.quit()
