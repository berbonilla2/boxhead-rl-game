import pygame
import random
import math
import time

pygame.init()

# ==============================
# 🧩 CONFIGURATION
# ==============================
WINDOW_WIDTH, WINDOW_HEIGHT = 640, 480
FPS = 60

# Speeds (tweakable)
PLAYER_SPEED = 4.0
ZOMBIE_SPEED = 1.2
DEMON_SPEED = 1.4

# Spawning
ENEMY_SPAWN_RATE = 0.003   # Probability per frame
SPAWN_MIN_DISTANCE = 50     # Min distance from center

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
LASER_COLOR = (255, 255, 255, 60)  # semi-transparent white

# ==============================
# 🧠 HELPER FUNCTIONS
# ==============================
def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def random_spawn_position():
    """Spawn outside the 50px radius from center."""
    while True:
        x, y = random.randint(0, WINDOW_WIDTH), random.randint(0, WINDOW_HEIGHT)
        if math.hypot(x - WINDOW_WIDTH / 2, y - WINDOW_HEIGHT / 2) > SPAWN_MIN_DISTANCE:
            return x, y

# ==============================
# 🎮 CLASSES
# ==============================
class Player:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.radius = 12
        self.speed = PLAYER_SPEED
        self.health = 100
        self.last_shot = 0
        self.shot_delay = 0.25
        self.direction = (0, -1)  # facing up

    def move(self, keys):
        moved = False
        if keys[pygame.K_w]:
            self.y -= self.speed
            self.direction = (0, -1)
            moved = True
        elif keys[pygame.K_s]:
            self.y += self.speed
            self.direction = (0, 1)
            moved = True
        elif keys[pygame.K_a]:
            self.x -= self.speed
            self.direction = (-1, 0)
            moved = True
        elif keys[pygame.K_d]:
            self.x += self.speed
            self.direction = (1, 0)
            moved = True

        # ✅ Clamp position inside screen boundaries
        self.x = max(self.radius, min(self.x, WINDOW_WIDTH - self.radius))
        self.y = max(self.radius, min(self.y, WINDOW_HEIGHT - self.radius))

        return moved


    def shoot(self, bullets):
        now = time.time()
        if now - self.last_shot >= self.shot_delay:
            dx, dy = self.direction
            bullets.append(Bullet(self.x + dx * 15, self.y + dy * 15, 8, self.direction, "player"))
            self.last_shot = now

    def draw(self, surface):
        # Body
        pygame.draw.circle(surface, BLUE, (int(self.x), int(self.y)), self.radius)

        # Laser sight
        fx, fy = self.x + self.direction[0] * 200, self.y + self.direction[1] * 200
        laser_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        pygame.draw.line(laser_surface, LASER_COLOR, (self.x, self.y), (fx, fy), 2)
        surface.blit(laser_surface, (0, 0))

        # Facing direction tip
        fx_tip, fy_tip = self.x + self.direction[0] * 15, self.y + self.direction[1] * 15
        pygame.draw.line(surface, YELLOW, (self.x, self.y), (fx_tip, fy_tip), 3)

        # Health bar
        pygame.draw.rect(surface, RED, (self.x - 15, self.y - 25, 30, 4))
        pygame.draw.rect(surface, GREEN, (self.x - 15, self.y - 25, 30 * (self.health / 100), 4))


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

    def draw(self, surface):
        color = WHITE if self.owner == "player" else ORANGE
        pygame.draw.circle(surface, color, (int(self.x), int(self.y)), self.radius)


class Enemy:
    def __init__(self, x, y, color, speed, radius=10, health=30):
        self.x, self.y = x, y
        self.color = color
        self.speed = speed
        self.radius = radius
        self.health = health

    def move_toward(self, target_x, target_y):
        dx, dy = target_x - self.x, target_y - self.y
        dist = math.hypot(dx, dy)
        if dist > 0:
            dx, dy = dx / dist, dy / dist
            self.x += dx * self.speed
            self.y += dy * self.speed

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)


class Demon(Enemy):
    def __init__(self, x, y):
        super().__init__(x, y, PURPLE, DEMON_SPEED, radius=14, health=50)
        self.last_shot = time.time()

    def shoot(self, bullets, target):
        now = time.time()
        if now - self.last_shot >= 5:
            for _ in range(3):
                dx, dy = target.x - self.x, target.y - self.y
                dist = math.hypot(dx, dy)
                if dist == 0:
                    continue
                dx, dy = dx / dist, dy / dist
                spread = random.uniform(-0.2, 0.2)
                bullets.append(Bullet(self.x, self.y, 5, (dx + spread, dy + spread), "demon"))
            self.last_shot = now


class Barrel:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.radius = 10
        self.exploded = False

    def draw(self, surface):
        pygame.draw.circle(surface, ORANGE, (int(self.x), int(self.y)), self.radius)

    def explode(self, enemies):
        if not self.exploded:
            self.exploded = True
            pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), 30, 2)
            for e in enemies:
                if distance((self.x, self.y), (e.x, e.y)) < 30:
                    e.health -= 20


class Package:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.radius = 8

    def draw(self, surface):
        pygame.draw.circle(surface, GREEN, (int(self.x), int(self.y)), self.radius)

# ==============================
# 🚀 GAME INITIALIZATION
# ==============================
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Boxhead 2D - RL Ready Version")
clock = pygame.time.Clock()

player = Player(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
bullets = []
zombies = [Enemy(*random_spawn_position(), RED, ZOMBIE_SPEED) for _ in range(5)]
demons = [Demon(*random_spawn_position()) for _ in range(2)]
barrels = [Barrel(*random_spawn_position()) for _ in range(3)]
packages = [Package(*random_spawn_position()) for _ in range(2)]

# ==============================
# 🎮 MAIN LOOP
# ==============================
running = True
while running:
    dt = clock.tick(FPS)
    screen.fill(BLACK)

    # Background grid
    for x in range(0, WINDOW_WIDTH, 32):
        pygame.draw.line(screen, GRAY, (x, 0), (x, WINDOW_HEIGHT))
    for y in range(0, WINDOW_HEIGHT, 32):
        pygame.draw.line(screen, GRAY, (0, y), (WINDOW_WIDTH, y))

    keys = pygame.key.get_pressed()
    player.move(keys)

    # Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            player.shoot(bullets)

    # Bullets
    for b in bullets[:]:
        b.update()
        if not (0 <= b.x <= WINDOW_WIDTH and 0 <= b.y <= WINDOW_HEIGHT):
            bullets.remove(b)
            continue
        if b.owner == "player":
            for e in zombies + demons:
                if distance((b.x, b.y), (e.x, e.y)) <= e.radius:
                    e.health -= 25
                    if b in bullets:
                        bullets.remove(b)
                    break
            for br in barrels[:]:
                if distance((b.x, b.y), (br.x, br.y)) <= br.radius:
                    br.explode(zombies + demons)
                    barrels.remove(br)
                    if b in bullets:
                        bullets.remove(b)
                    break
        elif b.owner == "demon":
            if distance((b.x, b.y), (player.x, player.y)) <= player.radius:
                player.health -= 15
                bullets.remove(b)

    # Enemies
    for z in zombies[:]:
        z.move_toward(player.x, player.y)
        if z.health <= 0:
            zombies.remove(z)
        elif distance((z.x, z.y), (player.x, player.y)) <= z.radius + player.radius:
            player.health -= 0.4

    for d in demons[:]:
        d.move_toward(player.x, player.y)
        d.shoot(bullets, player)
        if d.health <= 0:
            demons.remove(d)
        elif distance((d.x, d.y), (player.x, player.y)) <= d.radius + player.radius:
            player.health -= 0.6

    # Random spawns
    if random.random() < ENEMY_SPAWN_RATE:
        if random.random() < 0.8:
            zombies.append(Enemy(*random_spawn_position(), RED, ZOMBIE_SPEED))
        else:
            demons.append(Demon(*random_spawn_position()))

    # Packages
    for p in packages[:]:
        if distance((p.x, p.y), (player.x, player.y)) <= p.radius + player.radius:
            player.health = 100
            packages.remove(p)

    # Respawn barrels/packages
    if random.random() < 0.002:
        barrels.append(Barrel(*random_spawn_position()))
    if random.random() < 0.002:
        packages.append(Package(*random_spawn_position()))

    # Draw
    player.draw(screen)
    for b in bullets: b.draw(screen)
    for z in zombies: z.draw(screen)
    for d in demons: d.draw(screen)
    for br in barrels: br.draw(screen)
    for p in packages: p.draw(screen)

    pygame.display.flip()

    if player.health <= 0:
        print("Game Over!")
        running = False

pygame.quit()
