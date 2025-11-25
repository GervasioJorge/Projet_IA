import pygame
import cv2
import mediapipe as mp
import numpy as np
import random

# --------------------- INITIALISATION ---------------------
pygame.init()
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

SCREEN_WIDTH, SCREEN_HEIGHT = 600, 400
WHITE, BLACK, RED = (255,255,255), (0,0,0), (255,0,0)


screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Shooter Game - Hand Control")

PLAYER_WIDTH, PLAYER_HEIGHT = 50, 30
BULLET_WIDTH, BULLET_HEIGHT = 5, 10
ENEMY_WIDTH, ENEMY_HEIGHT = 40, 30

player = pygame.Rect(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50, PLAYER_WIDTH, PLAYER_HEIGHT)
bullets, enemies = [], []

player_speed = 8
bullet_speed = 8
enemy_speed = 2
score = 0
can_shoot = True   # anti-spam pour tir

font = pygame.font.SysFont("Arial", 30)

# Charger les images
background_img = pygame.image.load("background.png")
player_img = pygame.image.load("player.png")
enemy_img = pygame.image.load("enemy.png")

# Redimensionnement et correction
player_img = pygame.transform.scale(player_img, (PLAYER_WIDTH, PLAYER_HEIGHT))
enemy_img = pygame.transform.scale(enemy_img, (ENEMY_WIDTH, ENEMY_HEIGHT))

# --------------------- FUNCTIONS ---------------------

def detect_pinch_thumb_index(hand):
    """Détecte si le pouce touche l'index"""
    thumb = hand.landmark[4]
    index_finger = hand.landmark[8]
    dist = np.hypot(thumb.x - index_finger.x, thumb.y - index_finger.y)
    return dist < 0.05  # seuil

def detect_hands():
    ret, frame = cap.read()
    if not ret:
        return None, None, None

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    main_gauche, main_droite = None, None

    if results.multi_hand_landmarks:
        hands_sorted = sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x)
        if len(hands_sorted) >= 2:
            main_gauche = hands_sorted[0]
            main_droite = hands_sorted[1]

    cv2.imshow("Camera", frame)
    cv2.waitKey(1)
    return main_gauche, main_droite, results

def spawn_enemy():
    if random.random() < 0.02:
        x = random.randint(0, SCREEN_WIDTH - ENEMY_WIDTH)
        enemies.append(pygame.Rect(x, 0, ENEMY_WIDTH, ENEMY_HEIGHT))

def move_bullets():
    global score
    for b in bullets[:]:
        b.y -= bullet_speed
        if b.y < 0:
            bullets.remove(b)
        for e in enemies[:]:
            if b.colliderect(e):
                enemies.remove(e)
                bullets.remove(b)
                score += 10

def move_enemies():
    for e in enemies[:]:
        e.y += enemy_speed
        if e.y > SCREEN_HEIGHT:
            enemies.remove(e)

def draw():
    screen.blit(background_img, (0, 0))
    screen.blit(player_img, (player.x, player.y))
    for b in bullets:
        pygame.draw.rect(screen, WHITE, b)
    for e in enemies:
        screen.blit(enemy_img, (e.x, e.y))
    screen.blit(font.render(f"Score: {score}", True, WHITE), (10, 10))
    pygame.display.update()

# --------------------- GAME LOOP ---------------------
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
clock = pygame.time.Clock()
running = True

while running:

    main_gauche, main_droite, results = detect_hands()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # ------------------- MOVE PLAYER (Main DROITE) -------------------
    if main_droite:
        x = main_droite.landmark[9].x
        player.x = int(x * SCREEN_WIDTH - PLAYER_WIDTH / 2)

    # ------------------- SHOOT (Main GAUCHE = pouce + index) -------------------
    if main_gauche:
        if detect_pinch_thumb_index(main_gauche):
            if can_shoot:
                bullets.append(pygame.Rect(player.centerx, player.y, BULLET_WIDTH, BULLET_HEIGHT))
                can_shoot = False
        else:
            can_shoot = True

    spawn_enemy()
    move_enemies()
    move_bullets()

    for e in enemies:
        if e.colliderect(player):
            running = False

    draw()
    clock.tick(60)

cap.release()
cv2.destroyAllWindows()
pygame.quit()
