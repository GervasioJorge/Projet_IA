
import pygame
import cv2
import mediapipe as mp
import numpy as np
import random

# --------------------- INITIALISATION ---------------------
pygame.init()
pygame.mixer.init()  # pour la musique

mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

SCREEN_WIDTH, SCREEN_HEIGHT = 600, 400
WHITE, BLACK, RED, BLUE = (255,255,255), (0,0,0), (255,0,0), (0,120,255)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Shooter Game - Hand Control (2 joueurs)")

PLAYER_WIDTH, PLAYER_HEIGHT = 50, 30
BULLET_WIDTH, BULLET_HEIGHT = 5, 10
ENEMY_WIDTH, ENEMY_HEIGHT = 40, 30

# J1 & J2
player1 = pygame.Rect(SCREEN_WIDTH // 3, SCREEN_HEIGHT - 50, PLAYER_WIDTH, PLAYER_HEIGHT)
player2 = pygame.Rect(2 * SCREEN_WIDTH // 3, SCREEN_HEIGHT - 50, PLAYER_WIDTH, PLAYER_HEIGHT)

bullets_p1, bullets_p2, enemies = [], [], []

bullet_speed = 8
enemy_speed = 2
score = 0
# anti-spam pour tir : un "pince" -> une balle
can_shoot_p1 = True
can_shoot_p2 = True

font = pygame.font.SysFont("Arial", 30)

# Charger les images
background_img = pygame.image.load("background.png")
player_img = pygame.image.load("player.png")
enemy_img = pygame.image.load("enemy.png")

# Redimensionnement
player_img = pygame.transform.scale(player_img, (PLAYER_WIDTH, PLAYER_HEIGHT))
enemy_img = pygame.transform.scale(enemy_img, (ENEMY_WIDTH, ENEMY_HEIGHT))

# Image joueur 2 : si player2.png existe, on l'utilise; sinon on teinte le sprite J1
try:
    player2_img = pygame.image.load("player2.png")
    player2_img = pygame.transform.scale(player2_img, (PLAYER_WIDTH, PLAYER_HEIGHT))
except Exception:
    player2_img = player_img.copy()
    player2_img.fill(BLUE, special_flags=pygame.BLEND_MULT)

# Musique de fond (boucle). Si absente ou illisible, le jeu continue.
try:
    pygame.mixer.music.load("music.mp3")  # remplace par .ogg si besoin
    pygame.mixer.music.set_volume(0.6)
    pygame.mixer.music.play(-1)  # -1 => boucle infinie
except Exception as e:
    print("Musique non trouvée ou illisible :", e)

# --------------------- FONCTIONS ---------------------

def clamp_player(p):
    """Empêche un joueur de sortir à gauche/droite de l'écran."""
    p.x = max(0, min(p.x, SCREEN_WIDTH - PLAYER_WIDTH))

def detect_pinch_thumb_index(hand_lm):
    """Détecte si le pouce touche l'index (pince)."""
    thumb = hand_lm.landmark[4]
    index_finger = hand_lm.landmark[8]
    dist = np.hypot(thumb.x - index_finger.x, thumb.y - index_finger.y)
    return dist < 0.05  # seuil

def get_hands_info(hands_processor):
    """Retourne une liste d'infos mains: [{'lm': landmarks, 'x': float, 'label': 'Left'/'Right'/'Unknown'}]"""
    ret, frame = cap.read()
    if not ret:
        return [], None

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_processor.process(rgb)

    infos = []
    if results.multi_hand_landmarks:
        # multi_handedness correspond indice à indice aux landmarks
        for i, lm in enumerate(results.multi_hand_landmarks):
            label = "Unknown"
            if results.multi_handedness and i < len(results.multi_handedness):
                if results.multi_handedness[i].classification:
                    label = results.multi_handedness[i].classification[0].label  # 'Left' ou 'Right'
            x = lm.landmark[0].x  # poignet (coordonnée normalisée)
            infos.append({"lm": lm, "x": x, "label": label})

    # Affiche le flux (optionnel)
    cv2.imshow("Camera", frame)
    cv2.waitKey(1)
    return infos, results

def split_groups_left_right(hands_infos):
    """
    Sépare les mains en 2 groupes (gauche/droite de l'image) en fonction de la médiane des x.
    - Groupe gauche => Joueur 1
    - Groupe droite => Joueur 2
    """
    if not hands_infos:
        return [], []
    xs = [h["x"] for h in hands_infos]
    thresh = float(np.median(xs))
    g_left = [h for h in hands_infos if h["x"] <= thresh]
    g_right = [h for h in hands_infos if h["x"] > thresh]

    # Si 1 seule main, l'associer au côté correspondant à sa x vs 0.5
    if len(hands_infos) == 1:
        g_left = hands_infos if hands_infos[0]["x"] < 0.5 else []
        g_right = [] if g_left else hands_infos

    return g_left, g_right

def pick_movement_hand(group):
    """Choisit la main pour le déplacement: priorité 'Right', sinon la première du groupe."""
    if not group:
        return None
    rights = [h for h in group if h["label"] == "Right"]
    return (rights[0] if rights else group[0])["lm"]

def group_shoots_now(group):
    """
    Le tir est déclenché par une pince de la main 'Left' du groupe.
    Fallback: si aucune 'Left' n'est visible, on autorise une pince de n'importe quelle main du groupe.
    """
    if not group:
        return False
    lefts = [h for h in group if h["label"] == "Left"]
    candidates = lefts if lefts else group
    return any(detect_pinch_thumb_index(h["lm"]) for h in candidates)

def spawn_enemy():
    if random.random() < 0.02:
        x = random.randint(0, SCREEN_WIDTH - ENEMY_WIDTH)
        enemies.append(pygame.Rect(x, 0, ENEMY_WIDTH, ENEMY_HEIGHT))

def move_bullets_for(bullets_list):
    """Déplacement + collisions balle->ennemi, pour la liste de balles donnée."""
    global score
    for b in bullets_list[:]:
        b.y -= bullet_speed
        if b.y < 0:
            bullets_list.remove(b)
            continue
        for e in enemies[:]:
            if b.colliderect(e):
                enemies.remove(e)
                bullets_list.remove(b)
                score += 10
                break

def move_enemies():
    for e in enemies[:]:
        e.y += enemy_speed
        if e.y > SCREEN_HEIGHT:
            enemies.remove(e)

def draw():
    screen.blit(background_img, (0, 0))

    # joueurs
    screen.blit(player_img, (player1.x, player1.y))
    screen.blit(player2_img, (player2.x, player2.y))

    # balles J1 (blanches) / J2 (rouges)
    for b in bullets_p1:
        pygame.draw.rect(screen, WHITE, b)
    for b in bullets_p2:
        pygame.draw.rect(screen, RED, b)

    # ennemis
    for e in enemies:
        screen.blit(enemy_img, (e.x, e.y))

    # score
    screen.blit(font.render(f"Score: {score}", True, WHITE), (10, 10))

    pygame.display.update()

# --------------------- GAME LOOP ---------------------
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
clock = pygame.time.Clock()
running = True

while running:
    hands_infos, results = get_hands_info(hands)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Regrouper les mains en 2 groupes: gauche => J1, droite => J2
    group_left, group_right = split_groups_left_right(hands_infos)

    # --- Déplacement J1 (main "Right" du groupe gauche sinon n'importe laquelle)
    move_hand_p1 = pick_movement_hand(group_left)
    if move_hand_p1:
        x = move_hand_p1.landmark[9].x  # base de l'index
        player1.x = int(x * SCREEN_WIDTH - PLAYER_WIDTH / 2)
        clamp_player(player1)

    # --- Déplacement J2 (main "Right" du groupe droit sinon n'importe laquelle)
    move_hand_p2 = pick_movement_hand(group_right)
    if move_hand_p2:
        x = move_hand_p2.landmark[9].x
        player2.x = int(x * SCREEN_WIDTH - PLAYER_WIDTH / 2)
        clamp_player(player2)

    # --- Tir J1: pince main gauche du groupe gauche (anti-spam)
    if group_shoots_now(group_left):
        if can_shoot_p1:
            bullets_p1.append(pygame.Rect(player1.centerx - BULLET_WIDTH // 2, player1.y, BULLET_WIDTH, BULLET_HEIGHT))
            can_shoot_p1 = False
    else:
        can_shoot_p1 = True

    # --- Tir J2: pince main gauche du groupe droit (anti-spam)
    if group_shoots_now(group_right):
        if can_shoot_p2:
            bullets_p2.append(pygame.Rect(player2.centerx - BULLET_WIDTH // 2, player2.y, BULLET_WIDTH, BULLET_HEIGHT))
            can_shoot_p2 = False
    else:
        can_shoot_p2 = True

    # Spawns & mouvements
    spawn_enemy()
    move_enemies()
    move_bullets_for(bullets_p1)
    move_bullets_for(bullets_p2)

    # --- Collisions ennemis<->joueurs (pas de collision entre joueurs)
    for e in enemies:
        if e.colliderect(player1) or e.colliderect(player2):
            running = False

    draw()
    clock.tick(60)

cap.release()
cv2.destroyAllWindows()
pygame.quit()
