import gamedifficulty as GD

import json
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

# This is a demo of the algorithm presented in the paper A Comprehensive Model of Automated Evaluation of
# Difficulty in Platformer Games. This project includes helper function to extrapolate information images and data
# from mario levels to then process into a difficulty curve.
# You can find an example implementation in python of the actual algorithm in gamedifficulty/Processing.py,
# function CalculateDifficulty


def save_path_data(path_values: list[float], level_name, filename: str):
    """
    Sauvegarde les valeurs d'un chemin au format texte.
    Format : X-coordinate (échantillonné tous les 16 pixels) + valeur
    """
    folder = f"ressources/{level_name}"
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(f"{folder}/{filename}.txt", "w") as f:
        # En-tête (basé sur votre exemple)
        f.write("P W\n")

        # On parcourt les données avec un pas de 16
        for x_coord in range(0, len(path_values), 16):
            val = path_values[x_coord]
            # L'index x_coord correspond maintenant à la vraie position X sur l'image
            f.write(f"{x_coord} {val:.6g}\n")


def draw_path(image, path, color, thickness=2):
    """
    Dessine un chemin opaque sur l'image. Si le chemin repasse sur des pixels
    déjà colorés (différents du fond noir/transparent ou de l'image de base),
    les couleurs fusionnent sans transparence sur le reste du tracé.
    """
    if not path:
        return

    # 1. Créer un calque pour le chemin actuel (fond noir)
    layer = np.zeros_like(image)

    # 2. Dessiner le chemin actuel de manière totalement opaque sur ce calque
    for i in range(1, len(path)):
        prev = (int(round(path[i - 1][0])), int(round(path[i - 1][1])))
        curr = (int(round(path[i][0])), int(round(path[i][1])))
        cv.line(layer, prev, curr, color, thickness)
        cv.circle(layer, curr, 3, color, -1)

    # Points de départ (Rouge) et d'arrivée (Vert) fixes
    start = (int(round(path[0][0])), int(round(path[0][1])))
    cv.circle(layer, start, 5, (0, 0, 255), -1)
    goal = (int(round(path[-1][0])), int(round(path[-1][1])))
    cv.circle(layer, goal, 5, (0, 255, 0), -1)

    # 3. Créer des masques pour isoler les zones
    # Masque des pixels dessinés par le nouveau chemin
    new_path_mask = cv.cvtColor(layer, cv.COLOR_BGR2GRAY) > 0

    # Masque des pixels de l'image de destination qui ont déjà été modifiés par un ancien chemin
    # Note : On suppose ici que le fond "vide" de base ne correspond pas exactement à une couleur de chemin.
    # Pour être précis, on détecte ce qui n'est pas noir sur le calque accumulé ou modifié.
    # Si base_image contient le niveau de Mario en fond, on compare plutôt avec une image d'accumulation (voir save_all_paths)
    return layer


def save_all_paths(
    paths: list[list[tuple[int, int]]], level_name, base_image, type_image
):
    """
    Sauvegarde chaque chemin individuellement et une image regroupant tous les chemins.
    Pleine couleur partout, sauf aux intersections exactes où les couleurs se mélangent.
    """
    colors = [
        (0, 0, 255),
        (209, 177, 16),
        (209, 16, 132),
    ]

    folder = f"ressources/{level_name}"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # 1. Génération des images individuelles (pleine couleur sans fusion nécessaire)
    for idx, path in enumerate(paths):
        color = colors[idx % len(colors)]
        file_name = f"path_{idx + 1}_{type_image}"

        individual_img = base_image.copy()
        path_layer = np.zeros_like(individual_img)

        # Dessin standard direct
        for i in range(1, len(path)):
            prev = (int(round(path[i - 1][0])), int(round(path[i - 1][1])))
            curr = (int(round(path[i][0])), int(round(path[i][1])))
            cv.line(path_layer, prev, curr, color, 2)
            cv.circle(path_layer, curr, 3, color, -1)
        if path:
            cv.circle(
                path_layer,
                (int(round(path[0][0])), int(round(path[0][1]))),
                5,
                (0, 0, 255),
                -1,
            )
            cv.circle(
                path_layer,
                (int(round(path[-1][0])), int(round(path[-1][1]))),
                5,
                (0, 255, 0),
                -1,
            )

        # Application sur le fond du niveau
        mask = cv.cvtColor(path_layer, cv.COLOR_BGR2GRAY) > 0
        individual_img[mask] = path_layer[mask]
        cv.imwrite(f"{folder}/{file_name}.png", individual_img)

    # 2. Génération de l'image globale avec gestion de l'intersection de couleurs
    # On crée un calque d'accumulation pour collecter uniquement les tracés à part de l'image de fond
    paths_accumulation_layer = np.zeros_like(base_image)

    for idx, path in enumerate(paths):
        color = colors[idx % len(colors)]

        # On génère le calque isolé pour ce tracé spécifique
        current_layer = np.zeros_like(base_image)
        for i in range(1, len(path)):
            prev = (int(round(path[i - 1][0])), int(round(path[i - 1][1])))
            curr = (int(round(path[i][0])), int(round(path[i][1])))
            cv.line(current_layer, prev, curr, color, 2)
            cv.circle(current_layer, curr, 3, color, -1)
        if path:
            cv.circle(
                current_layer,
                (int(round(path[0][0])), int(round(path[0][1]))),
                5,
                (0, 0, 255),
                -1,
            )
            cv.circle(
                current_layer,
                (int(round(path[-1][0])), int(round(path[-1][1]))),
                5,
                (0, 255, 0),
                -1,
            )

        # Détection des masques binaires
        mask_current = cv.cvtColor(current_layer, cv.COLOR_BGR2GRAY) > 0
        mask_existing = cv.cvtColor(paths_accumulation_layer, cv.COLOR_BGR2GRAY) > 0

        # Intersection : Pixels où les deux calques se superposent
        intersection_mask = mask_current & mask_existing
        # Simple apport : Pixels du nouveau tracé qui arrivent sur une zone vide
        single_path_mask = mask_current & ~mask_existing

        # Application 1 : Zone classique -> On applique la pleine couleur
        paths_accumulation_layer[single_path_mask] = current_layer[single_path_mask]

        # Application 2 : Zone de superposition -> On fusionne les couleurs (Moyenne 50/50)
        if np.any(intersection_mask):
            blended_pixels = cv.addWeighted(
                paths_accumulation_layer, 0.5, current_layer, 0.5, 0
            )
            paths_accumulation_layer[intersection_mask] = blended_pixels[
                intersection_mask
            ]

    # Enfin, on fusionne le calque d'accumulation final sur l'image de fond du niveau de Mario
    combined_img = base_image.copy()
    final_mask = cv.cvtColor(paths_accumulation_layer, cv.COLOR_BGR2GRAY) > 0
    combined_img[final_mask] = paths_accumulation_layer[final_mask]

    cv.imwrite(f"{folder}/all_paths_{type_image}.png", combined_img)


def show_graph_for_path(
    path: list[tuple[int, int]],
    reach: cv.Mat[cv.CV_8U],
    danger: cv.Mat[cv.CV_8U],
    pathName,
):

    # Example path usage for CreatePathAccessibilityValue
    greenPathValue = GD.Processing.CreatePathAccessibilityValue(path, reach)
    save_path_data(greenPathValue, level, f"{pathName}_green_values")

    plt.figure(figsize=(10, 3))
    plt.plot(greenPathValue, label="Green reach value")
    plt.title(f"Green value graph for {pathName}")
    plt.xlabel("Level X coordinate")
    plt.ylabel("Green reach value")
    plt.ylim(0.0, 1.1)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"ressources/{level}/{pathName}_green_values")

    # Example path usage for CreatePathAccessibilityVariance
    greenPathVariance = GD.Processing.CreatePathAccessibilityVariance(path, reach)
    save_path_data(greenPathVariance, level, f"{pathName}_green_variation")

    plt.figure(figsize=(10, 3))
    plt.plot(greenPathVariance, label="Green reach variation")
    plt.title(f"Green variation graph for {pathName}")
    plt.xlabel("Level X coordinate")
    plt.ylabel("Green reach variation")
    plt.ylim(np.min(greenPathVariance) * 1.1, np.max(greenPathVariance) * 1.1)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"ressources/{level}/{pathName}_green_variation")

    # Example path usage for CreatePathAccessibilityValue
    redPathValue = GD.Processing.CreatePathAccessibilityValue(path, danger)
    save_path_data(redPathValue, level, f"{pathName}_red_values")

    plt.figure(figsize=(10, 3))
    plt.plot(redPathValue, label="Red reach value")
    plt.title(f"Red value graph for {pathName}")
    plt.xlabel("Level X coordinate")
    plt.ylabel("Red reach value")
    plt.ylim(0.0, 1.1)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"ressources/{level}/{pathName}_red_values")

    # Example path usage for CreatePathAccessibilityVariance
    redPathVariance = GD.Processing.CreatePathAccessibilityVariance(path, danger)
    save_path_data(redPathVariance, level, f"{pathName}_red_variation")

    plt.figure(figsize=(10, 3))
    plt.plot(redPathVariance, label="Red reach variation")
    plt.title(f"Red variation graph for {pathName}")
    plt.xlabel("Level X coordinate")
    plt.ylabel("Red reach variation")
    plt.ylim(np.min(redPathVariance) * 1.1, np.max(redPathVariance) * 1.1)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"ressources/{level}/{pathName}_red_variation")

    mergeValue = greenPathValue * redPathValue
    save_path_data(mergeValue, level, f"{pathName}_red_and_green_value")

    plt.figure(figsize=(10, 3))
    plt.plot(mergeValue, label="Green and Red values")
    plt.title(f"Green and Red values graph for {pathName}")
    plt.xlabel("Level X coordinate")
    plt.ylabel("Green and Red values")
    plt.ylim(0.0, 1.1)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"ressources/{level}/{pathName}_red_and_green_value")

    mergeVariation = greenPathVariance * redPathVariance
    save_path_data(mergeVariation, level, f"{pathName}_red_and_green_variation")

    plt.figure(figsize=(10, 3))
    plt.plot(mergeVariation, label="Green and Red variation")
    plt.title(f"Green and Red variation graph for {pathName}")
    plt.xlabel("Level X coordinate")
    plt.ylabel("Green and Red variation")
    plt.ylim(np.min(mergeVariation) * 1.1, np.max(mergeVariation) * 1.1)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"ressources/{level}/{pathName}_red_and_green_variation")


# Load the level image
# level = "Niveau_6_3"
level = "Niveau_1_1"
levelImage = cv.imread(f"ressources/{level}/level.png")

# cv.imshow("Level", levelImage)

# Load the sprite set (ground, enemies, etc)
spriteSet = GD.Classes.SpriteSet("ressources/Sprite")

# Detect all ground and pipe tiles that form the collisions of the level.
collisionPositions = GD.Detection.DetectPatternMulti(
    levelImage, spriteSet.GetCollisionsTextures(), 0.85
)
collisionPositions += GD.Detection.DetectPatternMulti(
    levelImage, spriteSet.GetPipesTextures(), 0.85
)

jumpBoardPositions = GD.Detection.DetectPatternMulti(
    levelImage, spriteSet.GetJumpBoardTextures(), 0.85
)
collisionPositions += jumpBoardPositions

# moving platforms must be processed a bit differently.
passthroughPositions = GD.Processing.CreateMovingPlatform(
    levelImage,
    spriteSet.GetPlatformsTextures(),
    spriteSet.GetBalancePointsLeft(),
    spriteSet.GetBalancePointsRight(),
)

# Transform all positions into an image mask
collisionMask = GD.Processing.CreateMaskFromPatternResult(
    collisionPositions, levelImage.shape[:2]
)

# Create static danger map (holes)
staticDanger = GD.Processing.CreateStaticDanger(collisionMask)

cv.imwrite(f"ressources/{level}/staticDanger.png", staticDanger * 255)

# Mark all pixels mario can (theoretically) reach
reach = GD.Processing.CreateReachTextureFromPatternResults(
    levelImage.shape[:2],
    [
        (collisionPositions + passthroughPositions, int(GD.Constants.jumpHeight)),
        (jumpBoardPositions, int(GD.Constants.jumpBoardHeight)),
    ],
)

normalizedReachMap = GD.Processing.CreateReachNormalizedTexture(reach, collisionMask)
cv.imwrite(
    f"ressources/{level}/normalized_reach.png",
    (((normalizedReachMap * 100.0).astype(np.uint8) / 100) * 255).astype(np.uint8),
)

enemyDanger = np.zeros(levelImage.shape[:2], dtype=np.uint8)
enemyDetections = {}
# Create enemy danger map and collect enemy detections per type
for enemyType in GD.Types.EnemyType.GetAllTypes():
    # Find all enemies in the level
    enemyPositions = GD.Detection.DetectPatternMulti(
        levelImage, spriteSet.GetEnemyTextures(enemyType), 0.85
    )
    enemyDetections[enemyType] = enemyPositions

    # Find their possible positions
    ed = GD.Processing.CreateDisplacementTexture(
        enemyType, enemyPositions, collisionMask
    )

    # merge static and enemy danger
    enemyDanger = np.maximum(ed, enemyDanger)

# Merge enemy danger and static danger
danger = np.maximum(staticDanger, enemyDanger)

# Create a pheromone map from enemy displacement zones
enemyPheromoneMap = GD.Processing.CreateEnemyPheromoneMap(
    enemyDetections, collisionMask
)
cv.imwrite(
    f"ressources/{level}/enemy_pheromone_map.png",
    (np.clip(enemyPheromoneMap, 0.0, 1.0) * 255).astype(np.uint8),
)

accessibleDanger = GD.Processing.CreateAccessibleDangerMap(
    collisionMask, enemyDetections, normalizedReachMap
)

# Optionnel : Sauvegarder pour visualiser
cv.imwrite(
    f"ressources/{level}/accessible_danger_map.png",
    (np.clip(accessibleDanger, 0.0, 1.0) * 255).astype(np.uint8),
)

# save reach image, collision image and danger image
cv.imwrite(f"ressources/{level}/reach.png", reach * 255)
cv.imwrite(f"ressources/{level}/collision.png", collisionMask * 255)
cv.imwrite(f"ressources/{level}/danger.png", danger * 255)
cv.imwrite(f"ressources/{level}/enemyDanger.png", enemyDanger * 255)

json_path = f"ressources/{level}/paths.json"
raw_paths = []

if os.path.exists(json_path):
    with open(json_path, "r") as f:
        raw_paths = json.load(f)
else:
    print(
        f"⚠️ Aucun fichier {json_path} trouvé. Lance l'interface graphique pour en créer un."
    )

paths_to_save = []

for path_coords in raw_paths:
    # Le JSON stocke des listes [x, y], la fonction attend des tuples (x, y)
    formatted_points = [(point[0], point[1]) for point in path_coords]

    # Calcul du pathfinding pour ces points
    computed_path = GD.Processing.CreateMultiPointAStarPath(
        formatted_points, normalizedReachMap, True
    )

    # On ne sauvegarde que si le chemin a pu être résolu
    if computed_path:
        paths_to_save.append(computed_path)

# print(path3)

height, width = normalizedReachMap.shape
overlay_img = np.zeros((height, width, 3), dtype=np.uint8)

# Conversion des matrices normalisées (0-1) en intensités de pixels (0-255)
# Pour s'assurer qu'elles sont bien entre 0 et 1, on peut utiliser np.clip si nécessaire
reach_intensity = (np.clip(normalizedReachMap, 0.0, 1.0) * 255).astype(np.uint8)
danger_intensity = (np.clip(accessibleDanger, 0.0, 1.0) * 255).astype(np.uint8)

# Canal Rouge : intensité de accessibleDanger
overlay_img[:, :, 2] = danger_intensity

# Canal Vert : intensité de normalizedReachMap
overlay_img[:, :, 1] = reach_intensity

# Appelez la fonction après avoir défini vos paths
save_all_paths(paths_to_save, level, overlay_img, "overlay")
save_all_paths(paths_to_save, level, levelImage, "level")

path_name = ["Basic Path", "Higher Path"]

for i in range(len(paths_to_save)):
    show_graph_for_path(
        paths_to_save[i], normalizedReachMap, accessibleDanger, path_name[i]
    )

cv.imwrite(f"ressources/{level}/reach_danger_overlay.png", overlay_img)
