import gamedifficulty as GD

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
    Format : X-coordinate (pas de 16) + valeur
    """
    folder = f"ressources/{level_name}"
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    with open(f"{folder}/{filename}.txt", "w") as f:
        # En-tête (basé sur votre exemple)
        f.write("P W\n")
        
        # Le pas est de 16 pixels
        for i, val in enumerate(path_values):
            x_coord = (i + 1) * 16
            # On écrit avec une précision standard
            f.write(f"{x_coord} {val:.6g}\n")

def draw_path(image, path, color, thickness=2):
    """Dessine un chemin sur une image donnée."""
    for i in range(1, len(path)):
        prev = (int(round(path[i - 1][0])), int(round(path[i - 1][1])))
        curr = (int(round(path[i][0])), int(round(path[i][1])))
        cv.line(image, prev, curr, color, thickness)
        cv.circle(image, curr, 3, color, -1)
    
    if path:
        start = (int(round(path[0][0])), int(round(path[0][1])))
        cv.circle(image, start, 5, (0, 0, 255), -1) # Point de départ en rouge fixe
        goal = (int(round(path[-1][0])), int(round(path[-1][1])))
        cv.circle(image, goal, 5, (0, 255, 0), -1)   # Point d'arrivée en vert fixe

def save_all_paths(paths: list[list[tuple[int, int]]], level_name, base_image, type_image):
    """
    Sauvegarde chaque chemin individuellement et une image regroupant tous les chemins.
    """
    # Génération de couleurs aléatoires uniques pour chaque chemin
    colors = [
        (0, 0, 255),
        (209, 177, 16),
        (209, 16, 132),
    ]

    # Création de l'image globale
    combined_img = base_image.copy()

    for idx, path in enumerate(paths):
        color = colors[idx]
        file_name = f"path_{idx + 1}_{type_image}"
        
        # 1. Image individuelle
        individual_img = base_image.copy()
        draw_path(individual_img, path, color)
        cv.imwrite(f"ressources/{level_name}/{file_name}.png", individual_img)
        
        # 2. Ajout à l'image globale
        draw_path(combined_img, path, color)
        
    # Sauvegarde de l'image globale
    cv.imwrite(f"ressources/{level_name}/all_paths_{type_image}.png", combined_img)

def show_graph_for_path(path: list[tuple[int, int]], reach: cv.Mat[cv.CV_8U], danger: cv.Mat[cv.CV_8U], pathName):

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
collisionPositions = GD.Detection.DetectPatternMulti(levelImage, spriteSet.GetCollisionsTextures(), 0.85)
collisionPositions += GD.Detection.DetectPatternMulti(levelImage, spriteSet.GetPipesTextures(), 0.85)

jumpBoardPositions = GD.Detection.DetectPatternMulti(levelImage, spriteSet.GetJumpBoardTextures(), 0.85)
collisionPositions += jumpBoardPositions

# moving platforms must be processed a bit differently.
passthroughPositions = GD.Processing.CreateMovingPlatform(levelImage, spriteSet.GetPlatformsTextures(), spriteSet.GetBalancePointsLeft(), spriteSet.GetBalancePointsRight())

# Transform all positions into an image mask
collisionMask = GD.Processing.CreateMaskFromPatternResult(collisionPositions, levelImage.shape[:2])

# Create static danger map (holes)
staticDanger = GD.Processing.CreateStaticDanger(collisionMask)

cv.imwrite(f"ressources/{level}/staticDanger.png", staticDanger * 255)

# Mark all pixels mario can (theoretically) reach
reach = GD.Processing.CreateReachTextureFromPatternResults(
    levelImage.shape[:2],
    [
        (collisionPositions + passthroughPositions, int(GD.Constants.jumpHeight)),
        (jumpBoardPositions, int(GD.Constants.jumpBoardHeight))
    ],
)

normalizedReachMap = GD.Processing.CreateReachNormalizedTexture(reach, collisionMask)
cv.imwrite(f"ressources/{level}/normalized_reach.png", (((normalizedReachMap * 100.0).astype(np.uint8) / 100) * 255).astype(np.uint8))

enemyDanger = np.zeros(levelImage.shape[:2], dtype=np.uint8)
enemyDetections = {}
# Create enemy danger map and collect enemy detections per type
for enemyType in GD.Types.EnemyType.GetAllTypes():
    # Find all enemies in the level
    enemyPositions = GD.Detection.DetectPatternMulti(levelImage, spriteSet.GetEnemyTextures(enemyType), 0.85)
    enemyDetections[enemyType] = enemyPositions

    # Find their possible positions
    ed = GD.Processing.CreateDisplacementTexture(enemyType, enemyPositions, collisionMask)

    # merge static and enemy danger
    enemyDanger = np.maximum(ed, enemyDanger)

# Merge enemy danger and static danger
danger = np.maximum(staticDanger, enemyDanger)

# Create a pheromone map from enemy displacement zones
enemyPheromoneMap = GD.Processing.CreateEnemyPheromoneMap(enemyDetections, collisionMask)
cv.imwrite(f"ressources/{level}/enemy_pheromone_map.png", (np.clip(enemyPheromoneMap, 0.0, 1.0) * 255).astype(np.uint8))

accessibleDanger = GD.Processing.CreateAccessibleDangerMap(
    collisionMask,
    enemyDetections,
    normalizedReachMap
)

# Optionnel : Sauvegarder pour visualiser
cv.imwrite(f"ressources/{level}/accessible_danger_map.png", 
           (np.clip(accessibleDanger, 0.0, 1.0) * 255).astype(np.uint8))

# save reach image, collision image and danger image
cv.imwrite(f"ressources/{level}/reach.png", reach * 255)
cv.imwrite(f"ressources/{level}/collision.png", collisionMask * 255)
cv.imwrite(f"ressources/{level}/danger.png", danger * 255)
cv.imwrite(f"ressources/{level}/enemyDanger.png", enemyDanger * 255)


start = (50,199)
end = (3175,175)

paths_to_save = [
    GD.Processing.CreateMultiPointAStarPath([
        (50, 199),
        (465, 167),
        (520, 199),
        (625, 151),
        (690, 90),
        (767, 135),
        (840, 199),
        (943, 135),
        (980, 75),
        (1030, 119),
        (1100, 199),
        (1120, 175),
        (1160, 199),
        (1375, 199),
        (1400, 175),
        (1425, 199),
        (2200, 135),
        (2225, 115),
        (2245, 135),
        (2320, 199),
        (2445, 135),
        (2460, 115),
        (2485, 135),
        (3175, 175),
    ], normalizedReachMap, True),
    GD.Processing.CreateMultiPointAStarPath([
        (50, 199),
        (265, 135),
        (295, 110),
        (325, 135),
        (415, 120),
        (465, 167),
        (520, 199),
        (625, 151),
        (690, 90),
        (767, 135),
        (840, 199),
        (943, 135),
        (980, 75),
        (1030, 119),
        (1100, 199),
        (1120, 175),
        (1160, 199),
        (1235, 135),
        (1285, 71),
        (1405, 71),
        (1430, 45),
        (1460, 71),
        (1515, 71),
        (1600, 135),
        (1700, 135),
        (1750, 71),
        (1800, 135),
        (1900, 135),
        (1940, 71),
        (1982, 71),
        (2010, 45),
        (2050, 71),
        (2100, 71),
        (2200, 135),
        (2225, 115),
        (2245, 135),
        (2320, 199),
        (2445, 135),
        (2460, 115),
        (2485, 135),
        (2625, 165),
        (2700, 135),
        (2880, 160),
        (3030, 71),
        (3175, 175),
    ], normalizedReachMap, True)
]

#print(path3)

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
    show_graph_for_path(paths_to_save[i], normalizedReachMap, accessibleDanger, path_name[i])

cv.imwrite(f"ressources/{level}/reach_danger_overlay.png", overlay_img)