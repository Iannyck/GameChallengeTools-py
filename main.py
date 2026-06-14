import gamedifficulty as GD

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# This is a demo of the algorithm presented in the paper A Comprehensive Model of Automated Evaluation of
# Difficulty in Platformer Games. This project includes helper function to extrapolate information images and data
# from mario levels to then process into a difficulty curve.
# You can find an example implementation in python of the actual algorithm in gamedifficulty/Processing.py,
# function CalculateDifficulty

def save_path_img(path: list[tuple[int, int]], name):
    pathImg = levelImage.copy()
    for i in range(1, len(path)):
        prev = (int(round(path[i - 1][0])), int(round(path[i - 1][1])))
        curr = (int(round(path[i][0])), int(round(path[i][1])))
        cv.line(pathImg, prev, curr, (0, 255, 255), 2)
        cv.circle(pathImg, curr, 3, (0, 255, 255), -1)
    if path:
        start = (int(round(path[0][0])), int(round(path[0][1])))
        cv.circle(pathImg, start, 4, (0, 0, 255), -1)
        goal = (int(round(path[-1][0])), int(round(path[-1][1])))
        cv.circle(pathImg, goal, 4, (0, 255, 0), -1)
    cv.imwrite(f"ressources/{level}/{name}.png", pathImg)

def show_graph_for_path(path: list[tuple[int, int]], normalizedReachMap: cv.Mat[cv.CV_8U]):

    # Example path usage for CreatePathAccessibilityValue
    pathValue = GD.Processing.CreatePathAccessibilityValue(path, normalizedReachMap)

    plt.figure(figsize=(10, 3))
    plt.plot(pathValue, label="Path Accessibility Value")
    plt.title(f"Path accessibility value curve for level {level}")
    plt.xlabel("Level X coordinate")
    plt.ylabel("Possible access to this x value")
    plt.ylim(np.min(pathValue) * 1.1, np.max(pathValue) * 1.1)
    plt.grid(True)
    plt.legend()
    plt.show()

    # Example path usage for CreatePathAccessibilityVariance
    pathVariance = GD.Processing.CreatePathAccessibilityVariance(path, normalizedReachMap)

    plt.figure(figsize=(10, 3))
    plt.plot(pathVariance, label="Path Accessibility Variance")
    plt.title(f"Path accessibility variance curve for level {level}")
    plt.xlabel("Level X coordinate")
    plt.ylabel("Variance between previous x accessiblity")
    plt.ylim(np.min(pathVariance) * 1.1, np.max(pathVariance) * 1.1)
    plt.grid(True)
    plt.legend()
    plt.show()


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
normalizedReach = normalizedReachMap / 100.0
cv.imwrite(f"ressources/{level}/normalized_reach.png", (normalizedReach * 255).astype(np.uint8))

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


start = (50,195)
end = (3175,175)

path1 = GD.Processing.CreateReachAStarPath(start, end, normalizedReach, staticDanger ,True)
path2 = GD.Processing.CreateSmoothHighPath(start, end, normalizedReach, staticDanger ,True)

save_path_img(path1, "path_1")
show_graph_for_path(path1, normalizedReachMap)
show_graph_for_path(path1, accessibleDanger)

save_path_img(path2, "path_2")
show_graph_for_path(path2, normalizedReachMap)
show_graph_for_path(path2, accessibleDanger)