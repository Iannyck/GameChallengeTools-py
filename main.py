import gamedifficulty as GD

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

# This is a demo of the algorithm presented in the paper A Comprehensive Model of Automated Evaluation of
# Difficulty in Platformer Games. This project includes helper function to extrapolate information images and data
# from mario levels to then process into a difficulty curve.
# You can find an example implementation in python of the actual algorithm in gamedifficulty/Processing.py,
# function CalculateDifficulty

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

# moving platforms must be processed a bit differently.
passthroughPositions = GD.Processing.CreateMovingPlatform(levelImage, spriteSet.GetPlatformsTextures(), spriteSet.GetBalancePointsLeft(), spriteSet.GetBalancePointsRight())

# Transform all collisions into an image mask
collisionMask = GD.Processing.CreateMaskFromPatternResult(collisionPositions, levelImage.shape[:2])

# Mark all pixels mario can (theoretically) reach
reach = GD.Processing.CreateReachTextureFromPatternResult(levelImage.shape[:2], collisionPositions + passthroughPositions, int(GD.Constants.jumpHeight))

# reach2 = cv.imread(f"ressources/{level}/reach_filled.png", cv.IMREAD_GRAYSCALE)

# Create static danger map (holes)
danger = GD.Processing.CreateStaticDanger(collisionMask)

cv.imwrite(f"ressources/{level}/staticDanger.png", danger * 255)

enemyDanger = np.zeros(levelImage.shape[:2], dtype=np.uint8)

# Create enemy danger map for each enemy type
for type in GD.Types.EnemyType.GetAllTypes():
    # Find all enemies in the level
    enemyPositions = GD.Detection.DetectPatternMulti(levelImage, spriteSet.GetEnemyTextures(type), 0.85)
    # Find their possible positions
    ed = GD.Processing.CreateDisplacementTexture(type, enemyPositions, collisionMask)

    # merge static and enemy danger
    enemyDanger = np.maximum(ed, enemyDanger)

# Merge enemy danger and static danger
danger = np.maximum(danger, enemyDanger)

difficultyCurves = []
durations = []
# Calculate difficulty for different window sizes
for wsize in [16, 32, 64, 128, 160, 224, 256, 288, 304, 320]:

    start = time.time()
    difficultyCurve = GD.Processing.CalculateDifficulty(danger, reach, wsize)
    end = time.time()
    # difficultyCurve2 = GD.Processing.CalculateDifficulty(danger, reach2, wsize)

    difficultyCurves.append(difficultyCurve)
    durations.append(end - start)

    # plot
    plt.plot(difficultyCurve)
    plt.ylim(0, 1.2)
    # plt.title(f"Difficulty for level {level}, window size {wsize}")
    plt.show()

# Plot all difficulty curves on the same graph
for i, curve in enumerate(difficultyCurves):
    plt.plot(curve, label=f"Window size {i}")
plt.legend()
plt.ylim(0, 1)
plt.title(f"Difficulty for level {level}")
plt.show()

# Plot duration by window size
plt.plot(durations)
# plt.title(f"Duration for level {level}")
plt.xlabel("Window size")
plt.ylabel("Duration (s)")
plt.xticks(range(len(durations)), [f"{i}" for i in [16, 32, 64, 128, 160, 224, 256, 288, 304, 320]])
plt.ylim(0, np.max(durations) + .2)
plt.show()

# Display the level image
plt.imshow(cv.cvtColor(levelImage, cv.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# display collision as blue, danger as red, and reach as green
demoImg = np.zeros(levelImage.shape, dtype=np.uint8)
demoImg[:, :, 0] = collisionMask * 255
demoImg[:, :, 1] = reach * 255
demoImg[:, :, 2] = danger * 255

# save reach image, collision image and danger image
cv.imwrite(f"ressources/{level}/reach.png", reach * 255)
cv.imwrite(f"ressources/{level}/collision.png", collisionMask * 255)
cv.imwrite(f"ressources/{level}/danger.png", danger * 255)
cv.imwrite(f"ressources/{level}/enemyDanger.png", enemyDanger * 255)
# save demo image
cv.imwrite(f"ressources/{level}/demo.png", demoImg)