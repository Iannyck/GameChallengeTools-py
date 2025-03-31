import gamedifficulty as GD

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# This is a demo of the algorithm presented in the paper A Comprehensive Model of Automated Evaluation of
# Difficulty in Platformer Games. This project includes helper function to extrapolate information images and data
# from mario levels to then process into a difficulty curve.
# You can find an example implementation in python of the actual algorithm in gamedifficulty/Processing.py,
# function CalculateDifficulty

# Load the level image
level = "Niveau_1_1"
levelImage = cv.imread(f"ressources/{level}/level.png")

cv.imshow("Level", levelImage)

# Load the sprite set (ground, enemies, etc)
spriteSet = GD.Classes.SpriteSet("ressources/Sprite")

# Detect all ground and pipe tiles that form the collisions of the level.
collisionPositions = GD.Detection.DetectPatternMulti(levelImage, spriteSet.GetCollisionsTextures(), 0.85)
collisionPositions += GD.Detection.DetectPatternMulti(levelImage, spriteSet.GetPipesTextures(), 0.85)

# Transform all positions into an image mask
collisionMask = GD.Processing.CreateMaskFromPatternResult(collisionPositions, levelImage.shape[:2])

# Mark all pixels mario can (theoretically) reach
reach = GD.Processing.CreateReachTextureFromPatternResult(levelImage.shape[:2], collisionPositions, int(GD.Constants.jumpHeight))

# Create static danger map (holes)
staticDanger = GD.Processing.CreateStaticDanger(collisionMask)

# Create goomba danger map
# Find goombas
goombaPositions = GD.Detection.DetectPatternMulti(levelImage, spriteSet.GetEnemyTextures(GD.Types.EnemyType.GOOMBA), 0.85)
# Find their possible positions
goombaDanger = GD.Processing.CreateDisplacementTexture(GD.Types.EnemyType.GOOMBA, goombaPositions, collisionMask)

# merge static and goomba danger
danger = np.maximum(goombaDanger, staticDanger)

difficultyCurves = []
# Calculate difficulty for different window sizes
for wsize in [16, 32, 64, 128, 160, 224, 256, 288, 304, 320]:
    difficultyCurve = GD.Processing.CalculateDifficulty(danger, reach, wsize)

    difficultyCurves.append(difficultyCurve)

    # plot
    plt.plot(difficultyCurve)
    plt.ylim(0, 1.2)
    plt.title(f"Difficulty for level {level}, window size {wsize}")
    plt.show()

# Plot all difficulty curves
for i, curve in enumerate(difficultyCurves):
    plt.plot(curve, label=f"Window size {i}")
plt.legend()
plt.ylim(0, 1)
plt.title(f"Difficulty for level {level}")
plt.show()

plt.imshow(cv.cvtColor(levelImage, cv.COLOR_BGR2RGB))
plt.axis("off")
plt.show()