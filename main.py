from gamedifficulty import *

import cv2 as cv
import numpy as np

# Load the level image
level = "Niveau_1_1"
levelImage = cv.imread(f"ressources/{level}/level.png")

# Load the sprite set (ground, enemies, etc)
spriteSet = SpriteSet("ressources/Sprite")

# Select an enemy to detect
enemy = EnemyType.KOOPA

# Detect the positions of the enemies in the level image
patternPositions = DetectPatternMulti(levelImage, spriteSet.GetEnemiesTextures()[enemy], 0.8)

# Create image of same size as levelImage, full black
modifiedImage = np.copy(levelImage)

(enemySizeX, enemySizeY) = GetEnnemySize(enemy)

# Mark enemies on the image
for (x, y) in patternPositions:
    modifiedImage[y:y+enemySizeY, x:x+enemySizeX] = [0, 0, 255]

cv.imshow("Analysed level", modifiedImage)
cv.waitKey(0)