from gamedifficulty import *

import cv2 as cv
import numpy as np

# Load the level image
level = "Niveau_1_1"
levelImage = cv.imread(f"ressources/{level}/level.png")

# Load the sprite set (ground, enemies, etc)
spriteSet = Classes.SpriteSet("ressources/Sprite")

# Detect the positions of the enemies in the level image
positions = Detection.DetectPatternMulti(levelImage, spriteSet.GetCollisionsTextures(), 0.85)
positions += Detection.DetectPatternMulti(levelImage, spriteSet.GetPipesTextures(), 0.85)

mask = Processing.CreateMaskFromPatternResult(positions, levelImage.shape[:2])

platformTexture = Processing.CreatePlatformTextureFromMask(mask)

cv.imshow("Analysed level", platformTexture * 255)
cv.waitKey(0)