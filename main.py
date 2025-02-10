from gamedifficulty import *

import cv2 as cv
import numpy as np

# Load the level image
level = "Niveau_6_3"
levelImage = cv.imread(f"ressources/{level}/level.png")

cv.imshow("Level", levelImage)
cv.waitKey(0)

# Load the sprite set (ground, enemies, etc)
spriteSet = Classes.SpriteSet("ressources/Sprite")

# Detect the positions of the enemies in the level image
positions = Detection.DetectPatternMulti(levelImage, spriteSet.GetCollisionsTextures(), 0.85)
positions += Detection.DetectPatternMulti(levelImage, spriteSet.GetPipesTextures(), 0.85)

mask = Processing.CreateMaskFromPatternResult(positions, levelImage.shape[:2])

platformTexture = Processing.CreatePlatformTextureFromMask(mask)

jumpShape = spriteSet.GetJumpShapeTexture()

test = Processing.CreateReachFromPlatformTexture(platformTexture, jumpShape)

cv.imshow("Test", test)
cv.waitKey(0)

cv.imwrite(f"Platforms.png", platformTexture * 255)
cv.imwrite(f"CollisionMask.png", mask * 255)
