import gamedifficulty as GD

import cv2 as cv
import numpy as np

from gamedifficulty.Constants import marioVelocity, gravity

# Load the level image
level = "Niveau_1_1"
levelImage = cv.imread(f"ressources/{level}/level.png")

cv.imshow("Level", levelImage)

# Load the sprite set (ground, enemies, etc)
spriteSet = GD.Classes.SpriteSet("ressources/Sprite")

# Detect the positions of the enemies in the level image
positions = GD.Detection.DetectPatternMulti(levelImage, spriteSet.GetCollisionsTextures(), 0.85)
positions += GD.Detection.DetectPatternMulti(levelImage, spriteSet.GetPipesTextures(), 0.85)

mask = GD.Processing.CreateMaskFromPatternResult(positions, levelImage.shape[:2])

platformTexture = GD.Processing.CreatePlatformTextureFromMask(mask)

jumpShape = spriteSet.GetJumpShapeTexture()

processingContext = GD.Processing.ProcessingContext()

test = processingContext.PropagateMovementPheromones(platformTexture, mask)

test = cv.max(test, platformTexture * 255)

cv.imshow("Test", test)
cv.waitKey(0)

cv.imwrite(f"Platforms.png", platformTexture * 255)
cv.imwrite(f"CollisionMask.png", mask * 255)
