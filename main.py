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

positions = GD.Detection.DetectPatternMulti(levelImage, spriteSet.GetCollisionsTextures(), 0.85)
positions += GD.Detection.DetectPatternMulti(levelImage, spriteSet.GetPipesTextures(), 0.85)

mask = GD.Processing.CreateMaskFromPatternResult(positions, levelImage.shape[:2])

cv.imshow("Mask", mask * 255)
cv.waitKey(0)

staticDanger = GD.Processing.CreateStaticDanger(mask)

cv.imshow("Static Danger", staticDanger * 255)
cv.waitKey(0)

goombaPositions = GD.Detection.DetectPatternMulti(levelImage, spriteSet.GetEnemyTextures(GD.Types.EnemyType.GOOMBA), 0.85)
goombaDanger = GD.Processing.CreateDisplacementTexture(GD.Types.EnemyType.GOOMBA, goombaPositions, mask)

cv.imshow("Goomba Danger", goombaDanger * 255)
cv.waitKey(0)

enemyImage = GD.Processing.CreateMaskFromPatternResult(goombaPositions, levelImage.shape[:2])

cv.imwrite("enemy.png", enemyImage * 255)

# write mask in r channel, static danger in g channel, goomba danger in b channel
danger = np.zeros((*mask.shape, 3), dtype=np.uint8)
danger[:, :, 0] = mask * 255
danger[:, :, 1] = staticDanger * 255
danger[:, :, 2] = goombaDanger * 255

cv.imshow("Danger", danger)
cv.waitKey(0)

cv.imwrite(f"danger.png", danger)

# platformTexture = GD.Processing.CreatePlatformTextureFromMask(mask)
#
# reachTexture = GD.Processing.CreateReachTextureFromPatternResult(levelImage.shape[:2], positions, int(GD.Constants.jumpHeight))
#
# cv.imshow("Platform", platformTexture * 255)
# cv.imshow("Reach", reachTexture * 255)
# cv.waitKey(0)