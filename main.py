import gamedifficulty as GD

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from gamedifficulty.Constants import marioVelocity, gravity

# Load the level image
level = "Niveau_1_1"
levelImage = cv.imread(f"ressources/{level}/level.png")

cv.imshow("Level", levelImage)

# Load the sprite set (ground, enemies, etc)
spriteSet = GD.Classes.SpriteSet("ressources/Sprite")

collisionPositions = GD.Detection.DetectPatternMulti(levelImage, spriteSet.GetCollisionsTextures(), 0.85)
collisionPositions += GD.Detection.DetectPatternMulti(levelImage, spriteSet.GetPipesTextures(), 0.85)

mask = GD.Processing.CreateMaskFromPatternResult(collisionPositions, levelImage.shape[:2])

reach = GD.Processing.CreateReachTextureFromPatternResult(levelImage.shape[:2], collisionPositions, int(GD.Constants.jumpHeight))

# cv.imshow("Mask", mask * 255)
# cv.waitKey(0)

staticDanger = GD.Processing.CreateStaticDanger(mask)

# cv.imshow("Static Danger", staticDanger * 255)
# cv.waitKey(0)

goombaPositions = GD.Detection.DetectPatternMulti(levelImage, spriteSet.GetEnemyTextures(GD.Types.EnemyType.GOOMBA), 0.85)
goombaDanger = GD.Processing.CreateDisplacementTexture(GD.Types.EnemyType.GOOMBA, goombaPositions, mask)

# cv.imshow("Goomba Danger", goombaDanger * 255)
# cv.waitKey(0)

# merge static and goomba danger
danger = np.maximum(goombaDanger, staticDanger)

for wsize in [16, 32, 64, 128, 160, 224, 256, 288, 304, 320]:
    difficultyCurve = GD.Processing.CalculateDifficulty(danger, reach, wsize)

    # plot
    plt.plot(difficultyCurve)
    plt.title(f"Difficulty for level {level}, window size {wsize}")
    plt.show()