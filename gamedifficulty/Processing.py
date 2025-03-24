import cv2 as cv
import numpy as np
from gamedifficulty.Constants import *
from gamedifficulty.Types import EnemyType

def CreateMaskFromPatternResult(detections: list[(int, int, int, int)], imageSize: (int, int)) -> cv.Mat[cv.CV_8U]:
    """
    Returns a mask from the detections positions. Returns 1 if the pixel is part of a detection, 0 otherwise.
    :param detections: the detections to create the mask from
    :param imageSize: the size of the image to create the mask from
    :return: the mask
    """
    result = np.zeros(imageSize, dtype=np.uint8)

    for (y, x, sizeY, sizeX) in detections:
        result[y:y + sizeY, x:x + sizeX] = 1

    return result


def CreatePlatformTextureFromMask(mask: cv.Mat[cv.CV_8U]) -> cv.Mat[cv.CV_8U]:
    """
    Returns the top edges of the mask (the platforms the player can stand upon).
    1 if the pixel is the top edge of a platform, 0 otherwise.
    :param mask: a boolean texture, the mask to create the texture from
    :return: unsigned int 8 texture (grayscale)
    """
    kernel = np.array([[-1], [1]])

    return cv.filter2D(mask, -1, kernel)


def CreateJumpUpTexture() -> cv.Mat[cv.CV_8U]:
    result = np.zeros((int(jumpHeight), int(jumpHalfWidth * 2)), dtype=np.uint8)
    for y in range(result.shape[0]):
        hd = -(np.sqrt(2. * gravity * -y)) / gravity * marioVelocity[1]
        for x in range(int(jumpHalfWidth - hd)):
            result[y, x] = 1

    return result


def CreateJumpDownTexture() -> cv.Mat[cv.CV_8U]:
    result = np.zeros((int(jumpHeight), int(jumpHalfWidth * 2)), dtype=np.uint8)
    for y in range(result.shape[0]):
        hd = -(np.sqrt(2. * gravity * -y)) / gravity * marioVelocity[1]
        for x in range(int(jumpHalfWidth + hd)):
            result[y, x] = 1

    return result


def CreateStaticDanger(collisionMask: cv.Mat[cv.CV_8U]) -> cv.Mat[cv.CV_8U]:
    danger = np.zeros(collisionMask.shape, dtype=np.uint8)

    danger[-1, collisionMask[-1, :] < 1] = 1

    for y in range(collisionMask.shape[0] - 2, -1, -1):
        for x in range(1, collisionMask.shape[1] - 1):
            if collisionMask[y, x] < 1:
                if danger[y + 1, x] == 1 or danger[y + 1, x - 1] == 1 or danger[y + 1, x + 1] == 1:
                    danger[y, x] = 1

    return danger


def CreateDisplacementTexture(ennemyType: EnemyType, detections: list[(int, int, int, int)], collisionMask: cv.Mat[cv.CV_8U]) -> cv.Mat[cv.CV_8U]:

    result = np.zeros(collisionMask.shape, dtype=np.uint8)

    # detection is y, x, ennemy size y, ennemy size x
    for (y, x, sizeY, sizeX) in detections:


    return result


def CreateReachTextureFromPatternResult(shape: (int, int), detections: list[(int, int, int, int)], height: int) -> cv.Mat[cv.CV_8U]:
    """
    Returns a mask from the detections positions. Returns 1 if the pixel is part of a detection, 0 otherwise.
    :param detections: the detections to create the mask from
    :param height: the jump height
    :return: the mask
    """
    result = np.zeros(shape, dtype=np.uint8)

    for (y, x, sizeY, sizeX) in detections:
        result[y - height:y, x:x + sizeX] = 1

    return result


def CalculateDifficulty(pheromones: cv.Mat[cv.CV_8U], reach: cv.Mat[cv.CV_8U], windowSize: int) -> np.array(np.float32):
    pass
