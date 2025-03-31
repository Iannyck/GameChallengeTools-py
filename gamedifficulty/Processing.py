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
    """
    Unused : creates a texture the shape of the jump
    """
    result = np.zeros((int(jumpHeight), int(jumpHalfWidth * 2)), dtype=np.uint8)
    for y in range(result.shape[0]):
        hd = -(np.sqrt(2. * gravity * -y)) / gravity * marioVelocity[1]
        for x in range(int(jumpHalfWidth - hd)):
            result[y, x] = 1

    return result


def CreateJumpDownTexture() -> cv.Mat[cv.CV_8U]:
    """
    Unused : creates a texture the shape of the jump
    """
    result = np.zeros((int(jumpHeight), int(jumpHalfWidth * 2)), dtype=np.uint8)
    for y in range(result.shape[0]):
        hd = -(np.sqrt(2. * gravity * -y)) / gravity * marioVelocity[1]
        for x in range(int(jumpHalfWidth + hd)):
            result[y, x] = 1

    return result


def CreateStaticDanger(collisionMask: cv.Mat[cv.CV_8U]) -> cv.Mat[cv.CV_8U]:
    """
    Using the collision mask, we find potential dangerous pixels that are above a hole mario can reach
    :param collisionMask: the collision mask
    :return: a mask of the dangerous pixels
    """
    danger = np.zeros(collisionMask.shape, dtype=np.uint8)

    danger[-1, collisionMask[-1, :] < 1] = 1

    for y in range(collisionMask.shape[0] - 2, -1, -1):
        for x in range(1, collisionMask.shape[1] - 1):
            if collisionMask[y, x] < 1:
                if danger[y + 1, x] == 1 or danger[y + 1, x - 1] == 1 or danger[y + 1, x + 1] == 1:
                    danger[y, x] = 1

    return danger


def CreateDisplacementTexture(ennemyType: EnemyType, detections: list[(int, int, int, int)], collisionMask: cv.Mat[cv.CV_8U]) -> cv.Mat[cv.CV_8U]:
    """
    Creates a displacement texture for an enemy type.
    :param ennemyType: the type of enemy
    :param detections: the detections positions and sizes of the enemy
    :param collisionMask: the collision mask
    :return: the displacement texture
    """
    if ennemyType == EnemyType.GOOMBA:
        return CreateGoombaDisplacementTexture(detections, collisionMask)

    return np.zeros(collisionMask.shape, dtype=np.uint8)

def CreateGoombaDisplacementTexture(detections: list[(int, int, int, int)], collisionMask: cv.Mat[cv.CV_8U]) -> cv.Mat[cv.CV_8U]:
    """
    Goomba specific implementation of CreateDisplacementTexture
    """
    result = np.zeros(collisionMask.shape, dtype=np.uint8)

    for (y, x, sizeY, sizeX) in detections:
        result[y:y + sizeY, x:x + sizeX] = 1

        direction = -1

        iter = 0
        maxIter = 1000
        while True and iter < maxIter:
            iter += 1

            for i in range(0, int(np.abs(gravity))):
                if y < 0 or y + sizeY >= collisionMask.shape[0] or not collisionMask[y + sizeY, x:x + sizeX].any():
                    y -= 1 * int(np.sign(gravity))
                else:
                    break

            if collisionMask[y:y+sizeY - 1, x if direction == -1 else x + sizeX].any():
                direction = 1 if direction == -1 else -1

            x += direction

            result[y:y + sizeY, x:x + sizeX] = 1

            # if any of the pixels outside of image break
            if y < 0 or y + sizeY >= collisionMask.shape[0] or x < 0 or x + sizeX >= collisionMask.shape[1]:
                break

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
    """
    Actual interesting part of the project. Here are done the difficulty calculations, from a reach map and a danger map.
    :param pheromones: danger map, where 1 is a dangerous pixel and 0 is not
    :param reach: reach map, where 1 is a pixel mario can reach and 0 is not
    :param windowSize: the size of the sliding window used in the calculations
    :return: array with the difficulty values at each point in the level [0: image size - window size]
    """
    assert pheromones.shape == reach.shape

    result = []

    for x in range(0, pheromones.shape[1] - windowSize):

        # Returns a 2D array the size of the window where element is true if pixel can be reached
        reachable = reach[:, x:x + windowSize] > 0

        # Returns a 2D array the size of the window where element is true if pixel is dangerous
        dangerous = pheromones[:, x:x + windowSize] > 0

        # np.logical_and returns the array of pixels that are dangerous and can be reached.
        # we then use np.sum to count the number (true is one and false is zero)
        count = np.sum(np.logical_and(reachable, dangerous))

        # edge case
        if np.isnan(float(np.sum(reachable))):
            result += [0]
            continue

        value = count / float(np.sum(reachable))
        result += [value]

    return result
