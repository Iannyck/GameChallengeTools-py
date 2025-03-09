import cv2 as cv
import numpy as np

from gamedifficulty.Constants import *
from gamedifficulty.Helpers import CoordsInImage


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

def CreateReachFromPlatformTexture(platformTexture: cv.Mat[cv.CV_8U], platformMask: cv.Mat[cv.CV_8U]) -> cv.Mat[cv.CV_8U]:

    result = np.zeros(platformTexture.shape, dtype=np.uint8)

    # TODO Move into function
    jumpUpRight = np.zeros((int(jumpHeight), int(jumpHalfWidth * 2)), dtype=np.uint8)
    for y in range(jumpUpRight.shape[0]):
        hd = -(np.sqrt(2. * gravity * -y)) / gravity * marioVelocity[0]
        for x in range(int(jumpHalfWidth - hd)):
            jumpUpRight[y, x] = 1

    jumpUpLeft = cv.flip(jumpUpRight, 1)

    jumpDownRight = np.zeros((int(jumpHeight), int(jumpHalfWidth * 2)), dtype=np.uint8)
    for y in range(jumpDownRight.shape[0]):
        hd = -(np.sqrt(2. * gravity * -y)) / gravity * marioVelocity[0]
        for x in range(int(jumpHalfWidth + hd)):
            jumpDownRight[y, x] = 1

    jumpDownLeft = cv.flip(jumpDownRight, 1)

    cv.imshow("JumpDown", jumpDownRight * 255)

    # For each pixel of the platform texture that is a top edge
    platformPixelCoords = np.transpose((platformTexture>0).nonzero())

    tempText = np.zeros(platformTexture.shape, dtype=np.uint8)

    for (y, x) in platformPixelCoords:
        # Copy jumpUp to the right position
        top = max(y-int(jumpHeight), 0)
        left = max(x-int(jumpHalfWidth), 0)
        right = min(x+int(jumpHalfWidth), result.shape[1])

        # right jump
        tempText[top:y, x:right] = jumpUpRight[0:top+y, 0:right-x]
        # left jump
        tempText[top:y, left:x] = jumpUpLeft[0:top+y, jumpUpLeft.shape[1]-x+left:]

        result += tempText
        tempText = np.zeros(platformTexture.shape, dtype=np.uint8)


    return result