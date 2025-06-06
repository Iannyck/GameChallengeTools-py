import cv2 as cv
import numpy as np


def DetectPattern(image: cv.Mat, pattern: cv.Mat, threshold=0.8) -> list[(int, int, int, int)]:
    """
    Returns a list of locations matching the pattern to a certain threshold
    :param image: the image to analyse
    :param pattern: the pattern to match
    :param threshold: minimum value of similarity to the pattern to consider a match
    :param debugShowMatchResult: DEBUG show the matchTemplate result
    :return: a list of (int, int, int, int) representing the (y,x, sizeY, sizeX) positions of the matches
    """
    result = cv.matchTemplate(image, pattern, cv.TM_CCOEFF_NORMED)

    locations = np.argwhere(result > threshold)
    (imSizeY, imSizeX) = pattern.shape[:2]
    return [(y, x, imSizeY, imSizeX) for (y, x) in locations]


def DetectPatternMulti(image: cv.Mat, patterns: list[cv.Mat], threshold=0.8) -> list[(int, int, int, int)]:
    """
    See DetectPattern, same function but takes in a list of pattern instead of just one
    """
    result = []
    for pattern in patterns:
        result += DetectPattern(image, pattern, threshold)

    # Filter duplicates
    result = list(set(result))

    return result
