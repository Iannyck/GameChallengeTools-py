import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join
from gamedifficulty.Types import EnemyType


def LoadTexturesInFolder(path: str) -> list[cv.Mat]:
    return [cv.imread(join(path, f), cv.IMREAD_COLOR) for f in listdir(path) if isfile(join(path, f))]

def BooleanMaskAsGrayscale(mask: cv.Mat) -> cv.Mat:
    return mask.astype(np.uint8) * 255

def CoordsInImage(coords: (int, int), image : cv.Mat) -> bool:
    return 0 <= coords[0] < image.shape[0] and 0 <= coords[1] < image.shape[1]