import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join
from gamedifficulty.Types import EnemyType


def LoadTexturesInFolder(path: str) -> list[cv.Mat]:
    return [cv.imread(join(path, f), cv.IMREAD_COLOR) for f in listdir(path) if isfile(join(path, f))]

def BooleanMaskAsGrayscale(mask: cv.Mat) -> cv.Mat:
    return mask.astype(np.uint8) * 255