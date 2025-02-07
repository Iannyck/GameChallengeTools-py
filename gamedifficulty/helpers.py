import cv2 as cv
from os import listdir
from os.path import isfile, join
from gamedifficulty.types import EnemyType


def LoadTexturesInFolder(path: str) -> list[cv.Mat]:
    return [cv.imread(join(path, f), cv.IMREAD_COLOR) for f in listdir(path) if isfile(join(path, f))]


def GetEnnemySize(enemyType: EnemyType) -> (int, int):
    match enemyType:
        case EnemyType.HAMMER_BRO:
            return (16, 23)
        case EnemyType.KOOPA:
            return (16, 23)
        case EnemyType.FLYING_KOOPA:
            return (16, 23)
        case EnemyType.LAKITU:
            return (16, 23)
        case EnemyType.PIRANHA_PLANT:
            return (16, 23)
        case _:
            return (16, 16)