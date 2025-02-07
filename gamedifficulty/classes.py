from gamedifficulty.types import EnemyType
from gamedifficulty.helpers import LoadTexturesInFolder

import cv2 as cv
from os import listdir
from os.path import join, isfile


class Enemy:
    x = 0
    y = 0
    width = 0
    heigt = 0
    enemyType = EnemyType.GOOMBA

    def __init__(self, x, y, width, height, ennemyType):
        self.x = x
        self.y = y
        self.width = width
        self.heigt = height
        self.enemyType = ennemyType

    def ToJsonDict(self):
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.heigt,
            "type": self.enemyType.value
        }

    @staticmethod
    def FromJsonDict(json_dict):
        return Enemy(json_dict["x"], json_dict["y"], json_dict["width"], json_dict["height"],
                     EnemyType(json_dict["type"]))


class Pipe:
    x = 0
    y = 0
    height = 0
    width = 0
    id = 0
    inside = False
    go_id = 0
    dynamicPipe = False

    def __init__(self, x, y, height, width, id, inside, goId, dynamicPipe):
        self.x = x
        self.y = y
        self.height = height
        self.width = width
        self.id = id
        self.inside = inside
        self.go_id = goId
        self.dynamicPipe = dynamicPipe

    def ToJsonDict(self):
        return {
            "x": self.x,
            "y": self.y,
            "height": self.height,
            "width": self.width,
            "id": self.id,
            "inside": self.inside,
            "go_id": self.go_id,
            "dynamic_pipe": self.dynamicPipe
        }

    @staticmethod
    def FromJsonDict(json_dict):
        return Pipe(json_dict["x"], json_dict["y"], json_dict["height"], json_dict["width"], json_dict["id"],
                    json_dict["inside"], json_dict["go_id"], json_dict["dynamic_pipe"])


class SpriteSet:
    """
    Helper class used to load sprites (ground, enemies, etc) from the given path
    """

    def __init__(self, path: str):
        self.path = path
        self.collisionsTextures = []
        self.endTextures = []
        self.enemiesTextures = dict()
        self.magicBeanTextures = []
        self.pipesTextures = []
        self.platformsTextures = []
        self.spawnTextures = []

        self.Load()

    def Load(self):
        self.collisionsTextures = LoadTexturesInFolder(f"{self.path}/CollisionBlock")
        self.endTextures = LoadTexturesInFolder(f"{self.path}/End")
        self.magicBeanTextures = LoadTexturesInFolder(f"{self.path}/MagicBean")
        self.pipesTextures = LoadTexturesInFolder(f"{self.path}/Pipes")
        self.platformsTextures = LoadTexturesInFolder(f"{self.path}/Platforms")
        self.spawnTextures = LoadTexturesInFolder(f"{self.path}/Spawn")

        # Todo : maybe change they way sprites are organized in the folders ?
        for enemy in EnemyType:
            self.enemiesTextures[enemy] = []

        enemyFilesPath = f"{self.path}/Enemies"
        enemyFiles = [join(enemyFilesPath, f) for f in listdir(enemyFilesPath) if isfile(join(enemyFilesPath, f))]
        alreadyReadFiles = []

        for enemyType in EnemyType:
            for file in enemyFiles:
                if EnemyType.GetFileName(enemyType) in file and file not in alreadyReadFiles:
                    self.enemiesTextures[enemyType].append(cv.imread(file, cv.IMREAD_COLOR))
                    alreadyReadFiles.append(file)

    def GetCollisionsTextures(self) -> list:
        return self.collisionsTextures

    def GetEndTextures(self) -> list:
        return self.endTextures

    def GetEnemiesTextures(self) -> dict:
        """
        Return a dictionary with the key being the enemy type and the value being a list of textures representing the enemy
        :return: dict
        """
        return self.enemiesTextures

    def GetMagicBeanTextures(self) -> list:
        return self.magicBeanTextures

    def GetPipesTextures(self) -> list:
        return self.pipesTextures

    def GetPlatformsTextures(self) -> list:
        return self.platformsTextures

    def GetSpawnTextures(self) -> list:
        return self.spawnTextures
