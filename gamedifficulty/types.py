from enum import Enum

class EnemyType(Enum):
    FLYING_KOOPA = "flying_koopa"  # MUST BE BEFORE KOOPA BECAUSE OF LOADING
    GOOMBA = "goomba"
    KOOPA = "koopa"
    BOWSER = "bowser"
    LAKITU = "lakitu"
    TURTLE_SPIKE = "turtle"
    HAMMER_BRO = "hammer_bro"
    FLYING_FISH = "flying_fish"
    PIRANHA_PLANT = "piranha_plant"
    TURTLE = "turtle"

    def __str__(self):
        return self.value

    # Workaround for the fact that the value of the enum (used in json) is not the same as the file name
    @staticmethod
    def GetFileName(type) -> str:
        if type == EnemyType.FLYING_KOOPA:
            return "koopa_volant"
        else:
            return type.value
