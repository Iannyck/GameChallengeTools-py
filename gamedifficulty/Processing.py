import cv2 as cv
import heapq
import numpy as np
from gamedifficulty.Constants import *
from gamedifficulty.Types import EnemyType
from gamedifficulty.Detection import DetectPatternMulti
from concurrent.futures import ThreadPoolExecutor


def CreateMaskFromPatternResult(
    detections: list[(int, int, int, int)], imageSize: (int, int)
) -> cv.Mat[cv.CV_8U]:
    """
    Returns a mask from the detections positions. Returns 1 if the pixel is part of a detection, 0 otherwise.
    :param detections: the detections to create the mask from
    :param imageSize: the size of the image to create the mask from
    :return: the mask
    """
    result = np.zeros(imageSize, dtype=np.uint8)

    for y, x, sizeY, sizeX in detections:
        result[y : y + sizeY, x : x + sizeX] = 1

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
        hd = -(np.sqrt(2.0 * gravity * -y)) / gravity * marioVelocity[1]
        for x in range(int(jumpHalfWidth - hd)):
            result[y, x] = 1

    return result


def CreateJumpDownTexture() -> cv.Mat[cv.CV_8U]:
    """
    Unused : creates a texture the shape of the jump
    """
    result = np.zeros((int(jumpHeight), int(jumpHalfWidth * 2)), dtype=np.uint8)
    for y in range(result.shape[0]):
        hd = -(np.sqrt(2.0 * gravity * -y)) / gravity * marioVelocity[1]
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
                if (
                    danger[y + 1, x] == 1
                    or danger[y + 1, x - 1] == 1
                    or danger[y + 1, x + 1] == 1
                ):
                    danger[y, x] = 1

    return danger


def CreateDisplacementTexture(
    ennemyType: EnemyType,
    detections: list[(int, int, int, int)],
    collisionMask: cv.Mat[cv.CV_8U],
) -> cv.Mat[cv.CV_8U]:
    """
    Creates a displacement texture for an enemy type.
    :param ennemyType: the type of enemy
    :param detections: the detections positions and sizes of the enemy
    :param collisionMask: the collision mask
    :return: the displacement texture
    """
    if (
        ennemyType == EnemyType.GOOMBA
        or ennemyType == EnemyType.KOOPA
        or ennemyType == EnemyType.TURTLE
    ):
        return CreateGoombaDisplacementTexture(detections, collisionMask)
    if ennemyType == EnemyType.RED_KOOPA:
        return CreateRedKoopaDisplacementTexture(detections, collisionMask)
    if ennemyType == EnemyType.PIRANHA_PLANT:
        return CreatePiranaPlantDisplacementTexture(detections, collisionMask)
    if ennemyType == EnemyType.BULLET_BILL:
        return CreateBulletBillDisplacementTexture(detections, collisionMask)
    if ennemyType == EnemyType.FLYING_FISH:
        return CreateFlyingFishDisplacementTexture(detections, collisionMask)
    if ennemyType == EnemyType.LAKITU:
        return CreateLakituDisplacementTexture(detections, collisionMask)

    return np.zeros(collisionMask.shape, dtype=np.uint8)


def CreateEnemyPheromoneMap(
    enemyDetections: dict[EnemyType, list[(int, int, int, int)]],
    collisionMask: cv.Mat[cv.CV_8U],
    pheromoneValues: dict[EnemyType, float] = None,
) -> np.ndarray:
    """
    Creates a pheromone map from enemy displacement zones.
    Each enemy contributes a configurable amount of pheromone on all pixels it can reach.
    If multiple enemies can reach the same pixel, the values are summed.

    :param enemyDetections: mapping from enemy type to list of detected enemy bounding boxes
    :param collisionMask: collision mask used to compute displacement zones
    :param pheromoneValues: optional weights per enemy type
    :return: float32 pheromone map
    """
    if pheromoneValues is None:
        pheromoneValues = {
            EnemyType.GOOMBA: 0.2,
            EnemyType.KOOPA: 0.2,
            EnemyType.TURTLE: 0.2,
            EnemyType.RED_KOOPA: 0.3,
            EnemyType.FLYING_KOOPA: 0.3,
            EnemyType.BOWSER: 0.5,
            EnemyType.LAKITU: 0.5,
            EnemyType.TURTLE_SPIKE: 0.25,
            EnemyType.HAMMER_BRO: 0.3,
            EnemyType.FLYING_FISH: 0.3,
            EnemyType.PIRANHA_PLANT: 0.1,
            EnemyType.BULLET_BILL: 0.25,
        }

    pheromones = np.zeros(collisionMask.shape, dtype=np.float32)
    total_pixels = float(collisionMask.size)
    scaleFactor = 2
    maxScale = 3

    for enemyType, detections in enemyDetections.items():
        if not detections:
            continue

        base_weight = float(pheromoneValues.get(enemyType, 0.0))
        if base_weight <= 0:
            continue

        for detection in detections:
            singleDetection = [detection]
            displacement = CreateDisplacementTexture(
                enemyType, singleDetection, collisionMask
            )
            area = float(np.count_nonzero(displacement))
            if area <= 0.0:
                continue

            area_frac = area / total_pixels

            scale = 1.0 + (1.0 - area_frac) * scaleFactor
            if scale > maxScale:
                scale = maxScale

            pheromones += displacement.astype(np.float32) * (base_weight * scale)

    max_pheromones = np.max(pheromones)
    if max_pheromones > 0:
        pheromones /= max_pheromones

    return pheromones


def CreateAccessibleDangerMap(
    collisionMask: cv.Mat[cv.CV_8U],
    enemyDetections: dict,
    normalizedReach: cv.Mat[cv.CV_8U],
) -> np.ndarray:
    """
    Crée une carte des dangers accessibles.
    Combine les dangers statiques (trous) et les phéromones ennemies,
    puis les filtre par l'accessibilité normalisée.

    :return: Matrice float32 normalisée (0.0 à 1.0)
    """

    static_danger = CreateStaticDanger(collisionMask).astype(np.float32)
    static_danger = NormalizeStaticDanger(static_danger)

    enemy_pheromones = CreateEnemyPheromoneMap(enemyDetections, collisionMask)

    combined_danger = np.maximum(static_danger, enemy_pheromones)

    access_mask = (normalizedReach > 0).astype(np.float32)
    accessible_danger = combined_danger * access_mask

    return accessible_danger


def NormalizeStaticDanger(collisionMask: cv.Mat[cv.CV_8U]) -> np.ndarray:
    """
    Normalise les dangers statiques (trous).
    Le danger est maximal au fond du trou et diminue au fur et à mesure
    que l'on remonte vers le niveau du sol.

    :param collisionMask: Masque où les trous sont à 1, le reste à 0
    :param decayRate: Valeur soustraite à chaque pixel de remontée
    :return: Matrice float32 normalisée entre 0.0 et 1.0
    """
    height, width = collisionMask.shape

    normalized_danger = np.zeros((height, width), dtype=np.float32)

    for x in range(width):

        col = collisionMask[:, x]

        current_danger = height

        for y in range(height - 1, -1, -1):
            if col[y] == 1:
                normalized_danger[y, x] = current_danger

            current_danger = max(0.0, current_danger - 1)

    result = normalized_danger / height

    return result


def CreateGoombaDisplacementTexture(
    detections: list[(int, int, int, int)], collisionMask: cv.Mat[cv.CV_8U]
) -> cv.Mat[cv.CV_8U]:
    """
    Goomba specific implementation of CreateDisplacementTexture
    """
    result = np.zeros(collisionMask.shape, dtype=np.uint8)

    for y, x, sizeY, sizeX in detections:
        result[y : y + sizeY, x : x + sizeX] = 1

        direction = -1

        iter = 0
        maxIter = 1000
        while True and iter < maxIter:
            iter += 1

            for i in range(0, int(np.abs(gravity))):
                if (
                    y < 0
                    or y + sizeY >= collisionMask.shape[0]
                    or not collisionMask[y + sizeY, x : x + sizeX].any()
                ):
                    y -= 1 * int(np.sign(gravity))
                else:
                    break

            if collisionMask[
                y : y + sizeY - 1, x if direction == -1 else x + sizeX
            ].any():
                direction = 1 if direction == -1 else -1

            x += direction

            result[y : y + sizeY, x : x + sizeX] = 1

            if (
                y < 0
                or y + sizeY >= collisionMask.shape[0]
                or x < 0
                or x + sizeX >= collisionMask.shape[1]
            ):
                break

    return result


def CreateRedKoopaDisplacementTexture(
    detections: list[(int, int, int, int)], collisionMask: cv.Mat[cv.CV_8U]
) -> cv.Mat[cv.CV_8U]:
    """
    Red Koopa specific implementation of CreateDisplacementTexture.
    Red Koopas move horizontally but do not fall off edges, they reverse when a ledge or wall is ahead.
    """
    result = np.zeros(collisionMask.shape, dtype=np.uint8)
    levelHeight, levelWidth = collisionMask.shape

    for y, x, sizeY, sizeX in detections:
        current_y = y
        current_x = x
        direction = -1
        iteration = 0
        max_iterations = 1000

        while iteration < max_iterations:
            iteration += 1

            if (
                current_y < 0
                or current_y + sizeY >= levelHeight
                or current_x < 0
                or current_x + sizeX > levelWidth
            ):
                break

            while (
                current_y + sizeY < levelHeight
                and not collisionMask[
                    current_y + sizeY, current_x : current_x + sizeX
                ].any()
            ):
                current_y -= int(np.sign(gravity))
                if current_y < 0 or current_y + sizeY >= levelHeight:
                    break

            if (
                current_y < 0
                or current_y + sizeY >= levelHeight
                or current_x < 0
                or current_x + sizeX > levelWidth
            ):
                break

            result[current_y : current_y + sizeY, current_x : current_x + sizeX] = 1

            front_x = current_x + direction
            support_x_start = front_x
            support_x_end = front_x + sizeX
            has_support_ahead = (
                0 <= support_x_start
                and support_x_end <= levelWidth
                and collisionMask[
                    current_y + sizeY, support_x_start:support_x_end
                ].any()
            )

            front_edge_x = current_x - 1 if direction == -1 else current_x + sizeX
            has_wall_ahead = (
                front_edge_x < 0
                or front_edge_x >= levelWidth
                or collisionMask[current_y : current_y + sizeY - 1, front_edge_x].any()
            )

            if not has_support_ahead or has_wall_ahead:
                opposite_direction = -direction
                opposite_front_x = current_x + opposite_direction
                opposite_support_x_start = opposite_front_x
                opposite_support_x_end = opposite_front_x + sizeX
                opposite_has_support = (
                    0 <= opposite_support_x_start
                    and opposite_support_x_end <= levelWidth
                    and collisionMask[
                        current_y + sizeY,
                        opposite_support_x_start:opposite_support_x_end,
                    ].any()
                )
                opposite_front_edge_x = (
                    current_x - 1 if opposite_direction == -1 else current_x + sizeX
                )
                opposite_has_wall = (
                    opposite_front_edge_x < 0
                    or opposite_front_edge_x >= levelWidth
                    or collisionMask[
                        current_y : current_y + sizeY - 1, opposite_front_edge_x
                    ].any()
                )

                if not opposite_has_support or opposite_has_wall:
                    break

                direction = opposite_direction
                continue

            current_x += direction

    return result


def CreateBulletBillDisplacementTexture(
    detections: list[(int, int, int, int)], collisionMask: cv.Mat[cv.CV_8U]
) -> cv.Mat[cv.CV_8U]:
    """
    Bullet Bill specific implementation of CreateDisplacementTexture
    """
    result = np.zeros(collisionMask.shape, dtype=np.uint8)

    for y, x, sizeY, sizeX in detections:
        result[y : y + sizeY, x : x + sizeX] = 1

        for currX in range(0, collisionMask.shape[1] - sizeX + 1):
            result[y : y + sizeY, currX : currX + sizeX] = 1

    return result


def CreateFlyingFishDisplacementTexture(
    detections: list[(int, int, int, int)], collisionMask: cv.Mat[cv.CV_8U]
) -> cv.Mat[cv.CV_8U]:
    """
    Flying Fish specific implementation of CreateDisplacementTexture.
    Flying fish move horizontally across the level while oscillating vertically,
    so we mark the full horizontal span and a vertical band around the detected position.
    """
    result = np.zeros(collisionMask.shape, dtype=np.uint8)
    levelWidth = collisionMask.shape[1]
    levelHeight = collisionMask.shape[0]

    for y, x, sizeY, sizeX in detections:
        result[y : y + sizeY, x : x + sizeX] = 1

        vertical_margin = max(1, sizeY * 2)
        top = max(0, y - vertical_margin)
        bottom = min(levelHeight, y + sizeY + vertical_margin)

        for currX in range(0, levelWidth - sizeX + 1):
            result[top:bottom, currX : currX + sizeX] = 1

    return result


def CreateLakituDisplacementTexture(
    detections: list[(int, int, int, int)], collisionMask: cv.Mat[cv.CV_8U]
) -> cv.Mat[cv.CV_8U]:
    """
    Lakitu specific implementation of CreateDisplacementTexture.
    Lakitu can attack from almost anywhere except directly above him,
    so the entire level becomes dangerous except the vertical zone above each Lakitu.
    """
    result = np.zeros(collisionMask.shape, dtype=np.uint8)
    if not detections:
        return result

    result[:, :] = 1
    height, width = collisionMask.shape

    for y, x, sizeY, sizeX in detections:
        safe_y_end = max(0, y)
        if safe_y_end > 0:
            result[0:safe_y_end, :] = 0

    return result


def CreatePiranaPlantDisplacementTexture(
    detections: list[(int, int, int, int)], collisionMask: cv.Mat[cv.CV_8U]
) -> cv.Mat[cv.CV_8U]:

    result = np.zeros(collisionMask.shape, dtype=np.uint8)

    for y, x, sizeY, sizeX in detections:
        result[y : y + sizeY, x : x + sizeX] = 1

    return result


def CalculateHorizontalExpansion(jumpHeight: int, baseJumpHeight: int) -> int:
    """
    Calculate horizontal expansion based on jump height ratio.
    When Mario jumps higher, he can reach further horizontally.
    Uses physics: time_in_air ∝ sqrt(height), so distance ∝ sqrt(height)
    :param jumpHeight: the height of the jump
    :param baseJumpHeight: the base jump height for comparison
    :return: horizontal expansion in pixels
    """
    if baseJumpHeight <= 0:
        return 0

    expansion_factor = np.sqrt(jumpHeight / baseJumpHeight)

    expansion = int(jumpHalfWidth * (expansion_factor - 1))
    return expansion


def CreateReachTextureFromPatternResults(
    shape: (int, int),
    patternGroups: list[tuple[list[(int, int, int, int)], int]],
    horizontalExpansions: list[int] = None,
) -> cv.Mat[cv.CV_8U]:
    """
    Creates a reach mask from one or more groups of platform detections.
    Each group may use a different jump height, for example normal platforms and trampoline/jump pad platforms.
    :param shape: the shape of the output mask
    :param patternGroups: list of (detections, height) tuples
    :param horizontalExpansions: optional list of horizontal expansions for each group
    :return: the reach texture
    """
    result = np.zeros(shape, dtype=np.uint8)

    if horizontalExpansions is None:
        horizontalExpansions = [0] * len(patternGroups)

    for (detections, height), expansion in zip(patternGroups, horizontalExpansions):
        for y, x, sizeY, sizeX in detections:

            reach_y_start = max(y - height, 0)
            reach_y_end = y

            reach_x_start = max(x - expansion, 0)
            reach_x_end = min(x + sizeX + expansion, shape[1])

            result[reach_y_start:reach_y_end, reach_x_start:reach_x_end] = 1

    for detections, _ in patternGroups:
        for y, x, sizeY, sizeX in detections:
            result[y : y + sizeY, x : x + sizeX] = 0

    return result


def CreateReachTextureFromPatternResult(
    shape: (int, int), detections: list[(int, int, int, int)], height: int
) -> cv.Mat[cv.CV_8U]:
    """
    Returns a mask from the detections positions. Returns 1 if the pixel is part of a detection, 0 otherwise.
    :param detections: the detections to create the mask from
    :param height: the jump height
    :return: the mask
    """
    return CreateReachTextureFromPatternResults(shape, [(detections, height)])


import cv2 as cv
import numpy as np
from numba import njit


@njit
def _compute_reach_loops(reachable, collisionMask, height, width):

    result = np.zeros((height, width), dtype=np.uint32)

    for x in range(width):
        for y in range(1, height):
            if not reachable[y, x] and reachable[y - 1, x]:
                value = 100
                for oy in range(y - 1, 0, -1):
                    if collisionMask[oy, x]:
                        break

                    result[oy, x] += max(0, value)

                    valueX = value
                    for ox in range(x, x - value, -1):
                        if ox < 0 or collisionMask[oy, ox]:
                            break

                        vX = max(0, valueX)
                        result[oy, ox] += vX

                        for ooy in range(oy + 1, height - 1, 1):
                            if collisionMask[ooy, ox]:
                                break

                            result[ooy, ox] += vX

                        valueX -= 1

                    valueX = value
                    for ox in range(x, x + value, 1):
                        if ox >= width or collisionMask[oy, ox]:
                            break

                        vX = max(0, valueX)
                        result[oy, ox] += vX

                        for ooy in range(oy + 1, height - 1, 1):
                            if collisionMask[ooy, ox]:
                                break

                            result[ooy, ox] += vX

                        valueX -= 1

                    value -= 1
    return result


def CreateReachNormalizedTexture(reach: cv.Mat, collisionMask: cv.Mat) -> cv.Mat:
    """
    Version optimisée de la création de texture de reach normalisée.
    Conserve 100% de la logique d'origine mais s'exécute instantanément grâce à Numba.
    """
    height, width = reach.shape
    reachable = reach > 0

    result_compiled = _compute_reach_loops(reachable, collisionMask, height, width)

    result = result_compiled.astype(np.float32)

    max_val = np.max(result)
    if max_val > 0:
        result = result / max_val

    return result


def CreatePathAccessibilityVariance(
    path: list[tuple[int, int]], normalizedReach: cv.Mat
) -> np.ndarray:
    """
    Computes a difficulty-variance curve from a Mario path and a normalized reach map.
    The variance is based on the average reach of Mario's 12x16 hitbox (bottom-right anchored).
    """
    if normalizedReach is None or len(path) < 2:
        return (
            np.zeros((normalizedReach.shape[1],), dtype=np.float32)
            if normalizedReach is not None
            else np.array([], dtype=np.float32)
        )

    height, width = normalizedReach.shape
    variance_by_x = np.zeros((width,), dtype=np.float32)
    counts_by_x = np.zeros((width,), dtype=np.int32)

    mario_width = 12
    mario_height = 16

    def get_area_mean(px: int, py: int) -> float:
        """Extrait la zone 12x16 ancrée en bas à droite et renvoie la moyenne."""
        start_x = max(0, px - mario_width + 1)
        end_x = min(width, px + 1)
        start_y = max(0, py - mario_height + 1)
        end_y = min(height, py + 1)

        roi = normalizedReach[start_y:end_y, start_x:end_x]
        return float(np.mean(roi)) if roi.size > 0 else 0.0

    for (prev_x, prev_y), (x, y) in zip(path, path[1:]):
        dist = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
        if dist > 150:
            continue

        current_x = int(round(x))
        current_y = int(round(y))
        previous_x = int(round(prev_x))
        previous_y = int(round(prev_y))

        if not (
            0 <= current_x < width
            and 0 <= current_y < height
            and 0 <= previous_x < width
            and 0 <= previous_y < height
        ):
            continue

        prev_val = get_area_mean(previous_x, previous_y)
        curr_val = get_area_mean(current_x, current_y)

        delta = prev_val - curr_val

        variance_by_x[current_x] += delta
        counts_by_x[current_x] += 1

    nonzero = counts_by_x > 0
    variance_by_x[nonzero] /= counts_by_x[nonzero].astype(np.float32)

    return variance_by_x


def CreatePathAccessibilityValue(
    path: list[tuple[int, int]], normalizedReach: cv.Mat
) -> np.ndarray:
    """
    Computes an accessibility value curve from a Mario path and a normalized reach map.
    The value is averaged over Mario's 12x16 hitbox (bottom-right anchored).
    """
    height, width = normalizedReach.shape
    variance_by_x = np.zeros((width,), dtype=np.float32)

    mario_width = 12
    mario_height = 16

    for x, y in path:
        curr_x = int(round(x))
        curr_y = int(round(y))

        if 0 <= curr_x < width and 0 <= curr_y < height:

            start_x = max(0, curr_x - mario_width + 1)
            end_x = min(width, curr_x + 1)
            start_y = max(0, curr_y - mario_height + 1)
            end_y = min(height, curr_y + 1)

            roi = normalizedReach[start_y:end_y, start_x:end_x]

            if roi.size > 0:
                valeur = np.mean(roi)
                variance_by_x[curr_x] = float(valeur) / 100.0

    max_val = np.max(variance_by_x)
    if max_val > 0:
        variance_by_x = variance_by_x / max_val

    return variance_by_x


def CreateReachAStarPath(
    start: tuple[int, int],
    goal: tuple[int, int],
    normalizedReach: cv.Mat,
    dangerMask: cv.Mat = None,
    allowDiagonal: bool = True,
) -> list[tuple[int, int]]:
    """
    Finds a path from start to goal using A* over the normalized reach map.
    This pathfinder simulates Mario gravity: whenever the current move is not an upward jump,
    the character falls to the lowest reachable point in that column.
    100 is treated as the easiest location to reach and 0 as impassable.

    :param start: (x, y) start coordinate
    :param goal: (x, y) goal coordinate
    :param normalizedReach: normalized reach map with values from 0 to 100
    :param dangerMask: Optional static danger mask (e.g., holes). Prevents Mario from landing in lethal zones.
    :param allowDiagonal: if True, allows diagonal moves for jump arcs; otherwise uses 4-connected movement
    :return: ordered list of (x, y) positions from start to goal, or [] if no path exists
    """
    if normalizedReach is None:
        return []

    height, width = normalizedReach.shape
    sx, sy = int(round(start[0])), int(round(start[1]))
    gx, gy = int(round(goal[0])), int(round(goal[1]))

    if not (
        0 <= sx < width and 0 <= sy < height and 0 <= gx < width and 0 <= gy < height
    ):
        return []
    if normalizedReach[sy, sx] == 0 or normalizedReach[gy, gx] == 0:
        return []

    def heuristic(x: int, y: int) -> float:
        return float(abs(x - gx) + abs(y - gy))

    def fall_to_lowest(x: int, y: int) -> int:
        while y + 1 < height and normalizedReach[y + 1, x] > 0:
            y += 1
        return y

    sy = fall_to_lowest(sx, sy)
    if normalizedReach[sy, sx] == 0:
        return []

    if sy >= height - 1 or (dangerMask is not None and dangerMask[sy, sx] > 0):
        return []

    neighbors = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    if allowDiagonal:
        neighbors += [
            (-1, -1, 1.41421356),
            (-1, 1, 1.41421356),
            (1, -1, 1.41421356),
            (1, 1, 1.41421356),
        ]

    g_score = np.full((height, width), np.inf, dtype=np.float32)
    g_score[sy, sx] = 0.0

    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    open_heap = []
    counter = 0
    heapq.heappush(open_heap, (heuristic(sx, sy), counter, (sx, sy)))

    while open_heap:
        _, _, (cx, cy) = heapq.heappop(open_heap)
        current_g = g_score[cy, cx]
        if current_g == np.inf:
            continue
        if (cx, cy) == (gx, gy):
            path = [(gx, gy)]
            while path[-1] != (sx, sy):
                path.append(came_from[path[-1]])
            return list(reversed(path))

        for dx, dy, move_cost in neighbors:
            nx = cx + dx
            ny = cy + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if normalizedReach[ny, nx] == 0:
                continue

            if dy >= 0:
                ny = fall_to_lowest(nx, ny)
                if normalizedReach[ny, nx] == 0:
                    continue

                if ny >= height - 1 or (
                    dangerMask is not None and dangerMask[ny, nx] > 0
                ):
                    continue

            step_cost = 101.0 - float(normalizedReach[ny, nx])
            tentative_g = current_g + step_cost * move_cost
            if tentative_g < g_score[ny, nx]:
                g_score[ny, nx] = tentative_g
                counter += 1
                heapq.heappush(
                    open_heap, (tentative_g + heuristic(nx, ny), counter, (nx, ny))
                )
                came_from[(nx, ny)] = (cx, cy)

    return []


def _solve_single_segment(
    start: tuple[int, int],
    goal: tuple[int, int],
    normalizedReach: np.ndarray,
    neighbors: list[tuple[int, int, float]],
) -> list[tuple[int, int]]:
    """
    Calcule le chemin A* pour un unique segment.
    Cette fonction isolée est exécutée en parallèle dans un thread séparé.
    """
    height, width = normalizedReach.shape
    sx, sy = int(round(start[0])), int(round(start[1]))
    gx, gy = int(round(goal[0])), int(round(goal[1]))

    if not (
        0 <= sx < width and 0 <= sy < height and 0 <= gx < width and 0 <= gy < height
    ):
        return []
    if normalizedReach[sy, sx] == 0 or normalizedReach[gy, gx] == 0:
        return []

    g_score = np.full((height, width), np.inf, dtype=np.float32)
    g_score[sy, sx] = 0.0

    came_from = {}
    open_heap = []
    counter = 0

    h_start = float(abs(sx - gx) + abs(sy - gy))
    heapq.heappush(open_heap, (h_start, counter, (sx, sy)))

    max_reach_scale = 100.0 if np.max(normalizedReach) > 2 else 1.0

    while open_heap:
        _, _, (cx, cy) = heapq.heappop(open_heap)

        if (cx, cy) == (gx, gy):
            segment_path = []
            curr = (gx, gy)
            while curr != (sx, sy):
                segment_path.append(curr)
                curr = came_from[curr]
            segment_path.reverse()
            return segment_path

        for dx, dy, move_cost in neighbors:
            nx, ny = cx + dx, cy + dy

            if (
                not (0 <= nx < width and 0 <= ny < height)
                or normalizedReach[ny, nx] == 0
            ):
                continue

            reach_val = float(normalizedReach[ny, nx])
            base_cost = (max_reach_scale + 1.0) - reach_val
            variance_cost = abs(reach_val - float(normalizedReach[cy, cx])) * 2.0

            step_cost = (base_cost + variance_cost) * move_cost
            tentative_g = g_score[cy, cx] + step_cost

            if tentative_g < g_score[ny, nx]:
                g_score[ny, nx] = tentative_g
                came_from[(nx, ny)] = (cx, cy)
                counter += 1
                h_next = float(abs(nx - gx) + abs(ny - gy))
                heapq.heappush(open_heap, (tentative_g + h_next, counter, (nx, ny)))

    return []


def CreateMultiPointAStarPath(
    points: list[tuple[int, int]], normalizedReach: cv.Mat, allowDiagonal: bool = True
) -> list[tuple[int, int]]:
    """
    Calcule un chemin A* passant par une liste ordonnée de points en parallélisant
    le calcul des segments intermédiaires.
    """
    if normalizedReach is None or len(points) < 2:
        return []

    neighbors = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    if allowDiagonal:
        neighbors += [
            (-1, -1, 1.4142),
            (-1, 1, 1.4142),
            (1, -1, 1.4142),
            (1, 1, 1.4142),
        ]

    segments = [(points[i], points[i + 1]) for i in range(len(points) - 1)]

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                _solve_single_segment, start, goal, normalizedReach, neighbors
            )
            for start, goal in segments
        ]
        results = [f.result() for f in futures]

    complete_path = []

    first_start = points[0]
    complete_path.append((int(round(first_start[0])), int(round(first_start[1]))))

    for segment_path in results:

        if not segment_path:
            continue
        complete_path.extend(segment_path)

    return complete_path


def CreateSmoothHighPath(
    start: tuple[int, int],
    goal: tuple[int, int],
    normalizedReach: cv.Mat,
    dangerMask: cv.Mat = None,
    allowDiagonal: bool = True,
) -> list[tuple[int, int]]:
    height, width = normalizedReach.shape
    sx, sy = int(round(start[0])), int(round(start[1]))
    gx, gy = int(round(goal[0])), int(round(goal[1]))

    def fall_to_lowest(x: int, y: int) -> int:
        while y + 1 < height and normalizedReach[y + 1, x] > 0:
            y += 1
        return y

    sy, gy = fall_to_lowest(sx, sy), fall_to_lowest(gx, gy)

    g_score = np.full((height, width), np.inf, dtype=np.float32)

    prev_reach = np.full((height, width), -1.0, dtype=np.float32)

    g_score[sy, sx] = 0.0
    came_from = {}
    open_heap = [(0.0, 0, (sx, sy))]

    def heuristic(x, y):

        return float(abs(x - gx) + abs(y - gy))

    while open_heap:
        priority, _, (cx, cy) = heapq.heappop(open_heap)

        if (cx, cy) == (gx, gy):
            path = [(gx, gy)]
            while path[-1] != (sx, sy):
                path.append(came_from[path[-1]])
            return list(reversed(path))

        for dx, dy, move_cost in [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1)] + (
            [(-1, -1, 1.4), (-1, 1, 1.4), (1, -1, 1.4), (1, 1, 1.4)]
            if allowDiagonal
            else []
        ):
            nx, ny = cx + dx, cy + dy
            if (
                not (0 <= nx < width and 0 <= ny < height)
                or normalizedReach[ny, nx] == 0
            ):
                continue

            if dy >= 0:
                ny = fall_to_lowest(nx, ny)
                if ny >= height - 1 or (
                    dangerMask is not None and dangerMask[ny, nx] > 0
                ):
                    continue

            reach_val = float(normalizedReach[ny, nx])

            base_cost = 101.0 - reach_val

            variance_cost = abs(reach_val - float(normalizedReach[cy, cx])) * 2.0

            altitude_bonus = (ny / height) * 500.0

            step_cost = (base_cost + variance_cost + altitude_bonus) * move_cost
            tentative_g = g_score[cy, cx] + step_cost

            if tentative_g < g_score[ny, nx]:
                g_score[ny, nx] = tentative_g
                came_from[(nx, ny)] = (cx, cy)

                heapq.heappush(
                    open_heap, (tentative_g + heuristic(nx, ny), id((nx, ny)), (nx, ny))
                )

    return []


def MergeDetection(
    detections: list[(int, int, int, int)],
) -> list[(int, int, int, int)]:

    detections.sort(key=lambda box: box[1])

    merged = []

    for box in detections:
        y, x, h, w = box

        if not merged:
            merged.append(box)
            continue

        last_y, last_x, last_h, last_w = merged[-1]

        if y == last_y and h == last_h and x <= last_x + last_w:
            new_x = min(last_x, x)
            new_w = max(last_x + last_w, x + w) - new_x
            merged[-1] = (y, new_x, h, new_w)
        else:
            merged.append(box)

    return merged


def CreateMovingPlatform(
    levelImage: cv.Mat,
    platformImages: list[cv.Mat],
    balancePointsLeft: list[cv.Mat],
    balancePointsRight: list[cv.Mat],
) -> list[(int, int, int, int)]:

    pointsLeft = DetectPatternMulti(levelImage, balancePointsLeft)
    pointsRight = DetectPatternMulti(levelImage, balancePointsRight)

    pointsLeft.sort(key=lambda box: box[0])
    pointsRight.sort(key=lambda box: box[0])

    platforms = DetectPatternMulti(levelImage, platformImages)

    mergedPlatforms = MergeDetection(platforms)

    links = []

    for i, left in enumerate(pointsLeft):
        for i, right in enumerate(pointsRight):
            if left[1] < right[1] and left[0] == right[0]:
                links.append((left, right))
                pointsRight.remove(right)
                break

    mergedPlatforms.sort(key=lambda box: box[0])

    def FindPlatformBelow(point):
        px, py = point[1], point[0]
        for platform in mergedPlatforms:
            y, x, h, w = platform
            if x <= px <= x + w and y >= py:
                return platform
        return None

    result = []

    for left, right in links:
        platformLeft = FindPlatformBelow(left)
        platformRight = FindPlatformBelow(right)

        if platformLeft and platformRight and platformLeft != platformRight:
            yl, xl, hl, wl = platformLeft
            yr, xr, hr, wr = platformRight

            maxHeight = max(yl, yr)
            minHeight = min(yl, yr)

            for y in range(minHeight, maxHeight + 1):
                result.append((y, xl, hl, wl))
                result.append((y, xr, hr, wr))

    for platform in mergedPlatforms:
        if platform not in result:
            result.append(platform)

    return result


def CalculateDifficulty(
    pheromones: cv.Mat[cv.CV_8U], reach: cv.Mat[cv.CV_8U], windowSize: int
) -> np.array(np.float32):
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

        reachable = reach[:, x : x + windowSize] > 0

        dangerous = pheromones[:, x : x + windowSize] > 0

        count = np.sum(np.logical_and(reachable, dangerous))

        if np.isnan(float(np.sum(reachable))):
            result += [0]
            continue

        value = count / float(np.sum(reachable))
        result += [value]

    return result
