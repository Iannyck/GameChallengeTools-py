import cv2 as cv
import heapq
import numpy as np
from gamedifficulty.Constants import *
from gamedifficulty.Types import EnemyType
from gamedifficulty.Detection import DetectPatternMulti


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


def CreateDisplacementTexture(ennemyType: EnemyType, detections: list[(int, int, int, int)],
                              collisionMask: cv.Mat[cv.CV_8U]) -> cv.Mat[cv.CV_8U]:
    """
    Creates a displacement texture for an enemy type.
    :param ennemyType: the type of enemy
    :param detections: the detections positions and sizes of the enemy
    :param collisionMask: the collision mask
    :return: the displacement texture
    """
    if ennemyType == EnemyType.GOOMBA or ennemyType == EnemyType.KOOPA or ennemyType == EnemyType.TURTLE:
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


def CreateEnemyPheromoneMap(enemyDetections: dict[EnemyType, list[(int, int, int, int)]],
                             collisionMask: cv.Mat[cv.CV_8U],
                             pheromoneValues: dict[EnemyType, float] = None) -> np.ndarray:
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

        # Process each enemy individually so overlaps accumulate and scale by area
        for detection in detections:
            singleDetection = [detection]
            displacement = CreateDisplacementTexture(enemyType, singleDetection, collisionMask)
            area = float(np.count_nonzero(displacement))
            if area <= 0.0:
                continue

            # area fraction in level
            area_frac = area / total_pixels

            # smaller area -> larger scale; clamp the scale
            scale = 1.0 + (1.0 - area_frac) * scaleFactor
            if scale > maxScale:
                scale = maxScale

            pheromones += displacement.astype(np.float32) * (base_weight * scale)

    return pheromones


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

            if collisionMask[y:y + sizeY - 1, x if direction == -1 else x + sizeX].any():
                direction = 1 if direction == -1 else -1

            x += direction

            result[y:y + sizeY, x:x + sizeX] = 1

            # if any of the pixels outside of image break
            if y < 0 or y + sizeY >= collisionMask.shape[0] or x < 0 or x + sizeX >= collisionMask.shape[1]:
                break

    return result


def CreateRedKoopaDisplacementTexture(detections: list[(int, int, int, int)], collisionMask: cv.Mat[cv.CV_8U]) -> cv.Mat[cv.CV_8U]:
    """
    Red Koopa specific implementation of CreateDisplacementTexture.
    Red Koopas move horizontally but do not fall off edges, they reverse when a ledge or wall is ahead.
    """
    result = np.zeros(collisionMask.shape, dtype=np.uint8)
    levelHeight, levelWidth = collisionMask.shape

    for (y, x, sizeY, sizeX) in detections:
        current_y = y
        current_x = x
        direction = -1
        iteration = 0
        max_iterations = 1000

        while iteration < max_iterations:
            iteration += 1

            if current_y < 0 or current_y + sizeY >= levelHeight or current_x < 0 or current_x + sizeX > levelWidth:
                break

            # fall until on solid ground
            while current_y + sizeY < levelHeight and not collisionMask[current_y + sizeY, current_x:current_x + sizeX].any():
                current_y -= int(np.sign(gravity))
                if current_y < 0 or current_y + sizeY >= levelHeight:
                    break

            if current_y < 0 or current_y + sizeY >= levelHeight or current_x < 0 or current_x + sizeX > levelWidth:
                break

            result[current_y:current_y + sizeY, current_x:current_x + sizeX] = 1

            front_x = current_x + direction
            support_x_start = front_x
            support_x_end = front_x + sizeX
            has_support_ahead = (
                0 <= support_x_start and support_x_end <= levelWidth and
                collisionMask[current_y + sizeY, support_x_start:support_x_end].any()
            )

            front_edge_x = current_x - 1 if direction == -1 else current_x + sizeX
            has_wall_ahead = (
                front_edge_x < 0 or front_edge_x >= levelWidth or
                collisionMask[current_y:current_y + sizeY - 1, front_edge_x].any()
            )

            if not has_support_ahead or has_wall_ahead:
                opposite_direction = -direction
                opposite_front_x = current_x + opposite_direction
                opposite_support_x_start = opposite_front_x
                opposite_support_x_end = opposite_front_x + sizeX
                opposite_has_support = (
                    0 <= opposite_support_x_start and opposite_support_x_end <= levelWidth and
                    collisionMask[current_y + sizeY, opposite_support_x_start:opposite_support_x_end].any()
                )
                opposite_front_edge_x = current_x - 1 if opposite_direction == -1 else current_x + sizeX
                opposite_has_wall = (
                    opposite_front_edge_x < 0 or opposite_front_edge_x >= levelWidth or
                    collisionMask[current_y:current_y + sizeY - 1, opposite_front_edge_x].any()
                )

                if not opposite_has_support or opposite_has_wall:
                    break

                direction = opposite_direction
                continue

            current_x += direction

    return result

def CreateBulletBillDisplacementTexture(detections: list[(int, int, int, int)], collisionMask: cv.Mat[cv.CV_8U]) -> cv.Mat[cv.CV_8U]:
    """
    Bullet Bill specific implementation of CreateDisplacementTexture
    """
    result = np.zeros(collisionMask.shape, dtype=np.uint8)

    for (y, x, sizeY, sizeX) in detections:
        result[y:y + sizeY, x:x + sizeX] = 1
        
        for currX in range(0, collisionMask.shape[1] - sizeX + 1):
            result[y:y + sizeY, currX:currX + sizeX] = 1

    return result


def CreateFlyingFishDisplacementTexture(detections: list[(int, int, int, int)], collisionMask: cv.Mat[cv.CV_8U]) -> cv.Mat[cv.CV_8U]:
    """
    Flying Fish specific implementation of CreateDisplacementTexture.
    Flying fish move horizontally across the level while oscillating vertically,
    so we mark the full horizontal span and a vertical band around the detected position.
    """
    result = np.zeros(collisionMask.shape, dtype=np.uint8)
    levelWidth = collisionMask.shape[1]
    levelHeight = collisionMask.shape[0]

    for (y, x, sizeY, sizeX) in detections:
        result[y:y + sizeY, x:x + sizeX] = 1

        vertical_margin = max(1, sizeY * 2)
        top = max(0, y - vertical_margin)
        bottom = min(levelHeight, y + sizeY + vertical_margin)

        for currX in range(0, levelWidth - sizeX + 1):
            result[top:bottom, currX:currX + sizeX] = 1

    return result


def CreateLakituDisplacementTexture(detections: list[(int, int, int, int)], collisionMask: cv.Mat[cv.CV_8U]) -> cv.Mat[cv.CV_8U]:
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

    for (y, x, sizeY, sizeX) in detections:
        safe_y_end = max(0, y)
        if safe_y_end > 0:
            result[0:safe_y_end, :] = 0

    return result


def CreatePiranaPlantDisplacementTexture(detections: list[(int, int, int, int)], collisionMask: cv.Mat[cv.CV_8U]) -> \
cv.Mat[cv.CV_8U]:
    # they don't move

    result = np.zeros(collisionMask.shape, dtype=np.uint8)

    for (y, x, sizeY, sizeX) in detections:
        result[y:y + sizeY, x:x + sizeX] = 1

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
    
    # Expansion proportional to sqrt of height ratio
    expansion_factor = np.sqrt(jumpHeight / baseJumpHeight)
    # expansion is the additional distance compared to base jump
    expansion = int(jumpHalfWidth * (expansion_factor - 1))
    return expansion


def CreateReachTextureFromPatternResults(shape: (int, int), patternGroups: list[tuple[list[(int, int, int, int)], int]], 
                                        horizontalExpansions: list[int] = None) -> cv.Mat[cv.CV_8U]:
    """
    Creates a reach mask from one or more groups of platform detections.
    Each group may use a different jump height, for example normal platforms and trampoline/jump pad platforms.
    :param shape: the shape of the output mask
    :param patternGroups: list of (detections, height) tuples
    :param horizontalExpansions: optional list of horizontal expansions for each group
    :return: the reach texture
    """
    result = np.zeros(shape, dtype=np.uint8)
    
    # If no expansions provided, use 0 for all groups
    if horizontalExpansions is None:
        horizontalExpansions = [0] * len(patternGroups)
    
    for (detections, height), expansion in zip(patternGroups, horizontalExpansions):
        for (y, x, sizeY, sizeX) in detections:
            # Apply vertical reach (above the platform)
            reach_y_start = max(y - height, 0)
            reach_y_end = y
            
            # Apply horizontal expansion (left and right of the platform)
            reach_x_start = max(x - expansion, 0)
            reach_x_end = min(x + sizeX + expansion, shape[1])
            
            result[reach_y_start:reach_y_end, reach_x_start:reach_x_end] = 1

    # Remove the platforms themselves from reach (can't be inside them)
    for detections, _ in patternGroups:
        for (y, x, sizeY, sizeX) in detections:
            result[y:y + sizeY, x:x + sizeX] = 0

    return result


def CreateReachTextureFromPatternResult(shape: (int, int), detections: list[(int, int, int, int)], height: int) -> \
cv.Mat[cv.CV_8U]:
    """
    Returns a mask from the detections positions. Returns 1 if the pixel is part of a detection, 0 otherwise.
    :param detections: the detections to create the mask from
    :param height: the jump height
    :return: the mask
    """
    return CreateReachTextureFromPatternResults(shape, [(detections, height)])


def CreateReachNormalizedTexture(reach: cv.Mat[cv.CV_8U], collisionMask: cv.Mat[cv.CV_8U]) -> cv.Mat[cv.CV_8U]:
    """
    Creates a normalized reach texture from an existing reach mask.
    For each "ground" point where the pixel is not reachable but the pixel immediately above is,
    assigns a value of 100 and then creates a vertical gradient upwards in the same column,
    subtracting 1 for each pixel until reaching 0.
    It also detects horizontal platform edges and propagates values left and right across each row,
    then extends those values downward in each unobstructed column.
    :param reach: binary reach mask where 1 means reachable and 0 means not reachable
    :return: normalized reach texture with values from 0 to 100
    """
    height, width = reach.shape
    reachable = reach > 0
    collision = collisionMask > 0

    # Build vertical seed values from the ground edge points.
    rowValues = np.zeros(reach.shape, dtype=np.uint8)
    seeds = np.nonzero(np.logical_and(~reachable[1:, :], reachable[:-1, :]))
    for seed_y, seed_x in zip(seeds[0], seeds[1]):
        y = seed_y + 1
        value = 100
        oy = y - 1
        while oy > 0 and value > 0:
            if collision[oy, seed_x]:
                break
            if rowValues[oy, seed_x] < value:
                rowValues[oy, seed_x] = value
            oy -= 1
            value -= 1

    result = np.zeros_like(rowValues)

    # Horizontal propagation along each row from the vertical sources.
    for oy in range(1, height):
        row = rowValues[oy]
        if not row.any():
            continue

        row_ext = row.copy()
        current = 0

        for ox in range(width):
            if collision[oy, ox]:
                current = 0
                continue
            current = max(current - 1, int(row[ox])) if current > 0 else int(row[ox])
            if current > row_ext[ox]:
                row_ext[ox] = current

        current = 0
        for ox in range(width - 1, -1, -1):
            if collision[oy, ox]:
                current = 0
                continue
            current = max(current - 1, int(row[ox])) if current > 0 else int(row[ox])
            if current > row_ext[ox]:
                row_ext[ox] = current

        result[oy] = row_ext

    # Downward extension from each row value along columns.
    for ox in range(width):
        current = 0
        for oy in range(height - 1):
            if collision[oy, ox]:
                current = 0
                continue
            current = max(current, int(result[oy, ox]))
            if current > result[oy, ox]:
                result[oy, ox] = current

    return result


def CreatePathDifficultyVariance(path: list[tuple[int, int]], normalizedReach: cv.Mat[cv.CV_8U]) -> np.ndarray:
    """
    Computes a difficulty-variance curve from a Mario path and a normalized reach map.
    Each successive step on the path produces a delta between the normalized reach value
    at the new point and the previous point. Results are aggregated per x coordinate,
    so the returned array can be plotted as a progression over the level x axis.

    :param path: ordered list of (x, y) positions representing Mario's path
    :param normalizedReach: normalized reach map with values from 0 to 100
    :return: 1D float32 array indexed by x coordinate with average signed variance
    """
    if normalizedReach is None or len(path) < 2:
        return np.zeros((normalizedReach.shape[1],), dtype=np.float32) if normalizedReach is not None else np.array([], dtype=np.float32)

    height, width = normalizedReach.shape
    variance_by_x = np.zeros((width,), dtype=np.float32)
    counts_by_x = np.zeros((width,), dtype=np.int32)

    for (prev_x, prev_y), (x, y) in zip(path, path[1:]):
        current_x = int(round(x))
        current_y = int(round(y))
        previous_x = int(round(prev_x))
        previous_y = int(round(prev_y))

        if not (0 <= current_x < width and 0 <= current_y < height and 0 <= previous_x < width and 0 <= previous_y < height):
            continue

        delta = float(normalizedReach[previous_y, previous_y]) - float(normalizedReach[current_y, current_x])
        print(delta)
        variance_by_x[current_x] += delta
        counts_by_x[current_x] += 1

    nonzero = counts_by_x > 0
    variance_by_x[nonzero] /= counts_by_x[nonzero].astype(np.float32)

    return variance_by_x


def CreateReachAStarPath(start: tuple[int, int], goal: tuple[int, int], normalizedReach: cv.Mat[cv.CV_8U], allowDiagonal: bool = True) -> list[tuple[int, int]]:
    """
    Finds a coherent path from start to goal using A* over the normalized reach map.
    Higher normalized reach values are preferred: 100 is lowest traversal cost and 0 is impassable.

    :param start: (x, y) start coordinate
    :param goal: (x, y) goal coordinate
    :param normalizedReach: normalized reach map with values from 0 to 100
    :param allowDiagonal: if True, allows 8-connected movement; otherwise uses 4-connected movement
    :return: ordered list of (x, y) positions from start to goal, or [] if no path exists
    """
    if normalizedReach is None:
        return []

    height, width = normalizedReach.shape
    sx, sy = int(round(start[0])), int(round(start[1]))
    gx, gy = int(round(goal[0])), int(round(goal[1]))

    if not (0 <= sx < width and 0 <= sy < height and 0 <= gx < width and 0 <= gy < height):
        return []
    if normalizedReach[sy, sx] == 0 or normalizedReach[gy, gx] == 0:
        return []

    def heuristic(x: int, y: int) -> float:
        return float(abs(x - gx) + abs(y - gy))

    neighbors = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    if allowDiagonal:
        neighbors += [(-1, -1, 1.41421356), (-1, 1, 1.41421356), (1, -1, 1.41421356), (1, 1, 1.41421356)]

    g_score = np.full((height, width), np.inf, dtype=np.float32)
    g_score[sy, sx] = 0.0

    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    open_heap = []
    counter = 0
    heapq.heappush(open_heap, (heuristic(sx, sy), counter, (sx, sy)))

    while open_heap:
        _, _, (cx, cy) = heapq.heappop(open_heap)
        if (cx, cy) == (gx, gy):
            path = [(gx, gy)]
            while path[-1] != (sx, sy):
                path.append(came_from[path[-1]])
            return list(reversed(path))

        current_g = g_score[cy, cx]
        for dx, dy, move_cost in neighbors:
            nx = cx + dx
            ny = cy + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if normalizedReach[ny, nx] == 0:
                continue

            step_cost = 101.0 - float(normalizedReach[ny, nx])
            tentative_g = current_g + step_cost * move_cost
            if tentative_g < g_score[ny, nx]:
                g_score[ny, nx] = tentative_g
                counter += 1
                heapq.heappush(open_heap, (tentative_g + heuristic(nx, ny), counter, (nx, ny)))
                came_from[(nx, ny)] = (cx, cy)

    return []


def MergeDetection(detections: list[(int, int, int, int)]) -> list[(int, int, int, int)]:
    # Sort by x-axis (left coordinate)
    detections.sort(key=lambda box: box[1])

    merged = []

    for box in detections:
        y, x, h, w = box

        if not merged:
            merged.append(box)
            continue

        last_y, last_x, last_h, last_w = merged[-1]

        # Merge if on same horizontal line (exact y and height) and x-ranges overlap
        if y == last_y and h == last_h and x <= last_x + last_w:
            new_x = min(last_x, x)
            new_w = max(last_x + last_w, x + w) - new_x
            merged[-1] = (y, new_x, h, new_w)
        else:
            merged.append(box)

    return merged


def CreateMovingPlatform(levelImage: cv.Mat, platformImages: list[cv.Mat], balancePointsLeft: list[cv.Mat],
                              balancePointsRight: list[cv.Mat]) -> list[(int, int, int, int)]:
    # find balance points
    pointsLeft = DetectPatternMulti(levelImage, balancePointsLeft)
    pointsRight = DetectPatternMulti(levelImage, balancePointsRight)

    pointsLeft.sort(key=lambda box: box[0])
    pointsRight.sort(key=lambda box: box[0])

    # find platforms
    platforms = DetectPatternMulti(levelImage, platformImages)

    # merge platforms that are next to each other
    mergedPlatforms = MergeDetection(platforms)

    links = []

    # link each left point to its right point
    for i, left in enumerate(pointsLeft):
        for i, right in enumerate(pointsRight):
            if left[1] < right[1] and left[0] == right[0]:
                links.append((left, right))
                pointsRight.remove(right)
                break

    # sort from top to bottom
    mergedPlatforms.sort(key=lambda box: box[0])

    def FindPlatformBelow(point):
        px, py = point[1], point[0]
        for platform in mergedPlatforms:
            y, x, h, w = platform
            if x <= px <= x + w and y >= py:
                return platform
        return None

    result = []

    # find for each link which platform is under the right and what platform is under the left
    for left, right in links:
        platformLeft = FindPlatformBelow(left)
        platformRight = FindPlatformBelow(right)

        if platformLeft and platformRight and platformLeft != platformRight:
            yl, xl, hl, wl = platformLeft
            yr, xr, hr, wr = platformRight

            maxHeight = max(yl, yr)
            minHeight = min(yl, yr)

            for y in range(minHeight, maxHeight+1):
                result.append((y, xl, hl, wl))
                result.append((y, xr, hr, wr))

    # add missing moving platforms
    for platform in mergedPlatforms:
        if platform not in result:
            result.append(platform)

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
