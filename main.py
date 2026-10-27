from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import traceback

import cv2 as cv
import gamedifficulty as GD
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESOURCE_DIRECTORY = Path("ressources")
LEVELS_DIRECTORY = RESOURCE_DIRECTORY / "levels"
SPRITE_DIRECTORY = RESOURCE_DIRECTORY / "Sprite"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
WINDOW_SIZES = [224, 256, 288, 304, 320]

_OUTPUT_SUFFIXES = (
    "_staticDanger",
    "_enemyDanger",
    "_collision",
    "_reach",
    "_danger",
    "_demo",
    "_source",
    "_difficulty_all",
)
_SPRITE_SET = None


def discover_level_images() -> list[Path]:
    """Find source images recursively, avoiding this script's generated images."""
    if LEVELS_DIRECTORY.is_dir():
        image_root = LEVELS_DIRECTORY
        candidates = image_root.rglob("*")
    else:
        # Keep the repository's existing ressources/Niveau_*/level.png layout working.
        image_root = RESOURCE_DIRECTORY
        candidates = (
            path for path in image_root.rglob("*") if path.name.lower() == "level.png"
        )

    return sorted(
        path
        for path in candidates
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not is_generated_image(path)
    )


def is_generated_image(path: Path) -> bool:
    stem = path.stem
    return (
        any(stem.endswith(suffix) for suffix in _OUTPUT_SUFFIXES)
        or "_difficulty_" in stem
    )


def initialize_worker() -> None:
    global _SPRITE_SET
    cv.setNumThreads(1)
    _SPRITE_SET = GD.Classes.SpriteSet(str(SPRITE_DIRECTORY))


def save_figure(figure, output_path: Path) -> None:
    try:
        figure.savefig(output_path, bbox_inches="tight")
    finally:
        plt.close(figure)


def save_image(image: np.ndarray, output_path: Path) -> None:
    if not cv.imwrite(str(output_path), image):
        raise OSError(f"Could not save image: {output_path}")


def process_level_image(image_path: Path) -> str:
    if _SPRITE_SET is None:
        raise RuntimeError("Worker sprite set was not initialized.")

    level_image = cv.imread(str(image_path), cv.IMREAD_COLOR)
    if level_image is None:
        raise ValueError(f"Could not read image: {image_path}")

    output_prefix = f"{image_path.stem}_{image_path.suffix.lower().lstrip('.')}"

    def output_path(name: str) -> Path:
        return image_path.with_name(f"{output_prefix}_{name}.png")

    sprite_set = _SPRITE_SET

    collision_positions = GD.Detection.DetectPatternMulti(
        level_image, sprite_set.GetCollisionsTextures(), 0.85
    )
    collision_positions += GD.Detection.DetectPatternMulti(
        level_image, sprite_set.GetPipesTextures(), 0.85
    )

    passthrough_positions = GD.Processing.CreateMovingPlatform(
        level_image,
        sprite_set.GetPlatformsTextures(),
        sprite_set.GetBalancePointsLeft(),
        sprite_set.GetBalancePointsRight(),
    )

    collision_mask = GD.Processing.CreateMaskFromPatternResult(
        collision_positions, level_image.shape[:2]
    )
    reach = GD.Processing.CreateReachTextureFromPatternResult(
        level_image.shape[:2],
        collision_positions + passthrough_positions,
        int(GD.Constants.jumpHeight),
    )
    danger = GD.Processing.CreateStaticDanger(collision_mask)
    # save_image(danger * 255, output_path("staticDanger"))

    enemy_danger = np.zeros(level_image.shape[:2], dtype=np.uint8)
    for enemy_type in GD.Types.EnemyType.GetAllTypes():
        enemy_positions = GD.Detection.DetectPatternMulti(
            level_image, sprite_set.GetEnemyTextures(enemy_type), 0.85
        )
        displacement = GD.Processing.CreateDisplacementTexture(
            enemy_type, enemy_positions, collision_mask
        )
        enemy_danger = np.maximum(displacement, enemy_danger)

    danger = np.maximum(danger, enemy_danger)

    difficulty_curves = []
    for window_size in WINDOW_SIZES:
        difficulty_curve = GD.Processing.CalculateDifficulty(danger, reach, window_size)
        difficulty_curves.append(difficulty_curve)

        figure, axis = plt.subplots()
        axis.plot(difficulty_curve)
        axis.set_ylim(0, 1.2)
        axis.set_title(f"Difficulty for {image_path.stem}, window size {window_size}")
        save_figure(figure, output_path(f"difficulty_{window_size}"))

    figure, axis = plt.subplots()
    for index, curve in enumerate(difficulty_curves):
        axis.plot(curve, label=f"Window size {WINDOW_SIZES[index]}")
    axis.legend()
    axis.set_ylim(0, 1)
    axis.set_title(f"Difficulty for {image_path.stem}")
    save_figure(figure, output_path("difficulty_all"))

    figure, axis = plt.subplots()
    axis.imshow(cv.cvtColor(level_image, cv.COLOR_BGR2RGB))
    axis.axis("off")
    save_figure(figure, output_path("source"))

    demo_image = np.zeros(level_image.shape, dtype=np.uint8)
    demo_image[:, :, 0] = collision_mask * 255
    demo_image[:, :, 1] = reach * 255
    demo_image[:, :, 2] = danger * 255

    # save_image(reach * 255, output_path("reach"))
    # save_image(collision_mask * 255, output_path("collision"))
    # save_image(danger * 255, output_path("danger"))
    # save_image(enemy_danger * 255, output_path("enemyDanger"))
    # save_image(demo_image, output_path("demo"))

    return str(image_path)


def main() -> None:
    image_paths = discover_level_images()
    if not image_paths:
        raise FileNotFoundError(
            f"No supported level images found under {LEVELS_DIRECTORY} "
            f"or {RESOURCE_DIRECTORY}."
        )

    worker_count = min(os.cpu_count() or 1, len(image_paths))
    print(f"Processing {len(image_paths)} images with {worker_count} workers.")

    failures = []
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=initialize_worker,
    ) as executor:
        futures = {
            executor.submit(process_level_image, image_path): image_path
            for image_path in image_paths
        }
        for future in as_completed(futures):
            image_path = futures[future]
            try:
                print(f"Processed {future.result()}")
            except Exception:
                failures.append(image_path)
                print(f"Failed to process {image_path}:", flush=True)
                traceback.print_exc()

    if failures:
        raise RuntimeError(
            f"Failed to process {len(failures)} of {len(image_paths)} images."
        )


if __name__ == "__main__":
    main()
