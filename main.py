from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
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
WINDOW_SIZES = [256]
PEAK_MIN_PROMINENCE = 0.05

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
_REFERENCE_CURVES = None
_REFERENCE_IMAGE_PATH = None
AREA_COMPARISON_COLUMNS = (
    "image",
    "algorithm",
    "area_under_curve",
    "reference_area_under_curve",
    "area_delta",
    "absolute_area_delta",
    "area_between_curves",
    "difficulty_peak_count",
    "mean_peak_spacing_px",
    "peaks_per_1000px",
    "reference_peak_count",
    "reference_mean_peak_spacing_px",
    "reference_peaks_per_1000px",
)
AreaComparisonValue = str | int | float | None


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


def initialize_worker(
    reference_curves=None, reference_image_path: Path | None = None
) -> None:
    global _REFERENCE_CURVES, _REFERENCE_IMAGE_PATH, _SPRITE_SET
    cv.setNumThreads(1)
    _SPRITE_SET = GD.Classes.SpriteSet(str(SPRITE_DIRECTORY))
    _REFERENCE_CURVES = reference_curves
    _REFERENCE_IMAGE_PATH = reference_image_path


def save_figure(figure, output_path: Path) -> None:
    try:
        figure.savefig(output_path, bbox_inches="tight")
    finally:
        plt.close(figure)


def save_image(image: np.ndarray, output_path: Path) -> None:
    if not cv.imwrite(str(output_path), image):
        raise OSError(f"Could not save image: {output_path}")


def process_level_image(image_path: Path) -> dict[int, np.ndarray]:
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

    difficulty_curves = {}
    for window_size in WINDOW_SIZES:
        difficulty_curve = np.asarray(
            GD.Processing.CalculateDifficulty(danger, reach, window_size),
            dtype=np.float64,
        )
        difficulty_curves[window_size] = difficulty_curve

        figure, axis = plt.subplots()
        axis.plot(difficulty_curve)
        axis.set_ylim(0, 1.2)
        axis.set_title(f"Difficulty for {image_path.stem}, window size {window_size}")
        save_figure(figure, output_path(f"difficulty_{window_size}"))

    figure, axis = plt.subplots()
    for window_size, curve in difficulty_curves.items():
        axis.plot(curve, label=f"Window size {window_size}")
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

    return difficulty_curves


def find_reference_image(image_paths: list[Path]) -> Path:
    reference_folder = "Niveau_1_1".casefold()
    candidates = [
        path
        for path in image_paths
        if any(part.casefold() == reference_folder for part in path.parts[:-1])
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Expected exactly one source image in a folder named Niveau_1_1, "
            f"found {len(candidates)}."
        )
    return candidates[0]


def calculate_curve_areas(
    curve: np.ndarray, reference_curve: np.ndarray
) -> tuple[float, float, float]:
    curve = np.asarray(curve, dtype=np.float64)
    reference_curve = np.asarray(reference_curve, dtype=np.float64)
    if curve.ndim != 1 or reference_curve.ndim != 1:
        raise ValueError("Difficulty curves must be one-dimensional.")
    if len(curve) < 2 or len(reference_curve) < 2:
        raise ValueError("Difficulty curves must contain at least two points.")
    if not np.isfinite(curve).all() or not np.isfinite(reference_curve).all():
        raise ValueError("Difficulty curves contain non-finite values.")

    curve_x = np.linspace(0.0, 1.0, len(curve))
    reference_x = np.linspace(0.0, 1.0, len(reference_curve))
    curve_area = float(np.trapezoid(curve, x=curve_x))
    reference_area = float(np.trapezoid(reference_curve, x=reference_x))

    comparison_x = np.union1d(curve_x, reference_x)
    difference = np.interp(comparison_x, curve_x, curve) - np.interp(
        comparison_x, reference_x, reference_curve
    )
    crossing_indexes = np.flatnonzero(difference[:-1] * difference[1:] < 0)
    crossing_x = comparison_x[crossing_indexes] - (
        difference[crossing_indexes]
        * (comparison_x[crossing_indexes + 1] - comparison_x[crossing_indexes])
        / (difference[crossing_indexes + 1] - difference[crossing_indexes])
    )
    area_x = np.sort(np.concatenate((comparison_x, crossing_x)))
    area_difference = np.interp(area_x, comparison_x, difference)
    area_between = float(np.trapezoid(np.abs(area_difference), x=area_x))
    return curve_area, reference_area, area_between


def calculate_curve_pacing(
    curve: np.ndarray,
    window_size: int,
    min_prominence: float = PEAK_MIN_PROMINENCE,
) -> dict[str, int | float | None]:
    values = np.asarray(curve, dtype=np.float64)
    if values.ndim != 1 or len(values) < 3:
        raise ValueError("Difficulty curve must contain at least three points.")
    if not np.isfinite(values).all():
        raise ValueError("Difficulty curve contains non-finite values.")
    if window_size < 1:
        raise ValueError("Window size must be positive.")
    if min_prominence < 0:
        raise ValueError("Minimum peak prominence cannot be negative.")

    tolerance = 1e-12
    candidates = []
    index = 1
    while index < len(values) - 1:
        plateau_end = index
        while (
            plateau_end + 1 < len(values) - 1
            and abs(values[plateau_end + 1] - values[index]) <= tolerance
        ):
            plateau_end += 1

        if (
            values[index] > values[index - 1] + tolerance
            and values[plateau_end] > values[plateau_end + 1] + tolerance
        ):
            peak_index = (index + plateau_end) // 2
            peak_value = values[peak_index]

            higher_left = np.flatnonzero(values[:peak_index] > peak_value + tolerance)
            higher_right = np.flatnonzero(
                values[peak_index + 1 :] > peak_value + tolerance
            )
            left_bound = int(higher_left[-1] + 1) if len(higher_left) else 0
            right_bound = (
                peak_index + 1 + int(higher_right[0])
                if len(higher_right)
                else len(values) - 1
            )
            prominence = peak_value - max(
                float(np.min(values[left_bound : peak_index + 1])),
                float(np.min(values[peak_index : right_bound + 1])),
            )
            if prominence >= min_prominence:
                candidates.append((peak_index, prominence, peak_value))

        index = plateau_end + 1

    selected_peaks = []
    for candidate in sorted(
        candidates, key=lambda peak: (peak[1], peak[2]), reverse=True
    ):
        if all(
            abs(candidate[0] - selected[0]) >= window_size
            for selected in selected_peaks
        ):
            selected_peaks.append(candidate)

    peak_indices = sorted(peak[0] for peak in selected_peaks)
    spacings = np.diff(peak_indices)
    level_width = len(values) + window_size
    return {
        "difficulty_peak_count": len(peak_indices),
        "mean_peak_spacing_px": (float(np.mean(spacings)) if len(spacings) else None),
        "peaks_per_1000px": len(peak_indices) * 1000.0 / level_width,
    }


def build_area_comparison_rows(
    image_path: Path,
    curves: dict[int, np.ndarray],
    reference_curves: dict[int, np.ndarray],
) -> list[dict[str, AreaComparisonValue]]:
    root = LEVELS_DIRECTORY if LEVELS_DIRECTORY.is_dir() else RESOURCE_DIRECTORY
    image_name = image_path.relative_to(root).as_posix()
    rows = []

    for window_size in WINDOW_SIZES:
        curve = curves[window_size]
        reference_curve = reference_curves[window_size]
        curve_area, reference_area, area_between = calculate_curve_areas(
            curve, reference_curve
        )
        curve_pacing = calculate_curve_pacing(curve, window_size)
        reference_pacing = calculate_curve_pacing(reference_curve, window_size)
        area_delta = curve_area - reference_area
        rows.append(
            {
                "image": image_name,
                "algorithm": image_path.parent.name,
                "area_under_curve": curve_area,
                "reference_area_under_curve": reference_area,
                "area_delta": area_delta,
                "absolute_area_delta": abs(area_delta),
                "area_between_curves": area_between,
                **curve_pacing,
                **{
                    f"reference_{key.removeprefix('difficulty_')}": value
                    for key, value in reference_pacing.items()
                },
            }
        )
    return rows


def process_and_compare_image(
    image_path: Path,
) -> tuple[str, list[dict[str, AreaComparisonValue]]]:
    if _REFERENCE_CURVES is None or _REFERENCE_IMAGE_PATH is None:
        raise RuntimeError("Reference difficulty curves were not initialized.")

    curves = process_level_image(image_path)
    return str(image_path), build_area_comparison_rows(
        image_path, curves, _REFERENCE_CURVES
    )


def write_area_comparison(rows: list[dict[str, AreaComparisonValue]]) -> Path:
    output_root = LEVELS_DIRECTORY if LEVELS_DIRECTORY.is_dir() else RESOURCE_DIRECTORY
    output_path = output_root / "difficulty_area_comparison.csv"
    rows.sort(key=lambda row: (str(row["algorithm"]), str(row["image"])))
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=AREA_COMPARISON_COLUMNS, delimiter=";"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        format(value, ".17g").replace(".", ",")
                        if isinstance(value, float)
                        else value
                    )
                    for key, value in row.items()
                }
            )
    return output_path


def main() -> None:
    image_paths = discover_level_images()
    if not image_paths:
        raise FileNotFoundError(
            f"No supported level images found under {LEVELS_DIRECTORY} "
            f"or {RESOURCE_DIRECTORY}."
        )

    reference_path = find_reference_image(image_paths)
    initialize_worker(reference_image_path=reference_path)
    reference_curves = process_level_image(reference_path)
    comparison_rows = build_area_comparison_rows(
        reference_path, reference_curves, reference_curves
    )

    images_to_process = [
        image_path for image_path in image_paths if image_path != reference_path
    ]
    failures = []
    if images_to_process:
        worker_count = min(os.cpu_count() or 1, len(images_to_process))
        print(
            f"Processing {len(images_to_process)} images with {worker_count} workers "
            f"(reference: {reference_path})."
        )

        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=initialize_worker,
            initargs=(reference_curves, reference_path),
        ) as executor:
            futures = {
                executor.submit(process_and_compare_image, image_path): image_path
                for image_path in images_to_process
            }
            for future in as_completed(futures):
                image_path = futures[future]
                try:
                    processed_path, rows = future.result()
                    comparison_rows.extend(rows)
                    print(f"Processed {processed_path}")
                except Exception:
                    failures.append(image_path)
                    print(f"Failed to process {image_path}:", flush=True)
                    traceback.print_exc()

    if not failures:
        print(f"Area comparison saved to {write_area_comparison(comparison_rows)}")

    if failures:
        raise RuntimeError(
            f"Failed to process {len(failures)} of {len(images_to_process)} images."
        )


if __name__ == "__main__":
    main()
