import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import json
import os
import glob
import cv2 as cv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gamedifficulty as GD


def save_path_data(path_values: list[float], level_name, filename: str):
    folder = f"ressources/{level_name}/path_results"
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(f"{folder}/{filename}.txt", "w") as f:
        f.write("P W\n")
        for x_coord in range(0, len(path_values), 16):
            val = path_values[x_coord]
            f.write(f"{x_coord} {val:.6g}\n")


def draw_path(image, path, color, thickness=2):
    if not path:
        return

    layer = np.zeros_like(image)
    for i in range(1, len(path)):
        prev = (int(round(path[i - 1][0])), int(round(path[i - 1][1])))
        curr = (int(round(path[i][0])), int(round(path[i][1])))

        dist = np.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)
        if dist > 150:

            cv.circle(layer, prev, 6, (255, 255, 255), 2)
            cv.circle(layer, curr, 6, (255, 255, 255), 2)
            continue

        cv.line(layer, prev, curr, color, thickness)
        cv.circle(layer, curr, 3, color, -1)

    start = (int(round(path[0][0])), int(round(path[0][1])))
    cv.circle(layer, start, 5, (0, 0, 255), -1)
    goal = (int(round(path[-1][0])), int(round(path[-1][1])))
    cv.circle(layer, goal, 5, (0, 255, 0), -1)

    return layer


def save_all_paths(
    paths: list[list[tuple[int, int]]], level_name, base_image, type_image
):
    colors = [(0, 0, 255), (209, 177, 16), (209, 16, 132)]
    folder = f"ressources/{level_name}/path_results"
    if not os.path.exists(folder):
        os.makedirs(folder)

    for idx, path in enumerate(paths):
        color = colors[idx % len(colors)]
        file_name = f"path_{idx + 1}_{type_image}"
        individual_img = base_image.copy()
        path_layer = np.zeros_like(individual_img)

        for i in range(1, len(path)):
            prev = (int(round(path[i - 1][0])), int(round(path[i - 1][1])))
            curr = (int(round(path[i][0])), int(round(path[i][1])))

            dist = np.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)
            if dist > 150:
                continue

            cv.line(path_layer, prev, curr, color, 2)
            cv.circle(path_layer, curr, 3, color, -1)

        if path:
            cv.circle(
                path_layer,
                (int(round(path[0][0])), int(round(path[0][1]))),
                5,
                (0, 0, 255),
                -1,
            )
            cv.circle(
                path_layer,
                (int(round(path[-1][0])), int(round(path[-1][1]))),
                5,
                (0, 255, 0),
                -1,
            )

        mask = cv.cvtColor(path_layer, cv.COLOR_BGR2GRAY) > 0
        individual_img[mask] = path_layer[mask]
        cv.imwrite(f"{folder}/{file_name}.png", individual_img)

    paths_accumulation_layer = np.zeros_like(base_image)

    for idx, path in enumerate(paths):
        color = colors[idx % len(colors)]
        current_layer = np.zeros_like(base_image)

        for i in range(1, len(path)):
            prev = (int(round(path[i - 1][0])), int(round(path[i - 1][1])))
            curr = (int(round(path[i][0])), int(round(path[i][1])))

            dist = np.sqrt((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2)
            if dist > 150:
                continue

            cv.line(current_layer, prev, curr, color, 2)
            cv.circle(current_layer, curr, 3, color, -1)

        if path:
            cv.circle(
                current_layer,
                (int(round(path[0][0])), int(round(path[0][1]))),
                5,
                (0, 0, 255),
                -1,
            )
            cv.circle(
                current_layer,
                (int(round(path[-1][0])), int(round(path[-1][1]))),
                5,
                (0, 255, 0),
                -1,
            )

        mask_current = cv.cvtColor(current_layer, cv.COLOR_BGR2GRAY) > 0
        mask_existing = cv.cvtColor(paths_accumulation_layer, cv.COLOR_BGR2GRAY) > 0
        intersection_mask = mask_current & mask_existing
        single_path_mask = mask_current & ~mask_existing

        paths_accumulation_layer[single_path_mask] = current_layer[single_path_mask]
        if np.any(intersection_mask):
            blended_pixels = cv.addWeighted(
                paths_accumulation_layer, 0.5, current_layer, 0.5, 0
            )
            paths_accumulation_layer[intersection_mask] = blended_pixels[
                intersection_mask
            ]

    combined_img = base_image.copy()
    final_mask = cv.cvtColor(paths_accumulation_layer, cv.COLOR_BGR2GRAY) > 0
    combined_img[final_mask] = paths_accumulation_layer[final_mask]
    cv.imwrite(f"{folder}/all_paths_{type_image}.png", combined_img)


def _sanitize_paths(raw_data):
    def is_number(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    clean_paths = []
    if not isinstance(raw_data, list):
        return clean_paths

    for path in raw_data:
        if not isinstance(path, list):
            continue
        clean_path = []
        for seg in path:
            if not isinstance(seg, list):
                continue
            clean_seg = []
            for pt in seg:
                if (
                    isinstance(pt, (list, tuple))
                    and len(pt) >= 2
                    and is_number(pt[0])
                    and is_number(pt[1])
                ):
                    clean_seg.append((pt[0], pt[1]))
            if clean_seg:
                clean_path.append(clean_seg)
        if clean_path:
            clean_paths.append(clean_path)

    return clean_paths


def show_graph_for_path(
    path: list[tuple[int, int]], reach: cv.Mat, danger: cv.Mat, pathName, level
):

    greenPathValue = GD.Processing.CreatePathAccessibilityValue(path, reach)
    save_path_data(greenPathValue, level, f"{pathName}_green_values")
    plt.figure(figsize=(10, 3))
    plt.plot(greenPathValue, label="Green reach value")
    plt.title(f"Green value graph for {pathName}")
    plt.xlabel("Level X coordinate")
    plt.ylabel("Green reach value")
    plt.ylim(0.0, 1.1)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"ressources/{level}/path_results/{pathName}_green_values")
    plt.close()

    greenPathVariance = GD.Processing.CreatePathAccessibilityVariance(path, reach)
    save_path_data(greenPathVariance, level, f"{pathName}_green_variation")
    plt.figure(figsize=(10, 3))
    plt.plot(greenPathVariance, label="Green reach variation")
    plt.title(f"Green variation graph for {pathName}")
    plt.xlabel("Level X coordinate")
    plt.ylabel("Green reach variation")
    plt.ylim(np.min(greenPathVariance) * 1.1, np.max(greenPathVariance) * 1.1)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"ressources/{level}/path_results/{pathName}_green_variation")
    plt.close()

    redPathValue = GD.Processing.CreatePathAccessibilityValue(path, danger)
    save_path_data(redPathValue, level, f"{pathName}_red_values")
    plt.figure(figsize=(10, 3))
    plt.plot(redPathValue, label="Red reach value")
    plt.title(f"Red value graph for {pathName}")
    plt.xlabel("Level X coordinate")
    plt.ylabel("Red reach value")
    plt.ylim(0.0, 1.1)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"ressources/{level}/path_results/{pathName}_red_values")
    plt.close()

    redPathVariance = GD.Processing.CreatePathAccessibilityVariance(path, danger)
    save_path_data(redPathVariance, level, f"{pathName}_red_variation")
    plt.figure(figsize=(10, 3))
    plt.plot(redPathVariance, label="Red reach variation")
    plt.title(f"Red variation graph for {pathName}")
    plt.xlabel("Level X coordinate")
    plt.ylabel("Red reach variation")
    plt.ylim(np.min(redPathVariance) * 1.1, np.max(redPathVariance) * 1.1)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"ressources/{level}/path_results/{pathName}_red_variation")
    plt.close()

    mergeValue = greenPathValue * redPathValue
    save_path_data(mergeValue, level, f"{pathName}_red_and_green_value")
    plt.figure(figsize=(10, 3))
    plt.plot(mergeValue, label="Green and Red values")
    plt.title(f"Green and Red values graph for {pathName}")
    plt.xlabel("Level X coordinate")
    plt.ylabel("Green and Red values")
    plt.ylim(0.0, 1.1)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"ressources/{level}/path_results/{pathName}_red_and_green_value")
    plt.close()

    mergeVariation = greenPathVariance * redPathVariance
    save_path_data(mergeVariation, level, f"{pathName}_red_and_green_variation")
    plt.figure(figsize=(10, 3))
    plt.plot(mergeVariation, label="Green and Red variation")
    plt.title(f"Green and Red variation graph for {pathName}")
    plt.xlabel("Level X coordinate")
    plt.ylabel("Green and Red variation")
    plt.ylim(np.min(mergeVariation) * 1.1, np.max(mergeVariation) * 1.1)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"ressources/{level}/path_results/{pathName}_red_and_green_variation")
    plt.close()


def run_analysis(level):
    """
    Exécute toute la logique de l'ex-main.py pour un niveau donné.
    """
    print(f"--- Début de l'analyse pour le niveau : {level} ---")
    levelImage = cv.imread(f"ressources/{level}/level.png")
    spriteSet = GD.Classes.SpriteSet("ressources/Sprite")

    collisionPositions = GD.Detection.DetectPatternMulti(
        levelImage, spriteSet.GetCollisionsTextures(), 0.85
    )
    collisionPositions += GD.Detection.DetectPatternMulti(
        levelImage, spriteSet.GetPipesTextures(), 0.85
    )

    jumpBoardPositions = GD.Detection.DetectPatternMulti(
        levelImage, spriteSet.GetJumpBoardTextures(), 0.85
    )
    collisionPositions += jumpBoardPositions

    passthroughPositions = GD.Processing.CreateMovingPlatform(
        levelImage,
        spriteSet.GetPlatformsTextures(),
        spriteSet.GetBalancePointsLeft(),
        spriteSet.GetBalancePointsRight(),
    )

    collisionMask = GD.Processing.CreateMaskFromPatternResult(
        collisionPositions, levelImage.shape[:2]
    )
    staticDanger = GD.Processing.CreateStaticDanger(collisionMask)
    cv.imwrite(f"ressources/{level}/staticDanger.png", staticDanger * 255)

    reach = GD.Processing.CreateReachTextureFromPatternResults(
        levelImage.shape[:2],
        [
            (collisionPositions + passthroughPositions, int(GD.Constants.jumpHeight)),
            (jumpBoardPositions, int(GD.Constants.jumpBoardHeight)),
        ],
    )

    normalizedReachMap = GD.Processing.CreateReachNormalizedTexture(
        reach, collisionMask
    )
    cv.imwrite(
        f"ressources/{level}/normalized_reach.png",
        (((normalizedReachMap * 100.0).astype(np.uint8) / 100) * 255).astype(np.uint8),
    )

    enemyDanger = np.zeros(levelImage.shape[:2], dtype=np.uint8)
    enemyDetections = {}
    for enemyType in GD.Types.EnemyType.GetAllTypes():
        enemyPositions = GD.Detection.DetectPatternMulti(
            levelImage, spriteSet.GetEnemyTextures(enemyType), 0.85
        )
        enemyDetections[enemyType] = enemyPositions
        ed = GD.Processing.CreateDisplacementTexture(
            enemyType, enemyPositions, collisionMask
        )
        enemyDanger = np.maximum(ed, enemyDanger)

    danger = np.maximum(staticDanger, enemyDanger)
    enemyPheromoneMap = GD.Processing.CreateEnemyPheromoneMap(
        enemyDetections, collisionMask
    )
    cv.imwrite(
        f"ressources/{level}/enemy_pheromone_map.png",
        (np.clip(enemyPheromoneMap, 0.0, 1.0) * 255).astype(np.uint8),
    )

    accessibleDanger = GD.Processing.CreateAccessibleDangerMap(
        collisionMask, enemyDetections, normalizedReachMap
    )
    cv.imwrite(
        f"ressources/{level}/accessible_danger_map.png",
        (np.clip(accessibleDanger, 0.0, 1.0) * 255).astype(np.uint8),
    )

    cv.imwrite(f"ressources/{level}/reach.png", reach * 255)
    cv.imwrite(f"ressources/{level}/collision.png", collisionMask * 255)
    cv.imwrite(f"ressources/{level}/danger.png", danger * 255)
    cv.imwrite(f"ressources/{level}/enemyDanger.png", enemyDanger * 255)

    json_path = f"ressources/{level}/paths.json"
    raw_paths = []
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            raw_paths = json.load(f)

    # Sanitize loaded paths to ensure points are valid (no None values)
    raw_paths = _sanitize_paths(raw_paths)

    paths_to_save = []
    for path_segments in raw_paths:
        full_path = []
        for segment_coords in path_segments:
            if len(segment_coords) < 2:
                print("⚠️ Un segment a moins de 2 points, ignoré.")
                continue

            formatted_points = [(pt[0], pt[1]) for pt in segment_coords]
            computed_segment = GD.Processing.CreateMultiPointAStarPath(
                formatted_points, normalizedReachMap, True
            )

            if computed_segment:
                full_path.extend(computed_segment)

        if full_path:
            paths_to_save.append(full_path)

    height, width = normalizedReachMap.shape
    overlay_img = np.zeros((height, width, 3), dtype=np.uint8)

    reach_intensity = (np.clip(normalizedReachMap, 0.0, 1.0) * 255).astype(np.uint8)
    danger_intensity = (np.clip(accessibleDanger, 0.0, 1.0) * 255).astype(np.uint8)
    overlay_img[:, :, 2] = danger_intensity
    overlay_img[:, :, 1] = reach_intensity

    save_all_paths(paths_to_save, level, overlay_img, "overlay")
    save_all_paths(paths_to_save, level, levelImage, "level")

    for i in range(len(paths_to_save)):
        path_name = f"Path_{i+1}"
        show_graph_for_path(
            paths_to_save[i], normalizedReachMap, accessibleDanger, path_name, level
        )

    cv.imwrite(f"ressources/{level}/reach_danger_overlay.png", overlay_img)
    print("--- Analyse terminée avec succès ! ---")


def load_sprites_with_mask(scale_factor):
    """
    Charge les sprites et crée un masque Alpha pour ignorer le fond transparent.
    """
    folder_path = os.path.join("ressources", "Sprite", "Mario")

    sprites = []
    valid_exts = ("*.png", "*.jpg", "*.bmp")
    files = []
    for ext in valid_exts:
        files.extend(glob.glob(os.path.join(folder_path, ext)))

    for filepath in files:

        img_rgba = cv.imread(filepath, cv.IMREAD_UNCHANGED)
        if img_rgba is None:
            continue

        if img_rgba.shape[2] == 4:
            bgr = img_rgba[:, :, :3]
            alpha = img_rgba[:, :, 3]
        else:
            bgr = img_rgba
            alpha = None

        sw = int(round(bgr.shape[1] * scale_factor))
        sh = int(round(bgr.shape[0] * scale_factor))

        resized_bgr = cv.resize(bgr, (sw, sh), interpolation=cv.INTER_NEAREST)
        resized_alpha = (
            cv.resize(alpha, (sw, sh), interpolation=cv.INTER_NEAREST)
            if alpha is not None
            else None
        )

        sprites.append(
            {
                "name": os.path.basename(filepath),
                "image": resized_bgr,
                "mask": resized_alpha,
                "w_nes": bgr.shape[1],
                "h_nes": bgr.shape[0],
                "w_vid": sw,
                "h_vid": sh,
            }
        )
    return sprites


def track_mario_fixed(level, video_path):
    folder_path = os.path.join("ressources", level)

    level_img_path = os.path.join(folder_path, "level.png")

    level_bg = cv.imread(level_img_path, cv.IMREAD_COLOR)
    if level_bg is None:
        raise FileNotFoundError(f"Impossible de charger {level_img_path}")

    level_h, level_w = level_bg.shape[:2]
    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir {video_path}")

    fps = cap.get(cv.CAP_PROP_FPS)
    vid_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    scale_factor = vid_h / float(level_h)
    scaled_sprites = load_sprites_with_mask(scale_factor)

    hud_height_nes = 32
    hud_height_vid = int(hud_height_nes * scale_factor)

    last_camera_x = 0
    max_camera_speed = 8
    search_window = 120

    tracking_data = []
    positions = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        scaled_frame_w = int(round(vid_w / scale_factor))
        frame_nes = cv.resize(
            frame, (scaled_frame_w, level_h), interpolation=cv.INTER_NEAREST
        )

        frame_crop = frame_nes[hud_height_nes:, :]
        level_crop = level_bg[hud_height_nes:, :]

        if frame_idx == 0:

            search_min_x = 0
            search_max_x = min(500, level_w - scaled_frame_w)
        else:

            search_min_x = max(0, int(last_camera_x - 10))
            search_max_x = min(
                level_w - scaled_frame_w, int(last_camera_x + search_window)
            )

        level_search_region = level_crop[
            :, search_min_x : search_max_x + scaled_frame_w
        ]

        res_cam = cv.matchTemplate(level_search_region, frame_crop, cv.TM_CCOEFF_NORMED)
        _, max_val_cam, _, max_loc_cam = cv.minMaxLoc(res_cam)

        camera_x = search_min_x + max_loc_cam[0]

        if camera_x >= last_camera_x:
            last_camera_x = camera_x

        play_area_vid = frame[hud_height_vid:, :]
        best_mario = {"val": -1.0, "x_vid": None, "y_vid": None, "sprite": None}

        for sp in scaled_sprites:
            if (
                sp["h_vid"] > play_area_vid.shape[0]
                or sp["w_vid"] > play_area_vid.shape[1]
            ):
                continue

            if sp["mask"] is not None:
                res_mario = cv.matchTemplate(
                    play_area_vid, sp["image"], cv.TM_SQDIFF_NORMED, mask=sp["mask"]
                )

                min_val, _, min_loc, _ = cv.minMaxLoc(res_mario)
                score = 1.0 - min_val
                match_loc = min_loc
            else:
                res_mario = cv.matchTemplate(
                    play_area_vid, sp["image"], cv.TM_CCOEFF_NORMED
                )
                _, score, _, match_loc = cv.minMaxLoc(res_mario)

            if score > best_mario["val"]:
                best_mario["val"] = score
                best_mario["x_vid"] = match_loc[0] + sp["w_vid"] // 2
                best_mario["y_vid"] = match_loc[1] + sp["h_vid"] // 2 + hud_height_vid
                best_mario["sprite"] = sp["name"]

        if best_mario["val"] >= 0.45 and best_mario["x_vid"] is not None:
            mario_screen_x_nes = best_mario["x_vid"] / scale_factor
            mario_screen_y_nes = best_mario["y_vid"] / scale_factor

            world_x = round(last_camera_x + mario_screen_x_nes, 2)
            world_y = round(mario_screen_y_nes, 2)
            detected_sprite = best_mario["sprite"]
            confidence = round(float(best_mario["val"]), 3)
        else:
            world_x, world_y = None, None
            detected_sprite = None
            confidence = (
                round(float(best_mario["val"]), 3) if best_mario["val"] > 0 else 0.0
            )

        timestamp = round(frame_idx / fps, 3) if fps > 0 else 0.0

        tracking_data.append(
            {
                "frame": frame_idx,
                "timestamp_sec": timestamp,
                "camera_x": last_camera_x,
                "world_x": world_x,
                "world_y": world_y,
                "sprite": detected_sprite,
                "confidence": confidence,
            }
        )

        positions.append([world_x, world_y])

        frame_idx += 1

    cap.release()

    return positions


def analyze_video_folder(level):
    video_folder = os.path.join("ressources", level, "Video")

    files = glob.glob(os.path.join(video_folder, "*.mp4"))

    positions = []
    for file in files:
        positions.append([track_mario_fixed(level, file)])

    path_output = os.path.join("ressources", level, "paths.json")

    with open(path_output, "w") as f:
        json.dump(positions, f)


class PathEditor:
    def __init__(self, root, level_name):
        self.root = root
        self.level_name = level_name
        self.root.title(f"Mario Level - Éditeur de Path ({self.level_name})")

        self.all_paths = []
        self.current_segments = [[]]
        self.image = None
        self.tk_image = None
        self.image_path = f"ressources/{self.level_name}/level.png"

        self.colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta"]

        self.setup_ui()
        self.root.after(100, self.load_level_data)

    def setup_ui(self):
        control_frame = tk.Frame(self.root, bg="#333333")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Button(control_frame, text="➕ Nouveau Chemin", command=self.next_path).pack(
            side=tk.LEFT, padx=10, pady=10
        )

        tk.Button(
            control_frame,
            text="🚪 Ajouter une Téléportation (Tuyau)",
            command=self.add_teleport,
            bg="#2196F3",
            fg="white",
        ).pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(
            control_frame, text="↩️ Annuler dernier point", command=self.undo_point
        ).pack(side=tk.LEFT, padx=10, pady=10)

        self.run_btn = tk.Button(
            control_frame,
            text="💾 Sauvegarder & Lancer l'analyse",
            command=self.save_and_run,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
        )
        self.run_btn.pack(side=tk.RIGHT, padx=10, pady=10)

        self.video_btn = tk.Button(
            control_frame,
            text="🎬 Analyser les Vidéos",
            command=self.save_and_analyze_videos,
            bg="#FF9800",
            fg="white",
            font=("Arial", 10, "bold"),
        )
        self.video_btn.pack(side=tk.RIGHT, padx=10, pady=10)

        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="gray", cursor="crosshair")
        hbar = tk.Scrollbar(
            canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview
        )
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click)

    def load_level_data(self):
        if not os.path.exists(self.image_path):
            messagebox.showerror(
                "Erreur", f"L'image {self.image_path} est introuvable !"
            )
            return

        self.image = Image.open(self.image_path)
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.config(scrollregion=(0, 0, self.image.width, self.image.height))

        json_file_path = f"ressources/{self.level_name}/paths.json"
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, "r") as f:

                    raw_data = json.load(f)
                    self.all_paths = _sanitize_paths(raw_data)
            except Exception as e:
                print(f"Impossible de charger l'ancien json : {e}")

        self.current_segments = [[]]
        self.redraw_paths()

    def on_click(self, event):
        if not self.image:
            return
        x = int(self.canvas.canvasx(event.x))
        y = int(self.canvas.canvasy(event.y))

        self.current_segments[-1].append((x, y))
        self.redraw_paths()

    def add_teleport(self):
        if self.current_segments[-1]:

            self.current_segments.append([])
            self.redraw_paths()

    def next_path(self):

        if any(self.current_segments):
            self.all_paths.append(self.current_segments)
            self.current_segments = [[]]
            self.redraw_paths()

    def undo_point(self):
        if not self.current_segments:
            return

        if self.current_segments[-1]:
            self.current_segments[-1].pop()
        elif len(self.current_segments) > 1:
            self.current_segments.pop()
            if self.current_segments[-1]:
                self.current_segments[-1].pop()

        self.redraw_paths()

    def redraw_paths(self):
        self.canvas.delete("all")
        if self.tk_image:
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        for idx, path_segments in enumerate(self.all_paths):
            color = self.colors[idx % len(self.colors)]
            self.draw_path_segments(path_segments, color)

        current_color = self.colors[len(self.all_paths) % len(self.colors)]
        self.draw_path_segments(self.current_segments, current_color)

    def draw_path_segments(self, segments, color):
        r = 3
        last_pt = None
        for seg in segments:
            if not seg:
                continue

            if last_pt:
                self.canvas.create_line(
                    last_pt[0],
                    last_pt[1],
                    seg[0][0],
                    seg[0][1],
                    fill=color,
                    dash=(4, 4),
                    width=2,
                )

            for i, (x, y) in enumerate(seg):
                self.canvas.create_oval(
                    x - r, y - r, x + r, y + r, fill=color, outline=color
                )
                if i > 0:
                    px, py = seg[i - 1]
                    self.canvas.create_line(px, py, x, y, fill=color, width=2)

            last_pt = seg[-1]

    def save_and_run(self):
        paths_to_export = list(self.all_paths)
        if any(self.current_segments):
            paths_to_export.append(self.current_segments)

        if not paths_to_export:
            messagebox.showwarning("Attention", "Aucun point n'a été placé !")
            return

        json_file_path = f"ressources/{self.level_name}/paths.json"

        try:
            with open(json_file_path, "w") as f:
                json.dump(paths_to_export, f, indent=4)

            self.run_btn.config(
                text="Analyse en cours...", bg="orange", state=tk.DISABLED
            )
            self.root.update()

            run_analysis(self.level_name)

            messagebox.showinfo("Succès", "L'analyse a été effectuée avec succès !")
        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur est survenue :\n{e}")
        finally:
            self.run_btn.config(
                text="💾 Sauvegarder & Lancer l'analyse", bg="#4CAF50", state=tk.NORMAL
            )

    def save_and_analyze_videos(self):
        paths_to_export = list(self.all_paths)
        if any(self.current_segments):
            paths_to_export.append(self.current_segments)

        if not paths_to_export:
            messagebox.showwarning("Attention", "Aucun point n'a été placé !")
            return

        json_file_path = f"ressources/{self.level_name}/paths.json"

        try:
            with open(json_file_path, "w") as f:
                json.dump(paths_to_export, f, indent=4)

            self.video_btn.config(
                text="Analyse vidéos...", bg="orange", state=tk.DISABLED
            )
            self.root.update()

            analyze_video_folder(self.level_name)

            messagebox.showinfo(
                "Succès", "L'analyse des vidéos a été effectuée avec succès !"
            )
        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur est survenue :\n{e}")
        finally:
            self.video_btn.config(
                text="🎬 Sauvegarder & Analyser Vidéo", bg="#FF9800", state=tk.NORMAL
            )


if __name__ == "__main__":

    TARGET_LEVEL = "Niveau_1_1"

    root = tk.Tk()
    root.geometry("1400x700")
    app = PathEditor(root, TARGET_LEVEL)
    root.mainloop()
