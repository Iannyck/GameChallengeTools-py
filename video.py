import cv2
import glob
import json
import os

level = "Niveau_1_1"


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

        img_rgba = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
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

        resized_bgr = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_NEAREST)
        resized_alpha = (
            cv2.resize(alpha, (sw, sh), interpolation=cv2.INTER_NEAREST)
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


def track_mario_fixed():
    folder_path = os.path.join("ressources", level)

    level_img_path = os.path.join(folder_path, "level.png")
    video_path = os.path.join(folder_path, "level.mp4")

    level_bg = cv2.imread(level_img_path, cv2.IMREAD_COLOR)
    if level_bg is None:
        raise FileNotFoundError(f"Impossible de charger {level_img_path}")

    level_h, level_w = level_bg.shape[:2]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
        frame_nes = cv2.resize(
            frame, (scaled_frame_w, level_h), interpolation=cv2.INTER_NEAREST
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

        res_cam = cv2.matchTemplate(
            level_search_region, frame_crop, cv2.TM_CCOEFF_NORMED
        )
        _, max_val_cam, _, max_loc_cam = cv2.minMaxLoc(res_cam)

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
                res_mario = cv2.matchTemplate(
                    play_area_vid, sp["image"], cv2.TM_SQDIFF_NORMED, mask=sp["mask"]
                )

                min_val, _, min_loc, _ = cv2.minMaxLoc(res_mario)
                score = 1.0 - min_val
                match_loc = min_loc
            else:
                res_mario = cv2.matchTemplate(
                    play_area_vid, sp["image"], cv2.TM_CCOEFF_NORMED
                )
                _, score, _, match_loc = cv2.minMaxLoc(res_mario)

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

    path_output = os.path.join(folder_path, "paths.json")

    with open(path_output, "w") as f:
        json.dump([[positions]], f)

    cap.release()


track_mario_fixed()
