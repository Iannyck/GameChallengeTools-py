import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import json
import os
import cv2 as cv
import numpy as np

# --- AJOUTE CES DEUX LIGNES ICI ---
import matplotlib

matplotlib.use("Agg")
# ----------------------------------

import matplotlib.pyplot as plt
import gamedifficulty as GD

# ==========================================
# FONCTIONS D'ANALYSE (Ex-main.py)
# ==========================================


def save_path_data(path_values: list[float], level_name, filename: str):
    folder = f"ressources/{level_name}"
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
    folder = f"ressources/{level_name}"
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


def show_graph_for_path(
    path: list[tuple[int, int]], reach: cv.Mat, danger: cv.Mat, pathName, level
):
    # Green reach value
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
    plt.savefig(f"ressources/{level}/{pathName}_green_values")
    plt.close()  # Important pour libérer la mémoire lors des appels multiples

    # Green reach variation
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
    plt.savefig(f"ressources/{level}/{pathName}_green_variation")
    plt.close()

    # Red danger value
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
    plt.savefig(f"ressources/{level}/{pathName}_red_values")
    plt.close()

    # Red danger variation
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
    plt.savefig(f"ressources/{level}/{pathName}_red_variation")
    plt.close()

    # Merged Value
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
    plt.savefig(f"ressources/{level}/{pathName}_red_and_green_value")
    plt.close()

    # Merged Variation
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
    plt.savefig(f"ressources/{level}/{pathName}_red_and_green_variation")
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

    paths_to_save = []
    for path_coords in raw_paths:
        formatted_points = [(point[0], point[1]) for point in path_coords]
        computed_path = GD.Processing.CreateMultiPointAStarPath(
            formatted_points, normalizedReachMap, True
        )
        if computed_path:
            paths_to_save.append(computed_path)

    height, width = normalizedReachMap.shape
    overlay_img = np.zeros((height, width, 3), dtype=np.uint8)

    reach_intensity = (np.clip(normalizedReachMap, 0.0, 1.0) * 255).astype(np.uint8)
    danger_intensity = (np.clip(accessibleDanger, 0.0, 1.0) * 255).astype(np.uint8)
    overlay_img[:, :, 2] = danger_intensity
    overlay_img[:, :, 1] = reach_intensity

    save_all_paths(paths_to_save, level, overlay_img, "overlay")
    save_all_paths(paths_to_save, level, levelImage, "level")

    # Si tu as plus de 2 chemins tu peux générer dynamiquement le nom
    for i in range(len(paths_to_save)):
        path_name = f"Path_{i+1}"
        show_graph_for_path(
            paths_to_save[i], normalizedReachMap, accessibleDanger, path_name, level
        )

    cv.imwrite(f"ressources/{level}/reach_danger_overlay.png", overlay_img)
    print("--- Analyse terminée avec succès ! ---")


# ==========================================
# INTERFACE GRAPHIQUE (Ex-path_editor_gui.py)
# ==========================================


class PathEditor:
    def __init__(self, root, level_name):
        self.root = root
        self.level_name = level_name
        self.root.title(f"Mario Level - Éditeur de Path ({self.level_name})")

        self.all_paths = []
        self.current_path = []
        self.image = None
        self.tk_image = None
        self.image_path = f"ressources/{self.level_name}/level.png"

        self.colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta"]

        self.setup_ui()

        # Charge l'image directement après le lancement
        self.root.after(100, self.load_level_data)

    def setup_ui(self):
        control_frame = tk.Frame(self.root, bg="#333333")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Button(control_frame, text="➕ Nouveau Chemin", command=self.next_path).pack(
            side=tk.LEFT, padx=10, pady=10
        )
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

        # Chargement automatique des chemins s'ils existent déjà
        json_file_path = f"ressources/{self.level_name}/paths.json"
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, "r") as f:
                    self.all_paths = json.load(f)
                    # Conversion des listes en tuples pour Tkinter
                    self.all_paths = [
                        [(pt[0], pt[1]) for pt in path] for path in self.all_paths
                    ]
            except Exception as e:
                print(f"Impossible de charger l'ancien json : {e}")

        self.current_path = []
        self.redraw_paths()

    def on_click(self, event):
        if not self.image:
            return
        x = int(self.canvas.canvasx(event.x))
        y = int(self.canvas.canvasy(event.y))
        self.current_path.append((x, y))
        self.redraw_paths()

    def next_path(self):
        if self.current_path:
            self.all_paths.append(self.current_path)
            self.current_path = []
            self.redraw_paths()

    def undo_point(self):
        if self.current_path:
            self.current_path.pop()
            self.redraw_paths()

    def redraw_paths(self):
        self.canvas.delete("all")
        if self.tk_image:
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        for idx, path in enumerate(self.all_paths):
            color = self.colors[idx % len(self.colors)]
            self.draw_path(path, color)

        current_color = self.colors[len(self.all_paths) % len(self.colors)]
        self.draw_path(self.current_path, current_color)

    def draw_path(self, path, color):
        r = 3
        for i, (x, y) in enumerate(path):
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r, fill=color, outline=color
            )
            if i > 0:
                px, py = path[i - 1]
                self.canvas.create_line(px, py, x, y, fill=color, width=2)

    def save_and_run(self):
        paths_to_export = list(self.all_paths)
        if self.current_path:
            paths_to_export.append(self.current_path)

        if not paths_to_export:
            messagebox.showwarning("Attention", "Aucun point n'a été placé !")
            return

        json_file_path = f"ressources/{self.level_name}/paths.json"

        try:
            with open(json_file_path, "w") as f:
                json.dump(paths_to_export, f, indent=4)
            print(f"[OK] Fichier sauvegardé : {json_file_path}")

            self.run_btn.config(
                text="Analyse en cours...", bg="orange", state=tk.DISABLED
            )
            self.root.update()  # Rafraîchit l'interface pour afficher le bouton en orange

            # Appel direct de la fonction de calcul
            run_analysis(self.level_name)

            messagebox.showinfo(
                "Succès",
                "L'analyse a été effectuée avec succès !\nLes images et graphiques ont été mis à jour.",
            )

        except Exception as e:
            messagebox.showerror(
                "Erreur", f"Une erreur est survenue durant l'analyse :\n{e}"
            )

        finally:
            # Remet le bouton à son état normal une fois terminé
            self.run_btn.config(
                text="💾 Sauvegarder & Lancer l'analyse", bg="#4CAF50", state=tk.NORMAL
            )


if __name__ == "__main__":
    # Paramètre global du niveau (tu peux le modifier ici selon tes besoins)
    TARGET_LEVEL = "Niveau_8_3"

    root = tk.Tk()
    root.geometry("1400x700")
    app = PathEditor(root, TARGET_LEVEL)
    root.mainloop()
