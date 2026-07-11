import tkinter as tk
import json
import os
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class PathEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Mario Level - Éditeur de Path")

        # État interne
        self.all_paths = []
        self.current_path = []
        self.image = None
        self.tk_image = None

        # Couleurs pour différencier les multiples chemins
        self.colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta"]

        self.setup_ui()

        # Demande automatiquement une image au lancement
        self.root.after(100, self.prompt_load_image)

    def setup_ui(self):
        # Panneau de contrôle (Boutons en haut)
        control_frame = tk.Frame(self.root, bg="#333333")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Button(
            control_frame, text="📁 Charger Image (PNG)", command=self.prompt_load_image
        ).pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(control_frame, text="➕ Nouveau Chemin", command=self.next_path).pack(
            side=tk.LEFT, padx=10, pady=10
        )
        tk.Button(
            control_frame, text="↩️ Annuler dernier point", command=self.undo_point
        ).pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(
            control_frame,
            text="💾 Générer Code Python",
            command=self.generate_code,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(side=tk.RIGHT, padx=10, pady=10)

        # Canvas avec Barres de défilement (Scrollbars)
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

        # Binding du clic gauche
        self.canvas.bind("<Button-1>", self.on_click)

    def prompt_load_image(self):
        filepath = filedialog.askopenfilename(
            title="Sélectionne l'image du niveau",
            filetypes=[("PNG Files", "*.png"), ("All Files", "*.*")],
        )
        if not filepath:
            return

        self.image_path = filepath  # <-- On sauvegarde le chemin du fichier
        self.image = Image.open(filepath)
        self.tk_image = ImageTk.PhotoImage(self.image)

        self.canvas.config(scrollregion=(0, 0, self.image.width, self.image.height))

        self.all_paths = []
        self.current_path = []
        self.redraw_paths()

    def on_click(self, event):
        if not self.image:
            return

        # Obtenir les vraies coordonnées de l'image (en prenant en compte le scroll)
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

        # Dessiner les anciens chemins verrouillés
        for idx, path in enumerate(self.all_paths):
            color = self.colors[idx % len(self.colors)]
            self.draw_path(path, color)

        # Dessiner le chemin en cours de création
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

    def generate_code(self):
        # Ajoute le chemin actuel à la liste d'export s'il n'est pas vide
        paths_to_export = list(self.all_paths)
        if self.current_path:
            paths_to_export.append(self.current_path)

        if not paths_to_export:
            messagebox.showwarning("Attention", "Aucun point n'a été placé !")
            return

        if not hasattr(self, "image_path") or not self.image_path:
            messagebox.showerror("Erreur", "Aucune image n'a été chargée.")
            return

        # Récupération du dossier où se trouve l'image du level
        level_dir = os.path.dirname(self.image_path)
        json_file_path = os.path.join(level_dir, "paths.json")

        try:
            # Sauvegarde en format JSON
            with open(json_file_path, "w") as f:
                json.dump(paths_to_export, f, indent=4)

            messagebox.showinfo(
                "Succès", f"Tes chemins ont été sauvegardés dans :\n{json_file_path}"
            )
        except Exception as e:
            messagebox.showerror(
                "Erreur", f"Impossible de sauvegarder le fichier JSON :\n{e}"
            )


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1400x700")  # Grande fenêtre par défaut
    app = PathEditor(root)
    root.mainloop()
