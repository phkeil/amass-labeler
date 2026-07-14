#!/usr/bin/env python3
"""AMASS 3D Data Labeling Utility Application.

Description
-----------
This application loads raw AMASS coordinate movement profiles (.npz files), 
passes them sequentially through the converted SMPL framework structure, and 
renders the skeleton dynamics within a 3D Matplotlib animation container. 
It facilitates categorical sorting (Very Useful, Useful, Not Useful) to isolate 
relevant motion vectors for specialized tracking configurations.

Environment Setup
-----------------
1. Open a terminal shell and navigate to the local project container folder:
   cd path/to/REPO_ROOT/amass-labeler

2. Build the isolated runtime environment utilizing the provided environment configuration file:
   conda env create -f environment_amass_labeler.yml

Execution Instructions
----------------------
1. Initialize the environment:
   conda activate amass_labeler

2. Launch the graphical tracking controller interface:
   python amass_labeler.py

3. Choose or paste the target absolute directory mapping pathway containing your 
   unpacked AMASS subfolders in the top-left section and press Enter.

4. Interact with the selection elements to flag and append specific movement behaviors. 
   Labels are automatically saved to the amass_labels.csv file.

5. After labelling, copy the amass_labels.csv file to the place specified in projector_pipeline.py
   as AMASS_LABELS_PATH, to ensure the labels are taken into account while dataset generation.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import queue
import threading
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import torch
import numpy as np
import smplx
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.animation as animation


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_PATH = REPO_ROOT / "smpl" / "models" / "converted_basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"
SAVE_PATH = SCRIPT_DIR / "amass_labels.csv"

# Numeric to Text mapping for display
LABEL_MAP = {
    "2": "🌟 Very Useful (Fall)",
    "1": "✅ Useful (Walk)",
    "0": "❌ Not Useful",
    "None": "Unlabeled"
}

DEFAULT_BTN_STYLE = {
    "bg": "#f0f0f0",
    "fg": "#000000",
    "activebackground": "#e5e5e5",
    "activeforeground": "#000000",
    "disabledforeground": "#a0a0a0",
    "relief": tk.RAISED,
    "bd": 3,
    "font": ("TkDefaultFont", 10, "normal"),
}

PRESSED_BTN_STYLE = {
    "bg": "#bdc3c7",
    "fg": "#000000",
    "activebackground": "#bdc3c7",
    "activeforeground": "#000000",
    "disabledforeground": "#000000",
    "relief": tk.SOLID,
    "bd": 3,
    "font": ("TkDefaultFont", 10, "bold"),
}

LIST_COLOR_MAP = {
    "2": "#d5f5e3",
    "1": "#e6f9ef",
    "0": "#fadbd8",
}


class AnimationWorker(threading.Thread):
    def __init__(self, file_path, model, device, result_queue):
        super().__init__(daemon=True)
        self.file_path = file_path
        self.model = model
        self.device = device
        self.result_queue = result_queue

    def run(self):
        try:
            bdata = np.load(self.file_path)
            step = 10
            poses = torch.tensor(bdata['poses'][::step], dtype=torch.float32, device=self.device)
            trans = torch.tensor(bdata['trans'][::step], dtype=torch.float32, device=self.device)

            with torch.no_grad():
                output = self.model(body_pose=poses[:, 3:72], global_orient=poses[:, :3], transl=trans)

            joints = output.joints[:, :24, :].cpu().numpy()
            avg_root = np.mean(joints[:, 0, :2], axis=0)
            joints[:, :, 0] -= avg_root[0]
            joints[:, :, 1] -= avg_root[1]
            self.result_queue.put(joints)
        except Exception as e:
            print(f"Worker Error: {e}")
            self.result_queue.put(None)


class AmassLabeler(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AMASS Native Labeler")
        self.geometry("1400x900")
        self.minsize(1000, 650)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.smpl_model = smplx.SMPL(model_path=MODEL_PATH, gender='neutral', batch_size=1).to(self.device)
        self.file_list = []
        self.amass_root = ""

        # Ensure label column is read as string to match button keys
        if os.path.exists(SAVE_PATH):
            self.df_labels = pd.read_csv(SAVE_PATH, dtype={'label': str})
        else:
            self.df_labels = pd.DataFrame(columns=['relative_path', 'label'])

        self.ani = None
        self.worker = None
        self.path_placeholder = "Paste AMASS Root Path and press Enter..."
        self.path_var = tk.StringVar()
        self.init_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def init_ui(self):
        main_splitter = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=6)
        main_splitter.pack(fill=tk.BOTH, expand=True)

        left_panel = tk.Frame(main_splitter, padx=8, pady=8)
        right_panel = tk.Frame(main_splitter, padx=8, pady=8)
        main_splitter.add(left_panel, minsize=320, width=450)
        main_splitter.add(right_panel, minsize=600, width=950)

        path_row = tk.Frame(left_panel)
        path_row.pack(fill=tk.X)

        self.path_input = tk.Entry(path_row, textvariable=self.path_var)
        self.path_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.path_input.bind("<Return>", lambda _event: self.scan_files())
        self.path_input.bind("<FocusIn>", self.clear_path_placeholder)
        self.path_input.bind("<FocusOut>", self.restore_path_placeholder)
        self.restore_path_placeholder()

        btn_browse = tk.Button(path_row, text="Browse", command=self.browse_folder)
        btn_browse.pack(side=tk.RIGHT)

        self.lbl_stats = tk.Label(
            left_panel,
            text="Files: 0 | Labeled: 0",
            anchor=tk.W,
            font=("TkDefaultFont", 10, "bold"),
            fg="#2c3e50",
        )
        self.lbl_stats.pack(fill=tk.X, pady=(8, 4))

        list_frame = tk.Frame(left_panel)
        list_frame.pack(fill=tk.BOTH, expand=True)

        y_scroll = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        x_scroll = tk.Scrollbar(list_frame, orient=tk.HORIZONTAL)
        self.list_widget = tk.Listbox(
            list_frame,
            selectmode=tk.SINGLE,
            exportselection=False,
            yscrollcommand=y_scroll.set,
            xscrollcommand=x_scroll.set,
        )
        y_scroll.config(command=self.list_widget.yview)
        x_scroll.config(command=self.list_widget.xview)
        self.list_widget.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        self.list_widget.bind("<<ListboxSelect>>", lambda _event: self.load_selected_file())

        self.figure = Figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.figure, master=right_panel)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.lbl_status = tk.Label(
            right_panel,
            text="Status: Ready",
            anchor=tk.W,
            font=("TkDefaultFont", 11, "bold"),
            fg="#34495e",
        )
        self.lbl_status.pack(fill=tk.X, pady=(8, 4))

        self.btn_group = tk.Frame(right_panel)
        self.btn_group.pack(fill=tk.X)

        self.btn_v_useful = tk.Button(self.btn_group, text="🌟 Very Useful", command=lambda: self.label_current("2"))
        self.btn_useful = tk.Button(self.btn_group, text="✅ Useful", command=lambda: self.label_current("1"))
        self.btn_not_useful = tk.Button(self.btn_group, text="❌ Not Useful", command=lambda: self.label_current("0"))

        self.buttons = {
            "2": self.btn_v_useful,
            "1": self.btn_useful,
            "0": self.btn_not_useful,
        }

        for btn in self.buttons.values():
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4, pady=4, ipady=14)
            self.apply_button_style(btn, DEFAULT_BTN_STYLE)
            btn.config(state=tk.DISABLED)

    def apply_button_style(self, btn, style):
        btn.config(**style)

    def clear_path_placeholder(self, _event=None):
        if self.path_var.get() == self.path_placeholder:
            self.path_var.set("")
            self.path_input.config(fg="#000000")

    def restore_path_placeholder(self, _event=None):
        if not self.path_var.get():
            self.path_var.set(self.path_placeholder)
            self.path_input.config(fg="#7f8c8d")

    def get_path_text(self):
        text = self.path_var.get().strip()
        if text == self.path_placeholder:
            return ""
        return text

    def set_status(self, text, color="#34495e"):
        self.lbl_status.config(text=text, fg=color)

    def highlight_buttons(self, current_label):
        for label, btn in self.buttons.items():
            if label == current_label:
                self.apply_button_style(btn, PRESSED_BTN_STYLE)
            else:
                self.apply_button_style(btn, DEFAULT_BTN_STYLE)

    def browse_folder(self):
        path = filedialog.askdirectory(title="Select AMASS Root")
        if path:
            self.path_var.set(path)
            self.path_input.config(fg="#000000")
            self.scan_files()

    def update_stats(self):
        total = len(self.file_list)
        labeled = len(self.df_labels[self.df_labels['relative_path'].isin(
            [os.path.relpath(f, self.amass_root) for f in self.file_list]
        )])
        self.lbl_stats.config(text=f"Files Found: {total} | Labeled: {labeled}")

    def update_list_colors(self):
        for i in range(self.list_widget.size()):
            rel_path = self.list_widget.get(i)
            label = None
            if rel_path in self.df_labels['relative_path'].values:
                label = str(self.df_labels.loc[self.df_labels['relative_path'] == rel_path, 'label'].values[0])

            bg_color = LIST_COLOR_MAP.get(label, "#ffffff")
            self.list_widget.itemconfig(i, bg=bg_color)

    def scan_files(self):
        # normalize path to remove trailing slashes which mess up relpath
        path = os.path.normpath(self.get_path_text())
        if not os.path.isdir(path):
            self.set_status("Status: Error - Invalid Directory", "#c0392b")
            return

        self.amass_root = path
        self.list_widget.delete(0, tk.END)

        # We sort to ensure consistency
        raw_files = []
        for r, d, fs in os.walk(self.amass_root):
            for f in fs:
                if f.endswith('.npz'):
                    raw_files.append(os.path.join(r, f))

        self.file_list = sorted(raw_files)

        for f in self.file_list:
            # Consistent relative path calculation
            rel = os.path.relpath(f, self.amass_root)
            self.list_widget.insert(tk.END, rel)

        self.update_stats()
        self.update_list_colors()

    def get_current_row(self):
        selection = self.list_widget.curselection()
        if not selection:
            return -1
        return selection[0]

    def label_current(self, label_code):
        row = self.get_current_row()
        if row < 0:
            return

        # 1. Get the current relative path
        rel_path = self.list_widget.get(row)

        # 2. Update existing OR add new
        if rel_path in self.df_labels['relative_path'].values:
            idx = self.df_labels.index[self.df_labels['relative_path'] == rel_path].tolist()[0]
            self.df_labels.at[idx, 'label'] = label_code
        else:
            new_row = pd.DataFrame({'relative_path': [rel_path], 'label': [label_code]})
            self.df_labels = pd.concat([self.df_labels, new_row], ignore_index=True)

        # 3. Clean duplicates just in case (Sanity Check)
        self.df_labels = self.df_labels.drop_duplicates(subset=['relative_path'], keep='last')

        # 4. Save
        self.df_labels.to_csv(SAVE_PATH, index=False)

        self.update_stats()
        self.update_list_colors()
        self.highlight_buttons(label_code)

        if row < self.list_widget.size() - 1:
            next_row = row + 1
            self.list_widget.selection_clear(0, tk.END)
            self.list_widget.selection_set(next_row)
            self.list_widget.activate(next_row)
            self.list_widget.see(next_row)
            self.load_selected_file()

    def set_buttons_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        for btn in self.buttons.values():
            btn.config(state=state)

    def load_selected_file(self):
        selected = self.get_current_row()
        if selected < 0:
            return

        if self.ani:
            self.ani.event_source.stop()

        self.set_buttons_enabled(False)
        self.highlight_buttons(None)

        file_path = self.file_list[selected]
        rel_path = os.path.relpath(file_path, self.amass_root)

        self.ax.clear()
        self.ax.set_title(f"Processing: {rel_path}", fontsize=10)
        self.canvas.draw()

        # Update status to show background work
        self.set_status("Status: Calculating Animation...", "#d35400")

        result_queue = queue.Queue()
        self.worker = AnimationWorker(file_path, self.smpl_model, self.device, result_queue)
        self.worker.start()
        self.poll_worker_queue(result_queue, rel_path)

    def poll_worker_queue(self, result_queue, rel_path):
        try:
            joints = result_queue.get_nowait()
        except queue.Empty:
            self.after(50, lambda: self.poll_worker_queue(result_queue, rel_path))
            return

        self.start_animation(joints, rel_path)

    def start_animation(self, joints, rel_path):
        if joints is None:
            self.ax.set_title(f"FAILED TO LOAD: {rel_path}", color='red')
            self.set_status("Status: ❌ Error loading file", "#c0392b")
            self.canvas.draw()
            return

        self.ax.clear()
        self.ax.set_title(rel_path, fontsize=10)
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_zlim(0, 2)
        scat = self.ax.scatter([], [], [], c='blue', s=10)

        def update(frame):
            f = joints[frame]
            scat._offsets3d = (f[:, 0], f[:, 1], f[:, 2])
            return scat,

        self.ani = animation.FuncAnimation(self.figure, update, frames=len(joints), interval=50, blit=False)
        self.canvas.draw()

        # Display the numeric label as a text status
        current_val = "None"
        if rel_path in self.df_labels['relative_path'].values:
            current_val = str(self.df_labels.loc[self.df_labels['relative_path'] == rel_path, 'label'].values[0])
            self.highlight_buttons(current_val)

        self.set_status(f"Status: {LABEL_MAP.get(current_val, 'Unlabeled')}")
        self.set_buttons_enabled(True)

    def on_close(self):
        if self.ani:
            self.ani.event_source.stop()
        self.destroy()


if __name__ == "__main__":
    app = AmassLabeler()
    app.mainloop()
