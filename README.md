# AMASS 3D Data Labeling Utility

A Tkinter and Matplotlib desktop application for sequentially viewing, rendering, and categorizing human motion sequences from the **AMASS (Archive of Motion Capture As a Surface Shape)** dataset using the **SMPL** body model framework.

This tool streamlines dataset curation by allowing users to rapidly flag motion sequences (`Very Useful`, `Useful`, `Not Useful`) to filter data prior to downstream dataset generation.

---

## Interface Demo

<video src="https://github.com/user-attachments/assets/27398f1e-1bfa-4504-a423-d0a23b14b1a3" controls="controls" muted="muted" style="max-height: 500px;">
</video>

---

## Environment Setup

### 1. Navigate to the Labeler Directory
Open your terminal and change into the application folder:
```bash
  cd path/to/REPO_ROOT/amass-labeler
```
### 2. Create the Conda Environment
Build the isolated environment using the provided YAML file:
```bash
  conda env create -f environment_amass_labeler.yml
```

---

## Execution Instructions
### 1. Activate the Environment
```bash
  conda activate amass_labeler
```
### 2. Launch the Application
```bash
  python amass_labeler.py
```
### 3. Load AMASS Sequences
- Paste or type the absolute path to your unpacked AMASS root directory in the top-left text field and press Enter (or use the Browse button).
- The tool recursively scans all .npz files, displays total file/labeled counts, and populates the sequence navigation list.
### 4. Label Sequences
- Select a sequence from the list. The app processes motion frames in a background thread and streams a 3D animated skeleton view.
- Click one of the three classification buttons to assign a label:
  - 2 — Very Useful (e.g., Falls, target movements)
  - 1 — Useful (e.g., Walking, standard locomotion)
  - 0 — Not Useful (e.g., Kickboxing, irrelevant movements)
- Clicking a label automatically saves the selection and advances to the next sequence.
- Rows in the sequence list update dynamically with muted colors reflecting their current label status.
## Output & Dataset Pipeline Integration
- Saved Output: Labels are saved automatically to amass_labels.csv inside the application folder.
- Format: relative_path,label ACCAD/Female1General_c3d/A10_-_lie_to_crouch_stageii.npz,2 ACCAD/Female1Running_c3d/C10_-__run_backwards_stop_run_forward_stageii.npz,0
- Pipeline Integration: Once labeling is complete, copy amass_labels.csv to the destination defined in projector_pipeline.py under AMASS_LABELS_PATH to apply filters during dataset generation.

---

## References & Acknowledgments
* **AMASS Dataset:** [https://amass.is.tue.mpg.de/](https://amass.is.tue.mpg.de/)
* **SMPL Body Model:** [https://smpl.is.tue.mpg.de/](https://smpl.is.tue.mpg.de/)
