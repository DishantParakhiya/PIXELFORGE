# PixelForge

A computer vision darkroom built with Python, OpenCV and Streamlit. Upload a photo, pick an algorithm, tune it, and compare the result with the original. It runs fully offline: no database, no paid API, no GPU.

## Run it

You need Python 3.10 or newer.

**Windows:** double-click `run.bat`

**macOS / Linux:** `bash run.sh`

Or manually:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. Click **Load demo** in the sidebar if you have no image handy.

## What is inside

| Mode | What it does |
|---|---|
| **Lab** | One algorithm at a time on the original image. Live parameters, a drag-to-reveal before/after print, side-by-side view, a delta map showing where pixels changed, histograms, and PNG export (preview size or full resolution). |
| **Forge Chain** | Stack algorithms into a pipeline. Every stage appears as a frame on a film strip. Reorder, remove, inspect any stage, load a preset, and copy the recipe for your report. |
| **Arena** | Run competing algorithms on the same image: filters (with PSNR scores on injected noise), edge detectors, thresholds, and morphology. |
| **From Scratch** | Convolution, histogram equalization and Otsu thresholding written in plain NumPy, run next to OpenCV with the difference measured. |
| **Codex** | Every operation with its parameters and the maths behind it. |

27 operations across 7 categories: Basic, Enhance, Filters, Edges, Segment, Morphology, Effects.

## How it maps to the project plan

All three levels are implemented: Level 1 (upload, resize, grayscale, brightness/contrast, Gaussian and median filters, Canny, thresholding, download), Level 2 (bilateral, Sobel, Laplacian, morphology, histograms, stage comparison) and Level 3 (pencil sketch, cartoon, K-means, processing chain, comparison dashboard).

Changes from the written plan, and why:

- **Independent Mode and Pipeline Mode both exist.** Lab always processes the original. Forge Chain is the explicit pipeline mode.
- **Morphology binarizes first** (Otsu, fixed, or none), so the threshold-then-morphology demo works inside one module.
- **A draggable reveal slider was added** even though the plan called it optional.
- **From Scratch mode was added** because viva examiners ask what happens underneath `cv2.something()`.
- **The syllabus unit mappings from the plan are not in the app.** Check them against your actual syllabus before putting them in the report.

## Architecture

```
app.py              Streamlit entry point (the five modes)
core/               Pure functions: image in, image out. No Streamlit imports.
  registry.py       One table of all operations, parameters and theory. Drives Lab, Chain, Codex and tests.
  basic.py enhancement.py filters.py edges.py segmentation.py morphology.py effects.py
  histograms.py     Histogram numbers (plotting lives in ui/)
  scratch.py        NumPy-only implementations for the viva
  utils.py          Decoding, validation, colour handling, preview, export, demo image
ui/                 Theme (cyanotype darkroom) and reusable components
tests/              pytest: 64 core tests + 21 end-to-end app tests
docs/VIVA_NOTES.md  Revision notes and likely examiner questions
```

Key design rules:

- Images are `uint8` NumPy arrays in OpenCV's BGR order. They are converted to RGB only when displayed.
- The original upload is never modified. Live processing runs on a copy downscaled to 860 px wide; export can run on the full-resolution original.
- Even kernel sizes are prevented by the slider step and forced odd inside the functions.
- Uploads are validated: JPG or PNG only, corrupted files and images over 60 MP are rejected with a clear message, EXIF rotation is applied, and transparency is flattened onto white.

## Tests

```bash
pytest                      # everything (about 10 seconds)
pytest tests/test_core.py   # algorithms only
```

## Troubleshooting

- **Port already in use:** `streamlit run app.py --server.port 8502`
- **Large images feel slow:** the live preview is already downscaled; bilateral filter, cartoon and K-means are the heaviest operations. Full-resolution export can take several seconds.
- **`pip install` fails behind a proxy:** install the packages in `requirements.txt` from a connection that can reach PyPI. After installation the app needs no internet.
