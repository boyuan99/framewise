# Framewise

A multi-panel synchronized viewer for calcium imaging, behavior video, and TDT electrophysiology. All panels subscribe to a single master clock for frame-locked playback and scrubbing, with an embedded Jupyter console / Lab that can access the live objects.

## Tech stack

- **PyQt6 + pyqtgraph** — GUI and plotting
- **h5py + dask** — lazy HDF5; **tifffile** — TIFF stacks; **imageio[ffmpeg]** — compressed video
- **tdt** — TDT blocks; **scipy** — .mat / signal processing
- **qtconsole + ipykernel + jupyterlab + PyQt6-WebEngine** — embedded console and Jupyter Lab

## Installation

Requires Python ≥ 3.10.

```bash
conda create -n framewise python=3.11
conda activate framewise
pip install -e .
```

Optional analysis extras (scikit-learn / statsmodels / opencv / meegkit):

```bash
pip install -e ".[analysis]"
```

## Usage

```bash
framewise [files or folders ...]   # entry-point script
python -m framewise_py [...]        # or run as a module
```

Arguments are optional — once the window is open, load data via **File → Open…** (video/stacks) or **File → Open Folder…** (TDT block / segmentation result folder).

Supported sources: HDF5 `.h5/.hdf5`, TIFF `.tif/.tiff`, video `.mp4/.avi/.mov/.mkv`, TDT block folders, and CNMF/CaImAn segmentation result folders (`SEG/` + optional `rmbg/`).
