# Framewise (Python / napari)

钙成像多视频同步查看器，基于 napari 实现。

## 安装

```bash
conda create -n framewise python=3.11
conda activate framewise
pip install -e .
```

## 使用

```bash
python -m framewise_py video1.h5 video2.h5 video3.mp4
```

每个文件会在独立的 napari 窗口中打开。第一个窗口附带「同步组」控制面板，可手动选择哪些视频组成同步组（同步组内拖拽时间滑块时联动）。

## 支持格式

- HDF5 (`.h5`, `.hdf5`) — 假设数据集形状为 `(T, H, W)` 或 `(T, C, H, W)`
- TIFF 堆栈 (`.tif`, `.tiff`)
- 视频文件 (`.mp4`, `.avi`) — 建议使用全关键帧编码以获得流畅 seek
