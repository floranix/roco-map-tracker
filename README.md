# roco-map-tracker

基于纯 `biliwiki` 世界地图的《洛克王国世界》小地图定位原型。

项目目标是：

- 自动使用项目内置的完整大地图，不要求用户手动输入底图路径
- 在 macOS 上提供中文 GUI
- 支持单张截图定位和屏幕区域持续识别
- 在大地图上叠加 biliwiki 采集点与按需生成的资源路线

## 当前方案

当前仓库已经彻底切到 `biliwiki` 体系：

- 识别底图：`data/rocom_biliwiki/rocom_base_z8.png`
- 采集点来源：`wiki.biligame.com/rocom/大地图`
- 点位坐标：使用 wiki `version=4` 原生地图坐标
- 点位投影：直接映射到 biliwiki 大地图像素空间

不再依赖旧版第三方底图、坐标对齐矩阵和相关脚本。

## 功能

- 中文 GUI，适合 macOS 直接使用
- 输入方式：
  - 单张小地图截图
  - 屏幕区域持续采集
- 小地图预处理：
  - 圆区提取
  - 中心箭头与边缘 UI 遮罩
  - 局部干扰抑制
- 定位流程：
  - 全局模板候选
  - 全局分块匹配
  - 局部跟踪与失联回退
- 地图显示：
  - 完整大地图缩放与滚动
  - 当前位置标记
  - 采集点图标叠加
  - 按所选素材生成路线

## 环境

推荐在 macOS 下使用项目虚拟环境：

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

注意：

- GUI 依赖 `Tkinter`
- 持续屏幕采集依赖 `mss`
- macOS 需要给当前 Python 或 Codex 授予“屏幕录制”权限

## 启动

GUI：

```bash
./launch_gui.command
```

或：

```bash
./.venv/bin/python main.py --gui
```

命令行单图定位：

```bash
./.venv/bin/python main.py --frame /absolute/path/to/minimap.png --save-visualizations
```

命令行持续识别：

```bash
./.venv/bin/python main.py --screen-region 1800,260,220,220 --capture-interval-ms 250 --visualize
```

## 内置资源

程序默认使用 `data/rocom_biliwiki/` 下的内置资源：

- `rocom_base_z8.png`：完整大地图
- `rocom_caiji_points.json`：采集点缓存
- `rocom_caiji_categories.json`：素材分类
- `icon_cache_wiki/`：素材图标缓存

启动时会优先使用本地缓存，不会因为点位文件“过期”就强制联网刷新。只有点位缺失，或在 GUI 里主动点击“刷新 biliwiki”时，才会重新拉取最新数据。

## 目录说明

- `configs/rocom_tracker.yaml`：默认配置
- `src/gui.py`：中文 GUI
- `src/main.py`：CLI 入口
- `src/preprocess.py`：小地图预处理
- `src/global_localizer.py`：全局定位核心
- `src/pipeline.py`：定位主流程
- `src/poi_overlay.py`：采集点与图标叠加
- `src/resource_routes.py`：路线生成与缓存
- `src/resource_sources.py`：biliwiki 资源准备与默认配置注入

## 验证

当前建议的基础验证：

```bash
python3 -m py_compile src/*.py tests/*.py main.py
./.venv/bin/python -m unittest discover -s tests -v
```

单图验证示例：

```bash
./.venv/bin/python main.py --frame /Users/lanier/Downloads/QQ20260413-235108.png --save-visualizations
```

## 内存说明

完整大地图较大，单次建立定位索引本身就会占用较多内存。正常 GUI/CLI 使用只应保留一套定位管线。

不要并行启动多份“完整大地图建索引”的调试任务，尤其不要同时跑多套 `orb/sift` 对比，否则内存会被整张大图和特征索引重复放大。
