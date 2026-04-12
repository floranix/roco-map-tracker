# roco-map-tracker

这是一个交接给下一位 Codex 的中文 README。

项目目标是把《洛克王国世界》的整张互动地图接入本地程序，并基于玩家截图在世界地图上自动定位，然后把定位结果和后续规划路线需要的点位一起可视化出来。

**目前已经实现的效果**

- 已接入 `17173` 世界地图数据，并按网页最大缩放层 `z=13` 下载瓦片后拼成完整底图。
- 默认配置已经切到高精度整图 [data/rocom_17173/rocom-shijie-z13.png](/Users/lanier/PyProjects/roco-map-tracker/data/rocom_17173/rocom-shijie-z13.png)，而不是旧的低清整图。
- 已接入 `pois.json`、`categories.json`、`regions.json` 等地图数据。
- GUI 已支持按分类和关键词按需显示 POI，并可控制是否显示标签。
- GUI 可在 macOS 上直接启动，CLI 也可用。
- 大图定位流程已加入多尺度全局匹配和分块候选匹配，能在 `8192x8192` 世界图上做首帧搜索。
- 定位结果展示已改成“局部最大层级视图 + 整图总览小窗”，更适合看落点。
- 已加入一版小地图截图预处理，会遮掉圆外区域、中心箭头和一部分常见干扰图标。

**这个项目当前能做到什么**

- 对“从整张世界图裁出来的局部地图视野”已经可以比较稳定地定位。
- 对部分真实小地图截图，能给出候选位置并按最大层级展示结果。
- 地图标注已经不是固定写死展示，而是支持按需筛选和按需叠加。

**这个项目还没做到什么**

- 真实用户提供的小地图截图仍然不能稳定自动定位，每张图成功率不够。
- 小地图里如果有大量资源点图标、数字标记、角落 UI 遮挡，现有特征匹配容易跑偏。
- 中文文字绘制还没有真正处理好，目前 OpenCV 默认字体会把中文渲染成 `????`。
- “小地图截图输入”和“真实场景截图输入”现在仍基本共用一条主流程，后续最好拆成明确的两种模式。

**建议下一个 Codex 继续做的事**

1. 把“小地图截图”做成独立输入模式。
2. 继续加强遮罩逻辑，尤其是右上角资源堆叠、数字标记、圆心箭头和浮层图标。
3. 不要只依赖 ORB，建议引入边缘/海岸线/颜色区域等更适合小地图的匹配方式。
4. 把玩家箭头朝向也纳入定位约束，用来过滤候选区域。
5. 把结果图的中文渲染切到 `Pillow` 或其他支持中文字体的方案。

**关键文件**

- [configs/rocom_17173.yaml](/Users/lanier/PyProjects/roco-map-tracker/configs/rocom_17173.yaml)：默认运行配置，当前主配置。
- [scripts/fetch_rocom_17173.py](/Users/lanier/PyProjects/roco-map-tracker/scripts/fetch_rocom_17173.py)：抓取 `17173` 地图页面数据并重建高缩放底图。
- [src/global_localizer.py](/Users/lanier/PyProjects/roco-map-tracker/src/global_localizer.py)：全局定位核心逻辑。
- [src/preprocess.py](/Users/lanier/PyProjects/roco-map-tracker/src/preprocess.py)：小地图预处理与遮罩逻辑。
- [src/poi_overlay.py](/Users/lanier/PyProjects/roco-map-tracker/src/poi_overlay.py)：POI 过滤、投影和绘制。
- [src/pipeline.py](/Users/lanier/PyProjects/roco-map-tracker/src/pipeline.py)：主处理流程。
- [src/gui.py](/Users/lanier/PyProjects/roco-map-tracker/src/gui.py)：mac 友好的 Tk GUI。
- [src/utils.py](/Users/lanier/PyProjects/roco-map-tracker/src/utils.py)：配置结构、可视化绘制、展示裁剪。
- [README-2d-game-map-localizer.md](/Users/lanier/PyProjects/roco-map-tracker/README-2d-game-map-localizer.md)：项目使用说明。

**已验证情况**

- `py_compile` 已通过。
- `python -m unittest discover -s tests -v` 已通过。
- GUI 已成功初始化。
- 对大图裁剪视野的定位是稳定的。
- 对真实小地图截图，当前只能算“已有实验性能力”，还不算稳定完成。

**运行方式**

```bash
python main.py --gui
```

也可以直接走命令行：

```bash
python main.py --config configs/rocom_17173.yaml --frame /path/to/image.png --save-visualizations
```

重建 `17173` 地图数据：

```bash
python scripts/fetch_rocom_17173.py
```

**提交说明**

- 这次提交不应包含分析过程中的 `tmp_*` 图片。
- 这次提交不应包含 `data/rocom_17173/tiles_z13/` 这种瓦片缓存目录。
- 这次提交也不包含抓取网页时留下的 `bootstrap.js`、`bundle.js`、`page.html`。
- 当前程序真正依赖的是拼好的 [data/rocom_17173/rocom-shijie-z13.png](/Users/lanier/PyProjects/roco-map-tracker/data/rocom_17173/rocom-shijie-z13.png) 以及相关元数据和 POI 数据。
