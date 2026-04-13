# roco-map-tracker

这是一个交接给下一位 Codex 的中文 README。

项目目标是把《洛克王国世界》的整张互动地图接入本地程序，并基于玩家截图在世界地图上自动定位，然后把定位结果和后续规划路线需要的点位一起可视化出来。

## 当前阶段

当前处于“原型已跑通，但真实小地图定位仍需继续打磨”的阶段。

- 底图、POI、GUI、CLI、实时屏幕区域采集已经串起来了。
- 当前默认数据源还是 `17173`，但渲染层已经兼容 biligame/wiki 风格的 `pixel_space` 点位和 icon 缓存。
- 对“从整张世界图裁出来的局部视野”定位比较稳。
- 对真实玩家小地图截图，已经有可用候选结果，但还没有达到稳定、可依赖的程度。

## 已完成

- 已接入 `17173` 世界地图数据，并按网页最大缩放层 `z=13` 下载瓦片后拼成完整底图。
- 默认配置已经切到高精度整图 `data/rocom_17173/rocom-shijie-z13.png`。
- 已接入 `pois.json`、`categories.json`、`regions.json` 等地图数据。
- GUI 已支持按分类和关键词筛选 POI，并可控制是否显示标签。
- POI 叠加已优先使用站点原始 icon，不再只画统一小圆点。
- POI 渲染已兼容两类数据：
  - `17173` 的 `category_id + longitude/latitude`
  - biligame/wiki 风格的 `markType + point{lng,lat}`，以及 `pixel_space` 投影
- 已支持像 `zkjisj/luoke_location` 那样选择一个屏幕矩形区域，按较短间隔持续采集并实时定位。
- 全局定位流程已经包含：
  - 多尺度模板匹配
  - 分块候选匹配
  - 特征匹配 + 几何验证
- 小地图输入遵循“输入图本身不旋转”的约束，保留了小角度容错，但不会主动把输入图当作会跟随移动旋转。
- 已移除之前默认关掉的模板颜色校验冗余逻辑。
- 结果展示已改成“局部聚焦视图 + 整图总览小窗”。

## 还没完成

- 真实小地图截图的稳定性还不够，尤其是纯色区域、资源点密集区域、数字标记和 UI 遮挡明显时仍会跑偏。
- 当前“实时矩形采集”只是把框选区域持续送入现有主流程，还没有针对真实屏幕流单独做一套更稳的专用策略。
- 默认运行数据源仍是 `17173`，还没有真正把 biligame/wiki 的整套底图、点位、icon 和 metadata 落到本仓库里替换默认配置。
- 中文文字绘制仍用 OpenCV 默认字体，中文标签会有 `????` 问题。
- 小地图输入模式和普通截图/视频模式仍基本共用一条主流程，后续建议拆开。

## 下一个 Codex 优先做什么

1. 真正接入 biligame/wiki 数据为默认数据源。
   - 参考项目：`https://github.com/zkjisj/luoke_location`
   - 已确认参考仓库里有：
     - `out/rocom_base_z8.png`
     - `out/rocom_points.json`
     - `out/rocom_caiji_points.json`
     - `out/icon_cache_wiki/`
     - `out/icon_cache_caiji/`
   - 还需要把这些数据正式整理到本仓库 `data/...` 下，并补一份 `metadata.json`，把 `pixel_space` 的缩放和偏移写明确。
2. 把“真实小地图输入”做成独立模式。
   - 单独配置预处理、遮罩和阈值。
   - 不要和“从大图裁出的局部视野”继续共用同一组策略。
3. 继续强化小地图遮罩。
   - 重点是右上角资源堆叠、数字标记、圆心箭头、边缘 UI 和圆外屏幕内容。
4. 不要只依赖 ORB。
   - 可以尝试边缘图、海岸线、区域块相似度，或参考仓库的 SIFT 思路做一套并行候选器。
5. 把中文渲染切到 `Pillow` 或别的中文字体方案。

## 关键文件

- `configs/rocom_17173.yaml`：当前主配置。
- `scripts/fetch_rocom_17173.py`：抓取 `17173` 地图页面数据并重建高缩放底图。
- `src/global_localizer.py`：全局定位核心逻辑。
- `src/preprocess.py`：小地图预处理与遮罩逻辑。
- `src/poi_overlay.py`：POI 过滤、投影、icon 加载和绘制。
- `src/screen_pick.py`：屏幕区域框选和持续采集。
- `src/pipeline.py`：主处理流程。
- `src/gui.py`：mac 友好的 Tk GUI。
- `src/main.py`：CLI 入口。
- `src/utils.py`：配置结构、可视化绘制、展示裁剪。
- `tests/test_poi_overlay.py`：POI 投影和 icon 渲染测试。
- `README-2d-game-map-localizer.md`：项目使用说明。

## 已验证

- `python3 -m py_compile src/*.py tests/*.py main.py` 已通过。
- `python3 -m unittest discover -s tests -v` 已通过。
- `.venv` 中已安装 `mss`，可直接通过 `launch_gui.command` 或 `./.venv/bin/python main.py --gui` 使用屏幕区域采集。
- 对大图裁剪视野的定位是稳定的。
- 对真实小地图截图，目前仍只能算“实验性可用”，不能说已经稳定完成。

**运行方式**

```bash
python main.py --gui
```

也可以直接走命令行：

```bash
python main.py --config configs/rocom_17173.yaml --frame /path/to/image.png --save-visualizations
```

如果想持续抓取屏幕上的一个矩形区域：

```bash
python main.py --config configs/rocom_17173.yaml --screen-region 1800,260,220,220 --capture-interval-ms 150 --visualize
```

GUI 中也可以把输入源切到“屏幕区域”，再点击“选择”直接框选。

重建 `17173` 地图数据：

```bash
python scripts/fetch_rocom_17173.py
```

**提交说明**

- 这次提交不应包含分析过程中的 `tmp_*` 图片。
- 这次提交不应包含 `data/rocom_17173/tiles_z13/` 这种瓦片缓存目录。
- 这次提交也不包含抓取网页时留下的 `bootstrap.js`、`bundle.js`、`page.html`。
- 当前程序真正依赖的是拼好的 `data/rocom_17173/rocom-shijie-z13.png` 以及相关元数据和 POI 数据。
