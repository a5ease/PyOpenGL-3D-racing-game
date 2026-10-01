# 🏎️ 动力滑行

**3D Mountain Racing Game** built with **Python + SDL2 + PyOpenGL**.

Realistic Haruna mountain pass racing simulation with PBR (Cook-Torrance) rendering, advanced car physics (bicycle model + Pacejka tires), and drift-style cornering.

基于榛名山真实地形的 3D 山路赛车模拟，使用 PBR Cook-Torrance 光照渲染，还原山路赛道场景与"沟渠跑法"物理。

---
## 📝 作者的话

> 本作品为我只使用 AI 的情况下，并在两周内完成。

> 此作品仅是为了实验 Python 能不能做出画质与质感为 PS4 左右的作品。虽然并不懂编程和代码，只知道怎么打游戏，但还是为了想试一试。

> 我在游玩《头文字D激斗》之后久久不能忘怀，并且因为学校周围没有街机厅，于是创造了此作品。

> 使用的 AI 以及平台分别为：DeepSeek V4.1 Flash、GLM 5.3/5.3 Flash、Hy 4、TRAE CN、WorkBuddy

**Keywords:** `racing-game` `mountain-racing` `3d-game` `python` `sdl2` `pyopengl` `opengl` `pbr-rendering` `car-physics` `drift` `simulation` `catmull-rom` `haruna`

---

## 🎮 功能特性

- **榛名山赛道** — 基于真实 DEM 数据构建，Catmull-Rom 样条驱动，全长约 7.5 km
- **PBR 渲染管线** — Cook-Torrance BRDF + 程序化纹理 + 真实 PBR 贴图支持
- **车辆物理** — 自行车模型 + Pacejka 轮胎 + 载荷转移 + 摩擦圆，模拟街机手感
- **沟渠跑法** — 路肩 U 型混凝土侧沟物理检测，重现"沟渠跑法"过弯
- **赛道边界防入侵** — signed_distance 实时碰撞检测，车体不可越界
- **完整 HUD** — 速度表、转速表、档位显示、小地图、弯道预警
- **场景丰富** — 杉树林（实例化 + LOD）、建筑群、路灯、护栏、弯道标志、防护网等
- **摄像机调节** — 滚轮调节距离/高度/预瞄，支持热键切换，自动保存配置

## 📸 截图

>![alt text](image.png)![alt text](image-1.png)

## 🧰 技术栈

| 组件 | 技术 |
|------|------|
| 窗口 & 输入 | SDL2 |
| 图形渲染 | OpenGL (PyOpenGL) |
| 物理引擎 | 自研（自行车模型 + Pacejka） |
| 赛道生成 | Catmull-Rom 样条 + 地形网格 signed_distance 压平 |
| 光照模型 | Cook-Torrance PBR |
| 碰撞检测 | signed_distance 场 |
| 场景数据 | NumPy 缓存系统 |

## 🚀 快速开始

```bash
# 安装依赖
pip install sdl2 PyOpenGL numpy Pillow PyOpenGL_accelerate

# 运行游戏
python main.py
```

> 首次运行会自动构建赛道、地形、植被等场景数据，耗时约 1–2 分钟。

## 📁 项目结构

```
动力滑行/
├── main.py              # 主游戏入口（渲染循环、场景构建）
├── hud.py               # HUD 界面（速度表、小地图、主菜单）
├── mountain.py          # 赛道与地形生成器（Catmull-Rom 样条驱动）
├── physics_car.py       # 车辆物理引擎（自行车模型 + Pacejka）
├── corner_analyzer.py   # 弯道检测与分类
├── track_bounds.py      # 赛道边界 signed_distance 碰撞检测
├── dem_loader.py        # DEM 地形数据加载
├── pbr_loader.py        # PBR 纹理加载器
├── batch_renderer.py    # 批量离屏渲染工具
├── README.md
├── run_game.bat         # Windows 一键启动脚本
├── .gitignore
├── assets/textures/     # PBR 材质贴图（albedo/normal/roughness...）
├── .cache/              # 场景缓存（首次运行自动生成）
└── _scratch/            # 开发测试脚本（不影响运行时）
```

## 🎮 操作说明

| 按键 | 功能 |
|------|------|
| W / ↑ | 加速 |
| S / ↓ | 刹车 / 倒车 |
| A / D 或 ← / → | 转向 |
| C | 切换摄像机调节模式（距离/高度/预瞄） |
| 滚轮 | 调节当前摄像机参数 |
| ESC | 暂停 / 返回主菜单 |
| F | 切换全屏 |

## ⚖️ 免责声明

本作品为同人粉丝项目，仅供学习交流，不用于商业用途。作品中出现的品牌、车辆型号、地点名称均为其各自所有者的财产。如有权利方认为构成侵权，请联系作者删除相关内容。

## 📄 许可

本项目仅供学习与交流，未经许可不得用于商业用途。
