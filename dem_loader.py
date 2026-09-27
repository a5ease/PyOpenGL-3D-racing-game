"""
dem_loader.py - 榛名山真实 DEM 地形装载器
=========================================
从 fetch_dem.py 采集的 haruna_dem.npy 中加载真实高程数据。
策略：DEM 网格直接映射到游戏世界网格，赛道控制点位置不变，
      只把 DEM 的高度起伏按比例缩放到游戏高度范围。

用法：
    from dem_loader import dem_get_height, init_dem
    init_dem()
    h = dem_get_height(x, z)   # 替换 mountain.get_terrain_height(x, z)
"""

import math
import numpy as np


# ============================================================
# 一、DEM 元数据
# ============================================================
LON_MIN, LON_MAX = 138.82, 138.93
LAT_MIN, LAT_MAX = 36.42, 36.52
GRID_SIZE = None  # 由 init_dem() 从加载的 DEM 数据自动设置

# 参考点：榛名山扫部岳（最高峰）
REF_LAT = 36.476
REF_LON = 138.878

# 游戏世界边界（与 mountain.py 的地形网格匹配）
WORLD_SCALE = 80  # 1:1 水平比例（GAME_WORLD_SIZE = 11200）
GAME_WORLD_SIZE = int(140 * WORLD_SCALE)  # 11200, -5600 ~ 5600
GAME_PEAK_HEIGHT = 18.0 * WORLD_SCALE / 2.5    # 原版山顶高度（无DEM回退用）
GAME_BASE_HEIGHT = -3.0 * WORLD_SCALE / 2.5    # 原版山脚高度（无DEM回退用）

# 全局状态
_dem_raw = None         # 原始高程 (150, 150)
_dem_smooth = None      # 平滑后的高程
_dem_ready = False

# 高度映射参数（线性映射：DEM 高度 → 游戏高度）
_dem_min = 0.0          # DEM 上的道路最低点
_dem_max = 0.0          # DEM 上的道路最高点
_height_scale = 1.0     # 缩放因子
_height_offset = 0.0    # 偏移量


# ============================================================
# 二、DEM 加载与预处理
# ============================================================

def init_dem(npy_path=None, smooth=True):
    """
    加载 DEM 数据，计算高度映射参数。
    """
    global _dem_raw, _dem_smooth, _dem_ready
    global _dem_min, _dem_max, _height_scale, _height_offset, GRID_SIZE

    if npy_path is None:
        import os
        npy_path = os.path.join(os.path.dirname(__file__), "haruna_dem.npy")

    # 加载 NPY
    data = np.load(npy_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == np.object_:
        data = data.item()
    if isinstance(data, dict):
        _dem_raw = data["elevation"].astype(np.float32)
    else:
        _dem_raw = np.array(data, dtype=np.float32)

    # 自动适配 DEM 网格大小（支持 150~1000 任意分辨率）
    GRID_SIZE = _dem_raw.shape[0]
    print(f"[DEM] 原始数据: {_dem_raw.shape}, "
          f"高程 {_dem_raw.min():.0f}m ~ {_dem_raw.max():.0f}m")

    # 平滑（轻量平滑，保留更多地形细节）
    if smooth:
        from scipy.ndimage import gaussian_filter
        # 仅一次轻量高斯平滑，保留真实地形细节
        _dem_smooth = gaussian_filter(_dem_raw, sigma=1.2)
        print(f"[DEM] 平滑完成（高斯σ1.2，保留细节）")
    else:
        _dem_smooth = _dem_raw.copy()

    # ---- 关键：计算高度映射 ----
    # 当前游戏赛道的控制点 XZ 范围，映射到 DEM 网格上
    # 取赛道覆盖区域的 DEM 高度范围
    _calibrate_height_map()

    _dem_ready = True
    print(f"[DEM] 就绪！")
    print(f"[DEM] 映射: DEM [{_dem_min:.0f}m ~ {_dem_max:.0f}m] → "
          f"游戏高度 [{_dem_min * _height_scale + _height_offset:.1f} ~ "
          f"{_dem_max * _height_scale + _height_offset:.1f}]")


def _calibrate_height_map():
    """
    计算 DEM 高度到游戏高度的映射参数。
    关键改进：大幅压缩垂直幅度，让地形像"真正的山路"而非"锯齿山"。
    策略：DEM 只提供宏观趋势（哪里高哪里低），不保留剧烈起伏。
    """
    global _dem_min, _dem_max, _height_scale, _height_offset

    # 取整个 DEM 的统计范围（用于归一化）
    _dem_min = float(_dem_smooth.min())
    _dem_max = float(_dem_smooth.max())

    dem_range = _dem_max - _dem_min

    # ---- 核心参数：垂直压缩 ----
    # 1:1 真实高程，零压缩。山区地形不做任何垂直缩放。
    VERTICAL_COMPRESSION = 1  # 1:1 真实高程，无压缩
    TARGET_HEIGHT_RANGE = max(25, int(dem_range / VERTICAL_COMPRESSION))

    if dem_range > 1.0:
        _height_scale = TARGET_HEIGHT_RANGE / dem_range
        # 让中间高度对应游戏 2 点附近（略高于地面）
        _height_offset = -_dem_min * _height_scale - TARGET_HEIGHT_RANGE * 0.3
    else:
        _height_scale = 0.01
        _height_offset = 0.0

    print(f"[DEM] 校准: DEM [{_dem_min:.0f}m, {_dem_max:.0f}m] "
          f"→ 游戏 [{dem_to_game_height(_dem_min):.1f}, {dem_to_game_height(_dem_max):.1f}]")
    print(f"[DEM] 参数: scale={_height_scale:.5f}, offset={_height_offset:.2f}")
    print(f"[DEM] ⚠️ 垂直压缩: 真实{dem_range:.0f}m → 游戏{TARGET_HEIGHT_RANGE:.0f}单位 (约{dem_range/TARGET_HEIGHT_RANGE:.0f}:1)")


def dem_to_game_height(dem_h):
    """DEM 高程（米）→ 游戏高度"""
    return dem_h * _height_scale + _height_offset


def game_to_dem_height(game_h):
    """游戏高度 → DEM 高程（米）"""
    return (game_h - _height_offset) / _height_scale


# ============================================================
# 三、高度采样（核心接口）
# ============================================================

def _sample_dem(lon, lat):
    """
    对 DEM 进行双线性插值采样。
    返回 DEM 原始高程（米）。
    """
    if not _dem_ready:
        return 0.0

    # 边界裁剪
    lon = max(LON_MIN, min(LON_MAX, lon))
    lat = max(LAT_MIN, min(LAT_MAX, lat))

    # 网格坐标
    col_f = (lon - LON_MIN) / (LON_MAX - LON_MIN) * (GRID_SIZE - 1)
    row_f = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * (GRID_SIZE - 1)

    col0 = int(col_f)
    row0 = int(row_f)
    col1 = min(col0 + 1, GRID_SIZE - 1)
    row1 = min(row0 + 1, GRID_SIZE - 1)

    fx = col_f - col0
    fy = row_f - row0

    h00 = _dem_smooth[row0, col0]
    h10 = _dem_smooth[row0, col1]
    h01 = _dem_smooth[row1, col0]
    h11 = _dem_smooth[row1, col1]

    h0 = h00 * (1 - fx) + h10 * fx
    h1 = h01 * (1 - fx) + h11 * fx
    return float(h0 * (1 - fy) + h1 * fy)


def sample_geo_height(lon, lat):
    """经纬度 → DEM 高程（米）"""
    return _sample_dem(lon, lat)


def sample_game_height(x, z):
    """游戏坐标 → DEM 对应游戏高度"""
    # 游戏坐标 → 经纬度
    # 1 游戏单位 ≈ 对应 DEM 上的固定距离
    # DEM 覆盖约 0.11° × 0.10°，对应游戏世界约 140 × 140 单位
    # 所以将 DEM 网格线性映射到游戏世界网格
    lon_frac = (x + GAME_WORLD_SIZE / 2) / GAME_WORLD_SIZE
    lat_frac = (z + GAME_WORLD_SIZE / 2) / GAME_WORLD_SIZE
    lon = LON_MIN + lon_frac * (LON_MAX - LON_MIN)
    lat = LAT_MIN + lat_frac * (LAT_MAX - LAT_MIN)

    dem_h = _sample_dem(lon, lat)
    return dem_to_game_height(dem_h)


def sample_game_height_vec(x_arr, z_arr):
    """向量化版本：对一批游戏坐标批量查询 DEM 高度。

    参数:
        x_arr, z_arr: 同形状的 numpy 数组（float64/float32）。
    返回:
        与输入同形状的 numpy 数组（float64）。
    """
    if not _dem_ready or GRID_SIZE is None:
        return np.full_like(x_arr, GAME_BASE_HEIGHT, dtype=np.float64)

    orig_shape = x_arr.shape
    x_f = x_arr.ravel().astype(np.float64)
    z_f = z_arr.ravel().astype(np.float64)

    # 游戏坐标 → 经纬度
    half = GAME_WORLD_SIZE / 2.0
    lon_frac = (x_f + half) / GAME_WORLD_SIZE
    lat_frac = (z_f + half) / GAME_WORLD_SIZE
    lon = LON_MIN + lon_frac * (LON_MAX - LON_MIN)
    lat = LAT_MIN + lat_frac * (LAT_MAX - LAT_MIN)

    # 边界裁剪
    lon = np.clip(lon, LON_MIN, LON_MAX)
    lat = np.clip(lat, LAT_MIN, LAT_MAX)

    # 网格坐标
    max_idx = GRID_SIZE - 1
    col_f = (lon - LON_MIN) / (LON_MAX - LON_MIN) * max_idx
    row_f = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * max_idx

    col0 = np.floor(col_f).astype(np.int64)
    row0 = np.floor(row_f).astype(np.int64)
    col1 = np.clip(col0 + 1, 0, max_idx).astype(np.int64)
    row1 = np.clip(row0 + 1, 0, max_idx).astype(np.int64)

    fx = col_f - col0.astype(np.float64)
    fy = row_f - row0.astype(np.float64)

    h00 = _dem_smooth[row0, col0]
    h10 = _dem_smooth[row0, col1]
    h01 = _dem_smooth[row1, col0]
    h11 = _dem_smooth[row1, col1]

    h0 = h00 * (1.0 - fx) + h10 * fx
    h1 = h01 * (1.0 - fx) + h11 * fx
    result = h0 * (1.0 - fy) + h1 * fy

    # 转换为游戏高度
    result = result * _height_scale + _height_offset

    # 边界外平滑衰减到 GAME_BASE_HEIGHT
    margin = 0.02
    fade = np.ones_like(lon_frac)
    fade *= np.clip(lon_frac / margin, 0.0, 1.0)
    fade *= np.clip((1.0 - lon_frac) / margin, 0.0, 1.0)
    fade *= np.clip(lat_frac / margin, 0.0, 1.0)
    fade *= np.clip((1.0 - lat_frac) / margin, 0.0, 1.0)
    fade = np.clip(fade, 0.0, 1.0)
    result = result * fade + GAME_BASE_HEIGHT * (1.0 - fade)

    return result.reshape(orig_shape)


# ============================================================
# 四、对外接口（兼容 mountain.py 的签名）
# ============================================================

def dem_get_height(x, z):
    """DEM 版 get_terrain_height(x, z)"""
    h = sample_game_height(x, z)

    # 超出 DEM 边界时平滑衰减
    lon_frac = (x + GAME_WORLD_SIZE / 2) / GAME_WORLD_SIZE
    lat_frac = (z + GAME_WORLD_SIZE / 2) / GAME_WORLD_SIZE
    if not (0 <= lon_frac <= 1 and 0 <= lat_frac <= 1):
        # 超出部分用指数衰减
        dist = math.sqrt(x * x + z * z)
        falloff = max(0, 1 - (dist - 50) / 50)
        h = GAME_BASE_HEIGHT + (h - GAME_BASE_HEIGHT) * falloff

    return h


def dem_get_normal(x, z, eps=0.5):
    """DEM 版 get_terrain_normal"""
    hx = dem_get_height(x + eps, z)
    hx_ = dem_get_height(x - eps, z)
    hz = dem_get_height(x, z + eps)
    hz_ = dem_get_height(x, z - eps)
    dx = (hx - hx_) / (2.0 * eps)
    dz = (hz - hz_) / (2.0 * eps)
    n = np.array([-dx, 1.0, -dz], dtype=np.float32)
    norm = np.linalg.norm(n)
    if norm < 0.001:
        return (0.0, 1.0, 0.0)
    n = n / norm
    return (float(n[0]), float(n[1]), float(n[2]))


def dem_get_slope(x, z):
    """DEM 版坡度角（度）"""
    eps = 0.5
    hx = dem_get_height(x + eps, z)
    hx_ = dem_get_height(x - eps, z)
    hz = dem_get_height(x, z + eps)
    hz_ = dem_get_height(x, z - eps)
    dx = (hx - hx_) / (2.0 * eps)
    dz = (hz - hz_) / (2.0 * eps)
    return math.degrees(math.atan(math.sqrt(dx * dx + dz * dz)))


def print_dem_info():
    """打印 DEM 统计信息"""
    if not _dem_ready:
        print("[DEM] 未初始化")
        return

    print("=" * 50)
    print("  榛名山 DEM 地形信息")
    print("=" * 50)
    print(f"  覆盖: {LON_MIN:.3f}°~{LON_MAX:.3f}°E, "
          f"{LAT_MIN:.3f}°~{LAT_MAX:.3f}°N")
    print(f"  网格: {GRID_SIZE}×{GRID_SIZE}")
    print(f"  高程: {_dem_raw.min():.0f}m ~ {_dem_raw.max():.0f}m")
    print(f"  映射: DEM {_dem_min:.0f}m~{_dem_max:.0f}m → "
          f"游戏 {dem_to_game_height(_dem_min):.1f}~"
          f"{dem_to_game_height(_dem_max):.1f}")
    print(f"  参数: scale={_height_scale:.4f}, offset={_height_offset:.2f}")
    print(f"  平滑: 高斯σ1.2（轻量保留细节）")

    # 游戏坐标中的关键点
    points = [
        ("游戏原点", 0, 0),
        ("赛道起点", 3, 3),
        ("发夹弯1", -8, 0),
        ("发夹弯2", 3, -11.5),
        ("发夹弯3", 2.5, 14.5),
        ("赛道终点", 14, -14),
        ("远处东南", 50, 50),
        ("远处西北", -50, -50),
    ]
    print(f"\n  游戏坐标采样:")
    for name, x, z in points:
        h = dem_get_height(x, z)
        lon_frac = (x + GAME_WORLD_SIZE / 2) / GAME_WORLD_SIZE
        lat_frac = (z + GAME_WORLD_SIZE / 2) / GAME_WORLD_SIZE
        in_dem = 0 <= lon_frac <= 1 and 0 <= lat_frac <= 1
        mark = "" if in_dem else " [DEM外]"
        print(f"    {name:10s} ({x:5.1f},{z:5.1f}) → {h:7.2f}{mark}")
    print("=" * 50)


# ============================================================
# 五、DEM 雕刻：在山上凿路
# ============================================================

def carve_track_into_dem(path_points):
    """
    path_points: [(x, z, y, w, bank_rad), ...] 赛道路径点（游戏坐标/单位）
    修改 _dem_smooth 在赛道区域的地形，模拟"在山上凿路"的效果。
    赛道路面被压平，赛道边缘外 smoothstep 过渡回 DEM 高度。
    此函数在赛道网格和地形网格生成之前调用，一次雕刻，所有系统受益。
    """
    global _dem_smooth
    if not _dem_ready:
        print("[DEM 雕刻] DEM 未初始化，跳过")
        return

    n_path = len(path_points)
    print(f"[DEM 雕刻] 开始雕刻赛道 ({n_path} 个截面) 到 {GRID_SIZE}×{GRID_SIZE} DEM...")

    # ---- 路径数据数组化 ----
    path_xy = np.array([[p[0], p[1]] for p in path_points], dtype=np.float64)   # (N, 2)
    path_y  = np.array([p[2] for p in path_points], dtype=np.float64)            # (N,) 路中心高
    path_w  = np.array([p[3] for p in path_points], dtype=np.float64)            # (N,) 路宽
    path_bank = np.array([p[4] for p in path_points], dtype=np.float64)          # (N,) banking 弧度
    path_hw = path_w * 0.5                                                       # (N,) 半宽

    # ---- 路径法线（2D 水平法线，垂直于前进方向） ----
    path_nx = np.zeros(n_path, dtype=np.float64)
    path_nz = np.zeros(n_path, dtype=np.float64)
    for i in range(n_path):
        ni = min(i + 1, n_path - 1)
        dx = path_xy[ni, 0] - path_xy[i, 0]
        dz = path_xy[ni, 1] - path_xy[i, 1]
        length = math.hypot(dx, dz)
        if length > 1e-8:
            path_nx[i] = -dz / length
            path_nz[i] =  dx / length
        else:
            path_nx[i] = 1.0
            path_nz[i] = 0.0

    # ---- 影响半径与雕刻参数（游戏单位） ----
    INFLUENCE_RADIUS = 40.0  # 赛道边缘外 40 单位内被影响（扩大一倍，使过渡更平缓）
    CARVE_DEPTH = 0.8        # 赛道内部下挖深度（防止 smoothstep 凸起覆盖赛道）
    MAX_SLOPE = 0.35         # 过渡区最大坡度（约 19°，防止地形墙）
    FLANK_EXTRA = 3.0        # 赛道两侧沟槽强化宽度（超出半宽）

    # ---- 构建 KD-Tree 加速最近邻搜索 ----
    from scipy.spatial import cKDTree
    tree = cKDTree(path_xy)

    # ---- 逐格点处理 ----
    # 将 DEM 网格的游戏坐标预先算好
    col_frac = np.arange(GRID_SIZE, dtype=np.float64) / (GRID_SIZE - 1)
    row_frac = np.arange(GRID_SIZE, dtype=np.float64) / (GRID_SIZE - 1)
    grid_x = col_frac * GAME_WORLD_SIZE - GAME_WORLD_SIZE / 2  # (GRID_SIZE,)
    grid_z = row_frac * GAME_WORLD_SIZE - GAME_WORLD_SIZE / 2  # (GRID_SIZE,)
    grid_xx, grid_zz = np.meshgrid(grid_x, grid_z)  # 两个 (GRID_SIZE, GRID_SIZE)

    # 展平以便批量查询
    flat_x = grid_xx.ravel()  # (250000,)
    flat_z = grid_zz.ravel()
    cell_xy = np.column_stack([flat_x, flat_z])  # (250000, 2)

    # KD-Tree 批量查询：每个 DEM 格点找最近赛道点
    batch_size = 10000
    modified_count = 0
    for batch_start in range(0, len(cell_xy), batch_size):
        batch_end = min(batch_start + batch_size, len(cell_xy))
        batch_xy = cell_xy[batch_start:batch_end]

        # 查询最近赛道点
        dists, indices = tree.query(batch_xy)

        for j in range(len(batch_xy)):
            dist = dists[j]
            if dist > INFLUENCE_RADIUS:
                continue

            n_idx = indices[j]
            bx, bz = batch_xy[j]
            px, pz = path_xy[n_idx]
            dx_off = bx - px
            dz_off = bz - pz

            # 有符号横向距离（法线方向）
            lateral = dx_off * path_nx[n_idx] + dz_off * path_nz[n_idx]
            abs_lat = abs(lateral)
            hw = path_hw[n_idx]

            # 赛道高度（含 banking）
            h_road = float(path_y[n_idx]) + lateral * math.sin(float(path_bank[n_idx]))

            orig_h = dem_to_game_height(float(_dem_smooth.flat[batch_start + j]))

            if abs_lat <= hw:
                # 赛道内部：下挖 CARVE_DEPTH，确保地形低于路面
                target_h = h_road - CARVE_DEPTH
            else:
                # 赛道外部：smoothstep 过渡到 DEM，同时受坡度限制
                edge_dist = abs_lat - hw  # 距赛道边缘的距离
                t = edge_dist / INFLUENCE_RADIUS
                t = max(0.0, min(1.0, t))
                s = t * t * (3.0 - 2.0 * t)  # smoothstep
                smooth_h = h_road * (1.0 - s) + orig_h * s
                # 坡度限制：地形升高速度不得超过 MAX_SLOPE
                max_allowed = h_road + edge_dist * MAX_SLOPE
                target_h = min(smooth_h, max_allowed)

            # 转回 DEM 海拔并写入
            _dem_smooth.flat[batch_start + j] = game_to_dem_height(target_h)
            modified_count += 1

    print(f"[DEM 雕刻] 完成！修改了 {modified_count}/{GRID_SIZE*GRID_SIZE} 个 DEM 格点")

    # ---- 第二遍：赛道沟槽强化 ----
    # 确保赛道及其两侧 FLANK_EXTRA 范围内有明显下凹，
    # 抵消 smoothstep 过渡在赛道边缘处可能产生的凸起。
    GROOVE_OFFSET = 0.5  # 沟槽面比赛道中心再低 0.5
    groove_modified = 0
    for batch_start in range(0, len(cell_xy), batch_size):
        batch_end = min(batch_start + batch_size, len(cell_xy))
        batch_xy = cell_xy[batch_start:batch_end]
        dists, indices = tree.query(batch_xy)

        for j in range(len(batch_xy)):
            n_idx = indices[j]
            dist = dists[j]
            bx, bz = batch_xy[j]
            px, pz = path_xy[n_idx]
            dx_off = bx - px
            dz_off = bz - pz
            lateral = dx_off * path_nx[n_idx] + dz_off * path_nz[n_idx]
            abs_lat = abs(lateral)
            hw = path_hw[n_idx]

            # 只处理赛道附近区域
            if abs_lat > hw + FLANK_EXTRA:
                continue
            if dist > hw + FLANK_EXTRA + 1.0:
                continue

            h_road = float(path_y[n_idx]) + lateral * math.sin(float(path_bank[n_idx]))
            current_game_h = dem_to_game_height(float(_dem_smooth.flat[batch_start + j]))

            # 沟槽目标高度：赛道内 = h_road - CARVE_DEPTH，边缘 = h_road - 0.2
            if abs_lat <= hw:
                groove_target = h_road - CARVE_DEPTH * 0.9
            else:
                blend = (abs_lat - hw) / FLANK_EXTRA
                groove_target = h_road - 0.2 * (1.0 - blend) - GROOVE_OFFSET * blend

            if current_game_h > groove_target:
                _dem_smooth.flat[batch_start + j] = game_to_dem_height(groove_target)
                groove_modified += 1

    print(f"[DEM 雕刻] 沟槽强化: 额外修改 {groove_modified} 格点")


# ============================================================
# 六、主入口：测试
# ============================================================
if __name__ == "__main__":
    init_dem()
    print()
    print_dem_info()

    # 对比原版和 DEM 版的赛道高度剖面
    print("\n赛道高度剖面对比（沿控制点）:")
    print(f"  {'点':>4s}  {'游戏X':>6s} {'游戏Z':>6s}  "
          f"{'原版':>8s} {'DEM':>8s} {'差':>8s}")
    print("  " + "-" * 44)

    from mountain import get_terrain_height as old_height
    for i, (x, z, w, b) in enumerate([
        (3, 3, 7, 0), (5, 0.5, 6.5, 2), (-8, 0, 5.5, -7),
        (3, -11.5, 5.5, -6), (2.5, 14.5, 6, -6), (14, -14, 8, 0)
    ]):
        old_h = old_height(x, z)
        new_h = dem_get_height(x, z)
        print(f"  CP{i:02d}  {x:6.1f} {z:6.1f}  "
              f"{old_h:8.2f} {new_h:8.2f} {new_h - old_h:+8.2f}")