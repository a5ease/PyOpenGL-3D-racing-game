"""
赛道边界防入侵系统（距离场 + 精确查询）
==========================================
提供统一的有符号距离场 API，用于判断任意点是否侵入赛道。
所有需要避开赛道的模块（树林、地形、护栏等）统一通过此模块查询。

核心策略：
  1. 精确模式（默认）——遍历路径点找最近邻，O(N) 但 N≈8100，单次≈0.01ms
  2. 距离场模式（可选）——预计算 grid 缓存到 .npy，O(1) 批量查询
"""

import numpy as np
import math
import os
import hashlib

# ============================================================
# 全局数据（由 _ensure_data() 惰性初始化）
# ============================================================
_TRACK_POINTS = None       # (M, 2) float32 — 赛道截面中心点 (x, z)
_TRACK_HALFWIDTHS = None   # (M,) float32 — 每个截面的赛道半宽（= TRACK_EDGE_RATIO * w）
_TRACK_NORMALS = None      # (M, 2) float32 — 每个截面的水平法线 (nx, nz)
_TRACK_HEIGHTS = None      # (M,) float32 — 每个截面的路面高度 (y)
_TRACK_BANK = None         # (M,) float32 — 每个截面的 banking 弧度
_TRACK_CENTER_Y = None     # (M,) float32 — 每个截面中心高度
_KD_TREE = None            # cKDTree 缓存（惰性构建，加速最近邻）

# ⚠️ signed_distance 的 0 点约定：sd = |横向偏移| - TRACK_EDGE_RATIO * w
#    即 sd=0 落在**沥青路缘**（±3.5m，w=10），不含 3.5~5.0 的褐色路肩。
#    全工程有三种"半宽"，改动前务必分清：
#      0.35w = 沥青路缘      ← signed_distance / is_safe / 空气墙
#      0.50w = 赛道棱柱外缘  （视觉 PROFILE 覆盖到 ±5.0，地形压平带基准）
#      距中心线距离 = sd + TRACK_EDGE_RATIO * w   ← 坡脚/道钉等反算用这个
TRACK_EDGE_RATIO = 0.35

# 距离场 grid（可选）
_DISTANCE_FIELD = None     # (H, W) float32
_FIELD_ORIGIN = None       # (x_min, z_min)
_FIELD_RESOLUTION = None   # 步长
_FIELD_SHAPE = None        # (H, W)

# 缓存目录
_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


# ============================================================
# 数据初始化
# ============================================================
def _ensure_data():
    """惰性加载赛道数据（首次查询时自动初始化）"""
    global _TRACK_POINTS, _TRACK_HALFWIDTHS, _TRACK_NORMALS, _TRACK_HEIGHTS
    global _TRACK_BANK, _TRACK_CENTER_Y
    if _TRACK_POINTS is not None:
        return
    # 函数内导入避免循环依赖（mountain 也会导入本模块）
    from mountain import get_akina_path
    path = get_akina_path()
    M = len(path)
    pts = np.zeros((M, 2), dtype=np.float32)
    hw = np.zeros(M, dtype=np.float32)
    nxs = np.zeros(M, dtype=np.float32)
    nzs = np.zeros(M, dtype=np.float32)
    bank = np.zeros(M, dtype=np.float32)
    center_y = np.zeros(M, dtype=np.float32)
    for i in range(M):
        x, z, y, w, bank_rad = path[i]
        pts[i] = [x, z]
        hw[i] = w * TRACK_EDGE_RATIO  # 半宽（匹配视觉沥青路边缘 ±3.5，不包含褐色路肩 ±3.5~±5.0）
        # 注意：地形压平用 0.5w（覆盖路肩到碎石带），这里 0.35w 是空气墙/物理边界。
        # 二者偏移差 (0.5-0.35)w = 1.5m(w=10) 构成路肩视觉缓冲区，是故意设计。
        bank[i] = bank_rad
        center_y[i] = y
        # 水平法线 = 切线旋转 90°（垂直于前进方向）
        ni = min(i + 1, M - 1)
        dx = path[ni][0] - x
        dz = path[ni][1] - z
        length = math.hypot(dx, dz)
        if length > 1e-8:
            tx, tz = dx / length, dz / length
            nxs[i] = -tz   # 法线 = 切线逆时针旋转 90°
            nzs[i] = tx
        else:
            nxs[i] = 1.0
            nzs[i] = 0.0
    # 存储高度（用于 3D 边缘坐标）
    heights = np.array([p[2] for p in path], dtype=np.float32)
    _TRACK_HEIGHTS = heights
    _TRACK_POINTS = pts
    _TRACK_HALFWIDTHS = hw
    _TRACK_NORMALS = np.column_stack([nxs, nzs])
    _TRACK_BANK = bank
    _TRACK_CENTER_Y = center_y


def _cache_path(name):
    """获取缓存文件路径"""
    p = os.path.join(_CACHE_DIR, f"{name}.npy")
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
    except OSError:
        return None
    return p


def _geometry_signature():
    """赛道几何签名：控制点/缩放/裁剪参数任一变化 → 缓存自动失效。

    v2（2026-09）：额外纳入 path 指纹（mountain._path_fingerprint）。
    旧版只哈希 CONTROL_POINTS，而 path 还会被 _preprocess_control_points
    （重采样间距/平滑系数）、_limit_path_slope（坡度裁剪）、_add_pullout_bays
    （避车道）改写 —— 这些改动不改控制点，签名却不变，于是
    track_height_field / tri_index / distance_field 会命中旧路的缓存，
    物理高度查询与渲染路面对不上（车悬空或陷地）。
    """
    from mountain import (CONTROL_POINTS, _S, PROFILE_VERSION,
                          get_akina_path, _path_fingerprint)
    # CONTROL_POINTS = [(x, z, width, bank_deg), ...] — 4 元组
    blob = repr([(round(float(p[0]), 3), round(float(p[1]), 3),
                  round(float(p[2]), 3), round(float(p[3]), 3)) for p in CONTROL_POINTS]).encode()
    blob += b"|S=%s|slope=0.08|seg=20" % str(_S).encode()
    blob += b"|PROFILE=%s" % str(PROFILE_VERSION).encode()
    blob += b"|vertex_layout=v2"  # v2：顶点打包 60B→28B，需重建 tri_index 缓存
    # path 指纹：path 一变，几何缓存整体换代
    try:
        blob += b"|path=%s" % _path_fingerprint(get_akina_path()).encode()
    except Exception as e:
        blob += b"|path=unavailable"
        print(f"[track_bounds] path 指纹不可用（缓存可能命中旧路）: {e}")
    return hashlib.md5(blob).hexdigest()[:10]

def _versioned_cache_path(base):
    """带几何签名的缓存路径，赛道一变缓存自动失效"""
    return _cache_path(f"{base}_{_geometry_signature()}")


def _clean_stale_geometry_caches(base, keep_name):
    """删除同一基名下签名不同的旧缓存（赛道几何换代后不留化石）。

    base="track_height_field" 的清理不会碰 "track_tex_*.png"（后缀不符/前缀不符）。
    """
    try:
        names = os.listdir(_CACHE_DIR)
    except OSError:
        return
    keep = {keep_name, keep_name + ".npy"}
    legacy = {base + ".npy"}
    pre = base + "_"
    removed = 0
    for fname in names:
        if fname in keep or not fname.endswith(".npy"):
            continue
        if fname in legacy or fname.startswith(pre):
            try:
                os.remove(os.path.join(_CACHE_DIR, fname))
                removed += 1
            except OSError:
                pass
    if removed:
        print(f"[track_bounds] 清理 {removed} 个过期赛道几何缓存 ({base}*)")


# ============================================================
# 段投影 + 段内线性插值（与视觉三角形插值同构）—— v2 核心修复
# ============================================================

def _project_nearest_segment(x, z, window=2):
    """
    在最近路径点附近的局部窗口内做"点到线段"投影。
    v3：使用 cKDTree 缓存，从 O(M) 线性搜索降为 O(log M)。
    """
    global _KD_TREE
    pts = _TRACK_POINTS
    M = len(pts)

    # 惰性构建 KDTree（只一次）
    if _KD_TREE is None:
        try:
            from scipy.spatial import cKDTree
            _KD_TREE = cKDTree(pts)
            print(f"[track_bounds] KDTree 已构建 ({M} 路径点)")
        except ImportError:
            _KD_TREE = False

    if _KD_TREE is not False:
        _, idx = _KD_TREE.query([x, z])
        idx = int(idx)
    else:
        # 回退：无 scipy 时用线性搜索
        dx = pts[:, 0] - x
        dz = pts[:, 1] - z
        idx = int(np.argmin(dx * dx + dz * dz))

    best_d2 = 1e18
    bi, bt, bpx, bpz = idx, 0.0, float(pts[idx, 0]), float(pts[idx, 1])
    lo = max(0, idx - window)
    hi = min(M - 1, idx + window)
    for i in range(lo, hi):
        ax, az = float(pts[i, 0]), float(pts[i, 1])
        bx, bz = float(pts[i + 1, 0]), float(pts[i + 1, 1])
        ex, ez = bx - ax, bz - az
        L2 = ex * ex + ez * ez
        if L2 < 1e-12:
            t = 0.0
        else:
            t = ((x - ax) * ex + (z - az) * ez) / L2
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        px, pz = ax + ex * t, az + ez * t
        d2 = (x - px) * (x - px) + (z - pz) * (z - pz)
        if d2 < best_d2:
            best_d2 = d2
            bi, bt, bpx, bpz = i, t, px, pz
    return bi, bt, bpx, bpz


# ============================================================
# 有符号距离：精确模式（遍历路径点）
# ============================================================
def signed_distance(x, z):
    """
    计算点到赛道边缘的有符号距离（v2：段投影版）。
    在局部窗口内做"点到线段"投影，弯道折叠处不再被对臂绑架。

    返回值：
        > 0 → 赛道外（安全）
        = 0 → 赛道边缘
        < 0 → 赛道内（侵入）
    """
    _ensure_data()
    i, t, px, pz = _project_nearest_segment(x, z)
    ex = float(_TRACK_POINTS[i + 1, 0] - _TRACK_POINTS[i, 0])
    ez = float(_TRACK_POINTS[i + 1, 1] - _TRACK_POINTS[i, 1])
    L = math.hypot(ex, ez)
    nx, nz = (-ez / L, ex / L) if L > 1e-10 else (1.0, 0.0)
    lateral = (x - px) * nx + (z - pz) * nz
    hw = _TRACK_HALFWIDTHS[i] * (1.0 - t) + _TRACK_HALFWIDTHS[i + 1] * t
    return abs(lateral) - hw


def nearest_track_info(x, z):
    """
    获取最近路径点的详细信息（v2：段投影版）。
    返回 (idx, 半宽, 法线_nx, 法线_nz, 横向距离, 赛道高度)
    """
    _ensure_data()
    i, t, px, pz = _project_nearest_segment(x, z)
    ex = float(_TRACK_POINTS[i + 1, 0] - _TRACK_POINTS[i, 0])
    ez = float(_TRACK_POINTS[i + 1, 1] - _TRACK_POINTS[i, 1])
    L = math.hypot(ex, ez)
    nx, nz = (-ez / L, ex / L) if L > 1e-10 else (1.0, 0.0)
    lateral = (x - px) * nx + (z - pz) * nz
    hw = _TRACK_HALFWIDTHS[i] * (1.0 - t) + _TRACK_HALFWIDTHS[i + 1] * t
    y = _TRACK_HEIGHTS[i] * (1.0 - t) + _TRACK_HEIGHTS[i + 1] * t
    return i, hw, nx, nz, lateral, y


def is_safe(x, z, margin=0.0):
    """检查点是否安全（赛道外且距离边缘 >= margin）"""
    return signed_distance(x, z) >= margin


# ============================================================
# 距离场 Grid（批量查询加速）
# ============================================================
GRID_RESOLUTION = 2.0      # 每格 2 单位，精度足够
FIELD_EXTRA_MARGIN = 80.0  # 赛道包围盒外扩范围

def _build_field_data():
    """
    用 numpy 分批构建有符号距离场 grid。
    结果缓存到 .npy，首次运行约需 10-30 秒。
    """
    _ensure_data()
    pts = _TRACK_POINTS     # (M, 2)
    hw = _TRACK_HALFWIDTHS  # (M,)
    norms = _TRACK_NORMALS  # (M, 2)
    M = len(pts)

    # 包围盒
    x_min = float(pts[:, 0].min()) - FIELD_EXTRA_MARGIN
    x_max = float(pts[:, 0].max()) + FIELD_EXTRA_MARGIN
    z_min = float(pts[:, 1].min()) - FIELD_EXTRA_MARGIN
    z_max = float(pts[:, 1].max()) + FIELD_EXTRA_MARGIN

    res = GRID_RESOLUTION
    cols = int(math.ceil((x_max - x_min) / res)) + 1
    rows = int(math.ceil((z_max - z_min) / res)) + 1
    total = rows * cols

    print(f"[track_bounds] 构建距离场 {cols}×{rows} ({total/1e6:.1f}M 格点)...")

    # 生成网格坐标
    x_vals = np.linspace(x_min, x_min + (cols - 1) * res, cols, dtype=np.float32)
    z_vals = np.linspace(z_min, z_min + (rows - 1) * res, rows, dtype=np.float32)
    X, Z = np.meshgrid(x_vals, z_vals)
    X_f = X.ravel()  # (total,)
    Z_f = Z.ravel()  # (total,)

    # 分批处理
    field = np.zeros(total, dtype=np.float32)
    batch_size = 2000
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        bx = X_f[start:end]  # (B,)
        bz = Z_f[start:end]  # (B,)
        B = end - start

        # 距离矩阵 (B, M)
        d = np.zeros((B, M, 2), dtype=np.float32)
        d[:, :, 0] = bx[:, np.newaxis] - pts[np.newaxis, :, 0]
        d[:, :, 1] = bz[:, np.newaxis] - pts[np.newaxis, :, 1]
        dist2 = d[:, :, 0] * d[:, :, 0] + d[:, :, 1] * d[:, :, 1]

        # 最近路径点
        idx = np.argmin(dist2, axis=1)  # (B,)

        # 取法线、半宽、距离分量
        nx = norms[idx, 0]   # (B,)
        nz = norms[idx, 1]   # (B,)
        hw_i = hw[idx]       # (B,)
        dxi = d[np.arange(B), idx, 0]  # (B,)
        dzi = d[np.arange(B), idx, 1]  # (B,)

        lateral = dxi * nx + dzi * nz
        field[start:end] = np.abs(lateral) - hw_i

    field = field.reshape(rows, cols)
    return field, (x_min, z_min), res


def build_field(force=False):
    """
    构建距离场 grid，自动缓存到 .npy。
    force=True 强制重建。
    """
    global _DISTANCE_FIELD, _FIELD_ORIGIN, _FIELD_RESOLUTION, _FIELD_SHAPE

    cache_p = _versioned_cache_path("distance_field")
    if cache_p:
        _clean_stale_geometry_caches(
            "distance_field", os.path.splitext(os.path.basename(cache_p))[0])
    if not force and cache_p and os.path.exists(cache_p):
        print("[track_bounds] 加载距离场缓存...")
        data = np.load(cache_p, allow_pickle=True).item()
        _DISTANCE_FIELD = data["field"]
        _FIELD_ORIGIN = tuple(data["origin"])
        _FIELD_RESOLUTION = float(data["resolution"])
        _FIELD_SHAPE = _DISTANCE_FIELD.shape
        print(f"[track_bounds]  距离场加载完成: {_FIELD_SHAPE}")
        return

    field, origin, res = _build_field_data()
    _DISTANCE_FIELD = field
    _FIELD_ORIGIN = origin
    _FIELD_RESOLUTION = res
    _FIELD_SHAPE = field.shape

    if cache_p:
        np.save(cache_p, {
            "field": field,
            "origin": origin,
            "resolution": res,
        })
        print(f"[track_bounds]  距离场已缓存 ({field.shape})")


def query_field(x, z):
    """
    从距离场 grid 双线性插值查询。
    如果 grid 未构建，回退到精确模式。
    """
    if _DISTANCE_FIELD is None:
        return signed_distance(x, z)

    x_min, z_min = _FIELD_ORIGIN
    res = _FIELD_RESOLUTION

    fx = (x - x_min) / res
    fz = (z - z_min) / res
    rows, cols = _FIELD_SHAPE

    ix = int(fx)
    iz = int(fz)
    if ix < 0 or ix >= cols - 1 or iz < 0 or iz >= rows - 1:
        return signed_distance(x, z)

    rx = fx - ix
    rz = fz - iz

    d00 = _DISTANCE_FIELD[iz, ix]
    d10 = _DISTANCE_FIELD[iz, ix + 1]
    d01 = _DISTANCE_FIELD[iz + 1, ix]
    d11 = _DISTANCE_FIELD[iz + 1, ix + 1]

    d0 = d00 * (1.0 - rx) + d10 * rx
    d1 = d01 * (1.0 - rx) + d11 * rx
    return d0 * (1.0 - rz) + d1 * rz


def _query_field_batch(points):
    """
    向量化批量查询距离场（双线性插值）。
    points: (P, 2) float32 数组
    返回: (P,) float32 数组 — 每个点的有符号距离
    """
    x_min, z_min = _FIELD_ORIGIN
    res = _FIELD_RESOLUTION
    rows, cols = _FIELD_SHAPE

    fx = (points[:, 0] - x_min) / res
    fz = (points[:, 1] - z_min) / res
    ix = np.floor(fx).astype(np.int32)
    iz = np.floor(fz).astype(np.int32)

    # 边界掩码（至少需要 2×2 格点做双线性插值）
    valid = (ix >= 0) & (ix < cols - 1) & (iz >= 0) & (iz < rows - 1)

    result = np.empty(len(points), dtype=np.float32)

    # 网格内的点：向量化双线性插值
    if np.any(valid):
        vx = ix[valid]
        vz = iz[valid]
        vrx = fx[valid] - vx.astype(np.float32)
        vrz = fz[valid] - vz.astype(np.float32)

        d00 = _DISTANCE_FIELD[vz, vx]
        d10 = _DISTANCE_FIELD[vz, vx + 1]
        d01 = _DISTANCE_FIELD[vz + 1, vx]
        d11 = _DISTANCE_FIELD[vz + 1, vx + 1]

        d0 = d00 * (1.0 - vrx) + d10 * vrx
        d1 = d01 * (1.0 - vrx) + d11 * vrx
        result[valid] = d0 * (1.0 - vrz) + d1 * vrz

    # 网格外的点：回退精确模式
    invalid = ~valid
    if np.any(invalid):
        invalid_idx = np.where(invalid)[0]
        for i in invalid_idx:
            result[i] = signed_distance(points[i, 0], points[i, 1])

    return result


def is_safe_field(x, z, margin=0.0):
    """从距离场查询安全状态"""
    return query_field(x, z) >= margin


# ============================================================
# 批量查询（用于生成阶段一次性检查大量点）
# ============================================================
def batch_safe_mask(points, margin=0.0):
    """
    批量检查多个点是否安全。
    points: (P, 2) float32 数组
    返回: (P,) bool 数组 — True = 安全
    """
    _ensure_data()
    if _DISTANCE_FIELD is not None:
        # 使用距离场——向量化批处理
        return _query_field_batch(points) >= margin

    # 精确模式 + numpy 分批
    pts = _TRACK_POINTS
    hw = _TRACK_HALFWIDTHS
    norms = _TRACK_NORMALS
    M = len(pts)
    P = len(points)
    mask = np.zeros(P, dtype=bool)

    batch_size = 5000
    for start in range(0, P, batch_size):
        end = min(start + batch_size, P)
        batch = points[start:end]
        B = end - start

        # 距离矩阵 (B, M)
        dx = batch[:, 0, np.newaxis] - pts[np.newaxis, :, 0]  # (B, M)
        dz = batch[:, 1, np.newaxis] - pts[np.newaxis, :, 1]  # (B, M)
        dist2 = dx * dx + dz * dz
        idx = np.argmin(dist2, axis=1)

        nx = norms[idx, 0]
        nz = norms[idx, 1]
        hw_i = hw[idx]
        dxi = dx[np.arange(B), idx]
        dzi = dz[np.arange(B), idx]

        lateral = dxi * nx + dzi * nz
        dist = np.abs(lateral) - hw_i
        mask[start:end] = dist >= margin

    return mask


# ============================================================
# 七、赛道路径边缘点（供护栏/路障/地形放置使用）
# ============================================================

def get_track_edge_points():
    """
    返回赛道左右两侧边缘点坐标。
    所有放置代码（护栏、路障、地形裁剪）都应使用此函数，
    确保与 signed_distance 使用的边缘保持完全一致。

    Returns:
        left_edge: (M, 2) float32 — 左侧边缘点 (x, z)
        right_edge: (M, 2) float32 — 右侧边缘点 (x, z)
        heights: (M,) float32 — 每个截面的路面高度 (y)
    """
    _ensure_data()

    pts = _TRACK_POINTS          # (M, 2)
    nx_all = _TRACK_NORMALS[:, 0]  # (M,)
    nz_all = _TRACK_NORMALS[:, 1]  # (M,)
    hw = _TRACK_HALFWIDTHS       # (M,)

    # 左侧 = 中心线 + 法线 * 半宽
    left_edge = np.column_stack([
        pts[:, 0] + nx_all * hw,
        pts[:, 1] + nz_all * hw,
    ])
    # 右侧 = 中心线 - 法线 * 半宽
    right_edge = np.column_stack([
        pts[:, 0] - nx_all * hw,
        pts[:, 1] - nz_all * hw,
    ])

    return left_edge, right_edge, _TRACK_HEIGHTS


# ============================================================
# 八、Catmull-Rom 样条参数化统一（视觉赛道与物理赛道同源）
# ============================================================

# 全局样条数据（由 _ensure_spline_data() 惰性初始化）
_SPLINE_PATH = None       # (S, 5) float64 — 高密度样条点 [x, z, y, w, bank]
_SPLINE_ARCLEN = None     # (S,) float64 — 累积弧长
_SPLINE_TOTAL_LEN = 0.0   # 总弧长
_SPLINE_NORMALS = None    # (S, 2) float64 — 每个样条点的水平法线 (nx, nz)
_SPLINE_SEGMENTS = 80     # 每段路径的细分段数（视觉赛道用 segments=20，这里用 80 提升精度）


def _ensure_spline_data():
    """
    构建高密度 Catmull-Rom 样条数据，用于精确的赛道表面查询。
    与视觉赛道使用相同的样条公式，保证物理查询与视觉渲染同源。
    """
    global _SPLINE_PATH, _SPLINE_ARCLEN, _SPLINE_TOTAL_LEN, _SPLINE_NORMALS
    if _SPLINE_PATH is not None:
        return

    from mountain import get_akina_path
    # 获取高密度路径点 + RMF 标架（无翻转的切线/法线/副法线）
    path, tangents, normals, binormals = get_akina_path(
        segments=_SPLINE_SEGMENTS, return_tangents=True
    )
    n = len(path)
    if n < 2:
        _SPLINE_PATH = np.zeros((0, 5), dtype=np.float64)
        _SPLINE_ARCLEN = np.zeros(0, dtype=np.float64)
        _SPLINE_NORMALS = np.zeros((0, 2), dtype=np.float64)
        _SPLINE_TOTAL_LEN = 0.0
        return

    # 提取 5 维数据和水平法线
    pts = np.zeros((n, 5), dtype=np.float64)
    spline_normals = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        pts[i] = [path[i][0], path[i][1], path[i][2], path[i][3], path[i][4]]
        # ★ 水平法线 = 切线旋转 90°（绝对一致于 build_mountain_road 的 nx_2d = -tz, nz_2d = tx）
        #   不用 RMF 法线投影，避免坡道处两者方向微小偏差导致 banking 修正高度不一致
        tx = float(tangents[i][0])
        tz = float(tangents[i][2])
        length = math.hypot(-tz, tx)
        if length > 1e-8:
            spline_normals[i] = [-tz / length, tx / length]
        else:
            spline_normals[i] = [1.0, 0.0]

    # 计算累积弧长
    arclen = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        dx = pts[i, 0] - pts[i-1, 0]
        dy = pts[i, 2] - pts[i-1, 2]  # 3D 弧长包含高度变化
        dz = pts[i, 1] - pts[i-1, 1]
        arclen[i] = arclen[i-1] + math.sqrt(dx*dx + dy*dy + dz*dz)

    total = arclen[-1]
    if total < 1e-10:
        total = 1.0

    _SPLINE_PATH = pts
    _SPLINE_ARCLEN = arclen
    _SPLINE_NORMALS = spline_normals
    _SPLINE_TOTAL_LEN = total


def _spline_sample_by_arclength(s):
    """
    根据弧长参数 s（实际距离，非归一化）采样样条。
    返回插值后的 (x, z, y, w, bank)。
    """
    _ensure_spline_data()
    pts = _SPLINE_PATH
    arclen = _SPLINE_ARCLEN
    total = _SPLINE_TOTAL_LEN

    if len(pts) < 2:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    # 夹紧到有效范围
    s = max(0.0, min(total, s))

    # 二分查找来越过 s 的路径段
    idx = int(np.searchsorted(arclen, s, side='right') - 1)
    idx = max(0, min(idx, len(pts) - 2))

    # 段内线性插值（基于弧长）
    seg_len = arclen[idx + 1] - arclen[idx]
    if seg_len < 1e-10:
        t = 0.0
    else:
        t = (s - arclen[idx]) / seg_len
    t = max(0.0, min(1.0, t))

    # 线性插值 5 维数据
    p0 = pts[idx]
    p1 = pts[idx + 1]
    result = p0 + (p1 - p0) * t

    return (float(result[0]), float(result[1]), float(result[2]),
            float(result[3]), float(result[4]))


def _sample_track_height(x, z):
    """
    查询 (x, z) 处赛道表面高度 —— v2：段投影 + 段内线性插值。

    视觉赛道把相邻截面的 profile 顶点连成三角形，GPU 在三角形内
    对顶点位置线性插值。车道层顶点 y_i = cy_i + o·sin(bank_i)，
    沿段线性插值 ⇒ 本函数在垂足所在段内对 (cy, bank) 线性插值后
    叠加 lateral·sin(bank)，与渲染表面数学同构。

    旧缺陷：最近邻单点外推产生"阶梯高度场"，弯道 banking 变化处
    与连续插值的视觉表面系统性偏差。
    """
    _ensure_data()
    if len(_TRACK_POINTS) < 2:
        return 0.0

    i, t, px, pz = _project_nearest_segment(x, z)

    # 段内线性插值：中心高度与 banking
    y  = float(_TRACK_HEIGHTS[i]) * (1.0 - t) + float(_TRACK_HEIGHTS[i + 1]) * t
    bk = float(_TRACK_BANK[i])    * (1.0 - t) + float(_TRACK_BANK[i + 1])    * t

    # 段方向 → 水平法线（与视觉 nx_2d=-tz, nz_2d=tx 完全一致）
    ex = float(_TRACK_POINTS[i + 1, 0] - _TRACK_POINTS[i, 0])
    ez = float(_TRACK_POINTS[i + 1, 1] - _TRACK_POINTS[i, 1])
    L = math.hypot(ex, ez)
    nx, nz = (-ez / L, ex / L) if L > 1e-10 else (1.0, 0.0)
    lateral = (x - px) * nx + (z - pz) * nz

    return float(y + lateral * math.sin(bk))


# ============================================================
# 九、轻量级赛道高度场缓存（O(1) 查询加速）
# ============================================================

_HEIGHT_FIELD = None       # (H, W) float32 — 赛道表面高度
_HF_ORIGIN = None          # (x_min, z_min)
_HF_RESOLUTION = 1.0       # 每格 1 单位，精度高
_HF_SHAPE = None           # (H, W)
_HF_MARGIN = 30.0          # 赛道包围盒外扩范围


def _build_height_field():
    """
    构建赛道表面高度场 grid。
    使用样条参数化查询每个格点的高度，保证与视觉赛道完全一致。
    结果缓存到 .npy，首次运行约需几秒。
    """
    _ensure_data()

    pts = _TRACK_POINTS

    # 包围盒
    margin = _HF_MARGIN
    x_min = float(pts[:, 0].min()) - margin
    x_max = float(pts[:, 0].max()) + margin
    z_min = float(pts[:, 1].min()) - margin
    z_max = float(pts[:, 1].max()) + margin

    res = _HF_RESOLUTION
    cols = int(math.ceil((x_max - x_min) / res)) + 1
    rows = int(math.ceil((z_max - z_min) / res)) + 1
    total = rows * cols

    print(f"[track_bounds] 构建赛道高度场 {cols}×{rows} ({total:,} 格点, 分辨率 {res})...")
    t0 = __import__('time').time()

    # 生成网格坐标
    x_vals = np.linspace(x_min, x_min + (cols - 1) * res, cols, dtype=np.float32)
    z_vals = np.linspace(z_min, z_min + (rows - 1) * res, rows, dtype=np.float32)
    X, Z = np.meshgrid(x_vals, z_vals)
    X_f = X.ravel()
    Z_f = Z.ravel()

    # 分批构建高度场
    height_field = np.zeros(total, dtype=np.float32)
    batch_size = 5000

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        bx = X_f[start:end]
        bz = Z_f[start:end]
        B = end - start

        # 对 batch 中每个点，找最近路径点
        d_all = np.zeros((B, len(pts)), dtype=np.float32)
        d_all[:, :] = (bx[:, np.newaxis] - pts[np.newaxis, :, 0]) ** 2
        d_all[:, :] += (bz[:, np.newaxis] - pts[np.newaxis, :, 1]) ** 2
        nearest_idx = np.argmin(d_all, axis=1)

        # 对每个点用样条采样高度
        for j in range(B):
            # 跳过距离赛道太远的点（远于 margin）
            idx_j = int(nearest_idx[j])
            dx_j = bx[j] - pts[idx_j, 0]
            dz_j = bz[j] - pts[idx_j, 1]
            dist = math.sqrt(dx_j*dx_j + dz_j*dz_j)
            if dist > margin:
                height_field[start + j] = -999.0  # 标记为无效
                continue

            # 使用样条统一高度查询
            h = _sample_track_height(float(bx[j]), float(bz[j]))
            height_field[start + j] = float(h)

        # 进度提示
        progress = (start + B) / total * 100
        if int(progress / 10) > int((start) / total * 100 / 10):
            print(f"  ... {progress:.0f}%")

    height_field = height_field.reshape(rows, cols)
    elapsed = __import__('time').time() - t0
    print(f"[track_bounds]  高度场构建完成 ({elapsed:.1f}s)")

    return height_field, (x_min, z_min), res


def build_height_field(force=False):
    """
    构建/加载赛道高度场缓存。
    force=True 强制重建。
    """
    global _HEIGHT_FIELD, _HF_ORIGIN, _HF_RESOLUTION, _HF_SHAPE

    cache_p = _versioned_cache_path("track_height_field")
    if cache_p:
        _clean_stale_geometry_caches(
            "track_height_field", os.path.splitext(os.path.basename(cache_p))[0])
    if not force and cache_p and os.path.exists(cache_p):
        print("[track_bounds] 加载赛道高度场缓存...")
        data = np.load(cache_p, allow_pickle=True).item()
        _HEIGHT_FIELD = data["field"]
        _HF_ORIGIN = tuple(data["origin"])
        _HF_RESOLUTION = float(data["resolution"])
        _HF_SHAPE = _HEIGHT_FIELD.shape
        print(f"[track_bounds]  高度场加载完成: {_HF_SHAPE}")
        return

    field, origin, res = _build_height_field()
    _HEIGHT_FIELD = field
    _HF_ORIGIN = origin
    _HF_RESOLUTION = res
    _HF_SHAPE = field.shape

    if cache_p:
        try:
            np.save(cache_p, {
                "field": field,
                "origin": origin,
                "resolution": res,
            })
            print(f"[track_bounds]  高度场已缓存 ({field.shape})")
        except Exception as e:
            print(f"[track_bounds]  高度场缓存失败: {e}")


def _query_height_field(x, z):
    """
    从高度场双线性插值查询赛道表面高度。
    如果点超出高度场范围，回退到样条查询。
    """
    if _HEIGHT_FIELD is None:
        return _sample_track_height(x, z)

    x_min, z_min = _HF_ORIGIN
    res = _HF_RESOLUTION

    fx = (x - x_min) / res
    fz = (z - z_min) / res
    rows, cols = _HF_SHAPE

    ix = int(fx)
    iz = int(fz)
    if ix < 0 or ix >= cols - 1 or iz < 0 or iz >= rows - 1:
        return _sample_track_height(x, z)

    rx = fx - ix
    rz = fz - iz

    h00 = _HEIGHT_FIELD[iz, ix]
    h10 = _HEIGHT_FIELD[iz, ix + 1]
    h01 = _HEIGHT_FIELD[iz + 1, ix]
    h11 = _HEIGHT_FIELD[iz + 1, ix + 1]

    # 检查是否有无效值
    if h00 < -900 or h10 < -900 or h01 < -900 or h11 < -900:
        return _sample_track_height(x, z)

    h0 = h00 * (1.0 - rx) + h10 * rx
    h1 = h01 * (1.0 - rx) + h11 * rx
    return h0 * (1.0 - rz) + h1 * rz


# ============================================================
# 十、重写 surface_height_at — 使用统一高度查询
# ============================================================

def surface_height_at(x, z):
    """统一入口：赛道面上走三角形（误差0），面外走段投影 fallback（连续过渡）"""
    # 三角形精确查询优先级最高
    h_tri = _surface_height_at_tri(x, z)
    if h_tri is not None:
        return h_tri
    # 出赛道面/路肩外 → 段投影外推，确保连续过渡
    return _sample_track_height(x, z)


# ============================================================
# 十一、视觉三角形精确查询（与 GPU 渲染完全同构）
# ============================================================

_TRI = {"grid": None, "xz": None, "y": None, "origin": None, "res": 1.5}

def build_triangle_index(tri_verts=None, force=False):
    """
    构建视觉赛道三角形的均匀网格索引。
    tri_verts: build_mountain_road() 的输出 (cnt, 11)，直接复用避免二次构建。
    一次性构建（~2-4s），按几何签名缓存到 .npy。
    """
    st = _TRI
    if st["grid"] is not None and not force:
        return
    if tri_verts is None:
        from mountain import build_mountain_road
        tri_verts = build_mountain_road()

    cache_p = _cache_path(f"tri_index_{_geometry_signature()}")
    if cache_p:
        _clean_stale_geometry_caches(
            "tri_index", os.path.splitext(os.path.basename(cache_p))[0])
    if not force and cache_p and os.path.exists(cache_p):
        try:
            d = np.load(cache_p, allow_pickle=True).item()
            st["xz"] = d["xz"]
            st["y"] = d["y"]
            st["origin"] = tuple(d["origin"])
            st["res"] = float(d["res"])
            st["grid"] = {tuple(map(int, k.split("_"))): v
                          for k, v in d["grid"].items()}
            print(f"[track_bounds] 三角形索引缓存加载: {len(st['xz'])} 三角形")
            return
        except Exception as e:
            print(f"[track_bounds] 三角形索引缓存损坏，重建: {e}")

    # 从顶点数组提取三角形 (x, y, z)，reshape 为 (T, 3, 3)
    tri = np.asarray(tri_verts, dtype=np.float32)[:, :3].reshape(-1, 3, 3)
    xz = tri[:, :, [0, 2]].copy()   # (T, 3, 2) — 仅存 xz 平面
    lo = xz.min(1)                   # (T, 2) — 每三角形包围盒左下
    hi = xz.max(1)                   # (T, 2) — 每三角形包围盒右上
    res = 1.5
    origin = lo.min(0) - res          # 网格原点（略偏移避免负格）
    g0 = np.floor((lo - origin) / res).astype(np.int64)
    g1 = np.floor((hi - origin) / res).astype(np.int64)

    # 构建网格哈希：每个网格格→覆盖它的三角形列表
    grid = {}
    for ti in range(len(tri)):
        for gz in range(int(g0[ti, 1]), int(g1[ti, 1]) + 1):
            for gx in range(int(g0[ti, 0]), int(g1[ti, 0]) + 1):
                grid.setdefault((gx, gz), []).append(ti)
    grid = {k: np.array(v, dtype=np.int32) for k, v in grid.items()}

    st["xz"] = xz
    st["y"] = tri[:, :, 1].copy()    # (T, 3) — Y 坐标（已含 ASPHALT_THICKNESS）
    st["origin"] = tuple(origin)
    st["res"] = res
    st["grid"] = grid
    print(f"[track_bounds] 三角形索引完成: {len(tri)} 三角形, {len(grid)} 格")

    if cache_p:
        try:
            np.save(cache_p, {
                "xz": xz,
                "y": st["y"],
                "origin": origin,
                "res": res,
                "grid": {f"{k[0]}_{k[1]}": v for k, v in grid.items()},
            })
        except Exception as e:
            print(f"[track_bounds] 三角形索引缓存失败: {e}")


def _surface_height_at_tri(x, z):
    """
    在视觉三角形网格上做重心插值（= GPU 渲染表面）。
    返回理论路面高度（已减 ASPHALT_THICKNESS，接口语义与旧版一致）。
    点不在任何赛道三角形上（出赛道面）返回 None。
    """
    st = _TRI
    if st["grid"] is None:
        build_triangle_index()
    res = st["res"]
    ox, oz = st["origin"]
    gx = int((x - ox) // res)
    gz = int((z - oz) // res)

    # 收集 3×3 邻域网格格中的候选三角形
    cands = []
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            c = st["grid"].get((gx + dx, gz + dz))
            if c is not None:
                cands.append(c)
    if not cands:
        return None
    cand = np.concatenate(cands)       # (K,) — 候选三角形索引

    t = st["xz"][cand]                 # (K, 3, 2)
    ax, az = t[:, 0, 0], t[:, 0, 1]
    bx, bz = t[:, 1, 0], t[:, 1, 1]
    cx, cz = t[:, 2, 0], t[:, 2, 1]
    det = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    l0 = ((bz - cz) * (x - cx) + (cx - bx) * (z - cz)) * inv
    l1 = ((cz - az) * (x - cx) + (ax - cx) * (z - cz)) * inv
    l2 = 1.0 - l0 - l1
    inside = ok & (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
    if not np.any(inside):
        return None
    # 取面积最大的包含三角形（多个候选时最稳定）
    scores = inside.astype(np.int32) * 10000 + np.where(ok, np.abs(det), 0.0)
    i = int(np.argmax(scores))
    lam = np.array([l0[i], l1[i], l2[i]])
    from mountain import ASPHALT_THICKNESS
    return float(np.dot(st["y"][cand[i]], lam)) - ASPHALT_THICKNESS


# ============================================================
# 十二、GPU Compute Shader 加速高度场查询
# ============================================================
# 将赛道三角形数据上传到 GPU SSBO，使用计算着色器批量查询高度。
# 适用于每帧多次调用的场景（车辆物理、轮胎痕迹等），
# 通过批量化减少 CPU-GPU 来回开销。

_GPU_INIT_DONE = False
_GPU_SSBO_TRI = None          # 三角形 mesh SSBO
_GPU_SSBO_QUERY = None        # 查询点 SSBO
_GPU_SSBO_RESULT = None       # 结果 SSBO
_GPU_PROG = None              # 计算着色器程序
_GPU_TRI_COUNT = 0            # 三角形数量

# Compute Shader 源码（与 main.py 中的 COMPUTE_HEIGHT_SRC 同构）
_COMPUTE_HEIGHT_SRC_TRACK = """
#version 430 core
layout(local_size_x = 64, local_size_y = 1) in;
layout(std430, binding=0) readonly buffer TriMesh {
    float tri_data[];
};
layout(std430, binding=1) buffer OutputHeights {
    float heights[256];
    float valid_flags[256];
};
layout(std430, binding=2) buffer QueryPoints {
    float queries[512];
};
uniform int uTriCount;
uniform int uQueryCount;

bool point_in_triangle(vec2 p, vec2 a, vec2 b, vec2 c, out vec3 lambda) {
    vec2 v0 = c - a;
    vec2 v1 = b - a;
    vec2 v2 = p - a;
    float dot00 = dot(v0, v0);
    float dot01 = dot(v0, v1);
    float dot02 = dot(v0, v2);
    float dot11 = dot(v1, v1);
    float dot12 = dot(v1, v2);
    float invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-12);
    float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
    lambda = vec3(1.0 - u - v, v, u);
    return (u >= -1e-5) && (v >= -1e-5) && (u + v <= 1.0 + 1e-5);
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= uQueryCount) return;
    float qx = queries[gid * 2];
    float qz = queries[gid * 2 + 1];
    vec2 q = vec2(qx, qz);
    float best_h = -999.0;
    float found = -1.0;
    for (int ti = 0; ti < uTriCount; ti++) {
        int base = ti * 9;
        vec2 a = vec2(tri_data[base + 0], tri_data[base + 1]);
        vec2 b = vec2(tri_data[base + 3], tri_data[base + 4]);
        vec2 c = vec2(tri_data[base + 6], tri_data[base + 7]);
        float ay = tri_data[base + 2];
        float by = tri_data[base + 5];
        float cy = tri_data[base + 8];
        vec3 lambda;
        if (point_in_triangle(q, a, b, c, lambda)) {
            best_h = ay * lambda.x + by * lambda.y + cy * lambda.z;
            found = 1.0;
            break;
        }
    }
    heights[gid] = best_h;
    valid_flags[gid] = found;
}
"""


def gpu_height_init(gl_context_ready=True):
    """
    初始化 GPU 高度查询系统。
    上传赛道三角形数据到 SSBO，编译计算着色器。
    gl_context_ready: 调用前确保 OpenGL 上下文已创建。
    如果 GPU 初始化失败，自动回退到 Python 实现。
    """
    global _GPU_INIT_DONE, _GPU_SSBO_TRI, _GPU_SSBO_QUERY
    global _GPU_SSBO_RESULT, _GPU_PROG, _GPU_TRI_COUNT

    if _GPU_INIT_DONE:
        return True

    if not gl_context_ready:
        print("[GPU高度] OpenGL 上下文未就绪，跳过 GPU 初始化")
        return False

    # 确保三角形索引已构建
    from mountain import ASPHALT_THICKNESS
    st = _TRI
    if st["grid"] is None or st["xz"] is None:
        print("[GPU高度] 三角形索引未构建，调用 build_triangle_index()")
        build_triangle_index()

    try:
        from OpenGL.GL import (
            glGenBuffers, glBindBuffer,
            glBufferData, glBindBufferBase, GL_SHADER_STORAGE_BUFFER,
            GL_STATIC_DRAW, GL_DYNAMIC_DRAW, GL_STREAM_READ,
            glCreateShader, glShaderSource, glCompileShader,
            glGetShaderiv, GL_COMPILE_STATUS, glGetShaderInfoLog,
            glCreateProgram, glAttachShader, glLinkProgram, glDeleteShader,
            glDeleteProgram,
            GL_COMPUTE_SHADER,
            glUseProgram, glUniform1i, glDispatchCompute,
            glMemoryBarrier, GL_ALL_BARRIER_BITS,
            glGetProgramiv, GL_LINK_STATUS, glGetProgramInfoLog,
            ctypes as gl_ctypes
        )
    except ImportError:
        print("[GPU高度] PyOpenGL 不可用，回退到 CPU 高度查询")
        return False

    # 构建三角形数据：每三角形 9 float (x0,z0,y0, x1,z1,y1, x2,z2,y2)
    xz_tri = st["xz"]
    y_tri = st["y"]
    T = xz_tri.shape[0]
    tri_data = np.zeros((T, 9), dtype=np.float32)
    tri_data[:, 0] = xz_tri[:, 0, 0]
    tri_data[:, 1] = xz_tri[:, 0, 1]
    tri_data[:, 2] = y_tri[:, 0]
    tri_data[:, 3] = xz_tri[:, 1, 0]
    tri_data[:, 4] = xz_tri[:, 1, 1]
    tri_data[:, 5] = y_tri[:, 1]
    tri_data[:, 6] = xz_tri[:, 2, 0]
    tri_data[:, 7] = xz_tri[:, 2, 1]
    tri_data[:, 8] = y_tri[:, 2]

    # 创建 SSBO 0: 三角形 mesh
    try:
        _GPU_SSBO_TRI = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, _GPU_SSBO_TRI)
        glBufferData(GL_SHADER_STORAGE_BUFFER, tri_data.nbytes, tri_data, GL_STATIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _GPU_SSBO_TRI)

        # SSBO 1: 结果 (高度 + 有效标志)
        MAX_Q = 256
        result_data = np.zeros(MAX_Q * 2, dtype=np.float32)
        _GPU_SSBO_RESULT = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, _GPU_SSBO_RESULT)
        glBufferData(GL_SHADER_STORAGE_BUFFER, result_data.nbytes, result_data, GL_STREAM_READ)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _GPU_SSBO_RESULT)

        # SSBO 2: 查询点 (x,z 对)
        query_data = np.zeros(MAX_Q * 2, dtype=np.float32)
        _GPU_SSBO_QUERY = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, _GPU_SSBO_QUERY)
        glBufferData(GL_SHADER_STORAGE_BUFFER, query_data.nbytes, query_data, GL_DYNAMIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _GPU_SSBO_QUERY)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    except Exception as e:
        print(f"[GPU高度] SSBO 分配失败: {e}，回退到 CPU 高度查询")
        return False

    # 编译计算着色器（传统编译流程，避免 glCreateShaderProgramv 的 ctypes 兼容问题）
    try:
        _GPU_PROG = glCreateProgram()
        shader = glCreateShader(GL_COMPUTE_SHADER)
        glShaderSource(shader, _COMPUTE_HEIGHT_SRC_TRACK)
        glCompileShader(shader)
        comp_status = gl_ctypes.c_int(0)
        glGetShaderiv(shader, GL_COMPILE_STATUS, comp_status)
        if comp_status.value != 1:
            log = glGetShaderInfoLog(shader).decode()
            print(f"[GPU高度] 计算着色器编译失败: {log[:200]}，回退到 CPU")
            glDeleteShader(shader)
            glDeleteProgram(_GPU_PROG)
            _GPU_PROG = None
            return False
        glAttachShader(_GPU_PROG, shader)
        glLinkProgram(_GPU_PROG)
        link_status = gl_ctypes.c_int(0)
        glGetProgramiv(_GPU_PROG, GL_LINK_STATUS, link_status)
        glDeleteShader(shader)
        if link_status.value != 1:
            log = glGetProgramInfoLog(_GPU_PROG).decode()
            print(f"[GPU高度] 计算着色器链接失败: {log[:200]}，回退到 CPU")
            glDeleteProgram(_GPU_PROG)
            _GPU_PROG = None
            return False
    except Exception as e:
        print(f"[GPU高度] 计算着色器异常: {e}，回退到 CPU")
        return False

    _GPU_TRI_COUNT = T
    _GPU_INIT_DONE = True
    print(f"[GPU高度] 初始化完成: {T} 三角形, {tri_data.nbytes/1e6:.1f}MB SSBO")
    return True


def gpu_height_query_batch(points):
    """
    批量 GPU 高度查询。
    points: list of (x, z) 或 numpy array (N, 2)
    返回: list of float，无效查询返回 None
    """
    global _GPU_INIT_DONE

    if not _GPU_INIT_DONE:
        # 回退到 CPU
        return [surface_height_at(x, z) for x, z in points]

    try:
        from OpenGL.GL import (
            glBindBuffer, glBufferSubData, glUseProgram,
            glUniform1i, glGetUniformLocation, glDispatchCompute, glMemoryBarrier,
            GL_SHADER_STORAGE_BUFFER, GL_ALL_BARRIER_BITS,
            glMapBufferRange, glUnmapBuffer, GL_MAP_READ_BIT,
            ctypes as gl_ctypes
        )
    except ImportError:
        return [surface_height_at(x, z) for x, z in points]

    pts = np.asarray(points, dtype=np.float32).ravel()
    n = len(points)
    if n == 0:
        return []

    # 上传查询点
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _GPU_SSBO_QUERY)
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, pts.nbytes, pts)

    # 调度
    glUseProgram(_GPU_PROG)
    glUniform1i(glGetUniformLocation(_GPU_PROG, "uTriCount"), _GPU_TRI_COUNT)
    glUniform1i(glGetUniformLocation(_GPU_PROG, "uQueryCount"), n)
    # 每组 64 线程，向上取整
    groups = (n + 63) // 64
    glDispatchCompute(groups, 1, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)
    glUseProgram(0)

    # 读取结果（修复：valid_flags 偏移 = 256*4 字节，不是 n*4）
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _GPU_SSBO_RESULT)
    # 读 heights[0:n]（偏移 0）
    ptr_h = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, n * 4, GL_MAP_READ_BIT)
    if ptr_h:
        raw_h = gl_ctypes.string_at(ptr_h, n * 4)
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        heights = np.frombuffer(raw_h, dtype=np.float32, count=n).copy()
    else:
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        return [surface_height_at(x, z) for x, z in points]
    # 读 valid_flags[0:n]（偏移 256*4 = 1024 字节，因为着色器里 float valid_flags[256]）
    ptr_v = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 256 * 4, n * 4, GL_MAP_READ_BIT)
    if ptr_v:
        raw_v = gl_ctypes.string_at(ptr_v, n * 4)
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        valid_flags = np.frombuffer(raw_v, dtype=np.float32, count=n).copy()
    else:
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        return [surface_height_at(x, z) for x, z in points]

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    from mountain import ASPHALT_THICKNESS
    out = []
    for i in range(n):
        if valid_flags[i] > 0.0:
            out.append(float(heights[i]) - ASPHALT_THICKNESS)
        else:
            # 出赛道面 → 回退到段投影
            x, z = points[i]
            out.append(_sample_track_height(x, z))
    return out


# ============================================================
# 公开 API（更新）
# ============================================================
__all__ = [
    "signed_distance",
    "is_safe",
    "nearest_track_info",
    "surface_height_at",
    "build_field",
    "query_field",
    "is_safe_field",
    "batch_safe_mask",
    "get_track_edge_points",
    "build_height_field",
    "build_triangle_index",
    "gpu_height_init",
    "gpu_height_query_batch",
]