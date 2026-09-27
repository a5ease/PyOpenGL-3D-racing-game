"""
动力滑行 — 榛名山赛道与地形生成器
===================================
核心设计（2026-09 重构 v2）：
1. 地形使用整块规则网格（Grid）+ signed_distance 精准压平赛道区域
   → 弯道内侧不再过生成，从根本上杜绝山体入侵赛道
2. 赛道使用独立 Catmull-Rom 样条 + 横截面 Profile，贴在地形上面
3. DEM 雕刻 + signed_distance 双重保险，确保赛道边缘平整
4. 输出格式完全兼容 main.py 的现有渲染管线
"""

import math
import random
import os
import re
import hashlib
import numpy as np
from OpenGL.GL import *
from scipy.ndimage import gaussian_filter, zoom as _ndizoom

# 赛道边界防入侵系统
from track_bounds import (is_safe, signed_distance, nearest_track_info,
                          surface_height_at, TRACK_EDGE_RATIO)

f32 = np.float32
_S = 1.0  # TRACK_SCALE

# 沥青厚度常量（视觉赛道抬高量，track_bounds 三角形查询需减回保持一致）
ASPHALT_THICKNESS = 0.15

# ============================================================
# 一、赛道路径控制点（从 1:1 真实 GPS 数据加载）
# ============================================================

def _load_1v1_control_points():
    """从 akina_1v1_track.json 加载真实 GPS 映射的控制点"""
    import json, os
    json_path = os.path.join(os.path.dirname(__file__), "akina_1v1_track.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cps = data.get("control_points", [])
            if cps and len(cps) > 10:
                xs = [c[0] for c in cps]
                zs = [c[1] for c in cps]
                print(f"[mountain] 已加载 1:1 真实赛道: {len(cps)} 控制点, "
                      f"x:[{min(xs):.1f},{max(xs):.1f}] z:[{min(zs):.1f},{max(zs):.1f}]")
                stats = data.get("stats", {})
                length = stats.get("length_m", 0)
                if length:
                    print(f"[mountain]   赛道总长: {length:.0f}m ({length/1000:.2f}km)")
                return cps
        except Exception as e:
            print(f"[mountain] 加载 1:1 赛道失败: {e}")
    return None


# ------------------------------------------------------------
# 修复 A（治根）：控制点均匀重采样 + 拉普拉斯平滑
# GPS 原始点在弯道处间距 2~5m、直道 20~50m，Catmull-Rom 在
# 密簇处过冲（曲线甩出真实轨迹打小圈）→ 道路网格扭动。
# 按弧长等距重采样消除不均匀性，拉普拉斯平滑抹掉亚米级噪声。
# ------------------------------------------------------------
def _preprocess_control_points(cps, spacing=10.0, smooth_iters=3, smooth_alpha=0.22):
    """GPS 控制点均匀化 + 平滑。
    1) 按弧长每 spacing 米重采样（消除"弯道密、直道疏"的不均匀，
       Catmull-Rom 的过冲随之消失）
    2) Laplacian 平滑 x/z（消除 GPS 亚米噪声导致的切线抖动）
    y/w/bank 线性插值携带。"""
    if len(cps) < 10:
        return cps
    xs = np.array([c[0] for c in cps], dtype=np.float64)
    zs = np.array([c[1] for c in cps], dtype=np.float64)
    ws = np.array([c[2] for c in cps], dtype=np.float64)
    bs = np.array([c[3] for c in cps], dtype=np.float64)

    # ---- 1. 弧长均匀重采样（线性即可，样条由 get_akina_path 再插） ----
    seg = np.hypot(np.diff(xs), np.diff(zs))
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = cum[-1]
    n_new = max(10, int(total / spacing))
    s_new = np.linspace(0.0, total, n_new)
    idx = np.clip(np.searchsorted(cum, s_new) - 1, 0, len(cum) - 2)
    t = (s_new - cum[idx]) / np.maximum(cum[idx + 1] - cum[idx], 1e-9)
    xs = xs[idx] + (xs[idx + 1] - xs[idx]) * t
    zs = zs[idx] + (zs[idx + 1] - zs[idx]) * t
    ws = ws[idx] + (ws[idx + 1] - ws[idx]) * t
    bs = bs[idx] + (bs[idx + 1] - bs[idx]) * t

    # ---- 2. Laplacian 平滑（首尾锚定，只动中间点） ----
    for _ in range(smooth_iters):
        xs[1:-1] += smooth_alpha * (0.5 * (xs[:-2] + xs[2:]) - xs[1:-1])
        zs[1:-1] += smooth_alpha * (0.5 * (zs[:-2] + zs[2:]) - zs[1:-1])

    print(f"[路径预处理] {len(cps)} → {n_new} 点 (间距{spacing}m, "
          f"平滑{smooth_iters}次), 总长{total:.0f}m")
    return [(float(xs[i]), float(zs[i]), float(ws[i]), float(bs[i]))
            for i in range(n_new)]


_1v1_cps = _load_1v1_control_points()

if _1v1_cps:
    CONTROL_POINTS = _preprocess_control_points(_1v1_cps)
else:
    # 内置手工赛道（仅在无 JSON 时回退）
    CONTROL_POINTS = [
        ( 3.0,  3.0, 7.0,  0),   ( 5.0,  0.5, 6.5,  2),
        ( 5.5, -3.0, 6.0,  4),   ( 4.0, -6.0, 5.5,  5),
        ( 1.0, -8.0, 5.5,  3),   (-2.0, -8.5, 5.5,  0),
        (-5.0, -7.0, 5.5, -3),   (-7.5, -4.0, 5.5, -6),
        (-8.0,  0.0, 5.5, -7),   (-6.5,  4.0, 5.5, -5),
        (-4.0,  6.5, 5.5,  0),   (-1.0,  8.0, 5.5,  3),
        ( 3.0,  8.0, 5.5,  5),   ( 6.5,  6.0, 5.5,  4),
        ( 9.0,  2.5, 5.5,  2),   (10.0, -2.0, 5.5,  0),
        ( 9.5, -6.0, 5.5, -2),   ( 7.0, -9.5, 5.5, -5),
        ( 3.0,-11.5, 5.5, -6),   (-1.5,-11.5, 5.5, -4),
        (-5.0, -9.5, 5.5,  0),   (-8.0, -6.5, 5.5,  3),
        (-10.0,-2.5, 5.5,  5),   (-10.5, 2.0, 6.0,  4),
        (-9.0,  6.5, 6.0,  0),   (-6.0, 10.0, 6.0, -3),
        (-2.0, 13.0, 6.0, -5),   ( 2.5, 14.5, 6.0, -6),
        ( 7.0, 14.0, 6.0, -4),   (11.0, 11.0, 6.0,  0),
        (14.0,  7.0, 6.5,  2),   (16.0,  2.0, 7.0,  3),
        (17.0, -3.5, 7.0,  0),   (16.5, -9.0, 7.5, -2),
        (14.0,-14.0, 8.0,  0),
    ]

# ============================================================
# 二、数学工具
# ============================================================

def smoothstep(t):
    return t * t * (3.0 - 2.0 * t)

def lerp(a, b, t):
    return a + (b - a) * t

def catmull_rom(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1) + (-p0 + p2) * t +
        (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
        (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )

def catmull_rom_derivative(p0, p1, p2, p3, t):
    t2 = t * t
    return 0.5 * (
        (-p0 + p2) +
        2.0 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t +
        3.0 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t2
    )


# ============================================================
# 三、DEM 真实地形集成
# ============================================================

_DEM_READY = False
_DEM_CARVED = False  # DEM 雕刻标志
_dem_available = False
try:
    from dem_loader import dem_get_height as _dem_get_height
    from dem_loader import dem_get_normal as _dem_get_normal
    from dem_loader import dem_get_slope as _dem_get_slope
    from dem_loader import init_dem as _init_dem
    from dem_loader import print_dem_info as _print_dem_info
    from dem_loader import carve_track_into_dem as _carve_dem
    from dem_loader import sample_game_height_vec as _dem_h_vec
    _dem_available = True
except Exception as e:
    print(f"[mountain] DEM 模块导入跳过: {e}")


def _ensure_dem_ready():
    """惰性初始化 DEM（首次调用时才会加载 SciPy 和 DEM 数据）"""
    global _DEM_READY
    if _DEM_READY or not _dem_available:
        return
    try:
        _init_dem()
        _print_dem_info()
        _DEM_READY = True
        print("[mountain] DEM 真实地形已激活（延迟加载）")
    except Exception as e:
        print(f"[mountain] DEM 初始化失败: {e}")


def get_terrain_height(x, z):
    """返回地形在 (x, z) 处的海拔高度"""
    _ensure_dem_ready()
    if _DEM_READY:
        return _dem_get_height(x, z)
    return 0.0  # 无 DEM 回退


def get_terrain_normal(x, z):
    """返回地形法线"""
    _ensure_dem_ready()
    if _DEM_READY:
        return _dem_get_normal(x, z)
    return (0.0, 1.0, 0.0)


def get_terrain_slope_angle(x, z):
    """返回坡度角（度）"""
    _ensure_dem_ready()
    if _DEM_READY:
        return _dem_get_slope(x, z)
    return 0.0


_terrain_cache = {}
_building_akina_path = False

# 缓存目录：用于缓存耗时几何体（杉树、地形）的 .npy 文件
_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")

# 地形缓存版本控制
# 升级地形算法时只需修改 _TERRAIN_CACHE_VERSION，旧缓存会被自动清理
_TERRAIN_CACHE_PREFIX = "akina_terrain"
_TERRAIN_CACHE_VERSION = "v35"  # v35：缓冲带 _surf 改用压平前的天然高（对齐标量版）

# 赛道横截面版本号（修改 PROFILE 时 +1，触发 track_bounds 几何缓存失效）
PROFILE_VERSION = "v3"  # v3：PROFILE 35→17 点精简

# 视觉赛道网格查询结构（用于地形压平与视觉赛道表面精确对齐）
_visual_track_lookup_cache = None
_mountain_road_mesh_cache = None

# path 完整缓存（segments=20）：(path, tangents, normals, binormals)
_akina_full_cache = None

def _clean_terrain_old_cache(keep_name=None):
    """
    自动清理地形系列的旧版本 .npy 缓存，防止旧版本堆积占用磁盘空间。
    keep_name: 如果指定，保留该名称的缓存文件不删除（通常是当前版本）。
    """
    if not os.path.exists(_CACHE_DIR):
        return
    prefix = _TERRAIN_CACHE_PREFIX
    removed = 0
    for fname in os.listdir(_CACHE_DIR):
        if fname.startswith(prefix) and (fname.endswith(".npy") or fname.endswith(".npz")):
            # 如果指定了 keep_name 且文件名匹配，跳过
            base = os.path.splitext(fname)[0]
            if keep_name and base == keep_name:
                continue
            try:
                os.remove(os.path.join(_CACHE_DIR, fname))
                print(f"[缓存] 删除旧地形缓存: {fname}")
                removed += 1
            except OSError as e:
                print(f"[缓存] 删除旧缓存失败 {fname}: {e}")
    if removed > 0:
        print(f"[缓存] 共清理 {removed} 个旧地形缓存文件")

def _get_cache_path(name):
    """获取缓存文件路径"""
    if not os.path.exists(_CACHE_DIR):
        try:
            os.makedirs(_CACHE_DIR)
        except OSError:
            return None
    return os.path.join(_CACHE_DIR, f"{name}.npy")

def _build_or_load(cache_name, build_fn, stale_base=None):
    """
    通用缓存器：有缓存 → 直接加载；无缓存 → 执行 build_fn 并保存
    build_fn 必须返回 numpy 数组

    stale_base: 保存新缓存成功后，清理同一基名下指纹不同的旧文件
                （环境构件带 path 指纹时用，防止旧路径缓存堆积）
    """
    cache_path = _get_cache_path(cache_name)
    if cache_path and os.path.exists(cache_path):
        print(f"[缓存] 加载 {cache_name}...")
        data = np.load(cache_path)
        print(f"[缓存]  {cache_name} 加载完成: {len(data)} 顶点")
        if stale_base:
            _clean_stale_env_caches(stale_base, cache_name)
        return data
    print(f"[缓存] 构建 {cache_name}（首次运行，下次将缓存）...")
    t0 = __import__('time').time()
    data = build_fn()
    elapsed = __import__('time').time() - t0
    if cache_path and len(data) > 0:
        try:
            np.save(cache_path, data)
            print(f"[缓存]  {cache_name} 已保存到缓存 ({elapsed:.1f}s)")
        except Exception as e:
            print(f"[缓存]  {cache_name} 保存失败: {e}")
    else:
        print(f"[缓存]  {cache_name} 构建完成 ({elapsed:.1f}s，未缓存)")
    if stale_base:
        _clean_stale_env_caches(stale_base, cache_name)
    return data

def clear_terrain_cache(clear_files=False):
    """清除地形成缓存
    clear_files=True 时同时删除地形系列 .npy 缓存文件。
    注意：只删除 akina_terrain_v*.npy 旧缓存，不会误删树木/纹理等其他缓存。
    """
    global _visual_track_lookup_cache, _mountain_road_mesh_cache
    _terrain_cache.clear()
    _visual_track_lookup_cache = None
    _mountain_road_mesh_cache = None
    if clear_files and os.path.exists(_CACHE_DIR):
        # 智能清理：只删除地形系列旧缓存，保留 akina_trees / 纹理 等其他缓存
        _clean_terrain_old_cache()
        print("[缓存] 地形缓存已清除")


# ============================================================
# 辅助函数
# ============================================================

def _clamp(v, lo, hi):
    """将 v 限制在 [lo, hi] 范围内"""
    return max(lo, min(v, hi))


# ============================================================
# path 指纹（缓存失效依据）
# ============================================================

def _path_fingerprint(path):
    """path 几何指纹（缓存失效依据）。

    v2（2026-09）：从"点数 + 宽度范围"的弱摘要升级为 (x, z, y, w) 全量 md5。
    旧版只对点数与 wmax 敏感 —— Laplacian 平滑系数、坡度裁剪只改坐标不改
    点数/宽度，指纹不变，环境缓存照样命中旧几何。现在任何 path 级改动
    （重采样间距、平滑系数、坡度上限、避车道长度/宽度）都会改指纹。

    取整到 0.01（厘米级）：<1cm 的浮点噪声通常不改指纹，但四舍五入的
    边界效应仍可能让它翻一次 —— 最坏结果是多重构建一次，无害。
    """
    if not path:
        return "empty"
    a = np.asarray(path, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 4:
        return "bad"
    q = np.round(a[:, :4], 2).astype(np.float32)
    return hashlib.md5(q.tobytes()).hexdigest()[:10]


def _env_cache_name(base, path=None):
    """环境构件缓存名 = 基名 + path 指纹。

    path 因重采样 / 避车道 / 平滑 / 坡度裁剪等任何调参而变
    → 所有 path 依赖的环境缓存自动失效重建。
    """
    if path is None:
        path = get_akina_path()
    return f"{base}_{_path_fingerprint(path)}"


def _cache_family(base):
    """从 "akina_guardrails_v4" 抽出构件家族前缀 "akina_guardrails"。

    没有 _vN 后缀时返回 base 自身（家族 = 自身）。
    家族前缀天然不会误伤兄弟构件：
      family("akina_signs_v1")   = "akina_signs"    ← 不匹配 akina_signs_em_v1_*
      family("akina_lamp_em_v1") = "akina_lamp_em"  ← 不匹配 akina_lamp_pool_v1_*
    """
    m = re.match(r"^(.*?)_v\d+$", base)
    return m.group(1) if m else base


def _clean_stale_env_caches(base, keep_name):
    """删除同一构件家族的过期缓存（旧版本号 + 旧 path 指纹）。

    命名代际：
      第一代  akina_guardrails_v1.npy / _v2 / _v3    版本号时代
      第二代  akina_guardrails_v4.npy                无指纹的固定名
      第三代  akina_guardrails_v4_<fingerprint>.npy  当前：版本号 + path 指纹

    所以必须按**家族**匹配 `^{family}_v\\d+`，只留 keep_name。
    只匹配 base 字面量是错的：_v4 的清理永远碰不到 _v3，旧版本号文件
    会在磁盘上永久堆积（实测遗留 ~110MB）。
    """
    if not os.path.exists(_CACHE_DIR):
        return
    family = _cache_family(base)
    pat = re.compile(r"^" + re.escape(family) + r"_v\d+")
    keep = {keep_name, keep_name + ".npy", keep_name + ".npz"}
    removed = []
    for fname in os.listdir(_CACHE_DIR):
        if fname in keep or not fname.endswith((".npy", ".npz")):
            continue
        if not pat.match(os.path.splitext(fname)[0]):
            continue
        try:
            os.remove(os.path.join(_CACHE_DIR, fname))
            removed.append(fname)
        except OSError:
            pass
    if removed:
        print(f"[缓存] 清理 {len(removed)} 个过期环境缓存 ({family}*)")


def _build_or_load_env(base, build_fn):
    """环境构件构建器统一入口：缓存名挂 path 指纹 + 自动清理旧指纹缓存。"""
    return _build_or_load(_env_cache_name(base, get_akina_path()),
                          build_fn, stale_base=base)


# ============================================================
# 带符号曲率分析（用于弯道标志牌/道钉）
# ============================================================

def _compute_path_curvature_signed(path):
    """逐点带符号曲率（1/m）：正 = 左转，负 = 右转。
    弯道外侧 = 转向反侧；normal=(-tz,tx) 指向行进方向左侧（护栏 side=1.0 同款约定）。"""
    n = len(path)
    curv = np.zeros(n, dtype=np.float64)
    for i in range(1, n - 1):
        p0, p1, p2 = path[i-1], path[i], path[i+1]
        v1x, v1z = p1[0]-p0[0], p1[1]-p0[1]
        v2x, v2z = p2[0]-p1[0], p2[1]-p1[1]
        l1 = math.hypot(v1x, v1z) or 1e-6
        l2 = math.hypot(v2x, v2z) or 1e-6
        cross = v1x * v2z - v1z * v2x
        dot = v1x * v2x + v1z * v2z
        ang = math.atan2(cross, dot)
        curv[i] = ang / max(0.5 * (l1 + l2), 1e-6)
    curv[0], curv[-1] = curv[1], curv[-2]
    return curv


def _outer_side_of(curv, peak_i):
    """外侧法线符号：左转(curv>0) → 外侧在右 → 法线负向。
    ★ 若实机发现放反了，把这个 return 的符号对调即可（一行修复）。"""
    return -1.0 if curv[peak_i] > 0.0 else 1.0


def _roadside_corner_clusters(curv):
    """扫描连续高曲率段 → [(入弯索引, 出弯索引, 峰值索引), ...]"""
    n = len(curv)
    clusters, i = [], 1
    while i < n - 1:
        if abs(curv[i]) > 0.012:
            j = i
            while j < n - 1 and abs(curv[j]) > 0.006:
                j += 1
            peak = max(range(i, j), key=lambda k: abs(curv[k]))
            clusters.append((i, j, peak))
            i = j + 15
        else:
            i += 1
    return clusters


def _scan_cut_segments(min_rise=None, step=2):
    """扫描两侧连续挖方段（格构护坡 / 落石防护网 / 落石标志共用判据）。

    判据与挡土墙一致（cut_foot_sample 的 rise），保证"有墙的坡才有护坡"。
    返回 [(side, [ (i, x, z, w, ox, oz), ... ]), ...]
    其中 (ox, oz) 已是**指向路外**的单位水平法线（= 法线 × side）。

    ⚠️ 实测这条赛道 rise 最大只有 ~1.26m（坡很缓，坡脚外 4m 才再抬 1.2m），
       阈值别按设计稿的 2~3m 设，否则一个都选不出来。
    ⚠️ 默认阈值用 None 延迟到调用时取 _CUT_MIN_RISE（该常量在本函数之后定义）。
    """
    if min_rise is None:
        min_rise = _CUT_MIN_RISE
    path, tangents = get_akina_path(return_tangents=True)[:2]
    n = len(path)
    out = []
    for side in (1.0, -1.0):
        active = []
        for i in range(0, n, step):
            x, z, y, w, bank = path[i]
            tx, ty, tz = tangents[i]
            nx2d, nz2d = -tz, tx
            L = math.hypot(nx2d, nz2d) or 1e-6
            nx2d, nz2d = nx2d / L, nz2d / L
            _, _, rise = cut_foot_sample(x, z, nx2d, nz2d, w, side)
            if rise > min_rise:
                active.append((i, x, z, w, nx2d * side, nz2d * side))
            else:
                if len(active) >= 3:
                    out.append((side, active))
                active = []
        if len(active) >= 3:
            out.append((side, active))
    return out


def _slope_point_xz(rec, run):
    """_slope_point 的水平分量（供发夹弯错腿检测用）。"""
    _i, x, z, w, ox, oz = rec
    lat = cut_foot_offset(w) + run
    return (x + ox * lat, z + oz * lat)


def _slope_point(rec, run, lift=0.0):
    """挖方边坡上"坡脚再往外 run 米"处的地表点（+lift 抬高）。"""
    _i, x, z, w, ox, oz = rec
    lat = cut_foot_offset(w) + run
    px = x + ox * lat
    pz = z + oz * lat
    return (px, get_ground_height(px, pz) + lift, pz)


# ============================================================
# 视觉赛道网格查询结构（用于地形压平与视觉赛道表面精确对齐）
# ============================================================

def _build_visual_track_lookup():
    """
    构建视觉赛道三角网格的 (x, z) → 精确高度 空间哈希查询结构。

    关键：直接使用 build_mountain_road() 产出的视觉三角形，
    保证地形压平高度与视觉赛道表面完全一致，
    彻底消除弯道、banking 变化处、路缘石区域的地形穿模。

    副作用：会触发 build_mountain_road 内部的 DEM 雕刻，
    使后续 get_terrain_height 采样到已雕刻的 DEM。

    返回：(road_verts, cell_map, cell_size)
    """
    global _visual_track_lookup_cache
    if _visual_track_lookup_cache is not None:
        return _visual_track_lookup_cache

    print("[地形] 构建视觉赛道查询结构（与视觉网格完全对齐）...")
    road_verts = build_mountain_road()
    n_tris = len(road_verts) // 3
    if n_tris == 0:
        _visual_track_lookup_cache = (road_verts, {}, 4.0)
        return _visual_track_lookup_cache

    # 空间哈希：每个三角形按其 XZ 包围盒落入若干网格单元
    CELL = 4.0
    cell_map = {}
    for ti in range(n_tris):
        b = ti * 3
        x0 = road_verts[b,     0]; z0 = road_verts[b,     2]
        x1 = road_verts[b + 1, 0]; z1 = road_verts[b + 1, 2]
        x2 = road_verts[b + 2, 0]; z2 = road_verts[b + 2, 2]

        xmin = x0 if x0 < x1 else x1
        if x2 < xmin: xmin = x2
        xmax = x0 if x0 > x1 else x1
        if x2 > xmax: xmax = x2
        zmin = z0 if z0 < z1 else z1
        if z2 < zmin: zmin = z2
        zmax = z0 if z0 > z1 else z1
        if z2 > zmax: zmax = z2

        cx0 = int(math.floor(xmin / CELL)); cx1 = int(math.floor(xmax / CELL))
        cz0 = int(math.floor(zmin / CELL)); cz1 = int(math.floor(zmax / CELL))
        for cx in range(cx0, cx1 + 1):
            for cz in range(cz0, cz1 + 1):
                key = (cx, cz)
                lst = cell_map.get(key)
                if lst is None:
                    cell_map[key] = [ti]
                else:
                    lst.append(ti)

    print(f"[地形]   {n_tris} 视觉三角形 → {len(cell_map)} 空间单元 (CELL={CELL})")
    _visual_track_lookup_cache = (road_verts, cell_map, CELL)
    return _visual_track_lookup_cache


def _query_visual_track_height(road_verts, cell_map, CELL, wx, wz):
    """
    在 (wx, wz) 处查询视觉赛道的精确重心插值高度。
    未落入任何三角形内返回 None。
    """
    cx = int(math.floor(wx / CELL))
    cz = int(math.floor(wz / CELL))
    tris = cell_map.get((cx, cz))
    if not tris:
        return None

    for ti in tris:
        b = ti * 3
        x0 = road_verts[b,     0]; y0 = road_verts[b,     1]; z0 = road_verts[b,     2]
        x1 = road_verts[b + 1, 0]; y1 = road_verts[b + 1, 1]; z1 = road_verts[b + 1, 2]
        x2 = road_verts[b + 2, 0]; y2 = road_verts[b + 2, 1]; z2 = road_verts[b + 2, 2]

        # XZ 平面上的 2D 重心坐标
        v0x = x1 - x0; v0z = z1 - z0
        v1x = x2 - x0; v1z = z2 - z0
        v2x = wx - x0; v2z = wz - z0

        denom = v0x * v1z - v1x * v0z
        if -1e-12 < denom < 1e-12:
            continue
        inv = 1.0 / denom
        u = (v2x * v1z - v1x * v2z) * inv
        v = (v0x * v2z - v2x * v0z) * inv

        if u >= -1e-4 and v >= -1e-4 and (u + v) <= 1.0 + 1e-4:
            return y0 + u * (y1 - y0) + v * (y2 - y0)

    return None


# ============================================================
# 四、赛道路径生成
# ============================================================

_akina_path_cache = None

def _get_path_height(x, z):
    """赛道高度 = DEM 地形高度 + 微抬高防 Z-fighting"""
    _ensure_dem_ready()
    if _DEM_READY:
        return _dem_get_height(x, z) + 0.25
    return 0.25


def _limit_path_slope(path, max_slope=0.08):
    """
    坡度裁剪：限制相邻路径点之间的坡度，消除 DEM 中过陡的上下坡路段。

    算法：双向传播约束
    - 正向传播：从起点到终点，限制上坡幅度不超过 max_slope
    - 反向传播：从终点到起点，限制下坡幅度不超过 max_slope
    - 整体偏移修正：保持裁剪前后的平均高度一致

    max_slope = 0.08 表示 8% 坡度（约 4.6°），适合山路驾驶。
    返回：新的 path 列表，保持与原格式一致 [(x, z, y, w, bank)]
    """
    n = len(path)
    if n < 2:
        return path

    # 提取高度和水平距离
    y = np.array([p[2] for p in path], dtype=np.float64)
    dists = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        dx = path[i][0] - path[i-1][0]
        dz = path[i][1] - path[i-1][1]
        dists[i] = math.hypot(dx, dz)

    y_original = y.copy()

    # ---- 正向传播：限制上坡 ----
    for i in range(1, n):
        max_dy = dists[i] * max_slope
        dy = y[i] - y[i-1]
        if dy > max_dy:
            y[i] = y[i-1] + max_dy
        elif dy < -max_dy:
            y[i] = y[i-1] - max_dy

    # ---- 反向传播：限制下坡 ----
    for i in range(n - 2, -1, -1):
        max_dy = dists[i + 1] * max_slope
        dy = y[i + 1] - y[i]
        if dy > max_dy:
            y[i] = y[i + 1] - max_dy
        elif dy < -max_dy:
            y[i] = y[i + 1] + max_dy

    # ---- 重复一次正向 + 反向，让约束充分收敛 ----
    for i in range(1, n):
        max_dy = dists[i] * max_slope
        dy = y[i] - y[i-1]
        if dy > max_dy:
            y[i] = y[i-1] + max_dy
        elif dy < -max_dy:
            y[i] = y[i-1] - max_dy
    for i in range(n - 2, -1, -1):
        max_dy = dists[i + 1] * max_slope
        dy = y[i + 1] - y[i]
        if dy > max_dy:
            y[i] = y[i + 1] - max_dy
        elif dy < -max_dy:
            y[i] = y[i + 1] + max_dy

    # ---- 整体偏移修正：保持平均高度不变 ----
    offset = np.mean(y_original - y)
    y += offset

    # 重新组装 path
    new_path = []
    for i in range(n):
        new_path.append((path[i][0], path[i][1], float(y[i]),
                         path[i][3], path[i][4]))

    return new_path


# ============================================================
# 路侧避车道（直道段局部加宽）
# ============================================================

def _add_pullout_bays(path, min_gap=120.0, bay_len=28.0, taper=14.0,
                      bay_w=3.2, max_curv=0.006):
    """直道段局部加宽的避车道。
    通过修改 path 的 w 分量实现，视觉赛道/地形压平/护栏/物理边界
    全部自动继承（它们的半宽都取 w*0.5）。
    """
    n = len(path)
    if n < 10:
        return path
    xs = np.array([p[0] for p in path]); zs = np.array([p[1] for p in path])
    # 弧长与航向
    dx = np.diff(xs); dz = np.diff(zs)
    seg = np.hypot(dx, dz)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    heading = np.arctan2(dx, dz)                      # (n-1,)
    dhead = np.abs((np.diff(heading) + np.pi) % (2*np.pi) - np.pi)
    # 每点的曲率近似（航向变化 / 弧长），窗口 3 点平滑
    curv = np.zeros(n)
    curv[1:-1] = dhead / np.maximum(seg[:-1] + seg[1:], 1e-6)
    curv = gaussian_filter(curv, 3.0)

    w = np.array([p[3] for p in path], dtype=np.float64)
    bay_w_arr = np.zeros(n)
    last_end = -1e9
    i = 0
    while i < n - 1:
        # 找一段足够长、足够直的直道
        j = i
        while j < n - 1 and curv[j] < max_curv:
            j += 1
        span = cum[min(j, n-1)] - cum[i]
        center = (cum[i] + cum[min(j, n-1)]) * 0.5
        if span >= bay_len + 2*taper and center - last_end >= min_gap:
            c0, c1 = center - bay_len/2, center + bay_len/2
            for k in range(n):
                s = cum[k]
                if c0 - taper <= s <= c1 + taper:
                    if s < c0:
                        t = (s - (c0 - taper)) / taper
                    elif s > c1:
                        t = (c1 + taper - s) / taper
                    else:
                        t = 1.0
                    bay_w_arr[k] = bay_w * smoothstep(max(0.0, min(1.0, t)))
            last_end = center
        i = max(j, i + 10)   # 至少跳 10 点防止在弯道反复试探

    total = w + bay_w_arr
    if np.max(bay_w_arr) <= 0.01:
        n_pts = int(np.sum(bay_w_arr > 0.01))
        print(f"[路径] 避车道: {n_pts} 点加宽（max_curv={max_curv}, min_gap={min_gap}），未找到足够直的直道段")
        return path
    n_pts = int(np.sum(bay_w_arr > 0.01))
    print(f"[路径] 避车道: {n_pts} 点加宽, 峰值 +{np.max(bay_w_arr):.1f}m")
    return [(path[k][0], path[k][1], path[k][2], float(total[k]), path[k][4])
            for k in range(n)]


# ============================================================
# 路径端部延长：起点/终点各直线延长 ~26m 的沥青赛道
# ============================================================

def _extend_track_ends(path, tangents, ext_len=26.0):
    """路径两端沿端部切向各直线延长 ext_len 米（延续端部纵坡）。

    ★ 这是"起点/终点场地与赛道衔接"的根本：场地（_venue_pads）、
      建筑、龙门架全部锚在路径端点上 —— 延长后赛道**开进**场地走廊，
      路在建筑群中间结束，而不是"路旁边贴一块地"。
    一切下游（视觉路面/地形压平/护栏/空气墙/小地图）都吃同一个 path，
    指纹一变全部自动重建，无需单独适配。
    """
    if ext_len <= 0.5 or len(path) < 20:
        return path, tangents
    for end in (0, -1):
        # 端部切向（水平归一）
        tx, ty, tz = tangents[end]
        Lt = math.hypot(tx, tz) or 1e-6
        tx, tz = tx / Lt, tz / Lt
        # 端部纵坡（限制在 8% 内，与 _limit_path_slope 同约束）
        i2 = -2 if end == -1 else 1
        p_end = path[end]
        dl = math.hypot(p_end[0] - path[i2][0], p_end[1] - path[i2][1]) or 1e-6
        slope = max(-0.08, min(0.08,
                    (p_end[2] - path[i2][2]) / dl))
        # 点距：端部 10 段的中位数
        rng = range(len(path) - 10, len(path)) if end == -1 else range(1, 11)
        segs = sorted(math.hypot(path[k][0] - path[k - 1][0],
                                 path[k][1] - path[k - 1][1]) for k in rng)
        step = segs[len(segs) // 2] or 0.5
        n_add = max(4, int(round(ext_len / step)))
        sgn = 1.0 if end == -1 else -1.0     # 离开路径的延长方向
        tl = math.sqrt(1.0 + slope * slope)  # 3D 切向归一
        add_p, add_t = [], []
        for k in range(1, n_add + 1):
            s = k * step
            add_p.append((p_end[0] + sgn * tx * s,
                          p_end[1] + sgn * tz * s,
                          p_end[2] + sgn * slope * s,
                          p_end[3], p_end[4]))
            # 前向切向恒为 (+tx, slope, +tz)（两端通用：指向索引增大方向）
            add_t.append((tx / tl, slope / tl, tz / tl))
        if end == -1:
            path, tangents = path + add_p, tangents + add_t
        else:
            path, tangents = add_p[::-1] + path, add_t[::-1] + tangents
    print(f"[路径] 两端各延长 ~{ext_len:.0f}m（起点/终点场地走廊）")
    return path, tangents


# ============================================================
# 地形表面高度查询（贴地放置专用）
# ============================================================
# ⚠️ 这些常量必须与 _build_akina_terrain_impl 保持一致
_TERRAIN_FLAT_END  = 5.0     # 平整带宽度：碎石路肩 + 排水沟（原 3.0 → 5.0）
_TERRAIN_CUT_SLOPE  = 0.75   # 上边坡（挖方）：每横向 1m 最多抬 0.75m ≈ 37°
_TERRAIN_FILL_SLOPE = 1.10   # 下边坡（填方路堤）：略缓于挖方
_TERRAIN_TRACK_PAD = 0.10   # 地形压平面低于视觉赛道的量

# 挖方坡脚采样（挡土墙 / 树木避让共用）
# ⚠️ 三种"半宽"别混（详见 track_bounds.TRACK_EDGE_RATIO）：
#      0.35w = 沥青路缘 = signed_distance 的 0 点
#      0.50w = 赛道棱柱外缘（视觉 PROFILE 到 ±5.0）
#      sd(点) = 距中心线距离 - 0.35w
#    地形压平带从路缘一路铺到 sd = _TERRAIN_FLAT_END。
#    历史 bug：旧代码把 `_TERRAIN_FLAT_END + 1.5` 当成**距中心线**的偏移
#    （= 路缘外 1.5m，sd≈1.5），整个探针埋在压平带里 → rise 恒为 0；
#    FLAT_END 由 3.0 调到 5.0 之后挡土墙判据永不成立，全工程 0 顶点，
#    树木的挡墙避让也一样失效。
_CUT_FOOT_SD = _TERRAIN_FLAT_END + 2.5   # 坡脚 sd：压平带外 2.5m，正好落在挖方坡脚
_CUT_MIN_RISE = 0.8         # 相对路肩抬升超过此值才算挖方段（砌墙 / 树避让）


# ============================================================
# 路肩 U 型混凝土侧沟（日本「上ぶた式 U 形側溝」简化几何）
# ============================================================
# 横向坐标 u：从沟内沿（0.38w，避让 0.36~0.375w 的红白路缘石）向外量，单位米。
#   0.00～0.08  内侧唇（贴路缘石）
#   0.08～0.50  沟腔（宽 0.42m × 深 0.30m）← 榛名山"沟渠跑法"的原型
#   0.50～0.58  外侧唇
#   0.58～0.80  外缓坡（衔接地形）
# ★ 沟腔 (u 0.08~0.50) 换算到 signed_distance 约 0.29~0.71m，落在空气墙
#   （sd ≤ 0.9）之内 → 保留"沟渠跑法"的可玩性。
#
# ⚠️⚠️ 为什么侧沟是"高出地面"而不是"下凹"：
#   地形压平带（plateau）紧贴路面下方 0.10m 且是**连续**的，任何比路面深
#   超过 0.10m 的下凹都会被地形填平（最终只剩两条 10cm 高的混凝土条）。
#   真开槽需要地形核心带 ~0.4m 网格，但这条 1:1 赛道 AABB 达 1701×2021m，
#   _build_akina_terrain_impl 会把核心带自动降级到 **3.0m**，0.8m 宽的
#   高斯开槽只会被采样成沿途随机散坑 —— 实测不可行，已废弃（v25 → v26）。
#   现方案：沟唇略高于路面、沟腔下凹但仍落在地形之上：
#     沟唇顶 = plateau + 0.20（高出路肩 0.10m，与红白路缘石齐平）
#     沟腔底 = plateau + 0.04（低于路面 0.06m，仍在地形之上 → 不会被填平）
#   可见沟腔 0.16m 深 × 0.42m 宽，外侧缓坡把沟体外沿埋进地形。
_GUTTER_START_RATIO = 0.38    # 沟内沿 / w（=0.38w）
_GUTTER_U_LIP0      = 0.00
_GUTTER_U_CH0       = 0.08    # 沟腔内壁
_GUTTER_U_CH1       = 0.50    # 沟腔外壁
_GUTTER_U_LIP1      = 0.58
_GUTTER_U_END       = 0.80
_GUTTER_LIFT        = 0.22    # 沟唇顶相对地形压平面的高度
_GUTTER_DEPTH       = 0.14    # 沟唇顶 → 沟腔底（沟腔底须比地形高 ≥0.06）
_GUTTER_APRON       = 0.28    # 沟唇顶 → 外缓坡末端（末端埋入地形 0.06m）
_GUTTER_BODY        = 0.40    # 沟体埋入段高度
_GUTTER_BASIN_EVERY = 40      # 每 N 个沟段设一处集水桝（下沉 + 钢格栅盖板）
# ★ 出现规则（v6）：侧沟只在"较急弯道的内侧"铺，直道 / 弯道外侧一律不铺。
#   判据用带符号曲率，不用 bank（实测 banking 只有 ±0.6°）。
#   入场 / 出场双阈值做滞回，否则曲率在阈值附近抖 → 一堆 2~4m 的碎片段。
_GUTTER_CURV_ENTER = 0.018    # 入场 ≈ R < 55m
_GUTTER_CURV_EXIT  = 0.010    # 出场 ≈ R < 100m（滞回）
_GUTTER_MIN_RUN    = 24.0     # 连续段最短里程（m），短于此的弯不铺
_GUTTER_STEP       = 3        # 每 N 个路径点取一个横截面站（≈1.5m）


def gutter_band(w=7.0):
    """沟腔在 signed_distance 坐标下的 (内壁, 外壁)。

    空气墙在 sd ≤ 0.9，沟腔 (0.29~0.71) 落在其中 → 沟渠跑法可玩。
    """
    base = (_GUTTER_START_RATIO - TRACK_EDGE_RATIO) * w
    return base + _GUTTER_U_CH0, base + _GUTTER_U_CH1


def gutter_hold_depth(wheel_sd, w=7.0):
    """外轮陷入沟腔的程度 0~1（0 = 未接触，1 = 正在沟腔中心）。

    超出外壁的部分作为 penetration 一并返回，供物理做硬约束
    （沟外壁把车挡住，不让外轮翻出沟外）。
    """
    lo, hi = gutter_band(w)
    if wheel_sd < lo - 0.10 or wheel_sd > hi + 0.30:
        return 0.0, 0.0
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) + 0.10
    depth = 1.0 - min(1.0, abs(wheel_sd - mid) / half)
    pen = max(0.0, wheel_sd - hi)
    return max(0.0, depth), pen


def gutter_dip(wheel_sd, w=7.0):
    """车辆外轮压进侧沟时的车身下沉量（米）。

    wheel_sd = 外侧车轮的 signed_distance。沟腔区间两侧各放宽 0.15m 做过渡，
    峰值 -0.10m（沟腔底低于路面 0.06m + 悬挂压缩）。
    车在路中时 wheel_sd 远小于下界 → 恒为 0。
    """
    c0 = (_GUTTER_START_RATIO - TRACK_EDGE_RATIO) * w + _GUTTER_U_CH0 - 0.15
    c1 = (_GUTTER_START_RATIO - TRACK_EDGE_RATIO) * w + _GUTTER_U_CH1 + 0.15
    if wheel_sd < c0 or wheel_sd > c1:
        return 0.0
    return -0.10 * math.sin(math.pi * (wheel_sd - c0) / (c1 - c0))


def cut_foot_offset(w):
    """挖方坡脚距中心线的横向偏移（米）＝ 0.35w + _CUT_FOOT_SD。"""
    return TRACK_EDGE_RATIO * w + _CUT_FOOT_SD


def cut_foot_sample(x, z, nx2d, nz2d, w, side):
    """探测该侧是否为挖方段。

    返回 (坡脚 x, 坡脚 z, 坡脚相对路肩的抬升 rise)。
    rise > _CUT_MIN_RISE → 这里有一道挖方边坡，坡脚需要挡土墙、树木要避让。
    """
    hw = w * 0.5
    edge_h = get_ground_height(x + nx2d * hw * side, z + nz2d * hw * side)
    off = cut_foot_offset(w)
    fx = x + nx2d * off * side
    fz = z + nz2d * off * side
    return fx, fz, get_ground_height(fx, fz) - edge_h


def get_ground_height(x, z):
    """
    返回渲染地形网格在 (x, z) 处的实际表面高度。

    与 _build_akina_terrain_impl 的高度计算逻辑完全同构：
      - sd ≤ FLAT_END  → 压平到赛道表面 (vis_h - TRACK_PAD)
      - sd > FLAT_END  → 以限定坡比向 DEM 爬升/下降（挖方/填方）

    树木 / 路灯 / 路障等物体贴地放置时必须调用此函数，
    而不是直接调 get_terrain_height()。
    """
    _ensure_dem_ready()
    _, _hw_b, _, _, _lat_b, _ = nearest_track_info(x, z)
    sd = abs(_lat_b) - _hw_b
    _ASPHALT = ASPHALT_THICKNESS
    TRACK_PAD  = _TERRAIN_TRACK_PAD
    FLAT_END   = _TERRAIN_FLAT_END
    CUT_SLOPE  = _TERRAIN_CUT_SLOPE
    FILL_SLOPE = _TERRAIN_FILL_SLOPE

    # ---- 路肩基准高 ----
    road_verts, track_cells, track_cell_size = _build_visual_track_lookup()
    vis_h = _query_visual_track_height(
        road_verts, track_cells, track_cell_size, x, z)
    if vis_h is not None:
        road_h = vis_h - TRACK_PAD
    else:
        road_h = surface_height_at(x, z) + _ASPHALT - TRACK_PAD

    # ---- 平整带内：直接压平 ----
    #   ⚠️ 起终点服务带压平必须**最后**做：带子内边缘在 sd=2.2 处（路缘外
    #      2.2m），属于 sd ≤ FLAT_END 的路肩范围，早退会让它漏压 → 带子里
    #      出现 0.2m 台阶。朝路面只用 2m 过渡 ⇒ sd<0.2 时权重已归 1，
    #      绝不会把沥青面顶起来（路面只有 0.15m 厚）。
    if sd <= FLAT_END:
        return _venue_flatten_h(x, z, road_h, sd)

    # ---- 坡比受限的边坡 ----
    d = sd - FLAT_END
    dem_h = get_terrain_height(x, z)
    dy = dem_h - road_h
    # clip 上下界
    lo = -FILL_SLOPE * d
    hi =  CUT_SLOPE  * d
    embank = road_h + max(lo, min(hi, dy))

    # 排水沟（坡脚集水浅槽）
    ditch = -0.35 * math.exp(-((sd - (FLAT_END + 0.8)) / 0.7) ** 2)
    # 场地压平（与地形构建器同构；sd≤FLAT_END 分支在保护带内，无需处理）
    return _venue_flatten_h(x, z, embank + ditch, sd)


def get_akina_path(segments=20, return_tangents=False):
    """
    生成路径点 [(x, z, y, road_width, bank_rad)]

    高度策略：在控制点处锚定DEM高度，然后Catmull-Rom
    对X/Z/Y/W/B五分量统一插值。
    插值完成后进行坡度裁剪（max_slope=0.08），
    确保赛道坡度不超过 8%。

    return_tangents=True 时返回 (path, tangents, normals, binormals)
    其中 normals/binormals 使用 Rotation Minimizing Frame（双反射法），
    解决弯道处 Frenet 标架翻转导致的赛道扭曲问题。
    """
    global _akina_path_cache, _akina_full_cache, _building_akina_path

    # segments=20 时 path/tangents/normals/binormals 全部缓存。
    # 旧版只缓存 path，导致 return_tangents=True 的 10 处调用各重建一次
    # 15601 点路径（每次 ~1s），还让各建造器拿到不同的列表对象。
    if segments == 20:
        if return_tangents:
            if _akina_full_cache is not None:
                return _akina_full_cache
        elif _akina_path_cache is not None:
            return _akina_path_cache

    if _building_akina_path:
        return [] if not return_tangents else ([], [], [], [])

    _building_akina_path = True
    try:
        n = len(CONTROL_POINTS)

        # 给每个控制点补充DEM高度，构建5分量锚点 (x, z, y, w, b)
        pts = []
        for p in CONTROL_POINTS:
            x, z = p[0] * _S, p[1] * _S
            y = _get_path_height(x, z)
            pts.append([x, z, y, p[2], p[3]])
        pts = np.array(pts, dtype=f32)

        path = []
        tangents = []
        for i in range(n - 1):
            p0 = pts[max(0, i - 1)]
            p1 = pts[i]
            p2 = pts[min(n - 1, i + 1)]
            p3 = pts[min(n - 1, i + 2)]
            for s in range(segments):
                t = s / segments
                ix, iz, iy, iw, ib = catmull_rom(p0, p1, p2, p3, t)
                path.append((float(ix), float(iz), float(iy),
                             float(iw), math.radians(float(ib))))
                dx, dz, dy, dw, db = catmull_rom_derivative(p0, p1, p2, p3, t)
                d_len = math.sqrt(dx*dx + dy*dy + dz*dz)
                if d_len > 1e-10:
                    tangents.append((dx/d_len, dy/d_len, dz/d_len))
                else:
                    tangents.append((1.0, 0.0, 0.0))

        last = pts[-1]
        path.append((float(last[0]), float(last[1]), float(last[2]),
                     float(last[3]), math.radians(float(last[4]))))
        tangents.append(tangents[-1] if tangents else (1.0, 0.0, 0.0))

        # ---- 坡度裁剪：限制相邻路径点之间的坡度不超过 8% ----
        path = _limit_path_slope(path, max_slope=0.08)

        # ---- 路侧避车道：直道段局部加宽 ----
        path = _add_pullout_bays(path)

        # ---- ★ 起终点场地走廊：路径两端各直线延长 ~26m ----
        path, tangents = _extend_track_ends(path, tangents, ext_len=26.0)

        # 计算 Rotation Minimizing Frame（双反射法）
        normals, binormals = _compute_rmf_frames(tangents)

        if segments == 20:
            if _akina_path_cache is None:
                _akina_path_cache = path
            _akina_full_cache = (path, tangents, normals, binormals)

        if return_tangents:
            return path, tangents, normals, binormals
        return path
    finally:
        _building_akina_path = False


def _compute_rmf_frames(tangents):
    """
    Rotation Minimizing Frame（双反射法）

    核心思想：将上一帧的 N/B 向量沿切线方向"平行运输"到当前点，
    保证标架旋转量最小，消除弯道处 Frenet 标架的突然翻转。

    参考文献：Wang et al. "Computation of Rotation Minimizing Frames"
    """
    n = len(tangents)
    if n == 0:
        return [], []

    normals = []
    binormals = []

    # ---- 初始标架 ----
    T0 = np.array(tangents[0], dtype=f32)
    UP = np.array([0.0, 1.0, 0.0], dtype=f32)
    # 如果切线几乎垂直，换参考方向
    if abs(np.dot(T0, UP)) > 0.999:
        UP = np.array([1.0, 0.0, 0.0], dtype=f32)
    B0 = np.cross(UP, T0)
    b_len = np.linalg.norm(B0)
    if b_len < 1e-10:
        B0 = np.array([0.0, 0.0, 1.0], dtype=f32)
    else:
        B0 = B0 / b_len
    N0 = np.cross(T0, B0)
    normals.append(N0)
    binormals.append(B0)

    # ---- 逐点平行运输 ----
    for i in range(1, n):
        Ti_prev = np.array(tangents[i - 1], dtype=f32)
        Ti_curr = np.array(tangents[i], dtype=f32)
        Ni_prev = normals[-1]
        Bi_prev = binormals[-1]

        # 双反射：反射向量 v = (T_{i-1} + T_i) 归一化
        v = Ti_prev + Ti_curr
        v_len = np.linalg.norm(v)
        if v_len < 1e-10:
            # 切线没变，直接沿用上一帧
            normals.append(Ni_prev)
            binormals.append(Bi_prev)
            continue
        v = v / v_len

        # 第一次反射（对 T_{i-1} 和 T_i 的平分面）
        N_ref = Ni_prev - 2.0 * np.dot(Ni_prev, v) * v
        B_ref = Bi_prev - 2.0 * np.dot(Bi_prev, v) * v

        # 第二次反射（对当前切线 T_i）
        Ni = N_ref - 2.0 * np.dot(N_ref, Ti_curr) * Ti_curr
        Bi = B_ref - 2.0 * np.dot(B_ref, Ti_curr) * Ti_curr

        # 符号跟踪：防止 B 在 U 型弯处翻转 180°
        # 原理：检查新 B 与上一帧 B 的点积，负值 = 翻转了，取反修复
        if np.dot(Bi, Bi_prev) < 0.0:
            Bi = -Bi
            Ni = -Ni

        # 重新归一化（防止浮点误差累积）
        Ni = Ni / max(np.linalg.norm(Ni), 1e-10)
        Bi = Bi / max(np.linalg.norm(Bi), 1e-10)

        normals.append(Ni)
        binormals.append(Bi)

    return normals, binormals


# ============================================================
# 六、地形颜色生成（基于 DEM 高度 + 坡度）
# ============================================================

def _terrain_color(h, slope_deg=0.0):
    """
    根据海拔高度和坡度返回自然地形颜色。
    颜色基于日本的榛名山真实植被分布特征：
    - 低海拔 (<180): 常绿阔叶林/农田 → 明亮的绿色
    - 中山腰 (180~350): 混合林（杉树 + 阔叶）→ 中绿色
    - 山腰高 (350~500): 杉树林为主 → 深绿色
    - 高海拔 (>500): 灌木+岩石 → 灰绿色/棕灰色
    - 陡坡 (>25°): 岩石/裸露土坡 → 棕灰色
    """
    # 基础色：按海拔分层（深灰绿，大幅去饱和，统一色温轴上）
    if h < 150:
        base = (0.20, 0.24, 0.15)          # 低地绿
    elif h < 230:
        t = (h - 150) / 80.0
        base = lerp_rgb((0.20, 0.24, 0.15), (0.18, 0.22, 0.13), t)
    elif h < 350:
        base = (0.18, 0.22, 0.13)          # 中海拔灰绿
    elif h < 480:
        t = (h - 350) / 130.0
        base = lerp_rgb((0.18, 0.22, 0.13), (0.14, 0.17, 0.11), t)
    else:
        base = (0.13, 0.15, 0.10)          # 高海拔灰褐

    # 坡度修正：陡坡 → 裸露岩石/土坡
    if slope_deg > 30:
        rock = (0.25, 0.20, 0.14)  # 岩石色（也压暗）
        t = min(1.0, (slope_deg - 30) / 25.0)
        base = lerp_rgb(base, rock, t)
    elif slope_deg > 18:
        mix = (0.20, 0.17, 0.11)  # 土石混合（压暗）
        t = (slope_deg - 18) / 12.0
        base = lerp_rgb(base, mix, t * 0.6)

    # 微观纹理扰动：基于世界坐标的伪随机，让大片绿色有细微变化
    noise = np.sin(h * 12.9898) * 43758.5453
    noise = noise - np.floor(noise)  # 伪随机 [0,1)
    tint = 0.88 + noise * 0.24       # 0.88~1.12
    return (
        max(0.0, min(1.0, base[0] * tint)),
        max(0.0, min(1.0, base[1] * tint)),
        max(0.0, min(1.0, base[2] * tint * 0.9)),
    )


def lerp_rgb(a, b, t):
    return (a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t, a[2] + (b[2]-a[2])*t)


# ============================================================
# 顶点打包工具（60B/顶点 → 28B/顶点）
# ============================================================

def pack_normal_2_10_10_10(nx, ny, nz):
    """
    将三个 [-1,1] 法线分量打包成一个 int32（GL_INT_2_10_10_10_REV 格式）。
    X → bits 0-9, Y → bits 10-19, Z → bits 20-29, W=0 → bits 30-31
    每分量 10-bit signed，编码: round(v * 511)
    """
    xi = np.clip(np.round(nx.astype(np.float64) * 511.0), -511, 511).astype(np.int32) & 0x3FF
    yi = np.clip(np.round(ny.astype(np.float64) * 511.0), -511, 511).astype(np.int32) & 0x3FF
    zi = np.clip(np.round(nz.astype(np.float64) * 511.0), -511, 511).astype(np.int32) & 0x3FF
    return xi | (yi << 10) | (zi << 20)


def pack_rgba_u8(r, g, b, a=None):
    """
    将四个 [0,1] 分量打包成一个 uint32（GL_UNSIGNED_BYTE 格式）。
    R → bits 0-7, G → bits 8-15, B → bits 16-23, A → bits 24-31
    编码: round(v * 255)
    """
    ri = np.clip(np.round(r.astype(np.float64) * 255.0), 0, 255).astype(np.uint32)
    gi = np.clip(np.round(g.astype(np.float64) * 255.0), 0, 255).astype(np.uint32)
    bi = np.clip(np.round(b.astype(np.float64) * 255.0), 0, 255).astype(np.uint32)
    if a is None:
        ai = np.uint32(255) << 24
    else:
        ai = (np.clip(np.round(a.astype(np.float64) * 255.0), 0, 255).astype(np.uint32) << 24)
    return ri | (gi << 8) | (bi << 16) | ai


def pack_half_uv(u, v):
    """
    将两个 UV 分量打包成一个 uint32（GL_HALF_FLOAT 格式）。
    U → bits 0-15, V → bits 16-31
    """
    uh = u.astype(np.float16).view(np.uint16).astype(np.uint32)
    vh = v.astype(np.float16).view(np.uint16).astype(np.uint32)
    return uh | (vh << 16)


def _terrain_color_vec(h_arr, slope_arr):
    """
    向量化版本的地形颜色生成。
    h_arr, slope_arr: 同形状 numpy 数组（float64/float32）
    返回: (*shape, 3) float32，与标量版 _terrain_color 逻辑 100% 一致
    """
    base_r = np.empty_like(h_arr, dtype=np.float64)
    base_g = np.empty_like(h_arr, dtype=np.float64)
    base_b = np.empty_like(h_arr, dtype=np.float64)

    # ---- 基础色：按海拔分段赋值 ----
    # h < 150
    m0 = h_arr < 150
    base_r[m0], base_g[m0], base_b[m0] = 0.20, 0.24, 0.15
    # 150 <= h < 230
    m1 = (h_arr >= 150) & (h_arr < 230)
    if np.any(m1):
        t = (h_arr[m1] - 150.0) / 80.0
        base_r[m1] = 0.20 + (0.18 - 0.20) * t
        base_g[m1] = 0.24 + (0.22 - 0.24) * t
        base_b[m1] = 0.15 + (0.13 - 0.15) * t
    # 230 <= h < 350
    m2 = (h_arr >= 230) & (h_arr < 350)
    base_r[m2], base_g[m2], base_b[m2] = 0.18, 0.22, 0.13
    # 350 <= h < 480
    m3 = (h_arr >= 350) & (h_arr < 480)
    if np.any(m3):
        t = (h_arr[m3] - 350.0) / 130.0
        base_r[m3] = 0.18 + (0.14 - 0.18) * t
        base_g[m3] = 0.22 + (0.17 - 0.22) * t
        base_b[m3] = 0.13 + (0.11 - 0.13) * t
    # h >= 480
    m4 = h_arr >= 480
    base_r[m4], base_g[m4], base_b[m4] = 0.13, 0.15, 0.10

    # ---- 坡度修正 ----
    # 陡坡 > 30° → 岩石色
    rock_m = slope_arr > 30
    if np.any(rock_m):
        t = np.clip((slope_arr[rock_m] - 30.0) / 25.0, 0.0, 1.0)
        base_r[rock_m] += (0.25 - base_r[rock_m]) * t
        base_g[rock_m] += (0.20 - base_g[rock_m]) * t
        base_b[rock_m] += (0.14 - base_b[rock_m]) * t
    # 中度 18°~30° → 土石混合
    mix_m = (slope_arr > 18) & (~rock_m)
    if np.any(mix_m):
        t = (slope_arr[mix_m] - 18.0) / 12.0 * 0.6
        base_r[mix_m] += (0.20 - base_r[mix_m]) * t
        base_g[mix_m] += (0.17 - base_g[mix_m]) * t
        base_b[mix_m] += (0.11 - base_b[mix_m]) * t

    # ---- 微观纹理扰动 ----
    noise = np.sin(h_arr * 12.9898) * 43758.5453
    noise = noise - np.floor(noise)
    tint = 0.88 + noise * 0.24

    r = np.clip(base_r * tint, 0.0, 1.0)
    g = np.clip(base_g * tint, 0.0, 1.0)
    b = np.clip(base_b * tint * 0.9, 0.0, 1.0)

    result = np.empty((*h_arr.shape, 3), dtype=np.float32)
    result[..., 0] = r.astype(np.float32)
    result[..., 1] = g.astype(np.float32)
    result[..., 2] = b.astype(np.float32)
    return result


# ============================================================
# 六B、PBR 材质权重计算（4 层混合：草/岩/土/碎石）
# ============================================================

def _compute_material_weights(h, slope_deg, sd):
    """
    根据高度/坡度/道路距离计算 4 层材质权重。
    返回 (w_grass, w_rock, w_dirt, w_gravel)，sum = 1.0

    sd: signed_distance（负=赛道上，正=赛道外）
    """
    # ---- 1. 基础草/岩分布（海拔 + 坡度） ----
    if h < 350:
        grass_base = 1.0
    elif h < 500:
        grass_base = (500 - h) / 150.0
    else:
        grass_base = 0.0

    rock_slope = max(0.0, min(1.0, (slope_deg - 30.0) / 15.0))

    # ---- 2. 道路影响带 ----
    # sd ∈ [0, 1]   → 纯碎石
    # sd ∈ [1, 5]   → 碎石→土
    # sd ∈ [5, 15]  → 土→草
    # sd > 15       → 纯自然
    if sd < 0:
        gravel_road = 0.0
        dirt_road = 0.0
    elif sd < 1.0:
        gravel_road = 1.0 - sd
        dirt_road = 0.0
    elif sd < 5.0:
        gravel_road = 0.0
        dirt_road = 1.0 - (sd - 1.0) / 4.0
    elif sd < 15.0:
        gravel_road = 0.0
        dirt_road = 1.0 - (sd - 5.0) / 10.0
    else:
        gravel_road = 0.0
        dirt_road = 0.0

    # ---- 3. 合成（★ 全部 max(0, ...) 保护，防止负数权重导致紫红色异常） ----
    road_influence = gravel_road + dirt_road
    residual = max(0.0, 1.0 - grass_base - rock_slope)

    w_grass  = max(0.0, grass_base * (1.0 - rock_slope) * (1.0 - road_influence))
    w_rock   = max(0.0, rock_slope * 0.85 + grass_base * rock_slope * 0.15)
    w_dirt   = max(0.0, dirt_road + residual * 0.3 * (1.0 - road_influence))
    w_gravel = max(0.0, gravel_road)

    total = w_grass + w_rock + w_dirt + w_gravel
    if total < 1e-6:
        return (1.0, 0.0, 0.0, 0.0)
    return (w_grass/total, w_rock/total, w_dirt/total, w_gravel/total)


def _compute_material_weights_vec(h_arr, slope_arr, sd_arr):
    """
    向量化版本：一次性计算所有格点的 4 层材质权重。
    h_arr, slope_arr, sd_arr: 同形状 numpy 数组（float64/float32）
    返回: (*shape, 4) float32，与标量版 _compute_material_weights 逻辑 100% 一致
    """
    # ---- 1. 基础草/岩分布 ----
    grass_base = np.where(h_arr < 350, 1.0,
                  np.where(h_arr < 500, (500.0 - h_arr) / 150.0, 0.0))
    rock_slope = np.clip((slope_arr - 30.0) / 15.0, 0.0, 1.0)

    # ---- 2. 道路影响带 ----
    # gravel_road: sd < 0 → 0, sd ∈ [0,1) → 1-sd, sd ≥ 1 → 0
    gravel_road = np.where(sd_arr < 0, 0.0,
                   np.where(sd_arr < 1.0, 1.0 - sd_arr, 0.0))
    # dirt_road: sd < 0 → 0; [0,1) → 0; [1,5) → 1-(sd-1)/4; [5,15) → 1-(sd-5)/10; ≥15 → 0
    dirt_road = np.where(sd_arr < 0, 0.0,
                np.where(sd_arr < 1.0, 0.0,
                  np.where(sd_arr < 5.0, 1.0 - (sd_arr - 1.0) / 4.0,
                    np.where(sd_arr < 15.0, 1.0 - (sd_arr - 5.0) / 10.0, 0.0))))

    # ---- 3. 合成 ----
    road_influence = gravel_road + dirt_road
    residual = np.maximum(0.0, 1.0 - grass_base - rock_slope)

    w_grass  = np.maximum(0.0, grass_base * (1.0 - rock_slope) * (1.0 - road_influence))
    w_rock   = np.maximum(0.0, rock_slope * 0.85 + grass_base * rock_slope * 0.15)
    w_dirt   = np.maximum(0.0, dirt_road + residual * 0.3 * (1.0 - road_influence))
    w_gravel = np.maximum(0.0, gravel_road)

    total = w_grass + w_rock + w_dirt + w_gravel
    total_safe = np.maximum(total, 1e-6)

    shape_4d = (*h_arr.shape, 4)
    result = np.empty(shape_4d, dtype=np.float32)
    result[..., 0] = (w_grass / total_safe).astype(np.float32)
    result[..., 1] = (w_rock / total_safe).astype(np.float32)
    result[..., 2] = (w_dirt / total_safe).astype(np.float32)
    result[..., 3] = (w_gravel / total_safe).astype(np.float32)

    # total 为 0 的点强制设为 (1,0,0,0)
    zero_mask = total < 1e-6
    if np.any(zero_mask):
        result[zero_mask] = (1.0, 0.0, 0.0, 0.0)

    return result


_hash_seed = 0.0
def _hash_float(v):
    """简单的伪随机哈希，用于颜色微扰动"""
    import struct
    b = struct.pack('f', float(v))
    h = b[0] ^ b[1] ^ b[2] ^ b[3]
    return (h & 0xFF) / 255.0


# ============================================================
# 七、整块规则网格地形（2026-09 新架构）
# ============================================================

def build_akina_terrain():
    """
    覆盖赛道区域的整块规则网格地形。
    使用 signed_distance 精准压平赛道区域，彻底消除弯道过生成。
    结果会被缓存到 .npy 文件，第二次启动直接加载。

    版本控制：修改 _TERRAIN_CACHE_VERSION 即可升级缓存，
    旧版本文件会在生成新缓存时被自动清理。
    """
    # 确保 DEM 已初始化并雕刻（雕刻已在 build_mountain_road 中完成）
    _ensure_dem_ready()

    # 缓存名 = 前缀_版本_path指纹：path 一变（重采样/平滑/坡度裁剪/避车道）
    # 地形压平带的形状就跟着变，必须重建，否则新赛道边缘露出旧地形。
    cache_name = (f"{_TERRAIN_CACHE_PREFIX}_{_TERRAIN_CACHE_VERSION}"
                  f"_{_path_fingerprint(get_akina_path())}")
    cache_path = _get_cache_path(cache_name)
    # 用 .npz 格式保存打包后的 dict
    npz_path = cache_path.replace('.npy', '.npz') if cache_path else None
    if npz_path and os.path.exists(npz_path):
        print(f"[缓存] 加载地形网格缓存 ({cache_name}.npz)...")
        data = np.load(npz_path)
        result = {
            'positions': data['positions'],
            'normals':   data['normals'],
            'colors':    data['colors'],
            'weights':   data['weights'],
            'uvs':       data['uvs'],
            'count':     len(data['positions']),
        }
        print(f"[缓存]  地形网格加载完成: {result['count']:,} 顶点")
        return result

    print(f"[缓存] 构建地形网格（首次运行或版本升级，耗时较长）...")
    t0 = __import__('time').time()
    result = _build_akina_terrain_impl()
    elapsed = __import__('time').time() - t0
    if npz_path and result['count'] > 0:
        try:
            # 保存新缓存前，先清理旧版本的地形缓存（只保留当前版本）
            _clean_terrain_old_cache(keep_name=cache_name)
            np.savez(npz_path,
                     positions=result['positions'],
                     normals=result['normals'],
                     colors=result['colors'],
                     weights=result['weights'],
                     uvs=result['uvs'])
            print(f"[缓存]  地形网格已缓存 ({cache_name}.npz, {elapsed:.1f}s)")
        except Exception as e:
            print(f"[缓存]  地形网格缓存失败: {e}")
    else:
        print(f"[缓存]  地形网格构建完成 ({elapsed:.1f}s)")
    return result


def _build_akina_terrain_impl():
    """整块规则网格地形（v12：适配 29 点赛道横截面 + 红白路缘石）"""
    """
    v12 改进（2026-09-15）：
    1. PROFILE 升级为 29 点精细版（含红白路缘石 + 反光柱）
    2. 新增路缘石红白交替翻转逻辑
    3. 地形压平表面适配新的赛道横截面
    """
    road_verts, track_cells, track_cell_size = _build_visual_track_lookup()

    path = get_akina_path()
    n = len(path)
    if n < 2:
        return {
            'positions': np.empty((0, 3), dtype=f32),
            'normals':   np.empty((0,), dtype=np.int32),
            'colors':    np.empty((0,), dtype=np.uint32),
            'weights':   np.empty((0,), dtype=np.uint32),
            'uvs':       np.empty((0,), dtype=np.uint32),
            'count':     0,
        }

    xs = np.array([p[0] for p in path], dtype=np.float64)
    zs = np.array([p[1] for p in path], dtype=np.float64)

    # ============================================================
    # 五级自适应网格坐标轴（v17：0.4m 核心带 → 8m 远山）
    # ============================================================
    # 赛道包围盒（保持原逻辑，用于计算整体覆盖范围）
    PAD_BASE = 25.0
    track_x_min = float(xs.min()) - PAD_BASE
    track_x_max = float(xs.max()) + PAD_BASE
    track_z_min = float(zs.min()) - PAD_BASE
    track_z_max = float(zs.max()) + PAD_BASE

    # ★ 核心带（赛道外扩 8m，用于 0.4m 密集采样区）
    CORE_PAD = 8.0
    core_x_min = float(xs.min()) - CORE_PAD
    core_x_max = float(xs.max()) + CORE_PAD
    core_z_min = float(zs.min()) - CORE_PAD
    core_z_max = float(zs.max()) + CORE_PAD

    # 五级参数（见设计文档）
    #   远山(8m) → 近山(3m) → 过渡带(1.2m) → 核心带(0.4m) → 过渡带(1.2m) → 近山(3m) → 远山(8m)
    CORE_RES     = 0.4   # 核心带（sd < 8m）
    TRANS_RES    = 1.2   # 过渡带（8m < sd < 50m）
    NEAR_RES     = 3.0   # 近山（50m < sd < 150m）
    FAR_RES      = 8.0   # 远山（sd > 150m）

    # 过渡带宽度（单侧）：从核心带边缘向外延伸 42m（= 50-8）
    TRANS_WIDTH  = 42.0
    # 近山宽度（单侧）：从过渡带边缘向外延伸 100m（= 150-50）
    NEAR_WIDTH   = 100.0
    # 远山额外外扩（确保地形覆盖完整）
    FAR_EXTRA    = 50.0

    def _build_axis_with_bands(lo, hi, road_lo, road_hi):
        """
        五级自适应 1D 坐标轴生成器。

        参数：
          lo, hi   : 地形整体范围（含远山外扩）
          road_lo, road_hi : 核心带范围（赛道 AABB 外扩 8m）

        生成策略：
          [lo, road_lo-62)         : 远山，间距 8m
          [road_lo-62, road_lo-20) : 近山，间距 3m
          [road_lo-20, road_lo)    : 过渡带，间距 1.2m
          [road_lo, road_hi)       : ★ 核心带，间距 0.4m
          [road_hi, road_hi+20)    : 过渡带，间距 1.2m
          [road_hi+20, road_hi+62) : 近山，间距 3m
          [road_hi+62, hi]         : 远山，间距 8m
        """
        pieces = []

        # ---- 左半区（lo → road_lo）----
        # 远山
        if lo < road_lo - 62:
            pieces.append(np.arange(lo, road_lo - 62, FAR_RES, dtype=np.float64))
        # 近山
        if road_lo - 62 < road_lo - 20:
            start_near = max(lo, road_lo - 62)
            pieces.append(np.arange(start_near, road_lo - 20, NEAR_RES, dtype=np.float64))
        # 过渡带
        if road_lo - 20 < road_lo:
            start_trans = max(lo, road_lo - 20)
            pieces.append(np.arange(start_trans, road_lo, TRANS_RES, dtype=np.float64))

        # ---- 核心带 ----
        pieces.append(np.arange(road_lo, road_hi + CORE_RES * 0.5,
                                CORE_RES, dtype=np.float64))

        # ---- 右半区（road_hi → hi）----
        # 过渡带
        if road_hi < road_hi + 20:
            pieces.append(np.arange(road_hi, road_hi + 20 + TRANS_RES * 0.5,
                                    TRANS_RES, dtype=np.float64))
        # 近山
        if road_hi + 20 < road_hi + 62:
            end_near = min(hi, road_hi + 62)
            pieces.append(np.arange(max(road_hi + 20, road_hi),
                                    end_near + NEAR_RES * 0.5,
                                    NEAR_RES, dtype=np.float64))
        # 远山
        if road_hi + 62 < hi:
            pieces.append(np.arange(max(road_hi + 62, road_hi),
                                    hi + FAR_RES * 0.5,
                                    FAR_RES, dtype=np.float64))

        # 合并、去重、排序
        axis = np.concatenate(pieces)
        axis = np.unique(axis)
        return axis

    # 整体地形覆盖范围
    FAR_LO = track_x_min - NEAR_WIDTH - FAR_EXTRA  # 约 min(xs) - 175
    FAR_HI = track_x_max + NEAR_WIDTH + FAR_EXTRA
    FAR_LO_Z = track_z_min - NEAR_WIDTH - FAR_EXTRA
    FAR_HI_Z = track_z_max + NEAR_WIDTH + FAR_EXTRA

    xs_axis = _build_axis_with_bands(FAR_LO, FAR_HI, core_x_min, core_x_max)
    zs_axis = _build_axis_with_bands(FAR_LO_Z, FAR_HI_Z, core_z_min, core_z_max)

    n_cols = len(xs_axis)
    n_rows = len(zs_axis)
    n_total = n_rows * n_cols

    # ---- 顶点数估算与自动调优 ----
    # 目标：预估顶点数在 2~3.5M 之间
    TARGET_VERTS_MAX = 3_500_000
    TARGET_VERTS_IDEAL = 2_500_000

    # 自动搜索合适的核心带间距（从 0.4 开始，逐步增加直到达标）
    # 说明：1:1 真实赛道 AABB 可能很大（>1000m），
    # 0.4m 间距会导致千万级顶点，必须根据实际尺寸自动调优。
    _core_res_candidates = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
    _chosen_core_res = _core_res_candidates[0]

    for _cr in _core_res_candidates:
        CORE_RES = _cr
        xs_axis = _build_axis_with_bands(FAR_LO, FAR_HI, core_x_min, core_x_max)
        zs_axis = _build_axis_with_bands(FAR_LO_Z, FAR_HI_Z, core_z_min, core_z_max)
        n_cols = len(xs_axis)
        n_rows = len(zs_axis)
        n_total = n_rows * n_cols
        if n_total == 0:
            continue
        estimated_tris = (n_rows - 1) * (n_cols - 1) * 2
        estimated_verts = estimated_tris * 3
        if estimated_verts <= TARGET_VERTS_MAX or _cr == _core_res_candidates[-1]:
            _chosen_core_res = _cr
            break

    # 打印最终选定的参数结果
    track_w = float(xs.max() - xs.min())
    track_h = float(zs.max() - zs.min())
    print(f"  网格地形（5 级自适应）: {n_cols}×{n_rows} = {n_total:,} 格点")
    print(f"    赛道 AABB: {track_w:.0f}m×{track_h:.0f}m | 核心带 {CORE_RES}m | "
          f"过渡带 {TRANS_RES}m | 近山 {NEAR_RES}m | 远山 {FAR_RES}m")
    print(f"    预估顶点数: {estimated_verts:,}（目标 4,000,000 ~ 6,000,000）")

    if _chosen_core_res > 0.4:
        print(f"    ℹ️ 赛道包围盒较大，核心带从 0.4m 自动调整为 {CORE_RES}m")
    if estimated_verts > TARGET_VERTS_MAX:
        print(f"    ⚠️ 顶点数仍偏高 ({estimated_verts:,})，建议减小过渡带宽度或加大 FAR_RES")
    elif estimated_verts < 2_000_000:
        print(f"    ℹ️ 顶点数偏少，可尝试减小核心带间距以提升接缘带细节")

    XX, ZZ = np.meshgrid(xs_axis, zs_axis)
    XX_flat = XX.ravel()
    ZZ_flat = ZZ.ravel()

    # ============================================================
    # 局部绑定加速
    # ============================================================
    _sd    = signed_distance
    _vis_q = _query_visual_track_height
    _tc    = _terrain_color
    _ASPHALT = ASPHALT_THICKNESS
    TRACK_PAD = 0.10     # v21：坡比模型，地形压平面低于视觉赛道的量

    print(f"  计算 {n_total:,} 格点的高度...")
    t0 = __import__('time').time()

    # ---- 阶段 1：向量化 SD + 视觉赛道高度查询 ----
    # 使用 cKDTree + 窗口段投影一次性批量计算所有格点的有符号距离
    from scipy.spatial import cKDTree
    M_path = len(path)
    # 赛道路径点 (x,z) 和半宽
    pts_sd = np.array([(p[0], p[1]) for p in path], dtype=np.float64)
    hw_sd  = np.array([p[3] * 0.5 for p in path], dtype=np.float64)
    path_y = np.array([p[2] for p in path], dtype=np.float64)  # 路径中心线高

    # 线段向量
    sd_ex = pts_sd[1:, 0] - pts_sd[:-1, 0]  # (M-1,)
    sd_ez = pts_sd[1:, 1] - pts_sd[:-1, 1]
    sd_L2 = sd_ex*sd_ex + sd_ez*sd_ez
    sd_L  = np.sqrt(np.maximum(sd_L2, 1e-12))
    # 法线（切线逆时针旋转 90°）
    sd_nx = -sd_ez / sd_L  # (M-1,)
    sd_nz =  sd_ex / sd_L
    # 线段起点
    sd_ax = pts_sd[:-1, 0]
    sd_az = pts_sd[:-1, 1]

    # KDTree 批量查询最近路径点
    _sd_tree = cKDTree(pts_sd)
    _sd_points = np.column_stack([XX_flat, ZZ_flat])
    _, sd_idx = _sd_tree.query(_sd_points)  # (N,) 最近路径点索引

    # 第二近路径点索引（双近腿取低者，修复 U 型弯两腿之间堆山）
    _, sd_idx2 = _sd_tree.query(_sd_points, k=2)  # (N, 2)
    seg_i_b = np.clip(sd_idx2[:, 1], 0, M_path - 2).astype(np.int64)   # 第二近点所在段
    del sd_idx2

    # 窗口段投影：检查 [idx-2, idx+2] 范围内的线段
    # 策略：找投影距离 d2 最小的段（与 _project_nearest_segment 一致），
    #       再取该段的 signed_distance
    _sd_window = 2
    best_d2 = np.full(n_total, 1e18, dtype=np.float64)
    sd_arr  = np.empty(n_total, dtype=np.float64)
    road_ref = np.empty(n_total, dtype=np.float64)  # 最近投影点的路径中心线高
    for d in range(-_sd_window, _sd_window + 1):
        seg_i = np.clip(sd_idx + d, 0, M_path - 2).astype(np.int64)
        # 提取线段数据
        ax = sd_ax[seg_i]; az = sd_az[seg_i]
        ex = sd_ex[seg_i]; ez = sd_ez[seg_i]
        L2 = sd_L2[seg_i]
        # 投影参数 t（点到线段的参数化位置）
        t = np.where(L2 > 1e-12,
                     ((XX_flat - ax) * ex + (ZZ_flat - az) * ez) / L2,
                     0.0)
        t = np.clip(t, 0.0, 1.0)
        # 投影点
        px = ax + ex * t
        pz = az + ez * t
        # 投影距离平方
        d2_j = (XX_flat - px)**2 + (ZZ_flat - pz)**2
        # 投影中心线高（与半宽插值同源）
        cy_proj = path_y[seg_i] * (1.0 - t) + path_y[seg_i + 1] * t
        if d == -_sd_window:
            # 第一次迭代：直接赋值
            # 半宽插值
            hw_proj = hw_sd[seg_i] * (1.0 - t) + hw_sd[seg_i + 1] * t
            lateral = (XX_flat - px) * sd_nx[seg_i] + (ZZ_flat - pz) * sd_nz[seg_i]
            sd_arr[:] = np.abs(lateral) - hw_proj
            road_ref[:] = cy_proj
            best_d2[:] = d2_j
        else:
            # 后续迭代：只在投影距离更小时更新
            mask = d2_j < best_d2
            if np.any(mask):
                hw_proj = hw_sd[seg_i] * (1.0 - t) + hw_sd[seg_i + 1] * t
                lateral = (XX_flat - px) * sd_nx[seg_i] + (ZZ_flat - pz) * sd_nz[seg_i]
                sd_arr[mask] = (np.abs(lateral) - hw_proj)[mask]
                road_ref[mask] = cy_proj[mask]
                best_d2[mask] = d2_j[mask]
    del best_d2
    # 清理临时变量
    del _sd_tree, _sd_points, sd_idx

    # vis_q 只对 sd < 28.0 的子集调用（约占总量的 20-30%，节省 70%+ 循环）
    vis_arr = np.full(n_total, np.nan, dtype=np.float64)
    near_mask = sd_arr < 28.0
    near_indices = np.where(near_mask)[0]
    for k in near_indices:
        wx = XX_flat[k]; wz = ZZ_flat[k]
        vh = _vis_q(road_verts, track_cells, track_cell_size, wx, wz)
        if vh is not None:
            vis_arr[k] = vh
    near_count = len(near_indices)
    del near_mask, near_indices

    print(f"    SD 计算完成 ({__import__('time').time()-t0:.1f}s, "
          f"vis_q 仅查询 {near_count:,}/{n_total:,} = {near_count/max(n_total,1)*100:.1f}%)")

    # ---- 阶段 2：高度计算（坡比受限的向量化路堤/路堑）----
    FLAT_END   = _TERRAIN_FLAT_END    # 5.0：平整带宽度
    CUT_SLOPE  = _TERRAIN_CUT_SLOPE   # 0.75：上边坡（挖方）
    FILL_SLOPE = _TERRAIN_FILL_SLOPE  # 1.10：下边坡（填方）

    # 路肩基准高
    dem_h = _dem_h_vec(XX_flat, ZZ_flat)
    shoulder = np.where(np.isnan(vis_arr), road_ref + _ASPHALT, vis_arr) - TRACK_PAD

    # ★ 抽成共享函数，避免双近腿重复代码
    def _leg_embank(seg_i):
        """对指定路径段索引，计算坡比受限的路堤高度（全向量化）"""
        # 投影参数 t
        _ax = sd_ax[seg_i]; _az = sd_az[seg_i]
        _ex = sd_ex[seg_i]; _ez = sd_ez[seg_i]
        _L2 = sd_L2[seg_i]
        _t = np.where(_L2 > 1e-12,
                      ((XX_flat - _ax) * _ex + (ZZ_flat - _az) * _ez) / _L2, 0.0)
        _t = np.clip(_t, 0.0, 1.0)
        # 横向距离（signed distance）
        _lat = (XX_flat - (_ax + _ex * _t)) * sd_nx[seg_i]              + (ZZ_flat - (_az + _ez * _t)) * sd_nz[seg_i]
        _hw = hw_sd[seg_i] * (1.0 - _t) + hw_sd[seg_i + 1] * _t
        _sd = np.abs(_lat) - _hw
        # 路肩基准高
        _sh = path_y[seg_i] * (1.0 - _t) + path_y[seg_i + 1] * _t + _ASPHALT - TRACK_PAD
        _d = _sd - FLAT_END
        _dy = dem_h - _sh
        _eb = _sh + np.clip(_dy, -FILL_SLOPE * np.maximum(_d, 0.0),
                                  CUT_SLOPE  * np.maximum(_d, 0.0))
        return np.where(_sd <= FLAT_END, _sh, _eb)  # ★ 不含排水沟，统一在外部加

    d = sd_arr - FLAT_END
    dy = dem_h - shoulder
    embank = shoulder + np.clip(dy, -FILL_SLOPE * np.maximum(d, 0.0),
                                     CUT_SLOPE  * np.maximum(d, 0.0))
    heights = np.where(sd_arr <= FLAT_END, shoulder, embank)

    # ★ 双近腿取低者：U 型弯两腿之间取较低的路堤，形成垭口
    embank_b = _leg_embank(seg_i_b)
    heights = np.minimum(heights, embank_b)
    del embank_b

    # ---- 路肩排水沟（坡脚集水浅槽，提升山路真实感）----
    ditch = -0.35 * np.exp(-((sd_arr - (FLAT_END + 0.8)) / 0.7) ** 2)
    heights = heights + ditch

    # （v25 曾在此处给侧沟开槽。地形核心带被自动降级到 3.0m，0.8m 宽的开槽
    #   只会被采样成沿途随机散坑，已废弃 —— 侧沟改为高出地面的混凝土几何体。）

    # ★ 近路高差硬上限：坡体在 25m 内不爬超过 5m（形成切削台地，比直坡自然）
    MAX_NEAR_RISE = 5.0
    near = sd_arr < 25.0
    heights = np.where(near & (dy > 0),
                       np.minimum(heights, shoulder + MAX_NEAR_RISE),
                       heights)

    # ---- 起终点"沿路服务带"压平（v31）：地形为场地让路 ----
    #   服务带紧贴赛道（内边缘离沥青路缘 _VENUE_GAP_ROAD ≈ 2.2m），
    #   平台标高 = 路肩 + 路缘高，并沿切向带路面纵坡 grade。
    #   ⚠️ 朝路面那侧必须用窄过渡带 _VENUE_FLATTEN_IN：用 10m 的话权重会
    #      伸到路面底下把路肩顶高 ~0.1m → 地形穿出 0.15m 厚的沥青面。
    #   标量对应实现在 get_ground_height → _venue_flatten_h，必须一致。
    for (vp_cx, vp_cz, vp_fx, vp_fz, vp_rx, vp_rz, vp_y, vp_HL, vp_HW,
         vp_g) in _venue_pads():
        _vu = (XX_flat - vp_cx) * vp_fx + (ZZ_flat - vp_cz) * vp_fz
        _vv = (XX_flat - vp_cx) * vp_rx + (ZZ_flat - vp_cz) * vp_rz
        _dU = np.abs(_vu) - vp_HL
        _dV = np.where(_vv < -vp_HW, -vp_HW - _vv,
                       np.where(_vv > vp_HW, _vv - vp_HW, 0.0))
        _vd = np.maximum(np.maximum(_dU, _dV), 0.0)
        _vW = np.where((_dV > 0.0) & (_vv < -vp_HW) & (_dV >= _dU),
                       _VENUE_FLATTEN_IN, _VENUE_FLATTEN_OUT)
        _vw = np.clip(_vd / _vW, 0.0, 1.0)
        _vw = _vw * _vw * (3.0 - 2.0 * _vw)
        _h0 = heights          # 压平前的天然高（np 运算生成新数组，引用不变）
        _vtgt = vp_y + vp_g * _vu - 0.06
        heights = _vtgt + (_h0 - _vtgt) * _vw

        # ★ 缓冲带（v ∈ [-HW-GAP, -HW] 且 |u| ≤ HL）：地形跟随铺面、恒低 9cm。
        #   与 _venue_flatten_h 的 strip 分支**逐项一致**（两份实现必须同步改，
        #   这里漏改 = 地形网格用旧压平，铺面照样被埋 —— v33 第一版就栽在这）。
        #   ⚠️ 必须用压平前的 _h0（标量版 strip 分支在 blend 之前 return）。
        _strip = ((np.abs(_vu) <= vp_HL) & (_vv <= -vp_HW)
                  & (_vv >= -vp_HW - _VENUE_GAP_ROAD))
        _t = np.clip((-vp_HW - _vv) / _VENUE_GAP_ROAD, 0.0, 1.0)
        _surf = ((vp_y + vp_g * _vu) * (1.0 - _t)
                 + (_h0 + _VENUE_APRON_UP) * _t - _VENUE_APRON_UP)
        heights = np.where(_strip, _surf, heights)

    heights = heights.reshape(n_rows, n_cols)
    print(f"    高度计算完成 ({__import__('time').time()-t0:.1f}s)")

    # ---- 阶段 3：梯度（非均匀网格，用实际坐标轴） ----
    # np.gradient(f, *coords) 支持非均匀间距
    dh_dz, dh_dx = np.gradient(heights, zs_axis, xs_axis)
    slope_map = np.degrees(np.arctan(np.sqrt(dh_dx**2 + dh_dz**2)))
    slope_map = np.clip(slope_map, 0.0, 60.0)

    colors = _terrain_color_vec(heights, slope_map)
    print(f"    颜色计算完成 ({__import__('time').time()-t0:.1f}s)")

    # ---- AO（环境光遮蔽）：山谷凹陷处自动变暗 ----
    # 用大核高斯模糊高度场作为"环境基准"，与原始高度的差 = 局部凹陷程度
    h_smooth = gaussian_filter(heights, sigma=2.5)
    ao_map = np.clip((heights - h_smooth) / 3.0 + 0.5, 0.75, 1.0)
    colors[..., 0] *= ao_map
    colors[..., 1] *= ao_map
    colors[..., 2] *= ao_map
    print(f"    AO 计算完成")

    # ---- 阶段 4：向量化法线 ----
    nx_arr = -dh_dx
    ny_arr = np.ones_like(heights)
    nz_arr = -dh_dz
    norm = np.sqrt(nx_arr*nx_arr + ny_arr*ny_arr + nz_arr*nz_arr)
    nx_arr /= norm; ny_arr /= norm; nz_arr /= norm

    # ---- 法线定向明暗（幅度减半，实时光照已有 NdotL，烘焙只留一点层次） ----
    ndot_up = ny_arr  # 法线Y分量（已归一化）
    sun_shade = 0.88 + 0.12 * np.maximum(ndot_up, 0.0)
    colors[..., 0] *= sun_shade
    colors[..., 1] *= sun_shade
    colors[..., 2] *= sun_shade

    # ---- 材质权重计算（4 层 PBR：草/岩/土/碎石，向量化） ----
    sd_2d = sd_arr.reshape(n_rows, n_cols)
    weights = _compute_material_weights_vec(heights, slope_map, sd_2d)
    print(f"    材质权重计算完成")

    # ---- 阶段 5：组装顶点（打包为 28B/顶点） ----
    us = ((XX - FAR_LO) / (FAR_HI - FAR_LO) * 10.0).astype(np.float32)
    vs = ((ZZ - FAR_LO_Z) / (FAR_HI_Z - FAR_LO_Z) * 10.0).astype(np.float32)

    # ---- 阶段 6：向量化三角化 ----
    iz_range = np.arange(n_rows - 1)
    ix_range = np.arange(n_cols - 1)
    IZ, IX = np.meshgrid(iz_range, ix_range, indexing='ij')
    IZ = IZ.ravel(); IX = IX.ravel()

    a = IZ * n_cols + IX
    b = IZ * n_cols + IX + 1
    c = (IZ + 1) * n_cols + IX + 1
    d = (IZ + 1) * n_cols + IX

    n_quads = len(a)
    indices = np.empty(n_quads * 6, dtype=np.int32)
    indices[0::6] = a
    indices[1::6] = b
    indices[2::6] = c
    indices[3::6] = a
    indices[4::6] = c
    indices[5::6] = d

    N = len(indices)

    # ---- 按索引从各分量提取顶点数据并打包 ----
    # 位置（3 × f32 = 12B，不动）
    pos = np.empty((N, 3), dtype=np.float32)
    pos[:, 0] = XX.astype(np.float32).ravel()[indices]
    pos[:, 1] = heights.astype(np.float32).ravel()[indices]
    pos[:, 2] = ZZ.astype(np.float32).ravel()[indices]

    # 法线打包（3 × 10-bit → 1 × int32 = 4B）
    nx_f = nx_arr.astype(np.float32).ravel()[indices]
    ny_f = ny_arr.astype(np.float32).ravel()[indices]
    nz_f = nz_arr.astype(np.float32).ravel()[indices]
    normals_pk = pack_normal_2_10_10_10(nx_f, ny_f, nz_f)

    # 颜色打包（RGB + A=255 → 1 × uint32 = 4B）
    col_r = colors[..., 0].ravel()[indices]
    col_g = colors[..., 1].ravel()[indices]
    col_b = colors[..., 2].ravel()[indices]
    colors_pk = pack_rgba_u8(col_r, col_g, col_b)

    # 权重打包（4 × u8 → 1 × uint32 = 4B）
    w0 = weights[..., 0].ravel()[indices]
    w1 = weights[..., 1].ravel()[indices]
    w2 = weights[..., 2].ravel()[indices]
    w3 = weights[..., 3].ravel()[indices]
    weights_pk = pack_rgba_u8(w0, w1, w2, w3)

    # UV 打包（2 × f16 → 1 × uint32 = 4B）
    u_f = us.ravel()[indices]
    v_f = vs.ravel()[indices]
    uvs_pk = pack_half_uv(u_f, v_f)

    elapsed = __import__('time').time() - t0
    per_vert = 12 + 4 + 4 + 4 + 4  # 28B
    total_mb = N * per_vert / (1024 * 1024)
    print(f"  网格地形顶点: {N:,} (打包后 {per_vert}B/顶点 = {total_mb:.0f} MB, 总耗时 {elapsed:.1f}s)")

    return {
        'positions': pos,
        'normals':   normals_pk,
        'colors':    colors_pk,
        'weights':   weights_pk,
        'uvs':       uvs_pk,
        'count':     N,
    }


# ============================================================
# 七、赛道网格构建
# ============================================================

def build_mountain_road():
    """
    3A 标准赛道生成：Profile + Banking + 地形色过渡（v2：numpy 向量化加速）。

    核心特性：
      1. 11 点横截面（路肩→路缘石→车道→标线→中心）
      2. 最外侧路肩使用地形颜色，实现道路→地形的无缝视觉过渡
      3. Banking（超高）：弯道倾斜
      4. DEM 雕刻：在赛道生成前先"在山上凿路"
      5. v2：使用 numpy broadcasting 批量计算横截面，加速 5~10 倍

    输出: (N, 11) float32 数组
    """
    global _mountain_road_mesh_cache
    if _mountain_road_mesh_cache is not None:
        return _mountain_road_mesh_cache

    # ---- DEM 雕刻（每轮首运行执行一次） ----
    global _DEM_CARVED
    _ensure_dem_ready()
    if not _DEM_CARVED and _DEM_READY:
        path, tangents = get_akina_path(return_tangents=True)[:2]
        _carve_dem(path)
        _DEM_CARVED = True
    else:
        path, tangents = get_akina_path(return_tangents=True)[:2]

    # ---- path 指纹（避车道等路径修改后自动切换缓存）----
    _fp = _path_fingerprint(path)
    print(f"[路径] 指纹: {_fp}")

    # ---- 尝试从磁盘加载缓存（带 path 指纹：避车道加宽后自动失效）----
    _road_cache_name = f"akina_road_mesh_v4_{_fp}"
    _road_cache_path = _get_cache_path(_road_cache_name)
    # 命中缓存也要清理旧指纹/旧版本号（否则 _v1/_v2 这类化石永远留着）
    _clean_stale_env_caches("akina_road_mesh_v4", _road_cache_name)
    if _road_cache_path and os.path.exists(_road_cache_path):
        print(f"[缓存] 加载赛道网格 ({_road_cache_name})...")
        cached = np.load(_road_cache_path)
        _mountain_road_mesh_cache = cached
        print(f"[缓存]  赛道网格加载完成: {len(cached)} 顶点")
        return cached

    n = len(path)
    if n < 2:
        return np.empty((0, 11), dtype=f32)

    # 赛道横截面（17 点精简版，保留关键结构 + 红白路缘石）
    PROFILE = [
        (-5.00, 0.00, (0.22, 0.26, 0.14)),   # 草地
        (-4.20, 0.03, (0.28, 0.26, 0.22)),   # 碎石带
        (-3.75, 0.09, (0.48, 0.16, 0.13)),   # 路缘石（红）
        (-3.60, 0.10, (0.72, 0.72, 0.68)),   # 路缘石（白）
        (-3.48, 0.05, (0.55, 0.55, 0.50)),   # 内侧过渡
        (-3.45, 0.00, (0.85, 0.85, 0.80)),   # 边缘线
        (-3.40, 0.00, (0.20, 0.19, 0.17)),   # 左车道外
        (-1.70, 0.00, (0.20, 0.19, 0.17)),   # 左车道
        ( 0.00, 0.00, (0.20, 0.19, 0.17)),   # 中心
        ( 1.70, 0.00, (0.20, 0.19, 0.17)),   # 右车道
        ( 3.40, 0.00, (0.20, 0.19, 0.17)),   # 右车道外
        ( 3.45, 0.00, (0.85, 0.85, 0.80)),   # 边缘线
        ( 3.48, 0.05, (0.55, 0.55, 0.50)),   # 内侧过渡
        ( 3.60, 0.10, (0.72, 0.72, 0.68)),   # 路缘石（白）
        ( 3.75, 0.09, (0.48, 0.16, 0.13)),   # 路缘石（红）
        ( 4.20, 0.03, (0.28, 0.26, 0.22)),   # 碎石带
        ( 5.00, 0.00, (0.22, 0.26, 0.14)),   # 草地
    ]
    VPS = 17  # 每段横截面的顶点数（35→17，精简过渡带）
    # 地形色覆盖的 profile 索引（最外侧草地）
    TERRAIN_COLOR_IDX = {0, 16}
    # 碎石带索引
    GRAVEL_IDX = {1}           # 左侧碎石带
    GRAVEL_IDX_R = {15}       # 右侧碎石带
    # 接缘带高度起伏索引
    RIPPLE_IDX = {1, 2, 14, 15}

    # ---- path/tangents 转 numpy ----
    path_arr = np.array(path, dtype=np.float32)        # (n, 5): x, z, y, w, bank
    tang_arr = np.array(tangents, dtype=np.float32)    # (n, 3): tx, ty, tz

    # ---- 向量化 2D 水平法线 ----
    tx = tang_arr[:, 0]; ty = tang_arr[:, 1]; tz = tang_arr[:, 2]
    nx_2d = -tz; nz_2d = tx
    n_len = np.sqrt(nx_2d**2 + nz_2d**2)
    n_len[n_len < 1e-10] = 1.0
    nx_2d /= n_len; nz_2d /= n_len

    # ---- 向量化 banking ----
    bank = path_arr[:, 4]
    cos_b = np.cos(bank); sin_b = np.sin(bank)

    # ---- 向量化 3D 法线（banking 后的向上方向） ----
    inv_cos = 1.0 - cos_b
    nx3 = -tz * sin_b + tx * ty * inv_cos
    ny3 = cos_b + ty * ty * inv_cos
    nz3 = tx * sin_b + tz * ty * inv_cos
    nl = np.sqrt(nx3*nx3 + ny3*ny3 + nz3*nz3)
    nl[nl < 1e-10] = 1.0
    nx3 /= nl; ny3 /= nl; nz3 /= nl

    # ---- 向量化横截面（broadcasting） ----
    offsets = np.array([p[0] for p in PROFILE], dtype=np.float32)  # (VPS,)
    y_offs  = np.array([p[1] for p in PROFILE], dtype=np.float32)  # (VPS,)
    width_scale = path_arr[:, 3] / 10.0                            # (n,)
    scaled_off = width_scale[:, None] * offsets[None, :]           # (n, VPS)

    # (n, 1) + (n, 1) * (1, VPS) → (n, VPS)
    rx = path_arr[:, 0][:, None] + nx_2d[:, None] * scaled_off
    rz = path_arr[:, 1][:, None] + nz_2d[:, None] * scaled_off
    # y = cy + y_off + scaled_off * sin_b
    ry = path_arr[:, 2][:, None] + y_offs[None, :] + scaled_off * sin_b[:, None]

    # ---- UV（横向→U，纵向→V，中线沿路延伸） ----
    dx = np.diff(path_arr[:, 0])
    dz = np.diff(path_arr[:, 1])
    dists = np.sqrt(dx**2 + dz**2)
    cum_len = np.concatenate(([0], np.cumsum(dists)))

    # ★ 横向 → 纹理 U（0.045 / 0.5 / 0.955 对应纹理上的车道横向位置）
    u_coord = (offsets + 5.0) / 10.0 * 0.625
    # ★ 沿路弧长 → 纹理 V（每 4 米重复一次纹理，颗粒细节更清晰）
    v_coord = cum_len / 4.0

    # ---- 颜色：基础色来自 PROFILE，最外侧路肩用地形色覆盖 ----
    base_colors = np.array([p[2] for p in PROFILE], dtype=np.float32)  # (VPS, 3)
    colors_arr = np.broadcast_to(base_colors[None, :, :], (n, VPS, 3)).copy()

    # 最外侧路肩采样地形色（PATH点数 × 4，循环量很小）
    for p_idx in TERRAIN_COLOR_IDX:
        for i in range(n):
            terr_h = get_terrain_height(float(rx[i, p_idx]), float(rz[i, p_idx]))
            slope  = min(60.0, get_terrain_slope_angle(float(rx[i, p_idx]), float(rz[i, p_idx])))
            colors_arr[i, p_idx] = _terrain_color(terr_h, slope)

    # ---- 路缘石红白交替（每 4 个路径截面翻转一次） ----
    CURB_RED_IDX   = {3, 14}   # 左侧路缘石红(3), 右侧路缘石红(14)
    CURB_WHITE_IDX = {4, 13}   # 左侧路缘石白(4), 右侧路缘石白(13)
    stripe_flip = ((np.arange(n) // 4) % 2 == 1)
    for p_idx in CURB_RED_IDX:
        colors_arr[stripe_flip, p_idx] = (0.72, 0.72, 0.68)
    for p_idx in CURB_WHITE_IDX:
        colors_arr[stripe_flip, p_idx] = (0.48, 0.16, 0.13)

    # ★ 碎石带噪声扰动：颗粒感明暗抖动
    def _hash_noise(x, z, seed):
        """基于世界坐标的确定性伪随机"""
        h = math.sin(x * 12.9898 + z * 78.233 + seed) * 43758.5453
        return h - math.floor(h)

    for p_idx in GRAVEL_IDX | GRAVEL_IDX_R:
        for i in range(n):
            noise = _hash_noise(float(rx[i, p_idx]), float(rz[i, p_idx]), 71.0)
            # 石子颗粒感：±15% 明暗抖动
            base = np.array(base_colors[p_idx], dtype=np.float32)
            colors_arr[i, p_idx] = base * (0.85 + noise * 0.30)

    # ★ 接缘带高度微起伏：碎石带/压实土 1~2cm 起伏
    for p_idx in RIPPLE_IDX:
        for i in range(n):
            n_val = _hash_noise(float(rx[i, p_idx]), float(rz[i, p_idx]), 33.0)
            # 在原有高度基础上 ±1.5cm
            ry[i, p_idx] += (n_val - 0.5) * 0.03

    # ---- 组装 verts (n, VPS, 11) ----
    verts_arr = np.empty((n, VPS, 11), dtype=np.float32)
    verts_arr[..., 0] = rx
    verts_arr[..., 1] = ry
    verts_arr[..., 2] = rz
    verts_arr[..., 3] = nx3[:, None]   # 每段所有截面共享同个法线
    verts_arr[..., 4] = ny3[:, None]
    verts_arr[..., 5] = nz3[:, None]
    verts_arr[..., 6] = u_coord[None, :]   # U 是横向（沿 profile 方向）
    verts_arr[..., 7] = v_coord[:, None]   # V 是纵向（沿弧长方向）
    verts_arr[..., 8:11] = colors_arr

    # ★ 碎石带法线粗糙化：向外倾斜 + 随机扰动
    GRAVEL_NORMAL_TILT = 0.35  # 向外倾斜系数
    for p_idx in GRAVEL_IDX | GRAVEL_IDX_R:
        side_sign = -1.0 if p_idx < VPS // 2 else 1.0
        # 混入向外倾斜的法线分量（绕 Y 轴旋转）
        for i in range(n):
            nx_orig = verts_arr[i, p_idx, 3]
            ny_orig = verts_arr[i, p_idx, 4]
            nz_orig = verts_arr[i, p_idx, 5]
            # 倾斜：朝向外侧（法线方向由 nx_2d/nz_2d * side_sign 决定）
            # 但这里不能直接用 nx2[i] 因为它是第 i 个路径截面的法线
            # 使用预计算的 nx_2d, nz_2d 数组（水平法线）
            tilt_x = nx_2d[i] * side_sign
            tilt_z = nz_2d[i] * side_sign
            # 混入倾斜分量 + 随机抖动
            rand_jitter = _hash_noise(float(rx[i, p_idx]), float(rz[i, p_idx]), 99.0) * 0.15
            nx_new = nx_orig * (1.0 - GRAVEL_NORMAL_TILT) + tilt_x * (GRAVEL_NORMAL_TILT + rand_jitter)
            ny_new = ny_orig * (1.0 - GRAVEL_NORMAL_TILT * 0.5)
            nz_new = nz_orig * (1.0 - GRAVEL_NORMAL_TILT) + tilt_z * (GRAVEL_NORMAL_TILT + rand_jitter)
            # 归一化
            nlen = math.sqrt(nx_new*nx_new + ny_new*ny_new + nz_new*nz_new)
            if nlen > 1e-10:
                verts_arr[i, p_idx, 3] = nx_new / nlen
                verts_arr[i, p_idx, 4] = ny_new / nlen
                verts_arr[i, p_idx, 5] = nz_new / nlen

    # ★ 赛道法线微扰：给沥青表面制造微观起伏感
    # 用世界坐标伪随机偏移法线，产生"骨料凸起"的视觉错觉
    for i in range(n):
        for p_idx in range(VPS):
            wx = verts_arr[i, p_idx, 0]
            wz = verts_arr[i, p_idx, 2]
            # 两个不同频率的伪随机扰动（高频 = 骨料，低频 = 路面起伏）
            n1 = _hash_noise(wx, wz, 17.0) - 0.5
            n2 = _hash_noise(wx, wz, 53.0) - 0.5
            # 主要扰动水平分量，Y 分量保持接近 1
            tilt_x = n1 * 0.18 + n2 * 0.08
            tilt_z = n2 * 0.18 + n1 * 0.08
            nx_old = verts_arr[i, p_idx, 3]
            ny_old = verts_arr[i, p_idx, 4]
            nz_old = verts_arr[i, p_idx, 5]
            nx_new = nx_old + tilt_x
            ny_new = ny_old
            nz_new = nz_old + tilt_z
            nlen = math.sqrt(nx_new*nx_new + ny_new*ny_new + nz_new*nz_new)
            if nlen > 1e-8:
                verts_arr[i, p_idx, 3] = nx_new / nlen
                verts_arr[i, p_idx, 4] = ny_new / nlen
                verts_arr[i, p_idx, 5] = nz_new / nlen

    # ---- 向量化三角化 ----
    iz = np.arange(n - 1)
    ix = np.arange(VPS - 1)
    IZ, IX = np.meshgrid(iz, ix, indexing='ij')
    IZ = IZ.ravel(); IX = IX.ravel()
    base  = IZ * VPS + IX
    nbase = (IZ + 1) * VPS + IX
    a = base; b = base + 1; c = nbase + 1; d = nbase

    n_quads = len(a)
    indices = np.empty(n_quads * 6, dtype=np.int32)
    indices[0::6] = a; indices[1::6] = b; indices[2::6] = c
    indices[3::6] = a; indices[4::6] = c; indices[5::6] = d

    verts_flat = verts_arr.reshape(-1, 11)
    result = verts_flat[indices].copy()

    # ---- 沥青厚度偏移 ----
    result[:, 1] += ASPHALT_THICKNESS

    _mountain_road_mesh_cache = result

    # ---- 保存到磁盘缓存（带 path 指纹）----
    if _road_cache_path and len(result) > 0:
        try:
            np.save(_road_cache_path, result)
            print(f"[缓存]  赛道网格已缓存 ({_road_cache_name})")
        except Exception as e:
            print(f"[缓存]  赛道网格缓存失败: {e}")

    print(f"  赛道顶点: {len(result)}")
    return result


# ============================================================
# 八、护栏构建（直接使用 get_akina_path 对齐视觉赛道，彻底解决弯道穿模）
# ============================================================

def _venue_rail_gap(px, pz):
    """护栏豁口判定：点落在起/终点**缓冲带**范围内 → 护栏断开。

    ⚠️ 缓冲带是沿整条服务带（|u| ≤ HL，60m）铺的连续地带，不是只在 u≈0
       开一个 8m 的小口。护栏在 sd=+0.4 ≈ 距中心线 0.35w+0.4，正好落在
       缓冲带（0.35w ~ 0.35w+2.2）中间 —— 不断开就等于横在缓冲带正中间，
       缓冲带看着还是"被隔断的另一块地"。
       只断**场地这一侧**的护栏，对侧不受影响。
    """
    for (cx, cz, fx, fz, rx, rz, base_y, HL, HW_, grade) in _venue_pads():
        u = (px - cx) * fx + (pz - cz) * fz
        v = (px - cx) * rx + (pz - cz) * rz
        if abs(u) < HL + 1.5 and \
           (-HW_ - _VENUE_GAP_ROAD - 0.8) < v < (-HW_ + 0.8):
            return True
    return False


def build_guardrails():
    """护栏：W-beam 连续护栏 — 已缓存（缓存名挂 path 指纹）"""
    return _build_or_load_env("akina_guardrails_v7", _build_guardrails_impl)

def _build_guardrails_impl():
    """
    W-beam 连续护栏 v5。
    ★ v5 关键修复：1:1 真实赛道的 banking 实测仅 ±0.6°（不是设计稿的 ±14°），
      原判据 |bank|>4° 恒假 → **弯道轮廓标一个都没生成**。改用曲率判据。
    v5 新增：弯道护栏的轮胎擦痕（现实榛名山护栏上满是车身刮痕）。
    观感：W 波形横梁、立柱加密、弯道轮廓标右白左黄、擦痕、端头处理。
    """
    path, tangents = get_akina_path(return_tangents=True)[:2]
    n = len(path)
    if n < 2:
        return np.empty((0, 11), dtype=np.float32)

    verts = []
    def quad(p1, p2, p3, p4, n, uv, c):
        verts.append([*p1, *n, *uv[0], *c])
        verts.append([*p2, *n, *uv[1], *c])
        verts.append([*p3, *n, *uv[2], *c])
        verts.append([*p1, *n, *uv[0], *c])
        verts.append([*p3, *n, *uv[2], *c])
        verts.append([*p4, *n, *uv[3], *c])
    def tri(p1, p2, p3, n, uv, c):
        verts.append([*p1, *n, *uv[0], *c])
        verts.append([*p2, *n, *uv[1], *c])
        verts.append([*p3, *n, *uv[2], *c])

    RAIL_UPPER = (0.45, 0.46, 0.48)  # 上横梁亮
    RAIL_LOWER = (0.22, 0.22, 0.24)  # 下横梁暗
    POST = (0.30, 0.30, 0.32)
    POST_TOP = (0.38, 0.38, 0.40)    # 柱顶略亮
    SCUFF = (0.62, 0.60, 0.56)       # 轮胎擦痕：磨掉镀锌层露出的灰白金属
    GH = 0.75  # 护栏高度（W-beam 标准高度，原 0.65 → 0.75）

    # ★ 曲率判据（banking 在这条真实赛道上只有 ±0.6°，不能用来判弯道）
    curv = _compute_path_curvature_signed(path)

    def _dsquad(p1, p2, p3, p4, nn, c):
        """双面 quad：竖直薄片无法保证绕序，正反都发一份。"""
        for pp in (p1, p2, p3, p1, p3, p4):
            verts.append([*pp, *nn, 0, 0, *c])
        for pp in (p1, p4, p3, p1, p3, p2):
            verts.append([*pp, *nn, 0, 0, *c])

    # ------------------------------------------------------------
    # 修复 B（治标）：用中心差分重建平滑切线，替代逐点样条导数。
    # 窗口 ±2 点覆盖约 ±1m，抹掉单点噪声又不把真弯道抹直。
    # 样条导数在 GPS 密簇处方向来回摆，中心差分度量真实前进方向，
    # 法线因此不再抖动，护栏不会翻到路内侧。
    # ------------------------------------------------------------
    def _smooth_tangent(i):
        i0 = max(0, i - 2)
        i1 = min(n - 1, i + 2)
        dxs = path[i1][0] - path[i0][0]
        dzs = path[i1][1] - path[i0][1]
        L = math.hypot(dxs, dzs)
        if L < 1e-9:
            return tangents[i]
        # y 分量从样条切线保留（坡度），水平方向用差分
        return (dxs / L, tangents[i][1], dzs / L)

    _tan_sm = [_smooth_tangent(i) for i in range(n)]

    # ------------------------------------------------------------
    # 修复 C（兜底）：_rail_point 用 signed_distance 梯度方向向外走，
    # 保证护栏落在 sd=+0.4 等值线上，不依赖任何可能有误的逐点法线。
    # ------------------------------------------------------------
    def _rail_point(cx, cz, rx, rz, want_sd=0.4):
        """从 (cx,cz) 沿法线方向走到 signed_distance == want_sd；
        若起点已在路上（法线翻转的症状），用 sd 场梯度方向向外推，
        保证护栏绝不落在赛道内。"""
        px, pz = cx, cz
        for _ in range(6):
            d = signed_distance(px, pz) - want_sd
            if d >= -0.05:
                break
            e = 0.5
            gx = (signed_distance(px + e, pz) - signed_distance(px - e, pz)) / (2 * e)
            gz = (signed_distance(px, pz + e) - signed_distance(px, pz - e)) / (2 * e)
            gl = math.hypot(gx, gz) or 1e-6
            px += gx / gl * max(-d, 0.1)
            pz += gz / gl * max(-d, 0.1)
        return px, pz

    # 分别处理左侧和右侧护栏
    for side_name, sign in [("左侧", 1.0), ("右侧", -1.0)]:
        prev_valid = False  # 跟踪连续段，用于端头处理
        first_i = None
        for i in range(0, n - 1, 2):  # 立柱加密：步长 3→2（约 2m 一柱）
            j = min(i + 1, n - 1)

            # 提取中心点数据和切线（★ 修复 B：使用平滑切线 _tan_sm）
            cx1, cz1, cy1, w1, bank1 = path[i]
            cx2, cz2, cy2, w2, bank2 = path[j]
            tx1, ty1, tz1 = _tan_sm[i]
            tx2, ty2, tz2 = _tan_sm[j]

            # ---- 计算水平法线 ----
            nx1_2d = -tz1; nz1_2d = tx1
            nx2_2d = -tz2; nz2_2d = tx2
            len1 = math.hypot(nx1_2d, nz1_2d)
            if len1 > 1e-10: nx1_2d /= len1; nz1_2d /= len1
            len2 = math.hypot(nx2_2d, nz2_2d)
            if len2 > 1e-10: nx2_2d /= len2; nz2_2d /= len2

            hw1 = w1 * 0.5; hw2 = w2 * 0.5
            sin_b1 = math.sin(bank1); sin_b2 = math.sin(bank2)

            x1 = cx1 + nx1_2d * hw1 * sign
            z1 = cz1 + nz1_2d * hw1 * sign
            y1 = cy1 - (hw1 * sign) * sin_b1
            x2 = cx2 + nx2_2d * hw2 * sign
            z2 = cz2 + nz2_2d * hw2 * sign
            y2 = cy2 - (hw2 * sign) * sin_b2

            dx = x2 - x1; dz = z2 - z1
            length = math.hypot(dx, dz)
            if length < 0.001:
                prev_valid = False
                continue

            rx1 = nx1_2d * sign; rz1 = nz1_2d * sign
            rx2 = nx2_2d * sign; rz2 = nz2_2d * sign

            # ★ 修复 C：用 signed_distance 等值线定位护栏，非法线偏移
            sx1, sz1 = _rail_point(x1, z1, rx1, rz1, 0.4)
            sy1 = get_ground_height(sx1, sz1) + 0.05
            sx2, sz2 = _rail_point(x2, z2, rx2, rz2, 0.4)
            sy2 = get_ground_height(sx2, sz2) + 0.05

            # ★ 起终点引道豁口：护栏在场地入口处整段断开
            #   （段长 ~20m，豁口比引道宽是真实的"开口渐变段"造型）
            if _venue_rail_gap(0.5 * (sx1 + sx2), 0.5 * (sz1 + sz2)):
                prev_valid = False
                continue

            # ---- 端头处理：段起始处外张 ----
            end_offset = 0.0
            end_drop = 0.0
            if not prev_valid:
                # 段起点：立柱外偏 0.6m，横梁下探 0.3m
                first_i = i
                end_offset = 0.6
                end_drop = 0.3
            # 段终点在下一轮判断（无法提前知道），在循环外补

            # ---- 立柱（含端头外偏）----
            for idx, (px, pz, py) in enumerate([(sx1, sz1, sy1), (sx2, sz2, sy2)]):
                offset_i = end_offset if idx == 0 else 0.0
                px_o = px + rx1 * offset_i
                pz_o = pz + rz1 * offset_i
                py_o = py - end_drop if idx == 0 else py

                hw2_p = 0.04
                hh = GH * 0.5
                quad((px_o - hw2_p, py_o, pz_o - hw2_p),
                     (px_o + hw2_p, py_o, pz_o - hw2_p),
                     (px_o + hw2_p, py_o + hh, pz_o - hw2_p),
                     (px_o - hw2_p, py_o + hh, pz_o - hw2_p),
                     (0, 0, 1), [(0,0)]*4, POST)
                quad((px_o - hw2_p, py_o + hh, pz_o - hw2_p),
                     (px_o + hw2_p, py_o + hh, pz_o - hw2_p),
                     (px_o + hw2_p, py_o + hh, pz_o + hw2_p),
                     (px_o - hw2_p, py_o + hh, pz_o + hw2_p),
                     (0, 1, 0), [(0,0)]*4, POST_TOP)
                quad((px_o - hw2_p, py_o, pz_o + hw2_p),
                     (px_o + hw2_p, py_o, pz_o + hw2_p),
                     (px_o + hw2_p, py_o + hh, pz_o + hw2_p),
                     (px_o - hw2_p, py_o + hh, pz_o + hw2_p),
                     (0, 0, -1), [(0,0)]*4, POST)

                # ---- 柱底接触阴影 ----
                sr = 0.08
                for s in range(6):
                    a1 = 2*math.pi*s/6
                    a2 = 2*math.pi*(s+1)/6
                    tri((px_o, py_o+0.005, pz_o),
                        (px_o+sr*math.cos(a1), py_o+0.005, pz_o+sr*math.sin(a1)),
                        (px_o+sr*math.cos(a2), py_o+0.005, pz_o+sr*math.sin(a2)),
                        (0,1,0), [(0,0)]*3, (0.0,0.0,0.0))

            # ---- W 波形横梁（5 点折线，中央内凹的 W 谷）----
            for y_off, rail_col in [(GH * 0.28, RAIL_LOWER), (GH * 0.72, RAIL_UPPER)]:
                mx1, mz1 = (sx1 + sx2) * 0.5, (sz1 + sz2) * 0.5
                avg_rx = (rx1 + rx2) * 0.5
                avg_rz = (rz1 + rz2) * 0.5
                # 1/4 与 3/4 点折出 W 谷
                qx1 = sx1 + (mx1 - sx1) * 0.5 - avg_rx * 0.035
                qz1 = sz1 + (mz1 - sz1) * 0.5 - avg_rz * 0.035
                qx2 = sx2 + (mx1 - sx2) * 0.5 + avg_rx * 0.035
                qz2 = sz2 + (mz1 - sz2) * 0.5 + avg_rz * 0.035
                # 端头横梁下探
                drop1 = end_drop if not prev_valid else 0.0
                poly = [(sx1, sy1 + y_off - drop1, sz1),
                        (qx1, sy1 + y_off + 0.025, qz1),
                        (mx1, sy1 + y_off - 0.01, mz1),
                        (qx2, sy2 + y_off + 0.025, qz2),
                        (sx2, sy2 + y_off, sz2)]
                for a in range(4):
                    (x1b, y1b, z1b), (x2b, y2b, z2b) = poly[a], poly[a+1]
                    quad((x1b, y1b, z1b), (x2b, y2b, z2b),
                         (x2b, y2b + 0.32, z2b), (x1b, y1b + 0.32, z1b),
                         (0, 1, 0), [(0,0)]*4, rail_col)

            # ---- ★ 弯道护栏擦痕：轮胎刮掉镀锌层露出的灰白金属条 ----
            # 现实榛名山护栏外侧满是黑灰刮痕，是山路最有辨识度的"战损"细节。
            if abs(curv[i]) > 0.008 and (i // 2) % 3 == 0:
                rnd = (math.sin(sx1 * 12.9898 + sz1 * 78.233 + 5.0) * 43758.5453)
                rnd -= math.floor(rnd)
                if rnd > 0.42:
                    off = 0.035                      # 略微挑出横梁面，防 Z-fighting
                    yb_a = sy1 + GH * 0.62
                    yb_b = sy2 + GH * 0.62
                    sh_h = 0.09 + rnd * 0.11
                    sa = (sx1 - rx1 * off, yb_a, sz1 - rz1 * off)
                    sb = (sx2 - rx2 * off, yb_b, sz2 - rz2 * off)
                    _dsquad(sa, sb,
                            (sb[0], sb[1] + sh_h, sb[2]),
                            (sa[0], sa[1] + sh_h, sa[2]),
                            (-rx1, 0.0, -rz1), SCUFF)

            # ---- 轮廓标：仅弯道放置，右白左黄（★ 判据从 banking 改为曲率）----
            if abs(curv[i]) > 0.006 and (i // 2) % 6 == 0:   # 约 6m 一根
                rf_y = sy1 + GH * 0.5
                rf_h = 0.05
                # 右侧白色、左侧橙黄（日规：沿行车方向）
                if side_name == "右侧":
                    rf_col = (0.92, 0.92, 0.88)   # 白
                else:
                    rf_col = (0.95, 0.65, 0.10)   # 橙黄
                quad(
                    (sx1 - rx1 * 0.001, rf_y - rf_h, sz1 - rz1 * 0.001),
                    (sx1 - rx1 * 0.001 + rx1 * 0.01, rf_y - rf_h, sz1),
                    (sx1 - rx1 * 0.001 + rx1 * 0.01, rf_y + rf_h, sz1),
                    (sx1 - rx1 * 0.001, rf_y + rf_h, sz1 - rz1 * 0.001),
                    (rx1 * -1, 0, rz1 * -1),
                    [(0,0)]*4, rf_col)

            prev_valid = True

    return np.array(verts, dtype=np.float32)


# ============================================================
# 九、混凝土路障（Jersey Barrier）
# ============================================================

def build_concrete_barriers():
    """混凝土路障（石墩子）— 2026-09-20 起**弯道不再放置**，返回空数组。

    原因：石墩子原先只生成在急弯外侧，与"侧沟移到弯道内侧"的新规则冲突
    （外侧重物 + 内侧沟 = 弯道两侧都堵死，看不出路肩层次）。

    保留空实现只为兼容 main.py 与 _scratch 冒烟脚本的既有调用；渲染侧按
    cbcnt == 0 自动跳过（subset 不注册、draw 不调用）。
    要恢复就把 _build_concrete_barriers_impl 从 git 历史里捞回来。
    """
    return np.empty((0, 11), dtype=np.float32)


# ============================================================
# 十、挖方挡土墙（石笼挡墙）
# ============================================================

def build_retaining_walls():
    """挖方挡土墙 — 已缓存（缓存名挂 path 指纹）"""
    return _build_or_load_env("akina_retaining_v7", _build_retaining_walls_impl)

def _build_retaining_walls_impl():
    """挖方挡土墙（v3）：
    - 墙顶跟随地形表面，封顶 2.8m（不再脱离坡面）
    - 分层砌石替代整面平板（每 0.45m 一层、逐层明暗抖动）
    - 压顶石略外挑深色收顶
    - 检测判据改用 get_ground_height 坡脚高差 > 0.9m

    顶点格式: [pos3, nrm3, uv2, col3]（与护栏一致，无需新着色器）
    """
    path, tangents = get_akina_path(return_tangents=True)[:2]
    n = len(path)
    FLAT_END = _TERRAIN_FLAT_END          # 5.0
    MIN_H, EMBED = 2.2, 0.5               # 保留旧常量，实际判据用 get_ground_height
    WALL_H_MAX = 6.0
    WALL_THICK_BOT = 0.35                 # 梯形断面底部外扩

    verts = []
    def _wall_quad(p1, p2, p3, p4, nn, col):
        verts.append([*p1, *nn, 0, 0, *col])
        verts.append([*p2, *nn, 0, 0, *col])
        verts.append([*p3, *nn, 0, 0, *col])
        verts.append([*p1, *nn, 0, 0, *col])
        verts.append([*p3, *nn, 0, 0, *col])
        verts.append([*p4, *nn, 0, 0, *col])

    STONE = (0.26, 0.24, 0.22)            # 石笼基色（原 0.34→压暗，抗环境光增益）
    COPING = (0.15, 0.14, 0.13)           # 压顶石近黑

    def _flush_segment(side, pts):
        """把连续激活段输出为一段分层石笼挡墙"""
        for a in range(0, len(pts) - 1, 2):
            b = min(a + 2, len(pts) - 1)
            (x1, z1, sh1, wh1), (x2, z2, sh2, wh2) = pts[a], pts[b]
            # 墙面法线（指向赛道内侧）
            dx, dz = x2 - x1, z2 - z1
            L = math.hypot(dx, dz) or 1e-6
            tx, tz = dx / L, dz / L
            # side=1 左墙法线朝右；side=-1 右墙法线朝左
            inn = (tz * side, -tx * side)

            # ★ 墙顶 = 墙自身位置处的地形表面高 + 0.25m 露头，绝对高度封顶 2.8m
            gh1 = get_ground_height(x1, z1)
            gh2 = get_ground_height(x2, z2)
            y_b1, y_b2 = sh1 - EMBED, sh2 - EMBED
            y_t1 = min(max(sh1 + 0.6, gh1 + 0.25), y_b1 + 2.8)
            y_t2 = min(max(sh2 + 0.6, gh2 + 0.25), y_b2 + 2.8)
            if y_t1 - y_b1 < 0.5 or y_t2 - y_b2 < 0.5:
                continue   # 太矮不砌

            # 梯形断面控制点
            v00 = (x1 - inn[0]*WALL_THICK_BOT, y_b1, z1 - inn[1]*WALL_THICK_BOT)
            v01 = (x1,                          y_b1, z1)
            v02 = (x1,                          y_t1, z1)
            v03 = (x1 - inn[0]*WALL_THICK_BOT, y_t1, z1 - inn[1]*WALL_THICK_BOT)
            w00 = (x2 - inn[0]*WALL_THICK_BOT, y_b2, z2 - inn[1]*WALL_THICK_BOT)
            w01 = (x2,                          y_b2, z2)
            w02 = (x2,                          y_t2, z2)
            w03 = (x2 - inn[0]*WALL_THICK_BOT, y_t2, z2 - inn[1]*WALL_THICK_BOT)

            n_in3 = (inn[0], 0.0, inn[1])       # 墙内侧法线（面向赛道）
            n_out3 = (-inn[0], 0.0, -inn[1])    # 墙外侧法线（背向赛道）

            # ---- 分层砌石（_coursed）----
            def _coursed(xa, za, y_ba, y_ta, xb, zb, y_bb, y_tb, nn, base):
                """沿高度切 0.45m 层，逐层抖动颜色"""
                y_c = y_ba
                ci = 0
                while y_c < y_ta - 0.03:
                    y_n = min(y_c + 0.45, y_ta)
                    # 两端高度不同时插值对齐
                    t_b = (y_c - y_ba) / max(y_ta - y_ba, 1e-6)
                    y_c2 = y_bb + t_b * (y_tb - y_bb)
                    t_n = (y_n - y_ba) / max(y_ta - y_ba, 1e-6)
                    y_n2 = y_bb + t_n * (y_tb - y_bb)
                    # 明暗抖动（基于位置哈希）
                    f = 0.74 + 0.20 * (((a * 7 + ci) * 7919 % 13) / 13.0)
                    col = tuple(min(1.0, c * f) for c in base)
                    _wall_quad((xa, y_c, za), (xb, y_c2, zb),
                               (xb, y_n2, zb), (xa, y_n, za), nn, col)
                    y_c = y_n
                    ci += 1

            # 内立面（面向赛道）
            _coursed(x1, z1, y_b1, y_t1, x2, z2, y_b2, y_t2, n_in3, STONE)
            # 外立面（背向赛道）
            _coursed(x1 - inn[0]*WALL_THICK_BOT, z1 - inn[1]*WALL_THICK_BOT,
                     y_b1, y_t1,
                     x2 - inn[0]*WALL_THICK_BOT, z2 - inn[1]*WALL_THICK_BOT,
                     y_b2, y_t2, n_out3, STONE)

            # 底面
            _wall_quad(v00, w00, w01, v01, (0, -1, 0), STONE)

            # ★ 压顶石（略外挑 5cm + 深色）
            CPT = WALL_THICK_BOT + 0.05
            c00 = (x1 - inn[0]*CPT, y_t1, z1 - inn[1]*CPT)
            c01 = (x1 + inn[0]*0.05, y_t1, z1 + inn[1]*0.05)
            c02 = (x1 + inn[0]*0.05, y_t1 + 0.10, z1 + inn[1]*0.05)
            c03 = (x1 - inn[0]*CPT, y_t1 + 0.10, z1 - inn[1]*CPT)
            d00 = (x2 - inn[0]*CPT, y_t2, z2 - inn[1]*CPT)
            d01 = (x2 + inn[0]*0.05, y_t2, z2 + inn[1]*0.05)
            d02 = (x2 + inn[0]*0.05, y_t2 + 0.10, z2 + inn[1]*0.05)
            d03 = (x2 - inn[0]*CPT, y_t2 + 0.10, z2 - inn[1]*CPT)
            # 压顶石顶面
            _wall_quad(c03, d03, d02, c02, (0, 1, 0), COPING)
            # 压顶石内侧立面
            _wall_quad(c01, d01, d02, c02, n_in3, COPING)
            # 压顶石外侧立面（略外挑 5cm）
            _wall_quad(c00, d00, d03, c03, n_out3, COPING)

            # 端面
            _wall_quad(v00, v01, v02, v03, (tx, 0, tz), STONE)
            _wall_quad(w00, w01, w02, w03, (-tx, 0, -tz), STONE)

    # ---- 逐点评估两侧是否需要墙（v3：get_ground_height 坡脚高差 >0.9m）----
    retaining_pts = []   # 收集所有墙段点坐标（用于树木避让）
    for side in (1.0, -1.0):
        active = []
        for i in range(n):
            x, z, y, w, bank = path[i]
            tx, ty, tz = tangents[i]
            # 2D 法线
            nx2d, nz2d = -tz, tx
            L = math.hypot(nx2d, nz2d) or 1e-6
            nx2d, nz2d = nx2d / L, nz2d / L

            # 坡脚点（压平带外的挖方坡脚，见 cut_foot_sample）
            fx, fz, rise_terrain = cut_foot_sample(x, z, nx2d, nz2d, w, side)

            # 路缘基准高（与护栏同源）
            hw = w * 0.5
            ex = x + nx2d * hw * side
            ez = z + nz2d * hw * side
            edge_h = get_ground_height(ex, ez)

            # ★ 坡脚地形真的比路肩高才砌
            if rise_terrain > _CUT_MIN_RISE:
                active.append((fx, fz, edge_h, rise_terrain))
                retaining_pts.append((fx, fz))
            else:
                if len(active) >= 4:
                    _flush_segment(side, active)
                active = []
        if len(active) >= 4:
            _flush_segment(side, active)

    result = np.array(verts, dtype=np.float32)
    print(f"[挡土墙] 顶点: {len(result)}, v4 分层砌石 + 墙顶贴地形")
    return result


# ============================================================
# v3 树木几何生成器 — 多层锥形树冠（杉树真实形态）
# 取代 v2 的变形 icosphere 叶簇方案
# 核心改进：
#   1. 树干：渐变锥形圆柱（底粗顶细），保留
#   2. 树枝：自然分叉向外伸展，保留
#   3. 树冠：多层锥形叠加（下大上小），替代球形簇
#   4. 着色：顶点亮/底边暗的双色渐变，有立体明暗
# ============================================================

def _make_crown_layer(cx, cy, cz, radius, height,
                      color_top, color_bottom,
                      segments, rng, droop=0.10, r_jitter=0.08):
    """
    生成杉树树冠的一个锥形层。

    形状：顶点位于 (cx, cy+height, cz)，
         底边环位于 (cx, cy, cz) 附近，半径 radius。

    改进点：
      - 底边顶点的半径有 r_jitter 的随机抖动 → 打破完美圆形
      - 底边顶点向下 droop*radius 的下垂 → 模拟枝条末端自然下垂
      - 顶点位置略有偏移 → 每层看起来不完全对称
      - 顶点色（受光）与底边色（阴影）分离 → 有立体明暗

    返回 [x,y,z, nx,ny,nz, u,v, r,g,b] × N
    """
    result = []

    # 顶点位置：略微偏移，让每层不完全对称
    apex_x = cx + (rng.random() - 0.5) * radius * 0.06
    apex_y = cy + height
    apex_z = cz + (rng.random() - 0.5) * radius * 0.06

    # 预计算底边顶点（相邻三角形共享，避免裂缝）
    bottom_pts = []
    for s in range(segments):
        a = 2 * math.pi * s / segments
        r = radius * (1.0 - r_jitter * rng.random())   # 半径抖动
        dy = -droop * radius * rng.random()            # 向下垂
        bx = cx + r * math.cos(a)
        by = cy + dy
        bz = cz + r * math.sin(a)
        bottom_pts.append((bx, by, bz, a))

    # 锥面坡度因子（用于计算面法线）
    slope = height / max(radius, 0.01)

    # 生成三角面：每段一个三角形（顶点 + 相邻两底点）
    for s in range(segments):
        s2 = (s + 1) % segments
        b1x, b1y, b1z, a1 = bottom_pts[s]
        b2x, b2y, b2z, a2 = bottom_pts[s2]

        # 朝外法线：沿径向 + 向上倾斜
        mid_a = (a1 + a2) * 0.5
        nx = math.cos(mid_a) * slope
        nz = math.sin(mid_a) * slope
        ny = 1.0
        nl = math.hypot(math.hypot(nx, ny), nz) or 1.0
        normal = (nx / nl, ny / nl, nz / nl)

        # 三角形：顶点(亮) → 底点1(暗) → 底点2(暗)
        result.append([apex_x, apex_y, apex_z, *normal, 0.0, 0.0,
                       color_top[0], color_top[1], color_top[2]])
        result.append([b1x, b1y, b1z, *normal, 0.0, 0.0,
                       color_bottom[0], color_bottom[1], color_bottom[2]])
        result.append([b2x, b2y, b2z, *normal, 0.0, 0.0,
                       color_bottom[0], color_bottom[1], color_bottom[2]])

    return result


def _make_tapered_cylinder(x0, y0, z0, x1, y1, z1,
                           radius_bottom, radius_top,
                           segments, color_base, color_top=None):
    """
    生成渐变粗细的圆柱体（锥形圆柱），支持任意朝向。
    用于树干（底粗顶细）和树枝。
    """
    if color_top is None:
        color_top = color_base

    dx = x1 - x0; dy = y1 - y0; dz = z1 - z0
    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    if length < 1e-8:
        return []

    # 方向向量
    up = (dx/length, dy/length, dz/length)

    # 构建正交基
    if abs(up[1]) < 0.99:
        ref = (0, 1, 0)
    else:
        ref = (1, 0, 0)

    rx = up[1]*ref[2] - up[2]*ref[1]
    ry = up[2]*ref[0] - up[0]*ref[2]
    rz = up[0]*ref[1] - up[1]*ref[0]
    rl = math.sqrt(rx*rx + ry*ry + rz*rz)
    rx /= rl; ry /= rl; rz /= rl

    fx = ry*up[2] - rz*up[1]
    fy = rz*up[0] - rx*up[2]
    fz = rx*up[1] - ry*up[0]

    result = []
    taper_sin = math.sin(math.atan2(radius_bottom - radius_top, length))

    for s in range(segments):
        a1 = 2 * math.pi * s / segments
        a2 = 2 * math.pi * (s + 1) / segments
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)

        # 底部圆
        b1x = x0 + (rx*c1 + fx*s1) * radius_bottom
        b1y = y0 + (ry*c1 + fy*s1) * radius_bottom
        b1z = z0 + (rz*c1 + fz*s1) * radius_bottom
        b2x = x0 + (rx*c2 + fx*s2) * radius_bottom
        b2y = y0 + (ry*c2 + fy*s2) * radius_bottom
        b2z = z0 + (rz*c2 + fz*s2) * radius_bottom

        # 顶部圆
        t1x = x1 + (rx*c1 + fx*s1) * radius_top
        t1y = y1 + (ry*c1 + fy*s1) * radius_top
        t1z = z1 + (rz*c1 + fz*s1) * radius_top
        t2x = x1 + (rx*c2 + fx*s2) * radius_top
        t2y = y1 + (ry*c2 + fy*s2) * radius_top
        t2z = z1 + (rz*c2 + fz*s2) * radius_top

        # 侧面法线
        mx = (c1 + c2) * 0.5
        mz = (s1 + s2) * 0.5
        wnx = rx*mx + fx*mz
        wny = ry*mx + fy*mz + taper_sin
        wnz = rz*mx + fz*mz
        nl = math.sqrt(wnx*wnx + wny*wny + wnz*wnz)
        if nl > 1e-8:
            wnx /= nl; wny /= nl; wnz /= nl

        # 两个三角形
        for tri_pts in [
            [(b1x,b1y,b1z), (b2x,b2y,b2z), (t2x,t2y,t2z)],
            [(b1x,b1y,b1z), (t2x,t2y,t2z), (t1x,t1y,t1z)],
        ]:
            for k, pt in enumerate(tri_pts):
                t_blend = 0.0 if k < 2 else 1.0
                col = (
                    color_base[0] * (1-t_blend) + color_top[0] * t_blend,
                    color_base[1] * (1-t_blend) + color_top[1] * t_blend,
                    color_base[2] * (1-t_blend) + color_top[2] * t_blend,
                )
                result.append([pt[0], pt[1], pt[2], wnx, wny, wnz, 0, 0, col[0], col[1], col[2]])

    return result


def _gen_tree_geometry(th, cr, ch, col, trunk_seg=8, crown_detail=1,
                       n_branches=3, n_clusters=6):
    """
    生成逼真的杉树几何体（v3：多层锥形树冠）。

    与 v2 的差异：
      - 弃用「变形 icosphere 叶簇」（顶点独立扰动会破碎成多边形碎片）
      - 改用「多层锥形叠加」：下大上小、层间重叠、边缘轻微抖动
      - 顶部汇聚成一个尖锥，符合针叶树自然形态

    参数保持与 v2 兼容（th/cr/ch/col/trunk_seg/crown_detail/n_branches/n_clusters）
    """
    all_verts = []
    rng = random.Random(hash((round(th, 2), round(cr, 2), round(ch, 2))) & 0xFFFFFFFF)

    BARK_BASE = (0.20, 0.12, 0.05)   # 树干底部暗褐色
    BARK_TOP  = (0.30, 0.20, 0.08)   # 树干顶部稍亮

    # ---- 1. 锥形树干（底粗顶细）----
    trunk_base_r = max(0.10, 0.14 * cr)
    trunk_top_r  = trunk_base_r * 0.35
    all_verts.extend(_make_tapered_cylinder(
        0, 0, 0, 0, th, 0,
        trunk_base_r, trunk_top_r,
        trunk_seg, BARK_BASE, BARK_TOP
    ))

    # ---- 2. 树枝（可选，向外伸展的粗枝条，从树干伸出）----
    if n_branches > 0:
        for bi in range(n_branches):
            h_frac = (0.30 + 0.60 * (bi / max(1, n_branches - 1))) if n_branches > 1 else 0.6
            branch_y = th * h_frac

            angle = (2 * math.pi * bi / max(1, n_branches)) + rng.uniform(-0.3, 0.3)
            branch_len = cr * rng.uniform(0.35, 0.55)
            branch_pitch = rng.uniform(0.2, 0.5)

            end_x = math.cos(angle) * branch_len
            end_y = branch_y + branch_len * math.sin(branch_pitch)
            end_z = math.sin(angle) * branch_len

            t_frac = h_frac
            trunk_r_at = trunk_base_r * (1 - t_frac) + trunk_top_r * t_frac
            start_x = math.cos(angle) * trunk_r_at * 0.7
            start_z = math.sin(angle) * trunk_r_at * 0.7

            br_base_r = max(0.03, trunk_r_at * 0.45)
            br_top_r  = br_base_r * 0.2

            all_verts.extend(_make_tapered_cylinder(
                start_x, branch_y, start_z,
                end_x, end_y, end_z,
                br_base_r, br_top_r,
                max(4, trunk_seg - 3),
                BARK_BASE, BARK_TOP
            ))

    # ---- 3. 多层锥形树冠 ----
    # 层数与分段数由 n_clusters 映射（保持旧接口不变）
    if n_clusters >= 6:      # LOD0
        n_layers, seg = 6, 12
    elif n_clusters >= 4:    # LOD1
        n_layers, seg = 4, 8
    else:                    # LOD2
        n_layers, seg = 2, 6

    # 树冠范围
    crown_start   = th * 0.35           # 树冠从树干 35% 高度开始
    crown_top_y   = th + ch             # 树冠最高点
    crown_total_h = crown_top_y - crown_start
    layer_spacing = crown_total_h / n_layers

    base_col = col

    for layer_i in range(n_layers):
        t = layer_i / max(1, n_layers - 1)     # 0=底 → 1=顶

        layer_cy = crown_start + layer_spacing * layer_i

        if layer_i == n_layers - 1:
            # 顶层：小半径，直接汇聚到树顶
            layer_r     = max(cr * 0.08, 0.25)
            this_height = (th + ch) - layer_cy
        else:
            # 中层/底层：半径由大到小，层高略大于间距形成重叠
            layer_r     = cr * (1.0 - t * 0.85)
            this_height = layer_spacing * 1.35

        # 颜色：顶部受光更亮，底部阴影更暗（形成纵向明暗渐变）
        brightness = 0.65 + 0.55 * t
        layer_col_top = tuple(
            max(0.0, min(1.0, c * brightness * 1.08)) for c in base_col
        )
        layer_col_bottom = tuple(
            max(0.0, min(1.0, c * brightness * 0.60)) for c in base_col
        )

        all_verts.extend(_make_crown_layer(
            0, layer_cy, 0,
            layer_r, this_height,
            layer_col_top, layer_col_bottom,
            segments=seg, rng=rng,
            droop=0.10,       # 底边下垂幅度（半径的 10%）
            r_jitter=0.08,    # 底边半径抖动幅度
        ))

    return np.array(all_verts, dtype=np.float32)


# ============================================================
# 3A 式针叶树 —— 枝干骨架 + 小枝面片（sprig cards）
# ============================================================
# 3A 游戏的针叶树**从来不是把整棵树冠画进一张贴图**，流程是：
#   1) 枝干骨架是真实几何（SpeedTree 建模，或程序生成）；
#   2) 美术/照片扫描只做"一根针叶小枝"（sprig，0.5~1.5m 的带针小枝）贴图；
#   3) 沿骨架撒几百上千个小面片，每片面片采一张小枝贴图（随机翻转/旋转）；
#   4) 树冠轮廓由枝干摆放"长"出来，冠内体积感靠烘进顶点色的 AO。
# 旧实现把整棵树冠画进一张 1024 贴图再糊到 6 张 4m×10m 全高竖直卡片上：
# 26 圈×3~4 个簇天然排成 3~4 条从底到顶的放射臂 + 巨型竖直卡片
# → 看起来像"海带"就是这两件事叠加的结果。
# ------------------------------------------------------------
# 图集布局（build_foliage_textures 与 _sprig_cell_uv 必须一致）：
#   v <  _CROWN_UV_V0 : 不透明树皮带（RGB=白，交给顶点色着色）
#   v >= _CROWN_UV_V0 : _SPRIG_COLS × _SPRIG_ROWS 株针叶小枝
# ------------------------------------------------------------
_BARK_UV_V = 0.03      # 树皮在图集里的采样 v
_CROWN_UV_V0 = 0.06    # 树冠（小枝）区域起始 v，以下为树皮带
_SPRIG_COLS = 2
_SPRIG_ROWS = 2
# cell 的纵横比（texel 空间）。面片宽高必须按它匹配，否则小枝会被拉伸
_SPRIG_CELL_ASPECT = (1.0 / _SPRIG_COLS) / ((1.0 - _CROWN_UV_V0) / _SPRIG_ROWS)


def _sprig_cell_uv(ix, iy):
    """第 (ix, iy) 株小枝在图集里的 UV 窗口 (u0, v0, uw, vh)。"""
    vh = (1.0 - _CROWN_UV_V0) / _SPRIG_ROWS
    return (ix / _SPRIG_COLS, _CROWN_UV_V0 + iy * vh,
            1.0 / _SPRIG_COLS, vh)


def _make_sprig_card(center, u_axis, v_axis, half_u, half_v,
                     cell_ix, cell_iy, color, mirror=False):
    """
    生成一片小枝面片（2 三角形，6 顶点）。

    顶点格式（14 float）：[pos3, nrm3, uv2, col3, tan3]
      - 法线 = 卡片**真实面法线**（不混 up：叶片着色阶段自己往上混，
        而切线空间必须用纯面法线构建，倾斜卡片才不会扭曲）
      - 切线 = u 轴（小枝的茎沿枝干方向），供叶片法线贴图构建 TBN
      - mirror 翻转 u，让同一株小枝以镜像出现，打破重复感
    """
    ux, uy, uz = u_axis
    vx, vy, vz = v_axis
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    _l = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    n = (nx / _l, ny / _l, nz / _l)

    sgn = -1.0 if mirror else 1.0
    u0, v0, uw, vh = _sprig_cell_uv(cell_ix, cell_iy)
    u_lo, u_hi = (u0 + uw, u0) if mirror else (u0, u0 + uw)
    uv = [(u_lo, v0), (u_hi, v0), (u_hi, v0 + vh), (u_lo, v0 + vh)]
    cx, cy, cz = center
    hx, hy, hz = ux * half_u, uy * half_u, uz * half_u
    wx, wy, wz = vx * half_v, vy * half_v, vz * half_v
    p0 = (cx - hx - wx, cy - hy - wy, cz - hz - wz)   # -u -v
    p1 = (cx + hx - wx, cy + hy - wy, cz + hz - wz)   # +u -v
    p2 = (cx + hx + wx, cy + hy + wy, cz + hz + wz)   # +u +v
    p3 = (cx - hx + wx, cy - hy + wy, cz - hz + wz)   # -u +v
    verts = []
    for pi, uvi in zip((p0, p1, p2, p0, p2, p3),
                       (uv[0], uv[1], uv[2], uv[0], uv[2], uv[3])):
        verts.append([pi[0], pi[1], pi[2], n[0], n[1], n[2], uvi[0], uvi[1],
                      color[0], color[1], color[2], ux * sgn, uy * sgn, uz * sgn])
    return verts


def _rot2(x, y, a):
    c, s = math.cos(a), math.sin(a)
    return x * c - y * s, x * s + y * c


def _bez3(p0, pc, p1, t):
    """三维二次贝塞尔。"""
    mt = 1.0 - t
    return (mt * mt * p0[0] + 2 * mt * t * pc[0] + t * t * p1[0],
            mt * mt * p0[1] + 2 * mt * t * pc[1] + t * t * p1[1],
            mt * mt * p0[2] + 2 * mt * t * pc[2] + t * t * p1[2])


def _bez3_dir(p0, pc, p1, t):
    """三维二次贝塞尔的单位切向。"""
    mt = 1.0 - t
    dx = 2 * mt * (pc[0] - p0[0]) + 2 * t * (p1[0] - pc[0])
    dy = 2 * mt * (pc[1] - p0[1]) + 2 * t * (p1[1] - pc[1])
    dz = 2 * mt * (pc[2] - p0[2]) + 2 * t * (p1[2] - pc[2])
    l = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    return (dx / l, dy / l, dz / l)


def _bez_points(p0, pc, p1, n):
    """二次贝塞尔采样 → [(x, y, t), ...]（n 个点）。"""
    ts = np.linspace(0.0, 1.0, n)
    mt = 1.0 - ts
    bx = mt * mt * p0[0] + 2 * mt * ts * pc[0] + ts * ts * p1[0]
    by = mt * mt * p0[1] + 2 * mt * ts * pc[1] + ts * ts * p1[1]
    return [(float(bx[i]), float(by[i]), float(ts[i])) for i in range(n)]


def _bez_dirs(p0, pc, p1, n):
    ts = np.linspace(0.0, 1.0, n)
    mt = 1.0 - ts
    dx = 2 * mt * (pc[0] - p0[0]) + 2 * ts * (p1[0] - pc[0])
    dy = 2 * mt * (pc[1] - p0[1]) + 2 * ts * (p1[1] - pc[1])
    l = np.hypot(dx, dy) + 1e-9
    return np.stack([dx / l, dy / l], axis=1)


def _walk_emit(pts, dirs, step):
    """沿折线每 step 像素输出一个 (x, y, t, dirx, diry)，保证针叶等距分布。"""
    out = []
    carry = 0.0
    for i in range(1, len(pts)):
        x0, y0, t0 = pts[i - 1]
        x1, y1, t1 = pts[i]
        dx, dy = x1 - x0, y1 - y0
        seg = math.hypot(dx, dy)
        if seg < 1e-9:
            continue
        ux, uy = dirs[i - 1]
        pos = 0.0
        while carry + (seg - pos) >= step:
            need = step - carry
            pos += need
            carry = 0.0
            f = pos / seg
            out.append((x0 + dx * f, y0 + dy * f, t0 + (t1 - t0) * f, ux, uy))
        carry += seg - pos
    x, y, t = pts[-1]
    ux, uy = dirs[-1]
    out.append((x, y, t, ux, uy))
    return out


def _stamp_needle(alpha, lit, wood, hgt, p0, p1, half_w, bright, is_wood=False):
    """把一根针叶（线段+半宽，两端收尖、圆脊截面）盖进 alpha/亮度/木质/高度场。"""
    H, W = alpha.shape
    x0 = max(0, int(min(p0[0], p1[0]) - half_w - 2))
    x1 = min(W, int(max(p0[0], p1[0]) + half_w + 3))
    y0 = max(0, int(min(p0[1], p1[1]) - half_w - 2))
    y1 = min(H, int(max(p0[1], p1[1]) + half_w + 3))
    if x1 <= x0 or y1 <= y0:
        return
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L2 = dx * dx + dy * dy
    if L2 < 1e-6:
        return
    px, py = np.meshgrid(np.arange(x0, x1, dtype=np.float32) + 0.5,
                         np.arange(y0, y1, dtype=np.float32) + 0.5)
    t = ((px - p0[0]) * dx + (py - p0[1]) * dy) / L2
    t = np.clip(t, 0.0, 1.0)
    cx, cy = p0[0] + t * dx, p0[1] + t * dy
    d = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    # 两端收尖（真实针叶是细长的，不是胶囊）
    tip = np.minimum(t, 1.0 - t) * math.sqrt(L2) / max(half_w * 1.7, 1e-3)
    hw = half_w * (0.30 + 0.70 * np.clip(tip, 0.0, 1.0))
    q = d / np.maximum(hw, 1e-3)
    cov = np.clip((1.0 - q) / 0.55, 0.0, 1.0)
    cov = cov * cov * (3.0 - 2.0 * cov)
    if not cov.any():
        return
    sl = (slice(y0, y1), slice(x0, x1))
    np.maximum(alpha[sl], cov, out=alpha[sl])
    if is_wood:
        np.maximum(wood[sl], cov, out=wood[sl])
        np.maximum(hgt[sl], cov * 0.55, out=hgt[sl])   # 茎是圆杆，脊较平
    else:
        # 针叶截面是圆脊 → 法线贴图立体感的直接来源
        np.maximum(hgt[sl], cov * np.clip(1.0 - q * q, 0.0, 1.0) * 0.95,
                   out=hgt[sl])
    np.maximum(lit[sl], cov * bright, out=lit[sl])


def _needles_on(alpha, lit, wood, hgt, walk, rng, nlen, nwid,
                b0, b1, drop=0.34):
    """沿已按弧长重采样的枝条撒针叶（两侧各一排，随机长短/角度/下垂）。"""
    for (x, y, tg, ux, uy) in walk:
        bright = b0 + (b1 - b0) * tg
        for s in (1.0, -1.0):
            ndx, ndy = _rot2(ux, uy, s * rng.uniform(0.55, 1.05))
            ndy -= drop                       # 针叶下垂
            nl = math.hypot(ndx, ndy) or 1.0
            ndx, ndy = ndx / nl, ndy / nl
            L = nlen * rng.uniform(0.72, 1.22)
            _stamp_needle(alpha, lit, wood, hgt, (x, y),
                          (x + ndx * L, y + ndy * L),
                          nwid * rng.uniform(0.85, 1.15), bright)


# 每株小枝的着生点 / 梢端（归一化 cell 坐标，y 向上）。4 株各不相同。
_SPRIG_ATTACH = [(0.07, 0.92), (0.93, 0.92), (0.07, 0.86), (0.93, 0.88)]
_SPRIG_TIP = [(0.94, 0.34), (0.06, 0.38), (0.92, 0.56), (0.08, 0.52)]


def _draw_sprig(cw, chh, rng, variant):
    """
    画一株针叶小枝（cell 局部坐标，行 0 = cell 底边）。

    结构：一根先拱后垂的主茎 + 7~10 根更短的侧枝，沿所有茎等距撒针叶。
    针叶亮度沿"基部 → 梢端"递增（梢端是新芽，明显更亮更黄）。
    """
    alpha = np.zeros((chh, cw), dtype=np.float32)
    lit = np.zeros_like(alpha)
    wood = np.zeros_like(alpha)
    hgt = np.zeros_like(alpha)

    p0 = _SPRIG_ATTACH[variant % 4]
    p2 = _SPRIG_TIP[variant % 4]
    a0 = (p0[0] * cw, p0[1] * chh)
    a2 = (p2[0] * cw, p2[1] * chh)
    ac = ((a0[0] + a2[0]) * 0.5 + rng.uniform(-0.04, 0.04) * cw,
          (a0[1] + a2[1]) * 0.5 + rng.uniform(0.10, 0.24) * chh)

    n = 30
    stem = _bez_points(a0, ac, a2, n)
    sdir = _bez_dirs(a0, ac, a2, n)

    # 主茎（木质）
    for i in range(n - 1):
        w = 2.7 * (1.0 - 0.60 * stem[i][2])
        _stamp_needle(alpha, lit, wood, hgt, stem[i][:2], stem[i + 1][:2],
                      w, 0.06, is_wood=True)

    # 主茎针叶
    _needles_on(alpha, lit, wood, hgt, _walk_emit(stem, sdir, 2.0),
                rng, nlen=chh * 0.040, nwid=1.35, b0=0.22, b1=0.95)

    # 侧枝 + 侧枝针叶。★ 侧枝要多、要长：小枝在 cell 里只占一条对角带，
    # 针叶不够密的话整张卡片几乎是透明的（实测曾只有 5.5% 覆盖率），
    # 树就会稀疏得像插了几根针
    n_bt = int(rng.integers(18, 24))
    for bi in range(n_bt):
        t0 = 0.04 + 0.94 * (bi + rng.uniform(0.1, 0.9)) / n_bt
        i0 = min(n - 2, int(t0 * (n - 1)))
        side = 1.0 if bi % 2 == 0 else -1.0
        bl = rng.uniform(0.28, 0.55) * (1.0 - 0.40 * t0) * chh
        dx, dy = _rot2(sdir[i0][0], sdir[i0][1], side * rng.uniform(0.85, 1.50))
        dy -= rng.uniform(0.10, 0.40)
        nl = math.hypot(dx, dy) or 1.0
        dx, dy = dx / nl, dy / nl
        sx, sy = stem[i0][0], stem[i0][1]
        e = (sx + dx * bl, sy + dy * bl)
        csub = (sx + dx * bl * 0.45,
                sy + dy * bl * 0.45 - bl * rng.uniform(0.18, 0.40))
        m = 18
        sub = _bez_points((sx, sy), csub, e, m)
        subdir = _bez_dirs((sx, sy), csub, e, m)
        for i in range(m - 1):
            w = 1.6 * (1.0 - 0.55 * sub[i][2])
            _stamp_needle(alpha, lit, wood, hgt, sub[i][:2], sub[i + 1][:2],
                          w, 0.06, is_wood=True)
        _needles_on(alpha, lit, wood, hgt, _walk_emit(sub, subdir, 2.0),
                    rng, nlen=chh * 0.034, nwid=1.25,
                    b0=0.26, b1=1.0, drop=0.40)

    return alpha, lit, wood, hgt


def _gen_sprig_tree(th, cr, ch, n_whorl, n_branch, n_cards, card_len,
                    trunk_seg=6, seed=0):
    """
    3A 式针叶树：树干 + 枝干骨架 + 沿骨架布置的小枝面片。

    th/cr/ch 与旧接口同义（干高 / 冠半径 / 冠高）。
    n_whorl   轮生层数（每层一圈枝干）
    n_branch  每层枝干数
    n_cards   每根枝干上的小枝面片数
    card_len  小枝面片沿枝干方向的长度（m）
    """
    rng = random.Random(7717 + seed * 131)
    all_verts = []

    def _push_tangent(verts, tx, ty, tz):
        for v in verts:
            v.extend([tx, ty, tz])

    # ---- 1. 树干（低模圆柱），UV 强制指向图集底部的树皮带 ----
    trunk = _make_tapered_cylinder(
        0, 0, 0, 0, th, 0,
        max(0.10, 0.13 * cr), max(0.05, 0.055 * cr),
        trunk_seg, (0.135, 0.095, 0.070), (0.185, 0.145, 0.105)
    )
    for v in trunk:
        v[6], v[7] = 0.5, _BARK_UV_V     # 采样树皮带（不透明）
    _push_tangent(trunk, 1.0, 0.0, 0.0)
    all_verts.extend(trunk)

    # ---- 2. 枝干骨架 + 小枝面片 ----
    crown_bottom = th * 0.10
    crown_top = th + ch * 0.95
    crown_h = crown_top - crown_bottom
    GA = 2.399963   # 黄金角：相邻轮生的枝干方位错开，避免十字/米字的机械感
    r_max = cr * 0.98

    for k in range(n_whorl):
        f = k / max(1, n_whorl - 1)
        y = crown_bottom + crown_h * (0.03 + 0.97 * f)
        # 越往上枝越短 → 锥形轮廓由骨架自然产生
        blen = r_max * max(0.05, (1.0 - f) ** 0.85) * rng.uniform(0.84, 1.12)
        if blen < 0.16:
            continue
        nb = n_branch + (1 if (k % 2) else 0)
        for j in range(nb):
            az = GA * k + 2.0 * math.pi * j / nb + rng.uniform(-0.20, 0.20)
            cax, saz = math.cos(az), math.sin(az)
            # 枝干姿态：上层上扬、下层下垂（杉树的自然形态）。
            # tip_dy/arch 压得比较平 —— 层与层之间要靠面片纵向互相搭接，
            # 否则冠体会变成"一摞分离的横带"
            tip_dy = 0.30 * (1.0 - f) ** 1.2 - 0.52 * f ** 0.9
            arch = 0.42 * (1.0 - f) ** 1.2 - 0.08 * f
            b0 = (0.0, y, 0.0)
            b1 = (cax * blen, y + blen * tip_dy, saz * blen)
            bc = (cax * blen * 0.5, y + blen * 0.5 * arch, saz * blen * 0.5)

            for ci in range(n_cards):
                # 面片沿枝干分布，并且**跨过枝端**（t 到 1.10）——
                # 梢端也要有叶，否则每根枝的末端是一条秃直线
                t = 0.14 + 0.96 * (ci + rng.uniform(0.15, 0.85)) / n_cards
                t = min(t, 1.10)
                c = _bez3(b0, bc, b1, t)
                d = _bez3_dir(b0, bc, b1, t)
                ux, uy, uz = d          # 面片 u 轴 = 枝干切线（小枝茎沿枝干长）
                # 面法线：先取垂直于枝干、背离树干的水平方向，再往上倾。
                # ★ 像瓦片一样朝上外倾 —— 既符合杉叶自然朝向，也让约 50° 的
                #   太阳真正照到面片（水平面片在 50° 太阳下直射项几乎为零）。
                n0x, n0z = -saz, cax
                tilt = rng.uniform(0.42, 0.88)          # 24°~50°
                if rng.random() < 0.25:
                    tilt = -rng.uniform(0.15, 0.40)     # 少量朝下的腹面小枝
                nx = n0x * math.cos(tilt)
                ny = math.sin(tilt)
                nz = n0z * math.cos(tilt)
                nl = math.sqrt(nx * nx + ny * ny + nz * nz)
                nx, ny, nz = nx / nl, ny / nl, nz / nl
                # v 轴 = cross(N, U)，保证 cross(U,V)=N（右手系、绕序正确）
                vx = ny * uz - nz * uy
                vy = nz * ux - nx * uz
                vz = nx * uy - ny * ux
                # 顶点色 = 冠内深度 AO（越靠树冠核心越暗越冷）。
                # 3A 给叶面片体积感的标准手法，比任何贴图都便宜。
                # ★ 下限不能太低：实测 0.40 起步会把整棵树压暗 1.5 倍，
                #   真机植被均值 RGB 从 (83,130,53) 掉到 (46,68,44)
                rad = math.hypot(c[0], c[2])
                rn = min(1.0, rad / max(r_max, 1e-3))
                sh = (0.72 + 0.28 * rn) * (0.92 + 0.08 * f) * rng.uniform(0.96, 1.04)
                col = (min(1.0, sh * 0.93), min(1.0, sh), min(1.0, sh * 0.90))
                half_u = card_len * 0.5 * rng.uniform(0.90, 1.15)
                half_v = half_u / _SPRIG_CELL_ASPECT
                all_verts.extend(_make_sprig_card(
                    c, (ux, uy, uz), (vx, vy, vz), half_u, half_v,
                    rng.randrange(_SPRIG_COLS), rng.randrange(_SPRIG_ROWS),
                    col, mirror=(rng.random() < 0.5)))

    # ---- 3. 顶端主梢：树最顶上的一撮直立小枝，否则树尖是秃的 ----
    top_y = crown_top
    for j in range(4):
        az = 2.399963 * 3.0 + 2.0 * math.pi * j / 4.0
        cax, saz = math.cos(az), math.sin(az)
        bl = 0.55 + 0.25 * (j % 2)
        c = (cax * bl * 0.45, top_y + bl * 0.55, saz * bl * 0.45)
        d = (cax, 0.85, saz)
        dl = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        d = (d[0] / dl, d[1] / dl, d[2] / dl)
        n0x, n0z = -saz, cax
        nx, ny, nz = n0x * 0.86, 0.50, n0z * 0.86
        nl = math.sqrt(nx * nx + ny * ny + nz * nz)
        nx, ny, nz = nx / nl, ny / nl, nz / nl
        vx = ny * d[2] - nz * d[1]
        vy = nz * d[0] - nx * d[2]
        vz = nx * d[1] - ny * d[0]
        sh = 1.02
        all_verts.extend(_make_sprig_card(
            c, d, (vx, vy, vz), 0.62, 0.62 / _SPRIG_CELL_ASPECT,
            rng.randrange(_SPRIG_COLS), rng.randrange(_SPRIG_ROWS),
            (sh * 0.92, sh, sh * 0.88), mirror=(j % 2 == 0)))

    return np.array(all_verts, dtype=np.float32)


_TREE_VARIANTS = [
    dict(th=4.0, cr=2.0, ch=7.5, whorl=12, branch=6, cards=4, clen=1.45, seg=6, seed=11),   # 标准杉
    dict(th=4.8, cr=1.5, ch=8.8, whorl=14, branch=5, cards=4, clen=1.30, seg=5, seed=23),   # 高瘦钻天
    dict(th=3.2, cr=2.7, ch=5.8, whorl=9,  branch=7, cards=4, clen=1.60, seg=6, seed=37),   # 矮胖老冠
    dict(th=4.3, cr=2.3, ch=6.8, whorl=11, branch=5, cards=3, clen=1.75, seg=5, seed=53),   # 疏朗
]
N_TREE_VARIANTS = len(_TREE_VARIANTS)

_TREE_LOD_SPECS = [   # (max_distance, whorl系数, branch增量, card_len系数)
    (150.0, 1.00,  0, 1.00),
    (400.0, 0.72, -1, 1.12),
    (1e9,   0.48, -1, 1.30),
]

def build_tree_templates():
    """4 变体 x 3 LOD = 12 个模板，打破'一棵模板盖章'的观感"""
    templates = []
    for v_i, V in enumerate(_TREE_VARIANTS):
        for lod_i, (md, wm, bd, cm) in enumerate(_TREE_LOD_SPECS):
            templates.append({
                'variant': v_i, 'lod': lod_i,
                'vertices': _gen_sprig_tree(
                    V['th'], V['cr'], V['ch'],
                    max(5, int(round(V['whorl'] * wm))),
                    max(3, V['branch'] + bd),
                    max(2, V['cards'] + (bd if bd < 0 else 0)),
                    V['clen'] * cm,
                    trunk_seg=max(4, V['seg'] - lod_i),
                    seed=V['seed'] + lod_i * 7),
                'max_distance': md,
            })
    return templates


# ------------------------------------------------------------
# ------------------------------------------------------------
# 杉树叶片图集 v4 —— 3A 做法：贴的是"一根针叶小枝"，不是整棵树冠
# ------------------------------------------------------------
# 3A 的针叶树贴图是一根 0.5~1.5m 的带针小枝（照片扫描或手绘），
# 由 _gen_sprig_tree 沿枝干骨架大量实例化。这里程序化画出 4 株这样的小枝。
# 每株 = 先拱后垂的主茎 + 7~10 根侧枝 + 沿所有茎等距撒的单根针叶。
# ------------------------------------------------------------
# 高度场梯度 → 切线空间法线的放大系数（纯强度，与贴图分辨率无关：
# 各频段梯度已按自身 RMS 归一化）。实测标定（1024 图集，alpha>0.5 的针叶像素）：
#   0.35 → 倾角 mean ~21°   （偏平）
#   0.60 → 倾角 mean 33.6°  p90 59.4°  >60° 仅 9.1%   ← 取此值
#   1.20 → 倾角 mean 48.7°  p90 73.5°  >60° 达 37.9%（噪声，显脏）
_NRM_K = 0.60


def _smoothstep_np(e0, e1, x):
    """支持 e0 > e1 的降序 smoothstep（Gauss 端裁剪）"""
    denom = e1 - e0
    if abs(denom) < 1e-6:
        denom = 1e-6
    t = np.clip((x - e0) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_foliage_textures(size=1024, nrm_k=_NRM_K, ao_floor=0.42):
    """
    程序化生成杉树叶片图集 v4（3A 式小枝图集）。

    返回 (albedo, normal)，均为 float32：
      albedo : RGB = 针叶色（线性空间）, A = 剪影覆盖率
      normal : RGB = 切线空间法线(n*0.5+0.5), A = 小枝内部遮蔽(AO)

    布局约定（必须与 _sprig_cell_uv / _gen_sprig_tree 一致）：
      v <  _CROWN_UV_V0 : 不透明树皮带（RGB=白，交给顶点色着色，法线平坦）
      v >= _CROWN_UV_V0 : _SPRIG_COLS × _SPRIG_ROWS 株针叶小枝

    为什么不再把整棵树冠画进一张图：
      树冠轮廓必须由枝干骨架"长"出来（见 _gen_sprig_tree 的注释）。
      贴图只负责"一根小枝长什么样"，冠内体积感靠顶点色 AO。
    """
    rng = np.random.default_rng(20260918)
    H = W = int(size)
    bark_rows = max(2, int(round(_CROWN_UV_V0 * H)))
    cell_h = (H - bark_rows) // _SPRIG_ROWS
    cell_w = W // _SPRIG_COLS

    alpha = np.zeros((H, W), dtype=np.float32)
    lit = np.zeros((H, W), dtype=np.float32)
    wood = np.zeros((H, W), dtype=np.float32)
    hgt = np.zeros((H, W), dtype=np.float32)

    for iy in range(_SPRIG_ROWS):
        for ix in range(_SPRIG_COLS):
            ca, cl, cw_, ch_ = _draw_sprig(cell_w, cell_h, rng,
                                           iy * _SPRIG_COLS + ix)
            y0 = bark_rows + iy * cell_h
            x0 = ix * cell_w
            sl = (slice(y0, y0 + cell_h), slice(x0, x0 + cell_w))
            np.maximum(alpha[sl], ca, out=alpha[sl])
            np.maximum(lit[sl], cl, out=lit[sl])
            np.maximum(wood[sl], cw_, out=wood[sl])
            np.maximum(hgt[sl], ch_, out=hgt[sl])

    alpha = np.clip(alpha, 0.0, 1.0)

    # ---- 颜色（线性空间三段色阶）：基部暗 → 梢端亮黄绿（新芽） ----
    deep = np.array([0.018, 0.054, 0.033], dtype=np.float32)
    midc = np.array([0.072, 0.178, 0.078], dtype=np.float32)
    tip = np.array([0.245, 0.448, 0.140], dtype=np.float32)
    L = np.clip(lit, 0.0, 1.0)[..., None]
    t1 = _smoothstep_np(0.06, 0.55, L)
    t2 = _smoothstep_np(0.52, 0.96, L)
    rgb = deep * (1.0 - t1) + midc * t1
    rgb = rgb * (1.0 - t2) + tip * t2
    # 木质茎单独上色（深棕）
    wmask = np.clip(wood[..., None] * 1.5, 0.0, 1.0)
    rgb = rgb * (1.0 - wmask) + np.array([0.085, 0.058, 0.038],
                                         dtype=np.float32) * wmask

    # ---- 法线贴图（多尺度高度场梯度）----
    # 与分辨率解耦：每个频段先按自身梯度 RMS 归一再加权，nrm_k 即纯强度。
    #   高频 = 针叶圆脊截面（近景细节）
    #   中频 = 针叶束成组
    #   低频 = 枝条级体积（远景 mip 之后唯一幸存的信息，必须有）
    gx = np.zeros((H, W), dtype=np.float32)
    gy = np.zeros((H, W), dtype=np.float32)
    for _fld, _sg, _wt in ((hgt, 0.9, 0.42),
                           (alpha, 2.4, 0.32),
                           (alpha, 8.0, 0.26)):
        _gyi, _gxi = np.gradient(gaussian_filter(_fld, _sg))
        _s = math.sqrt(float((_gxi * _gxi + _gyi * _gyi).mean())) + 1e-9
        gx += _gxi / _s * _wt
        gy += _gyi / _s * _wt
    nrm = np.stack([-gx * nrm_k, -gy * nrm_k, np.ones_like(gx)], axis=-1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
    nrm = (nrm * 0.5 + 0.5).astype(np.float32)

    # ---- AO：针叶互相遮挡（局部密度）+ 越靠小枝基部越暗 ----
    # 小枝基部的老针叶被上层遮蔽，梢端新芽最亮 —— 这是"枝"的立体线索
    dens = gaussian_filter(alpha, 2.4)
    dens_n = np.clip((dens - 0.22) / 0.55, 0.0, 1.0)
    yy, xx = np.mgrid[0:H, 0:W]
    tip_n = np.ones((H, W), dtype=np.float32)
    dmax = 1.05 * max(cell_w, cell_h)
    for iy in range(_SPRIG_ROWS):
        for ix in range(_SPRIG_COLS):
            att = _SPRIG_ATTACH[iy * _SPRIG_COLS + ix]
            ax_ = ix * cell_w + att[0] * cell_w
            ay_ = bark_rows + iy * cell_h + att[1] * cell_h
            tip_n = np.minimum(tip_n, np.clip(
                np.sqrt((xx - ax_) ** 2 + (yy - ay_) ** 2) / dmax, 0.0, 1.0))
    ao = np.clip(ao_floor + (1.0 - ao_floor)
                 * (0.40 * (1.0 - dens_n) + 0.60 * tip_n),
                 0.0, 1.0).astype(np.float32)

    normal = np.concatenate([nrm, ao[..., None]], axis=-1).astype(np.float32)

    albedo = np.concatenate([rgb, alpha[..., None]], axis=-1).astype(np.float32)
    bark = np.zeros((H, W), dtype=bool)
    bark[:bark_rows, :] = True
    albedo[bark] = np.array([0.10, 0.075, 0.055, 1.0], dtype=np.float32)  # ★ 深棕，与树干顶点色同族；不再在mip链中向小枝区渗白色
    # ★ 透明区必须用"叶片平均色"填充，不能留深色：
    #   mip 是 RGB 取平均，透明区若留暗色，小枝又稀疏（覆盖率仅 ~15%），
    #   远处树叶会被平均成一层暗绿泥 —— 真机实测植被 RGB 因此从
    #   (83,130,53) 掉到 (46,68,44)。这是稀疏 alpha 贴图的标准处理。
    op = alpha > 0.5
    if op.any():
        mean_rgb = rgb[op].mean(axis=0)
        hole = (~op) & (~bark)
        albedo[hole, 0] = mean_rgb[0]
        albedo[hole, 1] = mean_rgb[1]
        albedo[hole, 2] = mean_rgb[2]
    # 树皮带与透明区一律平坦法线 + 无遮蔽，避免 mip 混入假法线/假阴影
    flat = (bark | (alpha <= 0.02))[..., None]
    normal = np.where(flat, np.array([0.5, 0.5, 1.0, 1.0],
                                     dtype=np.float32), normal)
    normal = normal.astype(np.float32)

    cov_crown = float(alpha[bark_rows:].mean())
    print(f"[foliage] 小枝图集 {W}x{H}：{_SPRIG_COLS * _SPRIG_ROWS} 株针叶小枝, "
          f"整图覆盖率 {float(alpha.mean()):.3f}, 小枝区覆盖率 {cov_crown:.3f}")
    return albedo, normal


def build_foliage_texture(size=1024):
    """兼容旧接口：只返回 albedo 图集"""
    return build_foliage_textures(size)[0]


def build_tree_instances():
    """树实例 — 已缓存（缓存名挂 path 指纹：is_safe 判定基于新路）"""
    return _build_or_load_env("akina_tree_inst_v9", _build_tree_instances_impl)

def _build_tree_instances_impl():
    """生成所有树实例的位置/旋转/缩放/变体数据。输出: (M, 8) float32"""
    # 预取建筑位置（触发 get_ground_height 的初始化，防止循环触发递归）
    bldg_pos = get_building_positions()

    path = get_akina_path()
    rng = random.Random(42)
    n_path = len(path)
    instances = []
    COLORS = [
        (0.10, 0.26, 0.08), (0.12, 0.32, 0.10), (0.15, 0.38, 0.12),
        (0.13, 0.30, 0.09), (0.17, 0.42, 0.13), (0.08, 0.22, 0.06)
    ]

    # ---- 预计算挖方边坡「视廊」（防止树插墙 + 防止树挡住墙/护坡）----
    # 判据与 _build_retaining_walls_impl 共用 cut_foot_sample()：
    # 树避让的位置必须和真正砌出来的墙一致，否则树会插进墙里。
    #
    # ★ 旧版只给坡脚一个 1.5m 避让圈 —— 墙是砌出来了，但树冠（半径 2.5m）
    #   就贴在墙前 2.7m（实测坡脚→最近树中位数），挡土墙/格构护坡全被树遮死。
    #   实测挖方边坡 rise 最大仅 ~1.26m、坡度 ~0.29 ⇒ 坡面水平跨度只有 ~4.5m，
    #   所以沿外法线取 0 / 2.5 / 5.0m 三个偏移就能覆盖整片坡面。
    _CUT_VIEW_OFFSETS = (0.0, 2.5, 5.0)      # 坡脚 → 坡面（沿外法线）
    _CUT_VIEW_RADII   = (3.6, 3.6, 3.2)      # 含树冠半径 2.5m 的余量
    _retain_pts = []  # 收集 (x, z, 半径)
    for _i in range(0, n_path, 2):   # 每 2 点采样一次
        _x, _z, _y, _w, _bank = path[_i]
        _ni = min(_i + 1, n_path - 1)
        _dx = path[_ni][0] - _x
        _dz = path[_ni][1] - _z
        _L = math.hypot(_dx, _dz) or 1e-6
        _rx, _rz = -_dz / _L, _dx / _L   # 右法线
        for _side in (1.0, -1.0):
            _fx, _fz, _rise = cut_foot_sample(_x, _z, _rx, _rz, _w, _side)
            if _rise > _CUT_MIN_RISE:
                for _o, _r in zip(_CUT_VIEW_OFFSETS, _CUT_VIEW_RADII):
                    _retain_pts.append((_fx + _rx * _side * _o,
                                        _fz + _rz * _side * _o, _r))
    # 空间哈希（2 万个避让点，逐候选做全量距离是 O(N·M)，太慢）
    _CELL = 6.0
    _retain_grid = {}
    for _ax, _az, _ar in _retain_pts:
        _k = (int(math.floor(_ax / _CELL)), int(math.floor(_az / _CELL)))
        _b = _retain_grid.get(_k)
        if _b is None:
            _retain_grid[_k] = [(_ax, _az, _ar)]
        else:
            _b.append((_ax, _az, _ar))

    # ---- 预计算：路径中心线每 3m 一个采样 + 逐点曲率（视线走廊/弯道余量）----
    _sight_pts = []   # [(x, z), ...] 沿弧长 3m 间隔
    for _si in range(0, n_path - 1):
        _sx1, _sz1 = path[_si][0], path[_si][1]
        _sx2, _sz2 = path[_si + 1][0], path[_si + 1][1]
        _sl = math.hypot(_sx2 - _sx1, _sz2 - _sz1)
        _ns = max(1, int(_sl / 3.0))
        for _sk in range(_ns):
            _st = _sk / _ns
            _sight_pts.append((_sx1 + (_sx2 - _sx1) * _st, _sz1 + (_sz2 - _sz1) * _st))
    _sight_arr = np.array(_sight_pts, dtype=np.float64)
    # 逐点曲率半径倒数（离散弯道检测）
    _curv_arr = np.zeros(n_path, dtype=np.float64)
    for _ci in range(1, n_path - 1):
        _p0 = path[_ci - 1]; _p1 = path[_ci]; _p2 = path[_ci + 1]
        _v1x = _p1[0] - _p0[0]; _v1z = _p1[1] - _p0[1]
        _v2x = _p2[0] - _p1[0]; _v2z = _p2[1] - _p1[1]
        _l1 = math.hypot(_v1x, _v1z) or 1e-6
        _l2 = math.hypot(_v2x, _v2z) or 1e-6
        _dot = (_v1x * _v2x + _v1z * _v2z) / (_l1 * _l2)
        _curv_arr[_ci] = abs(1.0 - max(-1.0, min(1.0, _dot)))  # 0=直, 越大越弯
    SIGHT_HALF_W = 3.2   # 视线走廊半宽

    for i in range(0, n_path, 4):
        x, z, y, w, bank = path[i]
        ni = min(i + 1, n_path - 1)
        dx = path[ni][0] - x
        dz = path[ni][1] - z
        length = math.sqrt(dx*dx + dz*dz)
        if length < 0.001:
            continue
        tx, tz = dx/length, dz/length
        rx, rz = -tz, tx
        hw = w / 2.0

        for side in [-1, 1]:
            for _ in range(rng.randint(2, 4)):
                dist = hw + rng.uniform(2.0 * _S, 20.0 * _S)
                ox = rx * dist * side
                oz = rz * dist * side
                tx2 = x + ox + rng.uniform(-2.0 * _S, 2.0 * _S)
                tz2 = z + oz + rng.uniform(-2.0 * _S, 2.0 * _S)

                ground_y = get_ground_height(tx2, tz2) - 0.05
                if ground_y < -50 or ground_y > 500:
                    continue
                # 弯道段安全余量按曲率放大（内圈更空，改善视线）
                _curv_margin = 6.0 if _curv_arr[i] > 0.02 else 0.0
                if not is_safe(tx2, tz2, 3.5 + _curv_margin):
                    continue

                # ---- 视线走廊遮挡测试（U 型弯内侧不种树）----
                if len(_sight_arr) > 1:
                    _p0 = _sight_arr[:-1]; _p1 = _sight_arr[1:]
                    _dx = _p1[:, 0] - _p0[:, 0]; _dz = _p1[:, 1] - _p0[:, 1]
                    _L2 = np.maximum(_dx * _dx + _dz * _dz, 1e-9)
                    _t = np.clip(((tx2 - _p0[:, 0]) * _dx + (tz2 - _p0[:, 1]) * _dz) / _L2, 0, 1)
                    _cx = _p0[:, 0] + _t * _dx; _cz = _p0[:, 1] + _t * _dz
                    _d2 = (tx2 - _cx) ** 2 + (tz2 - _cz) ** 2
                    # 树冠投影半径 2.5m + 走廊半宽 → 距中心线 <5.7m 即算遮挡
                    if np.any(_d2 < (SIGHT_HALF_W + 2.5) ** 2):
                        continue

                # ---- 避开建筑（防止树穿墙）----
                if len(bldg_pos) > 0:
                    _dx_b = tx2 - bldg_pos[:, 0]
                    _dz_b = tz2 - bldg_pos[:, 2]
                    _r_b  = bldg_pos[:, 3]
                    if np.any(_dx_b*_dx_b + _dz_b*_dz_b < _r_b*_r_b):
                        continue

                # ---- 避开挖方边坡视廊（防插墙 + 让挡土墙/护坡露出来）----
                if _retain_grid:
                    _gx = int(math.floor(tx2 / _CELL))
                    _gz = int(math.floor(tz2 / _CELL))
                    _hit = False
                    for _ox in (-1, 0, 1):
                        for _oz in (-1, 0, 1):
                            _b = _retain_grid.get((_gx + _ox, _gz + _oz))
                            if not _b:
                                continue
                            for (_ax, _az, _ar) in _b:
                                _ddx = tx2 - _ax; _ddz = tz2 - _az
                                if _ddx*_ddx + _ddz*_ddz < _ar*_ar:
                                    _hit = True
                                    break
                            if _hit:
                                break
                        if _hit:
                            break
                    if _hit:
                        continue

                target_h = rng.uniform(3.0, 7.5)              # 树高目标，范围加大
                variant  = rng.randrange(N_TREE_VARIANTS)
                vth = _TREE_VARIANTS[variant]['th']
                sy  = target_h / vth
                sxz = sy * rng.uniform(0.82, 1.18)            # 宽度独立抖动：同树形也不同胖瘦
                _ = rng.choice(COLORS)  # 消耗随机状态，保持序列一致

                # 消耗树冠 offset 的随机状态
                for layer_ratio in [0.0, 0.5]:
                    _r = 2.0 * (1.0 if layer_ratio == 0.0 else 0.6)
                    _ = rng.uniform(-0.2, 0.2) * _r

                # 旋转角从位置哈希导出（不额外消耗 rng，保证位置与原版一致）
                rot = (tx2 * 7.341 + tz2 * 3.179) % (2 * math.pi)
                instances.append((tx2, ground_y, tz2, rot, sxz, sy, sxz, float(variant)))

    return np.array(instances, dtype=np.float32)


# ============================================================
# 七、赛道网格构建
# ============================================================

def build_akina_trees():
    """生态式植被分布（杉树为主），高度由地形决定
    结果会被缓存到 .npy 文件，第二次启动直接加载。
    缓存名挂 path 指纹（树木分布基于 is_safe，路变则必须重算）。
    """
    cache_name = _env_cache_name("akina_trees_v4", get_akina_path())
    cache_path = _get_cache_path(cache_name)
    if cache_path and os.path.exists(cache_path):
        print(f"[缓存] 加载杉树林缓存...")
        data = np.load(cache_path)
        print(f"[缓存]  杉树林加载完成: {len(data)} 顶点")
        return data

    print(f"[缓存] 构建杉树林（首次运行，耗时较长）...")
    t0 = __import__('time').time()
    result = _build_akina_trees_impl()
    elapsed = __import__('time').time() - t0
    if cache_path and len(result) > 0:
        try:
            np.save(cache_path, result)
            print(f"[缓存]  杉树林已缓存 ({elapsed:.1f}s)")
            _clean_stale_env_caches("akina_trees_v4", cache_name)
        except Exception as e:
            print(f"[缓存]  杉树林缓存失败: {e}")
    else:
        print(f"[缓存]  杉树林构建完成 ({elapsed:.1f}s)")
    return result


def _build_akina_trees_impl():
    """杉树林实际构建逻辑（极简低多边形版本，高密度、大体积、低顶点）"""
    verts = []
    def tri(p1, p2, p3, n, uv, c):
        verts.append([*p1, *n, *uv[0], *c])
        verts.append([*p2, *n, *uv[1], *c])
        verts.append([*p3, *n, *uv[2], *c])
    def quad(p1, p2, p3, p4, n, uv, c):
        tri(p1, p2, p3, n, [uv[0], uv[1], uv[2]], c)
        tri(p1, p3, p4, n, [uv[0], uv[2], uv[3]], c)

    TRUNK = (0.25, 0.15, 0.05)          # 树干受光面
    TRUNK_DARK = (0.16, 0.09, 0.03)     # 树干顶盖（略暗）
    COLORS = [
        (0.10, 0.26, 0.08), (0.12, 0.32, 0.10), (0.15, 0.38, 0.12),
        (0.13, 0.30, 0.09), (0.17, 0.42, 0.13), (0.08, 0.22, 0.06)
    ]

    path = get_akina_path()
    rng = random.Random(42)
    n_path = len(path)

    for i in range(0, n_path, 4):
        x, z, y, w, bank = path[i]
        ni = min(i + 1, n_path - 1)
        dx = path[ni][0] - x
        dz = path[ni][1] - z
        length = math.sqrt(dx*dx + dz*dz)
        if length < 0.001:
            continue
        tx, tz = dx/length, dz/length
        rx, rz = -tz, tx
        hw = w / 2.0

        for side in [-1, 1]:
            for _ in range(rng.randint(2, 4)):
                dist = hw + rng.uniform(2.0 * _S, 20.0 * _S)
                ox = rx * dist * side
                oz = rz * dist * side
                tx2 = x + ox + rng.uniform(-2.0 * _S, 2.0 * _S)
                tz2 = z + oz + rng.uniform(-2.0 * _S, 2.0 * _S)

                # ✅ 用与地形网格一致的高度函数贴地（不再浮空/穿模）
                ground_y = get_ground_height(tx2, tz2)
                # 树干稍微埋入地面 0.05，避免因采样误差露出根部
                ground_y -= 0.05
                # 限制高度范围，避免生成在太陡峭或太高的地方
                if ground_y < -50 or ground_y > 500:
                    continue
                # margin 从 2.0 提升到 3.5：让树远离压平路基，更自然
                if not is_safe(tx2, tz2, 3.5):
                    continue

                th = rng.uniform(3.0, 6.0)
                cr = rng.uniform(1.5, 3.5)
                ch = rng.uniform(4.0, 8.0)
                col = rng.choice(COLORS)

                # ============================================================
                # 1. 树干：6 边形棱柱（低多边形类圆柱）
                #    - 6 个侧面：法线朝外，各面光照不同 → 立体圆柱感
                #    - 1 个顶盖：朝上，颜色略暗
                # ============================================================
                TRUNK_SEG = 6
                tw = max(0.10, 0.15 * cr)
                top_y = ground_y + th

                cs = [math.cos(2*math.pi*s/TRUNK_SEG) for s in range(TRUNK_SEG)]
                sn = [math.sin(2*math.pi*s/TRUNK_SEG) for s in range(TRUNK_SEG)]

                # ---- 侧面 ----
                for s in range(TRUNK_SEG):
                    s2 = (s + 1) % TRUNK_SEG
                    b1x = tx2 + tw * cs[s];  b1z = tz2 + tw * sn[s]
                    b2x = tx2 + tw * cs[s2]; b2z = tz2 + tw * sn[s2]
                    mx = cs[s] + cs[s2]
                    mz = sn[s] + sn[s2]
                    m_len = math.sqrt(mx*mx + mz*mz)
                    if m_len > 1e-8:
                        mx /= m_len; mz /= m_len
                    else:
                        mx, mz = 0.0, 1.0
                    face_n = (mx, 0.0, mz)
                    quad((b1x, ground_y, b1z), (b2x, ground_y, b2z),
                         (b2x, top_y, b2z), (b1x, top_y, b1z),
                         face_n, [(0, 0)]*4, TRUNK)

                # ---- 顶部封盖（朝上，深色）----
                for s in range(TRUNK_SEG):
                    s2 = (s + 1) % TRUNK_SEG
                    p1 = (tx2 + tw*cs[s],  top_y, tz2 + tw*sn[s])
                    p2 = (tx2 + tw*cs[s2], top_y, tz2 + tw*sn[s2])
                    tri((tx2, top_y, tz2), p1, p2,
                        (0.0, 1.0, 0.0), [(0, 0)]*3, TRUNK_DARK)

                # ============================================================
                # 2. 树冠：每面按锥体几何计算真实法线
                # ============================================================
                layers = [(0.0, 1.0), (0.5, 0.6)]
                seg = 6
                for y_ratio, r_ratio in layers:
                    cy_tree = ground_y + th + ch * y_ratio
                    r = cr * r_ratio
                    h_layer = ch * 0.6
                    offset = rng.uniform(-0.2, 0.2) * r
                    apex_x = tx2 + offset
                    apex_z = tz2 + offset
                    apex_y = cy_tree + h_layer

                    for s in range(seg):
                        a1 = 2 * math.pi * s / seg
                        a2 = 2 * math.pi * (s + 1) / seg
                        bx1 = tx2 + r * math.cos(a1) + offset
                        bz1 = tz2 + r * math.sin(a1) + offset
                        bx2 = tx2 + r * math.cos(a2) + offset
                        bz2 = tz2 + r * math.sin(a2) + offset

                        mx = (bx1 + bx2) * 0.5 - apex_x
                        mz = (bz1 + bz2) * 0.5 - apex_z
                        m_len = math.sqrt(mx*mx + mz*mz)
                        if m_len > 1e-8:
                            mx /= m_len; mz /= m_len
                        H = max(h_layer, 1e-3)
                        nx_f = mx * H
                        ny_f = r
                        nz_f = mz * H
                        nl = math.sqrt(nx_f*nx_f + ny_f*ny_f + nz_f*nz_f)
                        face_n = (nx_f/nl, ny_f/nl, nz_f/nl) if nl > 1e-8 else (0.0, 1.0, 0.0)

                        tri((apex_x, apex_y, apex_z),
                            (bx1, cy_tree, bz1), (bx2, cy_tree, bz2),
                            face_n, [(0, 0)]*3, col)
                        tri((apex_x, cy_tree, apex_z),
                            (bx2, cy_tree, bz2), (bx1, cy_tree, bz1),
                            (0, -1, 0), [(0, 0)]*3,
                            (col[0]*0.6, col[1]*0.6, col[2]*0.6))

    return np.array(verts, dtype=np.float32)


# ============================================================
# 十二、路灯（重做：自发光灯罩 + 地面光池，取消圆盘光晕）
# ============================================================

_LIGHT_PARAMS = {
    "POLE_H":    5.0,
    "POLE_R":    0.07,
    "ARM_LEN":   1.20,
    "SPACING":   50.0,
    "HAIRPIN_SPACING": 28.0,
}


def _iter_light_positions(path, tangents):
    n = len(path)
    if n < 2:
        return
    cum = [0.0]
    for i in range(1, n):
        dx = path[i][0] - path[i-1][0]
        dz = path[i][1] - path[i-1][1]
        cum.append(cum[-1] + math.hypot(dx, dz))

    P = _LIGHT_PARAMS
    last_s = -1e9
    for i in range(n):
        s = cum[i]
        bank_deg = abs(math.degrees(path[i][4]))
        interval = P["HAIRPIN_SPACING"] if bank_deg > 5.0 else P["SPACING"]
        if s - last_s < interval:
            continue
        last_s = s

        x, z, y, w, bank = path[i]
        tx, ty, tz = tangents[i]
        nx_2d, nz_2d = -tz, tx
        L = math.hypot(nx_2d, nz_2d)
        if L < 1e-8:
            continue
        nx_2d /= L; nz_2d /= L

        side = 1.0 if (i // 3) % 2 == 0 else -1.0
        off = w * 0.35 + 1.1
        px = x + nx_2d * off * side
        pz = z + nz_2d * off * side
        py = get_ground_height(px, pz)

        # 灯头位置（悬臂末端，指向赛道内侧）
        ax = px - nx_2d * side * P["ARM_LEN"]
        az = pz - nz_2d * side * P["ARM_LEN"]
        ay = py + P["POLE_H"] - 0.10

        yield (px, py, pz, ax, ay, az, nx_2d, nz_2d, side)


def build_lights():
    """路灯 — 已缓存（缓存名挂 path 指纹）"""
    return _build_or_load_env("akina_lights_v1", _build_lights_impl)

def _build_lights_impl():
    """路灯几何：灯杆 + 弯臂 + 灯罩外壳。灯罩底面自发光在 build_lamp_emissive() 中。"""
    path, tangents = get_akina_path(return_tangents=True)[:2]
    verts = []
    def tri(p1, p2, p3, n, uv, c):
        verts.append([*p1, *n, *uv[0], *c])
        verts.append([*p2, *n, *uv[1], *c])
        verts.append([*p3, *n, *uv[2], *c])
    def quad(p1, p2, p3, p4, n, uv, c):
        tri(p1, p2, p3, n, [uv[0], uv[1], uv[2]], c)
        tri(p1, p3, p4, n, [uv[0], uv[2], uv[3]], c)

    POLE_COL = (0.32, 0.32, 0.35)
    ARM_COL  = (0.38, 0.38, 0.40)
    CASE_COL = (0.42, 0.42, 0.45)

    P = _LIGHT_PARAMS
    SEG = 6

    for (px, py, pz, ax, ay, az, *_rest) in _iter_light_positions(path, tangents):
        # 灯杆
        cs = [math.cos(2*math.pi*k/SEG) for k in range(SEG)]
        sn = [math.sin(2*math.pi*k/SEG) for k in range(SEG)]
        for k in range(SEG):
            k2 = (k + 1) % SEG
            b1x = px + P["POLE_R"] * cs[k];  b1z = pz + P["POLE_R"] * sn[k]
            b2x = px + P["POLE_R"] * cs[k2]; b2z = pz + P["POLE_R"] * sn[k2]
            mx = cs[k] + cs[k2]; mz = sn[k] + sn[k2]
            ml = math.hypot(mx, mz) or 1.0
            mx /= ml; mz /= ml
            quad((b1x, py, b1z), (b2x, py, b2z),
                 (b2x, py + P["POLE_H"], b2z), (b1x, py + P["POLE_H"], b1z),
                 (mx, 0.0, mz), [(0,0)]*4, POLE_COL)
        for k in range(SEG):
            k2 = (k + 1) % SEG
            tri((px, py + P["POLE_H"], pz),
                (px + P["POLE_R"]*cs[k],  py + P["POLE_H"], pz + P["POLE_R"]*sn[k]),
                (px + P["POLE_R"]*cs[k2], py + P["POLE_H"], pz + P["POLE_R"]*sn[k2]),
                (0, 1, 0), [(0,0)]*3, ARM_COL)

        # 弯臂（细盒体）
        ARM_W = 0.05
        hw = ARM_W * 0.5
        axx, azz = ax - px, az - pz
        aL_val = math.hypot(axx, azz) or 1.0
        axx /= aL_val; azz /= aL_val
        sx, sz = azz, -axx
        p0_y = py + P["POLE_H"] - 0.05
        p1_y = ay

        p0_ul = (px + sx*hw, p0_y + hw, pz + sz*hw)
        p0_ur = (px - sx*hw, p0_y + hw, pz - sz*hw)
        p0_dl = (px + sx*hw, p0_y - hw, pz + sz*hw)
        p0_dr = (px - sx*hw, p0_y - hw, pz - sz*hw)
        p1_ul = (ax + sx*hw, p1_y + hw, az + sz*hw)
        p1_ur = (ax - sx*hw, p1_y + hw, az - sz*hw)
        p1_dl = (ax + sx*hw, p1_y - hw, az + sz*hw)
        p1_dr = (ax - sx*hw, p1_y - hw, az - sz*hw)

        quad(p0_ul, p0_ur, p1_ur, p1_ul, (0, 1, 0), [(0,0)]*4, ARM_COL)
        quad(p0_dr, p0_dl, p1_dl, p1_dr, (0, -1, 0), [(0,0)]*4, POLE_COL)
        quad(p0_dl, p0_ul, p1_ul, p1_dl, (sx, 0, sz), [(0,0)]*4, ARM_COL)
        quad(p0_ur, p0_dr, p1_dr, p1_ur, (-sx, 0, -sz), [(0,0)]*4, ARM_COL)

        # 灯罩外壳（盒体，底面不做——留给自发光）
        LW, LH = 0.36, 0.14
        lx0, lx1 = ax - LW*0.5, ax + LW*0.5
        lz0, lz1 = az - LW*0.5, az + LW*0.5
        ly_top = ay - hw
        ly_bot = ly_top - LH
        quad((lx0, ly_top, lz0), (lx1, ly_top, lz0),
             (lx1, ly_top, lz1), (lx0, ly_top, lz1),
             (0, 1, 0), [(0,0)]*4, CASE_COL)
        quad((lx0, ly_bot, lz0), (lx1, ly_bot, lz0),
             (lx1, ly_top, lz0), (lx0, ly_top, lz0),
             (0, 0, -1), [(0,0)]*4, CASE_COL)
        quad((lx1, ly_bot, lz1), (lx0, ly_bot, lz1),
             (lx0, ly_top, lz1), (lx1, ly_top, lz1),
             (0, 0, 1), [(0,0)]*4, CASE_COL)
        quad((lx0, ly_bot, lz1), (lx0, ly_bot, lz0),
             (lx0, ly_top, lz0), (lx0, ly_top, lz1),
             (-1, 0, 0), [(0,0)]*4, CASE_COL)
        quad((lx1, ly_bot, lz0), (lx1, ly_bot, lz1),
             (lx1, ly_top, lz1), (lx1, ly_top, lz0),
             (1, 0, 0), [(0,0)]*4, CASE_COL)

    return np.array(verts, dtype=np.float32)


def build_lamp_emissive():
    """灯罩自发光 — 已缓存（缓存名挂 path 指纹）"""
    return _build_or_load_env("akina_lamp_em_v1", _build_lamp_emissive_impl)

def _build_lamp_emissive_impl():
    """
    灯罩底部的自发光面（用 uIsEmissive=1 单独渲染，不走光照）。
    + 地面椭圆光池（加法混合）。
    """
    path, tangents = get_akina_path(return_tangents=True)[:2]
    verts = []
    def tri(p1, p2, p3, n, uv, c):
        verts.append([*p1, *n, *uv[0], *c])
        verts.append([*p2, *n, *uv[1], *c])
        verts.append([*p3, *n, *uv[2], *c])

    # 灯罩底部自发光色（暖黄，非常亮）
    EMIT_COL = (1.00, 0.92, 0.65)

    P = _LIGHT_PARAMS

    for (px, py, pz, ax, ay, az, *_rest) in _iter_light_positions(path, tangents):
        LW = 0.36
        ly = ay - 0.05*0.5 - 0.14  # 灯罩底面
        quad_pts = [
            (ax-LW*0.5, ly, az-LW*0.5),
            (ax+LW*0.5, ly, az-LW*0.5),
            (ax+LW*0.5, ly, az+LW*0.5),
            (ax-LW*0.5, ly, az+LW*0.5),
        ]
        tri(quad_pts[0], quad_pts[1], quad_pts[2],
            (0, -1, 0), [(0,0)]*3, EMIT_COL)
        tri(quad_pts[0], quad_pts[2], quad_pts[3],
            (0, -1, 0), [(0,0)]*3, EMIT_COL)

    return np.array(verts, dtype=np.float32)


def build_lamp_ground_pool():
    """地面光池 — 已缓存（缓存名挂 path 指纹）"""
    return _build_or_load_env("akina_lamp_pool_v1", _build_lamp_ground_pool_impl)

def _build_lamp_ground_pool_impl():
    """
    地面光池：贴地的椭圆加法混合面片。
    半径 2.2m，略高于地面防 Z-fighting。
    """
    path, tangents = get_akina_path(return_tangents=True)[:2]
    verts = []
    def tri(p1, p2, p3, n, uv, c):
        verts.append([*p1, *n, *uv[0], *c])
        verts.append([*p2, *n, *uv[1], *c])
        verts.append([*p3, *n, *uv[2], *c])

    POOL_COL = (1.00, 0.78, 0.42)
    POOL_R = 2.2
    SEG = 16
    H_ABOVE = 0.04

    for (px, py, pz, ax, ay, az, *_rest) in _iter_light_positions(path, tangents):
        gy = get_ground_height(ax, az)
        cy = gy + H_ABOVE
        for k in range(SEG):
            a1 = 2*math.pi*k/SEG
            a2 = 2*math.pi*(k+1)/SEG
            tri((ax, cy, az),
                (ax + POOL_R*math.cos(a1), cy, az + POOL_R*math.sin(a1)),
                (ax + POOL_R*math.cos(a2), cy, az + POOL_R*math.sin(a2)),
                (0, 1, 0), [(0,0)]*3, POOL_COL)

    return np.array(verts, dtype=np.float32)


# ============================================================
# 十一、沿路建筑群（位置由 get_building_positions() 共享给树）
# ============================================================

def _iter_building_specs():
    """
    生成器：依次 yield 每栋建筑的规格。
    spec = (bx, by, bz, bw, bd, bh, facing, base_color, has_roof, rh, building_type, floors)
    新增 building_type: 'residential' / 'shop' / 'warehouse'
    """
    rng = random.Random(20240117)
    BCLRS = [(0.60,0.55,0.50),(0.42,0.44,0.52),(0.55,0.38,0.32),
             (0.68,0.64,0.56),(0.36,0.40,0.48),(0.52,0.32,0.26),
             (0.62,0.58,0.50),(0.32,0.36,0.42),(0.55,0.50,0.45),
             (0.46,0.50,0.56),(0.60,0.42,0.36),(0.50,0.55,0.60)]

    path = get_akina_path()
    n_path = len(path)
    if n_path < 4:
        return

    i = 4
    count = 0
    while i < n_path and count < 50:
        x, z, y, w, bank = path[i]
        ni = min(i + 1, n_path - 1)
        dx, dz = path[ni][0] - x, path[ni][1] - z
        L = math.hypot(dx, dz)
        if L < 1e-3:
            i += 10; continue
        tx, tz = dx / L, dz / L
        rx, rz = -tz, tx

        side = 1.0 if rng.random() < 0.5 else -1.0
        dist = w * 0.35 + rng.uniform(3.0, 10.0)    # ★ 贴近赛道
        bx = x + rx * dist * side
        bz = z + rz * dist * side
        # ★ 朝向：正面朝向赛道（在采样高度之前计算，供五角采样用）
        facing = math.atan2(rx * side, -rz * side)

        # ---- 日式建筑比例 ----
        bw = rng.uniform(3.0, 5.0)
        bd = rng.uniform(8.0, 15.0)
        bh = rng.uniform(6.0, 9.0)

        # ★ 采样建筑底面五角（四角+中心）地形高度，取最低点
        c_f_, s_f_ = math.cos(facing), math.sin(facing)
        _bhw = bw * 0.5 + 0.4
        _bhd = bd * 0.5 + 0.4
        _corners = [(-_bhw, -_bhd), (_bhw, -_bhd), (-_bhw, _bhd), (_bhw, _bhd), (0.0, 0.0)]
        by_min = 1e9
        for (lx, lz) in _corners:
            wx_c = bx + lx * c_f_ - lz * s_f_
            wz_c = bz + lx * s_f_ + lz * c_f_
            gh = get_ground_height(wx_c, wz_c)
            if gh < by_min:
                by_min = gh
        by = by_min

        # ---- 赛道防入侵检测 ----
        b_half_diag = math.hypot(bw * 0.5, bd * 0.5)
        safe_margin = b_half_diag + 1.0
        if not is_safe(bx, bz, safe_margin):
            side = -side
            bx = x + rx * dist * side
            bz = z + rz * dist * side
            facing = math.atan2(rx * side, -rz * side)
            # 重新采样五角最低点
            c_f_, s_f_ = math.cos(facing), math.sin(facing)
            by_min = 1e9
            for (lx, lz) in _corners:
                wx_c = bx + lx * c_f_ - lz * s_f_
                wz_c = bz + lx * s_f_ + lz * c_f_
                gh = get_ground_height(wx_c, wz_c)
                if gh < by_min:
                    by_min = gh
            by = by_min
            if not is_safe(bx, bz, safe_margin):
                i += rng.randint(18, 32)
                continue

        # 日式建筑配色：灰泥白墙、深棕木、浅灰瓦
        JCLRS = [(0.72,0.68,0.62),(0.65,0.60,0.55),(0.68,0.64,0.56),
                 (0.70,0.66,0.60),(0.62,0.58,0.52)]
        base = rng.choice(JCLRS)
        has_roof = True   # 日式建筑永远有屋顶
        rh = rng.uniform(0.6, 1.0)   # 缓坡屋脊高度

        # ---- 建筑类型选择 ----
        b_type_roll = rng.random()
        if dist < 18.0:
            b_type = 'shop' if b_type_roll < 0.7 else 'residential'
        elif dist > 25.0:
            b_type = 'warehouse' if b_type_roll < 0.6 else 'residential'
        else:
            b_type = 'residential' if b_type_roll < 0.6 else 'shop'

        # 计算楼层数（日式层高 2.8~3.2m）
        floors = max(2, int(bh / 3.0))

        yield (bx, by, bz, bw, bd, bh, facing, base, has_roof, rh, b_type, floors)

        count += 1
        i += rng.randint(18, 32)


def get_building_positions():
    """
    返回 (N, 4) float32 —— 每行: (bx, by, bz, 避让半径)。
    供 build_tree_instances() 避免树与建筑穿模。
    """
    specs = list(_iter_building_specs())
    if not specs:
        return np.zeros((0, 4), dtype=np.float32)
    arr = np.zeros((len(specs), 4), dtype=np.float32)
    for i, s in enumerate(specs):
        bx, by, bz, bw, bd, bh, *_ = s
        # 避让半径 = 建筑对角线的一半 + 树冠余量
        br = math.hypot(bw, bd) * 0.5 + 2.5
        arr[i] = (bx, by, bz, br)
    return arr


def build_lamp_light_cone():
    """光锥体积光 — 已缓存（缓存名挂 path 指纹）"""
    return _build_or_load_env("akina_lamp_cone_v1", _build_lamp_light_cone_impl)

def _build_lamp_light_cone_impl():
    """
    路灯体积光锥：顶点在灯罩底面中心，向下发散的圆锥体。
    顶点色：顶部暖黄不透明，底部完全透明。
    渲染时使用加法混合 + 关闭深度写入。
    """
    path, tangents = get_akina_path(return_tangents=True)[:2]
    verts = []
    def tri(p1, p2, p3, n, uv, c1, c2, c3):
        """三个顶点各自独立颜色（RGBA）"""
        verts.append([*p1, *n, *uv[0], *c1])
        verts.append([*p2, *n, *uv[1], *c2])
        verts.append([*p3, *n, *uv[2], *c3])

    SEG = 12           # 圆锥分割数
    CONE_H = 4.0       # 光锥高度（向下）
    BASE_R = 2.5       # 底部半径
    TOP_COL = (1.00, 0.85, 0.50, 0.65)   # 顶部：暖黄，半透明
    BOT_COL = (1.00, 0.85, 0.50, 0.0)    # 底部：完全透明

    for (px, py, pz, ax, ay, az, *_rest) in _iter_light_positions(path, tangents):
        # 圆锥顶点 = 灯罩底面中心
        apex_y = ay - 0.05 * 0.5 - 0.14  # 灯罩底面 Y
        apex = (ax, apex_y, az)

        # 圆锥底部：正下方 CONE_H 米
        base_y = apex_y - CONE_H
        ground_y = get_ground_height(ax, az)
        if base_y < ground_y + 0.02:
            base_y = ground_y + 0.02  # 不穿透地面

        # 生成圆锥面
        for k in range(SEG):
            a1 = 2 * math.pi * k / SEG
            a2 = 2 * math.pi * (k + 1) / SEG

            b1 = (ax + BASE_R * math.cos(a1), base_y, az + BASE_R * math.sin(a1))
            b2 = (ax + BASE_R * math.cos(a2), base_y, az + BASE_R * math.sin(a2))

            # 面法线（从锥面朝外）
            mid_x = (math.cos(a1) + math.cos(a2)) * 0.5
            mid_z = (math.sin(a1) + math.sin(a2)) * 0.5
            ml = math.hypot(mid_x, mid_z)
            if ml > 1e-8:
                mid_x /= ml
                mid_z /= ml

            # 锥面坡度法线
            slope = CONE_H / BASE_R if BASE_R > 1e-8 else 1.0
            n_len = math.hypot(slope, 1.0)
            nx = mid_x / n_len
            ny = slope / n_len
            nz = mid_z / n_len

            # 两个三角形组成一个四边形面（tri1: apex, b1, b2）
            tri(apex, b1, b2,
                (nx, ny, nz),
                [(0, 0)] * 3,
                TOP_COL,       # 顶点 = 顶部色
                BOT_COL,       # b1 = 底部色（透明）
                BOT_COL)       # b2 = 底部色（透明）

        # 底部圆盘（填充锥底，向地面投射）
        for k in range(SEG):
            a1 = 2 * math.pi * k / SEG
            a2 = 2 * math.pi * (k + 1) / SEG
            center = (ax, base_y, az)
            b1 = (ax + BASE_R * math.cos(a1), base_y, az + BASE_R * math.sin(a1))
            b2 = (ax + BASE_R * math.cos(a2), base_y, az + BASE_R * math.sin(a2))
            tri(center, b1, b2,
                (0, 1, 0),
                [(0, 0)] * 3,
                BOT_COL, BOT_COL, BOT_COL)

    return np.array(verts, dtype=np.float32)


def _build_aabb_verts(verts, tri, quad, P, x0, x1, z0, z1, y0, y1,
                      n_front, n_back, n_left, n_right, n_top, n_bottom,
                      c_top, c_bottom, c_front, c_back, c_left, c_right):
    """
    辅助函数：生成一个轴对齐盒子的 6 个面（12 个三角形）。
    所有参数已在外层准备好，直接传入。
    """
    quad(P(x0,y1,z0),P(x1,y1,z0),P(x1,y1,z1),P(x0,y1,z1), n_top,
         [(0,0),(1,0),(1,1),(0,1)], c_top)
    quad(P(x0,y0,z0),P(x1,y0,z0),P(x1,y0,z1),P(x0,y0,z1), n_bottom,
         [(0,0),(1,0),(1,1),(0,1)], c_bottom)
    quad(P(x0,y0,z1),P(x1,y0,z1),P(x1,y1,z1),P(x0,y1,z1), n_front,
         [(0,1),(1,1),(1,0),(0,0)], c_front)
    quad(P(x0,y0,z0),P(x1,y0,z0),P(x1,y1,z0),P(x0,y1,z0), n_back,
         [(0,1),(1,1),(1,0),(0,0)], c_back)
    quad(P(x0,y0,z0),P(x0,y0,z1),P(x0,y1,z1),P(x0,y1,z0), n_left,
         [(0,1),(1,1),(1,0),(0,0)], c_left)
    quad(P(x1,y0,z0),P(x1,y0,z1),P(x1,y1,z1),P(x1,y1,z0), n_right,
         [(0,1),(1,1),(1,0),(0,0)], c_right)


# ================================================================
# 日式建筑群辅助函数（v5: 8个独立几何特征）
# ================================================================

def _build_window_recess(verts, tri, quad, P, x0, x1, z0, z1, y0, y1,
                         n_front, n_back, n_left, n_right, n_top, n_bottom,
                         base_h, floors, bw, bd, b_type, rng):
    """
    特征1: 窗户（v3：凸出墙面修复版）

    ★ 主墙体 AABB 是完整的实体 quad，
      旧版玻璃在 z1 - recess（墙内侧），被主墙完全挡住。
      修复：窗户整位移到墙外（凸出），玻璃紧贴墙外，窗框凸出 8cm。
    """
    win_w = 0.7
    win_h = 0.55
    spacing = 1.4
    frame_w = 0.05
    frame_out = 0.08        # 窗框凸出量
    f_eps = 0.005

    WOOD_FRAME = (0.25, 0.15, 0.08, 1.0)
    GLASS_DARK = (0.08, 0.10, 0.14, 1.0)
    GLASS_LIT  = (1.80, 1.65, 1.05, 1.0)

    z_glass = z1 + f_eps                 # 玻璃面（紧贴墙外）
    z_frame_front = z_glass + frame_out  # 窗框前表面

    floor_h = (y1 - y0) / floors
    cols = max(2, int(bw / spacing))

    for fl in range(floors):
        wy = y0 + base_h + 0.3 + fl * floor_h
        if wy + win_h > y1 - 0.3:
            break
        for col in range(cols):
            wx = x0 + (col + 0.5) * (bw / cols)

            # 底层中间留门
            if fl == 0 and col == cols // 2:
                continue
            if fl == 0 and b_type == 'shop' and col in (cols // 2 - 1, cols // 2):
                continue

            lit = rng.random() < 0.40
            glass_c = GLASS_LIT if lit else GLASS_DARK

            wx0, wx1 = wx - win_w/2, wx + win_w/2
            wy0, wy1_w = wy, wy + win_h

            # ---- 玻璃面 ----
            quad(P(wx0, wy0, z_glass),
                 P(wx1, wy0, z_glass),
                 P(wx1, wy1_w, z_glass),
                 P(wx0, wy1_w, z_glass),
                 n_front, [(0, 0)] * 4, glass_c)

            # ---- 窗框正面 4 条 ----
            # 上横条
            quad(P(wx0 - frame_w, wy1_w, z_frame_front),
                 P(wx1 + frame_w, wy1_w, z_frame_front),
                 P(wx1 + frame_w, wy1_w + frame_w, z_frame_front),
                 P(wx0 - frame_w, wy1_w + frame_w, z_frame_front),
                 n_front, [(0, 0)] * 4, WOOD_FRAME)
            # 下横条
            quad(P(wx0 - frame_w, wy0 - frame_w, z_frame_front),
                 P(wx1 + frame_w, wy0 - frame_w, z_frame_front),
                 P(wx1 + frame_w, wy0, z_frame_front),
                 P(wx0 - frame_w, wy0, z_frame_front),
                 n_front, [(0, 0)] * 4, WOOD_FRAME)
            # 左竖条
            quad(P(wx0 - frame_w, wy0, z_frame_front),
                 P(wx0, wy0, z_frame_front),
                 P(wx0, wy1_w, z_frame_front),
                 P(wx0 - frame_w, wy1_w, z_frame_front),
                 n_front, [(0, 0)] * 4, WOOD_FRAME)
            # 右竖条
            quad(P(wx1, wy0, z_frame_front),
                 P(wx1 + frame_w, wy0, z_frame_front),
                 P(wx1 + frame_w, wy1_w, z_frame_front),
                 P(wx1, wy1_w, z_frame_front),
                 n_front, [(0, 0)] * 4, WOOD_FRAME)

            # ---- 窗框外侧面（从玻璃面到窗框前表面） ----
            # 上侧
            quad(P(wx0 - frame_w, wy1_w + frame_w, z_glass),
                 P(wx1 + frame_w, wy1_w + frame_w, z_glass),
                 P(wx1 + frame_w, wy1_w + frame_w, z_frame_front),
                 P(wx0 - frame_w, wy1_w + frame_w, z_frame_front),
                 n_top, [(0, 0)] * 4, WOOD_FRAME)
            # 下侧
            quad(P(wx0 - frame_w, wy0 - frame_w, z_frame_front),
                 P(wx1 + frame_w, wy0 - frame_w, z_frame_front),
                 P(wx1 + frame_w, wy0 - frame_w, z_glass),
                 P(wx0 - frame_w, wy0 - frame_w, z_glass),
                 n_bottom, [(0, 0)] * 4, WOOD_FRAME)
            # 左侧
            quad(P(wx0 - frame_w, wy0 - frame_w, z_glass),
                 P(wx0 - frame_w, wy1_w + frame_w, z_glass),
                 P(wx0 - frame_w, wy1_w + frame_w, z_frame_front),
                 P(wx0 - frame_w, wy0 - frame_w, z_frame_front),
                 n_left, [(0, 0)] * 4, WOOD_FRAME)
            # 右侧
            quad(P(wx1 + frame_w, wy0 - frame_w, z_frame_front),
                 P(wx1 + frame_w, wy1_w + frame_w, z_frame_front),
                 P(wx1 + frame_w, wy1_w + frame_w, z_glass),
                 P(wx1 + frame_w, wy0 - frame_w, z_glass),
                 n_right, [(0, 0)] * 4, WOOD_FRAME)


def _build_pilasters(verts, tri, quad, P, x0, x1, z0, z1, y0, y1,
                     n_front, n_back, n_left, n_right, n_top, n_bottom,
                     base_h, bw, bd):
    """
    特征2: 竖向壁柱（v3：凸出方向修复）

    ★ 旧版正面画在 z1 + f_eps（墙面位置），凸出量为0，被主墙遮住。
      修复：正面凸出到 z1 + f_eps + pillar_out，形成立体柱体。
    """
    pillar_out = 0.08
    pillar_w = 0.15
    PIL_COL = (0.22, 0.14, 0.07, 1.0)
    f_eps = 0.01

    z_back = z1 + f_eps                     # 贴近墙面
    z_front = z1 + f_eps + pillar_out       # 凸出正面

    spacing = 2.0
    n_segments = max(1, int(bw / spacing))

    # ---- 正面壁柱 ----
    for seg in range(n_segments + 1):
        px = x0 + seg * (bw / n_segments)

        # 正面（凸出）
        quad(P(px - pillar_w/2, y0 + base_h, z_front),
             P(px + pillar_w/2, y0 + base_h, z_front),
             P(px + pillar_w/2, y1, z_front),
             P(px - pillar_w/2, y1, z_front),
             n_front, [(0, 0)] * 4, PIL_COL)

        # 顶面
        quad(P(px - pillar_w/2, y1, z_back),
             P(px + pillar_w/2, y1, z_back),
             P(px + pillar_w/2, y1, z_front),
             P(px - pillar_w/2, y1, z_front),
             n_top, [(0, 0)] * 4, PIL_COL)

        # 底面
        quad(P(px - pillar_w/2, y0 + base_h, z_front),
             P(px + pillar_w/2, y0 + base_h, z_front),
             P(px + pillar_w/2, y0 + base_h, z_back),
             P(px - pillar_w/2, y0 + base_h, z_back),
             n_bottom, [(0, 0)] * 4, PIL_COL)

        # 左侧面
        quad(P(px - pillar_w/2, y0 + base_h, z_back),
             P(px - pillar_w/2, y0 + base_h, z_front),
             P(px - pillar_w/2, y1, z_front),
             P(px - pillar_w/2, y1, z_back),
             n_left, [(0, 0)] * 4, PIL_COL)

        # 右侧面
        quad(P(px + pillar_w/2, y0 + base_h, z_front),
             P(px + pillar_w/2, y0 + base_h, z_back),
             P(px + pillar_w/2, y1, z_back),
             P(px + pillar_w/2, y1, z_front),
             n_right, [(0, 0)] * 4, PIL_COL)

    # ---- 左右侧面壁柱 ----
    # 左侧面（法线朝 -X）
    x_side_l_back = x0 - f_eps
    x_side_l_front = x0 - f_eps - pillar_out
    for seg in range(max(1, int(bd / spacing)) + 1):
        pz = z0 + seg * (bd / max(1, int(bd / spacing)))
        if pz > z1:
            break
        # 正面（凸出）
        quad(P(x_side_l_front, y0 + base_h, pz - pillar_w/2),
             P(x_side_l_front, y0 + base_h, pz + pillar_w/2),
             P(x_side_l_front, y1, pz + pillar_w/2),
             P(x_side_l_front, y1, pz - pillar_w/2),
             n_left, [(0, 0)] * 4, PIL_COL)
        # 顶面
        quad(P(x_side_l_back, y1, pz - pillar_w/2),
             P(x_side_l_back, y1, pz + pillar_w/2),
             P(x_side_l_front, y1, pz + pillar_w/2),
             P(x_side_l_front, y1, pz - pillar_w/2),
             n_top, [(0, 0)] * 4, PIL_COL)
        # 底面
        quad(P(x_side_l_front, y0 + base_h, pz - pillar_w/2),
             P(x_side_l_front, y0 + base_h, pz + pillar_w/2),
             P(x_side_l_back, y0 + base_h, pz + pillar_w/2),
             P(x_side_l_back, y0 + base_h, pz - pillar_w/2),
             n_bottom, [(0, 0)] * 4, PIL_COL)

    # 右侧面（法线朝 +X）
    x_side_r_back = x1 + f_eps
    x_side_r_front = x1 + f_eps + pillar_out
    for seg in range(max(1, int(bd / spacing)) + 1):
        pz = z0 + seg * (bd / max(1, int(bd / spacing)))
        if pz > z1:
            break
        # 正面（凸出）
        quad(P(x_side_r_front, y0 + base_h, pz + pillar_w/2),
             P(x_side_r_front, y0 + base_h, pz - pillar_w/2),
             P(x_side_r_front, y1, pz - pillar_w/2),
             P(x_side_r_front, y1, pz + pillar_w/2),
             n_right, [(0, 0)] * 4, PIL_COL)
        # 顶面
        quad(P(x_side_r_back, y1, pz + pillar_w/2),
             P(x_side_r_back, y1, pz - pillar_w/2),
             P(x_side_r_front, y1, pz - pillar_w/2),
             P(x_side_r_front, y1, pz + pillar_w/2),
             n_top, [(0, 0)] * 4, PIL_COL)
        # 底面
        quad(P(x_side_r_front, y0 + base_h, pz + pillar_w/2),
             P(x_side_r_front, y0 + base_h, pz - pillar_w/2),
             P(x_side_r_back, y0 + base_h, pz - pillar_w/2),
             P(x_side_r_back, y0 + base_h, pz + pillar_w/2),
             n_bottom, [(0, 0)] * 4, PIL_COL)


def _build_belt_courses(verts, tri, quad, P, x0, x1, z0, z1, y0, y1,
                        n_front, n_back, n_left, n_right, n_top, n_bottom,
                        base_h, floors):
    """
    特征3: 横向腰线（v3：凸出方向修复）

    ★ 旧版正面画在 z1 + f_eps（墙面位置），凸出量为0，被主墙遮住。
      修复：正面凸出到 z1 + f_eps + belt_out，形成立体水平骨架。
    """
    belt_out = 0.06
    belt_h = 0.15
    BELT_COL = (0.22, 0.14, 0.07, 1.0)
    f_eps = 0.012

    z_back = z1 + f_eps
    z_front = z1 + f_eps + belt_out

    floor_h = (y1 - y0) / floors

    for fl in range(1, floors):
        by_wall = y0 + base_h + fl * floor_h
        if by_wall + belt_h > y1:
            break

        # ---- 正面腰线（凸出） ----
        # 正面
        quad(P(x0, by_wall, z_front),
             P(x1, by_wall, z_front),
             P(x1, by_wall + belt_h, z_front),
             P(x0, by_wall + belt_h, z_front),
             n_front, [(0, 0)] * 4, BELT_COL)
        # 顶面
        quad(P(x0, by_wall + belt_h, z_back),
             P(x1, by_wall + belt_h, z_back),
             P(x1, by_wall + belt_h, z_front),
             P(x0, by_wall + belt_h, z_front),
             n_top, [(0, 0)] * 4, BELT_COL)
        # 底面
        quad(P(x0, by_wall, z_front),
             P(x1, by_wall, z_front),
             P(x1, by_wall, z_back),
             P(x0, by_wall, z_back),
             n_bottom, [(0, 0)] * 4, BELT_COL)

        # ---- 左侧腰线 ----
        quad(P(x0 - f_eps - belt_out, by_wall, z0),
             P(x0 - f_eps - belt_out, by_wall, z1),
             P(x0 - f_eps - belt_out, by_wall + belt_h, z1),
             P(x0 - f_eps - belt_out, by_wall + belt_h, z0),
             n_left, [(0, 0)] * 4, BELT_COL)

        # ---- 右侧腰线 ----
        quad(P(x1 + f_eps + belt_out, by_wall, z1),
             P(x1 + f_eps + belt_out, by_wall, z0),
             P(x1 + f_eps + belt_out, by_wall + belt_h, z0),
             P(x1 + f_eps + belt_out, by_wall + belt_h, z1),
             n_right, [(0, 0)] * 4, BELT_COL)


def _build_layered_roof(verts, tri, quad, P, x0, x1, z0, z1, y1,
                        n_front, n_back, n_left, n_right, n_top, n_bottom):
    """
    特征7: 日式单层寄栋造（四面坡屋顶）
    - 深出檐 1.0m
    - 坡角 25°（约5寸勾配，传统日式缓坡）
    - 屋脊沿 X 方向，长度约为建筑宽度的60%
    - 屋脊两端各有鬼瓦装饰
    - 檐底深色AO阴影
    返回值兼容旧双层接口（y_lower=y1, y_upper=y1+rh）。
    """
    TILE_COL = (0.15, 0.15, 0.17, 1.0)       # 黑瓦
    EAVE_SHADOW = (0.09, 0.09, 0.10, 1.0)    # 檐底深色AO
    RIDGE_COL = (0.18, 0.18, 0.20, 1.0)       # 屋脊装饰
    ONI_COL = (0.10, 0.10, 0.11, 1.0)         # 鬼瓦色
    f_eps = 0.005

    eave = 1.0             # 出檐深度(m)
    slope_deg = 25         # 坡角（度）
    slope = math.radians(slope_deg)

    # 出檐范围
    ex0, ex1 = x0 - eave, x1 + eave
    ez0, ez1 = z0 - eave, z1 + eave

    # 屋脊高度：从 eave 边缘到屋脊的垂直升幅
    # 对于四面坡，屋脊高度由进深（Z方向）决定
    half_depth = (z1 - z0) / 2 + eave   # 从檐口到中心的水平距离
    rh = half_depth * math.tan(slope)
    y_ridge = y1 + rh                    # 屋脊Y坐标

    # 屋脊沿 X 方向，长度约为建筑宽度的 60%
    ridge_len = max(0.6, (x1 - x0) * 0.6)
    rx0, rx1 = -ridge_len / 2, ridge_len / 2
    # 屋脊在 Z 方向居中
    rz = (ez0 + ez1) / 2.0

    # 屋脊两端顶点
    apex_a = P(rx0, y_ridge, rz)   # 左端
    apex_b = P(rx1, y_ridge, rz)   # 右端

    # ======== 四个坡面 ========

    # 前坡面（梯形）：下边=前檐口，上边=屋脊
    quad(P(ex0, y1, ez1), P(ex1, y1, ez1),
         apex_b, apex_a,
         n_front, [(0, 0)] * 4, TILE_COL)

    # 后坡面（梯形）：下边=后檐口，上边=屋脊
    quad(P(ex1, y1, ez0), P(ex0, y1, ez0),
         apex_a, apex_b,
         n_back, [(0, 0)] * 4, TILE_COL)

    # 左侧面（三角形）
    tri(P(ex0, y1, ez1), P(ex0, y1, ez0), apex_a,
        n_left, [(0, 0)] * 3, TILE_COL)

    # 右侧面（三角形）
    tri(P(ex1, y1, ez0), P(ex1, y1, ez1), apex_b,
        n_right, [(0, 0)] * 3, TILE_COL)

    # ======== 屋檐底面（深色AO阴影） ========
    for (p1, p2, p3, p4) in [
        ((ex0, y1, z0), (ex1, y1, z0), (ex1, y1, ez0), (ex0, y1, ez0)),  # 前出檐底
        ((ex0, y1, ez1), (ex1, y1, ez1), (ex0, y1, z1), (ex1, y1, z1)),  # 后出檐底
        ((x0, y1, ez1), (x0, y1, ez0), (ex0, y1, ez0), (ex0, y1, ez1)),  # 左出檐底
        ((x1, y1, ez0), (x1, y1, ez1), (ex1, y1, ez1), (ex1, y1, ez0)),  # 右出檐底
    ]:
        quad(P(*p1), P(*p2), P(*p3), P(*p4),
             n_bottom, [(0, 0)] * 4, EAVE_SHADOW)

    # ======== 鬼瓦（屋脊两端小方块） ========
    oni_w, oni_h, oni_d = 0.15, 0.20, 0.12
    for gx, g_sign in [(rx0, -1), (rx1, 1)]:
        gz = rz
        # 前面
        quad(P(gx - oni_w / 2, y_ridge, gz - oni_d),
             P(gx + oni_w / 2, y_ridge, gz - oni_d),
             P(gx + oni_w / 2, y_ridge + oni_h, gz - oni_d),
             P(gx - oni_w / 2, y_ridge + oni_h, gz - oni_d),
             n_front, [(0, 0)] * 4, ONI_COL)
        # 后面
        quad(P(gx + oni_w / 2, y_ridge, gz + oni_d),
             P(gx - oni_w / 2, y_ridge, gz + oni_d),
             P(gx - oni_w / 2, y_ridge + oni_h, gz + oni_d),
             P(gx + oni_w / 2, y_ridge + oni_h, gz + oni_d),
             n_back, [(0, 0)] * 4, ONI_COL)
        # 左面
        quad(P(gx - oni_w / 2, y_ridge, gz - oni_d),
             P(gx - oni_w / 2, y_ridge, gz + oni_d),
             P(gx - oni_w / 2, y_ridge + oni_h, gz + oni_d),
             P(gx - oni_w / 2, y_ridge + oni_h, gz - oni_d),
             n_left, [(0, 0)] * 4, ONI_COL)
        # 右面
        quad(P(gx + oni_w / 2, y_ridge, gz + oni_d),
             P(gx + oni_w / 2, y_ridge, gz - oni_d),
             P(gx + oni_w / 2, y_ridge + oni_h, gz - oni_d),
             P(gx + oni_w / 2, y_ridge + oni_h, gz + oni_d),
             n_right, [(0, 0)] * 4, ONI_COL)

    # ======== 屋脊装饰条 ========
    ridge_w = 0.06
    quad(P(rx0, y_ridge - 0.02, rz - ridge_w),
         P(rx1, y_ridge - 0.02, rz - ridge_w),
         P(rx1, y_ridge + 0.04, rz + ridge_w),
         P(rx0, y_ridge + 0.04, rz + ridge_w),
         n_front, [(0, 0)] * 4, RIDGE_COL)
    quad(P(rx1, y_ridge - 0.02, rz + ridge_w),
         P(rx0, y_ridge - 0.02, rz + ridge_w),
         P(rx0, y_ridge + 0.04, rz + ridge_w),
         P(rx1, y_ridge + 0.04, rz + ridge_w),
         n_back, [(0, 0)] * 4, RIDGE_COL)
    quad(P(rx0, y_ridge - 0.02, rz - ridge_w),
         P(rx0, y_ridge - 0.02, rz + ridge_w),
         P(rx0, y_ridge + 0.04, rz + ridge_w),
         P(rx0, y_ridge + 0.04, rz - ridge_w),
         n_left, [(0, 0)] * 4, RIDGE_COL)
    quad(P(rx1, y_ridge - 0.02, rz + ridge_w),
         P(rx1, y_ridge - 0.02, rz - ridge_w),
         P(rx1, y_ridge + 0.04, rz - ridge_w),
         P(rx1, y_ridge + 0.04, rz + ridge_w),
         n_right, [(0, 0)] * 4, RIDGE_COL)

    # 返回值兼容旧接口：
    # y_lower_ridge = y1（下层屋顶底面），y_upper_ridge = y_ridge（屋脊顶）
    # eave_lower = eave, 出檐范围复用
    return y1, y_ridge, eave, ex0, ex1, ez0, ez1


def _build_rafter_tails(verts, tri, quad, P, x0, x1, z0, z1, y1,
                        n_front, n_back, n_left, n_right, n_bottom,
                        eave_lower, ex0_l, ex1_l, ez0_l, ez1_l):
    """
    特征4: 檐下椽子
    屋顶出檐下方一排小木椽，只做正面和左右侧面。
    每个椽子是0.06×0.08m截面的小木条，从屋檐边缘延伸到墙顶。
    """
    RAFTER_COL = (0.20, 0.12, 0.06, 1.0)   # 深棕椽木
    r_w = 0.06    # 椽宽(m)
    r_h = 0.08    # 椽高(m)
    spacing = 0.35  # 间距(m)
    r_y = y1 - 0.02  # 椽子Y（紧贴檐底）

    # ---- 正面椽子（沿X排列，从墙面z1到出檐边缘ez1_l） ----
    n_rafters = max(2, int((x1 - x0) / spacing))
    for ri in range(n_rafters + 1):
        rx = x0 + ri * (x1 - x0) / max(1, n_rafters)
        # 椽子从墙顶延伸到出檐边缘
        quad(P(rx - r_w/2, r_y, z1),
             P(rx + r_w/2, r_y, z1),
             P(rx + r_w/2, r_y, ez1_l),
             P(rx - r_w/2, r_y, ez1_l),
             n_bottom, [(0,0)]*4, RAFTER_COL)
        # 椽子前侧面（朝向观察者）
        quad(P(rx - r_w/2, r_y, ez1_l),
             P(rx + r_w/2, r_y, ez1_l),
             P(rx + r_w/2, r_y + r_h, ez1_l),
             P(rx - r_w/2, r_y + r_h, ez1_l),
             n_front, [(0,0)]*4, RAFTER_COL)

    # ---- 左侧面椽子（沿Z排列，从墙面x0到出檐边缘ex0_l） ----
    n_side = max(2, int((z1 - z0) / spacing))
    for si in range(n_side + 1):
        sz = z0 + si * (z1 - z0) / max(1, n_side)
        quad(P(x0, r_y, sz - r_w/2),
             P(x0, r_y, sz + r_w/2),
             P(ex0_l, r_y, sz + r_w/2),
             P(ex0_l, r_y, sz - r_w/2),
             n_bottom, [(0,0)]*4, RAFTER_COL)
        quad(P(ex0_l, r_y, sz - r_w/2),
             P(ex0_l, r_y, sz + r_w/2),
             P(ex0_l, r_y + r_h, sz + r_w/2),
             P(ex0_l, r_y + r_h, sz - r_w/2),
             n_left, [(0,0)]*4, RAFTER_COL)

    # ---- 右侧面椽子 ----
    for si in range(n_side + 1):
        sz = z0 + si * (z1 - z0) / max(1, n_side)
        quad(P(ex1_l, r_y, sz + r_w/2),
             P(ex1_l, r_y, sz - r_w/2),
             P(x1, r_y, sz - r_w/2),
             P(x1, r_y, sz + r_w/2),
             n_bottom, [(0,0)]*4, RAFTER_COL)
        quad(P(ex1_l, r_y, sz + r_w/2),
             P(ex1_l, r_y, sz - r_w/2),
             P(ex1_l, r_y + r_h, sz - r_w/2),
             P(ex1_l, r_y + r_h, sz + r_w/2),
             n_right, [(0,0)]*4, RAFTER_COL)


def _build_roof_tiles_caps(verts, tri, quad, P, x0, x1, z0, z1, y1,
                           n_front, n_left, n_right, n_top,
                           eave_lower, ex0_l, ex1_l, ez0_l, ez1_l):
    """
    特征6: 屋顶瓦当 — 沿出檐边缘的小圆盘装饰
    用8边形近似圆盘，只做正面和部分侧面以控制顶点数。
    """
    CAP_COL = (0.10, 0.10, 0.11, 1.0)   # 深灰瓦当
    cap_r = 0.10      # 瓦当半径(m)
    spacing = 0.50    # 间距(m)，放宽以控制顶点
    N_SEG = 8         # 8边形近似
    cap_y = y1        # 瓦当贴在屋檐边缘

    # 正面瓦当（沿出檐前边缘 ez1_l）
    n_front_caps = max(2, int((ex1_l - ex0_l) / spacing))
    for fi in range(n_front_caps + 1):
        cx = ex0_l + fi * (ex1_l - ex0_l) / max(1, n_front_caps)
        cz = ez1_l
        _build_wadang_disc(verts, tri, P, cx, cap_y, cz,
                           n_front, cap_r, N_SEG, CAP_COL)

    # 左侧面瓦当（沿出檐左边缘 ex0_l，只做前一半）
    # 侧面圆盘在 Z-Y 平面展开（X 固定）
    n_left_caps = max(2, int((ez1_l - ez0_l) * 0.5 / spacing))
    for li in range(n_left_caps + 1):
        cz = ez0_l + li * (ez1_l - ez0_l) * 0.5 / max(1, n_left_caps)
        cx = ex0_l
        cp = P(cx, cap_y, cz)
        for k in range(N_SEG):
            a1 = 2.0 * math.pi * k / N_SEG
            a2 = 2.0 * math.pi * (k + 1) / N_SEG
            p1 = P(cx, cap_y + cap_r * math.sin(a1), cz + cap_r * math.cos(a1))
            p2 = P(cx, cap_y + cap_r * math.sin(a2), cz + cap_r * math.cos(a2))
            tri(cp, p1, p2, n_left, [(0, 0)] * 3, CAP_COL)

    # 右侧面瓦当（沿出檐右边缘 ex1_l，只做前一半）
    for ri in range(n_left_caps + 1):
        cz = ez0_l + ri * (ez1_l - ez0_l) * 0.5 / max(1, n_left_caps)
        cx = ex1_l
        cp = P(cx, cap_y, cz)
        for k in range(N_SEG):
            a1 = 2.0 * math.pi * k / N_SEG
            a2 = 2.0 * math.pi * (k + 1) / N_SEG
            p1 = P(cx, cap_y + cap_r * math.sin(a1), cz + cap_r * math.cos(a1))
            p2 = P(cx, cap_y + cap_r * math.sin(a2), cz + cap_r * math.cos(a2))
            tri(cp, p1, p2, n_right, [(0, 0)] * 3, CAP_COL)


def _build_wadang_disc(verts, tri, P, cx, cy, cz, normal, radius, n_seg, col):
    """用三角形扇生成一个圆盘（瓦当）— 正面用（X-Y平面展开，法线朝Z）。"""
    cp = P(cx, cy, cz)
    for k in range(n_seg):
        a1 = 2.0 * math.pi * k / n_seg
        a2 = 2.0 * math.pi * (k + 1) / n_seg
        p1 = P(cx + radius * math.cos(a1), cy + radius * math.sin(a1), cz)
        p2 = P(cx + radius * math.cos(a2), cy + radius * math.sin(a2), cz)
        tri(cp, p1, p2, normal, [(0, 0)] * 3, col)


def _build_entrance_awning(verts, tri, quad, P, x0, x1, z0, z1, y0, y1,
                           n_front, n_left, n_right, n_top, n_bottom,
                           bw, bd, b_type):
    """
    特征5: 门头雨篷 + 台阶
    入口处加3级台阶和倾斜雨篷。
    """
    WOOD_COL = (0.25, 0.15, 0.08, 1.0)     # 雨篷深棕木
    STONE_COL = (0.45, 0.44, 0.42, 1.0)    # 台阶石材
    f_eps = 0.015

    # 门宽度（与窗洞函数一致）
    door_w = 1.2 if b_type == 'shop' else 0.9

    # ---- 台阶（3级） ----
    step_w = door_w + 0.4
    step_d = 0.25   # 每级深
    step_h = 0.15   # 每级高
    n_steps = 3

    for si in range(n_steps):
        sy0 = y0 + si * step_h
        sy1 = sy0 + step_h
        sz0 = z1 + f_eps + si * step_d
        sz1_w = z1 + f_eps + (si + 1) * step_d
        sx0, sx1 = -step_w/2, step_w/2

        # 台阶顶面
        quad(P(sx0, sy1, sz0),
             P(sx1, sy1, sz0),
             P(sx1, sy1, sz1_w),
             P(sx0, sy1, sz1_w),
             n_top, [(0, 0)] * 4, STONE_COL)
        # 台阶前面
        quad(P(sx0, sy0, sz1_w),
             P(sx1, sy0, sz1_w),
             P(sx1, sy1, sz1_w),
             P(sx0, sy1, sz1_w),
             n_front, [(0, 0)] * 4, STONE_COL)
        # 台阶左面
        quad(P(sx0, sy0, sz0),
             P(sx0, sy0, sz1_w),
             P(sx0, sy1, sz1_w),
             P(sx0, sy1, sz0),
             n_left, [(0, 0)] * 4, STONE_COL)
        # 台阶右面
        quad(P(sx1, sy0, sz1_w),
             P(sx1, sy0, sz0),
             P(sx1, sy1, sz0),
             P(sx1, sy1, sz1_w),
             n_right, [(0, 0)] * 4, STONE_COL)

    # ---- 雨篷 ----
    awning_w = door_w + 0.6
    awning_overhang = 0.8    # 挑出量(m)
    awning_angle = math.radians(15)  # 倾斜15°（外侧低）
    awning_y = y0 + 2.0      # 雨篷安装高度
    awning_thick = 0.06      # 雨篷板厚度

    ax0, ax1 = -awning_w/2, awning_w/2
    az_inner = z1 + f_eps * 3       # 靠墙侧
    az_outer = z1 + awning_overhang # 挑出侧

    # 外侧低：y_outer = awning_y - awning_overhang * tan(15°)
    y_drop = awning_overhang * math.tan(awning_angle)
    ay_outer = awning_y - y_drop

    # 雨篷顶面
    quad(P(ax0, awning_y, az_inner),
         P(ax1, awning_y, az_inner),
         P(ax1, ay_outer, az_outer),
         P(ax0, ay_outer, az_outer),
         n_top, [(0, 0)] * 4, WOOD_COL)
    # 雨篷底面
    quad(P(ax0, ay_outer - awning_thick, az_outer),
         P(ax1, ay_outer - awning_thick, az_outer),
         P(ax1, awning_y - awning_thick, az_inner),
         P(ax0, awning_y - awning_thick, az_inner),
         n_bottom, [(0, 0)] * 4, WOOD_COL)
    # 雨篷前面
    quad(P(ax0, awning_y - awning_thick, az_inner),
         P(ax1, awning_y - awning_thick, az_inner),
         P(ax1, awning_y, az_inner),
         P(ax0, awning_y, az_inner),
         n_front, [(0, 0)] * 4, WOOD_COL)
    # 雨篷前缘
    quad(P(ax0, ay_outer - awning_thick, az_outer),
         P(ax1, ay_outer - awning_thick, az_outer),
         P(ax1, ay_outer, az_outer),
         P(ax0, ay_outer, az_outer),
         n_front, [(0, 0)] * 4, WOOD_COL)
    # 雨篷左缘
    quad(P(ax0, awning_y, az_inner),
         P(ax0, ay_outer, az_outer),
         P(ax0, ay_outer - awning_thick, az_outer),
         P(ax0, awning_y - awning_thick, az_inner),
         n_left, [(0, 0)] * 4, WOOD_COL)
    # 雨篷右缘
    quad(P(ax1, ay_outer, az_outer),
         P(ax1, awning_y, az_inner),
         P(ax1, awning_y - awning_thick, az_inner),
         P(ax1, ay_outer - awning_thick, az_outer),
         n_right, [(0, 0)] * 4, WOOD_COL)

    # ---- 雨篷柱（2根，直径0.08m的方形柱） ----
    col_w = 0.08
    col_y_base = y0
    col_y_top = ay_outer
    for cx in (ax0 + col_w, ax1 - col_w):
        quad(P(cx - col_w/2, col_y_base, az_outer - col_w/2),
             P(cx + col_w/2, col_y_base, az_outer - col_w/2),
             P(cx + col_w/2, col_y_top, az_outer - col_w/2),
             P(cx - col_w/2, col_y_top, az_outer - col_w/2),
             n_front, [(0, 0)] * 4, WOOD_COL)
        quad(P(cx + col_w/2, col_y_base, az_outer + col_w/2),
             P(cx - col_w/2, col_y_base, az_outer + col_w/2),
             P(cx - col_w/2, col_y_top, az_outer + col_w/2),
             P(cx + col_w/2, col_y_top, az_outer + col_w/2),
             n_front, [(0, 0)] * 4, WOOD_COL)
        quad(P(cx - col_w/2, col_y_base, az_outer - col_w/2),
             P(cx - col_w/2, col_y_base, az_outer + col_w/2),
             P(cx - col_w/2, col_y_top, az_outer + col_w/2),
             P(cx - col_w/2, col_y_top, az_outer - col_w/2),
             n_left, [(0, 0)] * 4, WOOD_COL)
        quad(P(cx + col_w/2, col_y_base, az_outer + col_w/2),
             P(cx + col_w/2, col_y_base, az_outer - col_w/2),
             P(cx + col_w/2, col_y_top, az_outer - col_w/2),
             P(cx + col_w/2, col_y_top, az_outer + col_w/2),
             n_right, [(0, 0)] * 4, WOOD_COL)


def _build_chimney(verts, tri, quad, P, x0, x1, z0, z1, y1,
                   n_front, n_back, n_left, n_right, n_top, n_bottom,
                   y_lower_ridge, rng):
    """
    特征8: 烟囱（随机30%概率出现）
    细长盒子+顶部小帽，位于下层面屋顶上。
    """
    if rng.random() > 0.30:
        return

    chim_w = 0.3    # 宽(m)
    chim_d = 0.3    # 深(m)
    chim_h = rng.uniform(1.0, 1.8)  # 高度(m)
    CHIM_COL = (0.48, 0.47, 0.44, 1.0)   # 与基座同石材色
    CAP_COL = (0.15, 0.15, 0.17, 1.0)    # 帽檐色

    # 在下层屋脊附近随机放置
    cx = x0 + rng.uniform(0.3, 0.7) * (x1 - x0)
    cz = z0 + rng.uniform(0.3, 0.7) * (z1 - z0)
    # 放置在接近屋脊处（z=0附近）
    cz = (z0 + z1) / 2 + rng.uniform(-1.0, 1.0)

    cy0 = y_lower_ridge
    cy1 = y_lower_ridge + chim_h

    # 烟囱主体
    _build_aabb_verts(verts, tri, quad, P,
        cx - chim_w/2, cx + chim_w/2,
        cz - chim_d/2, cz + chim_d/2,
        cy0, cy1,
        n_front, n_back, n_left, n_right, n_top, n_bottom,
        CHIM_COL, CHIM_COL, CHIM_COL, CHIM_COL, CHIM_COL, CHIM_COL)

    # 烟囱帽（比主体宽0.1m每侧，高0.1m）
    cap_w = chim_w + 0.1
    cap_d = chim_d + 0.1
    cap_h = 0.10
    _build_aabb_verts(verts, tri, quad, P,
        cx - cap_w/2, cx + cap_w/2,
        cz - cap_d/2, cz + cap_d/2,
        cy1, cy1 + cap_h,
        n_front, n_back, n_left, n_right, n_top, n_bottom,
        CAP_COL, CAP_COL, CAP_COL, CAP_COL, CAP_COL, CAP_COL)


def build_buildings():
    """建筑群 — 已缓存（_iter_building_specs 沿 path 布点，缓存名挂 path 指纹）"""
    return _build_or_load_env("akina_buildings_v1", _build_buildings_impl)

def _build_buildings_impl():
    """
    v5: 日式建筑群 — 8大几何特征重写

    核心特征：
    1. 窗洞凹陷（凹进12cm + 窗框 + 40%亮灯）
    2. 竖向壁柱（每2m一根凸出木柱）
    3. 横向腰线（每层分界凸出横条）
    4. 檐下椽子（出檐下方小木椽）
    5. 门头雨篷+台阶（入口处3级台阶+倾斜雨篷）
    6. 屋顶瓦当（出檐边缘圆盘装饰）
    7. 屋顶分层（双层屋面：下层缓坡大出檐 + 上层陡坡小出檐）
    8. 烟囱（30%概率出现的细长盒+帽）
    """
    verts = []

    def tri(p1, p2, p3, n, uv, c):
        verts.append([*p1, *n, *uv[0], *c])
        verts.append([*p2, *n, *uv[1], *c])
        verts.append([*p3, *n, *uv[2], *c])

    def quad(p1, p2, p3, p4, n, uv, c):
        tri(p1, p2, p3, n, [uv[0], uv[1], uv[2]], c)
        tri(p1, p3, p4, n, [uv[0], uv[2], uv[3]], c)

    rng = random.Random(20240118)

    for spec in _iter_building_specs():
        bx, by, bz, bw, bd, bh, facing, base, has_roof, rh, b_type, floors = spec
        c_f, s_f = math.cos(facing), math.sin(facing)

        def P(px, py, pz):
            return (bx + px * c_f - pz * s_f, py, bz + px * s_f + pz * c_f)

        x0, x1 = -bw / 2, bw / 2
        z0, z1 = -bd / 2, bd / 2
        y0, y1 = by, by + bh

        # ---- 各面颜色（带假AO） ----
        c_top = tuple(min(1.0, c * 0.95) for c in base) + (1.0,)
        c_bottom = tuple(c * 0.40 for c in base) + (1.0,)
        c_front = tuple(min(1.0, c * 1.02) for c in base) + (1.0,)
        c_back = tuple(c * 0.60 for c in base) + (1.0,)
        c_left = tuple(c * 0.70 for c in base) + (1.0,)
        c_right = tuple(c * 0.75 for c in base) + (1.0,)

        n_front = (s_f, 0, c_f)
        n_back = (-s_f, 0, -c_f)
        n_left = (-c_f, 0, -s_f)
        n_right = (c_f, 0, s_f)
        n_top = (0, 1, 0)
        n_bottom = (0, -1, 0)

        # ================================================================
        # 基座：石砌基座（底部0.4m，微凸出墙体6cm）
        # ================================================================
        base_h = 0.4
        base_out = 0.06
        STONE_COL = (0.48, 0.47, 0.44, 1.0)
        STONE_DARK = (0.30, 0.29, 0.27, 1.0)

        # ★ 延伸基座向下2m（兜底防止地形起伏导致建筑悬空）
        EXTEND_H = 2.0
        _build_aabb_verts(verts, tri, quad, P,
            x0 - base_out, x1 + base_out,
            z0 - base_out, z1 + base_out,
            y0 - EXTEND_H, y0,
            n_front, n_back, n_left, n_right, n_top, n_bottom,
            STONE_COL, STONE_DARK, STONE_COL, STONE_DARK,
            STONE_DARK, STONE_DARK)

        # 原石砌基座（0.4m高）
        _build_aabb_verts(verts, tri, quad, P,
            x0 - base_out, x1 + base_out,
            z0 - base_out, z1 + base_out,
            y0, y0 + base_h,
            n_front, n_back, n_left, n_right, n_top, n_bottom,
            STONE_COL, STONE_DARK, STONE_COL, STONE_DARK,
            STONE_DARK, STONE_DARK)

        # ================================================================
        # 主体墙体：灰泥白墙
        # ================================================================
        _build_aabb_verts(verts, tri, quad, P,
            x0, x1, z0, z1, y0, y1,
            n_front, n_back, n_left, n_right, n_top, n_bottom,
            c_top, c_bottom, c_front, c_back, c_left, c_right)

        # ================================================================
        # 特征2: 竖向壁柱 — 墙面竖向凸出木柱
        # ================================================================
        _build_pilasters(verts, tri, quad, P, x0, x1, z0, z1, y0, y1,
                         n_front, n_back, n_left, n_right, n_top, n_bottom,
                         base_h, bw, bd)

        # ================================================================
        # 特征3: 横向腰线 — 每层分界处横条
        # ================================================================
        _build_belt_courses(verts, tri, quad, P, x0, x1, z0, z1, y0, y1,
                            n_front, n_back, n_left, n_right, n_top, n_bottom,
                            base_h, floors)

        # ================================================================
        # 特征1: 窗洞凹陷 ★ — 凹进12cm的窗户
        # ================================================================
        _build_window_recess(verts, tri, quad, P, x0, x1, z0, z1, y0, y1,
                             n_front, n_back, n_left, n_right, n_top, n_bottom,
                             base_h, floors, bw, bd, b_type, rng)

        # ================================================================
        # 特征7: 屋顶分层 — 双层屋面
        # ================================================================
        y_lower_ridge, y_upper_ridge, eave_lower, ex0_l, ex1_l, ez0_l, ez1_l = \
            _build_layered_roof(verts, tri, quad, P, x0, x1, z0, z1, y1,
                                n_front, n_back, n_left, n_right, n_top, n_bottom)

        # ================================================================
        # 特征4: 檐下椽子 — 出檐下方小木椽
        # ================================================================
        _build_rafter_tails(verts, tri, quad, P, x0, x1, z0, z1, y1,
                            n_front, n_back, n_left, n_right, n_bottom,
                            eave_lower, ex0_l, ex1_l, ez0_l, ez1_l)

        # ================================================================
        # 特征6: 屋顶瓦当 — 出檐边缘圆盘装饰
        # ================================================================
        _build_roof_tiles_caps(verts, tri, quad, P, x0, x1, z0, z1, y1,
                               n_front, n_left, n_right, n_top,
                               eave_lower, ex0_l, ex1_l, ez0_l, ez1_l)

        # ================================================================
        # 特征5: 门头雨篷 + 台阶
        # ================================================================
        _build_entrance_awning(verts, tri, quad, P, x0, x1, z0, z1, y0, y1,
                               n_front, n_left, n_right, n_top, n_bottom,
                               bw, bd, b_type)

        # ================================================================
        # 特征8: 烟囱（30%概率）
        # ================================================================
        _build_chimney(verts, tri, quad, P, x0, x1, z0, z1, y1,
                       n_front, n_back, n_left, n_right, n_top, n_bottom,
                       y_lower_ridge, rng)

    return np.array(verts, dtype=np.float32)
# ============================================================
# 弯道标志牌（凸面镜杆 + 箭头牌牌面）+ 反光道钉
# ============================================================

def build_roadside_signs():
    """弯道标志牌（凸面镜杆 + 箭头牌牌面）— 已缓存（弯道簇由 path 曲率重算）"""
    return _build_or_load_env("akina_signs_v4", _build_roadside_signs_impl)


def build_roadside_signs_emissive():
    """凸面镜镜面自发光 — 已缓存（缓存名挂 path 指纹）"""
    return _build_or_load_env("akina_signs_em_v2", _build_roadside_signs_em_impl)


def build_road_studs():
    """反光道钉 — 已缓存（自发光，白右黄左，直道 6m / 弯道 3m）"""
    return _build_or_load_env("akina_studs_v1", _build_road_studs_impl)


def _build_roadside_signs_impl():
    path, tangents = get_akina_path(return_tangents=True)[:2]
    n = len(path)
    curv = _compute_path_curvature_signed(path)
    cum = np.zeros(n)
    for i in range(1, n):
        cum[i] = cum[i-1] + math.hypot(path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])

    verts = []

    def quad(p1, p2, p3, p4, nn, c):
        for pp in (p1, p2, p3, p1, p3, p4):
            verts.append([*pp, *nn, 0, 0, *c])

    # ★ 双面薄板：正反两片（本项目几何 Pass 开背面剔除，单面片一旦绕序/朝向
    #   写反就彻底不可见 —— 历史 bug：箭头牌牌面朝路外，路上永远只能看到背面）。
    #   两片深度相同，但背面剔除保证同一像素只通过一片，不会 z-fight。
    def panel(p1, p2, p3, p4, nn, c):
        for pp in (p1, p2, p3, p1, p3, p4):
            verts.append([*pp, *nn, 0, 0, *c])
        nn2 = (-nn[0], -nn[1], -nn[2])
        for pp in (p1, p3, p2, p1, p4, p3):
            verts.append([*pp, *nn2, 0, 0, *c])

    def panel3(a, b, c_, nn, col):
        """双面三角"""
        verts.append([*a, *nn, 0, 0, *col])
        verts.append([*b, *nn, 0, 0, *col])
        verts.append([*c_, *nn, 0, 0, *col])
        nn2 = (-nn[0], -nn[1], -nn[2])
        verts.append([*a, *nn2, 0, 0, *col])
        verts.append([*c_, *nn2, 0, 0, *col])
        verts.append([*b, *nn2, 0, 0, *col])

    def _at_arc(s_target):
        """弧长 → (外侧位置, 外法线, 回看方向, 索引)"""
        i = min(n - 1, max(1, int(np.searchsorted(cum, s_target))))
        x, z, y, w, bank = path[i]
        tx, ty, tz = tangents[i]
        nx2d, nz2d = -tz, tx
        L = math.hypot(nx2d, nz2d) or 1e-6
        nx2d, nz2d = nx2d / L, nz2d / L
        side = _outer_side_of(curv, i)
        hw = w * 0.5
        off = hw + 1.5
        px = x + nx2d * off * side
        pz = z + nz2d * off * side
        py = surface_height_at(px, pz)
        return px, py, pz, nx2d * side, nz2d * side, tx, tz, i

    POLE = (0.30, 0.31, 0.33)
    BOARD_Y = (0.95, 0.76, 0.06)   # 警示黄
    BOARD_K = (0.05, 0.05, 0.05)   # 黑色箭纹
    FRAME = (0.45, 0.45, 0.47)

    clusters = _roadside_corner_clusters(curv)
    print(f"[标志牌] 检测到 {len(clusters)} 个弯道簇")

    for (i0, i1, peak) in clusters:
        # ---- 凸面镜：入弯前 10m，弯道外侧 ----
        mx, my, mz, onx, onz, ttx, ttz, _ = _at_arc(cum[i0] - 10.0)
        hx, hz = 0.03, 0.03
        # 杆：0.06×2.3m 细柱（四个侧面）
        panel((mx-hx, my, mz-hz), (mx+hx, my, mz-hz), (mx+hx, my+2.3, mz-hz), (mx-hx, my+2.3, mz-hz), (0,0,-1), POLE)
        panel((mx-hx, my, mz+hz), (mx+hx, my, mz+hz), (mx+hx, my+2.3, mz+hz), (mx-hx, my+2.3, mz+hz), (0,0,1), POLE)
        panel((mx-hx, my, mz-hz), (mx-hx, my, mz+hz), (mx-hx, my+2.3, mz+hz), (mx-hx, my+2.3, mz-hz), (-1,0,0), POLE)
        panel((mx+hx, my, mz-hz), (mx+hx, my, mz+hz), (mx+hx, my+2.3, mz+hz), (mx+hx, my+2.3, mz-hz), (1,0,0), POLE)
        # 背板（灰色圆盘扇面，朝向来车方向 = -tangent）
        nfx, nfz = -ttx, -ttz
        ufx, ufz = nfz, -nfx
        cy_m = my + 2.3
        SEG = 10
        # ★ 背板压到 0.05：与橙框(0.075)、镜面自发光(0.105)分层 ≥25mm，
        #   否则远距离深度精度不足 → 橙框与背板 z-fight 闪烁。
        for k in range(SEG):
            a1 = 2*math.pi*k/SEG; a2 = 2*math.pi*(k+1)/SEG
            p1 = (mx + 0.48*(ufx*math.cos(a1) + nfx*0.05), cy_m + 0.48*math.sin(a1), mz + 0.48*(ufz*math.cos(a1) + nfz*0.05))
            p2 = (mx + 0.48*(ufx*math.cos(a2) + nfx*0.05), cy_m + 0.48*math.sin(a2), mz + 0.48*(ufz*math.cos(a2) + nfz*0.05))
            c  = (mx + nfx*0.05, cy_m, mz + nfz*0.05)
            panel(c, p1, p2, c, (nfx, 0, nfz), FRAME)
        # ★ 橙色边框：真实日本カーブミラー是"橙框圆镜"，没有橙框一眼假
        RIM = (0.85, 0.42, 0.08)
        R_IN, R_OUT = 0.42, 0.54

        def _rim_pt(a, r):
            return (mx + r * (ufx * math.cos(a) + nfx * 0.075),
                    cy_m + r * math.sin(a),
                    mz + r * (ufz * math.cos(a) + nfz * 0.075))

        for k in range(SEG):
            a1 = 2 * math.pi * k / SEG
            a2 = 2 * math.pi * (k + 1) / SEG
            panel(_rim_pt(a1, R_IN), _rim_pt(a1, R_OUT),
                  _rim_pt(a2, R_OUT), _rim_pt(a2, R_IN),
                  (nfx, 0, nfz), RIM)

        # ---- 箭头牌：入弯前 60/40/20m 三连，外侧 ----
        # ★ 历史 bug：旧版牌面法线取"外法线"→ 牌背对来车，几何 Pass 开背面
        #   剔除 ⇒ 路上永远看不到牌面（只剩一根柱子）。现改为法线 = -前进方向，
        #   横轴取水平垂轴，车迎面正对牌面。
        # ★ 牌面放大到 1.20 × 0.84（原 0.80 × 0.50），高速下才有识别窗口。
        # ★ 黄底与黑图案分层 30mm（原共面 → z-fight 高速闪烁）。
        for s_off in (60.0, 40.0, 20.0):
            bx, by, bz, bnx, bnz, btx, btz, _ = _at_arc(cum[i0] - s_off)
            _tl = math.hypot(btx, btz) or 1e-6
            bnf_x, bnf_z = -btx / _tl, -btz / _tl          # 面向来车
            ufx2, ufz2 = bnf_z, -bnf_x                      # 牌面横轴
            PW, PH = 0.60, 0.42                             # 半宽 / 半高
            def _pt(lu, lv, lift=0.0):
                return (bx + ufx2*lu + bnf_x*(0.03+lift),
                        by + 1.85 + lv,
                        bz + ufz2*lu + bnf_z*(0.03+lift))
            NN = (bnf_x, 0.0, bnf_z)
            # 黄底
            panel(_pt(-PW,-PH), _pt(PW,-PH), _pt(PW,PH), _pt(-PW,PH), NN, BOARD_Y)

            # 黑色"つづら折り"折线箭头：竖笔画 → 斜笔画 → 箭头头
            sk2 = 1.0 if _outer_side_of(curv, peak) > 0 else -1.0
            def _seg(p0, p1, wid, lift=0.030):
                u0, v0 = p0; u1, v1 = p1
                du, dv = u1 - u0, v1 - v0
                Ls = math.hypot(du, dv) or 1e-6
                ax_, ay_ = du / Ls, dv / Ls
                qx_, qy_ = -ay_, ax_
                h = wid * 0.5
                panel(_pt(u0 - qx_*h, v0 - qy_*h, lift), _pt(u1 - qx_*h, v1 - qy_*h, lift),
                      _pt(u1 + qx_*h, v1 + qy_*h, lift), _pt(u0 + qx_*h, v0 + qy_*h, lift),
                      NN, BOARD_K)
            P0 = (0.0, -0.28); P1 = (0.0,  0.00); P2 = (sk2 * 0.30, 0.24)
            _seg(P0, P1, 0.145)
            _seg(P1, P2, 0.145)
            # 箭头头（三角，尖端在 P2 外沿）
            _dl = math.hypot(P2[0] - P1[0], P2[1] - P1[1]) or 1e-6
            _dx, _dy = (P2[0] - P1[0]) / _dl, (P2[1] - P1[1]) / _dl
            _qx, _qy = -_dy, _dx
            _bh = 0.185
            panel3(_pt(P2[0] + _dx * 0.12, P2[1] + _dy * 0.12, 0.030),
                   _pt(P2[0] + _qx * _bh, P2[1] + _qy * _bh, 0.030),
                   _pt(P2[0] - _qx * _bh, P2[1] - _qy * _bh, 0.030), NN, BOARD_K)
            # 立柱（双面）
            panel((bx-0.025, by, bz-0.025), (bx+0.025, by, bz-0.025), (bx+0.025, by+1.55, bz-0.025), (bx-0.025, by+1.55, bz-0.025), (0,0,-1), POLE)
            panel((bx-0.025, by, bz+0.025), (bx+0.025, by, bz+0.025), (bx+0.025, by+1.55, bz+0.025), (bx-0.025, by+1.55, bz+0.025), (0,0,1), POLE)

    # ============================================================
    # ★ 落石注意（日本警戒標識「落石注意」：黄底黑纹菱形）
    #   立在挖方段起点 + 每 180m 重复，与挡土墙/护坡同一判据。
    # ============================================================
    ROCK_Y = (0.95, 0.80, 0.10)    # 警示黄
    ROCK_K = (0.05, 0.05, 0.05)    # 黑（落石图形）
    SIGN_H, PLATE_Y, R_SIGN = 1.95, 2.25, 0.42   # 柱高 / 牌心高 / 菱形半对角

    cut_segs = _scan_cut_segments(min_rise=1.05)
    n_rock = 0
    # 扫描步长 step=2 → 每点约 1.0m，故 180m ≈ 180 个采样点
    _STRIDE = 180
    for side, recs in cut_segs:
        for k in range(0, len(recs), _STRIDE):
            _i, x, z, w, ox, oz = recs[k]
            # 立在侧沟之外（sd=1.30，与百米里程标同一横向规则）
            lat = (TRACK_EDGE_RATIO * w + 1.30)
            px = x + ox * lat
            pz = z + oz * lat
            py = get_ground_height(px, pz)
            # 立柱（0.044m 方柱）
            panel((px-0.022, py, pz-0.022), (px+0.022, py, pz-0.022),
                  (px+0.022, py+SIGN_H, pz-0.022), (px-0.022, py+SIGN_H, pz-0.022), (0,0,-1), POLE)
            panel((px-0.022, py, pz+0.022), (px+0.022, py, pz+0.022),
                  (px+0.022, py+SIGN_H, pz+0.022), (px-0.022, py+SIGN_H, pz+0.022), (0,0,1), POLE)
            panel((px-0.022, py, pz-0.022), (px-0.022, py, pz+0.022),
                  (px-0.022, py+SIGN_H, pz+0.022), (px-0.022, py+SIGN_H, pz-0.022), (-1,0,0), POLE)
            panel((px+0.022, py, pz-0.022), (px+0.022, py, pz+0.022),
                  (px+0.022, py+SIGN_H, pz+0.022), (px+0.022, py+SIGN_H, pz-0.022), (1,0,0), POLE)

            # 菱形牌面（朝向来车方向 = -切线）
            ti = min(len(path) - 1, _i)
            tx, ty, tz = tangents[ti]
            tl = math.hypot(tx, tz) or 1e-6
            fx, fz = tx / tl, tz / tl          # 前进方向
            nfx, nfz = -fx, -fz                # 面向来车
            ufx, ufz = nfz, -nfx               # 牌面横轴
            cy_s = py + PLATE_Y
            # 黄底菱形（4 个三角）
            N_, E_, S_, W_ = (0.0, R_SIGN), (R_SIGN, 0.0), (0.0, -R_SIGN), (-R_SIGN, 0.0)

            def _sp(u, v, lift=0.0):
                return (px + ufx * u + nfx * (0.03 + lift),
                        cy_s + v,
                        pz + ufz * u + nfz * (0.03 + lift))
            NF3 = (nfx, 0.0, nfz)
            # 黄底菱形（双面；旧版单面 + 绕序反 → 迎面看不到）
            for (qa, qb) in ((N_, E_), (E_, S_), (S_, W_), (W_, N_)):
                panel3(_sp(0.0, 0.0), _sp(*qa), _sp(*qb), NF3, ROCK_Y)

            # ★ 图案层抬 30mm（旧版与黄底共面 → 高速 z-fight 闪烁，图形时隐时现）
            LK = 0.030
            def _seg2(p0, p1, wid):
                u0, v0 = p0; u1, v1 = p1
                du, dv = u1 - u0, v1 - v0
                Ls = math.hypot(du, dv) or 1e-6
                ax_, ay_ = du / Ls, dv / Ls
                qx_, qy_ = -ay_, ax_
                h = wid * 0.5
                panel(_sp(u0 - qx_*h, v0 - qy_*h, LK), _sp(u1 - qx_*h, v1 - qy_*h, LK),
                      _sp(u1 + qx_*h, v1 + qy_*h, LK), _sp(u0 + qx_*h, v0 + qy_*h, LK),
                      NF3, ROCK_K)
            # 黑色"落石"图形：山坡斜线 + 岩块五边形 + 两颗落点（放大到占牌面 ~60%）
            _seg2((-0.31, -0.21), (0.27, -0.05), 0.080)
            rock = [(-0.13, 0.05), (0.03, 0.19), (0.19, 0.09), (0.15, -0.07), (-0.08, -0.08)]
            for j in range(len(rock)):
                a_, b_ = rock[j], rock[(j + 1) % len(rock)]
                panel3(_sp(0.0, 0.0, LK), _sp(*a_, lift=LK), _sp(*b_, lift=LK), NF3, ROCK_K)
            for (du, dv) in ((-0.20, -0.13), (0.17, -0.15)):
                r0 = 0.048
                for s_ in range(6):
                    a1 = 2 * math.pi * s_ / 6
                    a2 = 2 * math.pi * (s_ + 1) / 6
                    panel3(_sp(du, dv, LK),
                           _sp(du + r0 * math.cos(a1), dv + r0 * math.sin(a1), LK),
                           _sp(du + r0 * math.cos(a2), dv + r0 * math.sin(a2), LK), NF3, ROCK_K)
            n_rock += 1

    print(f"[标志牌] 落石注意标志 {n_rock} 处（挖方段 {len(cut_segs)} 段）")
    return np.array(verts, dtype=np.float32)


def _build_roadside_signs_em_impl():
    """凸面镜镜面：淡蓝自发光模拟弧面反光（触发 Bloom）。"""
    path, tangents = get_akina_path(return_tangents=True)[:2]
    n = len(path)
    curv = _compute_path_curvature_signed(path)
    cum = np.zeros(n)
    for i in range(1, n):
        cum[i] = cum[i-1] + math.hypot(path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
    verts = []
    def quad(p1, p2, p3, p4, nn, c):
        for pp in (p1, p2, p3, p1, p3, p4):
            verts.append([*pp, *nn, 0, 0, *c])
    MIRROR = (0.75, 0.92, 1.10)   # >1.0 触发 Bloom 的淡蓝
    for (i0, i1, peak) in _roadside_corner_clusters(curv):
        i = max(1, int(np.searchsorted(cum, cum[i0] - 10.0)))
        x, z, y, w, bank = path[i]
        tx, ty, tz = tangents[i]
        nx2d, nz2d = -tz, tx
        L = math.hypot(nx2d, nz2d) or 1e-6
        nx2d, nz2d = nx2d / L, nz2d / L
        side = _outer_side_of(curv, peak)
        px = x + nx2d * (w*0.5 + 1.5) * side
        pz = z + nz2d * (w*0.5 + 1.5) * side
        py = surface_height_at(px, pz) + 2.3
        nfx, nfz = -tx, -tz
        ufx, ufz = nfz, -nfx
        SEG = 10
        # ★ 镜面抬到 0.105：背板 0.05 / 橙框 0.075 / 镜面 0.105，三层互不 z-fight
        for k in range(SEG):
            a1 = 2*math.pi*k/SEG; a2 = 2*math.pi*(k+1)/SEG
            p1 = (px + 0.42*(ufx*math.cos(a1) + nfx*0.105), py + 0.42*math.sin(a1), pz + 0.42*(ufz*math.cos(a1) + nfz*0.105))
            p2 = (px + 0.42*(ufx*math.cos(a2) + nfx*0.105), py + 0.42*math.sin(a2), pz + 0.42*(ufz*math.cos(a2) + nfz*0.105))
            quad((px + nfx*0.105, py, pz + nfz*0.105), p1, p2, (px + nfx*0.105, py, pz + nfz*0.105), (nfx,0,nfz), MIRROR)
    return np.array(verts, dtype=np.float32)


def _build_road_studs_impl():
    path, tangents = get_akina_path(return_tangents=True)[:2]
    n = len(path)
    curv = _compute_path_curvature_signed(path)
    cum = np.zeros(n)
    for i in range(1, n):
        cum[i] = cum[i-1] + math.hypot(path[i][0]-path[i-1][0], path[i][1]-path[i-1][1])
    verts = []
    def quad(p1, p2, p3, p4, nn, c):
        for pp in (p1, p2, p3, p1, p3, p4):
            verts.append([*pp, *nn, 0, 0, *c])
    WHITE = (1.15, 1.12, 1.05)   # 右侧白（日规）
    AMBER = (1.20, 0.72, 0.10)   # 左侧橙黄
    R = 0.055; H = 0.022
    s = 4.0
    total_len = cum[-1]
    while s < total_len - 4.0:
        i = min(n - 2, max(1, int(np.searchsorted(cum, s))))
        step = 3.0 if abs(curv[i]) > 0.010 else 6.0
        s += step
        x, z, y, w, bank = path[i]
        tx, ty, tz = tangents[i]
        nx2d, nz2d = -tz, tx
        L = math.hypot(nx2d, nz2d) or 1e-6
        nx2d, nz2d = nx2d / L, nz2d / L
        hw = w * 0.5
        for side, col in ((-1.0, WHITE), (1.0, AMBER)):   # -1 = 行进方向右侧
            lat = 0.66 * hw   # 紧贴边缘线内侧
            px = x + nx2d * lat * side
            pz = z + nz2d * lat * side
            py = surface_height_at(px, pz) + H
            SEG = 6
            for k in range(SEG):
                a1 = 2*math.pi*k/SEG; a2 = 2*math.pi*(k+1)/SEG
                quad((px, py, pz),
                     (px + R*math.cos(a1), py, pz + R*math.sin(a1)),
                     (px + R*math.cos(a2), py, pz + R*math.sin(a2)),
                     (px, py, pz), (0,1,0), col)
    return np.array(verts, dtype=np.float32)


# ============================================================
# 十六、百米里程标（日式「100m 標」小柱）
# ============================================================

def build_kiloposts():
    """百米里程标 — 已缓存（缓存名挂 path 指纹）"""
    return _build_or_load_env("akina_kiloposts_v1", _build_kiloposts_impl)


def _build_kiloposts_impl():
    """
    日本山道每 100m 一根的白漆小柱（100m 標），立在侧沟之外的路肩上。
    造价极低（~3k 顶点），但给长直道提供了强烈的距离节奏感。

    位置：signed_distance = 1.30（侧沟外沿在 sd≈1.01，不冲突），
    立在**弯道外侧**，行车时永远在视野里。
    """
    path, tangents = get_akina_path(return_tangents=True)[:2]
    n = len(path)
    if n < 16:
        return np.empty((0, 11), dtype=np.float32)

    curv = _compute_path_curvature_signed(path)
    cum = np.zeros(n)
    for i in range(1, n):
        cum[i] = cum[i - 1] + math.hypot(path[i][0] - path[i - 1][0],
                                         path[i][1] - path[i - 1][1])
    total_len = float(cum[-1])

    WHITE = (0.80, 0.80, 0.77)
    BAND = (0.10, 0.10, 0.11)     # 柱顶黑箍
    H, HW2 = 0.95, 0.055          # 高 0.95m，截面 0.11m
    SD = 1.30

    verts = []

    def quad(A, B, C, D, nn, c):
        for pp in (A, B, C, A, C, D):
            verts.append([pp[0], pp[1], pp[2], nn[0], nn[1], nn[2],
                          0.0, 0.0, c[0], c[1], c[2]])

    s = 60.0
    while s < total_len - 20.0:
        i = int(np.searchsorted(cum, s))
        if i >= n - 1:
            break
        cx, cz, cy, w, bank = path[i]
        tx, ty, tz = tangents[i]
        L = math.hypot(tx, tz) or 1e-6
        nx, nz = -tz / L, tx / L
        side = _outer_side_of(curv, i)
        lat = side * (TRACK_EDGE_RATIO * w + SD)
        px = cx + nx * lat
        pz = cz + nz * lat
        py = get_ground_height(px, pz)

        # 四个侧面（朝外 / 朝内 / 顺路 / 逆路）
        ax, az = nx * side, nz * side          # 指向路外
        fx, fz = tx / L, tz / L
        c00 = (px - ax*HW2 - fx*HW2, py, pz - az*HW2 - fz*HW2)
        c10 = (px + ax*HW2 - fx*HW2, py, pz + az*HW2 - fz*HW2)
        c11 = (px + ax*HW2 + fx*HW2, py, pz + az*HW2 + fz*HW2)
        c01 = (px - ax*HW2 + fx*HW2, py, pz - az*HW2 + fz*HW2)
        t00 = (c00[0], py + H, c00[2]); t10 = (c10[0], py + H, c10[2])
        t11 = (c11[0], py + H, c11[2]); t01 = (c01[0], py + H, c01[2])
        quad(c10, c11, t11, t10, (ax, 0.0, az), WHITE)          # 朝外
        quad(c01, c00, t00, t01, (-ax, 0.0, -az), WHITE)        # 朝内
        quad(c11, c01, t01, t11, (fx, 0.0, fz), WHITE)          # 顺路
        quad(c00, c10, t10, t00, (-fx, 0.0, -fz), WHITE)        # 逆路
        # 柱顶
        quad(t00, t10, t11, t01, (0.0, 1.0, 0.0), BAND)
        # 顶部黑箍（0.06~0.16m）
        y0, y1 = py + H - 0.16, py + H - 0.06
        b00 = (c00[0], y0, c00[2]); b10 = (c10[0], y0, c10[2])
        b11 = (c11[0], y0, c11[2]); b01 = (c01[0], y0, c01[2])
        a00 = (c00[0], y1, c00[2]); a10 = (c10[0], y1, c10[2])
        a11 = (c11[0], y1, c11[2]); a01 = (c01[0], y1, c01[2])
        quad(b10, b11, a11, a10, (ax, 0.0, az), BAND)
        quad(b01, b00, a00, a01, (-ax, 0.0, -az), BAND)
        quad(b11, b01, a01, a11, (fx, 0.0, fz), BAND)
        quad(b00, b10, a10, a00, (-fx, 0.0, -fz), BAND)

        s += 100.0

    return np.array(verts, dtype=np.float32)


# ============================================================
# 十七、挖方边坡格构护坡（フレームワーク工法 / 法枠工）
# ============================================================

def build_slope_lattice():
    """挖方边坡混凝土格构护坡 — 已缓存（缓存名挂 path 指纹）"""
    return _build_or_load_env("akina_lattice_v4", _build_slope_lattice_impl)


def _beam_ribbon(pts, half_w, height, quad, c_top, c_side, sample=None):
    """沿中心线 pts=[(x,y,z)...] 铺一段混凝土梁（顶面 + 两侧立面）。

    pts 的 y 取地表高度；梁体从地表向上 height，横向半宽 half_w。

    ⚠️ sample 不为 None 时，**四个角各自采样地表高**（山体是斜的，
       用中心线高会让上坡侧埋进土里、下坡侧悬空）；传 None 则沿用 pts 的 y
       （用于本来就是直线的构件，如落石网的上部索）。
    """
    for j in range(len(pts) - 1):
        x0, y0, z0 = pts[j]
        x1, y1, z1 = pts[j + 1]
        dx, dz = x1 - x0, z1 - z0
        L = math.hypot(dx, dz)
        if L < 1e-6:
            continue
        px, pz = -dz / L, dx / L                    # 水平垂线
        if sample is not None:
            yc0 = sample(x0 - px*half_w, z0 - pz*half_w)
            yc1 = sample(x0 + px*half_w, z0 + pz*half_w)
            yd0 = sample(x1 - px*half_w, z1 - pz*half_w)
            yd1 = sample(x1 + px*half_w, z1 + pz*half_w)
        else:
            yc0 = yc1 = y0
            yd0 = yd1 = y1
        a0 = (x0 - px*half_w, yc0,     z0 - pz*half_w)
        b0 = (x0 + px*half_w, yc1,     z0 + pz*half_w)
        a1 = (x1 - px*half_w, yd0,     z1 - pz*half_w)
        b1 = (x1 + px*half_w, yd1,     z1 + pz*half_w)
        A0 = (x0 - px*half_w, yc0 + height, z0 - pz*half_w)
        B0 = (x0 + px*half_w, yc1 + height, z0 + pz*half_w)
        A1 = (x1 - px*half_w, yd0 + height, z1 - pz*half_w)
        B1 = (x1 + px*half_w, yd1 + height, z1 + pz*half_w)
        quad(A0, B0, B1, A1, (0.0, 1.0, 0.0), c_top)          # 顶面
        quad(a0, A0, A1, a1, (-px, 0.0, -pz), c_side)         # 外侧立面
        quad(b0, b1, B1, B0, (px, 0.0, pz), c_side)           # 内侧立面


def _slope_leg_ok(px, pz, w, run, tol=1.5):
    """该点是否真的在"本段路"的边坡上。

    ⚠️ 发夹弯两腿相距只有 20~30m，坡脚外 8~11m 处很可能**离另一条腿更近**，
       signed_distance / nearest_track_info 会选到另一条腿 → 中心线高直接跳
       好几米（实测 3m），梁体/网就会一半埋地里一半悬空。
       用"实测 sd vs 期望 sd"的偏差来剔除这类点。
    """
    expect = cut_foot_offset(w) + run - TRACK_EDGE_RATIO * w
    return abs(signed_distance(px, pz) - expect) <= tol


def _build_slope_lattice_impl():
    """
    日本山道挖方边坡的「法枠工」：现场浇筑的混凝土格构梁，框内植生。
    贴在挡土墙**上方**的坡面上（坡脚外 0.5m 起，向外铺 3.0m，2 排框格）。

    与挡土墙共用 cut_foot_sample 判据（rise > 0.9），保证"有墙才有护坡"。
    梁体沿地形起伏（采样 get_ground_height），不做整平 —— 真实法枠也是随坡就势。

    顶点格式: [pos3, nrm3, uv2, col3]
    """
    verts = []

    def quad(p1, p2, p3, p4, nn, c):
        for pp in (p1, p2, p3, p1, p3, p4):
            verts.append([pp[0], pp[1], pp[2], nn[0], nn[1], nn[2],
                          0.0, 0.0, c[0], c[1], c[2]])

    C_TOP = (0.44, 0.43, 0.39)     # 梁顶（新混凝土偏亮）
    C_SIDE = (0.30, 0.29, 0.27)    # 梁侧

    R0, R1, R2 = 0.5, 2.0, 3.5     # 三道纵向梁（沿路方向）的坡面距离
    BEAM_HW, BEAM_H = 0.15, 0.22   # 梁半宽 0.30m / 高 0.22m
    ALONG = 5                      # 横向梁间距（采样点，≈2.5m）

    segs = _scan_cut_segments(min_rise=0.9)
    n_cell = 0
    skipped = 0
    for side, recs in segs:
        m = len(recs)
        for k in range(0, m - 1, ALONG):
            k2 = min(k + ALONG, m - 1)
            a_rec, b_rec = recs[k], recs[k2]
            # ---- 纵向梁（沿路）：分 2 段贴合地形 ----
            for run in (R0, R1, R2):
                pts = []
                bad = False
                for t in (0.0, 0.5, 1.0):
                    # 两个记录之间按 t 插值中心与外法线
                    ia, xa, za, wa, oxa, oza = a_rec
                    ib, xb, zb, wb, oxb, ozb = b_rec
                    xm = xa + (xb - xa) * t
                    zm = za + (zb - za) * t
                    wm = wa + (wb - wa) * t
                    ox = oxa + (oxb - oxa) * t
                    oz = oza + (ozb - oza) * t
                    L = math.hypot(ox, oz) or 1e-6
                    lat = cut_foot_offset(wm) + run
                    px = xm + ox / L * lat
                    pz = zm + oz / L * lat
                    if not _slope_leg_ok(px, pz, wm, run):
                        bad = True
                        break
                    pts.append((px, get_ground_height(px, pz), pz))
                if bad or len(pts) < 2:
                    skipped += 1
                    continue
                _beam_ribbon(pts, BEAM_HW, BEAM_H, quad, C_TOP, C_SIDE,
                             sample=get_ground_height)
            # ---- 横向梁（跨坡）：从 R0 到 R2 ----
            for rec in (a_rec, b_rec):
                _i, _x, _z, wm = rec[0], rec[1], rec[2], rec[3]
                runs = (R0, (R0 + R2) * 0.5, R2)
                bad = any(not _slope_leg_ok(*_slope_point_xz(rec, r), wm, r)
                          for r in runs)
                if bad:
                    skipped += 1
                    continue
                pts = [_slope_point(rec, r) for r in runs]
                _beam_ribbon(pts, BEAM_HW, BEAM_H, quad, C_TOP, C_SIDE,
                             sample=get_ground_height)
            n_cell += 1

    print(f"[格构护坡] {len(segs)} 段挖方边坡, {n_cell} 个框格, "
          f"{len(verts):,} 顶点（跳过 {skipped} 处发夹弯错腿）")
    return np.array(verts, dtype=np.float32)


# ============================================================
# 十八、落石防护网（菱形金網 + 支柱 + 上部索）
# ============================================================

def build_rockfall_nets():
    """落石防护网 — 已缓存（缓存名挂 path 指纹）"""
    return _build_or_load_env("akina_rocknet_v5", _build_rockfall_nets_impl)


def _build_rockfall_nets_impl():
    """
    落石防護網：铺在格构护坡**上方**的坡面上（坡脚外 4.0~7.5m），
    由两组交叉钢丝形成菱形网目，上沿设 H 型钢支柱 + 上部横向索。

    ⚠️ 网用"竖向薄带"（贴地 0.02 → 抬 0.09）表示，不用透明贴图：
       本项目混合渲染会破坏延迟 G-Buffer，一律走 alpha 测试/实体几何。

    顶点格式: [pos3, nrm3, uv2, col3]
    """
    verts = []

    def quad(p1, p2, p3, p4, nn, c):
        for pp in (p1, p2, p3, p1, p3, p4):
            verts.append([pp[0], pp[1], pp[2], nn[0], nn[1], nn[2],
                          0.0, 0.0, c[0], c[1], c[2]])

    WIRE = (0.30, 0.31, 0.30)      # 镀锌钢丝（暗，避免满山发白）
    POST_C = (0.34, 0.35, 0.34)
    CABLE = (0.26, 0.27, 0.26)

    R0, R1 = 4.0, 7.5              # 网在坡面上的覆盖范围
    MESH = 0.80                    # 网目间距（沿路方向）
    WIRE_LO, WIRE_HI = 0.02, 0.09  # 竖向薄带的下/上沿
    ALONG = 5                      # 面板长度（采样点，≈2.5m）
    SUB = 3                        # 每根钢丝的地形采样数

    def _surf(rec_a, rec_b, t, run):
        """在两块记录之间插值，再沿坡面往外 run 米处的地表点。"""
        ia, xa, za, wa, oxa, oza = rec_a
        ib, xb, zb, wb, oxb, ozb = rec_b
        xm = xa + (xb - xa) * t
        zm = za + (zb - za) * t
        wm = wa + (wb - wa) * t
        ox = oxa + (oxb - oxa) * t
        oz = oza + (ozb - oza) * t
        L = math.hypot(ox, oz) or 1e-6
        lat = cut_foot_offset(wm) + run
        px = xm + ox / L * lat
        pz = zm + oz / L * lat
        # 发夹弯处最近腿可能变成另一条路 → 高度会跳几米，必须剔除
        if not _slope_leg_ok(px, pz, wm, run):
            return (px, 0.0, pz), False
        return (px, get_ground_height(px, pz), pz), True

    def _wire(rec_a, rec_b, t0, r0, t1, r1):
        """一根从 (t0,r0) 拉到 (t1,r1) 的钢丝（竖向薄带，贴地形）。"""
        poly = []
        for s_ in range(SUB + 1):
            t = t0 + (t1 - t0) * s_ / SUB
            r = r0 + (r1 - r0) * s_ / SUB
            (pt, ok) = _surf(rec_a, rec_b, t, r)
            if not ok:
                return          # 发夹弯错腿 → 整根钢丝丢弃
            poly.append(pt)
        for j in range(len(poly) - 1):
            x0, y0, z0 = poly[j]
            x1, y1, z1 = poly[j + 1]
            dx, dz = x1 - x0, z1 - z0
            L = math.hypot(dx, dz) or 1e-6
            px, pz = -dz / L, dx / L
            q0 = (x0, y0 + WIRE_LO, z0)
            q1 = (x1, y1 + WIRE_LO, z1)
            Q0 = (x0, y0 + WIRE_HI, z0)
            Q1 = (x1, y1 + WIRE_HI, z1)
            # 双面：竖向薄带无法保证绕序
            quad(q0, q1, Q1, Q0, (px, 0.0, pz), WIRE)
            quad(q0, Q0, Q1, q1, (-px, 0.0, -pz), WIRE)

    segs = _scan_cut_segments(min_rise=1.05)
    n_panel = 0
    for side, recs in segs:
        m = len(recs)
        for k in range(0, m - 1, ALONG):
            k2 = min(k + ALONG, m - 1)
            a_rec, b_rec = recs[k], recs[k2]
            # ---- 两组交叉钢丝 → 菱形网目 ----
            n_lines = max(2, int(2.5 / MESH))
            for s_ in range(n_lines + 1):
                t_ = s_ / n_lines
                _wire(a_rec, b_rec, t_, R0, min(t_ + 0.55, 1.0), R1)   # ↗
                _wire(a_rec, b_rec, t_, R1, min(t_ + 0.55, 1.0), R0)   # ↘
            # ---- 上沿支柱 + 上部索 + 下沿锚索 ----
            for t_ in (0.0, 1.0):
                (px, py, pz), _ok = _surf(a_rec, b_rec, t_, R1)
                if not _ok:
                    continue
                hh = 1.15
                quad((px-0.05, py, pz-0.05), (px+0.05, py, pz-0.05),
                     (px+0.05, py+hh, pz-0.05), (px-0.05, py+hh, pz-0.05), (0,0,-1), POST_C)
                quad((px-0.05, py, pz+0.05), (px+0.05, py, pz+0.05),
                     (px+0.05, py+hh, pz+0.05), (px-0.05, py+hh, pz+0.05), (0,0,1), POST_C)
                quad((px-0.05, py, pz-0.05), (px-0.05, py, pz+0.05),
                     (px-0.05, py+hh, pz+0.05), (px-0.05, py+hh, pz-0.05), (-1,0,0), POST_C)
                quad((px+0.05, py, pz-0.05), (px+0.05, py, pz+0.05),
                     (px+0.05, py+hh, pz+0.05), (px+0.05, py+hh, pz-0.05), (1,0,0), POST_C)
            # 上部索（沿路，柱顶）
            cab = [_surf(a_rec, b_rec, t_, R1) for t_ in (0.0, 0.5, 1.0)]
            if all(c[1] for c in cab):
                _beam_ribbon([(p[0][0], p[0][1] + 1.10, p[0][2]) for p in cab],
                             0.035, 0.045, quad, CABLE, CABLE)
            # 下沿锚索
            anc = [_surf(a_rec, b_rec, t_, R0) for t_ in (0.0, 0.5, 1.0)]
            if all(c[1] for c in anc):
                _beam_ribbon([(p[0][0], p[0][1] + 0.02, p[0][2]) for p in anc],
                             0.035, 0.045, quad, CABLE, CABLE)
            n_panel += 1

    print(f"[落石防护网] {len(segs)} 段高危边坡, {n_panel} 片网, {len(verts):,} 顶点")
    return np.array(verts, dtype=np.float32)


# ============================================================
# 十四、路肩 U 型混凝土侧沟（榛名山"沟渠跑法"的原型）
# ============================================================

def _gutter_corner_runs(curv, st_idx, s):
    """扫出"较急弯道"的连续段 → [(a0, a1, inner_side), ...]（站点索引，含首不含尾）。

    - 入场 |curv| ≥ _GUTTER_CURV_ENTER，出场 |curv| < _GUTTER_CURV_EXIT（滞回）
    - **曲率反号即断段**：S 弯的反向段不能算进同一个弯（否则内侧会判反）
    - 短于 _GUTTER_MIN_RUN 的段丢弃
    - inner_side = -_outer_side_of(峰值)：左转 → 内侧在左（side +1.0）

    返回站点索引区间，调用方用 path 索引 = st_idx[a]。
    """
    m = len(st_idx)
    out = []
    a = 0
    while a < m:
        if abs(curv[st_idx[a]]) < _GUTTER_CURV_ENTER:
            a += 1
            continue
        s0 = 1.0 if curv[st_idx[a]] > 0.0 else -1.0
        b = a
        while b < m and abs(curv[st_idx[b]]) >= _GUTTER_CURV_EXIT \
                and (1.0 if curv[st_idx[b]] > 0.0 else -1.0) == s0:
            b += 1
        i_end = min(st_idx[b - 1] + (st_idx[1] - st_idx[0]), len(s) - 1)
        if float(s[i_end] - s[st_idx[a]]) >= _GUTTER_MIN_RUN:
            apex = st_idx[a:b][int(np.argmax(np.abs(curv[st_idx[a:b]])))]
            out.append((a, b, -_outer_side_of(curv, apex)))
        a = b + 1
    return out


def build_drainage_ditches():
    """路肩 U 型混凝土侧沟 — 已缓存（缓存名挂 path 指纹）"""
    return _build_or_load_env("akina_ditches_v6", _build_drainage_ditches_impl)


# ---- 物理侧：与上面的视觉规则同源，供车辆物理查询"这里到底有没有沟" ----
_GUTTER_PHYS = None   # (mask[n], inner[n])，进程内只建一次


def _gutter_phys_tables():
    """逐 path 点的 (有沟掩码, 该处内侧 side 符号)。

    与 _build_drainage_ditches_impl 用同一套 _gutter_corner_runs + _GUTTER_STEP，
    保证"看得见的沟"和"咬得住的沟"是同一批。
    """
    global _GUTTER_PHYS
    if _GUTTER_PHYS is not None:
        return _GUTTER_PHYS
    _p = np.asarray(get_akina_path(), dtype=np.float64)
    n = len(_p)
    curv = _compute_path_curvature_signed(_p)
    arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(_p[:, 0]),
                                                    np.diff(_p[:, 1])))))
    st = np.arange(0, n, _GUTTER_STEP)
    mask = np.zeros(n, dtype=bool)
    inner = np.zeros(n, dtype=np.float64)
    for a, b, sd in _gutter_corner_runs(curv, st, arc):
        i0 = int(st[a])
        i1 = int(st[min(b, len(st) - 1)])
        mask[i0:i1 + 1] = True
        inner[i0:i1 + 1] = sd
    _GUTTER_PHYS = (mask, inner)
    return _GUTTER_PHYS


def gutter_phys_warmup():
    """启动时预热 —— 否则第一次开进沟里才算曲率（15601 点 Python 循环 ≈ 20ms 卡顿）。"""
    _gutter_phys_tables()


def gutter_present(i, side):
    """path 点 i 处、side 侧是否有可见侧沟（side: +1.0 = 法线正向 = 左）。

    i 来自 track_bounds.nearest_track_info()[0]，side = sign(该车中心横向偏移)。
    车辆物理（gutter_hold / gutter_dip）必须用它把关：v6 起侧沟只在急弯内侧
    存在，直道 / 弯外侧没有沟就不许咬地，否则车会被"看不见的沟"吸住。
    """
    mask, inner = _gutter_phys_tables()
    if i < 0 or i >= len(mask):
        return False
    return bool(mask[i]) and side == inner[i]


def _build_drainage_ditches_impl():
    """
    日本山道标配「上ぶた式 U 形側溝」的简化几何。

    ★ v6 出现规则：只在**较急弯道的内侧**铺 —— 直道不铺、弯道外侧不铺
      （这正是"沟渠跑法"发生的位置：内轮压进弯内侧的沟）。
      判据 = 带符号曲率滞回扫描（_gutter_corner_runs），不是 bank（实测 ±0.6°）。
      断头处补横截面端墙（_endcap），否则看穿成洞。

    ⚠️ 为什么不下凹进地形：地形压平带紧贴路面下方 0.10m 且连续，任何深于
       0.10m 的下凹都会被地形填平；而这条 1:1 赛道的地形核心带被自动降级到
       3.0m，开槽只会被采样成随机散坑（v25 试过，已废弃）。
       → 沟唇做成高出路面 0.10m 的混凝土边，沟腔底落在地形之上 0.04m，
         可见沟腔 0.16m 深 × 0.42m 宽，外缓坡把沟体外沿埋进地形。

    顶点格式: [pos3, nrm3, uv2, col3]（与护栏一致）
    """
    path, tangents = get_akina_path(return_tangents=True)[:2]
    n = len(path)
    if n < 16:
        return np.empty((0, 11), dtype=np.float32)

    curv = _compute_path_curvature_signed(path)  # ★ 曲率：正=左转，负=右转

    STEP = _GUTTER_STEP             # 每 3 个路径点一段（≈1.5m）
    U_LIST = (_GUTTER_U_LIP0, _GUTTER_U_CH0, _GUTTER_U_CH1,
              _GUTTER_U_LIP1, _GUTTER_U_END)
    _U_MID = 0.5 * (_GUTTER_U_CH0 + _GUTTER_U_CH1)   # 沟腔中心（查路面高用）

    CONC_TOP  = (0.46, 0.45, 0.42)  # 混凝土顶面（干）
    CONC_WALL = (0.33, 0.32, 0.30)  # 沟腔侧壁（背光）
    FLOOR     = (0.15, 0.16, 0.14)  # 沟底（潮气 + 青苔）
    GRATE     = (0.19, 0.20, 0.21)  # 集水桝钢格栅
    GRATE_DY  = -0.04               # 格栅略低于沟唇（形成下沉的集水口）
    CAP       = (0.27, 0.26, 0.24)  # 沟体断头端墙（略暗，读作"沟在这里断了"）

    verts = []

    def quad(A, B, C, D, nn, c):
        for pp in (A, B, C, A, C, D):
            verts.append([pp[0], pp[1], pp[2], nn[0], nn[1], nn[2],
                          0.0, 0.0, c[0], c[1], c[2]])

    def _noise(v, seed):
        h = math.sin(v * 12.9898 + seed * 78.233) * 43758.5453
        return h - math.floor(h)

    # ---- 每站的截面几何（横向 5 个位置 + 沟顶高）----
    st = [(path[i][0], path[i][1], path[i][2], path[i][3], path[i][4],
           math.atan2(tangents[i][0], tangents[i][2]))
          for i in range(0, n, STEP)]
    m = len(st)
    if m < 2:
        return np.empty((0, 11), dtype=np.float32)

    # ---- 出现规则（v6）：只在"较急弯道"的**内侧**铺，直道 / 弯道外侧不铺 ----
    st_idx = np.arange(0, n, STEP, dtype=np.int64)
    _p = np.asarray(path, dtype=np.float64)
    arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(_p[:, 0]),
                                                    np.diff(_p[:, 1])))))
    runs = _gutter_corner_runs(curv, st_idx, arc)
    if not runs:
        return np.empty((0, 11), dtype=np.float32)

    for (_a0, _a1, side) in runs:
        a0, a1 = _a0, min(_a1, m - 1)
        if a1 <= a0:
            continue
        # ★ 绕序翻转：u 轴（向外）随 side 反向，F×up 恒等于 +N0，
        #   因此 side<0 时同一套顶点顺序会变成背面 → 必须整体反转。
        flip = side < 0.0

        pts = {}    # pts[a][j] = (x, z)
        tops = {}
        nrm = {}    # 每站的外法线（side 已含）
        for a in range(a0, a1 + 1):
            cx, cz, cy, w, bank, ang = st[a]
            fx, fz = math.sin(ang), math.cos(ang)
            nx, nz = -fz, fx                      # 水平法线（与赛道网格同款）
            row = []
            for u in U_LIST:
                lat = side * (_GUTTER_START_RATIO * w + u)
                row.append((cx + nx * lat, cz + nz * lat))
            pts[a] = row
            nrm[a] = (side * nx, side * nz)       # 指向路外
            # 沟顶：用沟腔中心查**地形压平面**（必须与地形同源！）
            # ⚠️ 不能用 surface_height_at：它在赛道棱柱边界处有"三角形查询 /
            #    回退到中心线"的 0.15m 阶跃，会让沟底算出来比地形还低（被填平）。
            #    get_ground_height 与 _build_akina_terrain_impl 的 shoulder 同构。
            # ★ 在**沟腔内壁** (u=_GUTTER_U_CH0) 处查，而不是沟腔中心：
            #   路面横截面在路缘石外侧是往外下坡的（PROFILE y_off 从 0.09 降到 0），
            #   沟腔范围内地形最高点就在内壁这一侧。以它为基准才能保证
            #   "沟腔底比地形高 ≥ (_GUTTER_LIFT - _GUTTER_DEPTH)"，绝不被地形填平。
            latc = side * (_GUTTER_START_RATIO * w + _GUTTER_U_CH0)
            hx = cx + nx * latc
            hz = cz + nz * latc
            tops[a] = get_ground_height(hx, hz) + _GUTTER_LIFT

        def _face(pA0, pA1, pB1, pB0, nn, c):
            """(A,lo) (A,hi) (B,hi) (B,lo) —— lo/hi 沿 u 或 y。"""
            if flip:
                quad(pA0, pB0, pB1, pA1, nn, c)
            else:
                quad(pA0, pA1, pB1, pB0, nn, c)

        def _pt(a, j, dy):
            x, z = pts[a][j]
            return (x, tops[a] + dy, z)

        def _endcap(a):
            """沟体断头处的横截面端墙（4 片双面），否则断头看穿成洞。

            横截面带沟腔缺口 → 不是凸多边形，不能扇形三角化；按实体拆成
            4 个四边形：内侧唇柱 / 沟底下的墙 / 外侧唇柱 / 外缓坡楔形。
            """
            tt = (math.sin(st[a][5]), 0.0, math.cos(st[a][5]))
            lo = -_GUTTER_BODY

            def P(j, dy):
                x, z = pts[a][j]
                return (x, tops[a] + dy, z)

            def Q(p1, p2, p3, p4):
                quad(p1, p2, p3, p4, tt, CAP)
                quad(p1, p4, p3, p2, (-tt[0], -tt[1], -tt[2]), CAP)

            Q(P(0, 0.0), P(1, 0.0), P(1, lo), P(0, lo))
            Q(P(1, -_GUTTER_DEPTH), P(2, -_GUTTER_DEPTH), P(2, lo), P(1, lo))
            Q(P(2, 0.0), P(3, 0.0), P(3, lo), P(2, lo))
            Q(P(3, 0.0), P(4, -_GUTTER_APRON), P(4, lo), P(3, lo))

        _endcap(a0)
        _endcap(a1)

        for k in range(a0, a1):
            A, B = k, k + 1
            n_out = (nrm[A][0], 0.0, nrm[A][1])
            n_in = (-nrm[A][0], 0.0, -nrm[A][1])
            n_up = (0.0, 1.0, 0.0)
            kf = 0.88 + 0.24 * _noise(float(k), 11.0)
            c_top = (CONC_TOP[0] * kf, CONC_TOP[1] * kf, CONC_TOP[2] * kf)
            c_wall = (CONC_WALL[0] * kf, CONC_WALL[1] * kf, CONC_WALL[2] * kf)
            basin = ((k - a0) % _GUTTER_BASIN_EVERY) == 0
            floor_dy = -_GUTTER_DEPTH

            # 1 内侧唇顶面
            _face(_pt(A, 0, 0.0), _pt(A, 1, 0.0),
                  _pt(B, 1, 0.0), _pt(B, 0, 0.0), n_up, c_top)
            # 2 沟腔内壁（朝沟腔 = 向外）
            _face(_pt(A, 1, floor_dy), _pt(A, 1, 0.0),
                  _pt(B, 1, 0.0), _pt(B, 1, floor_dy), n_out, c_wall)
            # 3 沟底
            _face(_pt(A, 1, floor_dy), _pt(A, 2, floor_dy),
                  _pt(B, 2, floor_dy), _pt(B, 1, floor_dy), n_up, FLOOR)
            # 4 沟腔外壁（朝沟腔 = 向内）
            _face(_pt(A, 2, floor_dy), _pt(A, 2, 0.0),
                  _pt(B, 2, 0.0), _pt(B, 2, floor_dy), n_in, c_wall)
            # 5 外侧唇顶面
            _face(_pt(A, 2, 0.0), _pt(A, 3, 0.0),
                  _pt(B, 3, 0.0), _pt(B, 2, 0.0), n_up, c_top)
            # 6 外缓坡（衔接地形）
            _face(_pt(A, 3, 0.0), _pt(A, 4, -_GUTTER_APRON),
                  _pt(B, 4, -_GUTTER_APRON), _pt(B, 3, 0.0), n_up, c_wall)
            # 7 外侧立面（埋入开槽，遮住开槽侧壁）
            _face(_pt(A, 4, -_GUTTER_BODY), _pt(A, 4, -_GUTTER_APRON),
                  _pt(B, 4, -_GUTTER_APRON), _pt(B, 4, -_GUTTER_BODY),
                  n_out, c_wall)
            # 8 内侧立面（埋入开槽）
            _face(_pt(A, 0, -_GUTTER_BODY), _pt(A, 0, 0.0),
                  _pt(B, 0, 0.0), _pt(B, 0, -_GUTTER_BODY), n_in, c_wall)

            # 9 集水桝：钢格栅盖板（略低于沟唇）
            if basin:
                gy = GRATE_DY
                gA0 = (pts[A][0][0] + (pts[A][2][0] - pts[A][0][0]) * 0.02,
                       tops[A] + gy,
                       pts[A][0][1] + (pts[A][2][1] - pts[A][0][1]) * 0.02)
                gA1 = (pts[A][2][0], tops[A] + gy, pts[A][2][1])
                gB1 = (pts[B][2][0], tops[B] + gy, pts[B][2][1])
                gB0 = (pts[B][0][0] + (pts[B][2][0] - pts[B][0][0]) * 0.02,
                       tops[B] + gy,
                       pts[B][0][1] + (pts[B][2][1] - pts[B][0][1]) * 0.02)
                _face(gA0, gA1, gB1, gB0, n_up, GRATE)

    return np.array(verts, dtype=np.float32)


# ============================================================
# 十五、路面标线与磨耗（中央线 / 减速标线 / 补丁 / 胎痕）
# ============================================================

def build_road_markings():
    """路面标线 + 磨耗 — 已缓存（缓存名挂 path 指纹）"""
    return _build_or_load_env("akina_markings_v2", _build_road_markings_impl)


def _build_road_markings_impl():
    """
    日规山路路面要素，全部浮在视觉路面上方 8~15mm（防 Z-fighting）：

      · 中央线：白破线（实 4m / 空 6m，日规常用周期 10m）
      · 急弯段：黄实线（追越禁止）覆盖白破线
      · 弯道前：3 组横向白色减速标线（減速マーク）
      · 沥青补丁：每 ~70m 一块深色矩形补修痕
      · 漂移胎痕：弯道外侧两条黑色胎带

    顶点格式: [pos3, nrm3, uv2, col3]
    """
    path, tangents = get_akina_path(return_tangents=True)[:2]
    n = len(path)
    if n < 16:
        return np.empty((0, 11), dtype=np.float32)

    curv = _compute_path_curvature_signed(path)
    cum = np.zeros(n)
    for i in range(1, n):
        cum[i] = cum[i - 1] + math.hypot(path[i][0] - path[i - 1][0],
                                         path[i][1] - path[i - 1][1])
    total_len = float(cum[-1])

    verts = []

    def quad(A, B, C, D, nn, c):
        for pp in (A, B, C, A, C, D):
            verts.append([pp[0], pp[1], pp[2], nn[0], nn[1], nn[2],
                          0.0, 0.0, c[0], c[1], c[2]])

    def P(i, lat, lift):
        """路面点：与视觉赛道网格公式完全一致（含 banking 与沥青厚度）。"""
        cx, cz, cy, w, bank = path[i]
        tx, ty, tz = tangents[i]
        L = math.hypot(tx, tz) or 1e-6
        nx, nz = -tz / L, tx / L
        return (cx + nx * lat,
                cy + lat * math.sin(bank) + ASPHALT_THICKNESS + lift,
                cz + nz * lat)

    def strip(i0, i1, lat1, lat2, lift, col, step=2):
        """沿路径铺一条带（lat1 < lat2 → 正面朝上）。"""
        if lat1 > lat2:
            lat1, lat2 = lat2, lat1
        i0 = max(0, int(i0)); i1 = min(n - 2, int(i1))
        for i in range(i0, i1, step):
            j = min(i + step, n - 1)
            quad(P(i, lat1, lift), P(i, lat2, lift),
                 P(j, lat2, lift), P(j, lat1, lift), (0.0, 1.0, 0.0), col)

    WHITE = (0.86, 0.86, 0.82)
    YELLOW = (0.88, 0.72, 0.12)
    PATCH = (0.13, 0.125, 0.12)
    TIRE = (0.085, 0.085, 0.085)
    # ★ 四类标线互相重叠（补丁压胎痕、中央线压补丁）时若高度差只有 2~3mm，
    #   远距离深度精度不足 → z-fight，高速下整片标线闪烁。现按"磨耗在下、
    #   漆线在上"拉开到 ≥6mm：胎痕 6 / 补丁 12 / 中央线 20 / 减速标线 28 mm。
    L_TIRE, L_PATCH, L_MARK, L_BAR = 0.006, 0.012, 0.020, 0.028

    # ---- 1) 急弯黄实线（先铺，白破线让位）----
    yellow = np.zeros(n, dtype=bool)
    for (i0, i1, peak) in _roadside_corner_clusters(curv):
        a = max(0, i0 - 25)
        b = min(n - 1, i1 + 15)
        yellow[a:b] = True
        strip(a, b, -0.075, 0.075, L_MARK, YELLOW)

    # ---- 2) 中央白破线（4m 实 / 6m 空）----
    HW_LINE = 0.075
    run = None
    for i in range(0, n, 2):
        on = (not yellow[i]) and ((cum[i] % 10.0) < 4.0)
        if on and run is None:
            run = i
        elif (not on) and run is not None:
            strip(run, i, -HW_LINE, HW_LINE, L_MARK, WHITE)
            run = None
    if run is not None:
        strip(run, n - 1, -HW_LINE, HW_LINE, L_MARK, WHITE)

    # ---- 3) 弯道前横向减速标线（減速マーク）----
    BAR_HALF = 2.15          # 横跨沥青（路宽 4.9m，留 0.3m 边距）
    BAR_LEN = 0.35
    BAR_GAP = 0.45
    for (i0, i1, peak) in _roadside_corner_clusters(curv):
        s_base = float(cum[i0])
        for g in range(3):
            s0 = s_base - 45.0 + g * 8.0
            if s0 < 2.0:
                continue
            for b in range(3):
                sa = s0 + b * (BAR_LEN + BAR_GAP)
                sb = sa + BAR_LEN
                ia = int(np.searchsorted(cum, sa))
                ib = int(np.searchsorted(cum, sb))
                if ib <= ia:
                    ib = ia + 1
                strip(ia, ib, -BAR_HALF, BAR_HALF, L_BAR, WHITE, step=1)

    # ---- 4) 沥青补丁 ----
    def _hash(a, seed):
        h = math.sin(a * 12.9898 + seed * 78.233) * 43758.5453
        return h - math.floor(h)

    s = 40.0
    while s < total_len - 40.0:
        i = int(np.searchsorted(cum, s))
        i2 = min(n - 2, int(np.searchsorted(cum, s + 2.6)))
        if i2 > i:
            r1 = _hash(s, 3.0)
            r2 = _hash(s, 7.0)
            lat = -1.7 + r1 * 3.4
            half = 0.55 + r2 * 0.45
            strip(i, i2, lat - half, lat + half, L_PATCH, PATCH, step=1)
        s += 70.0 + _hash(s, 13.0) * 60.0

    # ---- 5) 漂移胎痕（弯道外侧双胎带）----
    for (i0, i1, peak) in _roadside_corner_clusters(curv):
        side = _outer_side_of(curv, peak)
        a = max(0, i0 - 8)
        b = min(n - 1, i1 + 12)
        for base in (1.15, 1.78):
            lat = side * base
            strip(a, b, lat - 0.065, lat + 0.065, L_TIRE, TIRE)

    return np.array(verts, dtype=np.float32)


# ============================================================
# 十九、起终点场地（沥青平台 + 停车位标线 + 挡轮杆）
# ============================================================

_VENUE_PADS   = None
# ★ 起终点 = **贴着赛道的沿路服务带**，不是远离路面的大平台：
#   内边缘离沥青路缘(0.35w) 只有 _VENUE_GAP_ROAD，建筑就立在路边 ~3m。
_VENUE_HALF_LEN  = 30.0    # 沿赛道半长 → 60m 长的服务带
_VENUE_GAP_ROAD  = 2.2     # 内边缘离沥青路缘的距离（护栏在 0.5w，落在带子外侧）
_VENUE_BANDS     = (5.5, 6.75, 8.0)   # 法向带宽候选（按挖填量择优）
_VENUE_CURB_UP   = 0.20    # 平台高出路肩 → 形成路缘石
_VENUE_APRON_UP  = 0.09    # 缓冲带铺面高出其下方地形的量（必须 >0，否则被埋）
_VENUE_FLATTEN_IN  = 2.0   # 朝路面一侧的地形过渡带（窄！否则顶起路面）
_VENUE_FLATTEN_OUT = 6.0   # 朝山一侧 / 两端的地形过渡带（太宽会削出大台地）


def _venue_pads():
    """起点/终点选址：贴着赛道划一条**沿路服务带**（★ 不是远离路面的大平台）。

    ⚠️ 旧版在法线方向 14~38m 外找一块平地铺 70×44 的大平台 ——
       等于把整组元素搬到半山腰上（终点那块离终点线 140m、比路面高 17m），
       完全不是"路边"。用户要的是：建筑就在起点/终点所在的这条路上，
       或者离路面 3m 以内。

    现在：以赛道端点为原点，沿切向 ±_VENUE_HALF_LEN、法向
    [0.35w + _VENUE_GAP_ROAD, +band] 划一条贴着路的服务带。
    - 平面标高 = 路肩高 + _VENUE_CURB_UP，并且**沿切向带路面纵坡 grade**：
      起点路面纵坡 −3.5%，不带纵坡的话带子末端会悬在路肩上方 1.3m。
    - 左右两侧 × 三种带宽全扫，取 max(挖方, 填方) 最小者（偏好宽带）。

    返回 [(cx, cz, fx, fz, rx, rz, base_y, HL, HW, grade), ...]（起点, 终点）
      cx/cz  = 带子中心（世界坐标）
      fx/fz  = 单位切向；rx/rz = 单位法向（指向背离路面一侧）
      base_y = u=0 处的平台标高；HL = 沿切向半长；HW = 法向半宽
      grade  = 沿 +u 的纵坡（m/m）
      局部 (u,v) → 世界：(cx + fx*u + rx*v, base_y + grade*u + py, cz + fz*u + rz*v)
    """
    global _VENUE_PADS
    if _VENUE_PADS is not None:
        return _VENUE_PADS
    path, tangents = get_akina_path(return_tangents=True)[:2]
    n = len(path)
    out = []
    for idx in (0, n - 1):
        x, z, y, w, bank = path[idx]
        tx, ty, tz = tangents[idx]
        # ⚠️ 3D 切向的 xz 分量不是单位长（|xz| = 1/√(1+坡度²)），必须归一，
        #    否则局部 (u,v) 与世界坐标互换不自洽（边缘点会算到带子外）。
        Lt = math.hypot(tx, tz) or 1e-6
        tx, tz = tx / Lt, tz / Lt
        nx, nz = -tz, tx
        hw = 0.35 * w                       # 沥青路缘
        best = None
        for side in (1.0, -1.0):
            # 纵坡：用 u ∈ [-HL, HL] 内的路径点线性拟合 y(u)
            us_, ys_ = [], []
            for k in range(max(0, idx - 100), min(n, idx + 100)):
                du = (path[k][0] - x) * tx + (path[k][1] - z) * tz
                if abs(du) <= _VENUE_HALF_LEN + 2.0:
                    us_.append(du)
                    ys_.append(path[k][2])
            if len(us_) >= 4:
                grade = float(np.polyfit(np.array(us_), np.array(ys_), 1)[0])
            else:
                grade = 0.0
            for band in _VENUE_BANDS:
                v0 = hw + _VENUE_GAP_ROAD
                v1 = v0 + band
                vc = 0.5 * (v0 + v1)
                # 基准标高：带子内侧 1.5m（= 路肩）去掉纵坡后的中位 + 路缘高
                hs = []
                for du in np.linspace(-_VENUE_HALF_LEN, _VENUE_HALF_LEN, 13):
                    for dv in np.linspace(v0, v0 + 1.5, 4):
                        hs.append(get_ground_height(
                            x + tx*du + nx*side*dv,
                            z + tz*du + nz*side*dv) - grade * du)
                hs.sort()
                base_y = hs[len(hs) // 2] + _VENUE_CURB_UP
                # 挖 / 填评估（相对带纵坡的平台面）
                cut = fill = 0.0
                for du in np.linspace(-_VENUE_HALF_LEN, _VENUE_HALF_LEN, 21):
                    for dv in np.linspace(v0, v1, 9):
                        r = get_ground_height(
                            x + tx*du + nx*side*dv,
                            z + tz*du + nz*side*dv) - (base_y + grade * du)
                        cut = max(cut, r)
                        fill = max(fill, -r)
                score = max(cut, fill) - 0.35 * band      # 偏好宽带
                if best is None or score < best[0]:
                    best = (score, side, vc, base_y, 0.5 * band, grade)
        _, side, vc, base_y, HW_, grade = best
        cx = x + nx * side * vc
        cz = z + nz * side * vc
        out.append((cx, cz, tx, tz, nx * side, nz * side, base_y,
                    _VENUE_HALF_LEN, HW_, grade))
    _VENUE_PADS = out
    return out


def _venue_flatten_h(x, z, h, sd):
    """把地形高 h 向沿路服务带的平台标高过渡（必须与地形构建器 v31 逐项一致）。

    - 带内（d ≤ 0）：压到 平台 − 0.06（下沉量防与沥青面共面 z-fight）
    - 带外：smoothstep 过渡回原地形，w(0)=0 ⇒ 边缘天然连续。
      ⚠️ 朝路面那一侧必须用**窄**过渡带（_VENUE_FLATTEN_IN = 2.0m）：
         用 10m 的话权重会一路伸到路面底下，把路肩顶高 ~0.1m，
         地形就穿出沥青面了（路面只有 0.15m 厚）。朝山一侧 / 两端用 10m。
      ⚠️ 别再加"道路保护带"之类的硬分支 —— 边缘浮点抖动会让它反复横跳
         （实测 ±0.6m）。连续公式才是解。
    """
    if _VENUE_PADS is None:
        return h
    for (cx, cz, fx, fz, rx, rz, base_y, HL, HW_, grade) in _VENUE_PADS:
        u = (x - cx) * fx + (z - cz) * fz
        v = (x - cx) * rx + (z - cz) * rz
        # ★ 缓冲带（路面↔平台之间那条连续缓冲地带，v ∈ [-HW-GAP, -HW]）：
        #   地形跟着缓冲带铺面走、恒定低 9cm。
        #   ⚠️ 旧版这里走"回天然高的 smoothstep 过渡" —— 结果地形被抬到只比
        #      铺面低 2~4cm，整条缓冲带被埋进地形里（用户"完全看不到相连部位"
        #      的直接原因）。铺面与地形必须留出确定间隙，不能靠 blend 碰运气。
        if abs(u) <= HL and (-HW_ - _VENUE_GAP_ROAD) <= v <= -HW_:
            t = (-HW_ - v) / _VENUE_GAP_ROAD      # 0=平台边, 1=沥青路缘
            h = (base_y + grade * u) * (1.0 - t) + (h + _VENUE_APRON_UP) * t \
                - _VENUE_APRON_UP
            continue
        dU = abs(u) - HL
        if v < -HW_:
            dV = -HW_ - v          # 朝路面那一侧
        elif v > HW_:
            dV = v - HW_           # 朝山那一侧
        else:
            dV = 0.0
        tgt = base_y + grade * u - 0.06
        if dU <= 0.0 and dV <= 0.0:
            h = tgt
            continue
        d = max(dU, dV, 0.0)
        W = _VENUE_FLATTEN_IN if (dV > 0.0 and v < -HW_ and dV >= dU) \
            else _VENUE_FLATTEN_OUT
        w = min(1.0, d / W)
        w = w * w * (3.0 - 2.0 * w)
        if w >= 1.0:
            continue
        h = tgt + (h - tgt) * w
    return h


def venue_wall_margin(px, pz, default=0.25):
    """★ 起终点场地走廊内的空气墙放宽量：车能开上建筑前的缓冲带。

    v 是相对场地中心的法向坐标（+r 指向背离路面）：
      v = -(HW+GAP) 是路缘、v ∈ [-(HW+GAP), -HW] 是缓冲带。
    车越过路缘进入场地一侧（v > -(HW+GAP+0.8)）→ 边界从路缘外推到
    建筑前 0.2m（sd≈2.0）；对侧（v 更负）保持默认 0.25 不受影响。
    """
    for (cx, cz, fx, fz, rx, rz, base_y, HL, HW_, grade) in _venue_pads():
        u = (px - cx) * fx + (pz - cz) * fz
        if abs(u) > HL + 1.5:
            continue
        v = (px - cx) * rx + (pz - cz) * rz
        if v > -(HW_ + _VENUE_GAP_ROAD + 0.8):
            return 2.0
    return default


def venue_surface_at(px, pz):
    """场地走廊内的铺面高（不在走廊返回 None）。

    走廊内地形已被压平：缓冲带铺面 = 地形+0.09、平台 = 地形+0.06，
    取 +0.075 最大误差 1.5cm。surface_height_at 的路棱柱外回退
    （cy + lat·sin(bank)）不认场地铺面，车开上缓冲带会陷进去 —— 这里兜底。
    """
    for (cx, cz, fx, fz, rx, rz, base_y, HL, HW_, grade) in _venue_pads():
        u = (px - cx) * fx + (pz - cz) * fz
        if abs(u) > HL + 1.5:
            continue
        v = (px - cx) * rx + (pz - cz) * rz
        if -(HW_ + _VENUE_GAP_ROAD + 0.8) <= v <= HW_ + 0.5:
            return get_ground_height(px, pz) + 0.075
    return None


def build_venue_ground():
    """起终点场地 — 已缓存（缓存名挂 path 指纹）"""
    return _build_or_load_env("akina_venue_ground_v18", _build_venue_ground_impl)


def _build_venue_ground_impl():
    """
    起点/终点各铺一条**贴着赛道的沿路服务带**（沥青 + 白边线 + 挡轮杆）。
    位置/尺寸/标高/纵坡全部由 _venue_pads() 决定（不再写死 70×44 的大平台）。

    顶点格式: [pos3, nrm3, uv2, col3]
    """
    verts = []

    def quad(p1, p2, p3, p4, n, c):
        """双面：正反各一片。

        ⚠️ 场地坐标系 (u=沿赛道, v=法线×side) 在 side=-1 时是左手系，
        所有按 (u,v) 逆时针写的绕序翻到世界空间就变成背面 → 整块不可见。
        起终点两块场地实测都选到 side=-1，所以这里一律双面；
        背面剔除保证同一像素只过一片，不会 z-fight。
        """
        for pp in (p1, p2, p3, p1, p3, p4):
            verts.append([pp[0], pp[1], pp[2], n[0], n[1], n[2],
                          0.0, 0.0, c[0], c[1], c[2]])
        nn = (-n[0], -n[1], -n[2])
        for pp in (p1, p4, p3, p1, p3, p2):
            verts.append([pp[0], pp[1], pp[2], nn[0], nn[1], nn[2],
                          0.0, 0.0, c[0], c[1], c[2]])

    ASPHALT     = (0.10, 0.10, 0.11)
    HATCH       = (0.62, 0.58, 0.20)   # 缓冲带斜向斑马线（黄）
    WHITE       = (0.88, 0.88, 0.84)
    WHEEL_STOP  = (0.85, 0.65, 0.10)
    STOP_DARK   = (0.10, 0.10, 0.10)
    CURB        = (0.45, 0.44, 0.42)
    SLOPE_C     = (0.30, 0.29, 0.26)   # 边坡裙（挖方切坡 / 填方护坡的混凝土面）
    UP          = (0.0, 1.0, 0.0)

    # ★ 场地位置/尺寸/标高/纵坡全部来自 _venue_pads（贴着赛道的沿路服务带）
    for _pi, (cx, cz, fx, fz, rx, rz, base_y, HL, HW_, grade) in \
            enumerate(_venue_pads()):

        def P(pu, py, pv):
            """局部 (沿赛道 u, 高 py, 法向 v) → 世界。
            平台沿 +u 带路面纵坡 grade，所有顶点必须逐点取高。"""
            return (cx + fx*pu + rx*pv, base_y + grade*pu + py,
                    cz + fz*pu + rz*pv)

        def W2(pu, pv):
            """只取平面坐标（缓冲带要逐点查地形高）"""
            return (cx + fx*pu + rx*pv, cz + fz*pu + rz*pv)

        # 带子四角 u=±HL / v=±HW（v 负侧朝路面）
        p1 = P(-HL, 0.0, -HW_)
        p2 = P(+HL, 0.0, -HW_)
        p3 = P(+HL, 0.0, +HW_)
        p4 = P(-HL, 0.0, +HW_)

        # ---- 沥青地面 ----
        quad(p1, p2, p3, p4, UP, ASPHALT)

        # ---- ★ 边坡裙：只在背离路面的三条边上做 ----
        #   朝路面那边地形本来就是路肩（比平台低 _VENUE_CURB_UP），
        #   做裙只会把路肩挖开 —— 那边用路缘石收边即可。
        #   ⚠️ 外法线必须显式给：旧版靠绕序推 rot(-90°)，side=-1 时整条
        #      裙边朝带子内部长，边缘压根没接到地形上。
        SKIRT_MAX = 12.0
        SKIRT_RATIO = 1.5      # 水平:垂直
        ROAD_CLEAR = 0.6       # 裙末端离沥青路缘至少这么远

        def _sd_of(sx, sz):
            _, _hw_i, _, _, _lat_i, _ = nearest_track_info(sx, sz)
            return abs(_lat_i) - _hw_i

        def skirt_len(px, pz, ep, y0):
            """沿外法线 ep 找裙长（m）：第一个坡度 ≤ 1:1.5 的落地点"""
            for k in range(22):                    # 1.5 → 12.0，步长 0.5
                s = 1.5 + 0.5 * k
                qx, qz = px + ep[0]*s, pz + ep[1]*s
                if _sd_of(qx, qz) < ROAD_CLEAR:    # 再往外就压到赛道了
                    return max(1.5, s - 0.5)
                if abs(get_ground_height(qx, qz) - y0) <= s / SKIRT_RATIO:
                    return s
            return SKIRT_MAX

        # 按 p1→p2→p3→p4 顺序，外法线 = (dv, -du)/L 恒指向带子外侧
        edges = [((-HL, -HW_), (+HL, -HW_)),     # 0 朝路面：只做路缘石
                 ((+HL, -HW_), (+HL, +HW_)),     # 1 端头
                 ((+HL, +HW_), (-HL, +HW_)),     # 2 朝山
                 ((-HL, +HW_), (-HL, -HW_))]     # 3 端头
        for ei, (a, b) in enumerate(edges):
            (ua, va), (ub, vb) = a, b
            du, dv = ub - ua, vb - va
            L_ed = math.hypot(du, dv) or 1e-6
            eu, ev = dv / L_ed, -du / L_ed        # 外法线（局部）
            ex, ez = fx*eu + rx*ev, fz*eu + rz*ev
            M = max(2, int(L_ed / 3.0))
            prev = None
            if ei == 0:
                # 朝路面：不做路缘石 —— 那一条现在是**连续缓冲带**，
                #   路缘石会把缓冲带和路面重新隔开（旧版"看不到相连部位"之一）。
                continue
            sn = (ex, 0.75, ez)
            Ls = math.hypot(sn[0], sn[1], sn[2])
            slope_n = (sn[0]/Ls, sn[1]/Ls, sn[2]/Ls)
            for k in range(M + 1):
                t_ = k / M
                cu, cv = ua + du*t_, va + dv*t_
                px, pz = P(cu, 0.0, cv)[0], P(cu, 0.0, cv)[2]
                y0 = base_y + grade * cu
                s = skirt_len(px, pz, (ex, ez), y0)
                qx, qz = px + ex*s, pz + ez*s
                # 末端埋进地形 0.40m：地形查询在路棱柱边界有跳变，
                #   "刚好贴地"会出现 ~0.2m 的悬空缝，宁埋勿浮（0.25 实测还差 0.14）
                cur = ((px, y0, pz),
                       (qx, get_ground_height(qx, qz) - 0.40, qz))
                if prev is not None:
                    quad(prev[0], cur[0], cur[1], prev[1], slope_n, SLOPE_C)
                prev = cur

        # ============================================================
        # ★ 缓冲地带：沿**整条服务带**从沥青路缘连续斜升到平台的铺面
        #   旧版只在 u≈0 开一条 8m 宽的小引道，其余 52m 全被路缘石封死，
        #   看着就是"路旁边另一块地"。现在整条 60m 都是相连的缓冲带。
        #   铺面高 = 地形 + _VENUE_APRON_UP（地形在带内跟着铺面走，见
        #   _venue_flatten_h），所以恒高出地形 9cm —— 不可能被埋。
        # ============================================================
        v_out = -HW_ - _VENUE_GAP_ROAD
        NU = max(2, int(2 * HL / 2.0))     # 沿 u 每 ~2m 一段（跟纵坡/地形起伏）
        NV = 4                              # 沿 v 每 ~0.55m 一段

        def v_out_at(pu):
            """★ 路面宽度沿赛道是变的（避车道 w 可到 10.2）—— 写死
               v=-HW-GAP 会让缓冲带压进路面（起点实测 sd=-0.47m，铺面
               直接盖在沥青上冒出 5cm 的台）。逐列把外缘推到 sd=+0.05。
               ★ 路端以外（u 越过端点）没有路了 —— sd 探测会把缓冲带
                 收成尖角；直接按矩形直延伸，视觉上"赛道开进场地后
                 仍在继续"（场地平台就是赛道的延续）。"""
            if (_pi == 1 and pu > 1.0) or (_pi == 0 and pu < -1.0):
                return -HW_ - _VENUE_GAP_ROAD
            vv = v_out
            for _ in range(16):
                x_, z_ = W2(pu, vv)
                if signed_distance(x_, z_) >= 0.05:
                    break
                vv += 0.10
            return min(vv, -HW_)

        def apron_y(pu, pv):
            if pv >= -HW_ - 1e-6:
                return base_y + grade * pu         # 平台边：与平台齐平
            x_, z_ = W2(pu, pv)
            return get_ground_height(x_, z_) + _VENUE_APRON_UP

        rows = []
        for k in range(NU + 1):
            pu = -HL + (2 * HL) * k / NU
            vo = v_out_at(pu)
            row = []
            for j in range(NV + 1):
                pv = -HW_ + (vo + HW_) * (j / NV)
                x_, z_ = W2(pu, pv)
                row.append((x_, apron_y(pu, pv), z_))
            rows.append(row)
        for k in range(NU):
            a_row, b_row = rows[k], rows[k + 1]
            for j in range(NV):
                quad(a_row[j], b_row[j], b_row[j + 1], a_row[j + 1],
                     UP, ASPHALT)

        # ---- pit 车位线：建筑前一排白框车位（与建筑立面 u 对齐）----
        #   ★ 旧版这里是黄斜纹（缓冲带/禁停区的画法）—— 那是"禁止驶入"的
        #     视觉语言，正是"这块地和赛道无关"疏离感的来源之一。
        #     现在改成维修区车位：起点 [管理栋, 看台]、终点 [车库, 便利店]，
        #     u 跨度与 _build_venue_buildings_impl 的建筑布局逐一对齐。
        BAY_W, BAY_L = 2.6, 2.6            # 车位宽 × 深（朝路方向）
        spans = ([(-23.0, -5.0), (4.0, 26.0)] if _pi == 0
                 else [(-26.0, -4.0), (8.0, 20.0)])
        for (s0, s1) in spans:
            n_bay = int((s1 - s0 - 1.0) // 3.0)
            for bi in range(max(0, n_bay)):
                uc = s0 + 1.0 + 3.0 * (bi + 0.5)
                v_in = -HW_ + 0.55         # 靠建筑一端（平台上）
                v_out_b = -HW_ - (BAY_L - 0.55)   # 朝路一端（缓冲带内）

                def _mp(pu_, pv_):
                    mx_, mz_ = W2(pu_, pv_)
                    return (mx_, get_ground_height(mx_, mz_) + 0.10, mz_)

                for (lu0, lv0, lu1, lv1) in (
                        (uc - BAY_W/2, v_in, uc - BAY_W/2, v_out_b),
                        (uc + BAY_W/2, v_in, uc + BAY_W/2, v_out_b)):
                    a_, b_ = _mp(lu0, lv0), _mp(lu1, lv1)
                    du_ = lu1 - lu0
                    dv_ = lv1 - lv0
                    Ll = math.hypot(du_, dv_) or 1e-6
                    wx_, wz_ = (fx*du_ + rx*dv_)/Ll, (fz*du_ + rz*dv_)/Ll
                    px_, pz_ = 0.05 * -wz_, 0.05 * wx_   # 线宽 0.10
                    quad((a_[0] - px_, a_[1], a_[2] - pz_),
                         (b_[0] - px_, b_[1], b_[2] - pz_),
                         (b_[0] + px_, b_[1], b_[2] + pz_),
                         (a_[0] + px_, a_[1], a_[2] + pz_), UP, WHITE)
                # 车位尾挡线（靠建筑那条短横线，宽 0.10）
                quad(_mp(uc - BAY_W/2, v_in - 0.05),
                     _mp(uc + BAY_W/2, v_in - 0.05),
                     _mp(uc + BAY_W/2, v_in + 0.05),
                     _mp(uc - BAY_W/2, v_in + 0.05), UP, WHITE)

        # ---- 起/终点线：横跨路面的黑白格纹（格子旗语言，一眼认出"这是终点"）----
        #   放在 u=±6（起端在车头前方 6m，终端在结束前 6m），确保在真实路面内。
        _line_u = 6.0 if _pi == 0 else -6.0
        _path_all = get_akina_path()
        _bi, _bd = 0, 1e18
        _ax, _az = cx + fx * _line_u, cz + fz * _line_u
        for k in range(len(_path_all)):
            d2 = (_path_all[k][0] - _ax) ** 2 + (_path_all[k][1] - _az) ** 2
            if d2 < _bd:
                _bd, _bi = d2, k
        _p0 = _path_all[_bi]
        _hw_r = 0.35 * _p0[3]
        _ka = max(0, _bi - 3)
        _kb = min(len(_path_all) - 1, _bi + 3)
        _tx = _path_all[_kb][0] - _path_all[_ka][0]
        _tz = _path_all[_kb][1] - _path_all[_ka][1]
        _tl = math.hypot(_tx, _tz) or 1e-6
        _tx, _tz = _tx / _tl, _tz / _tl

        def _cp(du, lat):
            """格纹顶点：du 沿切向、lat 横向（相对路径点 _p0）"""
            qx = _p0[0] + _tx * du - _tz * lat
            qz = _p0[1] + _tz * du + _tx * lat
            return (qx, get_ground_height(qx, qz) + 0.12, qz)

        _sq = 0.55
        _n_sq = int((2.0 * _hw_r - 0.5) / _sq)
        for _row in range(2):
            for _ci in range(_n_sq):
                _lat0 = -_hw_r + 0.25 + _ci * _sq
                _col = WHITE if (_ci + _row) % 2 == 0 else (0.10, 0.10, 0.11)
                quad(_cp(_row * _sq, _lat0),
                     _cp(_row * _sq, _lat0 + _sq),
                     _cp((_row + 1) * _sq, _lat0 + _sq),
                     _cp((_row + 1) * _sq, _lat0), UP, _col)

        # ---- 缓冲带靠路缘的白边线：与路面白线接上，读作"路在这里张开" ----
        for k in range(NU):
            ua = -HL + (2 * HL) * k / NU
            ub = -HL + (2 * HL) * (k + 1) / NU
            vl = v_out_at(ub) + 0.07
            pa = W2(ua, vl); pb = W2(ub, vl)
            quad((pa[0], apron_y(ua, vl) + 0.012, pa[1]),
                 (pb[0], apron_y(ub, vl) + 0.012, pb[1]),
                 (pb[0], apron_y(ub, vl + 0.13) + 0.012, pb[1]),
                 (pa[0], apron_y(ua, vl + 0.13) + 0.012, pa[1]), UP, WHITE)

        # ---- 标线 / 挡轮杆 ----
        def _seg(su, sv, d_len, dwid, axis, col):
            """沿轴画一条标线（axis=0 沿 u，axis=1 沿 v）。
            平台有纵坡，四角必须各自取高，不能共面。"""
            if axis == 0:
                au, av = d_len, 0.0
                wu, wv = 0.0, dwid * 0.5
            else:
                au, av = 0.0, d_len
                wu, wv = dwid * 0.5, 0.0
            c = [(su + wu, sv + wv), (su + au + wu, sv + av + wv),
                 (su + au - wu, sv + av - wv), (su - wu, sv - wv)]
            q = [P(u_, 0.012, v_) for (u_, v_) in c]
            quad(q[0], q[1], q[2], q[3], UP, col)

        # 平台与缓冲带的分界白线（缓冲带整条相连，不再留引道断口）
        _seg(-HL + 0.6, -HW_ + 0.16, 2 * HL - 1.2, 0.12, 0, WHITE)

        # ---- 挡轮杆（黄黑条纹，沿路侧一字排开，挡在建筑前面）----
        STOP_H, STOP_LU, STOP_LV = 0.14, 0.75, 0.12
        n_stop = int((2*HL - 2.0) // 3.2)
        for i in range(n_stop):
            u_c = -HL + 1.0 + 3.2 * (i + 0.5)
            v_c = -HW_ + 0.34
            cs = [(u_c - STOP_LU, v_c - STOP_LV),
                  (u_c + STOP_LU, v_c - STOP_LV),
                  (u_c + STOP_LU, v_c + STOP_LV),
                  (u_c - STOP_LU, v_c + STOP_LV)]
            top = [P(u_, STOP_H, v_) for (u_, v_) in cs]
            quad(top[0], top[1], top[2], top[3], UP, WHEEL_STOP)
            for k in range(4):
                a1, b1 = cs[k], cs[(k + 1) % 4]
                col = WHEEL_STOP if k % 2 == 0 else STOP_DARK
                du_, dv_ = b1[0] - a1[0], b1[1] - a1[1]
                Ls_ = math.hypot(du_, dv_) or 1e-6
                quad(P(a1[0], 0.0, a1[1]), P(b1[0], 0.0, b1[1]),
                     P(b1[0], STOP_H, b1[1]), P(a1[0], STOP_H, a1[1]),
                     (fx*(dv_/Ls_) + rx*(du_/Ls_), 0.0,
                      fz*(dv_/Ls_) + rz*(du_/Ls_)), col)

        # ---- 车位分位线（与挡轮杆同 3.2m 栅格，引道处断开）----
        for k in range(n_stop + 1):
            u_b = -HL + 1.0 + 3.2 * k
            _seg(u_b, -HW_ + 0.4, 2.3, 0.10, 1, WHITE)

    return np.array(verts, dtype=np.float32)


# ============================================================
# 二十、起终点专用建筑群（管理栋/观景台/维修车库/便利店）
# ============================================================

def build_venue_buildings():
    """起终点建筑群 — 已缓存（缓存名挂 path 指纹）"""
    return _build_or_load_env("akina_venue_bldg_v17", _build_venue_buildings_impl)


def _build_venue_buildings_impl():
    """
    起点：管理栋（2层玻璃幕墙 + 招牌）+ 观景台（阶梯看台 + 遮阳棚）
    终点：维修车库（钢结构 + 卷帘门 + 双坡顶）+ 便利店（雨篷 + 招牌 + 玻璃窗）

    顶点格式: [pos3, nrm3, uv2, col3]
    """
    verts = []

    def quad(p1, p2, p3, p4, n, c):
        """双面 + 几何法线在返回前统一重算（见函数末尾）。
        ⚠️ 单面片绕序一错整面不可见（历史 bug：签名牌/落石标志），
        且这里 n 是局部坐标而建筑系随赛道旋转，写死必错。"""
        for pp in (p1, p2, p3, p1, p3, p4):
            verts.append([pp[0], pp[1], pp[2], n[0], n[1], n[2],
                          0.0, 0.0, c[0], c[1], c[2]])
        nn = (-n[0], -n[1], -n[2])
        for pp in (p1, p4, p3, p1, p3, p2):
            verts.append([pp[0], pp[1], pp[2], nn[0], nn[1], nn[2],
                          0.0, 0.0, c[0], c[1], c[2]])

    # ---- 配色（★ 光强 light≈2.1 / ambient≈1.5，受光面亮度 ≈ 2.4×反照率；
    #      v14 的 0.30 钢墙直接削顶成"白模"，这里所有大面 ≤0.30，重点色 ≤0.45）----
    GLASS   = (0.05, 0.07, 0.10)   # 深色玻璃
    MULL    = (0.26, 0.26, 0.28)   # 竖梃 / 窗框 / 设备
    WALL_M  = (0.20, 0.20, 0.19)   # 管理栋实墙（暖灰）
    ROOF_C  = (0.08, 0.08, 0.09)   # 平屋顶
    FRAME   = (0.13, 0.13, 0.14)   # 深灰构件（腰线/封板/雨棚）
    SIGN_R  = (0.40, 0.05, 0.04)   # 招牌红
    SIGN_W  = (0.50, 0.49, 0.45)   # 招牌白条
    CONC    = (0.20, 0.20, 0.19)   # 混凝土（看台）
    PLINTH  = (0.09, 0.09, 0.10)   # 勒脚
    STEEL   = (0.11, 0.14, 0.18)   # 车库蓝钢墙
    SLAT_A  = (0.17, 0.18, 0.20)   # 卷帘门板条（亮）
    SLAT_B  = (0.06, 0.06, 0.07)   # 卷帘门板条（暗）
    PILLAR  = (0.19, 0.19, 0.20)   # 壁柱
    ROOF_R  = (0.07, 0.08, 0.10)   # 双坡顶
    ROOF_S  = (0.08, 0.08, 0.09)   # 便利店屋顶
    CREAM   = (0.30, 0.27, 0.20)   # 便利店奶黄墙
    AWN     = (0.38, 0.07, 0.05)   # 红雨篷
    DOOR_D  = (0.03, 0.03, 0.04)   # 门洞深色
    UP      = (0.0, 1.0, 0.0)

    def make_box(P):
        """局部坐标轴对齐盒（5 面贴片，底面省略 —— 反正看不见）。"""
        def box(x0, x1, y0, y1, z0, z1, c):
            quad(P(x0, y0, z0), P(x1, y0, z0), P(x1, y1, z0),
                 P(x0, y1, z0), (0, 0, -1), c)
            quad(P(x1, y0, z0), P(x1, y0, z1), P(x1, y1, z1),
                 P(x1, y1, z0), (1, 0, 0), c)
            quad(P(x1, y0, z1), P(x0, y0, z1), P(x0, y1, z1),
                 P(x1, y1, z1), (0, 0, 1), c)
            quad(P(x0, y0, z1), P(x0, y0, z0), P(x0, y1, z0),
                 P(x0, y1, z1), (-1, 0, 0), c)
            quad(P(x0, y1, z0), P(x1, y1, z0), P(x1, y1, z1),
                 P(x0, y1, z1), UP, c)
        return box

    # ============================================================
    # 起点建筑（★ 全部锚在 _venue_pads 给出的场地平台上）
    # ============================================================
    pads = _venue_pads()
    if len(pads) < 2:
        return np.empty((0, 11), dtype=np.float32)
    cx, cz, fx, fz, rx, rz, base_y, HL, HW_, grade = pads[0]

    # ★ 建筑进深上限：带子宽 2*HW，减去前 0.4 与后 0.5 的余量
    DMAX = 2.0 * HW_ - 0.9

    # ---- 管理栋 ----（玻璃幕墙 + 竖梃 + 入口雨棚 + 屋顶机组）
    BW, BD, BH = 18.0, min(5.5, DMAX), 7.5
    bu = -14.0
    bx = cx + fx*bu + rx*(-HW_ + 0.4 + BD/2)
    bz = cz + fz*bu + rz*(-HW_ + 0.4 + BD/2)
    by = base_y + grade*bu

    def B(px, py, pz):
        return (bx + fx*px + rx*pz, by + py + grade*px, bz + fz*px + rz*pz)

    boxB = make_box(B)

    # 勒脚（四周挑出 0.06）
    boxB(-BW/2-0.06, BW/2+0.06, 0.0, 0.6, -BD/2-0.06, BD/2+0.06, PLINTH)
    # 前墙（朝路）：幕墙玻璃
    quad(B(-BW/2, 0.6, -BD/2), B(BW/2, 0.6, -BD/2),
         B(BW/2, BH-0.4, -BD/2), B(-BW/2, BH-0.4, -BD/2), (0, 0, -1), GLASS)
    # 竖梃：每 2m 一根（挑出 45mm）
    x_ = -BW/2 + 1.0
    while x_ < BW/2 - 0.5:
        boxB(x_-0.07, x_+0.07, 0.6, BH-0.4, -BD/2-0.045, -BD/2, MULL)
        x_ += 2.0
    # 顶部实墙环带（女儿墙下 0.4m）
    boxB(-BW/2-0.05, BW/2+0.05, BH-0.4, BH, -BD/2-0.05, BD/2+0.05, FRAME)
    # 侧墙（暖灰实墙）+ 两道窗带
    quad(B(-BW/2, 0.6, -BD/2), B(-BW/2, 0.6, BD/2),
         B(-BW/2, BH-0.4, BD/2), B(-BW/2, BH-0.4, -BD/2), (-1, 0, 0), WALL_M)
    quad(B(BW/2, 0.6, -BD/2), B(BW/2, 0.6, BD/2),
         B(BW/2, BH-0.4, BD/2), B(BW/2, BH-0.4, -BD/2), (1, 0, 0), WALL_M)
    for y0w in (1.5, 4.5):
        quad(B(-BW/2-0.03, y0w, -BD/2+0.8), B(-BW/2-0.03, y0w, BD/2-0.8),
             B(-BW/2-0.03, y0w+1.1, BD/2-0.8), B(-BW/2-0.03, y0w+1.1, -BD/2+0.8),
             (-1, 0, 0), GLASS)
        quad(B(BW/2+0.03, y0w, -BD/2+0.8), B(BW/2+0.03, y0w, BD/2-0.8),
             B(BW/2+0.03, y0w+1.1, BD/2-0.8), B(BW/2+0.03, y0w+1.1, -BD/2+0.8),
             (1, 0, 0), GLASS)
    # 背墙（朝山，实墙）
    quad(B(-BW/2, 0.6, BD/2), B(BW/2, 0.6, BD/2),
         B(BW/2, BH-0.4, BD/2), B(-BW/2, BH-0.4, BD/2), (0, 0, 1), WALL_M)
    # 入口：深色门斗 + 雨棚 + 双柱
    #   ★ 柱是接地构件，必须落在压平带内（|v| ≤ HW+0.15，冒烟检查 6）：
    #     退到立面外 0.5m，1.2m 悬挑由雨棚板自己扛（悬空件允许外挑 1.4m）
    quad(B(-1.7, 0.6, -BD/2-0.03), B(1.7, 0.6, -BD/2-0.03),
         B(1.7, 3.0, -BD/2-0.03), B(-1.7, 3.0, -BD/2-0.03), (0, 0, -1), DOOR_D)
    boxB(-2.6, 2.6, 3.0, 3.22, -BD/2-1.7, -BD/2+0.05, FRAME)
    boxB(-2.25, -2.05, 0.0, 3.0, -BD/2-0.50, -BD/2-0.34, MULL)
    boxB(2.05, 2.25, 0.0, 3.0, -BD/2-0.50, -BD/2-0.34, MULL)
    # 楼板分层腰线
    quad(B(-BW/2-0.1, 3.5, -BD/2-0.1), B(BW/2+0.1, 3.5, -BD/2-0.1),
         B(BW/2+0.1, 3.5, BD/2+0.1), B(-BW/2-0.1, 3.5, BD/2+0.1), UP, FRAME)

    # 平屋顶 + 女儿墙
    quad(B(-BW/2-0.3, BH, -BD/2-0.3), B(BW/2+0.3, BH, -BD/2-0.3),
         B(BW/2+0.3, BH, BD/2+0.3), B(-BW/2-0.3, BH, BD/2+0.3), UP, ROOF_C)
    P_H = 0.5
    roof_edges = [
        (B(-BW/2-0.3, BH, -BD/2-0.3), B(BW/2+0.3, BH, -BD/2-0.3)),
        (B(BW/2+0.3, BH, -BD/2-0.3), B(BW/2+0.3, BH, BD/2+0.3)),
        (B(BW/2+0.3, BH, BD/2+0.3), B(-BW/2-0.3, BH, BD/2+0.3)),
        (B(-BW/2-0.3, BH, BD/2+0.3), B(-BW/2-0.3, BH, -BD/2-0.3)),
    ]
    for (a, b) in roof_edges:
        ah = (a[0], a[1]+P_H, a[2])
        bh = (b[0], b[1]+P_H, b[2])
        dx, dz = b[0]-a[0], b[2]-a[2]
        Le = math.hypot(dx, dz) or 1e-6
        en = (dz/Le, 0.0, -dx/Le)
        quad(a, b, bh, ah, en, FRAME)
    # 屋顶机组（空调箱）
    boxB(-4.0, -1.2, BH+0.02, BH+0.95, -1.0, 0.8, MULL)

    # 招牌（红底白条，坐在女儿墙上）
    boxB(-6.3, 6.3, BH+0.5, BH+1.7, -BD/2-0.38, -BD/2-0.30, SIGN_R)
    quad(B(-5.0, BH+0.85, -BD/2-0.39), B(5.0, BH+0.85, -BD/2-0.39),
         B(5.0, BH+1.35, -BD/2-0.39), B(-5.0, BH+1.35, -BD/2-0.39),
         (0, 0, -1), SIGN_W)

    # ---- 观景台 ----（阶梯看台 + 压顶矮墙 + 红饰条）
    STAND_W = 22.0
    STAND_D = min(5.0, DMAX)
    N_STEPS = 5
    STEP_H = 0.45
    STEP_D = STAND_D / N_STEPS
    gu = 15.0
    gx = cx + fx*gu + rx*(-HW_ + 0.4 + STAND_D/2)
    gz = cz + fz*gu + rz*(-HW_ + 0.4 + STAND_D/2)
    gy = base_y + grade*gu

    def G(px, py, pz):
        return (gx + fx*px + rx*pz, gy + py + grade*px, gz + fz*px + rz*pz)

    boxG = make_box(G)

    for i in range(N_STEPS):
        h = (i+1)*STEP_H
        d0 = -STAND_D/2 + i*STEP_D
        d1 = d0 + STEP_D
        quad(G(-STAND_W/2, h, d0), G(STAND_W/2, h, d0),
             G(STAND_W/2, h, d1), G(-STAND_W/2, h, d1), UP, CONC)
        if i < N_STEPS-1:
            quad(G(-STAND_W/2, h, d1), G(STAND_W/2, h, d1),
                 G(STAND_W/2, h+STEP_H, d1), G(-STAND_W/2, h+STEP_H, d1),
                 (0, 0, 1), CONC)

    top_h = N_STEPS*STEP_H
    quad(G(-STAND_W/2, top_h, STAND_D/2-STEP_D), G(STAND_W/2, top_h, STAND_D/2-STEP_D),
         G(STAND_W/2, top_h, STAND_D/2), G(-STAND_W/2, top_h, STAND_D/2), UP, CONC)

    # 压顶矮墙（前沿）+ 红饰条
    boxG(-STAND_W/2, STAND_W/2, top_h, top_h+0.85,
         -STAND_D/2-0.15, -STAND_D/2, CONC)
    boxG(-STAND_W/2-0.02, STAND_W/2+0.02, top_h+0.55, top_h+0.72,
         -STAND_D/2-0.18, -STAND_D/2-0.14, SIGN_R)

    # 遮阳棚（柱 + 棚面 + 棚沿红条）
    ROOF_H = 3.2
    for cx2 in (-STAND_W/2+0.5, STAND_W/2-0.5):
        for cz2 in (-STAND_D/2+0.5, STAND_D/2-0.5):
            q1 = G(cx2-0.12, 0, cz2-0.12)
            q2 = G(cx2+0.12, 0, cz2-0.12)
            q3 = G(cx2+0.12, 0, cz2+0.12)
            q4 = G(cx2-0.12, 0, cz2+0.12)
            h1 = G(cx2-0.12, ROOF_H, cz2-0.12)
            h2 = G(cx2+0.12, ROOF_H, cz2-0.12)
            h3 = G(cx2+0.12, ROOF_H, cz2+0.12)
            h4 = G(cx2-0.12, ROOF_H, cz2+0.12)
            quad(q1, q2, h2, h1, (0, 0, -1), MULL)
            quad(q2, q3, h3, h2, (1, 0, 0), MULL)
            quad(q3, q4, h4, h3, (0, 0, 1), MULL)
            quad(q4, q1, h1, h4, (-1, 0, 0), MULL)

    quad(G(-STAND_W/2-0.5, ROOF_H, -STAND_D/2-0.5),
         G(STAND_W/2+0.5, ROOF_H, -STAND_D/2-0.5),
         G(STAND_W/2+0.5, ROOF_H, STAND_D/2+0.5),
         G(-STAND_W/2-0.5, ROOF_H, STAND_D/2+0.5), UP, ROOF_C)
    boxG(-STAND_W/2-0.5, STAND_W/2+0.5, ROOF_H-0.25, ROOF_H-0.05,
         -STAND_D/2-0.55, -STAND_D/2-0.45, SIGN_R)

    # ============================================================
    # 终点建筑（★ 同样锚在终点场地平台上）
    # ============================================================
    cx, cz, fx, fz, rx, rz, base_y, HL, HW_, grade = pads[1]
    DMAX = 2.0 * HW_ - 0.9

    # ---- 维修车库 ----（卷帘门 + 壁柱 + 高窗 + 招牌带，卷帘门朝路面）
    GW, GD, GH = 22.0, min(5.5, DMAX), 7.0
    ru = -15.0
    gx2 = cx + fx*ru + rx*(-HW_ + 0.4 + GD/2)
    gz2 = cz + fz*ru + rz*(-HW_ + 0.4 + GD/2)
    gy2 = base_y + grade*ru

    def R(px, py, pz):
        return (gx2 + fx*px + rx*pz, gy2 + py + grade*px, gz2 + fz*px + rz*pz)

    boxR = make_box(R)

    # 地坪 + 勒脚
    quad(R(-GW/2, 0, -GD/2), R(GW/2, 0, -GD/2),
         R(GW/2, 0, GD/2), R(-GW/2, 0, GD/2), UP, (0.15, 0.15, 0.16))
    boxR(-GW/2-0.06, GW/2+0.06, 0.0, 0.8, -GD/2-0.06, GD/2+0.06, PLINTH)
    # 四面墙（蓝钢）
    quad(R(-GW/2, 0.8, -GD/2), R(GW/2, 0.8, -GD/2),
         R(GW/2, GH, -GD/2), R(-GW/2, GH, -GD/2), (0, 0, -1), STEEL)
    quad(R(GW/2, 0.8, -GD/2), R(GW/2, 0.8, GD/2),
         R(GW/2, GH, GD/2), R(GW/2, GH, -GD/2), (1, 0, 0), STEEL)
    quad(R(GW/2, 0.8, GD/2), R(-GW/2, 0.8, GD/2),
         R(-GW/2, GH, GD/2), R(GW/2, GH, GD/2), (0, 0, 1), STEEL)
    quad(R(-GW/2, 0.8, GD/2), R(-GW/2, 0.8, -GD/2),
         R(-GW/2, GH, -GD/2), R(-GW/2, GH, GD/2), (-1, 0, 0), STEEL)
    # 壁柱（角部 + 门两侧，通高挑出 50mm）
    for px_ in (-GW/2+0.1, -5.8, 5.8, GW/2-0.4):
        boxR(px_-0.15, px_+0.15, 0.8, GH, -GD/2-0.05, -GD/2, PILLAR)
    # 卷帘门（12 道板条交替明暗 + 门楣）
    for k in range(12):
        y0d = k * 0.5
        boxR(-4.0, 4.0, y0d + 0.02, y0d + 0.48, -GD/2-0.03, -GD/2+0.02,
             SLAT_A if k % 2 == 0 else SLAT_B)
    boxR(-4.2, 4.2, 6.0, 6.35, -GD/2-0.07, -GD/2+0.02, FRAME)
    # 侧门 + 两樘高窗
    boxR(9.4, 10.6, 0.8, 3.0, -GD/2-0.04, -GD/2, DOOR_D)
    for wx_ in (-8.6, 8.6):
        quad(R(wx_-1.3, 4.3, -GD/2-0.03), R(wx_+1.3, 4.3, -GD/2-0.03),
             R(wx_+1.3, 5.7, -GD/2-0.03), R(wx_-1.3, 5.7, -GD/2-0.03),
             (0, 0, -1), GLASS)
        boxR(wx_-0.05, wx_+0.05, 4.3, 5.7, -GD/2-0.06, -GD/2-0.02, MULL)
    # 招牌带（红底白条，压在檐口下）
    boxR(-5.5, 5.5, 6.4, 7.0, -GD/2-0.12, -GD/2-0.04, SIGN_R)
    quad(R(-4.2, 6.55, -GD/2-0.13), R(4.2, 6.55, -GD/2-0.13),
         R(4.2, 6.9, -GD/2-0.13), R(-4.2, 6.9, -GD/2-0.13), (0, 0, -1), SIGN_W)

    # 双坡顶 + 檐口封板
    ridge_h = GH + 2.0
    quad(R(-GW/2-0.5, GH, -GD/2-0.5), R(GW/2+0.5, GH, -GD/2-0.5),
         R(GW/2+0.5, ridge_h, 0), R(-GW/2-0.5, ridge_h, 0),
         (0, 0, -1), ROOF_R)
    quad(R(GW/2+0.5, GH, GD/2+0.5), R(-GW/2-0.5, GH, GD/2+0.5),
         R(-GW/2-0.5, ridge_h, 0), R(GW/2+0.5, ridge_h, 0),
         (0, 0, 1), ROOF_R)
    boxR(-GW/2-0.55, GW/2+0.55, GH-0.16, GH+0.04,
         -GD/2-0.58, -GD/2-0.46, FRAME)
    boxR(-GW/2-0.55, GW/2+0.55, GH-0.16, GH+0.04,
         GD/2+0.46, GD/2+0.58, FRAME)

    # ---- 便利店 ----（橱窗竖梃 + 红雨篷 + 招牌带，前脸朝路面）
    CW, CD, CH2 = 12.0, min(5.0, DMAX), 3.8
    su = 14.0
    sx2 = cx + fx*su + rx*(-HW_ + 0.4 + CD/2)
    sz2 = cz + fz*su + rz*(-HW_ + 0.4 + CD/2)
    sy2 = base_y + grade*su

    def C(px, py, pz):
        return (sx2 + fx*px + rx*pz, sy2 + py + grade*px, sz2 + fz*px + rz*pz)

    boxC = make_box(C)

    # 勒脚 + 四面墙（奶黄）
    boxC(-CW/2-0.05, CW/2+0.05, 0.0, 0.45, -CD/2-0.05, CD/2+0.05, PLINTH)
    quad(C(-CW/2, 0.45, -CD/2), C(CW/2, 0.45, -CD/2),
         C(CW/2, CH2, -CD/2), C(-CW/2, CH2, -CD/2), (0, 0, -1), CREAM)
    quad(C(CW/2, 0.45, -CD/2), C(CW/2, 0.45, CD/2),
         C(CW/2, CH2, CD/2), C(CW/2, CH2, -CD/2), (1, 0, 0), CREAM)
    quad(C(CW/2, 0.45, CD/2), C(-CW/2, 0.45, CD/2),
         C(-CW/2, CH2, CD/2), C(CW/2, CH2, CD/2), (0, 0, 1), CREAM)
    quad(C(-CW/2, 0.45, CD/2), C(-CW/2, 0.45, -CD/2),
         C(-CW/2, CH2, -CD/2), C(-CW/2, CH2, CD/2), (-1, 0, 0), CREAM)
    # 橱窗（大玻璃 + 竖梃）+ 门
    quad(C(-5.4, 0.9, -CD/2-0.03), C(2.4, 0.9, -CD/2-0.03),
         C(2.4, 2.9, -CD/2-0.03), C(-5.4, 2.9, -CD/2-0.03), (0, 0, -1), GLASS)
    xm_ = -4.8
    while xm_ < 2.4:
        boxC(xm_-0.05, xm_+0.05, 0.9, 2.9, -CD/2-0.06, -CD/2-0.02, MULL)
        xm_ += 1.6
    boxC(3.2, 4.4, 0.45, 2.7, -CD/2-0.05, -CD/2, DOOR_D)
    # 招牌带 + 红雨篷 + 吊杆
    boxC(-CW/2-0.05, CW/2+0.05, 3.05, 3.65, -CD/2-0.12, -CD/2-0.04, SIGN_R)
    quad(C(-4.6, 3.2, -CD/2-0.13), C(4.6, 3.2, -CD/2-0.13),
         C(4.6, 3.55, -CD/2-0.13), C(-4.6, 3.55, -CD/2-0.13), (0, 0, -1), SIGN_W)
    boxC(-CW/2+0.3, CW/2-0.3, 2.95, 3.07, -CD/2-1.5, -CD/2+0.05, AWN)
    # ★ 吊杆接地，退到立面外 0.42m（压平带内），1.1m 悬挑由雨篷承担
    boxC(-CW/2+0.75, -CW/2+0.83, 0.45, 2.95, -CD/2-0.42, -CD/2-0.34, MULL)
    boxC(CW/2-0.83, CW/2-0.75, 0.45, 2.95, -CD/2-0.42, -CD/2-0.34, MULL)
    # 屋顶（挑檐 0.3）
    quad(C(-CW/2-0.3, CH2, -CD/2-0.3), C(CW/2+0.3, CH2, -CD/2-0.3),
         C(CW/2+0.3, CH2, CD/2+0.3), C(-CW/2-0.3, CH2, CD/2+0.3), UP, ROOF_S)

    # ============================================================
    # ★ 起/终点龙门架：横跨路面的终点拱门 + 格纹横幅
    #   锚在场地中心沿切向 ±6m 处（确保在真实路面上方），
    #   车从门下穿过 —— 起终点最强的空间标识。
    # ============================================================
    _path_g = get_akina_path()
    for _gi, (cx, cz, fx, fz, rx, rz, base_y, HL, HW_, grade) in \
            enumerate(pads):
        _gu = 6.0 if _gi == 0 else -6.0
        _ax, _az = cx + fx * _gu, cz + fz * _gu
        _bi, _bd = 0, 1e18
        for k in range(len(_path_g)):
            d2 = (_path_g[k][0] - _ax) ** 2 + (_path_g[k][1] - _az) ** 2
            if d2 < _bd:
                _bd, _bi = d2, k
        _p0 = _path_g[_bi]
        _hw_r = 0.35 * _p0[3]
        # 切向（水平归一）与指向场地的法向
        _ka, _kb = max(0, _bi - 3), min(len(_path_g) - 1, _bi + 3)
        _tx = _path_g[_kb][0] - _path_g[_ka][0]
        _tz = _path_g[_kb][1] - _path_g[_ka][1]
        _tl = math.hypot(_tx, _tz) or 1e-6
        _tx, _tz = _tx / _tl, _tz / _tl
        _nx, _nz = -_tz, _tx
        if (_ax - _p0[0]) * _nx + (_az - _p0[1]) * _nz < 0:
            _nx, _nz = -_nx, -_nz

        # 梁：路中心高 +5.1 起，高 0.85、厚 0.5，格纹贴两侧
        _y_road = _p0[2]
        _y0, _y1 = _y_road + 5.1, _y_road + 5.95

        def GP(a, h, b):
            """龙门架局部 (法向 a / 高 h / 切向 b) → 世界"""
            return (_p0[0] + _nx * a + _tx * b,
                    _y_road + h,
                    _p0[1] + _nz * a + _tz * b)

        # 两柱：路面两侧各外 0.7m，0.45 方柱，底埋地形 0.25
        for _sgn in (1.0, -1.0):
            _ca = _sgn * (_hw_r + 0.7)
            _gy = get_ground_height(_p0[0] + _nx * _ca,
                                    _p0[1] + _nz * _ca) - 0.25
            _col_y0 = _gy - _y_road      # 转成 GP 的 h 参数
            box_g = make_box(lambda a, h, b, _ca=_ca, _y0c=_col_y0:
                             GP(_ca + a, _y0c + h, b))
            box_g(-0.225, 0.225, 0.0, (_y1 + 0.35) - _y_road - _col_y0,
                  -0.225, 0.225, FRAME)

        # 横梁（5 面，格纹单独贴在两侧大面）
        _half = _hw_r + 0.7
        quad(GP(-_half, _y1 - _y_road, -0.25), GP(_half, _y1 - _y_road, -0.25),
             GP(_half, _y0 - _y_road, -0.25), GP(-_half, _y0 - _y_road, -0.25),
             (0, 0, -1), FRAME)
        quad(GP(_half, _y1 - _y_road, 0.25), GP(-_half, _y1 - _y_road, 0.25),
             GP(-_half, _y0 - _y_road, 0.25), GP(_half, _y0 - _y_road, 0.25),
             (0, 0, 1), FRAME)
        quad(GP(-_half, _y1 - _y_road, -0.25), GP(-_half, _y1 - _y_road, 0.25),
             GP(-_half, _y0 - _y_road, 0.25), GP(-_half, _y0 - _y_road, -0.25),
             (-1, 0, 0), FRAME)
        quad(GP(_half, _y1 - _y_road, 0.25), GP(_half, _y1 - _y_road, -0.25),
             GP(_half, _y0 - _y_road, -0.25), GP(_half, _y0 - _y_road, 0.25),
             (1, 0, 0), FRAME)
        quad(GP(-_half, _y1 - _y_road, -0.25), GP(_half, _y1 - _y_road, -0.25),
             GP(_half, _y1 - _y_road, 0.25), GP(-_half, _y1 - _y_road, 0.25),
             UP, FRAME)

        # 两侧格纹横幅：0.55m 方格，白/黑交替
        _n_sq = int((2.0 * _half) // 0.55)
        for _side in (-1.0, 1.0):
            _a_out = 0.27 * _side
            for _ci in range(_n_sq):
                _b0 = -_half + _ci * 0.55
                _cc = SIGN_W if _ci % 2 == 0 else (0.05, 0.05, 0.06)
                quad(GP(_a_out, _y1 - _y_road, _b0),
                     GP(_a_out, _y1 - _y_road, _b0 + 0.55),
                     GP(_a_out, _y0 - _y_road, _b0 + 0.55),
                     GP(_a_out, _y0 - _y_road, _b0),
                     (0, 0, _side), _cc)

    # ★ 法线按几何重算：上面所有写死的 n 都是局部坐标，而建筑坐标系
    #   随赛道旋转（fx/fz ≠ 世界轴）—— 直接用会让光照完全错。
    arr = np.array(verts, dtype=np.float32).reshape(-1, 3, 11)
    pa, pb, pc = arr[:, 0, :3], arr[:, 1, :3], arr[:, 2, :3]
    nrm = np.cross(pb - pa, pc - pa)
    ln = np.linalg.norm(nrm, axis=1)
    ok = ln > 1e-9
    nrm[ok] /= ln[ok][:, None]
    arr[:, :, 3:6] = nrm[:, None, :]
    return arr.reshape(-1, 11)