"""
动力滑行 — 榛名山弯道检测与查询
=================================
分析赛道路径的曲率，提取弯道段并分类。
提供车辆位置实时查询，返回前方最近的弯道信息。

输出不含文字，只含方向/类型/距离，由 HUD 模块渲染纯箭头。
"""

import math
import numpy as np
from mountain import get_akina_path


# ============================================================
# 曲率计算
# ============================================================

def _curvature_2d(path):
    """
    计算路径序列在 xz 平面的二维曲率 kappa[i]。
    正 = 右弯，负 = 左弯。
    使用前后切线夹角除以弧长。
    """
    n = len(path)
    kappa = np.zeros(n, dtype=np.float64)

    for i in range(1, n - 1):
        dx0 = path[i][0] - path[i - 1][0]
        dz0 = path[i][1] - path[i - 1][1]
        dx1 = path[i + 1][0] - path[i][0]
        dz1 = path[i + 1][1] - path[i][1]

        s0 = math.hypot(dx0, dz0)
        s1 = math.hypot(dx1, dz1)
        if s0 < 0.01 or s1 < 0.01:
            continue

        a0 = math.atan2(dz0, dx0)
        a1 = math.atan2(dz1, dx1)

        # 处理角度环绕
        da = a1 - a0
        while da > math.pi:
            da -= 2.0 * math.pi
        while da < -math.pi:
            da += 2.0 * math.pi

        kappa[i] = da / ((s0 + s1) * 0.5)

    return kappa


def _smooth_kappa(kappa, window=5):
    """移动平均平滑曲率，消除噪声"""
    kernel = np.ones(window) / window
    return np.convolve(kappa, kernel, mode='same')


# ============================================================
# 弯道分类阈值（曲率 = 1/半径, m⁻¹）
# ============================================================
K_HAIRPIN = 1.0 / 30.0       # 发夹弯 R < 30m
K_SHARP   = 1.0 / 60.0       # 急弯   R < 60m
K_MEDIUM  = 1.0 / 120.0      # 中速弯 R < 120m
K_GENTLE  = 1.0 / 200.0      # 高速弯 R < 200m
K_MIN     = 1.0 / 300.0      # 最小弯道检测阈值


# ============================================================
# CornerAnalyzer 类
# ============================================================

class CornerAnalyzer:
    """
    弯道分析器。启动时分析赛道路径，提取弯道段。
    提供 query() 方法供每帧查询。
    """

    def __init__(self):
        # 获取赛道路径（包含切线/法线/副法线）
        path, tangents, normals, binormals = get_akina_path(return_tangents=True)
        self.path = path                    # [(x, z, y, w, bank), ...]
        self.tangents = tangents
        self.normals = normals
        self.binormals = binormals
        self.n = len(path)

        # 曲率
        kappa = _curvature_2d(path)
        self.kappa = _smooth_kappa(kappa)

        # 累计里程
        self.mileage = self._build_mileage()

        # 弯道段列表
        self.corners = self._extract_corners()
        print(f"[弯道分析] 检测到 {len(self.corners)} 个弯道段")

    def _build_mileage(self):
        """每个路径点的累计里程（从起点开始，沿路径弧长）"""
        mileage = np.zeros(self.n, dtype=np.float64)
        for i in range(1, self.n):
            dx = self.path[i][0] - self.path[i - 1][0]
            dz = self.path[i][1] - self.path[i - 1][1]
            mileage[i] = mileage[i - 1] + math.hypot(dx, dz)
        return mileage

    def _extract_corners(self):
        """
        从曲率中提取连续弯道段。
        返回: [(入口idx, 出口idx, direction, type, peak_kappa), ...]
        """
        k = self.kappa
        raw_segments = []

        i = 0
        while i < self.n:
            if abs(k[i]) < K_MIN:
                i += 1
                continue

            # 弯道入口
            start = i
            sgn = 1.0 if k[i] > 0 else -1.0
            peak_k = abs(k[i])

            while i < self.n and abs(k[i]) >= K_MIN:
                # 方向突变（正→负或负→正）视为弯道结束
                if k[i] * sgn < -K_MIN * 0.5:
                    break
                peak_k = max(peak_k, abs(k[i]))
                i += 1

            end = i - 1

            # 过滤短片段（< 15m）
            seg_len = self.mileage[end] - self.mileage[start]
            if seg_len < 15.0:
                continue

            direction = 'right' if sgn > 0 else 'left'

            if peak_k >= K_HAIRPIN:
                ctype = 'hairpin'
            elif peak_k >= K_SHARP:
                ctype = 'sharp'
            elif peak_k >= K_MEDIUM:
                ctype = 'medium'
            else:
                ctype = 'gentle'

            raw_segments.append((start, end, direction, ctype, peak_k))

        # 合并相邻反向弯道为 S 弯
        return self._merge_s_curves(raw_segments)

    def _merge_s_curves(self, segments):
        """
        检测距离很近（< 50m）的相邻反向弯道，合并为 S 弯。
        例如: [右→40m→左] → S弯
        """
        if not segments:
            return []

        merged = [segments[0]]
        for i in range(1, len(segments)):
            prev = merged[-1]
            cur = segments[i]

            # 相邻反向弯道且间隔小于 50m
            gap = self.mileage[cur[0]] - self.mileage[prev[1]]
            if prev[2] != cur[2] and gap < 50.0:
                # 合并为 S 弯
                s_start = prev[0]
                s_end = cur[1]
                s_dir = 's'
                s_type = 's'
                s_peak = max(prev[4], cur[4])
                merged[-1] = (s_start, s_end, s_dir, s_type, s_peak)
            else:
                merged.append(cur)

        return merged

    def query(self, car_x, car_z, car_yaw, lookahead=250.0):
        """
        查询车辆前方最近的弯道。

        参数:
            car_x, car_z: 车辆世界坐标
            car_yaw: 车辆朝向（弧度）
            lookahead: 前探距离（米）

        返回:
            dict:
                'has_upcoming': bool
                'direction': 'left' | 'right' | 's'
                'type': 'hairpin' | 'sharp' | 'medium' | 'gentle' | 's'
                'distance_m': float  # 距弯道入口的路径距离
        """
        # 找最近路径点
        nearest_idx = self._nearest_point(car_x, car_z)

        # 从最近点沿路径向前搜索第一个弯道
        search_end = min(nearest_idx + 800, self.n)

        for corner in self.corners:
            entry_idx = corner[0]

            # 弯道入口必须在车辆前方
            if entry_idx <= nearest_idx:
                continue
            if entry_idx > search_end:
                continue

            # 沿路径距离
            dist = self.mileage[entry_idx] - self.mileage[nearest_idx]

            if dist <= lookahead and dist > 0:
                return {
                    'has_upcoming': True,
                    'direction': corner[2],
                    'type': corner[3],
                    'distance_m': float(dist),
                }

        return {'has_upcoming': False}

    def _nearest_point(self, x, z):
        """
        找最近路径点索引。
        使用简单的向前搜索优化：假设车辆沿赛道前进，
        从上次最近点附近开始搜索。
        """
        # 全量搜索（n≈15000 足够快，~0.1ms）
        best_d = float('inf')
        best_i = 0
        for i in range(self.n):
            px = self.path[i][0]
            pz = self.path[i][1]
            d = (px - x) * (px - x) + (pz - z) * (pz - z)
            if d < best_d:
                best_d = d
                best_i = i
        return best_i