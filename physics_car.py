"""
physics_car.py — 头文字D街机风格车辆动力学
自行车模型 + Pacejka魔数公式轮胎 + 纵向载荷转移 + 摩擦圆
符号约定: yaw/yaw_rate 左转为正, steer 左转为正, v_lat 车体右侧滑动为正
"""
import math

def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


# ============================================================
# EngineSimulator —— 13B-REW 转子发动机模拟
# ============================================================
class EngineSimulator:
    """
    13B-REW 双转子 + 序列式双涡轮发动机模拟
    RPM 作为独立惯性状态量，由油门驱动扭矩曲线，车速通过传动比反推负载。

    扭矩曲线基于真实 FD3S 数据：
      峰值扭矩 305 N·m @ 5000 rpm
      峰值功率 255 ps @ 6500 rpm
      怠速 800 rpm / 红线 7000 rpm / 断油 8200 rpm
    """

    # ---- 传动参数（RX-7 FD3S 规格） ----
    FINAL_DRIVE = 4.100
    TIRE_CIRCUM = 1.983           # 米（225/50R16）
    TIRE_RADIUS = TIRE_CIRCUM / (2.0 * math.pi)

    GEAR_RATIOS = {
        '1': 3.483, '2': 2.015, '3': 1.391,
        '4': 1.000, '5': 0.719, '6': 0.621,
        'R': 3.200,
    }

    # ---- 换挡阈值 (m/s)，与 hud.GearDisplay 保持一致 ----
    UP_SHIFT = {"1": 17.3, "2": 30.0, "3": 43.5, "4": 60.4, "5": 84.1}
    DOWN_SHIFT = {"2": 10.0, "3": 14.5, "4": 20.2, "5": 28.0, "6": 32.4}

    # ---- 轮上驱动力上限（N）——街机量级，防止一档扭矩放大后 1.1g 弹射 ----
    MAX_WHEEL_FORCE = 8200.0

    # ---- 13B-REW 扭矩曲线关键点 (rpm, N·m) ----
    _TORQUE_TABLE = [
        (800,  140),   # 怠速
        (1500, 170),   # 涡轮起压前
        (2000, 200),   # 主涡轮开始工作
        (2500, 240),   # 扭矩爬升
        (3000, 275),
        (3500, 292),
        (4000, 298),
        (4500, 303),
        (5000, 305),   # 峰值扭矩
        (5500, 300),
        (6000, 290),
        (6500, 280),   # 峰值功率点
        (7000, 265),   # 红线起始
        (7500, 250),
        (8000, 230),
        (8200, 200),   # 断油前
    ]

    def __init__(self):
        self.rpm = 800.0           # 当前转速
        self.rpm_idle = 800.0      # 怠速
        self.rpm_max = 8200.0      # 断油转速
        self.rpm_redline = 7000.0  # 红线起始
        self.inertia = 0.28        # 转动惯量 kg·m²（含飞轮+双转子）
        self.rpm_history = []      # 调试用最近 10 帧
        self.gear = 'N'            # 当前档位（_calc_gear 维护）
        self._gear_num = 1         # 前进档记忆（迟滞判定用）

    def _torque(self, rpm):
        """查表线性插值获得当前 RPM 下的引擎扭矩（N·m）"""
        tbl = self._TORQUE_TABLE
        if rpm <= tbl[0][0]:
            return tbl[0][1]
        if rpm >= tbl[-1][0]:
            return tbl[-1][1]
        for i in range(1, len(tbl)):
            if rpm <= tbl[i][0]:
                r0, t0 = tbl[i - 1]
                r1, t1 = tbl[i]
                frac = (rpm - r0) / (r1 - r0) if r1 > r0 else 0.0
                return t0 + (t1 - t0) * frac
        return tbl[-1][1]

    def _calc_gear(self, speed_ms, throttle=0.0):
        """根据车速自动计算档位（带迟滞，阈值与 hud.GearDisplay 一致）。
        ★ 旧版每帧从 1 档重新判定：速度 ≥17.3m/s 永远报 2 档，
        3~6 档永远选不中 → 高速下负载反推 / RPM / 发动机制动全错。
        站立油门 >0.25 时直接进 1 档（旧版报 N → 轮上力恒 0，车永远无法起步）。"""
        if speed_ms < -0.5:
            self.gear = 'R'
            return self.gear
        if abs(speed_ms) < 0.5:
            self.gear = '1' if throttle > 0.25 else 'N'
            return self.gear
        cur = self._gear_num
        if cur <= 5 and speed_ms >= self.UP_SHIFT[str(cur)]:
            cur += 1                      # 升挡
        elif cur >= 2 and speed_ms < self.DOWN_SHIFT[str(cur)]:
            cur -= 1                      # 降挡
        self._gear_num = cur
        self.gear = str(cur)
        return self.gear

    def step(self, dt, throttle, gear, v_long):
        """
        单步引擎模拟

        参数:
            dt: 时间步长
            throttle: 油门 0~1
            gear: 档位字符串 'N'/'1'~'6'/'R'
            v_long: 当前车速 m/s

        返回:
            drive_force: 轮上驱动力（N），用于替换 engine_force 直接值
        """
        # ---- 1) 引擎自转扭矩 ----
        eng_torque = self._torque(self.rpm) * throttle

        # ---- 2) 离合器耦合扭矩（方向：快的一侧拖慢的一侧） ----
        #   引擎转速 > 轮端反推转速 → load>0：引擎被拖慢、轮子被往前推（驱动）
        #   引擎转速 < 轮端反推转速 → load<0：轮子拖引擎提速、车被拖慢（发动机制动）
        #   ⚠️ 旧版把 load 直接当"阻力"从引擎扭矩里扣、又以 -load*0.3 进轮上力：
        #      松油门时 load<0 → 轮上力为正 = 滑行反而被往前推，永远没有发动机制动；
        #      且驱动力没乘总齿比（引擎 300 N·m 直接除轮径 ≈ 950 N），全油门加速无力。
        ratio = self.GEAR_RATIOS.get(gear, 0.0)
        load = 0.0
        if ratio > 0.0:
            total_ratio = ratio * self.FINAL_DRIVE
            # 车轮转速 → 引擎端等效 RPM（v=0 时轮端转速 0，
            # 引擎空转转速差直接变成起步驱动力 —— 变矩器失速行为）
            wheel_rpm = abs(v_long) * 60.0 / self.TIRE_CIRCUM
            engine_rpm_from_wheel = wheel_rpm * total_ratio
            # 转速差产生耦合扭矩
            load = (self.rpm - engine_rpm_from_wheel) * 0.30

        # ---- 2b) 发动机内摩擦/泵气损失（发动机制动的来源，松油门才明显） ----
        fric = 0.0
        if throttle < 0.05:
            fric = 20.0 + self.rpm * 0.008

        # ---- 3) 断油保护 ----
        fuel_cut = 1.0
        if self.rpm >= self.rpm_max and throttle > 0.05:
            fuel_cut = max(0.0, 1.0 - (self.rpm - self.rpm_max) / 300.0)
        # 超 8500 完全断油
        if self.rpm >= 8500.0:
            fuel_cut = 0.0

        # ---- 4) 转速变化（牛顿第二定律：角加速度 = 扭矩 / 惯量） ----
        net_torque = eng_torque * fuel_cut - load - fric
        angular_accel = net_torque / self.inertia
        self.rpm += angular_accel * dt * 8.0  # 单位换算系数

        # ---- 5) 怠速锚定 ----
        if throttle < 0.05 and abs(net_torque) < 80.0:
            self.rpm += (self.rpm_idle - self.rpm) * min(1.0, dt * 3.0)

        # ---- 6) 限幅 ----
        self.rpm = max(500.0, min(self.rpm, 8600.0))

        # ---- 7) 轮上净力 = 耦合扭矩 × 总齿比 / 轮胎半径（传动放大） ----
        #   倒挡的进/退由 _step 的 reverse_accel 分支负责，这里不产生力。
        net_force = 0.0
        if ratio > 0.0 and gear != 'R':
            total_ratio = ratio * self.FINAL_DRIVE
            net_force = load * total_ratio / self.TIRE_RADIUS
            # 街机上限，防止低档扭矩放大后弹射
            net_force = max(-self.MAX_WHEEL_FORCE,
                            min(self.MAX_WHEEL_FORCE, net_force))
            # 极小值钳制防止空挡蠕行
            if abs(net_force) < 5.0:
                net_force = 0.0

        # 调试记录
        if len(self.rpm_history) >= 10:
            self.rpm_history.pop(0)
        self.rpm_history.append(self.rpm)

        return net_force


class VehiclePhysics:
    def __init__(self, x=0.0, z=0.0, yaw=0.0):
        # ---- 状态 ----
        self.x, self.z, self.yaw = x, z, yaw
        self.yaw_rate = 0.0
        self.v_long = 0.0          # 车身纵向速度 m/s（前正）
        self.v_lat = 0.0           # 车身横向速度 m/s（右正）
        self.steer = 0.0           # 当前前轮角 rad
        self._ax = 0.0             # 上一步纵向加速度（供载荷转移）

        # ---- 整车参数（RX-7 FD 量级）----
        self.mass = 1250.0         # kg
        self.wheelbase = 2.42
        self.a = 1.20              # 质心→前轴
        self.b = 1.22              # 质心→后轴
        self.hcg = 0.46            # 质心高度 m（决定载荷转移幅度）
        self.Iz = self.mass * self.a * self.b * 0.92   # 横摆惯量，系数越小越灵活

        # ---- 轮胎（魔数公式: Fy = D*mu*Fz*sin(C*atan(B*slip))）----
        # B 越小峰值滑移角越大 → 越容易漂、漂得越宽
        self.tire_f = dict(B=6.5, C=1.60, D=1.06)
        self.tire_r = dict(B=7.0, C=1.55, D=1.04)      # 后轮略"钝"，甩出更可控

        # ---- 动力/制动/阻力 ----
        self.engine_force = 8200.0     # N（后轮）
        self.engine_fade_v = 52.0      # 动力线性衰减到 0 的速度 → 决定极速
        self.brake_force = 13500.0     # N（11500→13500，≈10.8 m/s²）
        self.brake_dist = [0.62, 0.38] # 制动力前/后分配
        self.drag = 0.55               # F = drag*v*|v|
        self.rolling_res = 14.0        # N/(m/s)
        self.handbrake_lock = 11000.0  # 5200→11000：手刹锁轮 ~0.9g

        # ---- 倒挡（街机式：停稳后按住刹车 0.4s 进入倒挡）----
        self.reverse_max_v = 8.0       # 倒车极速 m/s ≈ 29 km/h
        self.reverse_accel = 3500.0    # 倒车驱动力 N
        self._brake_hold = 0.0         # 刹车持续计时（倒挡解锁钥匙）
        self._gear_reverse = False    # ★ 持久挡位：进倒挡后保持，踩油门才回前进挡

        # ---- 摩擦圆 / 转向权限 ----
        self.fx_cap_frac = 0.85       # 纵向力最多占用抓地预算 85%（侧向永远留余量）

        # ---- 下坡制动 ----
        self.engine_brake = 500.0      # 松油门滑行的发动机制动 N

        # ---- 转向 ----
        self.steer_max = math.radians(33.0)
        self.steer_speed = 8.0         # 5→8：20m/s 下转向延迟 0.5s→0.3s
        self.steer_return = 9.0        # 7→9
        self.steer_atten_v = 9.0       # ★ 24→9：108 km/h 满舵 ≈4°，54 km/h ≈9.7°

        # ---- 街机辅助（0=拟真, 1=全辅助）----
        self.lat_assist = 0.55         # 侧滑阻尼：抑制车尾摆动，越低越"甩"
        self.align_assist = 0.372      # 自回正力矩强度 = drift_deep_rate × 本值（原0.62，减小40%）
                                       # ★ 略小于玩家转向力（1.0）：松手收线、压舵能顶住
        self.yaw_damp = 0.20           # 横摆角速度阻尼

        # ---- IDAS 式漂移控制器（按住积累版：键盘数字输入的正确解）----
        self.drift_max_deg = 52.0        # 42→52：U 型弯需要更深的角度预算
        self.drift_deep_rate = 32.0
        self.drift_recov_rate = 110.0
        self.drift_decay_rate = 22.0
        self.drift_throttle_rate = 6.0
        self.drift_hb_rate = 22.0
        self.drift_entry_deg = 12.0
        self.drift_path_gain = 0.40   # 0.32→0.40
        self.drift_path_pull = 0.34   # 0.28→0.34：U 弯二次项增强
        self.drift_scrub_min = 0.03
        self.drift_scrub_max = 0.22
        self.drift_drive = 6.5
        self.drift_exit_deg = 6.0
        self._drift_on = False
        self._drift_sgn = 1.0
        self._drift_speed = 0.0
        self._drift_vdir = 0.0
        self._drift_ang = 0.0
        self._cs_hold = 0.0
        self._drift_cooldown = 0.0
        self._drift_spin = 0.0
        self._hb_hold = 0.0            # ★ 手刹按住计时（电平触发）
        self._drift_path_rate = 0.0
        self._hb_k = 0.0               # ★ 手刹权重平滑状态（旧版一帧 0→1）
        self._drift_vmax = 0.0         # ★ 速度棘轮：本段漂移速度上限
        self._wall_stick = 0.0         # ★ 连续楔入护栏帧计数

        # ---- 低速/动力学混合 ----
        self.dyn_min_v = 1.0           # 低于此速度纯运动学
        self.dyn_full_v = 3.0          # 高于此速度纯动力学

        # ---- 导出信号（给视觉/相机用）----
        self.slip_rear = 0.0           # 后轮滑移角 rad
        self.drift = 0.0               # 平滑漂移系数 0~1（胎痕/烟/FOV）
        self.ax_s = 0.0                # 平滑纵向加速度（车身俯仰视觉）
        self.ay_s = 0.0                # 平滑横向加速度（车身侧倾视觉）

        # ---- 引擎模拟器 ----
        self.engine = EngineSimulator()

    # ---------------- 公共接口 ----------------
    def set_state(self, x, z, yaw):
        self.x, self.z, self.yaw = x, z, yaw
        self.yaw_rate = self.v_long = self.v_lat = self.steer = 0.0
        self.drift = self.slip_rear = 0.0
        self._brake_hold = 0.0
        self._gear_reverse = False
        self._drift_on = False
        self._drift_ang = 0.0
        self._drift_speed = 0.0
        self._cs_hold = 0.0
        self._drift_cooldown = 0.0
        self._drift_spin = 0.0
        self._hb_hold = 0.0
        self._drift_path_rate = 0.0
        self._hb_k = 0.0
        self._drift_vmax = 0.0
        self._wall_stick = 0.0

    def wall_collide(self, nx, nz, penetration, dt=1.0 / 60.0,
                     restitution=0.0, scrape_decel=3.5, vn_decel=25.0):
        """
        空气墙两侧其实没有竖直挡板：这里是路面→土路肩/排水沟的软边界。

        ⚠️ 旧版对法向分量做逐帧反弹（vn → -vn·0.25）：车一旦以较大法向速度
        压出边界，整条速度被反向 → v_long 由 +11.5 变 -2.9（车瞬间变成倒着走），
        再叠加漂移模型每帧写回 v_long/v_lat ⇒ 动量在原地震荡、位置被投影钉住
        —— 玩家感知就是"莫名其妙被强行刹停"（发夹弯大角度漂移 100% 触发）。

        现行模型：
          1) 位置：硬投影回边界内（每帧做，无害）
          2) 法向：只做**减速**（vn_decel m/s²，∝ dt），绝不反向、不反弹
          3) 切向：刮擦减速 scrape_decel m/s²（∝ dt，与帧率无关）
        返回压出速度（m/s），供上层判定"是否该交还抓地模型"。
        """
        self.x -= nx * penetration
        self.z -= nz * penetration

        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        vxw = -sy * self.v_long + cy * self.v_lat
        vzw = -cy * self.v_long - sy * self.v_lat
        vn = vxw * nx + vzw * nz            # 外法向分量（朝边界外为正）
        tx, tz = -nz, nx                    # 墙面切向
        vt = vxw * tx + vzw * tz

        press = vn if vn > 0.0 else 0.0     # 压出速度（只有它算"撞"）
        # 法向：吃掉压出分量，不产生反向速度（≤0 的分量原样保留 = 允许离墙）
        if vn > 0.0:
            vn = max(0.0, vn - vn_decel * dt)

        # 刮擦减速：∝ 压墙强度（滑动摩擦 ∝ 正压力），且 ∝ dt 与帧率无关。
        #   ⚠️ 旧版只要蹭到边界就恒定扣 3.5 m/s² —— 车贴着边低速挪也会被一路
        #   磨到停，这是进沟后越来越慢、最后刹停的第二层原因。
        sdir = 1.0 if vt > 0.05 else (-1.0 if vt < -0.05 else 0.0)
        k_press = min(1.0, press / 1.5 + penetration / 0.35)
        vt -= sdir * min(scrape_decel * k_press * dt, abs(vt))

        vxw = tx * vt + nx * vn
        vzw = tz * vt + nz * vn
        self.v_long = -sy * vxw - cy * vzw
        self.v_lat  =  cy * vxw - sy * vzw

        if press > 0.0:
            self.yaw_rate *= 0.5            # ★ 只在真压出帧，不再每帧叠加
        # ★ 防卡死：连续楔入 0.3s 仍无新撞击 → 切向助推脱困
        if penetration > 0.15:
            self._wall_stick += dt
        else:
            self._wall_stick = 0.0
        if self._wall_stick > 0.3 and press < 0.5:
            # （旧版这里只改局部变量 vt，在写回之后执行 = 死代码，助推从未生效）
            # 上层仍需 sync_speed_after_boundary 把结果同步给漂移模型。
            cy2, sy2 = math.cos(self.yaw), math.sin(self.yaw)
            _vx = -sy2 * self.v_long + cy2 * self.v_lat
            _vz = -cy2 * self.v_long - sy2 * self.v_lat
            _vt = _vx * tx + _vz * tz
            _vt += (4.0 * dt if sdir >= 0.0 else -4.0 * dt)
            _vn = _vx * nx + _vz * nz
            _vx = tx * _vt + nx * _vn
            _vz = tz * _vt + nz * _vn
            self.v_long = -sy2 * _vx - cy2 * _vz
            self.v_lat  =  cy2 * _vx - sy2 * _vz
        return press

    def wall_nose_collide(self, nx, nz, penetration, dt=1.0 / 60.0):
        """车头楔入墙处理：推回 + 航向对齐墙面切向（把'顶墙'变'贴墙滑'）"""
        self.x -= nx * penetration
        self.z -= nz * penetration

        tx, tz = -nz, nx                       # 墙面切向
        fx, fz = -math.sin(self.yaw), -math.cos(self.yaw)
        if fx * tx + fz * tz < 0.0:            # 取与前向夹角小的方向
            tx, tz = -tx, -tz
        target_yaw = math.atan2(-tx, -tz)      # forward=(-sin,-cos) 反解
        dyaw = (target_yaw - self.yaw + math.pi) % (2.0 * math.pi) - math.pi
        self.yaw += _clamp(dyaw, -8.0 * dt, 8.0 * dt)   # 3→8 rad/s：0.1s 内完成对齐

        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        vxw = -sy * self.v_long + cy * self.v_lat
        vzw = -cy * self.v_long - sy * self.v_lat
        vn = vxw * nx + vzw * nz
        impact = abs(vn) if vn > 0.0 else 0.0
        if vn > 0.0:
            vxw -= nx * vn * 0.9               # 保留 10% 反弹
            vzw -= nz * vn * 0.9
        self.v_long = -sy * vxw - cy * vzw
        self.v_lat = cy * vxw - sy * vzw
        return impact

    def gutter_hold(self, nx, nz, penetration, depth, dt=1.0 / 60.0):
        """
        侧沟咬地（"沟渠跑法"的核心手感）：外轮陷进沟腔时，

          1) **硬约束**：外轮越过沟外壁 → 位置投影回沟内（沟壁是实体的）
          2) **横向支撑**：吃掉大部分"往路外"的速度分量，车被兜住不外滑
          3) **纵向拖拽**：沟底摩擦，轻微掉速（不是惩罚，是重量感）
          4) **咬地增益**：略微收敛横摆，让车贴着沟走而不是被沟弹开

        nx, nz   : 指向**路外**的单位水平法线
        penetration : 外轮越过沟外壁的米数（0 = 还在沟里）
        depth    : 0~1，陷入程度（gutter_hold_depth 的返回值）
        返回 True 表示本帧确实用上了沟。
        """
        if depth <= 0.001 and penetration <= 0.0:
            return False

        # 1) 位置：硬投影回沟外壁以内
        if penetration > 0.0:
            self.x -= nx * penetration
            self.z -= nz * penetration

        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        vxw = -sy * self.v_long + cy * self.v_lat
        vzw = -cy * self.v_long - sy * self.v_lat

        vn = vxw * nx + vzw * nz            # 朝路外为正
        tx, tz = -nz, nx                    # 沿路切向
        vt = vxw * tx + vzw * tz

        # 2) 横向支撑：沟壁吃掉往路外的速度分量。
        #    ⚠️ 旧版 `vn *= (1 - 0.70~0.88)` 是**每帧乘系数**：60fps 下 0.3^60≈0，
        #       外滑速度一帧清零 = 车被沟"吸住"，和 wall_collide 修过的
        #       "贴墙一秒=停死"是同一类帧率相关 bug。
        #    → 改成减速度（m/s²，∝ dt），外滑在 ~0.1s 内被吃掉，帧率无关。
        if vn > 0.0:
            vn = max(0.0, vn - (12.0 + 8.0 * depth) * dt)
        # 3) 纵向拖拽：沟底摩擦，0.8 m/s² 的绝对减速度（只在外轮真在沟里时）。
        #    ⚠️ 旧版 `vt *= (1 - 0.8*depth*dt)` 是**指数**衰减：每秒掉 55% 速度，
        #       与注释写的"每秒 0.8 m/s"差了 14 倍 —— 沟变成了减速带。
        _s = 1.0 if vt > 0.0 else (-1.0 if vt < 0.0 else 0.0)
        vt -= _s * min(abs(vt), 0.8 * depth * dt)
        # 4) 咬地：收敛横摆，让车"咬"住沟线（同样改成 ∝ dt，时间常数 ~0.33s）
        self.yaw_rate -= self.yaw_rate * min(0.9, 3.0 * depth * dt)
        if self._drift_on and depth > 0.55:
            # 沟把车兜住 → 漂移状态更容易收回（玩家能靠沟"救车"）
            self.drift *= max(0.0, 1.0 - 6.0 * depth * dt)

        vxw = tx * vt + nx * vn
        vzw = tz * vt + nz * vn
        self.v_long = -sy * vxw - cy * vzw
        self.v_lat = cy * vxw - sy * vzw
        return True

    def sync_speed_after_boundary(self, impact=0.0):
        """边界（空气墙 / 侧沟）改完速度后**必须**调用：把结果写回漂移模型。

        ⚠️ 漂移模型每帧无条件执行 v_long = _drift_speed·cos(ang)、
        v_lat = _drift_speed·sin(ang)。只改 v_long/v_lat 而不改 _drift_speed，
        下一帧就被原样覆盖 ⇒ 边界减速全部失效，同时位置被投影钉住 →
        车在原地震荡、越飘越慢直到停死（"莫名其妙被强行刹停"的根源）。
        强撞击（impact > 1.0）依旧直接交还抓地模型（原逻辑不变）。
        """
        if not self._drift_on:
            return
        ang = self._drift_ang
        # 沿漂移方向的剩余速度即为模型应承认的新速度
        along = max(0.0,
                    self.v_long * math.cos(ang) + self.v_lat * math.sin(ang))
        self._drift_speed = min(self._drift_speed, along)
        self._drift_vmax = min(self._drift_vmax, along)
        if impact > 1.0:
            self.abort_drift()

    def abort_drift(self):
        """撞墙即交还抓地模型。v_long/v_lat 已被墙面反射重新分解、
        与当前 yaw 一致，清除漂移标志即可无缝衔接。"""
        if self._drift_on:
            self._drift_on = False
            self._cs_hold = 0.0
            self._drift_cooldown = 0.15    # 防同一面墙上立刻重新入漂

    def update(self, dt, throttle, brake, steer_in, handbrake=False,
               surface_mu=1.0, pitch_slope=0.0):
        """固定 120Hz 子步进，保证不同帧率下手感一致"""
        t, step = 0.0, 1.0 / 120.0
        while t < dt - 1e-9:
            h = min(step, dt - t)
            if self._drift_update(h, throttle, brake, steer_in, handbrake,
                                  surface_mu, pitch_slope):
                # 漂移中：手动更新位置（_step 被跳过）
                cy, sy = math.cos(self.yaw), math.sin(self.yaw)
                vxw = -sy * self.v_long + cy * self.v_lat
                vzw = -cy * self.v_long - sy * self.v_lat
                self.x += vxw * h
                self.z += vzw * h
            else:
                self._step(h, throttle, brake, steer_in, handbrake,
                           surface_mu, pitch_slope)
            t += h
        # ---- 平滑导出信号 ----
        k = min(1.0, dt * 8.0)
        if self._drift_on:
            self.drift += (1.0 - self.drift) * k
        else:
            self.drift += (_clamp(abs(self.slip_rear) / math.radians(18.0)
                                  * _clamp(abs(self.v_long) / 8.0, 0, 1)
                                  - 0.28, 0.0, 1.0) - self.drift) * k

    # ---------------- IDAS 式漂移控制器 ----------------
    def _drift_update(self, dt, throttle, brake, steer_in, handbrake, mu_s, pitch):
        """按住积累版漂移控制器。返回 True = 本子步由漂移分支接管。
        ★ 不变量：yaw ≡ _drift_vdir + _drift_ang。vdir 唯一积分，yaw 只推导。"""
        v, vl, r = self.v_long, self.v_lat, self.yaw_rate
        speed = math.hypot(v, vl)
        D2R = math.pi / 180.0

        # ★ 电平触发：按住手刹滑到合适条件自动进入，不再依赖"松开再按"
        if handbrake:
            self._hb_hold += dt
        else:
            self._hb_hold = 0.0
        hb_want = self._hb_hold > 0.10

        if self._drift_cooldown > 0.0:
            self._drift_cooldown -= dt

        # ================= 退出判定 =================
        if self._drift_on:
            exit_now = False
            chain_now = False                      # ★ 链式换向标志
            unwind = speed < 3.5
            if abs(self._drift_ang) < math.radians(4.0):
                if handbrake and steer_in == -self._drift_sgn and speed >= 6.0:
                    chain_now = True               # ★ 主动连锁：手刹+反打+有速度
                else:
                    exit_now = True
            elif not unwind and steer_in == 0 \
                 and abs(self._drift_ang) < math.radians(self.drift_exit_deg):
                self._cs_hold += dt
                if self._cs_hold > 0.12:
                    exit_now = True
            else:
                self._cs_hold = 0.0
            if chain_now:
                # ★ 种子角 = 0：yaw = vdir + ang 在换向瞬间连续
                # （旧版硬跳 ±3° 种子 → yaw 一帧跳 ~7° = 瞬移）
                self._drift_sgn = -self._drift_sgn
                self._drift_ang = 0.0
                self._drift_vmax = speed
                self._cs_hold = 0.0
                # 不 return，直接落入下面的角度控制段继续本子步
            elif exit_now:
                ang = self._drift_ang
                self.v_long = self._drift_speed * math.cos(ang)
                self.v_lat = self._drift_speed * math.sin(ang)
                self.yaw_rate = _clamp(self._drift_path_rate, -0.6, 0.6)
                self.slip_rear = -ang
                self.steer = 0.0                    # ★ 清掉漂移前的残留舵角
                self._drift_on = False
                self._cs_hold = 0.0
                return False

        # ================= 进入判定 =================
        if not self._drift_on:
            if not hb_want or self._drift_cooldown > 0.0:   # ★ 电平触发
                return False
            if speed <= 4.5 or self.v_long <= 0.2:          # ★ 降门槛：6→4.5 m/s
                return False
            slide = math.atan2(vl, v) if v > 0.5 else 0.0
            # ★ 入漂角封顶 = entry_deg：手刹注的假滑移不再被整段吞下
            #   （旧版被 beta 顶到 ~26°，起步即 3 倍设计角）
            slide = _clamp(slide, -math.radians(self.drift_entry_deg),
                           math.radians(self.drift_entry_deg))
            if abs(slide) < math.radians(5.0) and steer_in == 0 \
               and abs(self.yaw_rate) < 0.5:
                return False
            self._drift_on = True
            # ★ 动能只用前进分量：旧版 hypot(v, vl) 把手刹造出的横向速度
            #   洗白成前进动能 = 凭空加速 12% 的元凶
            self._drift_speed = v
            self._drift_vmax = v * 1.05    # ★ 棘轮基准：入漂速度 +5% 余量
            if abs(slide) < math.radians(8.0):
                d0 = steer_in if steer_in != 0.0 else \
                     (1.0 if self.yaw_rate >= 0.0 else -1.0)
                slide = math.copysign(math.radians(8.0), d0)   # 15→8
            slide = _clamp(slide, -math.radians(self.drift_max_deg),
                           math.radians(self.drift_max_deg))
            self._drift_ang = slide
            self._drift_sgn = 1.0 if slide >= 0.0 else -1.0
            self._drift_vdir = self.yaw - slide
            self._cs_hold = 0.0
            self._drift_spin = 0.0
            self._drift_path_rate = 0.0
            self.v_long = speed * math.cos(slide)
            self.v_lat = speed * math.sin(slide)
            self.yaw_rate = 0.0
            self.slip_rear = -slide
            return True

        # ================= 角度控制：按住积累式 =================
        sgn = self._drift_sgn
        a_abs = abs(self._drift_ang)
        if speed < 3.5:
            # ★ 低速强制收线：刹车/手刹/转向都不再加深，收到 4° 由退出判定交还
            a_abs -= self.drift_recov_rate * 0.5 * D2R * dt
            a_abs = max(a_abs, 0.0)
        elif steer_in == sgn:
            a_abs = min(a_abs + self.drift_deep_rate * D2R * dt,
                        math.radians(self.drift_max_deg))
        elif steer_in == -sgn:
            a_abs -= self.drift_recov_rate * D2R * dt
            if a_abs <= 0.0:
                a_abs = 0.0        # ★ 反打把角收到 0，由退出判定接管交还
        else:
            a_abs -= self.drift_decay_rate * D2R * dt
            if throttle > 0.0:
                a_abs += (self.drift_decay_rate + self.drift_throttle_rate) * D2R * dt
            a_abs = max(a_abs, 0.0)
        if handbrake and speed >= 5.0:
            a_abs = min(a_abs + self.drift_hb_rate * D2R * dt,
                        math.radians(self.drift_max_deg))   # ★ 60→22°/s
        if brake > 0.0 and speed >= 5.0 and not handbrake:   # ★ 手刹按着时刹车不再削角
            a_abs = max(a_abs - self.drift_recov_rate * 0.5 * D2R * dt, 0.0)
        # ★ 自回正力矩（align_assist 此前定义了却从未接线 —— 漂移角只会单向积累）：
        #   轮胎侧偏的固有趋势，始终把车头拉回行进方向。强度 = 玩家转向力 × 0.62，
        #   → 顺着压舵仍能加深（净 +12°/s），松手/反打就被收线（净 −42/−130°/s）。
        #   旧版给油不反打时角度反而**加深**（+6°/s），维持漂移全靠油门是错的。
        a_abs = max(a_abs - self.drift_deep_rate * self.align_assist * D2R * dt,
                    0.0)
        ang = sgn * a_abs
        d_ang = (ang - self._drift_ang) / max(dt, 1e-6)

        # ==== 路径转向：由转向键驱动（旧版误绑 d_ang → 稳态漂移走直线）====
        sp_f = _clamp(speed / 10.0, 0.0, 1.0)
        fn = a_abs / math.radians(self.drift_max_deg)
        omega_path = (steer_in * self.drift_path_gain
                      + sgn * self.drift_path_pull * fn * (1.0 + 1.2 * fn)) * sp_f
        self._drift_vdir += omega_path * dt
        self._drift_path_rate = omega_path          # ★ 退出时用

        # ============ 纵向动力学 ============
        frac = abs(self._drift_ang) / math.radians(self.drift_max_deg)   # 0~1

        # ① 滑移摩擦：角度越大掉速越猛 —— 漂移的本质代价
        #    （头文字D：全漂过发夹掉 15~25 km/h，小角度巡航漂近似不亏速）
        scrub = self.drift_scrub_min + (self.drift_scrub_max - self.drift_scrub_min) * frac
        scrub *= (2.0 - mu_s)                     # ★ 顺手接上闲置的 mu_s：路肩漂移掉速更快
        self._drift_speed -= self._drift_speed * scrub * dt

        # ② 油门：上限压到抓地引擎水平，且随角度衰减 ——
        #    漂移中油门的职责是"维持"，不是"加速"
        if throttle > 0.0:
            drive = self.drift_drive * max(0.0, 1.0 - self._drift_speed / self.engine_fade_v)
            drive *= (1.0 - 0.55 * frac)          # 摩擦圆：滑移越大纵向驱动越弱
            # ★ 定律②：漂移永远减速。油门最多把 scrub 抵消到 80%，
            #   净加速度结构上 ≤ 0（旧版小角度时 6.5 > 3%×v，净加速）
            drive = min(drive, 0.8 * self._drift_speed * scrub)
            self._drift_speed += throttle * drive * dt

        if brake > 0.0:
            self._drift_speed -= brake * 35.0 * dt
        if handbrake:
            self._drift_speed -= 3.0 * dt         # ★ 手刹在漂移里重新有减速反馈

        # ③ 坡度（漂移分支一直漏掉）：上坡掉速、下坡增速，与抓地一致
        self._drift_speed -= 9.81 * pitch * dt

        # ④ 空气阻力 + 滚阻（保留）
        self._drift_speed -= (self.drag * self._drift_speed * abs(self._drift_speed)
                              + self.rolling_res * self._drift_speed) / self.mass * dt
        self._drift_speed = _clamp(self._drift_speed, 0.0, 60.0)
        # ★ 棘轮：任何情况下不允许超过本段漂移的速度峰值
        self._drift_speed = min(self._drift_speed, self._drift_vmax)

        # ====== 写回导出（yaw 永远推导，绝不独立积分）======
        self._drift_ang = ang
        self._drift_sgn = sgn
        self.yaw = self._drift_vdir + ang             # ★ 单一真值来源
        self.v_long = self._drift_speed * math.cos(ang)
        self.v_lat = self._drift_speed * math.sin(ang)
        self._drift_spin = omega_path + d_ang
        self.yaw_rate = self._drift_spin
        self.slip_rear = -ang
        return True

    # ---------------- 单步物理 ----------------
    def _step(self, dt, throttle, brake, steer_in, handbrake, mu_s, pitch):
        m, g = self.mass, 9.81
        v, vl, r = self.v_long, self.v_lat, self.yaw_rate

        # ---- ★ 刹车优先：W+S 同按 = 纯刹车。否则引擎 8200N 直接抵消
        #      制动力（净剩 3300N），这就是下坡刹不住的元凶 ----
        if brake > 0.0:
            throttle = 0.0

        # ---- 倒挡计时：接近停稳按住刹车累积，0.4s 进入倒挡 ----
        if brake > 0.0 and (abs(v) < 0.5 or v < -0.3):
            self._brake_hold += dt
        else:
            self._brake_hold = 0.0

        # ---- ★ 持久挡位：松开 S 倒车滑行时不再"瞬间变成前进挡"，
        #      限幅器也就不会再把 atan2(vl, v)≈±180° 读成深漂移 ----
        if self._brake_hold > 0.4 and abs(v) < 0.6:
            self._gear_reverse = True
        elif throttle > 0.0 and v > -0.3:
            self._gear_reverse = False
        in_reverse = self._gear_reverse

        # ============ 1. 转向输入：抓地预算 + 滑移补偿 ============
        # 旧公式是纯运动学（假设零滑移），在前轴接近极限时实际需要的
        # 舵角还要叠加 10~15° 滑移补偿 → 上限卡死 → 推头。
        a_lat_max = 9.0 * mu_s                       # 轮胎物理极限 ~1.0g，留操作余量
        st_grip = 1.35 * math.atan(a_lat_max * self.wheelbase / max(v * v, 4.0))
        st_grip = min(st_grip, self.steer_max)
        st_cap = st_grip
        target = steer_in * st_cap
        rate = self.steer_speed if steer_in != 0.0 else self.steer_return
        self.steer += (target - self.steer) * min(1.0, rate * dt)
        self.steer = _clamp(self.steer, -self.steer_max, self.steer_max)  # ★ 终极钳制
        st = self.steer

        # ============ 3. 横向动力学：抓地受限的指令跟踪模型 ============
        # ★ 架构裁决：彻底删除 Pacejka 力矩积分。yaw_rate 不再是被力
        #   驱动的状态，而是被一阶跟踪的"指令值"——180° 自旋在物理上
        #   不存在。漂移完全交给 _drift_update（角度受控，已验证稳定）。
        speed_tot = math.hypot(v, vl)
        a_lat_max = 9.0 * mu_s                      # 抓地上限 m/s²
        r_kin = v * math.tan(st) / self.wheelbase   # 运动学横摆指令
        r_cap = a_lat_max / max(speed_tot, 3.0)     # 轮胎给得起的横摆上限
        r_tgt = _clamp(r_kin, -r_cap, r_cap)

        # 手刹：车尾滑移角目标膨胀（甩尾发起器）+ 横摆跟踪变钝
        hb_k_tgt = 0.0
        if handbrake and speed_tot > 4.0:
            hb_k_tgt = _clamp((speed_tot - 4.0) / 6.0, 0.0, 1.0)
        # ★ 旧版 hb_k 一帧 0→1 → 目标角瞬跳 28°（突兀感的来源）
        self._hb_k += (hb_k_tgt - self._hb_k) * min(1.0, 4.0 * dt)
        hb_k = self._hb_k

        r += (r_tgt - r) * min(1.0, (8.0 - 5.0 * hb_k) * dt)
        r = _clamp(r, -3.5, 3.5)

        # ---- 车身滑移角 ----
        beta_nat = 2.2 * abs(r) * self.b / max(speed_tot, 4.0)
        beta_nat = min(beta_nat, 0.18)
        if hb_k > 0.0:
            # ★ 28°→16°：手刹只需把滑移顶进漂移接管区间(>14°)，不是甩满
            beta_nat = min(beta_nat * (1.0 + 1.2 * hb_k) + 0.08 * hb_k,
                           math.radians(16.0))
        vl_eq = math.tan(beta_nat) * v
        if r_tgt < 0.0:
            vl_eq = -vl_eq                                # 左转尾甩右（右正）
        # ★ 侧向加速度硬上限 ~0.7g：旧版 6/s 跟踪在 30m/s 下 = 90 m/s²（9g）
        #   —— "拉手刹瞬间横跳/瞬移"的直接推手。轮胎物理上就给不出这个力
        dvl = _clamp(vl_eq - vl, -7.0 * dt, 7.0 * dt)
        vl += dvl
        self.slip_rear = math.atan2(-vl - self.b * r, max(abs(v), 1.5))
        ay = r * v                                        # 视觉导出（侧倾/相机）
        # ---- 纵向力：引擎模拟器驱动 + 刹车/风阻/坡度 ----
        Fx_total = 0.0

        # 引擎模拟器（含档位自动计算、发动机制动反馈）
        gear = self.engine._calc_gear(v, throttle) if not in_reverse else 'R'
        eng_force = self.engine.step(dt, throttle, gear, v)
        Fx_total += eng_force

        if brake > 0.0:
            if v > 0.3:
                Fx_total -= brake * self.brake_force
            elif in_reverse and v > -self.reverse_max_v:
                Fx_total -= brake * self.reverse_accel
            elif v <= -self.reverse_max_v:
                Fx_total += brake * 2500.0
            else:
                Fx_total -= 4000.0
        if handbrake and not in_reverse:
            # ★ 1300·v 让它在中低速也有足够咬合；11000N 上限 = 8.8 m/s²
            Fx_total -= _clamp(v * 1300.0, -self.handbrake_lock, self.handbrake_lock)
        if handbrake and brake > 0.0:
            # ★ 组合上限 1.3g：猛而可控的急停，不是 1.5g+ 的硬撞
            Fx_total = max(Fx_total, -13000.0)
        # 引擎阻力（已由 EngineSimulator 负载计算包含，此处不再额外减 engine_brake）
        Fx_total -= self.drag * v * abs(v) + self.rolling_res * v + m * g * pitch
        ax = Fx_total / m - r * vl
        v += ax * dt
        # ---- 低速/停稳 ----
        if throttle <= 0.0 and abs(v) < 0.6:
            if handbrake:
                v = 0.0
            elif brake > 0.0 and not in_reverse:
                v = 0.0
            elif brake == 0.0 and abs(pitch) <= mu_s * 0.55:
                v = 0.0
        # ---- 守门员（保留，现在永远不触发，纯保险）----
        v = _clamp(v, -10.0, 60.0)
        vl = _clamp(vl, -30.0, 30.0)
        r = _clamp(r, -3.5, 3.5)
        if not (math.isfinite(v) and math.isfinite(vl) and math.isfinite(r)):
            v = vl = r = 0.0
            self._ax = 0.0
        self._ax = _clamp(ax, -50.0, 50.0)

        self.v_long = v
        self.v_lat = vl
        self.yaw_rate = r

        # ============ 4. 世界空间位置 ============
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        vxw = -sy * self.v_long + cy * self.v_lat
        vzw = -cy * self.v_long - sy * self.v_lat
        self.x += vxw * dt
        self.z += vzw * dt
        self.yaw += self.yaw_rate * dt

        # ---- 视觉导出 ----
        self.ax_s += (ax - self.ax_s) * min(1.0, dt * 6.0)
        self.ay_s += (ay - self.ay_s) * min(1.0, dt * 6.0)