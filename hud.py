"""
HUD 界面模块 - 小地图、速度表、档位显示等 2D 界面元素
独立于 3D 游戏逻辑，使用 OpenGL 2D 渲染

== 坐标系统 ==
所有 UI 元素使用"逻辑像素"定位（即 SDL_GetWindowSize 返回的尺寸），
渲染时通过 UICoordSystem 自动转换为 OpenGL 实际绘制坐标（适配高 DPI 缩放）。
"""

import numpy as np
import math
import random
import ctypes
from OpenGL.GL import *

f32 = np.float32


# ============================================================
# UI 坐标系统 —— 逻辑像素 ↔ 实际绘制像素 映射
# ============================================================
class UICoordSystem:
    """
    UI 坐标系统：将逻辑像素坐标映射到 OpenGL 视口坐标

    逻辑像素 = SDL_GetWindowSize 返回的窗口尺寸（用户感知的尺寸）
    实际像素 = SDL_GL_GetDrawableSize 返回的绘制表面尺寸（高 DPI 下可能放大）

    公式：实际像素 = 逻辑像素 × 缩放因子
    """

    def __init__(self, logical_w, logical_h, drawable_w, drawable_h):
        """
        参数:
            logical_w, logical_h: 窗口逻辑尺寸（SDL_GetWindowSize）
            drawable_w, drawable_h: 实际绘制尺寸（SDL_GL_GetDrawableSize）
        """
        self.logical_w = logical_w
        self.logical_h = logical_h
        self.drawable_w = drawable_w
        self.drawable_h = drawable_h
        # 计算缩放因子（高 DPI 下 > 1.0）
        self.scale_x = drawable_w / logical_w if logical_w > 0 else 1.0
        self.scale_y = drawable_h / logical_h if logical_h > 0 else 1.0

    def to_drawable(self, logical_x, logical_y, logical_w, logical_h):
        """将逻辑像素区域转换为实际绘制像素区域"""
        return (
            int(logical_x * self.scale_x),
            int(logical_y * self.scale_y),
            int(logical_w * self.scale_x),
            int(logical_h * self.scale_y),
        )

    def to_drawable_pos(self, logical_x, logical_y):
        """将逻辑像素位置转换为实际绘制位置"""
        return (int(logical_x * self.scale_x), int(logical_y * self.scale_y))


# ============================================================
# 4x4 矩阵工具
# ============================================================
def _mat4_ortho(left, right, bottom, top, near, far):
    """正交投影矩阵"""
    return np.array([
        [2/(right-left), 0, 0, -(right+left)/(right-left)],
        [0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom)],
        [0, 0, -2/(far-near), -(far+near)/(far-near)],
        [0, 0, 0, 1]
    ], dtype=f32)


# ============================================================
# 共享 HUD 着色器程序（懒加载单例）
# 所有 HUD 组件共用一份着色器，省 4 次编译 + 4 次 uniform 查询
# ============================================================
_shared_hud = {"prog": None, "uMVP": None}

def get_hud_program():
    """获取共享的 HUD 着色器程序（懒加载）"""
    if _shared_hud["prog"] is None:
        _shared_hud["prog"] = _create_shader_program(HUD_VERTEX_SRC, HUD_FRAGMENT_SRC)
        _shared_hud["uMVP"] = glGetUniformLocation(_shared_hud["prog"], "uMVP")
    return _shared_hud["prog"], _shared_hud["uMVP"]


# ============================================================
# HUD 着色器（2D 位置 + RGBA 颜色，支持透明度）
# ============================================================
HUD_VERTEX_SRC = """
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec4 aColor;
out vec4 vColor;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPos, 0.0, 1.0);
    vColor = aColor;
}
"""

HUD_FRAGMENT_SRC = """
#version 330 core
in vec4 vColor;
out vec4 FragColor;
void main() {
    FragColor = vColor;
}
"""


def _compile_shader(src, shader_type):
    """编译单个着色器"""
    shader = glCreateShader(shader_type)
    glShaderSource(shader, src)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(shader).decode()
        typ = "顶点" if shader_type == GL_VERTEX_SHADER else "片段"
        print(f"HUD {typ}着色器编译错误: {log}")
        glDeleteShader(shader)
        return None
    return shader


def _create_shader_program(vsrc, fsrc):
    """创建 HUD 着色器程序"""
    v = _compile_shader(vsrc, GL_VERTEX_SHADER)
    f = _compile_shader(fsrc, GL_FRAGMENT_SHADER)
    if not v or not f:
        return None
    p = glCreateProgram()
    glAttachShader(p, v)
    glAttachShader(p, f)
    glLinkProgram(p)
    if not glGetProgramiv(p, GL_LINK_STATUS):
        print(f"HUD 着色器链接错误: {glGetProgramInfoLog(p).decode()}")
        glDeleteProgram(p)
        return None
    glDeleteShader(v)
    glDeleteShader(f)
    return p


# ============================================================
# 小地图 (Minimap) 类
# ============================================================
class Minimap:
    """
    小地图组件：绘制赛道俯视轮廓 + 玩家位置箭头 + 边框角标
    使用 UI 坐标系统定位，自动适配各种分辨率和 DPI 缩放
    """

    # 小地图外观尺寸（逻辑像素）
    MM_SIZE = 200
    MM_PAD = 12  # 距屏幕边缘的内边距（逻辑像素）

    def __init__(self, ui_coord, track_path, pos="top_left"):
        """
        初始化小地图

        参数:
            ui_coord: UICoordSystem 实例
            track_path: 赛道路径点列表 [(x, z, y), ...]
            pos: 位置模式 "top_left"（左上角）| "center"（屏幕中心）
        """
        self._ui = ui_coord
        self._pos = pos

        # --- 使用共享 HUD 着色器 ---
        self._hud_prog, self._hud_uMVP = get_hud_program()

        # --- 计算赛道包围盒 ---
        self._track_path = track_path
        xs_path = [p[0] for p in track_path]
        zs_path = [p[1] for p in track_path]
        pad = 40  # 世界坐标边距
        mm_x_min = min(xs_path) - pad
        mm_x_max = max(xs_path) + pad
        mm_z_min = min(zs_path) - pad
        mm_z_max = max(zs_path) + pad

        self._mm_x_min = mm_x_min
        self._mm_x_max = mm_x_max
        self._mm_z_min = mm_z_min
        self._mm_z_max = mm_z_max

        # --- 正交投影矩阵（世界坐标 → NDC） ---
        # 交换 z_min/z_max 使赛道方向正确（z 大值在屏幕下方）
        self._proj = _mat4_ortho(mm_x_min, mm_x_max, mm_z_max, mm_z_min, -1, 1)

        # --- 用 UI 坐标系统计算屏幕定位 ---
        self._recalc_position()

    def _recalc_position(self):
        """根据位置模式重新计算屏幕定位"""
        if self._pos == "center":
            # 屏幕正中心
            logical_x = (self._ui.logical_w - self.MM_SIZE) // 2
            logical_y = (self._ui.logical_h - self.MM_SIZE) // 2
        else:
            # 默认：左上角
            logical_x = self.MM_PAD
            logical_y = self._ui.logical_h - self.MM_SIZE - self.MM_PAD
            if logical_y < 0:
                logical_y = self.MM_PAD
        # 转换为实际绘制像素
        self._screen_x, self._screen_y, self._screen_w, self._screen_h = \
            self._ui.to_drawable(logical_x, logical_y, self.MM_SIZE, self.MM_SIZE)

        # --- 构建背景四边形 ---
        self._init_background()

        # --- 构建赛道轮廓线 ---
        self._init_track_line()

        # --- 构建边框 + 角标 ---
        self._init_border()

        # --- 准备玩家标记（动态更新） ---
        self._init_player_marker()

        print(f"  小地图初始化: {len(self._track_path)} 路径点, "
              f"包围盒 x:[{self._mm_x_min:.0f},{self._mm_x_max:.0f}] "
              f"z:[{self._mm_z_min:.0f},{self._mm_z_max:.0f}], "
              f"位置模式: {self._pos}, "
              f"实际视口 ({self._screen_x}, {self._screen_y}, "
              f"{self._screen_w}, {self._screen_h}), "
              f"缩放因子 ({self._ui.scale_x:.2f}, {self._ui.scale_y:.2f})")

    # ---- 初始化: 背景 ----
    def _init_background(self):
        """创建半透明背景四边形"""
        col = (0.05, 0.18, 0.18, 0.75)  # 半透明深青色
        verts = np.array([
            [self._mm_x_min, self._mm_z_min, *col],
            [self._mm_x_max, self._mm_z_min, *col],
            [self._mm_x_max, self._mm_z_max, *col],
            [self._mm_x_min, self._mm_z_min, *col],
            [self._mm_x_max, self._mm_z_max, *col],
            [self._mm_x_min, self._mm_z_max, *col],
        ], dtype=f32)
        stride = 6 * 4
        self._bg_vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(self._bg_vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)
        self._bg_vbo = vbo

    # ---- 初始化: 赛道轮廓 ----
    def _init_track_line(self):
        """创建赛道轮廓线带（绿色线条）"""
        col = (0.29, 0.49, 0.25, 0.90)  # 复古绿
        vert_list = []
        for p in self._track_path:
            vert_list.extend([p[0], p[1], *col])
        data = np.array(vert_list, dtype=f32)
        stride = 6 * 4
        self._track_vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(self._track_vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)
        self._track_vbo = vbo
        self._track_count = len(self._track_path)

    # ---- 初始化: 边框 + 四角色标 ----
    def _init_border(self):
        """创建四边形边框和四角色标"""
        border_col = (0.63, 0.78, 0.88, 0.90)  # 浅蓝灰 #A1C8E0
        bw = (self._mm_x_max - self._mm_x_min) * 0.008  # 边框宽度（世界坐标）
        # 四角色标颜色
        corner_colors = [
            (1.0, 0.0, 0.0, 1.0),    # 左上 红
            (0.0, 1.0, 0.0, 1.0),    # 右上 绿
            (0.0, 0.0, 1.0, 1.0),    # 左下 蓝
            (1.0, 1.0, 0.0, 1.0),    # 右下 黄
        ]
        cs = (self._mm_x_max - self._mm_x_min) * 0.02  # 角标尺寸

        xmin, xmax = self._mm_x_min, self._mm_x_max
        zmin, zmax = self._mm_z_min, self._mm_z_max

        verts = np.array([
            # 上边
            [xmin, zmax, *border_col], [xmax, zmax, *border_col],
            [xmax, zmax-bw, *border_col], [xmin, zmax, *border_col],
            [xmax, zmax-bw, *border_col], [xmin, zmax-bw, *border_col],
            # 下边
            [xmin, zmin+bw, *border_col], [xmax, zmin+bw, *border_col],
            [xmax, zmin, *border_col], [xmin, zmin+bw, *border_col],
            [xmax, zmin, *border_col], [xmin, zmin, *border_col],
            # 左边
            [xmin, zmax, *border_col], [xmin+bw, zmax, *border_col],
            [xmin+bw, zmin, *border_col], [xmin, zmax, *border_col],
            [xmin+bw, zmin, *border_col], [xmin, zmin, *border_col],
            # 右边
            [xmax-bw, zmax, *border_col], [xmax, zmax, *border_col],
            [xmax, zmin, *border_col], [xmax-bw, zmax, *border_col],
            [xmax, zmin, *border_col], [xmax-bw, zmin, *border_col],
            # 四角色标（小三角形）
            # 左上 红
            [xmin, zmax, *corner_colors[0]], [xmin+cs, zmax, *corner_colors[0]],
            [xmin, zmax-cs, *corner_colors[0]],
            # 右上 绿
            [xmax-cs, zmax, *corner_colors[1]], [xmax, zmax, *corner_colors[1]],
            [xmax, zmax-cs, *corner_colors[1]],
            # 左下 蓝
            [xmin, zmin+cs, *corner_colors[2]], [xmin+cs, zmin, *corner_colors[2]],
            [xmin, zmin, *corner_colors[2]],
            # 右下 黄
            [xmax, zmin+cs, *corner_colors[3]], [xmax-cs, zmin, *corner_colors[3]],
            [xmax, zmin, *corner_colors[3]],
        ], dtype=f32)

        stride = 6 * 4
        self._border_vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(self._border_vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)
        self._border_vbo = vbo

    # ---- 初始化: 玩家箭头标记 ----
    def _init_player_marker(self):
        """创建动态玩家箭头标记"""
        self._player_vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(self._player_vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        init_data = np.zeros(3 * 6, dtype=f32)  # 3个顶点 × 6分量
        glBufferData(GL_ARRAY_BUFFER, init_data.nbytes, init_data, GL_DYNAMIC_DRAW)
        stride = 6 * 4
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)
        self._player_vbo = vbo

    # ---- 更新玩家标记位置 ----
    def update_player(self, car_pos, car_angle):
        """
        更新玩家箭头位置

        参数:
            car_pos: 车辆位置 [x, y, z]
            car_angle: 车辆朝向角（弧度）
        """
        marker_col = (0.83, 0.66, 0.33, 0.95)  # 暖金色 #d4a853
        ca = car_angle
        c_ca = math.cos(ca)
        s_ca = math.sin(ca)

        # 箭头尺寸（世界坐标）
        arrow_len = (self._mm_x_max - self._mm_x_min) * 0.035
        arrow_w = arrow_len * 0.4

        # 车辆前进方向：(-sin, -cos)
        tip = (car_pos[0] - s_ca * arrow_len, car_pos[2] - c_ca * arrow_len)
        left = (car_pos[0] + c_ca * arrow_w + s_ca * arrow_len * 0.3,
                car_pos[2] - s_ca * arrow_w + c_ca * arrow_len * 0.3)
        right = (car_pos[0] - c_ca * arrow_w + s_ca * arrow_len * 0.3,
                 car_pos[2] + s_ca * arrow_w + c_ca * arrow_len * 0.3)

        verts = np.array([
            [*tip, *marker_col],
            [*left, *marker_col],
            [*right, *marker_col],
        ], dtype=f32)

        glBindBuffer(GL_ARRAY_BUFFER, self._player_vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_DYNAMIC_DRAW)

    # ---- 渲染小地图 ----
    def render(self):
        """
        渲染小地图到屏幕左上角
        调用前需确保已保存/恢复 3D OpenGL 状态
        """
        # 使用 UI 坐标系统转换后的实际绘制像素
        glViewport(self._screen_x, self._screen_y,
                   self._screen_w, self._screen_h)

        glUseProgram(self._hud_prog)
        glUniformMatrix4fv(self._hud_uMVP, 1, GL_TRUE, self._proj)

        # 绘制背景
        glBindVertexArray(self._bg_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        # 绘制赛道轮廓线
        glBindVertexArray(self._track_vao)
        glDrawArrays(GL_LINE_STRIP, 0, self._track_count)

        # 绘制边框 + 角标
        glBindVertexArray(self._border_vao)
        glDrawArrays(GL_TRIANGLES, 0, 36)

        # 绘制玩家箭头
        glBindVertexArray(self._player_vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)

        # 恢复全屏视口（解耦后续组件，消除顺序依赖）
        glViewport(0, 0, self._ui.drawable_w, self._ui.drawable_h)

        # 恢复状态
        glBindVertexArray(0)
        glUseProgram(0)

    # ---- 清理资源 ----
    def destroy(self):
        """释放所有 OpenGL 资源（共享着色器由模块统一管理，不在此删除）"""
        for vao in [self._bg_vao, self._track_vao,
                    self._border_vao, self._player_vao]:
            glDeleteVertexArrays(1, [vao])
        for vbo in [self._bg_vbo, self._track_vbo,
                    self._border_vbo, self._player_vbo]:
            glDeleteBuffers(1, [vbo])

    # ---- 窗口尺寸更新 ----
    def resize(self, new_logical_w, new_logical_h, new_drawable_w, new_drawable_h):
        """窗口尺寸变化时重新计算定位"""
        self._ui = UICoordSystem(new_logical_w, new_logical_h,
                                 new_drawable_w, new_drawable_h)
        self._recalc_position()


# ============================================================
# 纹理着色器（用于文字渲染）
# ============================================================
TEXT_VERTEX_SRC = """
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPos, 0.0, 1.0);
    vUV = aUV;
}
"""

TEXT_FRAGMENT_SRC = """
#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
uniform vec4 uColor = vec4(1.0);
void main() {
    FragColor = texture(uTex, vUV) * uColor;
}
"""


# ============================================================
# TextRenderer —— 使用 Pillow 生成文字纹理并渲染
# ============================================================
class TextRenderer:
    """
    文字纹理渲染器（带缓存）
    用 Pillow 生成文字位图，上传为 OpenGL 纹理，渲染到屏幕指定位置。
    相同 (文字, 字号, 颜色) 组合只生成一次纹理，后续命中缓存直接绘制。
    """

    def __init__(self, ui_coord):
        from PIL import Image, ImageDraw, ImageFont

        self._ui = ui_coord

        # 加载字体（尝试多种系统字体）
        self._font_large = None  # 档位用（~64px）
        self._font_medium = None  # 速度数字（~32px）
        self._font_small = None   # 单位文字（~16px）
        # 优先中文字体（含 ASCII），确保中英文都能正常显示；找不到再回退西文字体
        font_candidates = [
            "msyh.ttc", "msyhbd.ttc",          # 微软雅黑（含中英文）
            "simhei.ttf",                       # 黑体
            "simsun.ttc", "simsunb.ttf",        # 宋体
            "arial.ttf", "Arial.ttf",
            "segoeui.ttf", "SegoeUI.ttf",
            "consola.ttf", "Consolas.ttf",
            "lucon.ttf",
        ]
        for fn in font_candidates:
            try:
                self._font_large = ImageFont.truetype(fn, 64)
                self._font_medium = ImageFont.truetype(fn, 32)
                self._font_small = ImageFont.truetype(fn, 16)
                break
            except IOError:
                continue
        if not self._font_large:
            # 回退默认字体
            self._font_large = ImageFont.load_default()
            self._font_medium = ImageFont.load_default()
            self._font_small = ImageFont.load_default()

        # 创建纹理着色器程序
        self._prog = _create_shader_program(TEXT_VERTEX_SRC, TEXT_FRAGMENT_SRC)
        if not self._prog:
            raise RuntimeError("纹理着色器创建失败")
        self._uMVP = glGetUniformLocation(self._prog, "uMVP")
        self._uTex = glGetUniformLocation(self._prog, "uTex")
        self._uColor = glGetUniformLocation(self._prog, "uColor")

        # 创建 UI 正交投影（逻辑像素 → NDC）
        # 左下角为 (0,0)，右上角为 (logical_w, logical_h)
        self._proj = _mat4_ortho(0, self._ui.logical_w,
                                 0, self._ui.logical_h, -1, 1)

        # 用于渲染纹理四边形的 VAO/VBO
        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        # 初始数据：位置 (2) + UV (2)，每帧动态更新
        init_data = np.zeros(6 * 4, dtype=f32)
        glBufferData(GL_ARRAY_BUFFER, init_data.nbytes, init_data, GL_DYNAMIC_DRAW)
        stride = 4 * 4
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

        # ---- 纹理缓存 ----
        # key = (text, font_key, color)  →  value = (tex_id, tw, th, tw_actual, th_actual)
        self._cache = {}
        self._cache_order = []
        self._cache_max = 256

    def _get_font(self, font_size):
        """根据字号选择预加载字体"""
        if font_size >= 48:
            return self._font_large, "lg"
        elif font_size >= 24:
            return self._font_medium, "md"
        else:
            return self._font_small, "sm"

    def _make_text_image(self, text, font, color):
        """用 Pillow 生成文字图像，直接使用实际尺寸（NPOT）"""
        from PIL import Image, ImageDraw
        # 获取文本边界
        bbox = font.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw < 1 or th < 1:
            # 空文字，返回 1x1 透明像素
            return b"\x00\x00\x00\x00", 1, 1, 0, 0
        img = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((-bbox[0], -bbox[1]), text, font=font, fill=color)
        return img.tobytes(), tw, th, tw, th

    def _get_text_tex(self, text, font, font_key, color):
        """从缓存获取或创建文字纹理"""
        key = (text, font_key, color)
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        # 缓存未命中：生成纹理
        data, tw, th, tw_a, th_a = self._make_text_image(text, font, color)
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tw, th, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        entry = (tex, tw, th, tw_a, th_a)
        self._cache[key] = entry
        self._cache_order.append(key)

        # FIFO 淘汰
        if len(self._cache_order) > self._cache_max:
            old_key = self._cache_order.pop(0)
            old_entry = self._cache.pop(old_key, None)
            if old_entry is not None:
                glDeleteTextures(1, [old_entry[0]])

        return entry

    def render(self, text, x, y, font_size=32,
               color=(255, 255, 255, 255), align="left"):
        """
        在逻辑像素坐标 (x, y) 处渲染文字

        参数:
            x, y: 文字左下角逻辑坐标
            font_size: 字体大小（自动选择预加载的字体）
            color: RGBA 颜色元组
            align: "left" 或 "center" 或 "right"
        """
        # 双保险：确保全屏视口（组件解耦）
        glViewport(0, 0, self._ui.drawable_w, self._ui.drawable_h)

        # 选择字体
        font, font_key = self._get_font(font_size)

        # 从缓存获取纹理
        tex_id, tw, th, tw_actual, th_actual = self._get_text_tex(
            text, font, font_key, color)

        # 空文字跳过
        if tw_actual < 1 or th_actual < 1:
            return

        # 根据对齐方式调整 x
        if align == "center":
            x -= tw_actual // 2
        elif align == "right":
            x -= tw_actual

        # 构建四边形顶点（逻辑像素坐标）
        uvw = tw_actual / tw
        uvh = th_actual / th
        verts = np.array([
            x,      y,       0.0, uvh,
            x+tw_actual, y,        uvw, uvh,
            x+tw_actual, y+th_actual, uvw, 0.0,
            x,      y,       0.0, uvh,
            x+tw_actual, y+th_actual, uvw, 0.0,
            x,      y+th_actual, 0.0, 0.0,
        ], dtype=f32)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_DYNAMIC_DRAW)

        # 渲染
        glUseProgram(self._prog)
        glUniformMatrix4fv(self._uMVP, 1, GL_TRUE, self._proj)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glUniform1i(self._uTex, 0)
        glUniform4f(self._uColor, 1.0, 1.0, 1.0, 1.0)
        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)

    def resize(self, new_logical_w, new_logical_h,
               new_drawable_w, new_drawable_h):
        """窗口尺寸变化时更新投影"""
        self._ui = UICoordSystem(new_logical_w, new_logical_h,
                                 new_drawable_w, new_drawable_h)
        self._proj = _mat4_ortho(0, self._ui.logical_w,
                                 0, self._ui.logical_h, -1, 1)

    def destroy(self):
        """释放资源（清理所有缓存纹理）"""
        # 清理缓存的纹理
        for key, (tex_id, *_) in list(self._cache.items()):
            glDeleteTextures(1, [tex_id])
        self._cache.clear()
        self._cache_order.clear()
        glDeleteVertexArrays(1, [self._vao])
        glDeleteBuffers(1, [self._vbo])
        glDeleteProgram(self._prog)


# ============================================================
# ---- RX-7 FD3S 变速箱齿比 & RPM 参数 ----
# 来源：Mazda RX-7 FD3S 6速手动变速箱典型值
# 公式：RPM = speed_mps × 60 × 终传比 × 档位齿比 / 轮胎周长
GEAR_RATIOS = {
    "R": 3.483,   # 倒挡齿比（实际减速用绝对值）
    "N": 0.0,
    "1": 3.483,
    "2": 2.015,
    "3": 1.391,
    "4": 1.000,
    "5": 0.719,
    "6": 0.621,
}
FINAL_DRIVE = 4.100          # 终传比
TIRE_CIRCUM = 1.983          # 225/50R16 轮胎周长（米），π × (胎高×2+轮毂直径)
IDLE_RPM = 800.0             # 怠速转速
MAX_RPM_VAL = 8000.0         # 表底转速
REDLINE_START = 7000.0       # 红线起始转速

# Speedometer —— 模拟指针式仪表盘（转速表版）
# ============================================================
# ============================================================
# 绘制辅助：平滑阶跃（用于纹理边缘抗锯齿）
# ============================================================
def _sstep(e0, e1, x):
    """平滑阶跃；e0 > e1 时方向自动反转"""
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _upload_rgba_texture(img):
    """PIL RGBA 图像 → OpenGL 纹理（线性过滤，边缘钳制）"""
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, img.tobytes())
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    return tex


def _panel_texture(size, radius, bg, bezel, ss=4, tex_scale=2, top_shade=0.30):
    """离屏绘制圆角面板（深色底 + 金属描边 + 顶部内阴影），返回 GL 纹理"""
    from PIL import Image, ImageDraw
    n = int(round(size * ss))
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    bg_c = tuple(int(v * 255) for v in bg[:3]) + (int(bg[3] * 255),)
    d.rounded_rectangle([0, 0, n - 1, n - 1], radius=radius * ss, fill=bg_c)
    bez = tuple(int(v * 255) for v in bezel)
    d.rounded_rectangle([0, 0, n - 1, n - 1], radius=radius * ss,
                        outline=bez + (255,), width=max(1, int(1.2 * ss)))
    arr = np.asarray(img, dtype=f32).copy()
    vvv = np.linspace(0.0, 1.0, n, dtype=f32)[:, None]
    arr[..., :3] = arr[..., :3] * (1.0 - top_shade
                                   * np.exp(-(vvv / 0.35) ** 2))[..., None]
    side = int(round(size * tex_scale))
    img = Image.fromarray(np.clip(arr, 0.0, 255.0).astype(np.uint8), "RGBA")
    img = img.resize((side, side), Image.LANCZOS)
    return _upload_rgba_texture(img)


class Speedometer:
    """
    转速表——传统机械仪表盘（转速指针 + 中央 LCD 速度数字）
    左下角：圆弧形表盘 + 转速指针 + 中央 7段数码管速度
    视觉主题：白色背景 + 黑色码表 + 红色指针 + 红色高转区
    """

    # ---- 仪表盘几何（逻辑像素坐标，圆表中心） ----
    CX = 466                       # 圆心 X（档位面板右缘 346 + 20 间距 + 半径 100，彻底让开 H 槽）
    CY = 172                       # 圆心 Y（抬高避开左下角调试文字）
    R_OUTER = 100                  # 表盘外缘（金属外圈）
    R_DIAL = 92                    # 盘面半径（金属外圈内缘）
    R_TICK_MAJOR = 84              # 大刻度外端半径
    R_TICK_MAJOR_IN = 68           # 大刻度内端半径
    R_TICK_MINOR = 82              # 小刻度外端半径
    R_TICK_MINOR_IN = 75           # 小刻度内端半径
    R_LABEL = 60                   # 刻度数字中心半径
    R_NEEDLE = 84                  # 指针长度（轴心→针尖）
    NEEDLE_TAIL = 16               # 指针尾部配重长度
    NEEDLE_HW = 4.0                # 指针根部半宽
    R_HUB = 10.5                   # 中心轴帽半径
    TEX_PAD = 6                    # 纹理留白（容纳外圈高光）
    CAPTION_PAD = 20               # 纹理顶部额外留白（x1000 r/min 标签用）

    SWEEP_DEG = 220                # 刻度弧总跨度（度）
    START_DEG = -110               # 起始角度（0°=12点方向，顺时针为正）
    MAX_RPM = 8000.0               # 表底转速
    TICK_MAJOR = 1000              # 大刻度间隔（RPM）
    TICK_MINOR = 500               # 小刻度间隔（RPM）
    REDLINE_RPM = 7000.0           # 红线区起点（此值以上刻度数字转红）

    SS = 4                         # 离屏超采样倍数
    TEX_SCALE = 2                  # 最终纹理分辨率 / 逻辑像素

    # ---- 配色（真实机械表：深黑盘面 + 金属外圈 + 橙红指针） ----
    DIAL_IN = (0.120, 0.126, 0.145)          # 盘面中心（略亮）
    DIAL_OUT = (0.022, 0.022, 0.028)         # 盘面边缘
    # 金属外圈横截面渐变 (相对位置 → 颜色)，左=内缘 右=外缘
    RING_STOPS = ((0.00, (0.10, 0.11, 0.13)),
                  (0.22, (0.72, 0.76, 0.82)),
                  (0.48, (0.34, 0.37, 0.42)),
                  (1.00, (0.07, 0.08, 0.09)))
    COL_TICK_MAJOR = (0.94, 0.95, 0.96)      # 大刻度
    COL_TICK_MINOR = (0.62, 0.65, 0.69)      # 小刻度
    COL_TICK_RED = (0.95, 0.22, 0.16)        # 红区内的刻度线
    COL_REDLINE = (0.84, 0.09, 0.10)         # 红线区弧带
    COL_NEEDLE_TIP = (1.00, 0.36, 0.12)      # 针尖（暖橙）
    COL_NEEDLE_BASE = (0.93, 0.07, 0.02)     # 针根（正红）
    COL_NEEDLE_EDGE = (0.05, 0.02, 0.01)     # 指针描边
    COL_HUB_FILL = (0.035, 0.038, 0.046)     # 轴帽底
    COL_HUB_EDGE = (0.64, 0.68, 0.74)        # 轴帽金属边

    # ---- 中央速度数码窗配置 ----
    LCD_BG = (0.045, 0.050, 0.058, 0.96)     # 窗口底色
    LCD_BEZEL = (0.42, 0.45, 0.50)           # 窗口边框
    LCD_DIGIT = (255, 176, 32)               # 点亮段（琥珀）
    LCD_OFF = (22, 15, 5)                    # 熄灭段（暗琥珀鬼影）
    LCD_GLOW = (255, 150, 20)                # 辉光色
    LCD_SEG_THICK = 4.0                      # 段厚
    LCD_DIGIT_W = 18                         # 每位宽
    LCD_DIGIT_H = 32                         # 每位高
    LCD_SPACING = 3                          # 位间距
    LCD_DIGITS = 3                           # 固定显示位数（窗口宽度恒定，数字居中）
    LCD_PAD_X = 6                            # 窗口横向内边距
    LCD_PAD_Y = 4                            # 窗口纵向内边距
    LCD_CY_OFF = -36                         # 窗口中心相对圆心的 Y 偏移
    LCD_TEX_SCALE = 2                        # 数码纹理分辨率倍数
    LCD_DIGIT_PAD = 4                        # 单位数字纹理留白（容纳辉光外溢）

    # 文字字号（纹理渲染）
    TICK_FONT = 17                           # 刻度数字
    RPM_FONT = 10                            # "x1000 r/min" 标签
    UNIT_FONT = 12                           # "km/h" 单位

    # ---- 7段数码管段定义 (A,B,C,D,E,F,G) ----
    _DIGIT_MAP = {
        '0': (1,1,1,1,1,1,0),
        '1': (0,1,1,0,0,0,0),
        '2': (1,1,0,1,1,0,1),
        '3': (1,1,1,1,0,0,1),
        '4': (0,1,1,0,0,1,1),
        '5': (1,0,1,1,0,1,1),
        '6': (1,0,1,1,1,1,1),
        '7': (1,1,1,0,0,0,0),
        '8': (1,1,1,1,1,1,1),
        '9': (1,1,1,1,0,1,1),
    }

    def __init__(self, ui_coord, text_renderer):
        self._ui = ui_coord
        self._text = text_renderer

        # 状态：速度（m/s→km/h）+ RPM
        self._speed_ms = 0.0      # 原始 m/s
        self._speed_kmh = 0.0     # 转换后 km/h
        self._rpm = IDLE_RPM      # 当前转速

        # 使用共享 HUD 着色器
        self._hud_prog, self._hud_uMVP = get_hud_program()

        # 投影矩阵
        self._proj = _mat4_ortho(0, self._ui.logical_w,
                                 0, self._ui.logical_h, -1, 1)

        # ---- 中央数码窗：窗底 + 逐位数字纹理全部预生成（运行时零构建） ----
        self._lcd_win_tex, self._lcd_win_w, self._lcd_win_h = \
            self._build_lcd_window_texture()
        self._lcd_digit_tex = {d: self._build_lcd_digit_texture(d)
                               for d in "0123456789"}

        # ---- 离屏绘制表盘 / 指针 / 轴帽纹理（启动时一次性） ----
        self._dial_tex, self._dial_w, self._dial_h = self._build_dial_texture()
        self._needle_tex, self._needle_w, self._needle_h = self._build_needle_texture()
        self._hub_tex, self._hub_w, self._hub_h = self._build_hub_texture()

        # ---- 通用纹理四边形 VBO（每帧更新，绘制表盘/指针/轴帽/LCD） ----
        self._quad_vao = glGenVertexArrays(1)
        self._quad_vbo = glGenBuffers(1)
        glBindVertexArray(self._quad_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._quad_vbo)
        init_quad = np.zeros(6 * 4, dtype=f32)   # 6 顶点 × (pos2 + uv2)
        glBufferData(GL_ARRAY_BUFFER, init_quad.nbytes, init_quad, GL_DYNAMIC_DRAW)
        stride = 4 * 4
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

    # -------------------- 离屏纹理构建 --------------------

    @staticmethod
    def _load_font(size_px, variation=None):
        """加载仪表字体（Bahnschrift 最接近车用码表数字），失败逐级回退"""
        from PIL import ImageFont
        import os
        fonts_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        for name in ("bahnschrift.ttf", "arialbd.ttf", "seguisb.ttf",
                     "arial.ttf", "msyhbd.ttc"):
            path = os.path.join(fonts_dir, name)
            if not os.path.exists(path):
                continue
            try:
                font = ImageFont.truetype(path, size_px)
            except Exception:
                continue
            if variation:
                try:
                    font.set_variation_by_name(variation)
                except Exception:
                    pass
            return font
        return ImageFont.load_default()

    @staticmethod
    def _upload_texture(img):
        """PIL 图像 → OpenGL 纹理（线性过滤，边缘钳制）"""
        return _upload_rgba_texture(img)

    @staticmethod
    def _blit_tick(rgb, lx, ly, a_deg, r_in, r_out, width, col, ss, cx_img, cy_img):
        """把一条矩形刻度（沿半径方向）叠加到 rgb 上，含覆盖度抗锯齿"""
        a = math.radians(a_deg)
        dx, dy = math.sin(a), math.cos(a)      # 刻度方向（屏幕 y 向上为正）
        nx, ny = -dy, dx                       # 垂直方向
        hw = width * 0.5
        xs, ys = [], []
        for rr in (r_in, r_out):
            for sgn in (-1.0, 1.0):
                xs.append(dx * rr + nx * hw * sgn)
                ys.append(dy * rr + ny * hw * sgn)
        nh, nw = lx.shape
        x0 = max(0, int(math.floor(cx_img + (min(xs) - 1.0) * ss)))
        x1 = min(nw, int(math.ceil(cx_img + (max(xs) + 1.0) * ss)) + 1)
        y0 = max(0, int(math.floor(cy_img - (max(ys) + 1.0) * ss)))
        y1 = min(nh, int(math.ceil(cy_img - (min(ys) - 1.0) * ss)) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        sx = lx[y0:y1, x0:x1]
        sy = ly[y0:y1, x0:x1]
        uu = sx * dx + sy * dy
        vv = -sx * dy + sy * dx
        aa = 0.55
        cov = np.clip((hw + aa - np.abs(vv)) / (2.0 * aa), 0.0, 1.0)
        cov = cov * np.clip((uu - r_in + aa) / (2.0 * aa), 0.0, 1.0)
        cov = cov * np.clip((r_out - uu + aa) / (2.0 * aa), 0.0, 1.0)
        cov3 = cov[..., None]
        cc = np.array(col, dtype=f32)[None, None, :]
        rgb[y0:y1, x0:x1] = rgb[y0:y1, x0:x1] * (1.0 - cov3) + cc * cov3

    def _build_dial_texture(self):
        """
        离屏绘制整张表盘底图：金属外圈 / 盘面渐变 / 红线区 / 刻度 /
        刻度数字 / 玻璃罩反光。SS 倍超采样后 LANCZOS 降采样。
        纹理非正方形：顶部额外留白放 "x1000 r/min" 标签。
        返回 (纹理, 逻辑宽, 逻辑高)，圆心位于纹理底边上方 half 处。
        """
        from PIL import Image, ImageDraw

        r_out = float(self.R_OUTER)
        r_dial = float(self.R_DIAL)
        ss = float(self.SS)
        half = r_out + self.TEX_PAD
        pad_top = float(self.CAPTION_PAD)
        nw = int(round(2.0 * half * ss))
        nh = int(round((2.0 * half + pad_top) * ss))
        cx_img = (nw - 1) * 0.5
        cy_img = (half + pad_top) * ss           # 圆心距图像顶部的像素数

        yy, xx = np.mgrid[0:nh, 0:nw].astype(f32)
        lx = (xx - cx_img) / ss                  # 相对圆心的 X（逻辑像素）
        ly = (cy_img - yy) / ss                  # 相对圆心的 Y（向上为正）
        r = np.sqrt(lx * lx + ly * ly)
        deg = 90.0 - np.degrees(np.arctan2(ly, lx))  # 0°=12 点方向，顺时针为正

        # ---- 1) 盘面径向渐变 + 极细同心纹（塑料盘面质感） ----
        t = np.clip(r / r_dial, 0.0, 1.0) ** 1.35
        dial_in = np.array(self.DIAL_IN, dtype=f32)
        dial_out = np.array(self.DIAL_OUT, dtype=f32)
        rgb = (dial_in[None, None, :] * (1.0 - t)[..., None]
               + dial_out[None, None, :] * t[..., None])
        rgb = rgb * (1.0 + 0.020 * np.sin(r * 1.15)
                     * np.clip(1.0 - r / r_dial, 0.0, 1.0))[..., None]

        # ---- 2) 金属外圈（横截面分段渐变） ----
        u = np.clip((r - r_dial) / (r_out - r_dial), 0.0, 1.0)
        ring = np.zeros_like(rgb)
        stops = self.RING_STOPS
        for k in range(len(stops) - 1):
            t0, col0 = stops[k]
            t1, col1 = stops[k + 1]
            sel = (u >= t0) & (u <= t1)
            if not sel.any():
                continue
            f = np.clip((u[sel] - t0) / (t1 - t0), 0.0, 1.0)[:, None]
            a0 = np.array(col0, dtype=f32)[None, :]
            a1 = np.array(col1, dtype=f32)[None, :]
            ring[sel] = a0 * (1.0 - f) + a1 * f
        w_ring = _sstep(r_dial - 0.8, r_dial + 0.8, r)
        rgb = rgb * (1.0 - w_ring)[..., None] + ring * w_ring[..., None]
        # 盘面与金属圈交界的高光细线（装饰环）
        gloss = np.exp(-((r - (r_dial - 1.0)) / 0.9) ** 2) * 0.30
        rgb = rgb + gloss[..., None] * np.array([0.58, 0.61, 0.67], dtype=f32)
        # 盘面靠外缘的一圈暗环：玻璃罩把盘面边缘压暗，制造纵深
        rim = np.exp(-((r - (r_dial - 3.0)) / 2.6) ** 2) * 0.42
        rgb = rgb * (1.0 - rim[..., None])

        # 外缘抗锯齿
        alpha = 1.0 - _sstep(r_out - 1.6, r_out, r)

        # ---- 3) 红线区弧带（刻度外侧一圈，带轻微径向明暗避免塑料感） ----
        red_deg = self.START_DEG + (self.REDLINE_RPM / self.MAX_RPM) * self.SWEEP_DEG
        band = ((deg >= red_deg) & (deg <= self.START_DEG + self.SWEEP_DEG)
                & (r >= self.R_TICK_MAJOR + 2.0) & (r <= r_dial - 1.5))
        if band.any():
            k = _sstep(red_deg, red_deg + 1.0, deg)
            shade_band = 0.86 + 0.14 * np.clip(
                (r - self.R_TICK_MAJOR - 2.0) / 4.5, 0.0, 1.0)
            rc = (np.array(self.COL_REDLINE, dtype=f32)[None, None, :]
                  * shade_band[..., None])
            blend = (k * band)[..., None] * 0.84
            rgb = rgb * (1.0 - blend) + rc * blend

        # ---- 4) 刻度线（红区内的刻度转红） ----
        for rpm_val in range(0, int(self.MAX_RPM) + 1, self.TICK_MINOR):
            a_deg = self.START_DEG + (rpm_val / self.MAX_RPM) * self.SWEEP_DEG
            in_red = rpm_val >= self.REDLINE_RPM
            if rpm_val % self.TICK_MAJOR == 0:
                r_in, r_out_t = self.R_TICK_MAJOR_IN, self.R_TICK_MAJOR
                wid = 3.2
                col = self.COL_TICK_RED if in_red else self.COL_TICK_MAJOR
            else:
                r_in, r_out_t = self.R_TICK_MINOR_IN, self.R_TICK_MINOR
                wid = 1.6
                col = self.COL_TICK_RED if in_red else self.COL_TICK_MINOR
            self._blit_tick(rgb, lx, ly, a_deg, r_in, r_out_t, wid, col, ss,
                            cx_img, cy_img)

        # ---- 5) 刻度数字 + 单位标签（正立文字） ----
        txt = Image.new("RGBA", (nw, nh), (0, 0, 0, 0))
        td = ImageDraw.Draw(txt)
        f_num = self._load_font(int(round(self.TICK_FONT * ss)), "SemiBold")
        for rpm_val in range(0, int(self.MAX_RPM) + 1, self.TICK_MAJOR):
            a = math.radians(self.START_DEG + (rpm_val / self.MAX_RPM) * self.SWEEP_DEG)
            ix = cx_img + math.sin(a) * self.R_LABEL * ss
            iy = cy_img - math.cos(a) * self.R_LABEL * ss
            if rpm_val >= self.REDLINE_RPM:
                col = (236, 58, 38, 255)
            else:
                col = (236, 238, 240, 255)
            td.text((ix, iy), str(rpm_val // 1000), font=f_num, fill=col, anchor="mm")
        f_cap = self._load_font(max(8, int(round(self.RPM_FONT * ss))), "SemiBold")
        cap = "x1000  r/min"
        cb = f_cap.getbbox(cap)
        td.text((cx_img - (cb[2] - cb[0]) * 0.5 - cb[0],
                 cy_img - (r_out + 5.0) * ss),
                cap, font=f_cap, fill=(214, 219, 226, 255))
        # km/h 单位（数码窗正下方，一并烘焙进表盘纹理，字体与盘面统一）
        f_unit = self._load_font(max(8, int(round(self.UNIT_FONT * ss))), "SemiBold")
        box_bottom = self.LCD_CY_OFF - (self.LCD_DIGIT_H * 0.5 + self.LCD_PAD_Y)
        unit_cy = box_bottom - 7.0 - self.UNIT_FONT * 0.5
        td.text((cx_img, cy_img - unit_cy * ss), "km/h",
                font=f_unit, fill=(178, 184, 192, 255), anchor="mm")
        ta = np.asarray(txt, dtype=f32)[..., 3:4] / 255.0
        tc = np.asarray(txt, dtype=f32)[..., :3] / 255.0
        rgb = rgb * (1.0 - ta) + tc * ta
        alpha = np.maximum(alpha, ta[..., 0])       # 盘外文字（x1000 r/min）也要不透明

        # ---- 6) 玻璃罩反光（叠在刻度与文字之上，低强度以免洗白盘面） ----
        glass = _sstep(r_dial, r_dial - 3.0, r)
        glare = np.zeros_like(rgb)
        s = lx * 0.60 + ly * 0.80
        glare += (np.exp(-((s - 46.0) / 20.0) ** 2) * 0.105)[..., None]   # 主斜向高光带
        up = np.clip(ly / (r + 1e-6), 0.0, 1.0)
        glare += (np.exp(-((r - r_dial * 0.78) / 13.0) ** 2) * 0.140 * up)[..., None]
        d1 = np.sqrt(lx * lx + (ly - 30.0) ** 2)
        glare += (np.exp(-(d1 / 58.0) ** 2) * 0.030)[..., None]           # 中央柔光
        d2 = np.sqrt((lx + 48.0) ** 2 + (ly - 54.0) ** 2)
        glare += (np.exp(-(d2 / 19.0) ** 2) * 0.050)[..., None]           # 副反光（穹顶曲面）
        glare *= glass[..., None]
        rgb = np.clip(rgb + glare, 0.0, 1.0)

        # ---- 7) 打包 → 降采样 → 上传 ----
        rgba = np.concatenate([rgb, alpha[..., None]], axis=2)
        img = Image.fromarray(
            (np.clip(rgba, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "RGBA")
        img = img.resize((int(round(2.0 * half * self.TEX_SCALE)),
                          int(round((2.0 * half + pad_top) * self.TEX_SCALE))),
                         Image.LANCZOS)
        return (self._upload_texture(img), 2.0 * half,
                2.0 * half + pad_top)

    def _build_needle_texture(self):
        """离屏绘制指针：楔形针身 + 尾部配重 + 深色描边 + 纵向渐变"""
        from PIL import Image, ImageDraw, ImageFilter

        ss = float(self.SS * 2)
        hw = 13.0                        # 逻辑半宽
        th = float(self.R_NEEDLE) + self.NEEDLE_TAIL
        nw = int(round(2.0 * hw * ss))
        nh = int(round(th * ss))
        ax = hw * ss                     # 轴心（图像坐标）
        ay = self.R_NEEDLE * ss

        mask = Image.new("L", (nw, nh), 0)
        md = ImageDraw.Draw(mask)
        tip_w = 1.0
        base_w = self.NEEDLE_HW
        tail_w = 2.6
        md.polygon([(ax - base_w * ss, ay), (ax - tip_w * ss, 0.5 * ss),
                    (ax + tip_w * ss, 0.5 * ss), (ax + base_w * ss, ay)], fill=255)
        md.polygon([(ax - base_w * ss, ay), (ax - tail_w * ss, nh - 1.5 * ss),
                    (ax + tail_w * ss, nh - 1.5 * ss), (ax + base_w * ss, ay)],
                   fill=255)

        m = np.asarray(mask, dtype=f32)[..., None] / 255.0
        inner = np.asarray(mask.filter(ImageFilter.MinFilter(5)),
                           dtype=f32)[..., None] / 255.0
        edge = np.clip(m - inner, 0.0, 1.0)

        ty = np.linspace(0.0, 1.0, nh, dtype=f32)[:, None, None]   # 0=针尖 1=针根
        tip_c = np.array(self.COL_NEEDLE_TIP, dtype=f32)
        base_c = np.array(self.COL_NEEDLE_BASE, dtype=f32)
        grgb = (tip_c * (1.0 - ty) + base_c * ty) * np.ones((1, nw, 1), dtype=f32)

        # ---- 圆柱体明暗：越靠针身中线越亮，两侧压暗 ----
        yv = np.arange(nh, dtype=f32)[:, None]                   # 图像行（0=针尖）
        y_axis = ay
        y_tail = nh - 1.5 * ss
        # 该行的针身半宽（逻辑像素）：楔形向针尖线性收窄，尾部再略放
        f_head = np.clip(yv / max(y_axis, 1.0), 0.0, 1.0)
        f_tail = np.clip((yv - y_axis) / max(y_tail - y_axis, 1.0), 0.0, 1.0)
        half_body = tip_w + (base_w - tip_w) * f_head
        tail_rows = (yv > y_axis).ravel()
        half_body[tail_rows, 0] = (base_w + (tail_w - base_w)
                                   * f_tail.ravel()[tail_rows])
        half_body = np.maximum(half_body, 0.35)
        q = ((np.arange(nw, dtype=f32)[None, :] - ax) / ss) / half_body    # ∈[-1,1]
        q2 = np.clip(q, -1.4, 1.4)
        # 侧翼压暗 + 窄高光带（高光乘在底色上，避免加白导致脱色）
        flank = 1.0 - 0.40 * np.clip(q2 * q2, 0.0, 1.4)
        hi = np.exp(-((q2 + 0.30) / 0.34) ** 2) * 0.30
        grgb = np.clip(grgb * (flank + hi)[..., None], 0.0, 1.0)

        edge_c = np.array(self.COL_NEEDLE_EDGE, dtype=f32)
        rgb = grgb * (1.0 - edge) + edge_c[None, None, :] * edge

        rgba = np.concatenate([np.clip(rgb, 0.0, 1.0), m], axis=2)
        img = Image.fromarray((rgba * 255.0 + 0.5).astype(np.uint8), "RGBA")
        w = int(round(2.0 * hw * self.TEX_SCALE))
        h = int(round(th * self.TEX_SCALE))
        img = img.resize((w, h), Image.LANCZOS)
        return self._upload_texture(img), 2.0 * hw, th

    def _build_hub_texture(self):
        """离屏绘制中心轴帽：金属外环（顶亮底暗）+ 深色内盘 + 中心柔光"""
        from PIL import Image

        ss = float(self.SS)
        radius = self.R_HUB
        half = radius + 2.0
        n = int(round(2.0 * half * ss))
        c = (n - 1) * 0.5
        yy, xx = np.mgrid[0:n, 0:n].astype(f32)
        lx = (xx - c) / ss
        ly = (c - yy) / ss
        r = np.sqrt(lx * lx + ly * ly)

        fill_c = np.array(self.COL_HUB_FILL, dtype=f32)
        edge_c = np.array(self.COL_HUB_EDGE, dtype=f32)
        rgb = np.broadcast_to(fill_c, (n, n, 3)).copy()
        # 金属外环：顶亮底暗 + 左上高光
        shade = 0.45 + 0.75 * np.clip(ly / (r + 1e-6), -1.0, 1.0)
        shade += 0.35 * np.clip((ly * 0.7 - lx * 0.7) / (r + 1e-6), 0.0, 1.0)
        ring = edge_c[None, None, :] * np.clip(shade, 0.30, 1.55)[..., None]
        w = _sstep(radius - 1.9, radius - 1.0, r)
        rgb = rgb * (1.0 - w)[..., None] + ring * w[..., None]
        # 内侧暗沟 + 中心柔光
        rgb = rgb * (1.0 - 0.55 * np.exp(-((r - (radius - 2.1)) / 0.6) ** 2))[..., None]
        rgb = rgb + (np.exp(-(r / (radius * 0.42)) ** 2) * 0.10
                     * np.clip(ly / (r + 1e-6) + 0.4, 0.0, 1.0))[..., None]

        alpha = 1.0 - _sstep(radius - 0.8, radius, r)
        rgba = np.concatenate([np.clip(rgb, 0.0, 1.0), alpha[..., None]], axis=2)
        img = Image.fromarray((rgba * 255.0 + 0.5).astype(np.uint8), "RGBA")
        side = int(round(2.0 * half * self.TEX_SCALE))
        img = img.resize((side, side), Image.LANCZOS)
        return self._upload_texture(img), 2.0 * half, 2.0 * half

    # -------------------- 公共接口 --------------------

    def update(self, speed_ms, gear="N", dt=1/60.0, drift=0.0, engine_rpm=None):
        """更新速度值（m/s 输入）和档位（用于计算 RPM），dt 用于 RPM 平滑
           drift=0~1 漂移系数，漂移时提升目标 RPM 模拟轮胎空转
           engine_rpm: 如果传入物理引擎的真实 RPM，则直接使用（不再反算）
           指针指向转速，中央数字显示速度。"""
        self._speed_ms = max(0.0, speed_ms)
        self._speed_kmh = self._speed_ms * 3.6

        if engine_rpm is not None:
            # 物理引擎提供了真实 RPM，直接使用（人工动画 hack 全部跳过）
            target_rpm = engine_rpm
            diff = target_rpm - self._rpm
            # 只做一阶低通滤波（模拟机械指针惯性），不做人工动画
            smooth = min(1.0, dt * 15.0)
            self._rpm += diff * smooth
            # 保留高转抖动（转子发动机振动是物理特性，不是动画 hack）
            jitter_amp = 0.0
            if self._rpm > 5500:
                jitter_amp = (self._rpm - 5500) / 2500.0 * 150.0
            if gear == "1" and self._rpm > 2000:
                jitter_amp += 80.0
            if jitter_amp > 0:
                self._rpm += random.uniform(-jitter_amp, jitter_amp) * dt * 60.0
        else:
            # 兼容旧模式：从速度反算 RPM（走原有动画管线）
            target_rpm = self._calc_rpm(speed_ms, gear, drift)
            diff = target_rpm - self._rpm
            if diff > 0:
                if gear == "1" and abs(diff) > 300:
                    smooth = min(1.0, dt * 22.0)
                else:
                    smooth = min(1.0, dt * 8.0)
                self._rpm += diff * smooth
            else:
                if abs(diff) > 800:
                    smooth = min(1.0, dt * 30.0)
                else:
                    smooth = min(1.0, dt * 18.0)
                self._rpm += diff * smooth
            # 高转抖动
            jitter_amp = 0.0
            if self._rpm > 5500:
                jitter_amp = (self._rpm - 5500) / 2500.0 * 150.0
            if gear == "1" and self._rpm > 2000:
                jitter_amp += 80.0
            if jitter_amp > 0:
                self._rpm += random.uniform(-jitter_amp, jitter_amp) * dt * 60.0

        # 限幅
        self._rpm = max(IDLE_RPM, min(self._rpm, MAX_RPM_VAL))

    def _calc_rpm(self, speed_ms, gear, drift=0.0):
        """根据车速(m/s)、档位和漂移系数计算发动机转速(RPM)
           漂移时轮胎空转，转速保持高位。"""
        # 空挡或静止 → 怠速
        if gear == "N" or abs(speed_ms) < 0.2:
            return IDLE_RPM
        ratio = GEAR_RATIOS.get(gear, 0.0)
        if ratio <= 0.0:
            return IDLE_RPM
        # RPM = speed × 60 × 终传比 × 档位齿比 / 轮胎周长
        rpm = abs(speed_ms) * 60.0 * FINAL_DRIVE * ratio / TIRE_CIRCUM

        # ---- 漂移转速补偿：模拟轮胎空转高转 ----
        if drift > 0.05:
            # 漂移系数 0.1→400, 0.5→2000, 1.0→4000 RPM 补偿
            rpm_boost = drift * 4000.0
            # 漂移越深、当前档位越低，补偿越大
            if gear in ("1", "2"):
                rpm_boost *= 1.3  # 低档位额外 +30%
            rpm += rpm_boost

        # 最低不低于怠速（低速蠕动时）
        if rpm < IDLE_RPM and speed_ms > 0.2:
            rpm = IDLE_RPM + (rpm * 0.15)
        return min(rpm, MAX_RPM_VAL)

    def render(self):
        """渲染仪表盘：表盘底图 → 中央数码窗 → 指针 → 中心轴帽 → 单位文字"""
        glUseProgram(self._text._prog)
        glUniformMatrix4fv(self._text._uMVP, 1, GL_TRUE, self._text._proj)
        glActiveTexture(GL_TEXTURE0)
        glUniform1i(self._text._uTex, 0)
        glUniform4f(self._text._uColor, 1.0, 1.0, 1.0, 1.0)
        glBindVertexArray(self._quad_vao)

        # 表盘底图（纹理比圆盘高：顶部额外留白给 x1000 r/min）
        half = self._dial_w * 0.5
        self._draw_quad(self._dial_tex, self.CX - half, self.CY - half,
                        self._dial_w, self._dial_h)

        # 中央数码速度窗（嵌在盘面内，指针不会扫到正下方）
        self._draw_lcd_speed()

        # 指针（绕轴心旋转）
        self._draw_needle()

        # 中心轴帽
        self._draw_quad(self._hub_tex,
                        self.CX - self._hub_w * 0.5,
                        self.CY - self._hub_h * 0.5,
                        self._hub_w, self._hub_h)

        glBindVertexArray(0)
        glUseProgram(0)

    # -------------------- 纹理四边形绘制 --------------------

    def _draw_quad(self, tex, x0, y0, w, h):
        """以左下角 (x0, y0) 绘制整张纹理（UV 与文字管线一致）"""
        verts = np.array([
            x0,     y0,     0.0, 1.0,
            x0 + w, y0,     1.0, 1.0,
            x0 + w, y0 + h, 1.0, 0.0,
            x0,     y0,     0.0, 1.0,
            x0 + w, y0 + h, 1.0, 0.0,
            x0,     y0 + h, 0.0, 0.0,
        ], dtype=f32)
        glUniform4f(self._text._uColor, 1.0, 1.0, 1.0, 1.0)
        glBindTexture(GL_TEXTURE_2D, tex)
        glBindBuffer(GL_ARRAY_BUFFER, self._quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_DYNAMIC_DRAW)
        glDrawArrays(GL_TRIANGLES, 0, 6)

    # -------------------- 指针绘制 --------------------

    def _draw_needle(self):
        """按当前转速旋转指针纹理（含落影：光源在左上，影子偏右下）"""
        deg = self.START_DEG + (self._rpm / self.MAX_RPM) * self.SWEEP_DEG
        rad = math.radians(-deg)
        cs, sn = math.cos(rad), math.sin(rad)

        ax = self._needle_w * 0.5
        ay = self.R_NEEDLE
        w, h = self._needle_w, self._needle_h
        # 纹理四角相对轴心（未旋转）
        corners = ((-ax, ay, 0.0, 0.0),
                   (w - ax, ay, 1.0, 0.0),
                   (w - ax, ay - h, 1.0, 1.0),
                   (-ax, ay - h, 0.0, 1.0))
        rot = []
        for (ox, oy, _u, _v) in corners:
            rot.append((ox * cs - oy * sn, ox * sn + oy * cs))

        order = (0, 1, 2, 0, 2, 3)

        def emit(dx, dy, color):
            data = []
            for i in order:
                _ox, _oy, u, v = corners[i]
                rx, ry = rot[i]
                data.extend([self.CX + rx + dx, self.CY + ry + dy, u, v])
            verts = np.array(data, dtype=f32)
            glUniform4f(self._text._uColor, *color)
            glBindBuffer(GL_ARRAY_BUFFER, self._quad_vbo)
            glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_DYNAMIC_DRAW)
            glDrawArrays(GL_TRIANGLES, 0, 6)

        glBindTexture(GL_TEXTURE_2D, self._needle_tex)
        emit(1.4, -1.8, (0.10, 0.06, 0.06, 0.42))    # 落影
        emit(0.0, 0.0, (1.0, 1.0, 1.0, 1.0))         # 指针本体

    # ---- 7 段几何工具（单位数字，坐标以数字框左上角为原点） ----

    def _seg_polys(self):
        """返回 7 个段的多边形（单位数字局部坐标，A..G 顺序）"""
        dw, dh = float(self.LCD_DIGIT_W), float(self.LCD_DIGIT_H)
        st = float(self.LCD_SEG_THICK)
        g = st * 0.42                       # 45° 斜切长度
        mid = dh * 0.5
        top, bot = st * 0.5, dh - st * 0.5
        gtop, gbot = mid - st * 0.5, mid + st * 0.5

        def hseg(x0, x1, y0):
            return [(x0 + g, y0), (x1 - g, y0), (x1, y0 + st * 0.5),
                    (x1 - g, y0 + st), (x0 + g, y0 + st), (x0, y0 + st * 0.5)]

        def vseg(x0, y0, y1):
            return [(x0, y0 + g), (x0 + st * 0.5, y0), (x0 + st, y0 + g),
                    (x0 + st, y1 - g), (x0 + st * 0.5, y1), (x0, y1 - g)]

        return (hseg(0, dw, 0),                     # A 上横
                vseg(dw - st, top, gtop),           # B 右上竖
                vseg(dw - st, gbot, bot),           # C 右下竖
                hseg(0, dw, dh - st),               # D 下横
                vseg(0, gbot, bot),                 # E 左下竖
                vseg(0, top, gtop),                 # F 左上竖
                hseg(0, dw, mid - st * 0.5))        # G 中横

    # -------------------- 中央数码速度窗 --------------------

    def _build_lcd_window_texture(self):
        """窗底：圆角深色板 + 金属描边 + 顶部内阴影（启动时一次性生成）"""
        from PIL import Image, ImageDraw

        ss = float(self.SS)
        lw = (self.LCD_DIGITS * self.LCD_DIGIT_W
              + (self.LCD_DIGITS - 1) * self.LCD_SPACING + self.LCD_PAD_X * 2)
        lh = self.LCD_DIGIT_H + self.LCD_PAD_Y * 2
        W = int(round(lw * ss))
        H = int(round(lh * ss))

        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        radius = 4.0 * ss
        bg = tuple(int(v * 255) for v in self.LCD_BG[:3]) + (int(self.LCD_BG[3] * 255),)
        d.rounded_rectangle([0, 0, W - 1, H - 1], radius=radius, fill=bg)
        bez = tuple(int(v * 255) for v in self.LCD_BEZEL)
        d.rounded_rectangle([0, 0, W - 1, H - 1], radius=radius,
                            outline=bez + (255,), width=max(1, int(ss)))
        arr = np.asarray(img, dtype=f32).copy()
        # 顶部内阴影 → 玻璃面板纵深
        vvv = np.linspace(0.0, 1.0, H, dtype=f32)[:, None]
        arr[..., :3] *= (1.0 - 0.32 * np.exp(-(vvv / 0.38) ** 2))[..., None]
        out = Image.fromarray(np.clip(arr, 0.0, 255.0).astype(np.uint8), "RGBA")
        out = out.resize((int(round(lw * self.LCD_TEX_SCALE)),
                          int(round(lh * self.LCD_TEX_SCALE))), Image.LANCZOS)
        return self._upload_texture(out), lw, lh

    def _build_lcd_digit_texture(self, ch):
        """单位数字位：熄灭段鬼影 + 点亮段 + 辉光（10 个字符各一张）"""
        from PIL import Image, ImageDraw, ImageFilter

        ss = float(self.SS)
        dw, dh = float(self.LCD_DIGIT_W), float(self.LCD_DIGIT_H)
        st, pad = float(self.LCD_SEG_THICK), float(self.LCD_DIGIT_PAD)
        lw, lh = dw + pad * 2.0, dh + pad * 2.0
        W, H = int(round(lw * ss)), int(round(lh * ss))
        polys = self._seg_polys()

        def to_px(poly):
            return [((x + pad) * ss, (y + pad) * ss) for (x, y) in poly]

        # 熄灭段（暗琥珀鬼影，勾勒出完整字位）
        base = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        bd = ImageDraw.Draw(base)
        off_c = tuple(self.LCD_OFF) + (255,)
        for poly in polys:
            bd.polygon(to_px(poly), fill=off_c)

        # 点亮段（灰度层，供辉光使用）
        lit = Image.new("L", (W, H), 0)
        ld = ImageDraw.Draw(lit)
        pattern = self._DIGIT_MAP.get(ch)
        if pattern:
            for idx, on in enumerate(pattern):
                if on:
                    ld.polygon(to_px(polys[idx]), fill=255)

        glow = lit.filter(ImageFilter.GaussianBlur(st * ss * 0.45))
        ba = np.asarray(base, dtype=f32)
        li = np.asarray(lit, dtype=f32)[..., None] / 255.0
        gi = np.asarray(glow, dtype=f32)[..., None] / 255.0
        lit_c = np.array(self.LCD_DIGIT, dtype=f32)[None, None, :]
        glow_c = np.array(self.LCD_GLOW, dtype=f32)[None, None, :]

        rgb = ba[..., :3] * (1.0 - li) + lit_c * li
        rgb = np.clip(rgb + glow_c * gi * 0.34, 0.0, 255.0)
        alpha = np.maximum(ba[..., 3],
                           np.maximum(li[..., 0] * 255.0, gi[..., 0] * 255.0 * 0.82))
        out = np.concatenate([rgb, alpha[..., None]], axis=2)
        img = Image.fromarray(np.clip(out, 0.0, 255.0).astype(np.uint8), "RGBA")
        img = img.resize((int(round(lw * self.LCD_TEX_SCALE)),
                          int(round(lh * self.LCD_TEX_SCALE))), Image.LANCZOS)
        return self._upload_texture(img), lw, lh

    def _draw_lcd_speed(self):
        """绘制中央数码窗：静态窗底 + 预生成的逐位数字纹理（零每帧构建）"""
        wl, wh_ = self._lcd_win_w, self._lcd_win_h
        wx = self.CX - wl * 0.5
        wy = self.CY + self.LCD_CY_OFF - wh_ * 0.5
        self._draw_quad(self._lcd_win_tex, wx, wy, wl, wh_)

        text = str(int(round(self._speed_kmh)))
        nch = len(text)
        step = self.LCD_DIGIT_W + self.LCD_SPACING
        # 数字在窗口内居中（空位均分到两侧）
        slot0 = self.LCD_PAD_X + max(0.0, (self.LCD_DIGITS - nch) * 0.5 * step)
        pad = self.LCD_DIGIT_PAD
        for i, ch in enumerate(text):
            info = self._lcd_digit_tex.get(ch)
            if info is None:
                continue
            tex, tw, th = info
            self._draw_quad(tex,
                            wx + slot0 + i * step - pad,
                            wy + self.LCD_PAD_Y - pad,
                            tw, th)

    # -------------------- 窗口适配与资源释放 --------------------

    def resize(self, new_logical_w, new_logical_h,
               new_drawable_w, new_drawable_h):
        """窗口尺寸变化时更新"""
        self._ui = UICoordSystem(new_logical_w, new_logical_h,
                                 new_drawable_w, new_drawable_h)
        self._proj = _mat4_ortho(0, self._ui.logical_w,
                                 0, self._ui.logical_h, -1, 1)
        # 几何体位置不变（逻辑像素定位），无需重建 VBO
        # 投影矩阵已更新，渲染自动适应实际像素

    def destroy(self):
        """释放所有 OpenGL 资源（共享着色器由模块统一管理）"""
        for tex in (self._dial_tex, self._needle_tex, self._hub_tex,
                    self._lcd_win_tex):
            glDeleteTextures(1, [tex])
        for tex, _, _ in self._lcd_digit_tex.values():
            glDeleteTextures(1, [tex])
        self._lcd_digit_tex.clear()
        glDeleteVertexArrays(1, [self._quad_vao])
        glDeleteBuffers(1, [self._quad_vbo])

# ============================================================
# GearDisplay —— H 型换挡槽档位显示器
# ============================================================
class GearDisplay:
    """
    H 型换挡槽组件（左下角，转速表左侧）
    绘制 FD3S 6 速手动 H 型换挡槽 + 档位高亮滑动动画

    H 型布局:
        1   3   5            上排
         │   │   │
        2   4   6   R        下排 + R（独立槽位，FD3S：R 在 6 正右侧）
    """

    # ---- 面板几何（逻辑像素） ----
    # ⚠️ 档位来源：main.py 直接传 phys.engine.gear（与物理共用一套迟滞判定），
    # 这里不再按车速独立推档 —— 双实现会在起步/倒挡判定上与 EngineSimulator 打架。
    PANEL_W = 330                     # 面板宽（R 槽独立，不再压在 6 上）
    PANEL_H = 252                     # 面板高（顶部加档位数码窗）
    PAD = 16                          # 距屏幕左下角边距
    CORNER_R = 10
    SLOT_W = 40                       # 槽位宽
    SLOT_H = 30                       # 槽位高
    SLOT_R = 5                        # 槽位圆角

    # ---- 三列中心 X（面板局部坐标，左下角为原点） ----
    _COL_X = (46, 146, 246)
    _ROW_TOP = 170                    # 上排 Y（1,3,5）
    _ROW_BOT = 40                     # 下排 Y（2,4,6）
    _N_Y = 105                        # N 位 Y
    _R_POS = (296, 40)                # R 位 (x, y)——独立槽位，与 6（x=246）错开

    # ---- 顶部档位数码窗（面板局部坐标，左下角原点） ----
    _GEARWIN = (26, 196, 316, 242)    # LCD 凹窗 (x0, y0, x1, y1)
    _GEARWIN_GLYPH = (286, 219)       # 当前档位大字中心（右对齐，仪表读数习惯）
    _GEARWIN_CAPTION = (56, 219)      # "GEAR" 刻字中心

    # ---- 档位锚点（面板局部坐标，中心点） ----
    _ANCHORS = {
        '1': (_COL_X[0], _ROW_TOP),
        '3': (_COL_X[1], _ROW_TOP),
        '5': (_COL_X[2], _ROW_TOP),
        'N': (_COL_X[1], _N_Y),
        '2': (_COL_X[0], _ROW_BOT),
        '4': (_COL_X[1], _ROW_BOT),
        '6': (_COL_X[2], _ROW_BOT),
        'R': _R_POS,
    }
    # ---- 上排档位（换挡路径规划用） ----
    _TOP_ROW = {'1', '3', '5'}
    _BOT_ROW = {'2', '4', '6'}

    # ---- 外观 ----
    PANEL_BG = (0.045, 0.050, 0.058, 0.94)
    PANEL_BEZEL = (0.42, 0.45, 0.50)
    SS = 4
    TEX_SCALE = 2
    FONT_SIZE = 44                    # 字形字号
    LABEL_FONT = 16                   # N/R 标签字号
    COL_GLYPH_DIM = (100, 100, 100, 140)   # 未选中字形颜色（暗灰）
    COL_GLOW = (255, 150, 20)         # 辉光色
    COL_DIGIT = (255, 176, 32)        # 选中字形色（琥珀）
    COL_HL_FRAME = (0.90, 0.65, 0.20) # 高亮框色
    HL_BLINK_SPEED = 3.0              # 脉冲闪烁速度

    # ---- 动画参数 ----
    ANIM_MOVE_SPEED = 380.0           # 高亮框移动速度 像素/秒
    ANIM_SETTLE_TIME = 0.04           # 弹性归位时长
    ANIM_SETTLE_AMP = 4.0             # 弹性振幅（像素）
    ANIM_SETTLE_DECAY = 14.0          # 弹性衰减系数

    def __init__(self, ui_coord, text_renderer):
        self._ui = ui_coord
        self._text = text_renderer
        self._gear = "N"

        self._hud_prog, self._hud_uMVP = get_hud_program()
        self._proj = _mat4_ortho(0, self._ui.logical_w,
                                 0, self._ui.logical_h, -1, 1)

        # ---- 初始化时烘焙完整 H 型槽背景纹理 ----
        self._bg_tex, self._bg_w, self._bg_h = self._build_hpattern_bg()

        # ---- 高亮框纹理（小尺寸，运行时复用） ----
        self._hl_tex = self._build_highlight_texture()

        # ---- 字形纹理缓存 ----
        self._glyph_tex = {}
        self._ensure_glyph("1")  # 预生成常用档位

        # ---- 动画状态 ----
        self._dt = 1.0 / 60.0          # 游戏帧间隔，由 update() 更新
        self._display_gear = "N"       # 当前显示位置（动画可能还在路上）
        self._anim_progress = 1.0      # 0~1：动画进度，≥1 表示完成
        self._anim_path = [(0, 0)]     # 路径锚点列表
        self._anim_path_len = 0.0      # 路径总长
        self._settle_t = 0.0           # 弹性归位计时
        self._flash_phase = 0.0        # 脉冲相位

        # ---- 纹理四边形 VBO（复用，每帧更新） ----
        self._quad_vao = glGenVertexArrays(1)
        self._quad_vbo = glGenBuffers(1)
        glBindVertexArray(self._quad_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._quad_vbo)
        init_quad = np.zeros(6 * 4, dtype=f32)
        glBufferData(GL_ARRAY_BUFFER, init_quad.nbytes, init_quad, GL_DYNAMIC_DRAW)
        stride = 4 * 4
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

    # -------------------- 纹理烘焙 --------------------

    def _load_font(self, size_px):
        """加载档位字体"""
        from PIL import ImageFont
        import os
        fonts_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        for name in ("bahnschrift.ttf", "arialbd.ttf", "seguisb.ttf",
                     "arial.ttf", "msyhbd.ttc"):
            path = os.path.join(fonts_dir, name)
            if not os.path.exists(path):
                continue
            try:
                return ImageFont.truetype(path, size_px)
            except Exception:
                continue
        return ImageFont.load_default()

    def _build_hpattern_bg(self):
        """
        离屏绘制完整的 H 型换挡槽背景纹理（真实机械换挡板风格）：
          拉丝金属面板 → 双层金属边框 → 雕刻沟槽 → 内凹槽位铭牌 →
          雕刻银漆档位字符（N 金属圈 / R 红漆）→ 四角螺丝
        一次烘焙，运行时不修改。
        """
        from PIL import Image, ImageDraw

        ss = float(self.SS)
        pw = float(self.PANEL_W)
        ph = float(self.PANEL_H)
        W = int(round(pw * ss))
        H = int(round(ph * ss))

        # ---- 1) 面板底：垂直渐变 + 拉丝金属纹理 + 暗角（真实换挡板质感） ----
        rng = np.random.default_rng(20260921)
        grad = (np.linspace(0.0, 1.0, H, dtype=f32)[:, None, None]
                * np.ones((1, W, 1), dtype=f32))
        top_c = np.array([0.098, 0.105, 0.118], dtype=f32)   # 顶部略亮（受光）
        bot_c = np.array([0.050, 0.053, 0.061], dtype=f32)
        rgb = bot_c[None, None, :] * grad + top_c[None, None, :] * (1.0 - grad)
        # 拉丝：沿水平方向拉伸的细噪声
        nrow, ncol = max(2, H // 10), max(2, W // 28)
        streaks = rng.normal(0.0, 1.0, (nrow, ncol)).astype(f32)
        smin = float(streaks.min())
        streaks = (streaks - smin) / max(1e-6, float(streaks.max()) - smin)
        streaks = np.asarray(
            Image.fromarray((streaks * 255).astype(np.uint8)).resize(
                (W, H), Image.BILINEAR),
            dtype=f32) / 255.0 - 0.5
        rgb = rgb * (1.0 + streaks[..., None] * 0.22)
        # 暗角：边缘压暗制造冲压件曲面纵深
        yy, xx = np.mgrid[0:H, 0:W].astype(f32)
        vg = np.sqrt(((xx - W * 0.5) / (W * 0.68)) ** 2
                     + ((yy - H * 0.55) / (H * 0.82)) ** 2)
        rgb = rgb * (1.0 - 0.30 * np.clip(vg, 0.0, 1.0) ** 2)[..., None]
        img = Image.fromarray(
            (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "RGB"
        ).convert("RGBA")
        # 圆角外透明
        pm = Image.new("L", (W, H), 0)
        ImageDraw.Draw(pm).rounded_rectangle(
            [0, 0, W - 1, H - 1], radius=self.CORNER_R * ss, fill=255)
        img.putalpha(pm)
        d = ImageDraw.Draw(img)

        # ---- 2) 金属边框：外亮圈 + 内暗缝 + 顶部受光高光 ----
        d.rounded_rectangle([0, 0, W - 1, H - 1], radius=self.CORNER_R * ss,
                            outline=(96, 102, 112, 255),
                            width=max(1, int(1.2 * ss)))
        inset = max(1, int(2.0 * ss))
        d.rounded_rectangle([inset, inset, W - 1 - inset, H - 1 - inset],
                            radius=int(self.CORNER_R * 0.75) * ss,
                            outline=(10, 11, 14, 255),
                            width=max(1, int(0.8 * ss)))
        d.line([int(self.CORNER_R * ss * 0.9), max(1, int(1.1 * ss)),
                W - int(self.CORNER_R * ss * 0.9), max(1, int(1.1 * ss))],
               fill=(205, 214, 226, 80), width=max(1, int(0.5 * ss)))

        def _px(x, y=None):
            """面板局部坐标 → 像素坐标。
            ⚠️ 局部 y 轴向上（面板底部=0），PIL 的 y 轴向下 → 必须在这里
            翻转落点：y_img = (ph - y) * ss。只翻位置、不翻字形，
            否则档位字符会上下镜像（历史 bug：1/3/5 与 2/4/6 两排颠倒）。"""
            if y is None:
                x, y = x
            return (int(round(x * ss)), int(round((ph - y) * ss)))

        def _rect(cx, cy, hw, hh):
            """中心+半宽高 → 像素 bbox（翻转后 y 自动排序）"""
            x0, y0 = _px(cx - hw, cy + hh)
            x1, y1 = _px(cx + hw, cy - hh)
            return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]

        # ---- 3) H 型导轨沟槽（雕刻进面板：深槽 + 下缘受光 + 上缘投影） ----
        gw = max(2, int(round(2.4 * ss)))
        ew = max(1, int(round(0.55 * ss)))
        g_dark = (5, 6, 8, 235)
        g_lit = (196, 204, 214, 70)
        g_shadow = (0, 0, 0, 120)
        eoff = gw // 2 + max(1, int(0.7 * ss))

        def _groove_h(x0, x1, y):
            ax, ay = _px(x0, y)
            bx, _ = _px(x1, y)
            d.line([ax, ay, bx, ay], fill=g_dark, width=gw)
            d.line([ax, ay + eoff, bx, ay + eoff], fill=g_lit, width=ew)
            d.line([ax, ay - eoff, bx, ay - eoff], fill=g_shadow, width=ew)

        def _groove_v(x, y0, y1):
            ax, ay = _px(x, y0)
            _, by = _px(x, y1)
            d.line([ax, ay, ax, by], fill=g_dark, width=gw)
            d.line([ax + eoff, ay, ax + eoff, by], fill=g_lit, width=ew)
            d.line([ax - eoff, ay, ax - eoff, by], fill=g_shadow, width=ew)

        # 横向导轨：上排 1-3-5 / 下排 2-4-6（下排直通 R）
        y_top = self._ROW_TOP
        y_bot = self._ROW_BOT
        _groove_h(self._COL_X[0] - self.SLOT_W * 0.35,
                  self._COL_X[2] + self.SLOT_W * 0.35, y_top)
        _groove_h(self._COL_X[0] - self.SLOT_W * 0.35,
                  self._R_POS[0] - self.SLOT_W * 0.35, y_bot)
        # 三条纵向连接轨（1/3/5 ↔ N ↔ 2/4/6）
        for cx in self._COL_X:
            _groove_v(cx, y_top - self.SLOT_H * 0.35, y_bot + self.SLOT_H * 0.35)

        # ---- 4) 槽位铭牌：内凹金属片（上缘阴影 / 下缘受光） ----
        hsw = self.SLOT_W * 0.5
        hsh = self.SLOT_H * 0.5
        sr = self.SLOT_R * ss
        for gear, (cx, cy) in self._ANCHORS.items():
            if gear == 'N':
                continue                      # N 用金属圈，单独画
            bx0, by0, bx1, by1 = _rect(cx, cy, hsw, hsh)
            d.rounded_rectangle([bx0, by0, bx1, by1], radius=sr,
                                fill=(27, 30, 36, 255))
            # 内凹斜面：上内缘阴影 / 下内缘受光
            d.line([bx0 + sr, by0 + ew, bx1 - sr, by0 + ew],
                   fill=(0, 0, 0, 150), width=max(1, int(0.6 * ss)))
            d.line([bx0 + sr, by1 - ew, bx1 - sr, by1 - ew],
                   fill=(185, 193, 204, 90), width=max(1, int(0.6 * ss)))
            d.rounded_rectangle([bx0, by0, bx1, by1], radius=sr,
                                outline=(56, 61, 69, 230),
                                width=max(1, int(0.45 * ss)))

        # ---- 5) N 位金属圈（真实 H 槽的中空回位位） ----
        ncx, ncy = self._ANCHORS['N']
        nr = self.SLOT_W * 0.46
        d.ellipse(_rect(ncx, ncy, nr + 1.2, nr + 1.2),
                  outline=(0, 0, 0, 140), width=max(1, int(0.7 * ss)))
        d.ellipse(_rect(ncx, ncy, nr, nr),
                  outline=(168, 176, 187, 210), width=max(1, int(0.7 * ss)))

        # ---- 5b) 顶部档位数码窗（LCD 凹窗，运行时叠加当前档位大字） ----
        gx0, gy0, gx1, gy1 = self._GEARWIN
        wx0, wy0 = _px(gx0, gy1)          # 局部 y 翻转：左上角
        wx1, wy1 = _px(gx1, gy0)          # 右下角
        d.rounded_rectangle([wx0, wy0, wx1, wy1], radius=4 * ss,
                            fill=(11, 13, 17, 255))
        # 内凹斜面（与槽位铭牌同语言：上内缘阴影 / 下内缘受光）
        d.line([wx0 + 4 * ss, wy0 + ew, wx1 - 4 * ss, wy0 + ew],
               fill=(0, 0, 0, 170), width=max(1, int(0.6 * ss)))
        d.line([wx0 + 4 * ss, wy1 - ew, wx1 - 4 * ss, wy1 - ew],
               fill=(185, 193, 204, 90), width=max(1, int(0.6 * ss)))
        d.rounded_rectangle([wx0, wy0, wx1, wy1], radius=4 * ss,
                            outline=(56, 61, 69, 230),
                            width=max(1, int(0.45 * ss)))
        # LCD 扫描线（极淡，每隔 3 逻辑像素一条）
        for sy in range(wy0 + max(1, int(3 * ss)), wy1 - 1, max(2, int(3 * ss))):
            d.line([wx0 + 2, sy, wx1 - 2, sy], fill=(0, 0, 0, 40), width=1)

        # ---- 6) 雕刻银漆档位字符（刻口下缘受光 / 上缘投影） ----
        def _engrave(cx, cy, s, font, col):
            bid = font.getbbox(s)
            tw, th = bid[2] - bid[0], bid[3] - bid[1]
            px, py = _px(cx, cy)
            pos = (px - tw * 0.5 - bid[0], py - th * 0.5 - bid[1])
            eo = max(1, int(round(0.7 * ss)))
            d.text((pos[0], pos[1] + eo), s, font=font,
                   fill=(212, 220, 230, 85))      # 刻口下缘受光
            d.text((pos[0], pos[1] - eo), s, font=font,
                   fill=(8, 9, 11, 160))          # 刻口上缘投影
            d.text(pos, s, font=font, fill=col)

        f_num = self._load_font(int(round(self.FONT_SIZE * ss)))
        for gear, (cx, cy) in self._ANCHORS.items():
            if gear in ('N', 'R'):
                continue
            _engrave(cx, cy, gear, f_num, (172, 179, 190, 235))
        # N / R 标签（R 用红漆，与真实挡板一致）+ 数码窗 GEAR 刻字
        f_lab = self._load_font(int(round(self.LABEL_FONT * ss)))
        _engrave(ncx, ncy, 'N', f_lab, (198, 205, 214, 235))
        _engrave(self._R_POS[0], self._R_POS[1], 'R', f_lab, (214, 106, 92, 235))
        f_cap = self._load_font(int(round(13 * ss)))
        _engrave(self._GEARWIN_CAPTION[0], self._GEARWIN_CAPTION[1], 'GEAR',
                 f_cap, (198, 205, 214, 235))

        # ---- 7) 四角螺丝（冲压件固定点） ----
        def _screw(scx, scy):
            x, y = _px(scx, scy)
            rp = 4.2 * ss
            d.ellipse([x - rp - ss * 0.8, y - rp - ss * 0.8,
                       x + rp + ss * 0.8, y + rp + ss * 0.8],
                      fill=(0, 0, 0, 90))
            steps = max(8, int(rp))
            for i in range(steps, 0, -1):
                t = i / rp
                c = int(58 + 132 * max(0.0, 1.0 - t) ** 1.5)
                d.ellipse([x - i, y - i, x + i, y + i],
                          fill=(c, c + 3, c + 8, 255))
            ang = math.radians(25)
            ca, sa = math.cos(ang) * rp * 0.66, math.sin(ang) * rp * 0.66
            lw2 = max(1, int(0.8 * ss))
            d.line([x - ca, y - sa, x + ca, y + sa],
                   fill=(13, 14, 17, 255), width=lw2)
            d.line([x + sa, y - ca, x - sa, y + ca],
                   fill=(13, 14, 17, 255), width=lw2)

        for scx, scy in ((13, 13), (pw - 13, 13), (13, ph - 13), (pw - 13, ph - 13)):
            _screw(scx, scy)

        # ---- 降采样上传 ----
        tex_w = int(round(pw * self.TEX_SCALE))
        tex_h = int(round(ph * self.TEX_SCALE))
        img = img.resize((tex_w, tex_h), Image.LANCZOS)
        return _upload_rgba_texture(img), pw, ph

    def _build_highlight_texture(self):
        """选中槽位特效：琥珀 LED 背光（内透晕光）+ 细亮描边（仪表指示灯质感）"""
        from PIL import Image, ImageDraw, ImageFilter

        ss = float(self.SS * 2)
        hsw = self.SLOT_W * 0.5 + 4      # 略大于槽位
        hsh = self.SLOT_H * 0.5 + 4
        w = int(round(hsw * 2 * ss))
        h = int(round(hsh * 2 * ss))

        # 背光：圆角矩形高斯晕（LED 从档位铭牌底下透光）
        glow = Image.new("L", (w, h), 0)
        ImageDraw.Draw(glow).rounded_rectangle(
            [int(3.5 * ss), int(3.5 * ss), w - int(3.5 * ss), h - int(3.5 * ss)],
            radius=self.SLOT_R * ss, fill=150)
        glow = glow.filter(ImageFilter.GaussianBlur(ss * 1.6))
        ga = np.asarray(glow, dtype=f32)[..., None] / 255.0
        amber = np.array([255.0, 178.0, 64.0], dtype=f32)
        arr = np.zeros((h, w, 4), dtype=f32)
        arr[..., :3] = np.clip(amber[None, None, :] * (0.55 + 0.45 * ga), 0, 255)
        arr[..., 3] = np.clip(ga[..., 0] * 150.0, 0.0, 150.0)
        img = Image.fromarray(arr.astype(np.uint8), "RGBA")

        # 细描边框
        ImageDraw.Draw(img).rounded_rectangle(
            [int(2.0 * ss), int(2.0 * ss), w - int(2.0 * ss), h - int(2.0 * ss)],
            radius=self.SLOT_R * ss, outline=(255, 200, 96, 240),
            width=max(2, int(ss * 0.5)))

        img = img.resize((int(round(hsw * 2 * self.TEX_SCALE)),
                          int(round(hsh * 2 * self.TEX_SCALE))), Image.LANCZOS)
        return _upload_rgba_texture(img)

    def _make_glyph_texture(self, ch):
        """档位字形纹理：琥珀色 + 辉光（与速度表数码窗同风格）"""
        from PIL import Image, ImageDraw, ImageFilter

        ss = float(self.SS)
        n = int(round(self.PANEL_H * 0.45 * ss))
        font = self._load_font(int(round(self.FONT_SIZE * ss)))
        lit = Image.new("L", (n, n), 0)
        bx0, by0, bx1, by1 = font.getbbox(ch)
        pos = (n * 0.5 - (bx1 - bx0) * 0.5 - bx0,
               n * 0.5 - (by1 - by0) * 0.5 - by0)
        ImageDraw.Draw(lit).text(pos, ch, font=font, fill=255)
        glow = lit.filter(ImageFilter.GaussianBlur(self.FONT_SIZE * ss * 0.10))
        li = np.asarray(lit, dtype=f32)[..., None] / 255.0
        gi = np.asarray(glow, dtype=f32)[..., None] / 255.0
        dig = np.array(self.COL_DIGIT, dtype=f32)[None, None, :]
        glw = np.array(self.COL_GLOW, dtype=f32)[None, None, :]
        rgb = np.clip(dig * li + glw * gi * 0.42, 0.0, 255.0)
        alpha = np.clip(np.maximum(li[..., 0], gi[..., 0] * 0.85) * 255.0, 0.0, 255.0)
        out = np.concatenate([rgb, alpha[..., None]], axis=2)
        img = Image.fromarray(out.astype(np.uint8), "RGBA")
        side = int(round(n * self.TEX_SCALE / ss))
        return _upload_rgba_texture(img)

    def _ensure_glyph(self, gear):
        """确保档位字形纹理已缓存"""
        if gear not in self._glyph_tex:
            self._glyph_tex[gear] = self._make_glyph_texture(gear)

    # -------------------- 换挡逻辑（保持原有齿比映射）--------------------

    def update(self, gear, dt=1.0/60.0):
        """接收物理引擎档位（phys.engine.gear，含迟滞/倒挡/起步油门判定），
        档位变化时触发 H 型换挡动画。"""
        self._dt = dt  # 保存 dt 供动画使用

        gear = str(gear)
        if gear not in self._ANCHORS:
            gear = 'N'

        if gear != self._gear:
            old_gear = self._gear
            self._gear = gear
            self._start_shift_animation(old_gear, gear)
            self._ensure_glyph(gear)

    # -------------------- H 型换挡动画 --------------------

    def _gate_path(self, from_gear, to_gear):
        """沿真实 H 型导轨规划路径（直角折线，绝不斜穿面板）：
        - 同列（1↔2 / 3↔N↔4 / 5↔6）：沿纵轨直上直下，不绕 N；
        - 跨列：本列纵轨 → 中央横轨 → 目标列纵轨（真实手掌动作）；
        - R 只与 6 直连（同排横向），其余档位进出 R 一律经 6 中转。"""
        pa = self._ANCHORS[from_gear]
        pb = self._ANCHORS[to_gear]

        if from_gear == 'R' or to_gear == 'R':
            if {from_gear, to_gear} == {'R', '6'}:
                return [pa, pb]
            if from_gear == 'R':
                return [pa] + self._gate_path('6', to_gear)
            return self._gate_path(from_gear, '6') + [pb]

        ax, ay = pa
        bx, by = pb
        if ax == bx:                       # 同列纵轨
            return [pa, pb]
        ny = self._N_Y                     # 跨列：走中央横轨
        pts = [(ax, ny), (bx, ny)]
        if by != ny:
            pts.append((bx, by))
        return [pa] + pts

    def _start_shift_animation(self, from_gear, to_gear):
        """计算 H 型换挡路径，启动动画"""
        if from_gear not in self._ANCHORS or to_gear not in self._ANCHORS:
            self._display_gear = to_gear
            self._anim_progress = 1.0
            return

        # 去除重复点（N 本身就在中央横轨与中列纵轨交点上）
        raw = self._gate_path(from_gear, to_gear)
        self._anim_path = [raw[0]]
        for p in raw[1:]:
            if p != self._anim_path[-1]:
                self._anim_path.append(p)

        # 计算路径总长
        self._anim_path_len = 0.0
        for i in range(1, len(self._anim_path)):
            dx = self._anim_path[i][0] - self._anim_path[i - 1][0]
            dy = self._anim_path[i][1] - self._anim_path[i - 1][1]
            self._anim_path_len += math.sqrt(dx * dx + dy * dy)

        self._anim_progress = 0.0
        self._settle_t = 0.0

    def _get_animated_pos(self, dt):
        """
        逐帧推进换挡动画，返回当前高亮框的 (x, y, 进度归一化)
        """
        if self._anim_progress >= 1.0:
            return (self._ANCHORS[self._gear][0],
                    self._ANCHORS[self._gear][1], 1.0)

        # 沿路径匀速移动
        dist = self.ANIM_MOVE_SPEED * dt
        progress_delta = dist / max(self._anim_path_len, 1.0)
        self._anim_progress += progress_delta

        t = self._anim_progress
        path = self._anim_path

        if t >= 1.0:
            # 进入弹性归位阶段
            self._settle_t += dt
            settle_progress = self._settle_t / self.ANIM_SETTLE_TIME
            if settle_progress > 1.0:
                self._anim_progress = 1.0
                self._display_gear = self._gear
                return (path[-1][0], path[-1][1], 1.0)
            # 指数衰减振荡
            overshoot = (math.exp(-settle_progress * self.ANIM_SETTLE_DECAY)
                         * math.sin(settle_progress * math.pi * 6.0)
                         * self.ANIM_SETTLE_AMP)
            return (path[-1][0] + overshoot, path[-1][1], t)

        # 计算当前沿路径的位置
        total_t = t * self._anim_path_len
        acc = 0.0
        for i in range(1, len(path)):
            seg_len = math.sqrt((path[i][0] - path[i - 1][0]) ** 2
                                + (path[i][1] - path[i - 1][1]) ** 2)
            if acc + seg_len >= total_t:
                seg_t = (total_t - acc) / max(seg_len, 1e-6)
                x = path[i - 1][0] + (path[i][0] - path[i - 1][0]) * seg_t
                y = path[i - 1][1] + (path[i][1] - path[i - 1][1]) * seg_t
                return (x, y, t)
            acc += seg_len

        return (path[-1][0], path[-1][1], t)

    # -------------------- 渲染 --------------------

    def render(self):
        """渲染 H 型换挡槽：背景 → 高亮框 → 档位字符（选中最亮，其余暗显）"""
        glUseProgram(self._text._prog)
        glUniformMatrix4fv(self._text._uMVP, 1, GL_TRUE, self._text._proj)
        glActiveTexture(GL_TEXTURE0)
        glUniform1i(self._text._uTex, 0)
        glUniform4f(self._text._uColor, 1.0, 1.0, 1.0, 1.0)
        glBindVertexArray(self._quad_vao)

        # 面板左下角（屏幕坐标）
        px, py = float(self.PAD), float(self.PAD)

        # ---- 1) 背景纹理 ----
        self._draw_quad(self._bg_tex, px, py, self._bg_w, self._bg_h)

        # ---- 2) 高亮特效（当前档位，含动画插值） ----
        hx, hy, progress = self._get_animated_pos(self._dt)
        # 纹理逻辑尺寸 = 槽位 + 2*4 内边距（与 _build_highlight_texture 一致）
        hl_w = self.SLOT_W + 8
        hl_h = self.SLOT_H + 8
        # 面板局部 → 屏幕坐标
        screen_cx = px + hx
        screen_cy = py + hy

        # 脉冲闪烁：高转时闪烁加速
        if self._gear not in ('N', 'R'):
            try:
                gr = int(self._gear) if self._gear in "123456" else 1
            except ValueError:
                gr = 1
        else:
            gr = 0
        blink_k = 1.0 + 0.15 * math.sin(self._flash_phase * self.HL_BLINK_SPEED)

        glUniform4f(self._text._uColor, blink_k, blink_k, blink_k, 1.0)
        self._draw_quad(self._hl_tex,
                        screen_cx - hl_w * 0.5, screen_cy - hl_h * 0.5,
                        hl_w, hl_h)

        # ---- 3) 当前档位字符（亮琥珀色，叠在高亮框上） ----
        glUniform4f(self._text._uColor, 1.0, 1.0, 1.0, 1.0)
        gt = self._glyph_tex.get(self._gear)
        if gt is not None:
            # 字形纹理为正方形（字形居中），按槽位高度约 1.35 倍绘制
            gw = gh = self.SLOT_H * 1.35
            self._draw_quad(gt,
                            screen_cx - gw * 0.5, screen_cy - gh * 0.5,
                            gw, gh)

        # ---- 4) 顶部数码窗：当前档位大字（琥珀辉光，随换挡即时切换） ----
        if gt is not None:
            gwx, gwy = self._GEARWIN_GLYPH
            gw2 = gh2 = 38
            self._draw_quad(gt,
                            px + gwx - gw2 * 0.5, py + gwy - gh2 * 0.5,
                            gw2, gh2)

        glBindVertexArray(0)
        glUseProgram(0)

        # 更新脉冲相位
        self._flash_phase += self._dt

    def _draw_quad(self, tex, x0, y0, w, h):
        """以左下角 (x0, y0) 绘制整张纹理"""
        verts = np.array([
            x0,     y0,     0.0, 1.0,
            x0 + w, y0,     1.0, 1.0,
            x0 + w, y0 + h, 1.0, 0.0,
            x0,     y0,     0.0, 1.0,
            x0 + w, y0 + h, 1.0, 0.0,
            x0,     y0 + h, 0.0, 0.0,
        ], dtype=f32)
        glBindTexture(GL_TEXTURE_2D, tex)
        glBindBuffer(GL_ARRAY_BUFFER, self._quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_DYNAMIC_DRAW)
        glDrawArrays(GL_TRIANGLES, 0, 6)

    # -------------------- 适配与释放 --------------------

    def resize(self, new_logical_w, new_logical_h,
               new_drawable_w, new_drawable_h):
        self._ui = UICoordSystem(new_logical_w, new_logical_h,
                                 new_drawable_w, new_drawable_h)
        self._proj = _mat4_ortho(0, self._ui.logical_w,
                                 0, self._ui.logical_h, -1, 1)

    def destroy(self):
        glDeleteTextures(1, [self._bg_tex, self._hl_tex])
        for tex in self._glyph_tex.values():
            glDeleteTextures(1, [tex])
        self._glyph_tex.clear()
        glDeleteVertexArrays(1, [self._quad_vao])
        glDeleteBuffers(1, [self._quad_vbo])


# ============================================================
# Button —— 通用按钮组件
# ============================================================
class Button:
    def __init__(self, x, y, w, h, text, font_size=24,
                 bg_color=(0.15, 0.30, 0.35, 0.70),
                 hover_color=(0.20, 0.40, 0.45, 0.85),
                 text_color=(255, 248, 240, 230),
                 text_hover_color=(255, 248, 240, 255)):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
        self.font_size = font_size
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.text_hover_color = text_hover_color
        self._hovered = False
        self._pressed = False
        stride = 6 * 4
        verts = self._make_verts(bg_color)
        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)
        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

    def _make_verts(self, color):
        x, y, w, h = self.x, self.y, self.w, self.h
        return np.array([
            [x, y, *color], [x+w, y, *color], [x+w, y+h, *color],
            [x, y, *color], [x+w, y+h, *color], [x, y+h, *color],
        ], dtype=f32)

    def contains(self, px, py):
        return (self.x <= px <= self.x + self.w and
                self.y <= py <= self.y + self.h)

    def set_hover(self, hovered):
        if hovered != self._hovered:
            self._hovered = hovered
            color = self.hover_color if hovered else self.bg_color
            verts = self._make_verts(color)
            glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
            glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_DYNAMIC_DRAW)

    def render(self, hud_prog, hud_uMVP, proj, text_renderer):
        glUseProgram(hud_prog)
        glUniformMatrix4fv(hud_uMVP, 1, GL_TRUE, proj)
        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)
        tx = self.x + self.w // 2
        ty = self.y + (self.h - self.font_size) // 2 + 2
        color = self.text_hover_color if self._hovered else self.text_color
        text_renderer.render(self.text, tx, ty, font_size=self.font_size,
                             color=color, align="center")

    def destroy(self):
        glDeleteVertexArrays(1, [self._vao])
        glDeleteBuffers(1, [self._vbo])


# ============================================================
# MainMenu —— 主菜单界面
# ============================================================
class MainMenu:
    """主菜单：开始游戏、设置、退出"""
    TITLE_COLOR = (255, 248, 240, 255)
    SUBTITLE_COLOR = (163, 200, 224, 200)
    BG_OVERLAY = (0.05, 0.18, 0.18, 0.55)

    def __init__(self, ui_coord, text_renderer):
        self._ui = ui_coord
        self._text = text_renderer
        self._hud_prog, self._hud_uMVP = get_hud_program()
        self._proj = _mat4_ortho(0, self._ui.logical_w, 0, self._ui.logical_h, -1, 1)

        # 背景
        self._init_background()

        # 三个按钮
        cx = self._ui.logical_w // 2
        btn_w, btn_h, gap = 260, 52, 20
        btn_x = cx - btn_w // 2
        bh = 3 * btn_h + 2 * gap
        sy = (self._ui.logical_h - bh) // 2

        self.btn_start = Button(
            btn_x, sy + 2*(btn_h+gap), btn_w, btn_h, "开 始 游 戏",
            font_size=22, bg_color=(0.15, 0.35, 0.35, 0.75),
            hover_color=(0.83, 0.66, 0.33, 0.85))
        self.btn_settings = Button(
            btn_x, sy + btn_h + gap, btn_w, btn_h, "设    置",
            font_size=22, bg_color=(0.15, 0.30, 0.35, 0.65),
            hover_color=(0.20, 0.40, 0.45, 0.85))
        self.btn_quit = Button(
            btn_x, sy, btn_w, btn_h, "退    出",
            font_size=22, bg_color=(0.15, 0.25, 0.30, 0.65),
            hover_color=(0.55, 0.20, 0.20, 0.80))

        self.buttons = [self.btn_start, self.btn_settings, self.btn_quit]
        self._active = True

    def _init_background(self):
        w, h = self._ui.logical_w, self._ui.logical_h
        col = self.BG_OVERLAY
        verts = np.array([
            [0, 0, *col], [w, 0, *col], [w, h, *col],
            [0, 0, *col], [w, h, *col], [0, h, *col],
        ], dtype=f32)
        stride = 6 * 4
        self._bg_vao = glGenVertexArrays(1)
        self._bg_vbo = glGenBuffers(1)
        glBindVertexArray(self._bg_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._bg_vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

    def set_active(self, active):
        self._active = active

    @property
    def is_active(self):
        return self._active

    def on_mouse_move(self, mx, my):
        # 翻转 Y：SDL (y=0 顶) → 逻辑坐标 (y=0 底)
        my = self._ui.logical_h - my
        for btn in self.buttons:
            btn.set_hover(btn.contains(mx, my))

    def on_mouse_click(self, mx, my):
        # 翻转 Y：SDL (y=0 顶) → 逻辑坐标 (y=0 底)
        my = self._ui.logical_h - my
        for i, btn in enumerate(self.buttons):
            if btn.contains(mx, my):
                return i
        return -1

    def render(self):
        if not self._active:
            return
        glViewport(0, 0, self._ui.drawable_w, self._ui.drawable_h)
        glUseProgram(self._hud_prog)
        glUniformMatrix4fv(self._hud_uMVP, 1, GL_TRUE, self._proj)
        glBindVertexArray(self._bg_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)
        glUseProgram(0)

        cx = self._ui.logical_w // 2
        title_y = self._ui.logical_h - int(self._ui.logical_h * 0.28)
        self._text.render("动力滑行", cx, title_y,
                          font_size=52, color=self.TITLE_COLOR, align="center")
        self._text.render("— 榛名山山道 —", cx, title_y - 42,
                          font_size=20, color=self.SUBTITLE_COLOR, align="center")
        self._text.render("─" * 20, cx, title_y - 70,
                          font_size=12, color=(163, 200, 224, 100), align="center")
        for btn in self.buttons:
            btn.render(self._hud_prog, self._hud_uMVP, self._proj, self._text)

    def resize(self, nw, nh, dw, dh):
        self._ui = UICoordSystem(nw, nh, dw, dh)
        self._proj = _mat4_ortho(0, self._ui.logical_w, 0, self._ui.logical_h, -1, 1)
        self._init_background()
        cx = self._ui.logical_w // 2
        btn_w, btn_h, gap = 260, 52, 20
        btn_x = cx - btn_w // 2
        bh = 3 * btn_h + 2 * gap
        sy = (self._ui.logical_h - bh) // 2
        for btn, ny in zip(self.buttons,
                           [sy + 2*(btn_h+gap), sy + btn_h + gap, sy]):
            btn.x = btn_x
            btn.y = ny
            c = btn.hover_color if btn._hovered else btn.bg_color
            verts_data = btn._make_verts(c)
            glBindBuffer(GL_ARRAY_BUFFER, btn._vbo)
            glBufferData(GL_ARRAY_BUFFER, verts_data.nbytes,
                         verts_data, GL_DYNAMIC_DRAW)

    def destroy(self):
        glDeleteVertexArrays(1, [self._bg_vao])
        glDeleteBuffers(1, [self._bg_vbo])
        for btn in self.buttons:
            btn.destroy()


# ============================================================
# SettingsPanel —— 设置面板
# ============================================================
class SettingsPanel:
    """设置面板：全屏、画质、垂直同步等选项"""
    TITLE_COLOR = (255, 248, 240, 255)
    LABEL_COLOR = (200, 220, 230, 220)
    VALUE_COLOR = (255, 220, 120, 255)
    BG_OVERLAY = (0.05, 0.18, 0.18, 0.60)
    PANEL_BG = (0.05, 0.12, 0.12, 0.75)

    def __init__(self, ui_coord, text_renderer):
        self._ui = ui_coord
        self._text = text_renderer
        self._hud_prog, self._hud_uMVP = get_hud_program()
        self._proj = _mat4_ortho(0, self._ui.logical_w, 0, self._ui.logical_h, -1, 1)

        self._items = [
            {"label": "全屏模式",   "key": "fullscreen",  "type": "toggle", "value": False},
            {"label": "画面质量",   "key": "quality",     "type": "choice",
             "options": ["低", "中", "高", "极高"], "value": 2},
            {"label": "垂直同步",   "key": "vsync",       "type": "toggle", "value": True},
            {"label": "显示小地图", "key": "show_minimap","type": "toggle", "value": True},
            {"label": "显示速度表", "key": "show_speedo", "type": "toggle", "value": True},
            {"label": "镜头抖动",   "key": "cam_shake",   "type": "toggle", "value": True},
        ]
        self._active = False
        self._init_panel()

        pw = int(self._ui.logical_w * 0.50)
        ph = int(self._ui.logical_h * 0.60)
        px = (self._ui.logical_w - pw) // 2
        py = (self._ui.logical_h - ph) // 2
        self.btn_back = Button(
            px + (pw - 160)//2, py + ph - 44 - 30, 160, 44,
            "返    回", font_size=20,
            bg_color=(0.15, 0.30, 0.35, 0.70),
            hover_color=(0.83, 0.66, 0.33, 0.85))

    def _init_panel(self):
        w, h = self._ui.logical_w, self._ui.logical_h
        pw = int(w * 0.50)
        ph = int(h * 0.60)
        px = (w - pw) // 2
        py = (h - ph) // 2
        verts = np.concatenate([
            np.array([[0,0,*self.BG_OVERLAY],[w,0,*self.BG_OVERLAY],
                      [w,h,*self.BG_OVERLAY],[0,0,*self.BG_OVERLAY],
                      [w,h,*self.BG_OVERLAY],[0,h,*self.BG_OVERLAY]], dtype=f32),
            np.array([[px,py,*self.PANEL_BG],[px+pw,py,*self.PANEL_BG],
                      [px+pw,py+ph,*self.PANEL_BG],[px,py,*self.PANEL_BG],
                      [px+pw,py+ph,*self.PANEL_BG],[px,py+ph,*self.PANEL_BG]], dtype=f32),
        ])
        stride = 6 * 4
        self._bg_vao = glGenVertexArrays(1)
        self._bg_vbo = glGenBuffers(1)
        glBindVertexArray(self._bg_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._bg_vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)
        self._bg_vert_count = 12
        self._panel_rect = (px, py, pw, ph)

    def set_active(self, a):
        self._active = a

    @property
    def is_active(self):
        return self._active

    def on_mouse_move(self, mx, my):
        # 翻转 Y：SDL (y=0 顶) → 逻辑坐标 (y=0 底)
        my = self._ui.logical_h - my
        self.btn_back.set_hover(self.btn_back.contains(mx, my))

    def on_mouse_click(self, mx, my):
        # 翻转 Y：SDL (y=0 顶) → 逻辑坐标 (y=0 底)
        my = self._ui.logical_h - my
        if self.btn_back.contains(mx, my):
            return "back"
        px, py, pw, ph = self._panel_rect
        iy = py + 70
        for item in self._items:
            if px+30 <= mx <= px+pw-30 and iy <= my <= iy+42:
                if item["type"] == "toggle":
                    item["value"] = not item["value"]
                elif item["type"] == "choice":
                    item["value"] = (item["value"] + 1) % len(item["options"])
                return "changed"
            iy += 42
        return None

    def render(self):
        if not self._active:
            return
        glViewport(0, 0, self._ui.drawable_w, self._ui.drawable_h)
        glUseProgram(self._hud_prog)
        glUniformMatrix4fv(self._hud_uMVP, 1, GL_TRUE, self._proj)
        glBindVertexArray(self._bg_vao)
        glDrawArrays(GL_TRIANGLES, 0, self._bg_vert_count)
        glBindVertexArray(0)
        glUseProgram(0)

        w, h = self._ui.logical_w, self._ui.logical_h
        pw = int(w * 0.50); ph = int(h * 0.60)
        px = (w - pw)//2; py = (h - ph)//2
        self._text.render("游 戏 设 置", px+pw//2, py+20,
                          font_size=28, color=self.TITLE_COLOR, align="center")
        self._text.render("─" * 30, px+pw//2, py+54,
                          font_size=10, color=(163,200,224,100), align="center")

        iy = py + 70
        for item in self._items:
            self._text.render(item["label"], px+40, iy+6,
                              font_size=18, color=self.LABEL_COLOR, align="left")
            if item["type"] == "toggle":
                vt = "ON" if item["value"] else "OFF"
                vc = (100, 220, 100, 255) if item["value"] else (180, 100, 100, 200)
            else:
                vt = item["options"][item["value"]]
                vc = self.VALUE_COLOR
            self._text.render(vt, px+pw-50, iy+6,
                              font_size=18, color=vc, align="right")
            iy += 42

        self.btn_back.render(self._hud_prog, self._hud_uMVP, self._proj, self._text)

    def resize(self, nw, nh, dw, dh):
        self._ui = UICoordSystem(nw, nh, dw, dh)
        self._proj = _mat4_ortho(0, self._ui.logical_w, 0, self._ui.logical_h, -1, 1)
        self._init_panel()
        pw = int(self._ui.logical_w * 0.50)
        ph = int(self._ui.logical_h * 0.60)
        px = (self._ui.logical_w - pw)//2
        py = (self._ui.logical_h - ph)//2
        self.btn_back.x = px + (pw - 160)//2
        self.btn_back.y = py + ph - 44 - 30
        c = self.btn_back.hover_color if self.btn_back._hovered else self.btn_back.bg_color
        verts_data = self.btn_back._make_verts(c)
        glBindBuffer(GL_ARRAY_BUFFER, self.btn_back._vbo)
        glBufferData(GL_ARRAY_BUFFER, verts_data.nbytes,
                     verts_data, GL_DYNAMIC_DRAW)

    def destroy(self):
        glDeleteVertexArrays(1, [self._bg_vao])
        glDeleteBuffers(1, [self._bg_vbo])
        self.btn_back.destroy()


# ============================================================
# 模块级清理：释放共享 HUD 着色器程序
# ============================================================
def destroy_shared_hud():
    """释放共享 HUD 着色器程序（在游戏退出时调用一次）"""
    if _shared_hud["prog"] is not None:
        glDeleteProgram(_shared_hud["prog"])
        _shared_hud["prog"] = None
        _shared_hud["uMVP"] = None


# ============================================================
# CornerWarning —— 弯道路标提示（日本山路弯道指示牌风格）
# ============================================================
# 屏幕顶部居中弹出路标，模仿日本山路上真实的弯道指示牌：
#   - 白底圆角矩形牌 + 颜色编码边框 + 黑色粗箭头
#   - 发夹弯=红框 | 急弯=橙框 | 中速弯=黄框 | 高速弯=绿框
#   - S 弯 = 红框双箭头
#   距弯道 <30m 闪烁，入弯后淡出
# ============================================================

# 弯道类型 → 路标参数（size=外框尺寸, border=边框颜色）
_CW_SIGN = {
    'hairpin': {'size': 110, 'border': (0.87, 0.13, 0.13)},  # 红
    'sharp':   {'size': 95,  'border': (0.87, 0.40, 0.20)},   # 橙
    'medium':  {'size': 80,  'border': (0.87, 0.67, 0.20)},   # 黄
    'gentle':  {'size': 65,  'border': (0.27, 0.73, 0.27)},   # 绿
    's':       {'size': 75,  'border': (0.87, 0.27, 0.27)},   # 红
}


def _bake_sign_tex(sign_size, border_rgb, direction, ss=4):
    """
    离屏绘制单个弯道路标纹理。

    视觉风格：白底圆角矩形 + 颜色编码边框 + 底部路标杆 + 黑色弯道箭头。
    模仿日本山道常见的弯道指示牌。

    参数:
        sign_size: 路标外框的逻辑尺寸（像素，正方形）
        border_rgb: 边框颜色 RGB (0~1)
        direction: 'right'（↗）或 'left'（↖）
        ss: 超采样倍率

    返回: OpenGL 纹理 ID
    """
    from PIL import Image, ImageDraw

    S = int(round(sign_size * ss))
    pad = int(round(6 * ss))                    # 内边距
    bw = max(1, int(round(3 * ss)))             # 边框宽度
    r = int(round(5 * ss))                      # 圆角半径

    # 颜色
    bg_c = (248, 244, 235, 255)                 # 米白底色（旧金属牌质感）
    bdr_c = tuple(int(v * 255) for v in border_rgb[:3]) + (255,)
    arr_c = (35, 35, 35, 255)                   # 深灰箭头
    post_c = (120, 120, 120, 255)               # 灰色路标杆

    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # ---- 1) 外框（圆角矩形 + 颜色边框） ----
    draw.rounded_rectangle([0, 0, S - 1, S - 1], radius=r,
                           fill=bg_c, outline=bdr_c, width=bw)

    # ---- 2) 内部箭头（黑色等边三角形，带 6px 内边距） ----
    inner = S - 2 * pad                         # 箭头绘制区尺寸
    if direction == 'right':
        # ↗：尖端右上
        poly = [
            (pad + inner * 0.88, pad + inner * 0.12),
            (pad + inner * 0.38, pad + inner * 0.92),
            (pad + inner * 0.04, pad + inner * 0.55),
        ]
    else:
        # ↖：尖端左上
        poly = [
            (pad + inner * 0.12, pad + inner * 0.12),
            (pad + inner * 0.96, pad + inner * 0.55),
            (pad + inner * 0.62, pad + inner * 0.92),
        ]
    draw.polygon(poly, fill=arr_c)

    # ---- 3) 底部路标杆（小梯形，模拟插在地上的杆子） ----
    post_w = max(2, int(round(6 * ss)))
    post_h = max(3, int(round(6 * ss)))
    cx = S // 2
    draw.polygon([
        (cx - post_w, S - post_h),
        (cx + post_w, S - post_h),
        (cx + post_w // 2, S),
        (cx - post_w // 2, S),
    ], fill=post_c)

    # 降采样到目标尺寸
    target = int(round(sign_size * 2))
    img = img.resize((target, target), Image.LANCZOS)
    return _upload_rgba_texture(img)


def _bake_s_sign_tex(sign_size, border_rgb, ss=4):
    """
    离屏绘制 S 弯路标（↖↗ 双箭头并排）。

    参数:
        sign_size: 单个箭头的逻辑尺寸（像素）
        border_rgb: 边框颜色 RGB
        ss: 超采样倍率

    返回: OpenGL 纹理 ID
    """
    from PIL import Image, ImageDraw

    half = int(round(sign_size * ss))
    gap = max(3, int(round(5 * ss)))
    S_w = half * 2 + gap
    S_h = half

    bw = max(1, int(round(3 * ss)))
    r = int(round(5 * ss))
    pad = int(round(4 * ss))

    bg_c = (248, 244, 235, 255)
    bdr_c = tuple(int(v * 255) for v in border_rgb[:3]) + (255,)
    arr_c = (35, 35, 35, 255)
    post_c = (120, 120, 120, 255)

    img = Image.new("RGBA", (S_w, S_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 外框
    draw.rounded_rectangle([0, 0, S_w - 1, S_h - 1], radius=r,
                           fill=bg_c, outline=bdr_c, width=bw)

    # 左箭头 ↖
    inner = half - 2 * pad
    poly_left = [
        (pad + inner * 0.12, pad + inner * 0.12),
        (pad + inner * 0.96, pad + inner * 0.55),
        (pad + inner * 0.62, pad + inner * 0.92),
    ]
    draw.polygon(poly_left, fill=arr_c)

    # 右箭头 ↗
    offset_x = half + gap
    poly_right = [
        (offset_x + pad + inner * 0.88, pad + inner * 0.12),
        (offset_x + pad + inner * 0.38, pad + inner * 0.92),
        (offset_x + pad + inner * 0.04, pad + inner * 0.55),
    ]
    draw.polygon(poly_right, fill=arr_c)

    # 路标杆（居中底部）
    post_w = max(2, int(round(5 * ss)))
    post_h = max(2, int(round(5 * ss)))
    cx = S_w // 2
    draw.polygon([
        (cx - post_w, S_h - post_h),
        (cx + post_w, S_h - post_h),
        (cx + post_w // 2, S_h),
        (cx - post_w // 2, S_h),
    ], fill=post_c)

    target_w = int(round(S_w * 2.0 / ss))
    target_h = int(round(S_h * 2.0 / ss))
    img = img.resize((target_w, target_h), Image.LANCZOS)
    return _upload_rgba_texture(img)


class CornerWarning:
    """
    弯道路标提示。

    屏幕顶部居中弹出日本山路风格的弯道指示牌：
      - 白底圆角矩形 + 颜色编码边框 + 底部路标杆
      - 距弯道 200m~30m：路标弹入并保持
      - 距弯道 <30m：路标闪烁（1.5Hz）
      - 进入弯道后：缩小淡出
      - S 弯显示 ↖↗ 双箭头牌
    """

    TOP_Y = 30                  # 路标中心距屏幕顶部逻辑像素
    TRIGGER_DIST = 200.0        # 触发显示的距离（米）
    BLINK_DIST = 30.0           # 闪烁阈值（米）
    BLINK_FREQ = 1.5            # 闪烁频率（Hz）

    def __init__(self, ui_coord, text_renderer):
        self._ui = ui_coord
        self._text = text_renderer
        self._proj = _mat4_ortho(0, ui_coord.logical_w, 0, ui_coord.logical_h, -1, 1)

        # ---- 烘焙所有路标纹理 ----
        self._tex_map = {}

        for ctype, cfg in _CW_SIGN.items():
            size = cfg['size']
            border = cfg['border']
            if ctype == 's':
                self._tex_map[('s', 's')] = _bake_s_sign_tex(size, border)
            else:
                self._tex_map[(ctype, 'right')] = _bake_sign_tex(size, border, 'right')
                self._tex_map[(ctype, 'left')] = _bake_sign_tex(size, border, 'left')

        # ---- 共享 Quad VAO/VBO ----
        self._quad_vao = glGenVertexArrays(1)
        self._quad_vbo = glGenBuffers(1)
        glBindVertexArray(self._quad_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, 6 * 4 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))
        glBindVertexArray(0)

        # ---- 状态 ----
        self._active = False
        self._tex_id = None
        self._tex_w = 0
        self._tex_h = 0
        self._anim_t = 0.0
        self._anim_done = False
        self._fade_out = 0.0
        self._blink_phase = 0.0
        self._was_in_corner = False
        self._should_blink = False

    def update(self, corner_info, dt):
        """更新弯道路标状态。"""
        if not corner_info or not corner_info.get('has_upcoming'):
            if self._active and not self._was_in_corner:
                self._fade_out = min(1.0, self._fade_out + dt * 3.0)
                if self._fade_out >= 1.0:
                    self._active = False
                    self._fade_out = 0.0
                    self._anim_t = 0.0
                    self._anim_done = False
            self._was_in_corner = False
            return

        dist = corner_info['distance_m']
        direction = corner_info['direction']
        ctype = corner_info['type']

        key = (ctype, direction)
        if key in self._tex_map:
            self._tex_id = self._tex_map[key]
        else:
            self._tex_id = None
            return

        size = _CW_SIGN.get(ctype, _CW_SIGN['medium'])['size']
        if ctype == 's':
            self._tex_w = size * 2 + 10
            self._tex_h = size
        else:
            self._tex_w = size
            self._tex_h = size

        if not self._active:
            self._active = True
            self._anim_t = 0.0
            self._anim_done = False
            self._fade_out = 0.0
            self._was_in_corner = False

        if not self._anim_done:
            self._anim_t += dt
            if self._anim_t >= 0.3:
                self._anim_done = True

        self._should_blink = dist < self.BLINK_DIST
        self._blink_phase += dt

        if dist < 5.0 and self._active:
            self._was_in_corner = True
        if self._was_in_corner:
            self._fade_out = min(1.0, self._fade_out + dt * 2.5)
            if self._fade_out >= 1.0:
                self._active = False
                self._fade_out = 0.0
                self._anim_t = 0.0
                self._anim_done = False
                self._was_in_corner = False

    @property
    def _scale(self):
        if self._anim_done:
            return 1.0
        t = min(1.0, self._anim_t / 0.3)
        return 1.0 + 2.70158 * t ** 3 - 1.70158 * t ** 2

    @property
    def _alpha(self):
        if self._fade_out > 0:
            return max(0.0, 1.0 - self._fade_out)
        if self._should_blink:
            blink = math.sin(self._blink_phase * self.BLINK_FREQ * 2.0 * math.pi)
            return 0.3 + 0.7 * max(0.0, blink)
        return 1.0

    def render(self):
        """渲染路标（屏幕顶部居中）"""
        if not self._active or self._tex_id is None:
            return

        glUseProgram(self._text._prog)
        glUniformMatrix4fv(self._text._uMVP, 1, GL_TRUE, self._proj)
        glActiveTexture(GL_TEXTURE0)
        glUniform1i(self._text._uTex, 0)

        alpha = self._alpha
        scale = self._scale
        if alpha < 0.01 or scale < 0.01:
            glUseProgram(0)
            return

        cx = self._ui.logical_w * 0.5
        cy = self._ui.logical_h - self.TOP_Y
        w = self._tex_w * scale
        h = self._tex_h * scale
        x0 = cx - w * 0.5
        y0 = cy - h * 0.5

        glUniform4f(self._text._uColor, 1.0, 1.0, 1.0, alpha)
        glBindVertexArray(self._quad_vao)
        self._draw_quad(self._tex_id, x0, y0, w, h)
        glBindVertexArray(0)
        glUseProgram(0)

    def _draw_quad(self, tex, x0, y0, w, h):
        """以左下角 (x0, y0) 绘制纹理"""
        verts = np.array([
            x0,     y0,     0.0, 1.0,
            x0 + w, y0,     1.0, 1.0,
            x0 + w, y0 + h, 1.0, 0.0,
            x0,     y0,     0.0, 1.0,
            x0 + w, y0 + h, 1.0, 0.0,
            x0,     y0 + h, 0.0, 0.0,
        ], dtype=f32)
        glBindTexture(GL_TEXTURE_2D, tex)
        glBindBuffer(GL_ARRAY_BUFFER, self._quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_DYNAMIC_DRAW)
        glDrawArrays(GL_TRIANGLES, 0, 6)

    def resize(self, new_logical_w, new_logical_h, new_drawable_w, new_drawable_h):
        self._ui = UICoordSystem(new_logical_w, new_logical_h, new_drawable_w, new_drawable_h)
        self._proj = _mat4_ortho(0, self._ui.logical_w, 0, self._ui.logical_h, -1, 1)

    def destroy(self):
        """释放所有纹理和 GPU 资源"""
        for tex in set(self._tex_map.values()):
            try:
                glDeleteTextures(1, [tex])
            except Exception:
                pass
        self._tex_map.clear()
        glDeleteVertexArrays(1, [self._quad_vao])
        glDeleteBuffers(1, [self._quad_vbo])
        self._tex_id = None