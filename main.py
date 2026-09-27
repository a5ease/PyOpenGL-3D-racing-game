"""
动力滑行 — SDL2 + PyOpenGL 山路赛车游戏
=========================================
PBR Cook-Torrance 光照 + 程序化纹理 + 榛名山赛道
赛道和地形由 mountain.py 的山脊骨架驱动生成
"""

import sdl2
from OpenGL.GL import *
import numpy as np
import sys, math, random, ctypes, os, gc, json
from PIL import Image

# ============================================================
# ★ 第三阶段：PBR 贴图加载器（替换程序化纹理生成）
# ============================================================
import pbr_loader

# ============================================================
# ★ 第三阶段：外置模型加载器（当前未使用，保留以备将来扩展）
# ============================================================

# PBR 纹理目录（外部资产），优先从此目录加载真实 PBR 贴图
_PBR_TEX_DIR = pbr_loader._DEFAULT_PBR_DIR  # assets/textures

def _load_tex_from_cache_or_pbr(cache_name, pbr_layer=None, width=2048, height=2048,
                                 fallback_build_fn=None):
    """
    优先从 PBR 目录加载真实纹理；如果不存在，从 .cache PNG 文件加载；
    最后回退到程序化生成（fallback_build_fn）。

    参数：
        cache_name: .cache 目录中的 PNG 文件名（不含扩展名）
        pbr_layer:  PBR 加载的层名（如 'track', 'terrain_grass'），None 则只用缓存
        width/height: 目标纹理尺寸
        fallback_build_fn: 当 PBR 和缓存都不存在时的程序化生成回调 (w, h) -> np.ndarray

    返回 OpenGL 纹理 ID
    """
    import os as _os
    tex_dir = _os.path.join(_os.path.dirname(__file__), ".cache")
    pbr_path = _os.path.join(_os.path.dirname(__file__), "assets", "textures")

    # 1) 尝试从 PBR 目录加载真实纹理
    if pbr_layer is not None and _os.path.isdir(pbr_path):
        albedo = pbr_loader.load_pbr_albedo(pbr_path, pbr_layer, width, height)
        if albedo is not None:
            print(f"[PBR] 加载真实纹理: {pbr_layer} (albedo)")
            return albedo

    # 2) 回退到 .cache 中的 PNG 缓存
    png_path = _os.path.join(tex_dir, f"{cache_name}.png")
    if _os.path.exists(png_path):
        print(f"[缓存] 加载纹理: {cache_name}.png")
        img = np.array(Image.open(png_path), dtype=np.uint8)
        actual_h, actual_w = img.shape[:2]
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        fmt = GL_RGBA if img.shape[2] == 4 else GL_RGB
        glTexImage2D(GL_TEXTURE_2D, 0, fmt, actual_w, actual_h, 0, fmt, GL_UNSIGNED_BYTE, img)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glGenerateMipmap(GL_TEXTURE_2D)
        try:
            max_aniso = glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, min(16.0, max_aniso))
        except Exception:
            pass
        return tex

    # 3) ★ 程序化生成回退（而不是灰色占位！）
    if fallback_build_fn is not None:
        print(f"[程序生成] 无 PBR/缓存，程序化生成纹理: {cache_name}")
        return _make_texture_cached(cache_name, width, height, fallback_build_fn)

    # 4) 最后兜底：灰色占位
    print(f"[警告] 纹理不存在且无回退函数: {cache_name}，生成占位纹理")
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    return tex


def _load_normal_from_cache_or_pbr(cache_name, pbr_layer=None, width=512, height=512,
                                    fallback_build_fn=None):
    """加载法线贴图，优先从 PBR 目录，回退缓存，最后程序化生成"""
    import os as _os
    pbr_path = _os.path.join(_os.path.dirname(__file__), "assets", "textures")
    tex_dir = _os.path.join(_os.path.dirname(__file__), ".cache")

    if pbr_layer is not None and _os.path.isdir(pbr_path):
        nrm = pbr_loader.load_pbr_normal(pbr_path, pbr_layer, width, height)
        if nrm is not None:
            return nrm
        # 尝试从高度图生成法线
        nrm = pbr_loader.load_pbr_normal_from_height(pbr_path, pbr_layer, strength=2.0, width=width, height=height)
        if nrm is not None:
            return nrm

    # 回退缓存
    png_path = _os.path.join(tex_dir, f"{cache_name}.png")
    if _os.path.exists(png_path):
        img = np.array(Image.open(png_path), dtype=np.uint8)
        if img.shape[2] == 4:
            img = img[:,:,:3]
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.shape[1], img.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glGenerateMipmap(GL_TEXTURE_2D)
        return tex

    # ★ 程序化生成回退
    if fallback_build_fn is not None:
        print(f"[程序生成] 无 PBR/缓存，程序化生成法线: {cache_name}")
        return _make_texture_cached(cache_name, width, height, fallback_build_fn)

    print(f"[警告] 法线纹理不存在: {cache_name}.png")
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    return tex

# ============================================================
# 导入山体地形模块（山脊骨架 + 地形高度 + 赛道网格）
# ============================================================
from mountain import (
    get_terrain_height,
    get_akina_path,
    build_mountain_road,
    build_guardrails,
    build_akina_trees,
    build_akina_terrain,
    build_concrete_barriers,
    build_retaining_walls,     # 挖方挡土墙
    clear_terrain_cache,
    build_tree_templates,      # 树模板（单棵）
    build_tree_instances,      # 树实例数据
    build_foliage_texture,     # 杉树叶片 alpha 贴图（程序化）
    build_lights,              # 路灯灯杆 + 弯臂 + 灯罩外壳
    build_lamp_emissive,       # 灯罩底部自发光
    build_lamp_ground_pool,    # 地面光池
    build_lamp_light_cone,     # 光锥体积光
    build_buildings,           # 沿路建筑群
    build_roadside_signs,      # 弯道标志牌（凸面镜 + 箭头牌）
    build_roadside_signs_emissive,  # 凸面镜镜面自发光
    build_road_studs,          # 反光道钉（自发光）
    build_drainage_ditches,    # 路肩 U 型混凝土侧沟（沟渠跑法）
    build_road_markings,       # 中央线 / 减速标线 / 补丁 / 胎痕
    build_kiloposts,           # 百米里程标（日式 100m 標）
    build_slope_lattice,       # 挖方边坡格构护坡（法枠工）
    build_rockfall_nets,       # 落石防护网（菱形金網 + 支柱 + 上部索）
    build_venue_ground,        # 起终点场地（沥青平台 + 停车位）
    build_venue_buildings,     # 起终点专用建筑群（管理栋/观景台/车库/便利店）
    gutter_dip,                # 侧沟下沉量（车辆物理）
    gutter_band,               # 沟腔的 signed_distance 区间
    gutter_hold_depth,         # 外轮陷入沟腔的程度
    gutter_present,            # 该处该侧到底有没有可见侧沟（与侧沟 v6 规则同源）
    gutter_phys_warmup,        # 预热上面那张查询表
    venue_wall_margin,         # 起终点场地走廊内的空气墙放宽（车能开上缓冲带）
    venue_surface_at,          # 起终点场地铺面高（路棱柱外回退不认场地，兜底用）
)

# ============================================================
# HUD 界面模块（小地图等 2D 覆盖层）
# ============================================================
from hud import Minimap, UICoordSystem, TextRenderer, Speedometer, GearDisplay, MainMenu, SettingsPanel, destroy_shared_hud, CornerWarning
from corner_analyzer import CornerAnalyzer

# ============================================================
# 赛道边界防入侵系统（实时碰撞检测）
# ============================================================
from track_bounds import signed_distance, nearest_track_info, surface_height_at, build_height_field, build_triangle_index, gpu_height_init, gpu_height_query_batch
from physics_car import VehiclePhysics

import json as _json
import struct as _struct
import io as _io
import base64 as _b64

# ============================================================
# 4x4 矩阵工具
# ============================================================
def mat4_identity():
    return np.eye(4, dtype=np.float32)

def mat4_perspective(fov_y, aspect, near, far):
    f = 1.0 / np.tan(fov_y / 2.0)
    nf = 1.0 / (near - far)
    return np.array([
        [f/aspect,0,0,0],[0,f,0,0],[0,0,(far+near)*nf,2*far*near*nf],[0,0,-1,0]
    ], dtype=np.float32)

def mat4_look_at(eye, target, up):
    eye, t, u = np.array(eye,f32), np.array(target,f32), np.array(up,f32)
    f = (t - eye) / np.linalg.norm(t - eye)
    s = np.cross(f, u) / np.linalg.norm(np.cross(f, u))
    u = np.cross(s, f)
    return np.array([
        [s[0],s[1],s[2],-np.dot(s,eye)],[u[0],u[1],u[2],-np.dot(u,eye)],
        [-f[0],-f[1],-f[2],np.dot(f,eye)],[0,0,0,1]
    ], dtype=np.float32)

def mat4_multiply(a, b):
    return a @ b

def mat4_translate(x, y, z):
    return np.array([
        [1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]
    ], dtype=np.float32)

def mat4_rotate_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]
    ], dtype=np.float32)

def mat4_rotate_x(angle):
    """绕X轴旋转（用于车辆俯仰pitch）"""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]
    ], dtype=np.float32)

def mat4_scale(sx, sy, sz):
    return np.array([
        [sx,0,0,0],[0,sy,0,0],[0,0,sz,0],[0,0,0,1]
    ], dtype=np.float32)

def mat4_ortho(left, right, bottom, top, near, far):
    """正交投影矩阵"""
    return np.array([
        [2/(right-left),0,0,-(right+left)/(right-left)],
        [0,2/(top-bottom),0,-(top+bottom)/(top-bottom)],
        [0,0,-2/(far-near),-(far+near)/(far-near)],
        [0,0,0,1]
    ], dtype=np.float32)

def mat4_multiply_vec(mat, vec):
    """4x4矩阵 × 4分量向量"""
    return np.dot(mat.astype(np.float64), vec.astype(np.float64)).astype(np.float32)

f32 = np.float32

def _clamp(v, lo, hi):
    """钳制 v 到 [lo, hi] 区间"""
    return lo if v < lo else (hi if v > hi else v)

def _angle_wrap(a):
    """将角度规整到 [-pi, pi) 区间"""
    return (a + math.pi) % (2.0 * math.pi) - math.pi

# ============================================================
# 赛道路径由 mountain.py 提供 (get_akina_path)
# 山脊骨架驱动地形和赛道，见 mountain.py
# ============================================================

# ============================================================
# 着色器（PBR Cook-Torrance GGX + 纹理）
# ============================================================
VERTEX_SHADER_SRC = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aTexCoord;
layout(location=3) in vec3 aColor;
layout(location=4) in vec4 aWeights;    // ★ 4 层 PBR 权重

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vTexCoord;
out vec3 vColor;
out vec4 vWeights;                      // ★ 传递到片段

uniform mat4 uMVP;
uniform mat4 uModel;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    gl_Position = uMVP * worldPos;
    vWorldPos = worldPos.xyz;
    vNormal = normalize(mat3(uModel) * aNormal);
    vTexCoord = aTexCoord;
    vColor = aColor;
    vWeights = aWeights;                // ★ 透传
}
"""

TREE_VERTEX_SHADER_SRC = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aTexCoord;
layout(location=3) in vec3 aColor;

// 实例属性（divisor=1） — loc4=aWeights(geo模式) / 未使用(tree模式)
layout(location=5) in vec3 aInstPos;
layout(location=6) in float aInstRot;
layout(location=7) in vec3 aInstScale;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vTexCoord;
out vec3 vColor;
out vec4 vWeights;                      // 兼容片段着色器（树不用地形混合）

uniform mat4 uMVP;
uniform mat4 uModel;
uniform float uTime;   // ★ 风摆动画

void main() {
    // 实例变换：scale → rotateY → translate
    vec3 scaled = aPos * aInstScale;
    float c = cos(aInstRot);
    float s = sin(aInstRot);
    vec3 rotated = vec3(
        scaled.x * c - scaled.z * s,
        scaled.y,
        scaled.x * s + scaled.z * c
    );
    vec3 worldPos = rotated + aInstPos;

    // ★ 风摆：高度越高摆幅越大（树冠动、树根不动）
    float swayPh = uTime * 1.5 + aInstPos.x * 0.37 + aInstPos.z * 0.29;
    float swayAmp = 0.035 * max(aPos.y, 0.0) * aInstScale.y;
    worldPos.x += sin(swayPh) * swayAmp;
    worldPos.z += cos(swayPh * 0.83) * swayAmp * 0.6;

    // 法线旋转
    vec3 normalRotated = vec3(
        aNormal.x * c - aNormal.z * s,
        aNormal.y,
        aNormal.x * s + aNormal.z * c
    );

    vec4 wp = uModel * vec4(worldPos, 1.0);
    gl_Position = uMVP * wp;
    vWorldPos = wp.xyz;
    vNormal = normalize(mat3(uModel) * normalRotated);
    vTexCoord = aTexCoord;

    // ★ 每棵树独立的色相微调（按实例坐标哈希），打破整片树林的均一感
    float thash = fract(sin(dot(floor(aInstPos.xz + 0.5), vec2(12.9898, 78.233))) * 43758.5453);
    vec3 treeTint = mix(vec3(0.80, 0.90, 0.74), vec3(1.16, 1.06, 0.86), thash);
    vColor = aColor * treeTint;
    vWeights = vec4(1.0, 0.0, 0.0, 0.0);  // 仅草地权重，不会被使用
}"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vTexCoord;
in vec3 vColor;
in vec4 vWeights;                         // ★ 4 层材质权重

out vec4 FragColor;

uniform sampler2D uTexture;
uniform sampler2D uGrassTex;               // ★ 草地 PBR 纹理
uniform sampler2D uRockTex;                // ★ 岩石 PBR 纹理
uniform sampler2D uDirtTex;                // ★ 泥土 PBR 纹理
uniform sampler2D uGravelTex;              // ★ 碎石 PBR 纹理
uniform float uUseTerrainBlend;            // ★ 地形纹理混合开关
uniform sampler2D uGrassNrm;               // ★ 草地法线贴图（阶段三）
uniform sampler2D uRockNrm;                // ★ 岩石法线贴图（阶段三）
uniform sampler2D uDirtNrm;                // ★ 泥土法线贴图（阶段三）
uniform sampler2D uGravelNrm;              // ★ 碎石法线贴图（阶段三）
uniform vec3 uLightDir;       // 方向光方向（世界空间）
uniform vec3 uLightColor;     // 方向光颜色
uniform vec3 uViewPos;        // 摄像机位置
uniform vec3 uAmbientColor;   // 环境光颜色（配合 uAmbientBoost 使用）
uniform float uMetallic;      // 金属度 0-1
uniform float uRoughness;     // 粗糙度 0-1
uniform float uHasTexture;    // 是否有纹理
uniform float uAmbientBoost;  // 环境光增益（地形专用，默认1.0）
uniform float uExposure;      // 曝光控制（默认1.0）
uniform int uDebugMode;        // ★ 调试模式：0=正常, 1=草, 2=岩, 3=土, 4=碎石, 5=权重可视化
uniform float uIsEmissive;    // =1.0 时直接输出顶点色，不走光照
uniform float uIsBuilding;    // =1.0 时启用建筑 PBR 纹理分支
uniform sampler2D uBuildingAlbedo;   // 建筑漫反射纹理
uniform sampler2D uBuildingNormal;   // 建筑法线贴图
// uniform sampler2D uBuildingEmissive; // 已移除：改用顶点色触发 Bloom
uniform sampler2D uBuildingAO;       // 建筑环境光遮蔽贴图
uniform float uApplyTonemap;  // =1.0 时执行 ACES+Gamma，=0.0 时输出线性 HDR
uniform sampler2D uFoliageTex;   // ★ 杉树叶片贴图（RGB=叶片色, A=剪影）
uniform float uUseFoliage;       // ★ =1.0 时按叶片贴图做 alpha 测试 + 着色
uniform sampler2D uDetailTex;    // ★ 高频细节层（路面）
uniform float uDetail;           // ★ =1.0 时启用细节层
uniform float uDetailScale;      // ★ 细节层世界平铺密度

const float PI = 3.14159265359;

// ---- GGX / Trowbridge-Reitz 法线分布函数 ----
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float nom = a2;
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    return nom / (PI * denom * denom);
}

// ---- Smith-GGX 几何遮蔽函数（单边） ----
float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

// ---- Smith 联合几何函数 ----
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return GeometrySchlickGGX(NdotV, roughness) * GeometrySchlickGGX(NdotL, roughness);
}

// ---- Schlick 菲涅尔近似 ----
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ---- 三平面投影采样（带 LOD 控制，远距离降级） ----
vec3 triplanar_sample_lod(sampler2D tex, vec3 wp, vec3 n, float scale, int lod) {
    if (lod >= 2) {
        // 远距离：只用 y 平面（地形主要以 Y 为主方向）
        return texture(tex, wp.xz * scale).rgb;
    } else {
        // 中近距离：完整三平面
        vec3 blend = abs(n);
        blend = pow(blend, vec3(4.0));
        blend /= dot(blend, vec3(1.0)) + 1e-6;
        vec3 xP = texture(tex, wp.yz * scale).rgb;
        vec3 yP = texture(tex, wp.xz * scale).rgb;
        vec3 zP = texture(tex, wp.xy * scale).rgb;
        return xP * blend.x + yP * blend.y + zP * blend.z;
    }
}

// ---- 三平面投影法线采样（法线贴图专用，做坐标系变换） ----
vec3 triplanar_normal(sampler2D tex, vec3 wp, vec3 N, float scale) {
    vec3 blend = abs(N);
    blend = pow(blend, vec3(4.0));
    blend /= dot(blend, vec3(1.0)) + 1e-6;

    // 三轴采样，从 [0,1] 解包到 [-1,1]
    vec3 nx = texture(tex, wp.yz * scale).rgb * 2.0 - 1.0;
    vec3 ny = texture(tex, wp.xz * scale).rgb * 2.0 - 1.0;
    vec3 nz = texture(tex, wp.xy * scale).rgb * 2.0 - 1.0;

    // 转换到世界空间（调换轴顺序以匹配各平面投影方向）
    nx = vec3(nx.y, nx.z, nx.x);
    ny = vec3(ny.x, ny.z, ny.y);
    nz = vec3(nz.x, nz.y, nz.z);

    return normalize(nx * blend.x + ny * blend.y + nz * blend.z);
}

// ---- 3D 值噪声（建筑程序化纹理用） ----
float hash31(vec3 p) {
    p = fract(p * vec3(443.8975, 397.2973, 491.1871));
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
}

float vnoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash31(i);
    float b = hash31(i + vec3(1, 0, 0));
    float c = hash31(i + vec3(0, 1, 0));
    float d = hash31(i + vec3(1, 1, 0));
    float e = hash31(i + vec3(0, 0, 1));
    float f_ = hash31(i + vec3(1, 0, 1));
    float g = hash31(i + vec3(0, 1, 1));
    float h = hash31(i + vec3(1, 1, 1));

    float x1 = mix(a, b, f.x);
    float x2 = mix(c, d, f.x);
    float x3 = mix(e, f_, f.x);
    float x4 = mix(g, h, f.x);

    float y1 = mix(x1, x2, f.y);
    float y2 = mix(x3, x4, f.y);

    return mix(y1, y2, f.z);
}

// ---- ACES 电影级色调映射 ----
vec3 ACESFilm(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    // ---- 自发光：直接输出顶点色，不走光照 ----
    if (uIsEmissive > 0.5) {
        FragColor = vec4(vColor, 1.0);
        return;
    }

    // ---- 建筑专用分支（纯顶点色 + 程序化噪点 + PBR 光照） ----
    if (uIsBuilding > 0.5) {
        vec3 wp_b = vWorldPos;
        vec3 N_b = normalize(vNormal);

        // 亮灯窗：直接自发光（>1.0 会触发 Bloom）
        if (vColor.r + vColor.g + vColor.b > 2.4) {
            FragColor = vec4(vColor * 1.5, 1.0);
            return;
        }

        // 顶点色作为基础 albedo
        vec3 albedo_b = vColor;

        // 程序化噪点：模拟灰泥/木材/石材的细微颗粒
        // 用世界坐标，避免顶点扰动导致的闪烁
        float n1 = vnoise(wp_b * 2.5);    // 粗颗粒（5~10cm）
        float n2 = vnoise(wp_b * 12.0);   // 细颗粒（1~2cm）
        albedo_b *= 0.94 + n1 * 0.08 + n2 * 0.04;

        // ---- PBR 光照 ----
        vec3 V_b = normalize(uViewPos - wp_b);
        vec3 L_b = normalize(uLightDir);
        vec3 H_b = normalize(V_b + L_b);
        float NdotL_b = max(dot(N_b, L_b), 0.0);
        float NdotV_b = max(dot(N_b, V_b), 0.0);
        float rough_b = clamp(uRoughness, 0.04, 1.0);
        float NDF_b = DistributionGGX(N_b, H_b, rough_b);
        float G_b = GeometrySmith(N_b, V_b, L_b, rough_b);
        vec3 F0_b = vec3(0.04);
        vec3 F_b = fresnelSchlick(max(dot(H_b, V_b), 0.0), F0_b);
        vec3 spec_b = NDF_b * G_b * F_b / (4.0 * NdotV_b * NdotL_b + 0.0001);
        vec3 kD_b = (vec3(1.0) - F_b);

        vec3 Lo_b = (kD_b * albedo_b / PI + spec_b) * uLightColor * NdotL_b;
        vec3 ambient_b = uAmbientColor * albedo_b * 1.4;   // 环境光增益略高
        vec3 color_b = Lo_b + ambient_b;

        // ---- 高度雾（晴天稀薄，浅蓝） ----
        float dist_b = length(wp_b - uViewPos);
        float fog_b = clamp(0.02 + (1.0 - exp(-0.0006 * dist_b)) * 0.15, 0.0, 0.25);
        color_b = mix(color_b, vec3(0.55, 0.75, 0.88), fog_b);

        FragColor = vec4(color_b, 1.0);
        return;
    }

    // ---- 基础色（所有纹理/顶点颜色都是显示值，不做 gamma 转换） ----
    // ★ LOD 距离分层（改动 A）
    float cam_dist = length(vWorldPos - uViewPos);
    int LOD_LEVEL = 0;
    if (cam_dist > 120.0) LOD_LEVEL = 2;
    else if (cam_dist > 30.0) LOD_LEVEL = 1;

    vec3 albedo;
    vec4 w_terrain = vec4(1.0, 0.0, 0.0, 0.0);  // ★ 法线混合用权重

    if (uUseTerrainBlend > 0.5) {
        // ★ 地形 4 层 PBR 纹理混合（三平面投影）
        vec3 wp = vWorldPos;
        vec3 N_sample = normalize(vNormal);  // ★ 用于三平面纹理采样的法线
        vec4 w = vWeights / (dot(vWeights, vec4(1.0)) + 1e-6);
        w_terrain = w;  // ★ 保存权重，供法线贴图混合使用

        // ★ 调试模式：始终完整采样（改动 E）
        if (uDebugMode != 0) {
            vec3 c_g = triplanar_sample_lod(uGrassTex,  wp, N_sample, 0.03, 0);
            vec3 c_r = triplanar_sample_lod(uRockTex,   wp, N_sample, 0.05, 0);
            vec3 c_d = triplanar_sample_lod(uDirtTex,   wp, N_sample, 0.08, 0);
            vec3 c_v = triplanar_sample_lod(uGravelTex, wp, N_sample, 0.50, 0);
            if (uDebugMode == 1) albedo = c_g;
            else if (uDebugMode == 2) albedo = c_r;
            else if (uDebugMode == 3) albedo = c_d;
            else if (uDebugMode == 4) albedo = c_v;
            else if (uDebugMode == 5) albedo = vec3(w.x, w.y, w.z) + vec3(w.w * 0.5);
        }
        // ★ 正常渲染模式按 LOD 分级（改动 C）
        else if (LOD_LEVEL == 0) {
            // 近处（<30m）：完整 4 层采样（12 次）
            vec3 c_g_far  = triplanar_sample_lod(uGrassTex,  wp, N_sample, 0.03, 0);
            vec3 c_g_near = triplanar_sample_lod(uGrassTex,  wp, N_sample, 0.50, 0);
            vec3 c_g = mix(c_g_far, c_g_near, clamp((30.0 - cam_dist) / 30.0, 0.0, 1.0));
            vec3 c_r = triplanar_sample_lod(uRockTex,   wp, N_sample, 0.05, 0);
            vec3 c_d = triplanar_sample_lod(uDirtTex,   wp, N_sample, 0.08, 0);
            vec3 c_v = triplanar_sample_lod(uGravelTex, wp, N_sample, 0.50, 0);
            albedo = c_g * w.x + c_r * w.y + c_d * w.z + c_v * w.w;
        } else if (LOD_LEVEL == 1) {
            // 中距（30~120m）：单层草地 + 其它层（6 次）
            vec3 c_g = triplanar_sample_lod(uGrassTex,  wp, N_sample, 0.10, 0);
            vec3 c_r = triplanar_sample_lod(uRockTex,   wp, N_sample, 0.05, 0);
            vec3 c_d = triplanar_sample_lod(uDirtTex,   wp, N_sample, 0.08, 0);
            vec3 c_v = triplanar_sample_lod(uGravelTex, wp, N_sample, 0.50, 0);
            albedo = c_g * w.x + c_r * w.y + c_d * w.z + c_v * w.w;
        } else {
            // 远处（>120m）：连续降采样率，保留4层分层
            float g_scale = mix(0.10, 0.05, clamp((cam_dist - 30.0) / 90.0, 0.0, 1.0));
            vec3 c_g = triplanar_sample_lod(uGrassTex,  wp, N_sample, g_scale, 2);
            vec3 c_r = triplanar_sample_lod(uRockTex,   wp, N_sample, g_scale, 2);
            vec3 c_d = triplanar_sample_lod(uDirtTex,   wp, N_sample, g_scale, 2);
            vec3 c_v = triplanar_sample_lod(uGravelTex, wp, N_sample, g_scale * 0.5, 2);
            albedo = c_g * w.x + c_r * w.y + c_d * w.z + c_v * w.w;
        }

        // ★ 修复1：vColor 只提供色调微调（烘焙已减轻压暗，这里只微量修正）
        if (uDebugMode == 0) {
            float ao = dot(vColor, vec3(0.3, 0.6, 0.1));
            albedo *= mix(0.92, 1.05, ao);
        }
    } else if (uHasTexture > 0.5) {
        // 把 vColor 归一化到最大分量=1，只做色调微调，不压暗纹理
        float vmax = max(max(vColor.r, vColor.g), vColor.b);
        vec3 tint = vColor / max(vmax, 0.001);
        // 仅 20% 的色调影响，保留 80% 的原始纹理亮度
        tint = mix(vec3(1.0), tint, 0.2);
        albedo = texture(uTexture, vTexCoord).rgb * tint;
    } else {
        albedo = vColor;
    }

    // ★ 植被交叉面片：alpha 测试裁剪出杉树剪影，并用叶片贴图着色
    if (uUseFoliage > 0.5) {
        vec4 leaf = texture(uFoliageTex, vTexCoord);
        if (leaf.a < 0.30) discard;
        // 冠内上亮下暗（模拟树冠自遮蔽），避免贴图平贴的感觉
        albedo *= leaf.rgb * mix(0.72, 1.28, vTexCoord.y);
    }

    // ★ 高频细节层（路面专用）：双层平铺噪声打破低频脏斑
    if (uDetail > 0.5) {
        float dt1 = texture(uDetailTex, vWorldPos.xz * uDetailScale).r;
        float dt2 = texture(uDetailTex, vWorldPos.xz * uDetailScale * 3.7 + 0.31).r;
        albedo *= mix(vec3(0.82), vec3(1.18), dt1 * 0.65 + dt2 * 0.35);
    }

    float metallic = uMetallic;
    float roughness = clamp(uRoughness, 0.04, 1.0);

    // ★ 阶段三：法线贴图混合（只在近处启用，改动 D）
    vec3 N_geo = normalize(vNormal);
    vec3 N = N_geo;
    if (uUseTerrainBlend > 0.5 && LOD_LEVEL == 0) {
        vec3 n_g = triplanar_normal(uGrassNrm,  vWorldPos, N_geo, 0.15);
        vec3 n_r = triplanar_normal(uRockNrm,   vWorldPos, N_geo, 0.10);
        vec3 n_d = triplanar_normal(uDirtNrm,   vWorldPos, N_geo, 0.25);
        vec3 n_v = triplanar_normal(uGravelNrm, vWorldPos, N_geo, 0.80);
        vec3 N_t = normalize(n_g * w_terrain.x + n_r * w_terrain.y + n_d * w_terrain.z + n_v * w_terrain.w);
        N = normalize(mix(N_geo, N_t, 0.85));
    }

    vec3 V = normalize(uViewPos - vWorldPos);

    // 非金属 F0 ≈ 0.04，金属 F0 = albedo
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // ========== Cook-Torrance BRDF 直接光照 ==========
    vec3 Lo = vec3(0.0);
    vec3 L = normalize(uLightDir);
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);

    float NDF = DistributionGGX(N, H, roughness);
    float G   = GeometrySmith(N, V, L, roughness);
    vec3  F   = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
    vec3 specular = numerator / denominator;

    // ---- 根据距离抑制近处高光（0.5米内无高光，5.5米外恢复正常） ----
    float dist_to_cam = length(vWorldPos - uViewPos);
    float spec_atten = clamp((dist_to_cam - 0.5) / 5.0, 0.0, 1.0);
    specular *= spec_atten;

    // 能量守恒：漫反射 = (1 - F) * (1 - metallic)
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    Lo = (kD * albedo / PI + specular) * uLightColor * NdotL;

    // ========== 环境光（半球环境光：天空蓝在上、地面暖灰反弹在下） ==========
    // 纯常数环境光会让背光面死黑（车辆/树尤甚），半球插值后暗部保有天空色
    vec3 ambSky   = uAmbientColor;
    vec3 ambGround = uAmbientColor * vec3(0.42, 0.38, 0.34);
    vec3 hemiAmb = mix(ambGround, ambSky, clamp(N.y * 0.5 + 0.5, 0.0, 1.0));
    vec3 ambient = hemiAmb * albedo * uAmbientBoost;

    vec3 color = Lo + ambient;

    // ★ 植被专用光照（头文字D 街机的杉树观感关键）：
    //   1) 逆光透光——对着太阳看时叶片发亮
    //   2) 顶部受光强化
    //   3) 环境光加成——防止背光面黑成剪影
    if (uUseFoliage > 0.5) {
        float back = pow(max(dot(-V, L), 0.0), 6.0);
        color += albedo * back * 0.55;
        color += vec3(1.0, 0.95, 0.75) * pow(max(dot(N, L), 0.0), 3.0) * 0.18;
        color += ambient * albedo * 0.35;
    }

    // ========== 正午晴空雾（极淡，亮蓝白）= ==========
    float dist = length(vWorldPos - uViewPos);
    float heightFogFactor = exp(-max(0.0, (vWorldPos.y - uViewPos.y + 5.0)) / 60.0);
    float fog = 0.01 + (1.0 - exp(-0.0003 * dist)) * heightFogFactor * 0.08;
    fog = clamp(fog, 0.0, 0.12);

    vec3 viewDir = normalize(vWorldPos - uViewPos);
    float sunDot = max(dot(viewDir, -uLightDir), 0.0);
    vec3 fogColor = mix(vec3(0.65, 0.82, 0.95),
                        vec3(0.80, 0.90, 1.00),
                        pow(sunDot, 4.0));

    color = mix(color, fogColor, fog);

    // ========== 白天中性色分级 ==========
    vec3 shadowTint = vec3(0.95, 0.97, 1.05);   // 冷色微蓝
    vec3 highlightTint = vec3(1.02, 1.00, 0.98); // 中性微暖
    float luma = dot(color, vec3(0.299, 0.587, 0.114));
    vec3 tint = mix(shadowTint, highlightTint, smoothstep(0.1, 0.6, luma));
    color *= tint;

    // ========== ACES 电影级色调映射 + Gamma 校正 ==========
    // uApplyTonemap==0.0 时输出线性 HDR（供 Bloom 后处理使用）
    // uApplyTonemap==1.0 时正常色调映射 + Gamma
    if (uApplyTonemap > 0.5) {
        color = ACESFilm(color * uExposure);
        color = pow(color, vec3(1.0 / 2.2));
    }

    FragColor = vec4(color, 1.0);
}
"""

# ============================================================
# 天空盒着色器
# ============================================================
SKY_VERTEX_SRC = """
#version 330 core
layout(location=0) in vec3 aPos;
out vec3 vDir;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vDir = aPos;   // 天空球以摄像机为中心，方向 = 局部位置
}
"""

SKY_FRAGMENT_SRC = """
#version 330 core
in vec3 vDir;
out vec4 FragColor;

uniform vec3 uSunDir;        // 指向太阳的单位向量
uniform vec3 uZenithColor;   // 天顶
uniform vec3 uHorizonColor;  // 地平线
uniform vec3 uGroundColor;   // 地平线以下（山谷雾色）
uniform vec3 uSunColor;      // 太阳光晕色
uniform float uTime;         // 云层缓慢漂移

// ---- 值噪声 + fbm（用于云层） ----
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i), b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0)), d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float fbm(vec2 p) {
    float s = 0.0, amp = 0.5;
    for (int i = 0; i < 5; i++) {
        s += amp * vnoise(p);
        p *= 2.03;
        amp *= 0.5;
    }
    return s;
}

void main() {
    vec3 dir = normalize(vDir);
    float h = dir.y;

    // 地平线到天顶的渐变（pow 让地平线暖色带更窄）
    float t = clamp(h, 0.0, 1.0);
    vec3 sky = mix(uHorizonColor, uZenithColor, pow(t, 0.55));

    // 地平线以下是雾海
    if (h < 0.0) {
        float g = clamp(-h * 4.0, 0.0, 1.0);
        sky = mix(uHorizonColor, uGroundColor, g);
    }

    // ---- 太阳盘 + 光晕（收敛强度，避免整片天空被冲白） ----
    vec3 sunDir = normalize(uSunDir);
    float sd = max(dot(dir, sunDir), 0.0);
    float disk = smoothstep(0.9993, 0.9998, sd);
    float glow = pow(sd, 220.0) * 0.55 + pow(sd, 22.0) * 0.10;
    sky += uSunColor * (disk * 2.2 + glow);

    // ---- 程序化云层（只在地平线以上，越靠地平线越密） ----
    if (h > -0.02) {
        // 把方向投影到「云平面」上做透视（近地平线压缩，形成纵深感）
        float planeH = max(h, 0.06);
        vec2 cuv = dir.xz / planeH * 1.15;
        cuv += vec2(uTime * 0.006, uTime * 0.0025);

        float base = fbm(cuv * 1.35);
        float detail = fbm(cuv * 3.10 + base * 1.4);
        float cover = mix(0.46, 0.34, smoothstep(0.0, 0.55, h));  // 高空云量减少
        float cloud = smoothstep(cover, cover + 0.30, base * 0.68 + detail * 0.42);

        // 太阳附近的云更亮（银边），远离太阳的云略暗
        float toSun = pow(max(dot(dir, sunDir), 0.0), 3.0);
        vec3 cloudLit = mix(vec3(0.52, 0.58, 0.66), vec3(1.02, 1.00, 0.95), toSun);
        vec3 cloudDark = vec3(0.42, 0.47, 0.56);
        vec3 cloudCol = mix(cloudDark, cloudLit, smoothstep(0.25, 0.85, base));

        float horizonFade = smoothstep(-0.02, 0.16, h);   // 地平线处淡出
        sky = mix(sky, cloudCol, cloud * 0.85 * horizonFade);
    }

    FragColor = vec4(sky, 1.0);
}
"""

# ============================================================
# Bloom 后处理着色器（全屏四边形）
# ============================================================
BLOOM_VERTEX_SRC = """
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aTexCoord;
out vec2 vTexCoord;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vTexCoord = aTexCoord;
}
"""

BLOOM_BRIGHT_FRAG_SRC = """
#version 330 core
in vec2 vTexCoord;
out vec4 FragColor;
uniform sampler2D uSceneTex;
uniform float uThreshold;  // 亮度阈值，默认1.0

void main() {
    vec3 color = texture(uSceneTex, vTexCoord).rgb;
    // 使用亮度权重提取高亮像素
    float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
    if (brightness > uThreshold) {
        FragColor = vec4(color, 1.0);
    } else {
        FragColor = vec4(0.0, 0.0, 0.0, 0.0);
    }
}
"""

BLOOM_BLUR_FRAG_SRC = """
#version 330 core
in vec2 vTexCoord;
out vec4 FragColor;
uniform sampler2D uImage;
uniform int uHorizontal;  // 1=水平模糊, 0=垂直模糊

// 13 抽头高斯模糊权重（sigma≈3.0, 归一化）
const float weights[13] = float[](
    0.0097, 0.0241, 0.0500, 0.0863, 0.1240,
    0.1488, 0.1488, 0.1240, 0.0863, 0.0500,
    0.0241, 0.0097, 0.000001  // 占位
);
// 实际只用 13 个抽头，weights[12] 几乎为 0

void main() {
    vec2 tex_offset = 1.0 / textureSize(uImage, 0);
    vec3 result = texture(uImage, vTexCoord).rgb * weights[6];  // 中心权重 0.1488

    vec2 dir = uHorizontal > 0 ? vec2(tex_offset.x, 0.0) : vec2(0.0, tex_offset.y);

    for (int i = 0; i < 6; i++) {
        float w = weights[6 + i + 1];  // 对称权重
        vec2 offset = dir * float(i + 1);
        result += texture(uImage, vTexCoord + offset).rgb * w;
        result += texture(uImage, vTexCoord - offset).rgb * w;
    }

    FragColor = vec4(result, 1.0);
}
"""

BLOOM_COMPOSITE_FRAG_SRC = """
#version 330 core
in vec2 vTexCoord;
out vec4 FragColor;
uniform sampler2D uSceneTex;  // HDR 场景
uniform sampler2D uBloomTex;  // 模糊后的亮部
uniform float uBloomStrength; // Bloom 强度，默认0.8
uniform float uExposure;      // 曝光控制
uniform float uContrast;      // ★ 对比度（S 曲线强度）
uniform float uSaturation;    // ★ 饱和度
uniform float uVignette;      // ★ 暗角强度

vec3 ACESFilm(vec3 x) {
    float a = 2.51; float b = 0.03; float c = 2.43;
    float d = 0.59; float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 sceneColor = texture(uSceneTex, vTexCoord).rgb;
    vec3 bloomColor = texture(uBloomTex, vTexCoord).rgb;

    // 加法混合 Bloom
    vec3 hdr = sceneColor + bloomColor * uBloomStrength;

    // 统一色调映射 + Gamma
    vec3 mapped = ACESFilm(hdr * uExposure);
    mapped = pow(mapped, vec3(1.0 / 2.2));

    // ========== ★ 色彩分级（头文字D 街机风格）==========
    float lum = dot(mapped, vec3(0.299, 0.587, 0.114));

    // 1) 对比度：以中灰为轴做 S 曲线，暗部更沉、亮部更跳
    mapped = (mapped - 0.5) * (1.0 + uContrast) + 0.5;

    // 2) 饱和度
    mapped = mix(vec3(lum), mapped, uSaturation);

    // 3) 青橙分离：暗部压向冷青，亮部推暖（街机动画的经典影调）
    vec3 shadowTint = vec3(0.93, 0.99, 1.09);
    vec3 highTint   = vec3(1.05, 1.00, 0.94);
    mapped *= mix(shadowTint, highTint, smoothstep(0.12, 0.78, lum));

    // 4) 暗角
    vec2 q = vTexCoord - 0.5;
    float vig = 1.0 - dot(q, q) * uVignette;
    mapped *= clamp(vig, 0.0, 1.0);

    FragColor = vec4(clamp(mapped, 0.0, 1.0), 1.0);
}
"""

# ============================================================
# ★ FXAA 快速近似抗锯齿（Deferred 管线拿不到 MSAA，用它收边）
# ============================================================
FXAA_FRAG_SRC = """
#version 330 core
in vec2 vTexCoord;
out vec4 FragColor;
uniform sampler2D uTex;
uniform vec2 uTexel;      // 1.0 / 分辨率

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main() {
    vec3 c = texture(uTex, vTexCoord).rgb;
    float lM  = luma(c);
    float lNW = luma(texture(uTex, vTexCoord + vec2(-1.0, -1.0) * uTexel).rgb);
    float lNE = luma(texture(uTex, vTexCoord + vec2( 1.0, -1.0) * uTexel).rgb);
    float lSW = luma(texture(uTex, vTexCoord + vec2(-1.0,  1.0) * uTexel).rgb);
    float lSE = luma(texture(uTex, vTexCoord + vec2( 1.0,  1.0) * uTexel).rgb);

    float lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
    float lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));

    // 平坦区域直接返回，省掉后续采样
    if (lMax - lMin < max(0.0312, lMax * 0.125)) {
        FragColor = vec4(c, 1.0);
        return;
    }

    // 边缘方向估计
    vec2 dir = vec2(
        -((lNW + lNE) - (lSW + lSE)),
         ((lNW + lSW) - (lNE + lSE))
    );
    float reduce = max((lNW + lNE + lSW + lSE) * 0.03125, 1.0 / 128.0);
    float rcpDir = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * rcpDir, vec2(-8.0), vec2(8.0)) * uTexel;

    vec3 rgbA = 0.5 * (texture(uTex, vTexCoord + dir * (1.0 / 3.0 - 0.5)).rgb +
                       texture(uTex, vTexCoord + dir * (2.0 / 3.0 - 0.5)).rgb);
    vec3 rgbB = rgbA * 0.5 + 0.25 * (texture(uTex, vTexCoord + dir * -0.5).rgb +
                                     texture(uTex, vTexCoord + dir *  0.5).rgb);
    float lB = luma(rgbB);

    FragColor = vec4((lB < lMin || lB > lMax) ? rgbA : rgbB, 1.0);
}
"""

# ============================================================
# ★ GPU 粒子 Compute Shader（SSBO 粒子状态 → billboard 顶点）
# ============================================================
PARTICLE_COMPUTE_SRC = """
#version 430 core
layout(local_size_x = 32) in;

layout(std430, binding=12) buffer ParticleStateBuf {
    float state[];  // 每粒子12个float: pos(3), vel(3), life, maxLife, size, type, pad(1)
};

layout(std430, binding=13) buffer VertexOutBuf {
    float vertices[];  // 输出顶点: pos(3)+nrm(3)+uv(2)+col(3) = 11 floats
};

layout(std430, binding=14) buffer CounterBuf {
    uint particleCount;
};

uniform float cs_dt;
uniform vec3 cs_camEye;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint count = particleCount;
    if (idx >= count) return;

    uint sOff = idx * 12;
    vec3 pos   = vec3(state[sOff+0], state[sOff+1], state[sOff+2]);
    float life = state[sOff+6];
    float maxL = state[sOff+7];
    float size = state[sOff+8];
    float ptyp = state[sOff+9];

    // 物理更新已在 Python 侧完成，此处仅生成顶点
    // 已死亡粒子不生成顶点
    if (life <= 0.0) return;

    // 颜色：尾气(typ<0.5)=白→灰，烟尘(typ>=0.5)=棕褐
    float lr = life / maxL;
    vec3 col;
    if (ptyp < 0.5) {
        float c = lr * 0.6 + 0.2;
        col = vec3(c, c, min(1.0, c * 1.05));
    } else {
        float c = lr * 0.5 + 0.08;
        col = vec3(c * 0.9, c * 0.5, c * 0.15);
    }

    // 面朝摄像机 billboard（仅绕 Y 轴）
    vec3 toCam = cs_camEye - pos;
    vec3 fwd = normalize(vec3(toCam.x, 0.0, toCam.z));
    if (length(vec2(toCam.x, toCam.z)) < 0.0001) fwd = vec3(1.0, 0.0, 0.0);
    vec3 rgt = vec3(-fwd.z, 0.0, fwd.x);
    float h = size * 0.5;

    // 6 个顶点 / 粒子（2 个三角形）
    vec3 vv[6] = vec3[6](
        vec3(-rgt.x*h - fwd.x*h, h, -rgt.z*h - fwd.z*h),
        vec3( rgt.x*h - fwd.x*h, h,  rgt.z*h - fwd.z*h),
        vec3( rgt.x*h + fwd.x*h, h,  rgt.z*h + fwd.z*h),
        vec3(-rgt.x*h - fwd.x*h, h, -rgt.z*h - fwd.z*h),
        vec3( rgt.x*h + fwd.x*h, h,  rgt.z*h + fwd.z*h),
        vec3(-rgt.x*h + fwd.x*h, h, -rgt.z*h + fwd.z*h)
    );
    vec3 nrm = vec3(fwd.x, 0.0, fwd.z);

    uint vBase = idx * 6 * 11;
    for (int vi = 0; vi < 6; vi++) {
        uint off = vBase + vi * 11;
        vertices[off+0] = pos.x + vv[vi].x;
        vertices[off+1] = pos.y + vv[vi].y;
        vertices[off+2] = pos.z + vv[vi].z;
        vertices[off+3] = nrm.x; vertices[off+4] = nrm.y; vertices[off+5] = nrm.z;
        vertices[off+6] = 0.0;  vertices[off+7] = 0.0;  // uv
        vertices[off+8] = col.r; vertices[off+9] = col.g; vertices[off+10] = col.b;
    }
}
"""

# ============================================================
# ★ 视锥剔除 Compute Shader（用于树实例等大量对象的 GPU 侧剔除）
# ============================================================
FRUSTUM_CULL_SRC = """
#version 430 core
layout(local_size_x = 64) in;

// 输入：所有树实例 (M, 4) — x, y, z, radius（注意 y 不可省略！）
layout(std430, binding=20) buffer InstanceBuf {
    float instances[];  // 每实例 4 float: x, y, z, radius
};

// 输出：可见实例索引（紧凑排列）
layout(std430, binding=21) buffer VisibleIdxBuf {
    uint visibleIndices[];
};

// 输出：每 LOD 的可见数量（最多 4 个 LOD）
layout(std430, binding=22) buffer LodCountBuf {
    uint lodCounts[4];
};

// 输出：总可见数量
layout(std430, binding=23) buffer CounterBuf2 {
    uint totalVisible;
};

uniform vec4 frustumPlanes[6];  // 6 个平面: (nx, ny, nz, d)
uniform uint totalInstances;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= totalInstances) return;

    uint off = idx * 4;
    float px = instances[off + 0];
    float py = instances[off + 1];
    float pz = instances[off + 2];
    float radius = instances[off + 3];

    // 球体-视锥相交测试（完整 3D，含 y 分量！）
    bool inside = true;
    for (int p = 0; p < 6; p++) {
        float d = frustumPlanes[p].x * px
                + frustumPlanes[p].y * py
                + frustumPlanes[p].z * pz
                + frustumPlanes[p].w;
        if (d < -radius) { inside = false; break; }
    }

    if (inside) {
        uint slot = atomicAdd(totalVisible, 1u);
        visibleIndices[slot] = idx;
    }
}
"""

# ============================================================
# 光锥着色器（体积光，顶点色 + 透明度，加法混合）
# ============================================================
LIGHT_CONE_VERTEX_SRC = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aTexCoord;
layout(location=3) in vec4 aColor;  // RGBA，Alpha 控制透明度
out vec4 vColor;
out vec3 vWorldPos;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vColor = aColor;
    vWorldPos = aPos;
}
"""

LIGHT_CONE_FRAG_SRC = """
#version 330 core
in vec4 vColor;
in vec3 vWorldPos;
out vec4 FragColor;

// 体积光噪声扰动（柔和边缘）
float noise3D(vec3 p) {
    float n = fract(sin(dot(p, vec3(12.9898, 78.233, 45.5432))) * 43758.5453);
    return n * 2.0 - 1.0;
}

float fbm(vec3 p) {
    float value = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    for (int i = 0; i < 3; i++) {
        value += amp * noise3D(p * freq);
        freq *= 2.0;
        amp *= 0.5;
    }
    return value * 0.5 + 0.5;
}

void main() {
    float alpha = vColor.a;

    // 基于世界坐标生成噪声扰动，使光锥边缘柔和
    float noise = fbm(vWorldPos * 0.8) * 0.15;
    alpha *= (1.0 - noise);

    // 光锥底部衰减（离顶点越远越暗）
    float dist_factor = 1.0 - length(vWorldPos.xy) * 0.05;
    alpha *= clamp(dist_factor, 0.3, 1.0);

    FragColor = vec4(vColor.rgb, alpha);
}
"""

# ============================================================
# ★ Deferred Rendering 着色器（OpenGL 4.3+ 几何Pass + 光照Pass）
# ============================================================

# —— 几何阶段 顶点着色器（与现有 VERTEX_SHADER_SRC 功能一致，但追加实例属性） ——
GEO_VERTEX_SRC = """
#version 430 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aTexCoord;
layout(location=3) in vec3 aColor;
layout(location=4) in vec4 aWeights;

// ★ 叶片卡切线（仅树木 VAO 启用 location 8，loc4 留给地形的 aWeights）
layout(location=8) in vec3 aTangent;

// 实例属性（树木 LOD）
layout(location=5) in vec3 aInstPos;
layout(location=6) in float aInstRot;
layout(location=7) in vec3 aInstScale;

out vec3 vWorldPos;
out vec3 vWorldNormal;
out vec2 vTexCoord;
out vec3 vColor;
out vec4 vWeights;
out vec3 vTangent;
flat out int vInstanceID;

uniform mat4 uMVP;
uniform mat4 uModel;
uniform int uIsInstanced;   // 0=普通, 1=实例化

void main() {
    vec3 pos;
    vec3 nrm;
    vec3 tan;
    if (uIsInstanced == 1) {
        vec3 scaled = aPos * aInstScale;
        float c = cos(aInstRot);
        float s = sin(aInstRot);
        vec3 rotated = vec3(scaled.x * c - scaled.z * s, scaled.y, scaled.x * s + scaled.z * c);
        pos = rotated + aInstPos;
        nrm = vec3(aNormal.x * c - aNormal.z * s, aNormal.y, aNormal.x * s + aNormal.z * c);
        tan = vec3(aTangent.x * c - aTangent.z * s, aTangent.y, aTangent.x * s + aTangent.z * c);
    } else {
        pos = aPos;
        nrm = aNormal;
        tan = aTangent;
    }
    vec4 worldPos = uModel * vec4(pos, 1.0);
    gl_Position = uMVP * worldPos;
    vWorldPos = worldPos.xyz;
    vWorldNormal = normalize(mat3(uModel) * nrm);
    vTexCoord = aTexCoord;
    vColor = aColor;
    vTangent = mat3(uModel) * tan;
    vWeights = uIsInstanced == 1 ? vec4(1.0, 0.0, 0.0, 0.0) : aWeights;
    vInstanceID = gl_InstanceID;
}
"""

# —— 几何阶段 顶点着色器（批量 Indirect 模式） ——
GEO_VERTEX_BATCH_SRC = """
#version 430 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aTexCoord;
layout(location=3) in vec3 aColor;
layout(location=4) in float aMaterialID;

out vec3 vWorldPos;
out vec3 vWorldNormal;
out vec2 vTexCoord;
out vec3 vColor;
flat out int vMaterialID;

uniform mat4 uMVP;
uniform mat4 uModel;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    gl_Position = uMVP * worldPos;
    vWorldPos = worldPos.xyz;
    vWorldNormal = normalize(mat3(uModel) * aNormal);
    vTexCoord = aTexCoord;
    vColor = aColor;
    vMaterialID = int(round(aMaterialID));
}
"""

# —— 几何阶段 片段着色器（批量 Indirect 模式，使用 SSBO 材质参数） ——
GEO_FRAGMENT_BATCH_SRC = """
#version 430 core

in vec3 vWorldPos;
in vec3 vWorldNormal;
in vec2 vTexCoord;
in vec3 vColor;
flat in int vMaterialID;

// ★ MRT 输出
layout(location=0) out vec4 gPosition;
layout(location=1) out vec4 gNormal;
layout(location=2) out vec4 gAlbedo;
layout(location=3) out vec4 gMaterial;

// ★ 材质参数 SSBO（binding=4，与 batch_renderer.py 一致）
layout(std430, binding=4) readonly buffer MaterialBuffer {
    vec4 materials[];
};

uniform mat4 uModel;
uniform vec3 uViewPos;

void main() {
    vec3 N = normalize(vWorldNormal);

    // 从 SSBO 读取材质参数
    int mid = max(vMaterialID, 0);
    vec4 matParams = materials[mid];
    float metallic  = matParams.x;
    float roughness = clamp(matParams.y, 0.04, 1.0);
    int flags       = int(round(matParams.z));

    float hasTexture = float((flags & 1) != 0);
    float isBuilding = float((flags & 2) != 0);
    float isEmissive = float((flags & 4) != 0);

    vec3 albedo = vColor;
    float ao = 1.0;
    float materialFlags = 0.0;

    if (isEmissive > 0.5) {
        materialFlags = 1.0;
        gPosition = vec4(vWorldPos, materialFlags);
        gNormal = vec4(N * 0.5 + 0.5, materialFlags);
        gAlbedo = vec4(albedo, metallic);
        gMaterial = vec4(roughness, ao, 0.0, 0.0);
        return;
    }

    // World-space position & normal
    gPosition = vec4(vWorldPos, materialFlags);
    gNormal = vec4(N * 0.5 + 0.5, materialFlags);

    // Albedo (with AO baked into alpha)
    gAlbedo = vec4(albedo, metallic);
    gMaterial = vec4(roughness, ao, isBuilding, 0.0);
}
"""

# —— 几何阶段 片段着色器（写入 G-Buffer，不计算光照） ——
GEO_FRAGMENT_SRC = """
#version 430 core

in vec3 vWorldPos;
in vec3 vWorldNormal;
in vec2 vTexCoord;
in vec3 vColor;
in vec4 vWeights;
in vec3 vTangent;
flat in int vInstanceID;

// ★ MRT 输出
layout(location=0) out vec4 gPosition;      // 世界坐标 + 材质标志
layout(location=1) out vec4 gNormal;         // 世界法线
layout(location=2) out vec4 gAlbedo;         // 反射率 + 金属度
layout(location=3) out vec4 gMaterial;       // 粗糙度 + AO + 自发光标志

uniform sampler2D uTexture;
uniform sampler2D uGrassTex;
uniform sampler2D uRockTex;
uniform sampler2D uDirtTex;
uniform sampler2D uGravelTex;
uniform float uUseTerrainBlend;
uniform sampler2D uGrassNrm;
uniform sampler2D uRockNrm;
uniform sampler2D uDirtNrm;
uniform sampler2D uGravelNrm;
uniform float uMetallic;
uniform float uRoughness;
uniform float uHasTexture;
uniform float uIsEmissive;
uniform float uIsBuilding;
uniform float uIsCar;
uniform sampler2D uBuildingAlbedo;
uniform sampler2D uBuildingNormal;
uniform sampler2D uBuildingAO;
uniform vec3 uViewPos;
uniform int uDebugMode;

// ★ 第三阶段：视差贴图（POM）高度图 + 参数
uniform sampler2D uHeightMap;     // 赛道高度图
uniform float uParallaxScale;     // 视差强度（0=禁用，建议 0.02~0.08）
uniform float uParallaxLayers;    // 步进层数（建议 8~16）
uniform sampler2D uTerrainHeightMap; // 地形统一高度图（三平面 POM）

// ★ PBR 材质通道纹理（替换 uniform 常数）
uniform sampler2D uTrackRoughness;
uniform sampler2D uTrackMetallic;
uniform sampler2D uTrackAO;
uniform sampler2D uTrackNormal;
uniform sampler2D uGrassRoughness;
uniform sampler2D uRockRoughness;
uniform sampler2D uDirtRoughness;
uniform sampler2D uGravelRoughness;
uniform sampler2D uGrassAO;
uniform sampler2D uRockAO;
uniform sampler2D uDirtAO;
uniform sampler2D uGravelAO;
uniform sampler2D uFoliageTex;   // ★ 杉树叶片贴图（RGB=叶片色, A=剪影）
uniform sampler2D uFoliageNrm;   // ★ 叶片法线贴图（RGB=切线空间法线, A=内部遮蔽）
uniform float uUseFoliage;       // ★ =1.0 时按剪影裁剪

const float PI = 3.14159265359;

// —— 世界空间切线计算（从法线和微分推导） ——
vec3 compute_tangent(vec3 N) {
    vec3 c1 = cross(N, vec3(0.0, 0.0, 1.0));
    vec3 c2 = cross(N, vec3(0.0, 1.0, 0.0));
    return normalize(abs(dot(c1, c1)) > 0.001 ? c1 : c2);
}

// —— 标准视差遮蔽贴图（UV 映射，用于赛道纹理） ——
vec2 parallax_occlusion_mapping(sampler2D heightMap, vec2 uv, vec3 V_ts, float heightScale, float numLayers) {
    float layerDepth = 1.0 / numLayers;
    float currentLayerDepth = 0.0;
    vec2 deltaUV = V_ts.xy / max(V_ts.z, 0.01) * heightScale / numLayers;

    vec2 currentUV = uv;
    float currentHeight = texture(heightMap, currentUV).r;

    // 向下步进
    for (float i = 0.0; i < numLayers; i += 1.0) {
        if (currentLayerDepth > currentHeight) break;
        currentUV -= deltaUV;
        currentHeight = texture(heightMap, currentUV).r;
        currentLayerDepth += layerDepth;
    }

    // 线性插值平滑
    vec2 prevUV = currentUV + deltaUV;
    float prevHeight = texture(heightMap, prevUV).r;
    float afterDepth  = currentLayerDepth - currentHeight;
    float beforeDepth = currentHeight - (currentLayerDepth - layerDepth);
    float weight = afterDepth / max(afterDepth + beforeDepth, 0.0001);
    return mix(currentUV, prevUV, weight);
}

// —— 三平面采样（无 POM，回退版本） ——
vec3 triplanar_sample(sampler2D tex, vec3 wp, vec3 n, float scale) {
    vec3 blend = abs(n);
    blend = pow(blend, vec3(4.0));
    blend /= dot(blend, vec3(1.0)) + 1e-6;
    vec3 xP = texture(tex, wp.yz * scale).rgb;
    vec3 yP = texture(tex, wp.xz * scale).rgb;
    vec3 zP = texture(tex, wp.xy * scale).rgb;
    return xP * blend.x + yP * blend.y + zP * blend.z;
}

// —— 三平面法线贴图采样（无 POM，回退版本） ——
vec3 triplanar_normal_map(sampler2D tex, vec3 wp, vec3 N, float scale) {
    vec3 blend = abs(N);
    blend = pow(blend, vec3(4.0));
    blend /= dot(blend, vec3(1.0)) + 1e-6;
    vec3 nx = texture(tex, wp.yz * scale).rgb * 2.0 - 1.0;
    vec3 ny = texture(tex, wp.xz * scale).rgb * 2.0 - 1.0;
    vec3 nz = texture(tex, wp.xy * scale).rgb * 2.0 - 1.0;
    nx = vec3(nx.y, nx.z, nx.x);
    ny = vec3(ny.x, ny.z, ny.y);
    nz = vec3(nz.x, nz.y, nz.z);
    return normalize(nx * blend.x + ny * blend.y + nz * blend.z);
}

// —— 三平面采样 + 视差贴图（地形专用） ——
vec3 triplanar_sample_pom(sampler2D tex, sampler2D hTex, vec3 wp, vec3 n, vec3 V, float scale, float heightScale, float numLayers) {
    vec3 blend = abs(n);
    blend = pow(blend, vec3(4.0));
    blend /= dot(blend, vec3(1.0)) + 1e-6;

    // 计算世界空间切线方向
    vec3 tangent = compute_tangent(n);

    // ---- XY 平面（投影到 Z） ----
    vec3 V_xy = normalize(vec3(dot(V, vec3(1,0,0)), dot(V, vec3(0,1,0)), dot(V, n)));
    float layerDepth = 1.0 / numLayers;
    vec2 deltaUV_xy = V_xy.xy / max(V_xy.z, 0.01) * heightScale / numLayers;
    vec2 uv_xy = wp.xy * scale;
    float curLayer = 0.0;
    float curH = texture(hTex, wp.xy * scale).r;
    for (float i = 0.0; i < numLayers; i += 1.0) {
        if (curLayer > curH) break;
        uv_xy -= deltaUV_xy;
        // 用 wp.xy 近似
        curH = texture(hTex, uv_xy).r;
        curLayer += layerDepth;
    }

    // ---- YZ 平面（投影到 X） ----
    vec3 V_yz = normalize(vec3(dot(V, vec3(0,1,0)), dot(V, vec3(0,0,1)), dot(V, n)));
    vec2 deltaUV_yz = V_yz.xy / max(V_yz.z, 0.01) * heightScale / numLayers;
    vec2 uv_yz = wp.yz * scale;
    curLayer = 0.0;
    curH = texture(hTex, wp.yz * scale).r;
    for (float i = 0.0; i < numLayers; i += 1.0) {
        if (curLayer > curH) break;
        uv_yz -= deltaUV_yz;
        curH = texture(hTex, uv_yz).r;
        curLayer += layerDepth;
    }

    // ---- XZ 平面（投影到 Y） ----
    vec3 V_xz = normalize(vec3(dot(V, vec3(1,0,0)), dot(V, vec3(0,0,1)), dot(V, n)));
    vec2 deltaUV_xz = V_xz.xy / max(V_xz.z, 0.01) * heightScale / numLayers;
    vec2 uv_xz = wp.xz * scale;
    curLayer = 0.0;
    curH = texture(hTex, wp.xz * scale).r;
    for (float i = 0.0; i < numLayers; i += 1.0) {
        if (curLayer > curH) break;
        uv_xz -= deltaUV_xz;
        curH = texture(hTex, uv_xz).r;
        curLayer += layerDepth;
    }

    vec3 xP = texture(tex, uv_yz).rgb;
    vec3 yP = texture(tex, uv_xz).rgb;
    vec3 zP = texture(tex, uv_xy).rgb;
    return xP * blend.x + yP * blend.y + zP * blend.z;
}

vec3 triplanar_normal_map_pom(sampler2D tex, sampler2D hTex, vec3 wp, vec3 N, vec3 V, float scale, float heightScale, float numLayers) {
    // 简化版：用 POM 偏移后的坐标采样法线贴图
    vec3 blend = abs(N);
    blend = pow(blend, vec3(4.0));
    blend /= dot(blend, vec3(1.0)) + 1e-6;

    vec3 tangent = compute_tangent(N);

    // 对 YZ 平面应用 POM
    vec3 V_yz = normalize(vec3(dot(V, vec3(0,1,0)), dot(V, vec3(0,0,1)), dot(V, N)));
    float layerDepth = 1.0 / numLayers;
    vec2 deltaUV_yz = V_yz.xy / max(V_yz.z, 0.01) * heightScale / numLayers;
    vec2 uv_yz = wp.yz * scale;
    float curLayer = 0.0;
    float curH = texture(hTex, wp.yz * scale).r;
    for (float i = 0.0; i < numLayers; i += 1.0) {
        if (curLayer > curH) break;
        uv_yz -= deltaUV_yz;
        curH = texture(hTex, uv_yz).r;
        curLayer += layerDepth;
    }

    // XZ 平面 POM
    vec3 V_xz = normalize(vec3(dot(V, vec3(1,0,0)), dot(V, vec3(0,0,1)), dot(V, N)));
    vec2 deltaUV_xz = V_xz.xy / max(V_xz.z, 0.01) * heightScale / numLayers;
    vec2 uv_xz = wp.xz * scale;
    curLayer = 0.0;
    curH = texture(hTex, wp.xz * scale).r;
    for (float i = 0.0; i < numLayers; i += 1.0) {
        if (curLayer > curH) break;
        uv_xz -= deltaUV_xz;
        curH = texture(hTex, uv_xz).r;
        curLayer += layerDepth;
    }

    // XY 平面 POM
    vec3 V_xy = normalize(vec3(dot(V, vec3(1,0,0)), dot(V, vec3(0,1,0)), dot(V, N)));
    vec2 deltaUV_xy = V_xy.xy / max(V_xy.z, 0.01) * heightScale / numLayers;
    vec2 uv_xy = wp.xy * scale;
    curLayer = 0.0;
    curH = texture(hTex, wp.xy * scale).r;
    for (float i = 0.0; i < numLayers; i += 1.0) {
        if (curLayer > curH) break;
        uv_xy -= deltaUV_xy;
        curH = texture(hTex, uv_xy).r;
        curLayer += layerDepth;
    }

    vec3 nx = texture(tex, uv_yz).rgb * 2.0 - 1.0;
    vec3 ny = texture(tex, uv_xz).rgb * 2.0 - 1.0;
    vec3 nz = texture(tex, uv_xy).rgb * 2.0 - 1.0;
    nx = vec3(nx.y, nx.z, nx.x);
    ny = vec3(ny.x, ny.z, ny.y);
    nz = vec3(nz.x, nz.y, nz.z);
    return normalize(nx * blend.x + ny * blend.y + nz * blend.z);
}

// —— 建筑程序化噪点 ——
float hash31(vec3 p) {
    p = fract(p * vec3(443.8975, 397.2973, 491.1871));
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
}
float vnoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash31(i), hash31(i+vec3(1,0,0)), f.x),
            mix(hash31(i+vec3(0,1,0)), hash31(i+vec3(1,1,0)), f.x), f.y),
        mix(mix(hash31(i+vec3(0,0,1)), hash31(i+vec3(1,0,1)), f.x),
            mix(hash31(i+vec3(0,1,1)), hash31(i+vec3(1,1,1)), f.x), f.y), f.z);
}

void main() {
    vec3 N_geo = normalize(vWorldNormal);
    vec3 albedo;
    float metallic = uMetallic;
    float roughness = clamp(uRoughness, 0.04, 1.0);
    float ao = 1.0;
    float materialFlags = 0.0;  // 0=normal, 1=emissive

    // ★ 植被交叉面片：按叶片剪影裁剪，避免 G-Buffer 里留下实心矩形
    vec3 foliageRGB = vec3(1.0);
    float foliageFlag = 0.0;   // ★ 1.0 = 树冠叶片（光照阶段走 wrap + 透射）
    float carFlag = 0.0;       // ★ 1.0 = 车身（光照阶段走车漆：清漆 + 环境反射）
                               //    走 gNormal.a 传给延迟光照 Pass（全屏四边形
                               //    没有逐物体 uniform，只能靠 G-Buffer 传标志）
    if (uUseFoliage > 0.5) {
        vec4 leaf = texture(uFoliageTex, vTexCoord);
        if (leaf.a < 0.30) discard;
        foliageRGB = leaf.rgb;
        // ★ v < 0.06 是树皮带（树干），只有树冠用叶片法线/叶片光照
        if (vTexCoord.y > 0.06) {
            foliageFlag = 1.0;
            vec4 nrmP = texture(uFoliageNrm, vTexCoord);
            vec3 nt = nrmP.xyz * 2.0 - 1.0;
            // ★ 3A 式小枝面片是任意倾斜的（像瓦片一样朝上外倾），切线空间
            //    必须用"卡片真实面法线"构建，不能再假设 B = 世界上方向 ——
            //    否则倾斜卡片上的法线会被系统性扭曲。
            //    T = 顶点切线（小枝茎沿枝干方向），Np = 卡片面法线，
            //    B = cross(Np, T)（Gram-Schmidt 正交化，右手系）。
            vec3 Np = normalize(N_geo);
            vec3 T = vTangent;
            if (dot(T, T) < 1e-6) {
                T = cross(vec3(0.0, 1.0, 0.0), Np);
            }
            T = normalize(T - Np * dot(T, Np));
            vec3 B = normalize(cross(Np, T));
            vec3 Nf = normalize(T * nt.x + B * nt.y + Np * nt.z);
            // 双面：背面翻转法线，让叶片两面都正确受光
            if (!gl_FrontFacing) Nf = -Nf;
            // 向"上"混合 25%，保留卡片体积假象。★ 原来混 45% 会把法线贴图的
            // 信息抹掉近一半，实测让法线对最终画面的影响只剩 ~1%（等于白做）
            N_geo = normalize(mix(Nf, vec3(0.0, 1.0, 0.0), 0.25));
            // 小枝内部遮蔽（贴图 A 通道，图集里已标定到完整 0~1 动态范围）
            ao = clamp(nrmP.a, 0.0, 1.0);
        }
    }

    // —— 自发光 ——
    if (uIsEmissive > 0.5) {
        materialFlags = 1.0;
        gPosition = vec4(vWorldPos, materialFlags);
        gNormal = vec4(N_geo * 0.5 + 0.5, 0.0);
        gAlbedo = vec4(vColor, 0.0);
        gMaterial = vec4(0.0, 0.0, 0.0, 1.0);  // w=1.0 标记自发光
        return;
    }

    // —— 建筑分支 ——
    if (uIsBuilding > 0.5) {
        if (vColor.r + vColor.g + vColor.b > 2.4) {
            // 亮灯窗 → 自发光
            materialFlags = 1.0;
            gPosition = vec4(vWorldPos, materialFlags);
            gNormal = vec4(N_geo * 0.5 + 0.5, 0.0);
            gAlbedo = vec4(0.0);
            gMaterial = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }
        albedo = vColor;
        float n1 = vnoise(vWorldPos * 2.5);
        float n2 = vnoise(vWorldPos * 12.0);
        albedo *= 0.94 + n1 * 0.08 + n2 * 0.04;
        roughness = 0.85;
        metallic = 0.0;
        gPosition = vec4(vWorldPos, 0.0);
        gNormal = vec4(N_geo * 0.5 + 0.5, 0.0);
        gAlbedo = vec4(albedo, metallic);
        gMaterial = vec4(roughness, 1.0, 0.0, 0.0);
        return;
    }

    // —— ★ POM 参数 ——
    float pomScale = uParallaxScale;
    float pomLayers = uParallaxLayers;
    vec3 V_dir = normalize(uViewPos - vWorldPos);

    // —— ★ 车辆分支：纯 albedo + 几何法线，完全绕开赛道 PBR/POM ——
    vec4 w = vec4(1.0, 0.0, 0.0, 0.0);
    if (uIsCar > 0.5) {
        float vmax = max(max(vColor.r, vColor.g), vColor.b);
        vec3 tint = vColor / max(vmax, 0.001);
        tint = mix(vec3(1.0), tint, 0.2);
        if (uHasTexture > 0.5) {
            albedo = texture(uTexture, vTexCoord).rgb * tint;
        } else {
            albedo = vColor;
        }
        metallic = 0.0;
        roughness = clamp(uRoughness, 0.04, 1.0);
        ao = 1.0;
        // N_geo 保持几何法线，不采样任何法线贴图、不做 POM
        carFlag = 1.0;
        // ★★ 这里曾经有一行 `if (dot(N_geo, V_dir) < 0.0) N_geo = -N_geo;`
        //    作为"闭合车壳双面修正"的兜底。它是个误诊产物：当时实测到车身
        //    G-Buffer 法线朝下（N.y≈-0.53、N·L<0 占 76%），以为是 GLB 法线反了；
        //    真实原因是 geo_prog 的 uViewPos 从未上传（恒为 0,0,0），
        //    V_dir 是个错误方向，这行按错误方向把本来正确的法线翻转了。
        //    现已在几何 Pass 补上 glUniform3fv(guViewPos, cam_eye)。
        //    离屏 A/B（_scratch/_car_check.py：真实着色器 + 真实 VAO）证明
        //    正确视点下车身 G-Buffer 法线是 N.y=+0.67、N·V<0 占 0.0% ——
        //    法线本来就对，不需要任何兜底翻转。切勿再加回来。
    } else if (uUseTerrainBlend > 0.5) {
        vec3 wp = vWorldPos;
        w = vWeights / (dot(vWeights, vec4(1.0)) + 1e-6);

        // ★ 地形三平面 POM（当启用了高度图时）
        bool usePom = pomScale > 0.001;
        if (uDebugMode != 0) {
            vec3 c_g = usePom ? triplanar_sample_pom(uGrassTex,  uTerrainHeightMap, wp, N_geo, V_dir, 0.03, pomScale * 0.1, pomLayers)
                              : triplanar_sample(uGrassTex,  wp, N_geo, 0.03);
            vec3 c_r = usePom ? triplanar_sample_pom(uRockTex,   uTerrainHeightMap, wp, N_geo, V_dir, 0.05, pomScale * 0.1, pomLayers)
                              : triplanar_sample(uRockTex,   wp, N_geo, 0.05);
            vec3 c_d = usePom ? triplanar_sample_pom(uDirtTex,   uTerrainHeightMap, wp, N_geo, V_dir, 0.08, pomScale * 0.1, pomLayers)
                              : triplanar_sample(uDirtTex,   wp, N_geo, 0.08);
            vec3 c_v = usePom ? triplanar_sample_pom(uGravelTex, uTerrainHeightMap, wp, N_geo, V_dir, 0.50, pomScale * 0.1, pomLayers)
                              : triplanar_sample(uGravelTex, wp, N_geo, 0.50);
            if (uDebugMode == 1) albedo = c_g;
            else if (uDebugMode == 2) albedo = c_r;
            else if (uDebugMode == 3) albedo = c_d;
            else if (uDebugMode == 4) albedo = c_v;
            else if (uDebugMode == 5) albedo = vec3(w.x, w.y, w.z) + vec3(w.w * 0.5);
        } else {
            float cam_dist = length(vWorldPos - uViewPos);
            vec3 wp2 = vWorldPos;
            vec3 c_g = usePom ? triplanar_sample_pom(uGrassTex,  uTerrainHeightMap, wp2, N_geo, V_dir, 0.03, pomScale * 0.1, pomLayers)
                              : triplanar_sample(uGrassTex,  wp2, N_geo, 0.03);
            vec3 c_r = usePom ? triplanar_sample_pom(uRockTex,   uTerrainHeightMap, wp2, N_geo, V_dir, 0.05, pomScale * 0.1, pomLayers)
                              : triplanar_sample(uRockTex,   wp2, N_geo, 0.05);
            vec3 c_d = usePom ? triplanar_sample_pom(uDirtTex,   uTerrainHeightMap, wp2, N_geo, V_dir, 0.08, pomScale * 0.1, pomLayers)
                              : triplanar_sample(uDirtTex,   wp2, N_geo, 0.08);
            vec3 c_v = usePom ? triplanar_sample_pom(uGravelTex, uTerrainHeightMap, wp2, N_geo, V_dir, 0.50, pomScale * 0.1, pomLayers)
                              : triplanar_sample(uGravelTex, wp2, N_geo, 0.50);
            albedo = c_g * w.x + c_r * w.y + c_d * w.z + c_v * w.w;
        }
        if (uDebugMode == 0) {
            float ao_t = dot(vColor, vec3(0.3, 0.6, 0.1));
            albedo *= mix(0.92, 1.05, ao_t);
        }
        // ★ 地形法线贴图混合（带 POM 版本）
        vec3 n_g = usePom ? triplanar_normal_map_pom(uGrassNrm,  uTerrainHeightMap, vWorldPos, N_geo, V_dir, 0.15, pomScale * 0.1, pomLayers)
                          : triplanar_normal_map(uGrassNrm,  vWorldPos, N_geo, 0.15);
        vec3 n_r = usePom ? triplanar_normal_map_pom(uRockNrm,   uTerrainHeightMap, vWorldPos, N_geo, V_dir, 0.10, pomScale * 0.1, pomLayers)
                          : triplanar_normal_map(uRockNrm,   vWorldPos, N_geo, 0.10);
        vec3 n_d = usePom ? triplanar_normal_map_pom(uDirtNrm,   uTerrainHeightMap, vWorldPos, N_geo, V_dir, 0.25, pomScale * 0.1, pomLayers)
                          : triplanar_normal_map(uDirtNrm,   vWorldPos, N_geo, 0.25);
        vec3 n_v = usePom ? triplanar_normal_map_pom(uGravelNrm, uTerrainHeightMap, vWorldPos, N_geo, V_dir, 0.80, pomScale * 0.1, pomLayers)
                          : triplanar_normal_map(uGravelNrm, vWorldPos, N_geo, 0.80);
        N_geo = normalize(mix(N_geo, normalize(n_g * w.x + n_r * w.y + n_d * w.z + n_v * w.w), 0.85));
    } else if (uHasTexture > 0.5) {
        // ★ 赛道纹理：标准 UV（POM 已禁用，不再需要）
        vec2 uv = vTexCoord;
        float cam_dist_t = length(vWorldPos - uViewPos);

        // ★ 导数式 TBN：T 沿 UV 的 U 方向，由屏幕空间梯度求出，
        //   banking/坡道上不再跳变（伪切线是水波根源）
        vec3 dpx = dFdx(vWorldPos), dpy = dFdy(vWorldPos);
        vec2 dux = dFdx(vTexCoord), duy = dFdy(vTexCoord);
        vec3 T = cross(dpy, N_geo) * duy.x + cross(N_geo, dpx) * dux.x;
        T = dot(T, T) > 1e-8 ? normalize(T) : compute_tangent(N_geo);   // 退化兜底
        vec3 B = normalize(cross(N_geo, T));
        // （POM 已禁用，若以后重新开，V_ts 也必须用这里的 T/B，不能用 compute_tangent）

        // ★ 删掉 *vColor：vColor 已经被归一化为纯色调微调，不再压暗纹理
        float vmax = max(max(vColor.r, vColor.g), vColor.b);
        vec3 tint = vColor / max(vmax, 0.001);
        tint = mix(vec3(1.0), tint, 0.2);
        albedo = texture(uTexture, uv).rgb * tint;
        // ★ PBR：从贴图采样 roughness/metallic/AO
        float tr = texture(uTrackRoughness, uv).r;
        if (tr > 0.0) {
            // 粗糙度保底 0.6，保留骨料微闪光
            roughness = clamp(tr, 0.6, 1.0);
        }
        float tm = texture(uTrackMetallic, uv).r;
        metallic = clamp(tm, 0.0, 0.05);  // 沥青不是金属
        float ta = texture(uTrackAO, uv).r;
        if (ta > 0.0) ao = ta;
        // ★ 真实法线贴图采样（导数式 TBN，帧方向与 UV 实际梯度一致）
        vec3 tn = texture(uTrackNormal, uv).rgb * 2.0 - 1.0;
        vec3 nrmNormal = normalize(T * tn.x + B * tn.y + N_geo * tn.z);
        // ★ 法线强度随距离淡出：远处高频凹凸只剩高光闪烁，没有细节收益
        float nrmFade = clamp(1.0 - cam_dist_t / 90.0, 0.25, 1.0);
        N_geo = normalize(mix(N_geo, nrmNormal, 0.85 * nrmFade));
    } else {
        albedo = vColor;
    }

    // ★ PBR：地形 roughness/AO 按权重混合（仅在 terrain blend 激活时）
    if (uUseTerrainBlend > 0.5 && uDebugMode == 0) {
        vec3 wp_t = vWorldPos;
        float r_g = texture(uGrassRoughness,  wp_t.xz * 0.03).r;
        float r_r = texture(uRockRoughness,   wp_t.xz * 0.05).r;
        float r_d = texture(uDirtRoughness,   wp_t.xz * 0.08).r;
        float r_v = texture(uGravelRoughness, wp_t.xz * 0.50).r;
        float a_g = texture(uGrassAO,  wp_t.xz * 0.03).r;
        float a_r = texture(uRockAO,   wp_t.xz * 0.05).r;
        float a_d = texture(uDirtAO,   wp_t.xz * 0.08).r;
        float a_v = texture(uGravelAO, wp_t.xz * 0.50).r;
        // 用当前 w 权重混合（与 albedo/法线使用同一组权重）
        vec4 w_t = vWeights / (dot(vWeights, vec4(1.0)) + 1e-6);
        roughness = clamp(r_g * w_t.x + r_r * w_t.y + r_d * w_t.z + r_v * w_t.w, 0.04, 1.0);
        ao = clamp(a_g * w_t.x + a_r * w_t.y + a_d * w_t.z + a_v * w_t.w, 0.0, 1.0);
        metallic = 0.0;  // 地形非金属
    }

    gPosition = vec4(vWorldPos, 0.0);
    gNormal = vec4(N_geo * 0.5 + 0.5, carFlag);
    gAlbedo = vec4(albedo * foliageRGB, metallic);
    gMaterial = vec4(roughness, ao, foliageFlag, 0.0);
}
"""

# —— 光照阶段 顶点着色器（全屏四边形） ——
LIGHT_VERTEX_SRC = """
#version 430 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aTexCoord;
out vec2 vTexCoord;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vTexCoord = aTexCoord;
}
"""
FULLSCREEN_VERT_SRC = LIGHT_VERTEX_SRC

# —— 光照阶段 片段着色器（全屏采样 G-Buffer，PBR 光照） ——
LIGHT_FRAGMENT_SRC = """
#version 430 core
in vec2 vTexCoord;
out vec4 FragColor;

// G-Buffer 纹理（由几何阶段填充）
layout(binding=0) uniform sampler2D gPosition;
layout(binding=1) uniform sampler2D gNormal;
layout(binding=2) uniform sampler2D gAlbedo;
layout(binding=3) uniform sampler2D gMaterial;

uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uViewPos;
uniform vec3 uAmbientColor;
uniform float uExposure;
uniform float uBloomThreshold;

// ★ CSM 级联阴影贴图（优化：3→2级联）
uniform sampler2D uShadowMap0;     // 级联0（近处，0~200m）
uniform sampler2D uShadowMap1;     // 级联1（远处，200~1000m）
uniform mat4 uLightVP[2];          // 2个级联的光源VP矩阵
uniform vec4 uCascadeSplits;       // x=split0(200), y=split1(1000), z=级联0分辨率, w=未使用
uniform float uShadowStrength;     // 阴影强度（0=无阴影, 1=全阴影）

// ★ SSAO 环境光遮蔽
uniform sampler2D uSSAOTex;        // SSAO 结果纹理
uniform float uSSAOEnabled;        // SSAO 开关

// ★★ 解析式环境天空辐亮度（替代缺失的 cubemap / 反射探针）
//    与延迟模式下"天空"的着色保持一致（light pass 里的 emissive 渐变）。
//    作用：光滑介电表面（尤其是黑车漆）的可读性几乎全部来自"反射环境"，
//    没有它，黑漆的 albedo≈0 ⇒ 漫反射≈0、伪镜面≈0 ⇒ 必然是一坨死黑。
uniform vec3 uSkyZenith;
uniform vec3 uSkyHorizon;
uniform vec3 uSkyGround;
uniform vec3 uSkySun;

vec3 envRadianceBase(vec3 d) {
    float h = d.y;
    vec3 c = mix(uSkyHorizon, uSkyZenith, pow(clamp(h, 0.0, 1.0), 0.55));
    if (h < 0.0) {
        c = mix(uSkyHorizon, uSkyGround, clamp(-h * 4.0, 0.0, 1.0));
    }
    return c;
}

vec3 envRadiance(vec3 d) {
    vec3 c = envRadianceBase(d);
    float sd = max(dot(d, normalize(uLightDir)), 0.0);
    c += uSkySun * (smoothstep(0.9993, 0.9998, sd) * 2.2
                    + pow(sd, 220.0) * 0.55 + pow(sd, 22.0) * 0.10);
    return c;
}

// ★ 车漆专用环境反射：天空底色 + 紧致太阳盘（镜面里该看到太阳），
//   去掉 pow(sd,22) 的宽角光晕 —— 它会在朝阳面板上糊一片暖白，
//   是车身"白模化"的成因之一。非车漆表面仍走 envRadiance，行为不变。
vec3 envRadianceCar(vec3 d) {
    float sd = max(dot(d, normalize(uLightDir)), 0.0);
    return envRadianceBase(d) + uSkySun * (smoothstep(0.9993, 0.9998, sd) * 2.2
                                           + pow(sd, 220.0) * 0.55);
}

// ★ IBL 球谐环境光照
uniform vec3 uSHCoeffs[9];       // 3阶SH系数（实际只用到前4项）

const float PI = 3.14159265359;

// —— PBR 函数 ——
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    return a2 / (PI * (NdotH2 * (a2 - 1.0) + 1.0) * (NdotH2 * (a2 - 1.0) + 1.0));
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return GeometrySchlickGGX(max(dot(N, V), 0.0), roughness)
         * GeometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ★ 3阶 SH 环境光照重建
vec3 shIrradiance(vec3 N) {
    return uSHCoeffs[0] * 0.282095
         + uSHCoeffs[1] * 0.488603 * N.y
         + uSHCoeffs[2] * 0.488603 * N.z
         + uSHCoeffs[3] * 0.488603 * N.x;
}

vec3 ACESFilm(vec3 x) {
    float a = 2.51; float b = 0.03; float c = 2.43;
    float d = 0.59; float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// ★ CSM 阴影采样（PCF 3x3 软阴影，2级联版）
float sampleShadowPCF(sampler2D shadowMap, vec3 ndc, float bias, float texelSize) {
    vec2 uv = ndc.xy * 0.5 + 0.5;
    float currentDepth = ndc.z * 0.5 + 0.5;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 1.0;

    float shadow = 0.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float pcfDepth = texture(shadowMap, uv + vec2(x, y) * texelSize).r;
            shadow += (currentDepth - bias > pcfDepth) ? 0.0 : 1.0;
        }
    }
    return shadow / 9.0;
}

// ★ 2级联 CSM 阴影计算（normal offset：沿法线推出 15cm 防自阴影粉刺）
float calcCSMShadow(vec3 worldPos, vec3 N, vec3 L) {
    float viewDist = length(worldPos - uViewPos);
    int cascade = 0;
    if (viewDist > uCascadeSplits.x) cascade = 1;

    vec3 shadowPos = worldPos + N * 0.15;
    mat4 lightVP = uLightVP[cascade];
    vec4 clip = lightVP * vec4(shadowPos, 1.0);
    vec3 ndc = clip.xyz / clip.w;

    float bias = max(0.002 * (1.0 - dot(N, L)), 0.0005);
    float texelSize = 1.0 / uCascadeSplits.z;

    float shadow;
    if (cascade == 0) shadow = sampleShadowPCF(uShadowMap0, ndc, bias, texelSize);
    else shadow = sampleShadowPCF(uShadowMap1, ndc, bias, texelSize);

    // 级联过渡混合（消除硬边）
    if (cascade == 1) {
        float t = smoothstep(uCascadeSplits.x - 10.0, uCascadeSplits.x + 10.0, viewDist);
        float s0 = sampleShadowPCF(uShadowMap0, ndc, bias, texelSize);
        shadow = mix(s0, shadow, t);
    }

    return mix(1.0, shadow, uShadowStrength);
}

void main() {
    // 读取 G-Buffer
    vec4 posPack   = texture(gPosition, vTexCoord);
    vec4 nrmPack   = texture(gNormal, vTexCoord);
    vec4 albPack   = texture(gAlbedo, vTexCoord);
    vec4 matPack   = texture(gMaterial, vTexCoord);

    float materialFlags = posPack.a;
    vec3 worldPos = posPack.xyz;
    vec3 N = normalize(nrmPack.xyz * 2.0 - 1.0);
    vec3 albedo = albPack.rgb;
    float metallic = albPack.a;
    float roughness = matPack.r;
    float ao = matPack.g;
    float isEmissive = matPack.a;
    float foliageFlag = matPack.b;   // ★ 1.0 = 树冠叶片
    float carFlag = nrmPack.a;        // ★ 1.0 = 车身（几何 Pass 写在 gNormal.a）

    // —— 自发光：跳过 PBR，直接输出亮色（用于 Bloom 提取） ——
    if (materialFlags > 0.5 || isEmissive > 0.5) {
        // 用屏幕坐标做简单的天空渐变（顶蓝→地平线淡蓝），让天空有层次
        vec2 uv = vTexCoord;
        vec3 skyColor = mix(vec3(0.75, 0.88, 1.05), vec3(0.35, 0.62, 1.05), uv.y);
        // albedo 几乎为零 = 天空（无顶点色）；否则是其它自发光物体（灯罩）
        vec3 emissiveColor = (length(albedo) < 0.01) ? skyColor : albedo;
        FragColor = vec4(emissiveColor, 1.0);
        return;
    }

    // —— PBR Cook-Torrance 直接光照 ——
    vec3 V = normalize(uViewPos - worldPos);
    vec3 L = normalize(uLightDir);
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 specular = NDF * G * F / (4.0 * NdotV * NdotL + 0.0001);

    // 近处高光抑制
    float dist_to_cam = length(worldPos - uViewPos);
    float spec_atten = clamp((dist_to_cam - 0.5) / 5.0, 0.0, 1.0);
    specular *= spec_atten;

    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

    // ★ CSM 阴影采样
    float shadowFactor = calcCSMShadow(worldPos, N, L);

    vec3 Lo;
    if (foliageFlag > 0.5) {
        // —— 叶片光照：wrap 漫反射 + 宽角透射 + 针叶微高光 ——
        float NdL = dot(N, L);
        // wrap：针叶是半透薄片，明暗交界比 Lambert 宽得多（"毛绒感"来源）。
        // ★ 再乘 2.2 的"叶片能量补偿"：叶片散射远强于不透明表面，纯 wrap 会让
        //    整棵树发黑（实测出屏 RGB 仅 (28,74,29)）。亮度必须从这里给 ——
        //    而不是从白色高光给，那是把树打成一团白丝的老路。
        float wrapNdotL = clamp((NdL + 0.45) / 1.45, 0.0, 1.0);
        // ★ 透射：不能直接用 dot(V,-L) 的四次方。本作太阳高度角约 50°（光向
        //    y=0.76），近水平的镜头视角下那个点积最大只有 ~0.57，四次方后
        //    ≈0.11，等于整局游戏都看不到叶背透光。改用带法线偏移的视线向量
        //    （Frostbite 的半透近似），既能在大角度下生效，又随叶片法线变化。
        vec3 Vt = normalize(V + N * 0.40);
        float trans = pow(clamp(dot(Vt, -L), 0.0, 1.0), 1.6)
                    * clamp(1.0 - NdL * 0.6, 0.0, 1.0);
        // ★ 针叶微高光：必须 (a) 很弱、(b) 用叶片色着色。
        //    旧写法是 vec3(0.92,0.96,0.82) * 0.40 * uLightColor 的纯白高光，
        //    峰值约为漫反射项的 8 倍 —— 实测让 20~24% 的树像素被打到近白，
        //    出屏 RGB 从 (28,74,29) 浊化成 (105,124,88)，就是用户看到的
        //    "满树白丝"。现在保留一点点高光（针叶确实有蜡质反光，也是法线贴图
        //    在平光下唯一的杠杆），但幅度压到 1/7 且跟随叶片色，不会白化。
        vec3 Hf = normalize(V + L);
        float sheen = pow(clamp(dot(N, Hf), 0.0, 1.0), 12.0);
        vec3 sheenCol = vec3(0.70, 0.78, 0.58) * 0.40 + albedo * 1.8;
        Lo  = (kD * albedo / PI * wrapNdotL * 2.2) * uLightColor * shadowFactor;
        Lo += albedo * vec3(1.45, 1.20, 0.55) * uLightColor
              * trans * (0.30 + 0.70 * shadowFactor) * 1.10;
        Lo += sheenCol * sheen * uLightColor * shadowFactor * 0.06;
    } else {
        Lo = (kD * albedo / PI + specular) * uLightColor * NdotL * shadowFactor;
    }

    // ★ 车漆清漆层（clearcoat）：替代旧的伪造紧高光，带能量守恒
    //   ★★ "白模"修复：旧写法三个坑 ——
    //   (a) 掠射角 NdotV→0 时分母 4·NdotV·NdotL 趋 0，GGX 峰值被除法推到天上，
    //       形成大块白色热斑；
    //   (b) 绕过了普通镜面项已有的 spec_atten 近距衰减，追逐相机 6~8m 贴脸时
    //       高光整片刷白；
    //   (c) 没有能量上限，与底漆高光/环境反射叠加后轻松超 1.6。
    if (carFlag > 0.5) {
        const float ccRough = 0.06;                       // 清漆层极光滑
        float NDF_cc = DistributionGGX(N, H, ccRough);
        float G_cc   = GeometrySmith(N, V, L, ccRough);
        vec3  F_cc   = fresnelSchlick(max(dot(H, V), 0.0), vec3(0.04));
        // NdotV 钳到 0.10：净效果 = 掠射角高光最多放大约 2.5 倍，不再爆炸
        vec3  specCc = NDF_cc * G_cc * F_cc / (4.0 * max(NdotV, 0.10) * NdotL + 0.0001);
        specCc = min(specCc, vec3(4.0));                  // 硬上限双保险
        specCc *= spec_atten;                             // 与普通镜面同规则
        // 能量守恒：清漆界面反射掉的部分，底漆漫反射/镜面都要衰减
        float F_cV = fresnelSchlick(NdotV, vec3(0.04)).r;
        Lo *= (1.0 - F_cV);
        Lo += specCc * uLightColor * NdotL * shadowFactor;
    }

    // ★ IBL 环境光照（SH 球谐 + 粗糙度退化镜面反射，替代纯色环境光）
    float ssao = 1.0;
    if (uSSAOEnabled > 0.5) {
        ssao = texture(uSSAOTex, vTexCoord).r;
    }
    vec3 irradiance = shIrradiance(N);
    irradiance = max(irradiance, uAmbientColor * 0.5); // 保底 50% 防垂直面黑
    vec3 kD_env = (vec3(1.0) - fresnelSchlick(NdotV, F0)) * (1.0 - metallic);
    vec3 diffuseIBL = irradiance * albedo * kD_env;
    if (foliageFlag > 0.5) {
        diffuseIBL *= vec3(1.06, 1.00, 0.86);   // ★ 抵消 SH 环境光的偏蓝
    }

    // ★ IBL 镜面反射近似（粗糙度退化的方向光 + 环境镜面）
    float specIBL = pow(1.0 - roughness, 3.0);
    vec3 reflDir = reflect(-V, N);
    float reflNdotL = max(dot(reflDir, L), 0.0);
    vec3 specularIBL = uLightColor * specIBL * reflNdotL * F0 * 0.12;

    // ★★ 环境反射（真正的 3A 车漆做法）
    //    旧代码的 specularIBL 是"拿光源色伪造镜面"，峰值只有 ~0.003 —— 等于没有。
    //    光滑表面的镜面能谱必须来自"反射环境"。本引擎没有 cubemap/反射探针，
    //    所以用与天空一致的解析式辐亮度按反射向量采样，再按 Fresnel 加权。
    //    ★ 必须按粗糙度衰减：路面 roughness≈0.75，若不衰减，掠射角下 F_env→1，
    //      整条路会变成一面镜子（实测全场景被抬亮 2.5 倍、车区过曝到 0.91）。
    float envGloss = pow(1.0 - roughness, 2.0);
    vec3 envRefl = envRadiance(reflDir);
    vec3 F_env = F0 + (vec3(1.0) - F0) * pow(1.0 - max(NdotV, 0.0), 5.0);
    vec3 envSpec = envRefl * F_env * (0.05 + 0.95 * envGloss);
    if (carFlag > 0.5) {
        // 车漆清漆层环境反射：清漆极光滑，底漆环境反射被清漆能量守恒衰减
        // ★★ "白模"修复三连：
        //   (a) 权重压到 0.55 —— 解析天空是均匀亮色，全权重等于给车身糊一层白纱，
        //       黄漆 albedo 被盖掉 → 整车褪色泛白；
        //   (b) edgeKill —— NdotV<0.20 的边缘区 Fresnel→1，白边最重，平滑压掉；
        //   (c) 换 envRadianceCar —— 只保留紧致太阳盘镜像，去掉宽角光晕。
        const float ccRough = 0.06;
        float F_cV = fresnelSchlick(NdotV, vec3(0.04)).r;
        float ccGloss = pow(1.0 - ccRough, 2.0);
        float edgeKill = smoothstep(0.02, 0.20, NdotV);
        vec3 envSpecCc = envRadianceCar(reflDir)
                         * fresnelSchlick(NdotV, vec3(0.04)) * ccGloss
                         * 0.55 * edgeKill;
        envSpec = envSpec * (1.0 - F_cV) + envSpecCc;
    }

    // ★ 叶片天空可见度
    //    叶片着色里环境光(IBL)其实是主导项（漫反射直射项在大约 50° 太阳 +
    //    近垂直卡片法线下只有它的 1/2 左右），而 SH 辐照度几乎不随法线变化，
    //    于是法线贴图再细也对画面没有杠杆。这里给叶片加一个"朝上的针叶吃到
    //    更多天光、朝下的被树冠自身遮住"的因子，让主导项也能读出叶簇体积。
    //    取值 0.60~1.32，均值约 0.96 ⇒ 整体亮度基本不变，只增加结构。
    float skyVis = 1.0;
    if (foliageFlag > 0.5) {
        skyVis = 0.60 + 0.72 * clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
        // ★ 叶片已有 贴图AO × 顶点AO 两层，SSAO 只保留 30%，否则冠内压成死灰
        ssao = mix(ssao, 1.0, 0.70);
        // ★ 掠射角下 F_env→1，天空蓝 envRefl 直接糊在叶面上（淡青主凶）
        envSpec *= 0.30;
        // ★ 灰白色伪镜面（F0=0.04 的灰）同步压半
        specularIBL *= 0.50;
    }
    vec3 ambient = (diffuseIBL * skyVis + specularIBL + envSpec) * ao * ssao;

    vec3 color = Lo + ambient;

    // ★ 车漆软限幅（保色相版）：亮度-色度分离压缩
    //   旧写法整色乘系数压缩，R/G 通道同时超限时色度被一起削掉 → 高光死白，
    //   黄漆变"白膜"。现在只压亮度、保留色度比例：过曝区域仍是
    //   "很亮的黄漆"而不是白色。色度上限防止压缩后色度占比过大发霓虹。
    if (carFlag > 0.5) {
        float luma_c = dot(color, vec3(0.299, 0.587, 0.114));
        if (luma_c > 1.2) {
            vec3 chroma = color - vec3(luma_c);              // 色度分量（有正有负）
            float over = luma_c - 1.2;
            float lumaNew = 1.2 + over / (1.0 + over * 0.8); // 软 knee，渐近 ~2.45
            float chromaMax = 0.40 * lumaNew;
            float cl = length(chroma);
            if (cl > chromaMax) chroma *= chromaMax / cl;
            color = vec3(lumaNew) + chroma;
        }
    }

    // —— 正午晴空雾（增强远处亮度补偿，修正远暗近亮） ——
    float dist = length(worldPos - uViewPos);
    float fog = 0.01 + (1.0 - exp(-0.00045 * dist)) * 0.22;
    fog = clamp(fog, 0.0, 0.30);

    vec3 viewDir = normalize(worldPos - uViewPos);
    float sunDot = max(dot(viewDir, -uLightDir), 0.0);
    vec3 fogColor = mix(vec3(0.65, 0.82, 0.95), vec3(0.80, 0.90, 1.00), pow(sunDot, 4.0));
    color = mix(color, fogColor, fog);

    // —— 白天中性色分级 ——
    vec3 shadowTint = vec3(0.95, 0.97, 1.05);
    vec3 highlightTint = vec3(1.02, 1.00, 0.98);
    float luma = dot(color, vec3(0.299, 0.587, 0.114));
    vec3 tint = mix(shadowTint, highlightTint, smoothstep(0.1, 0.6, luma));

    // —— HDR 输出（不做色调映射，留给 Bloom 后处理） ——
    FragColor = vec4(color, 1.0);
}
"""

# ============================================================
# CSM 级联阴影贴图 深度渲染着色器
# ============================================================
CSM_DEPTH_VERTEX_SRC = """
#version 430 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aTexCoord;
layout(location=3) in vec3 aColor;

// 实例属性（树木用）
layout(location=5) in vec3 aInstPos;
layout(location=6) in float aInstRot;
layout(location=7) in vec3 aInstScale;

out vec2 vTexCoord;   // ★ 供叶片 alpha 测试使用
uniform mat4 uLightVP;
uniform mat4 uModel;
uniform int uIsInstanced;

void main() {
    vec3 pos;
    if (uIsInstanced == 1) {
        vec3 scaled = aPos * aInstScale;
        float c = cos(aInstRot);
        float s = sin(aInstRot);
        vec3 rotated = vec3(scaled.x * c - scaled.z * s, scaled.y, scaled.x * s + scaled.z * c);
        pos = rotated + aInstPos;
    } else {
        pos = aPos;
    }
    vec4 worldPos = uModel * vec4(pos, 1.0);
    gl_Position = uLightVP * worldPos;
    vTexCoord = aTexCoord;
}
"""

CSM_DEPTH_FRAGMENT_SRC = """
#version 430 core
in vec2 vTexCoord;
uniform sampler2D uFoliageTex;   // ★ 叶片 alpha 贴图
uniform float uUseFoliage;       // ★ =1.0 时按剪影裁剪，避免面片投出矩形影子

void main() {
    if (uUseFoliage > 0.5) {
        if (texture(uFoliageTex, vTexCoord).a < 0.30) discard;
    }
    // 深度自动写入
}
"""

# ============================================================
# SSAO 屏幕空间环境光遮蔽 着色器
# ============================================================
SSAO_VERTEX_SRC = """
#version 430 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aTexCoord;
out vec2 vTexCoord;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vTexCoord = aTexCoord;
}
"""

SSAO_FRAGMENT_SRC = """
#version 430 core
in vec2 vTexCoord;
out vec4 FragColor;

layout(binding=0) uniform sampler2D gPosition;
layout(binding=1) uniform sampler2D gNormal;
uniform sampler2D uNoiseTex;       // 4x4 随机旋转纹理
uniform vec3 uSamples[32];          // 半球采样核
uniform mat4 uProjection;
uniform vec2 uScreenSize;
uniform float uSSAORadius;          // 采样半径
uniform float uSSAOBias;            // 深度偏移
uniform vec3 uViewPos;              // 视点位置（用于比较视空间 Z）

void main() {
    vec4 posPack = texture(gPosition, vTexCoord);
    vec3 worldPos = posPack.xyz;
    vec3 N = normalize(texture(gNormal, vTexCoord).xyz * 2.0 - 1.0);

    // 天空/无效像素跳过
    float materialFlags = posPack.a;
    if (materialFlags > 0.5 || length(N) < 0.01) {
        FragColor = vec4(1.0);
        return;
    }

    // 随机旋转向量（4x4 噪声纹理平铺）
    vec2 noiseScale = uScreenSize / 4.0;
    vec3 randVec = normalize(texture(uNoiseTex, vTexCoord * noiseScale).xyz * 2.0 - 1.0);

    // 构建 TBN 矩阵
    vec3 tangent = normalize(randVec - N * dot(randVec, N));
    vec3 bitangent = cross(N, tangent);

    float occlusion = 0.0;
    float radius = uSSAORadius;
    float currentZ = length(worldPos - uViewPos);  // 当前片元的视空间距离

    for (int i = 0; i < 32; i++) {
        // 半球采样方向（偏向法线方向）
        vec3 sampleDir = normalize(tangent * uSamples[i].x + bitangent * uSamples[i].y + N * uSamples[i].z);
        vec3 samplePos = worldPos + sampleDir * radius;

        // 投影到屏幕空间
        vec4 clipPos = uProjection * vec4(samplePos, 1.0);
        vec3 ndc = clipPos.xyz / clipPos.w;
        vec2 sampleUV = ndc.xy * 0.5 + 0.5;

        // 检查采样点是否在屏幕内
        if (sampleUV.x < 0.0 || sampleUV.x > 1.0 || sampleUV.y < 0.0 || sampleUV.y > 1.0)
            continue;

        // 正确的 SSAO 深度比较：比较视空间距离（Z）
        vec3 scenePos = texture(gPosition, sampleUV).xyz;
        float sceneZ = length(scenePos - uViewPos);   // 场景点离摄像机多远
        float sampleZ = length(samplePos - uViewPos);  // 采样点离摄像机多远
        float rangeCheck = smoothstep(0.0, 1.0, radius / max(abs(sceneZ - currentZ), 0.0001));

        // 如果场景点比采样点更靠近摄像机 → 遮挡
        if (sampleZ > sceneZ + uSSAOBias) occlusion += rangeCheck;
    }

    occlusion = 1.0 - (occlusion / 32.0);
    occlusion = pow(occlusion, 1.5);  // 对比度增强

    FragColor = vec4(vec3(occlusion), 1.0);
}
"""

# ============================================================
# SSAO 模糊着色器（4x4 双边滤波）
# ============================================================
SSAO_BLUR_FRAG_SRC = """
#version 430 core
in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler2D uSSAOTex;
uniform sampler2D gNormal;
uniform vec2 uScreenSize;

void main() {
    vec2 texelSize = 1.0 / uScreenSize;
    float result = 0.0;
    float totalWeight = 0.0;
    vec3 centerN = normalize(texture(gNormal, vTexCoord).xyz * 2.0 - 1.0);

    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            vec2 offset = vec2(float(x), float(y)) * texelSize * 1.5;
            vec3 nbrN = normalize(texture(gNormal, vTexCoord + offset).xyz * 2.0 - 1.0);
            float nDot = max(dot(centerN, nbrN), 0.0);
            float spatialW = 1.0 / (1.0 + float(x*x + y*y));
            float weight = spatialW * nDot;
            result += texture(uSSAOTex, vTexCoord + offset).r * weight;
            totalWeight += weight;
        }
    }

    FragColor = vec4(vec3(result / max(totalWeight, 0.001)), 1.0);
}
"""

# ============================================================
# SSR 屏幕空间反射 着色器
# ============================================================
SSR_VERTEX_SRC = """
#version 430 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aTexCoord;
out vec2 vTexCoord;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vTexCoord = aTexCoord;
}
"""

SSR_FRAGMENT_SRC = """
#version 430 core
in vec2 vTexCoord;
out vec4 FragColor;

layout(binding=0) uniform sampler2D gPosition;
layout(binding=1) uniform sampler2D gNormal;
layout(binding=2) uniform sampler2D gAlbedo;
layout(binding=3) uniform sampler2D gMaterial;
uniform sampler2D uSceneTex;         // HDR 场景颜色
uniform sampler2D uDepthTex;         // 深度纹理
uniform mat4 uProjection;
uniform mat4 uInvProjection;
uniform mat4 uView;
uniform vec3 uViewPos;
uniform vec2 uScreenSize;
uniform float uMaxDist;              // 最大追踪距离
uniform float uStepSize;             // 步长
uniform int uMaxSteps;               // 最大步数
uniform float uThickness;            // 深度容差

vec3 worldPosFromDepth(vec2 uv, float depth) {
    vec4 clipPos = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 worldPos = uInvProjection * clipPos;
    worldPos /= worldPos.w;
    return worldPos.xyz;
}

float linearDepth(float d) {
    float near = 0.5;
    float far = 2000.0;
    return (2.0 * near) / (far + near - d * (far - near));
}

void main() {
    vec4 posPack = texture(gPosition, vTexCoord);
    vec3 worldPos = posPack.xyz;
    vec3 N = normalize(texture(gNormal, vTexCoord).xyz);
    float roughness = texture(gMaterial, vTexCoord).r;
    float metallic = texture(gAlbedo, vTexCoord).a;

    // 只对金属度>0或粗糙度<0.5的表面做反射（赛道沥青≈metallic=0, rough=0.98 不做反射）
    if (metallic < 0.01 && roughness > 0.7) {
        FragColor = vec4(0.0);
        return;
    }
    // 天空跳过
    if (posPack.a > 0.5) {
        FragColor = vec4(0.0);
        return;
    }

    // 反射方向
    vec3 V = normalize(uViewPos - worldPos);
    vec3 R = reflect(-V, N);

    // 对粗糙表面抖动反射方向
    if (roughness > 0.1) {
        float r1 = fract(sin(dot(vTexCoord, vec2(12.9898, 78.233))) * 43758.5453);
        float r2 = fract(sin(dot(vTexCoord, vec2(93.131, 47.451))) * 78901.2345);
        vec3 randDir = normalize(vec3(r1 - 0.5, r2 - 0.5, r1 + r2 - 1.0));
        R = normalize(mix(R, randDir, roughness * 0.3));
    }

    // 屏幕空间 Raymarching
    vec4 startClip = uProjection * vec4(worldPos, 1.0);
    vec3 startNDC = startClip.xyz / startClip.w;

    vec4 endWorld = vec4(worldPos + R * uMaxDist, 1.0);
    vec4 endClip = uProjection * endWorld;
    vec3 endNDC = endClip.xyz / endClip.w;

    vec3 stepNDC = (endNDC - startNDC) / float(uMaxSteps);
    vec2 stepUV = stepNDC.xy * 0.5;
    float stepDepth = stepNDC.z;

    vec2 rayUV = vTexCoord;
    float rayDepth = startNDC.z;

    float hit = 0.0;
    vec3 hitColor = vec3(0.0);

    for (int i = 0; i < 128; i++) {
        if (i >= uMaxSteps) break;

        rayUV += stepUV;
        rayDepth += stepDepth;

        if (rayUV.x < 0.0 || rayUV.x > 1.0 || rayUV.y < 0.0 || rayUV.y > 1.0)
            break;

        float sceneDepth = texture(uDepthTex, rayUV).r;
        float sceneLinearDepth = linearDepth(sceneDepth);

        if (rayDepth > sceneDepth + uThickness * 0.01) {
            hit = 1.0;
            hitColor = texture(uSceneTex, rayUV).rgb;
            break;
        }
    }

    // 边缘衰减
    vec2 edgeDist = min(rayUV, 1.0 - rayUV);
    float edgeFade = smoothstep(0.0, 0.15, min(edgeDist.x, edgeDist.y));

    // 粗糙度模糊混合
    float reflStrength = (1.0 - roughness) * 0.5;
    hitColor *= hit * edgeFade * reflStrength;

    FragColor = vec4(hitColor, 1.0);
}
"""

# ============================================================
# 体积光（Volumetric Light / God Rays）着色器
# ============================================================
VOLUMETRIC_VERTEX_SRC = """
#version 430 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aTexCoord;
out vec2 vTexCoord;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vTexCoord = aTexCoord;
}
"""

VOLUMETRIC_FRAG_SRC = """
#version 430 core
in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler2D uDepthTex;         // 深度纹理
uniform sampler2D uShadowMap;        // CSM 阴影图（级联0，近处）
uniform mat4 uInvProjection;
uniform mat4 uLightVP;               // 光源 VP 矩阵
uniform vec3 uViewPos;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec2 uScreenSize;
uniform float uTime;
uniform int uNumSteps;               // 步进数
uniform float uScattering;           // 散射强度
uniform float uMaxDist;              // 最大追踪距离

// 3D 值噪声
float hash3D(vec3 p) {
    float h = dot(p, vec3(127.1, 311.7, 74.7));
    return fract(sin(h) * 43758.5453);
}

float noise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(mix(hash3D(i), hash3D(i + vec3(1,0,0)), f.x),
            mix(hash3D(i + vec3(0,1,0)), hash3D(i + vec3(1,1,0)), f.x), f.y),
        mix(mix(hash3D(i + vec3(0,0,1)), hash3D(i + vec3(1,0,1)), f.x),
            mix(hash3D(i + vec3(0,1,1)), hash3D(i + vec3(1,1,1)), f.x), f.y), f.z);
}

float fbm3D(vec3 p) {
    float v = 0.0, amp = 0.5, freq = 1.0;
    for (int i = 0; i < 4; i++) {
        v += amp * noise3D(p * freq);
        freq *= 2.0;
        amp *= 0.5;
    }
    return v;
}

float linearDepth(float d) {
    float near = 0.5, far = 2000.0;
    return (2.0 * near) / (far + near - d * (far - near));
}

float shadowSample(vec3 worldPos) {
    vec4 clip = uLightVP * vec4(worldPos, 1.0);
    vec3 ndc = clip.xyz / clip.w;
    vec2 uv = ndc.xy * 0.5 + 0.5;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 1.0;

    float currentDepth = ndc.z * 0.5 + 0.5;
    float shadow = 0.0;
    vec2 texelSize = 1.0 / vec2(2048.0);

    // 3x3 PCF
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float pcfDepth = texture(uShadowMap, uv + vec2(x, y) * texelSize).r;
            shadow += (currentDepth - 0.002 > pcfDepth) ? 0.0 : 1.0;
        }
    }
    return shadow / 9.0;
}

void main() {
    float depth = texture(uDepthTex, vTexCoord).r;
    if (depth >= 1.0) {
        FragColor = vec4(0.0);
        return;
    }

    // 重建世界坐标
    vec4 clipPos = vec4(vTexCoord * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 worldPosH = uInvProjection * clipPos;
    vec3 worldPos = worldPosH.xyz / worldPosH.w;

    vec3 V = normalize(worldPos - uViewPos);
    float viewDist = length(worldPos - uViewPos);
    float stepDist = min(viewDist, uMaxDist) / float(uNumSteps);

    // 太阳屏幕空间位置
    vec4 sunClip = uInvProjection * vec4(uLightDir * 100.0, 1.0);
    vec3 sunNDC = sunClip.xyz / sunClip.w;
    vec2 sunUV = sunNDC.xy * 0.5 + 0.5;

    vec3 rayPos = uViewPos;
    vec3 rayStep = V * stepDist;
    float scattering = 0.0;
    float transmittance = 1.0;

    for (int i = 0; i < 64; i++) {
        if (i >= uNumSteps) break;

        rayPos += rayStep;

        // 3D 噪声雾密度
        float noiseVal = fbm3D(rayPos * 0.05 + uTime * 0.02);
        float heightFog = exp(-max(0.0, rayPos.y - uViewPos.y + 5.0) / 30.0);
        float density = noiseVal * heightFog * 0.15;

        // 阴影采样
        float shadow = shadowSample(rayPos);
        float lightContrib = shadow * max(dot(normalize(uLightDir), -V), 0.0);

        // Mie 散射相位函数
        float cosTheta = dot(V, -uLightDir);
        float phase = (1.0 - 0.9 * 0.9) / pow(1.0 + 0.9 * 0.9 - 2.0 * 0.9 * cosTheta, 1.5);
        phase /= 4.0 * 3.14159;

        scattering += density * transmittance * lightContrib * phase;
        transmittance *= exp(-density * stepDist);
    }

    vec3 volumetricColor = scattering * uLightColor * uScattering * 3.0;
    FragColor = vec4(volumetricColor, 1.0);
}
"""

# ============================================================
# Compute Shader：GPU 加速赛道高度场查询
# ============================================================
COMPUTE_HEIGHT_SRC = """
#version 430 core
layout(local_size_x = 64, local_size_y = 1) in;

// 三角形网格数据（与视觉赛道完全同构）
layout(std430, binding=0) readonly buffer TriMesh {
    float tri_data[];  // 每三角形 9 个 float: (x0,z0,y0, x1,z1,y1, x2,z2,y2)
};

// 查询结果输出
layout(std430, binding=1) buffer OutputHeights {
    float heights[256];       // 每个查询点的高度结果
    float valid_flags[256];   // 1.0=有效, -1.0=无效
};

// 查询点数组
layout(std430, binding=2) buffer QueryPoints {
    float queries[512];       // (x,z) 成对存储
};

uniform int uTriCount;     // 三角形总数
uniform int uQueryCount;   // 查询点总数

// 2D 重心坐标判断点是否在三角形内
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

    // 读取查询点
    float qx = queries[gid * 2];
    float qz = queries[gid * 2 + 1];
    vec2 q = vec2(qx, qz);

    float best_h = -999.0;
    float found = -1.0;

    // 遍历所有三角形
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
            break;  // 找到第一个包含三角形即退出（同一网格格内三角形不重叠）
        }
    }

    heights[gid] = best_h;
    valid_flags[gid] = found;
}
"""

# ============================================================
# 程序化纹理生成（带 .png 缓存，比 .npy 节省约 80% 磁盘空间）
# ============================================================
_TEX_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")

def _tex_cache_path(name):
    """纹理缓存文件路径（使用 .png 格式，体积更小）"""
    path = os.path.join(_TEX_CACHE_DIR, f"{name}.png")
    if not os.path.exists(_TEX_CACHE_DIR):
        try:
            os.makedirs(_TEX_CACHE_DIR)
        except OSError:
            return None
    return path

def _make_texture_cached(cache_name, width, height, build_fn):
    """
    带缓存的纹理生成，使用 .png 压缩格式。
    build_fn(width, height) 返回 (height, width, 3) uint8 数组
    """
    cache_path = _tex_cache_path(cache_name)
    if cache_path and os.path.exists(cache_path):
        print(f"[缓存] 加载纹理 {cache_name}.png...")
        img = np.array(Image.open(cache_path), dtype=np.uint8)
    else:
        print(f"[缓存] 生成纹理 {cache_name}（首次运行）...")
        img = build_fn(width, height)
        if cache_path is not None and img is not None:
            try:
                # 先清理同名的旧 .npy 缓存（平滑升级）
                old_npy = os.path.join(_TEX_CACHE_DIR, f"{cache_name}.npy")
                if os.path.exists(old_npy):
                    os.remove(old_npy)
                # 保存为 .png，大幅减小磁盘占用
                Image.fromarray(img).save(cache_path, optimize=True)
                print(f"[缓存]  纹理 {cache_name}.png 已缓存")
            except Exception as e:
                print(f"[缓存]  纹理 {cache_name} 缓存失败: {e}")
    # 上传到 GPU
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glGenerateMipmap(GL_TEXTURE_2D)
    # 开启各向异性过滤，提升斜视角度下的纹理清晰度
    try:
        max_aniso = glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, min(16.0, max_aniso))
    except Exception:
        pass  # 如果 PyOpenGL 没绑定这个扩展，忽略即可
    return tex

def make_texture(width, height, data_callback):
    """保留原接口兼容，但内部使用缓存（基于回调函数名做缓存key不可行，直接透传）"""
    # 由于没有 cache_name，回退到无缓存版本
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            img[y, x] = data_callback(x, y, width, height)
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glGenerateMipmap(GL_TEXTURE_2D)
    return tex

# ★ 模块级纹理 build 函数（供 _load_tex_from_cache_or_pbr 的 fallback_build_fn 使用）
def _build_track_tex(w, h):
    """多层混合沥青纹理 2048x2048（随机骨料），U=横向，V=沿路方向"""
    size = 2048
    rng = np.random.RandomState(71)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    # ★ 语义反转：v_along 是沿赛道方向（纹理 Y 轴），u_across 是横向（纹理 X 轴）
    u_across = xx / size    # 横向 0~1
    v_along  = yy / size    # 沿路 0~1

    # ---- 1. 基础沥青底色 ----
    base = np.full((size, size), 50.0, dtype=np.float32)

    # ---- 2. 真实骨料 ----
    pebbles_small = rng.normal(0, 10, size=(size, size))
    pebbles_bright = (rng.rand(size, size) > 0.985).astype(np.float32) * rng.uniform(30, 70, size=(size, size))
    pebbles_dark = (rng.rand(size, size) > 0.99).astype(np.float32) * rng.uniform(-20, -5, size=(size, size))
    base += pebbles_small + pebbles_bright + pebbles_dark

    # ---- 3. 局部磨损 ----
    from scipy.ndimage import zoom, gaussian_filter
    patch_low_res = rng.normal(0, 1, size=(64, 64))
    patch = zoom(patch_low_res, (size/64, size/64), order=1) * 15
    base += patch

    # ---- 4. 裂缝 ----
    crack = np.zeros_like(base)
    for _ in range(25):
        start_x, start_y = rng.randint(0, size), rng.randint(0, size)
        length = rng.randint(50, 300)
        angle = rng.uniform(0, 2*np.pi)
        for t in range(length):
            angle += rng.normal(0, 0.2)
            cx = int(start_x + t * np.cos(angle))
            cy = int(start_y + t * np.sin(angle))
            if 0 <= cx < size and 0 <= cy < size:
                crack[cy, cx] = 1
                crack[max(0,cy-1):min(size,cy+2), max(0,cx-1):min(size,cx+2)] = 1
    crack = gaussian_filter(crack, sigma=0.8)
    base -= crack * 35

    # ---- 5. 组装 RGB ----
    img = np.stack([base, base * 0.98, base * 0.95], axis=-1)

    # ★ 中线沿赛道方向（在纹理 X 轴固定 = 横向位置 0.5 处画一条竖向线）
    #   竖向线 = 沿 Y 轴延伸 = 沿赛道纵向延伸 → 正确的车道中线
    center = np.abs(u_across - 0.5) < 0.006
    img[center] = [210, 180, 50]

    # ★ 两侧白色边缘线（横向位置 0.045 / 0.955）
    edge1 = np.abs(u_across - 0.045) < 0.005
    edge2 = np.abs(u_across - 0.955) < 0.005
    img[edge1] = [200, 200, 195]
    img[edge2] = [200, 200, 195]

    return np.clip(img, 0, 255).astype(np.uint8)


def gen_track_tex():
    """多层混合沥青纹理 2048x2048，带缓存"""
    return _make_texture_cached("track_tex_v8", 2048, 2048, _build_track_tex)


def _build_track_normal(w, h):
    """从程序化赛道高度场生成法线贴图（Sobel 梯度→切空间法线）"""
    size = 2048
    rng = np.random.RandomState(71)
    from scipy.ndimage import gaussian_filter, zoom

    # ---- 高度场（与 _build_track_tex 完全一致） ----
    base = np.full((size, size), 50.0, dtype=np.float32)
    pebbles_small = rng.normal(0, 10, size=(size, size))
    pebbles_bright = (rng.rand(size, size) > 0.985).astype(np.float32) * rng.uniform(30, 70, size=(size, size))
    pebbles_dark = (rng.rand(size, size) > 0.99).astype(np.float32) * rng.uniform(-20, -5, size=(size, size))
    base += pebbles_small + pebbles_bright + pebbles_dark
    patch_low_res = rng.normal(0, 1, size=(64, 64))
    patch = zoom(patch_low_res, (size/64, size/64), order=1) * 15
    base += patch
    crack = np.zeros_like(base)
    for _ in range(25):
        start_x, start_y = rng.randint(0, size), rng.randint(0, size)
        length = rng.randint(50, 300)
        angle = rng.uniform(0, 2*np.pi)
        for t in range(length):
            angle += rng.normal(0, 0.2)
            cx = int(start_x + t * np.cos(angle))
            cy = int(start_y + t * np.sin(angle))
            if 0 <= cx < size and 0 <= cy < size:
                crack[cy, cx] = 1
                crack[max(0,cy-1):min(size,cy+2), max(0,cx-1):min(size,cx+2)] = 1
    crack = gaussian_filter(crack, sigma=0.8)
    base -= crack * 35

    # ---- 标线略微凸起（+8 ≈ 3% 高度） ----
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    u_across = xx / size
    center = np.abs(u_across - 0.5) < 0.006
    edge1 = np.abs(u_across - 0.045) < 0.005
    edge2 = np.abs(u_across - 0.955) < 0.005
    base[center] += 8.0
    base[edge1] += 8.0
    base[edge2] += 8.0

    # ---- Sobel 梯度 → 切空间法线 ----
    gx = np.zeros_like(base)
    gy = np.zeros_like(base)
    gx[:, 1:-1] = (base[:, 2:] - base[:, :-2]) * 0.5
    gy[1:-1, :] = (base[2:, :] - base[:-2, :]) * 0.5
    strength = 1.5
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(base)
    norm = np.sqrt(nx*nx + ny*ny + nz*nz) + 1e-6
    nx /= norm; ny /= norm; nz /= norm

    # 编码到 0~1：R=横向(左负右正), G=纵向(下负上正), B=朝上
    nrm = np.stack([nx, ny, nz], axis=-1) * 0.5 + 0.5
    return np.clip(nrm * 255, 0, 255).astype(np.uint8)


def gen_track_normal():
    """赛道法线贴图 2048x2048，带缓存"""
    return _make_texture_cached("track_nrm_v1", 2048, 2048, _build_track_normal)


def gen_building_tex():
    """玻璃幕墙窗户纹理 512x512，带缓存"""
    def _build(w, h):
        rng = random.Random(89)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                if x % 26 < 4 or y % 30 < 4:
                    img[y, x] = [70, 75, 85]
                else:
                    img[y, x] = [
                        max(0, min(255, rng.randint(20, 100) + 130)),
                        max(0, min(255, rng.randint(20, 100) + 120)),
                        max(0, min(255, rng.randint(20, 100) + 100)),
                    ]
        return img
    return _make_texture_cached("building_tex", 512, 512, _build)


def gen_building_wall_tex():
    """
    日式建筑分层纹理 512x512：
      顶部 1/5（Y 0.00~0.20）: 瓦片（黑灰，横向瓦楞）
      中上部 1/5（Y 0.20~0.40）: 木檐板（深棕，垂直木纹）
      中部 3/5（Y 0.40~1.00）: 灰泥墙（米白色，有细微污渍）
    """
    def _build(w, h):
        size = 512
        rng = np.random.RandomState(20240118)
        img = np.zeros((size, size, 3), dtype=np.uint8)
        for y in range(size):
            v = y / size
            for x in range(size):
                u = x / size
                if v < 0.20:
                    tile_v = int(y / 16) % 2
                    base = 40 if tile_v == 0 else 32
                    shade = rng.randint(-6, 6)
                    img[y, x] = [base + shade, base + shade, base + shade + 3]
                elif v < 0.40:
                    grain = int(20 * math.sin(u * 60.0) + 20 * rng.rand())
                    img[y, x] = [max(0, 60 + grain), max(0, 38 + grain // 2), max(0, 18 + grain // 3)]
                else:
                    low_freq = 8 * math.sin(u * 8.0 + v * 6.0)
                    noise = rng.randint(-4, 4)
                    base = 178 + int(low_freq) + noise
                    base = max(140, min(210, base))
                    img[y, x] = [base, base - 4, base - 12]
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img.astype(np.float32), sigma=(2, 0, 0))
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    return _make_texture_cached("building_wall_tex_v2", 512, 512, _build)


def gen_building_emissive_tex():
    """建筑自发光贴图 512x512：随机亮窗和霓虹灯效果"""
    def _build(w, h):
        size = 512
        rng = np.random.RandomState(20240119)
        img = np.zeros((size, size, 3), dtype=np.uint8)
        # 大部分区域不发光
        # 随机亮窗网格
        cell_w, cell_h = 26, 30
        for y in range(0, size, cell_h):
            for x in range(0, size, cell_w):
                if rng.random() < 0.35:
                    bright = rng.uniform(0.6, 1.0)
                    y0 = min(y + 4, size-5)
                    y1 = min(y + cell_h - 4, size)
                    x0 = min(x + 4, size-5)
                    x1 = min(x + cell_w - 4, size)
                    for dy in range(y0, y1):
                        for dx in range(x0, x1):
                            img[dy, dx] = [
                                min(255, int(255 * bright * rng.uniform(0.8, 1.2))),
                                min(255, int(235 * bright * rng.uniform(0.7, 1.1))),
                                min(255, int(180 * bright * rng.uniform(0.6, 1.0))),
                            ]
                # 偶尔有霓虹灯条
                if rng.random() < 0.05:
                    for dx in range(min(cell_w, size - x)):
                        img[min(y+2, size-1), x+dx] = [255, 120, 40]
                        img[min(y+cell_h-3, size-1), x+dx] = [255, 120, 40]
        return img
    return _make_texture_cached("building_emissive_tex", 512, 512, _build)


def gen_building_ao_tex():
    """建筑环境光遮蔽贴图 512x512：角落和缝隙更暗"""
    def _build(w, h):
        size = 512
        rng = np.random.RandomState(20240120)
        base = np.full((size, size, 3), 200, dtype=np.uint8)
        # 网格状暗角（模拟窗户凹槽、砖缝）
        cell_w, cell_h = 26, 30
        for y in range(0, size, cell_h):
            for x in range(0, size, cell_w):
                # 边框暗化（加边界保护）
                y_end = min(y + cell_h, size)
                x_end = min(x + cell_w, size)
                for dy in range(y, y_end):
                    for dx in range(x, x_end):
                        ly, lx = dy - y, dx - x
                        if lx < 4 or lx > cell_w-5 or ly < 4 or ly > cell_h-5:
                            base[dy, dx] = base[dy, dx] // 2
        # 随机污渍
        for _ in range(50):
            cx, cy = rng.randint(0, size, 2)
            r = rng.randint(10, 40)
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    if dx*dx + dy*dy <= r*r:
                        nx, ny = cx+dx, cy+dy
                        if 0 <= nx < size and 0 <= ny < size:
                            dark = int(40 * (1 - (dx*dx+dy*dy)/(r*r)))
                            base[ny, nx] = np.maximum(base[ny, nx].astype(np.int16) - dark, 60).astype(np.uint8)
        return base
    return _make_texture_cached("building_ao_tex", 512, 512, _build)


def gen_building_normal_tex():
    """建筑法线贴图 512x512：砖墙法线"""
    size = 512
    rng = np.random.RandomState(20240121)
    h_field = rng.randn(size, size).astype(np.float32)
    from scipy.ndimage import gaussian_filter
    h_field = gaussian_filter(h_field, sigma=1.5)
    # 砖块凹凸
    brick_h, brick_w = 20, 40
    for y in range(0, size, brick_h):
        offset = (y // brick_h) % 2 * (brick_w // 2)
        for x in range(-offset, size, brick_w):
            cy = min(y + brick_h//2, size-1)
            cx = min(max(0, x + brick_w//2), size-1)
            h_field[cy-3:cy+3, cx-5:cx+5] += 0.15
    return _make_texture_cached("building_normal_tex", size, size,
        lambda w_, h_: _height_to_normal(h_field, strength=2.0))


def gen_car_tex():
    """赛车纹理：白色底 + 灰色条纹，带缓存"""
    def _build(w, h):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                if (y // 18) % 2 == 1 and 30 < x < w - 30:
                    img[y, x] = [235, 235, 235]
                else:
                    v = 250 - (y % 18) * 1
                    img[y, x] = [v, v, v]
        return img
    return _make_texture_cached("car_tex", 512, 256, _build)

def gen_white_tex():
    """纯白纹理 2x2"""
    return make_texture(2, 2, lambda x,y,w,h: (255, 255, 255))


# ============================================================
# 地形 PBR 纹理生成（4 层：草/岩/土/碎石）
# ============================================================

# ★ 模块级地形纹理 build 函数（供 fallback_build_fn 使用）
def _build_grass_tex(w, h):
    """程序化生成草地纹理 512x512"""
    size = 512
    rng = np.random.RandomState(101)
    from scipy.ndimage import gaussian_filter
    base = np.full((size, size, 3), 0, dtype=np.float32)
    for scale, weight, color in [
        (8,  0.25, (0.20, 0.32, 0.12)),
        (32, 0.40, (0.26, 0.42, 0.15)),
        (128, 0.35, (0.32, 0.50, 0.20)),
    ]:
        noise = rng.randn(size//scale + 1, size//scale + 1).astype(np.float32)
        noise = np.kron(noise, np.ones((scale, scale), dtype=np.float32))[:size, :size]
        noise = gaussian_filter(noise, sigma=1.0)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
        for c in range(3):
            base[:, :, c] += color[c] * weight * noise
    base += rng.randn(size, size, 3).astype(np.float32) * 0.03
    return np.clip(base * 255, 0, 255).astype(np.uint8)

def _build_rock_tex(w, h):
    """程序化生成岩石纹理 512x512"""
    size = 512
    rng = np.random.RandomState(202)
    base = np.full((size, size, 3), 0.35, dtype=np.float32)
    for _ in range(20):
        length = rng.randint(80, 300)
        angle = rng.uniform(0, 2*np.pi)
        x, y = rng.randint(0, size), rng.randint(0, size)
        for t in range(length):
            angle += rng.normal(0, 0.15)
            px = int(x + t * np.cos(angle)) % size
            py = int(y + t * np.sin(angle)) % size
            base[max(0,py-1):py+2, max(0,px-1):px+2] -= 0.15
    base = np.clip(base, 0.15, 0.55)
    base += rng.randn(size, size, 1).astype(np.float32) * 0.05
    rgb = np.stack([base[:,:,0]*1.05, base[:,:,1]*1.0, base[:,:,2]*0.92], axis=-1)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)

def _build_dirt_tex(w, h):
    """程序化生成泥土纹理 512x512"""
    size = 512
    rng = np.random.RandomState(303)
    from scipy.ndimage import gaussian_filter
    base = np.full((size, size, 3), 0.28, dtype=np.float32)
    noise = rng.randn(size, size).astype(np.float32)
    noise = gaussian_filter(noise, sigma=1.5)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
    for c, tint in enumerate([1.10, 0.95, 0.78]):
        base[:, :, c] = base[:, :, c] * (0.7 + 0.5 * noise) * tint
    return np.clip(base * 255, 0, 255).astype(np.uint8)

def _build_gravel_tex(w, h):
    """程序化生成碎石纹理 512x512"""
    size = 512
    rng = np.random.RandomState(404)
    base = np.full((size, size, 3), 0.30, dtype=np.float32)
    for _ in range(3000):
        cx, cy = rng.randint(0, size, 2)
        r = rng.randint(2, 6)
        bright = rng.uniform(0.5, 1.2)
        y0 = max(0, cy-r); y1 = min(size, cy+r+1)
        x0 = max(0, cx-r); x1 = min(size, cx+r+1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask = (yy - cy)**2 + (xx - cx)**2 <= r*r
        for c, tint in enumerate([1.0, 0.98, 0.92]):
            base[y0:y1, x0:x1, c][mask] *= bright * tint
    return np.clip(base * 255, 0, 255).astype(np.uint8)


def gen_terrain_textures():
    """生成 4 张地形 PBR 纹理（草地/岩石/泥土/碎石），返回 (grass, rock, dirt, gravel) 纹理对象"""
    grass_tex  = _make_texture_cached("terrain_grass_v1",  512, 512, _build_grass_tex)
    rock_tex   = _make_texture_cached("terrain_rock_v1",   512, 512, _build_rock_tex)
    dirt_tex   = _make_texture_cached("terrain_dirt_v1",   512, 512, _build_dirt_tex)
    gravel_tex = _make_texture_cached("terrain_gravel_v1", 512, 512, _build_gravel_tex)
    return grass_tex, rock_tex, dirt_tex, gravel_tex


# ============================================================
# 法线贴图生成（三阶段：法线贴图）
# ============================================================
def _height_to_normal(h, strength=2.0):
    """高度场 → 法线贴图（Sobel 求导）"""
    from scipy.ndimage import gaussian_filter
    h = gaussian_filter(h, sigma=0.8)
    dx = np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)
    dy = np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)
    nx = -dx * strength
    ny = -dy * strength
    nz = np.ones_like(h)
    norm = np.sqrt(nx*nx + ny*ny + nz*nz) + 1e-6
    r = ((nx/norm + 1.0) * 0.5 * 255).astype(np.uint8)
    g = ((ny/norm + 1.0) * 0.5 * 255).astype(np.uint8)
    b = ((nz/norm + 1.0) * 0.5 * 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)

# ★ 模块级法线贴图 build 函数（供 fallback_build_fn 使用）
def _build_grass_nm(w, h):
    size = 512
    rng = np.random.RandomState(111)
    from scipy.ndimage import gaussian_filter
    h = rng.randn(size, size).astype(np.float32)
    h = gaussian_filter(h, sigma=1.2)
    return _height_to_normal(h, strength=0.8)

def _build_rock_nm(w, h):
    size = 512
    rng = np.random.RandomState(222)
    from scipy.ndimage import gaussian_filter
    h = rng.randn(size, size).astype(np.float32)
    h = gaussian_filter(h, sigma=3.0)
    return _height_to_normal(h, strength=3.0)

def _build_dirt_nm(w, h):
    size = 512
    rng = np.random.RandomState(333)
    from scipy.ndimage import gaussian_filter
    h = rng.randn(size, size).astype(np.float32)
    h = gaussian_filter(h, sigma=2.0)
    return _height_to_normal(h, strength=1.5)

def _build_gravel_nm(w, h):
    size = 512
    rng = np.random.RandomState(444)
    from scipy.ndimage import gaussian_filter
    h = rng.randn(size, size).astype(np.float32)
    h = gaussian_filter(h, sigma=1.5)
    return _height_to_normal(h, strength=2.5)


def gen_terrain_normal_maps():
    """生成 4 张法线贴图（与阶段二的 4 张 albedo 对应），返回 (grass_nm, rock_nm, dirt_nm, gravel_nm)"""
    size = 512

    # 草地：细密草叶
    rng = np.random.RandomState(111)
    h_grass = rng.randn(size, size).astype(np.float32)
    from scipy.ndimage import gaussian_filter
    h_grass = gaussian_filter(h_grass, sigma=1.2)
    grass_nm = _height_to_normal(h_grass, strength=0.8)

    # 岩石：粗裂块
    rng = np.random.RandomState(222)
    h_rock = rng.randn(size, size).astype(np.float32)
    h_rock = gaussian_filter(h_rock, sigma=3.0)
    rock_nm = _height_to_normal(h_rock, strength=3.0)

    # 泥土：微起伏
    rng = np.random.RandomState(333)
    h_dirt = rng.randn(size, size).astype(np.float32)
    h_dirt = gaussian_filter(h_dirt, sigma=2.0)
    dirt_nm = _height_to_normal(h_dirt, strength=1.5)

    # 碎石：高频小凸起
    rng = np.random.RandomState(444)
    h_gravel = rng.randn(size, size).astype(np.float32)
    h_gravel = gaussian_filter(h_gravel, sigma=1.5)
    gravel_nm = _height_to_normal(h_gravel, strength=2.5)

    def _save_cached(name, img):
        from PIL import Image
        path = _tex_cache_path(name)
        if path and os.path.exists(path):
            loaded = np.array(Image.open(path))
            if loaded.shape[:2] == (size, size):
                return loaded
        if path:
            Image.fromarray(img).save(path, optimize=True)
        return img

    grass_nm  = _save_cached("terrain_grass_nm_v1",  grass_nm)
    rock_nm   = _save_cached("terrain_rock_nm_v1",   rock_nm)
    dirt_nm   = _save_cached("terrain_dirt_nm_v1",   dirt_nm)
    gravel_nm = _save_cached("terrain_gravel_nm_v1", gravel_nm)

    def _upload(img):
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, size, size, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, img)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glGenerateMipmap(GL_TEXTURE_2D)
        return tex

    return _upload(grass_nm), _upload(rock_nm), _upload(dirt_nm), _upload(gravel_nm)


# ============================================================
# 着色器编译
# ============================================================
def compile_shader(source, stype):
    s = glCreateShader(stype)
    glShaderSource(s, source)
    glCompileShader(s)
    if not glGetShaderiv(s, GL_COMPILE_STATUS):
        print(f"着色器错误: {glGetShaderInfoLog(s).decode()}")
        glDeleteShader(s); return None
    return s

def create_shader_program(vsrc, fsrc):
    v, f = compile_shader(vsrc, GL_VERTEX_SHADER), compile_shader(fsrc, GL_FRAGMENT_SHADER)
    if not v or not f: return None
    p = glCreateProgram()
    glAttachShader(p, v); glAttachShader(p, f)
    glLinkProgram(p)
    if not glGetProgramiv(p, GL_LINK_STATUS):
        print(f"链接错误: {glGetProgramInfoLog(p).decode()}")
        glDeleteProgram(p); return None
    glDeleteShader(v); glDeleteShader(f)
    return p


# ============================================================
# Compute Shader 基础设施
# ============================================================
def create_compute_program(csrc):
    """编译 Compute Shader 并链接为独立程序"""
    c = compile_shader(csrc, GL_COMPUTE_SHADER)
    if not c: return None
    p = glCreateProgram()
    glAttachShader(p, c)
    glLinkProgram(p)
    if not glGetProgramiv(p, GL_LINK_STATUS):
        print(f"Compute Shader 链接错误: {glGetProgramInfoLog(p).decode()}")
        glDeleteProgram(p); return None
    glDeleteShader(c)
    return p


def create_ssbo(data: np.ndarray, binding: int, usage=GL_DYNAMIC_DRAW):
    """创建 SSBO 并绑定到指定 binding point

    返回 (ssbo_id, size_bytes)
    data: numpy 数组，dtype 必须是 float32/uint32/int32
    """
    ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data, usage)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, ssbo)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    return ssbo, data.nbytes


def ssbo_sub_data(ssbo: int, data: np.ndarray, offset_bytes=0):
    """向 SSBO 写入子范围数据（glBufferSubData 封装）"""
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset_bytes, data.nbytes, data)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)


def ssbo_read(ssbo: int, count: int, dtype=np.uint8, offset_bytes=0):
    """从 SSBO 读取数据到 numpy 数组（调试用）
    count: 元素数量, dtype: 元素类型, 返回 (count,) 的 ndarray
    """
    byte_size = count * np.dtype(dtype).itemsize
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
    ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, offset_bytes, byte_size, GL_MAP_READ_BIT)
    if ptr:
        raw = ctypes.string_at(ptr, byte_size)
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
        # ★ np.frombuffer 对 bytes 返回只读视图，.copy() 确保可写
        data = np.frombuffer(raw, dtype=dtype).copy()
    else:
        data = np.zeros(count, dtype=dtype)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    return data


# ============================================================
# 视锥剔除辅助函数
# ============================================================
def extract_frustum_planes(vp: np.ndarray):
    """从 view-projection 矩阵提取 6 个视锥平面（Gribb-Hartmann 方法）

    vp 是行主序矩阵（NumPy 默认），平面必须从"行"构造。
    """
    planes = np.empty((6, 4), dtype=np.float32)
    planes[0] = vp[3] + vp[0]  # left
    planes[1] = vp[3] - vp[0]  # right
    planes[2] = vp[3] + vp[1]  # bottom
    planes[3] = vp[3] - vp[1]  # top
    planes[4] = vp[3] + vp[2]  # near
    planes[5] = vp[3] - vp[2]  # far
    # 归一化
    for i in range(6):
        n = np.linalg.norm(planes[i, :3])
        if n > 1e-8:
            planes[i] /= n
    return planes

def compute_sh_coeffs(light_dir, light_color, ambient_col):
    """从方向光 + 环境色计算 L0+L1 SH 球谐系数（9 x vec3）

    L00 承担主要均匀环境光，L1 只贡献方向性增量。
    返回适合 shIrradiance() 重建的系数数组（shape=9x3）
    """
    coeffs = np.zeros((9, 3), dtype=np.float32)
    # L00 (DC)：均匀环境辐照度（占主导）
    coeffs[0] = ambient_col * 2.5
    # L1：方向性贡献（太阳本身，降低强度避免垂直面过暗）
    coeffs[1] = light_color * light_dir[1] * 0.5  # Y
    coeffs[2] = light_color * light_dir[2] * 0.5  # Z
    coeffs[3] = light_color * light_dir[0] * 0.5  # X
    # L2（全零，完整3阶需要更多信息）
    return coeffs

def frustum_test_sphere(planes, center, radius):
    """球体 - 视锥相交测试"""
    for i in range(6):
        d = planes[i, 0] * center[0] + planes[i, 1] * center[1] + planes[i, 2] * center[2] + planes[i, 3]
        if d < -radius:
            return False
    return True


# ============================================================
# RX-7 GLB 加载：按材质分组 + 自检轴 + 朝向开关
# ============================================================
CAR_YAW_180 = True    # ★ 车头朝后就开这个（你的现状）
CAR_FLIP_UP = False   # ★ 只有车底朝天才开（目前不是）

_GLB_CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
           5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_GLB_NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}

def _glb_read_accessor(js, bin_data, idx):
    acc = js['accessors'][idx]
    view = js['bufferViews'][acc.get('bufferView', 0)]
    ncomp = _GLB_NC[acc['type']]
    dt = _GLB_CT[acc['componentType']]
    start = view.get('byteOffset', 0) + acc.get('byteOffset', 0)
    count = acc['count']
    itemsize = np.dtype(dt).itemsize
    stride = view.get('byteStride') or itemsize * ncomp
    if stride == itemsize * ncomp:
        return np.frombuffer(bin_data, dtype=dt, count=count * ncomp,
                             offset=start).reshape(count, ncomp).copy()
    out = np.empty((count, ncomp), dtype=dt)
    for i in range(count):
        out[i] = np.frombuffer(bin_data, dtype=dt, count=ncomp,
                               offset=start + i * stride)
    return out

def _glb_node_matrix(n):
    if 'matrix' in n:
        return np.array(n['matrix'], dtype=np.float64).reshape(4, 4).T  # glTF 列主序
    M = np.eye(4, dtype=np.float64)
    x, y, z, w = n.get('rotation', [0.0, 0.0, 0.0, 1.0])
    M[:3, :3] = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]], dtype=np.float64)
    M[:3, :3] *= np.array(n.get('scale', [1.0, 1.0, 1.0]), dtype=np.float64)
    M[:3, 3] = np.array(n.get('translation', [0.0, 0.0, 0.0]), dtype=np.float64)
    return M


def _load_glb_groups(glb_path, scale):
    """手工解析 GLB：每个 primitive 一组，材质/贴图严格配对（与 Blender 一致）"""
    from PIL import Image as _PILImage
    with open(glb_path, 'rb') as fp:
        data = fp.read()
    magic, _ver, total = _struct.unpack_from('<III', data, 0)
    assert magic == 0x46546C67, "不是 GLB 文件"
    off, js, bin_data = 12, None, None
    while off < total:
        clen, ctype = _struct.unpack_from('<II', data, off)
        chunk = data[off+8: off+8+clen]
        if ctype == 0x4E4F534A:    js = _json.loads(chunk.decode('utf-8'))   # JSON
        elif ctype == 0x004E4942:  bin_data = chunk                          # BIN
        off += 8 + clen
    g = js

    # ---- 图片 ----
    images = []
    for im in g.get('images', []):
        img = None
        try:
            if 'bufferView' in im:
                bv = g['bufferViews'][im['bufferView']]
                raw = bin_data[bv.get('byteOffset', 0):
                               bv.get('byteOffset', 0) + bv['byteLength']]
                img = _PILImage.open(_io.BytesIO(raw))
            elif 'uri' in im:
                if im['uri'].startswith('data:'):
                    img = _PILImage.open(_io.BytesIO(
                        _b64.b64decode(im['uri'].split(',', 1)[1])))
                else:
                    img = _PILImage.open(os.path.join(os.path.dirname(glb_path), im['uri']))
        except Exception:
            pass
        images.append(img)

    # ---- 材质 -> (贴图, baseColorFactor, emissiveFactor) ----
    def mat_info(mi):
        if mi is None:
            return None, (1.0, 1.0, 1.0), None
        m = g['materials'][mi]
        pbr = m.get('pbrMetallicRoughness', {})
        ti = pbr.get('baseColorTexture', {}).get('index')
        img = None
        if ti is not None:
            src = g['textures'][ti].get('source')
            if src is not None and src < len(images):
                img = images[src]
        bcf = pbr.get('baseColorFactor', [1.0, 1.0, 1.0, 1.0])
        return img, (float(bcf[0]), float(bcf[1]), float(bcf[2])), m.get('emissiveFactor')

    # ---- 遍历场景图，组合节点变换 ----
    prims = []   # (世界矩阵, mesh索引, prim索引)
    def walk(ni, parent_M):
        n = g['nodes'][ni]
        M = parent_M @ _glb_node_matrix(n)
        if 'mesh' in n:
            for pi in range(len(g['meshes'][n['mesh']]['primitives'])):
                prims.append((M, n['mesh'], pi))
        for c in n.get('children', []):
            walk(c, M)
    for root in g['scenes'][g.get('scene', 0)]['nodes']:
        walk(root, np.eye(4, dtype=np.float64))

    # ---- 逐 primitive 展开为三角形汤 ----
    groups = []
    for M, mesh_i, p_i in prims:
        prim = g['meshes'][mesh_i]['primitives'][p_i]
        attrs = prim.get('attributes', {})
        if 'POSITION' not in attrs:
            continue
        P = _glb_read_accessor(g, bin_data, attrs['POSITION']).astype(np.float64)
        idx_data = (_glb_read_accessor(g, bin_data, prim['indices']).reshape(-1)
                    if 'indices' in prim else np.arange(len(P), dtype=np.int64))
        if 'NORMAL' in attrs:
            N = _glb_read_accessor(g, bin_data, attrs['NORMAL']).astype(np.float64)
        else:
            tri = P[idx_data].reshape(-1, 3, 3)
            fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
            N = np.repeat(fn, 3, axis=0)
        UV = (_glb_read_accessor(g, bin_data, attrs['TEXCOORD_0']).astype(np.float32)
              if 'TEXCOORD_0' in attrs else np.zeros((len(P), 2), np.float32))

        R = M[:3, :3]
        v  = (P[idx_data] @ R.T + M[:3, 3]) * scale
        nn = N[idx_data] @ R.T
        uu = UV[idx_data].astype(np.float32)
        img, bcf, emis = mat_info(prim.get('material'))
        groups.append({
            'name': g['meshes'][mesh_i].get('name', 'mesh') + '_' + str(p_i),
            'raw': (v.astype(np.float32), nn.astype(np.float32), uu),
            'img': img, 'color': bcf, 'emissive': emis,
        })
    print(f"  [GLB] 解析到 {len(groups)} 个 primitive 组")
    return groups

_glb_tex_cache = {}
def _upload_glb_tex(img):
    if img is None:
        return 0
    key = id(img)
    if key in _glb_tex_cache:
        return _glb_tex_cache[key]
    arr = np.array(img.convert("RGB"), dtype=np.uint8)   # ★ 不再 FLIP_TOP_BOTTOM！
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, arr.shape[1], arr.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, arr)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glGenerateMipmap(GL_TEXTURE_2D)
    _glb_tex_cache[key] = tex
    return tex

def _map_group(raw, up_idx, up_min, up_max):
    v, n, uv = raw
    if up_idx == 1:
        gx, gy, gz = v[:,0].copy(), v[:,1]-up_min, v[:,2].copy()
        nx, ny, nz = n[:,0].copy(), n[:,1].copy(), n[:,2].copy()
    else:
        gx, gy, gz = v[:,0].copy(), v[:,2]-up_min, -v[:,1]
        nx, ny, nz = n[:,0].copy(), n[:,2].copy(), -n[:,1]
    if CAR_FLIP_UP:                     # 沿上轴翻转（含法线）
        gy = (up_max - up_min) - gy
        ny = -ny
    if CAR_YAW_180:                     # 绕 Y 转 180°：x、z 同时取负（防镜像）
        gx, gz = -gx, -gz
        nx, nz = -nx, -nz
    ln = np.sqrt(nx*nx + ny*ny + nz*nz)
    inv = 1.0 / np.where(ln > 1e-8, ln, 1.0)
    return gx, gy, gz, nx*inv, ny*inv, nz*inv, uv

CAR_TARGET_WIDTH = 2.6
_car_scale_factor = 1.0

def fix_car_winding(v):
    """按'外向评分'统一三角形绕序（从车外看全部逆时针），
    并输出与绕序一致的平直法线。返回 (v_fixed, n_fixed)。"""
    tri = v.reshape(-1, 3, 3)
    fn = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    flen = np.linalg.norm(fn, axis=1, keepdims=True)
    area = 0.5 * flen[:, 0]
    fn = fn / np.maximum(flen, 1e-12)
    center = v.mean(axis=0)
    outward = tri.mean(axis=1) - center  # 各三角形中心指向外的方向
    outward /= np.maximum(np.linalg.norm(outward, axis=1, keepdims=True), 1e-9)
    d = (fn * outward).sum(axis=1)
    # 全局多数决：整辆车若整体反向，先整体翻转
    if area[d < 0].sum() > area[d > 0].sum():
        tri = tri[:, [0, 2, 1]]
        d = -d
    # 个别凹面（轮拱内侧等）局部翻转
    flip = d < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    fn2 = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    fn2 /= np.maximum(np.linalg.norm(fn2, axis=1, keepdims=True), 1e-12)
    return (tri.reshape(-1, 3).astype(np.float32),
            np.repeat(fn2, 3, axis=0).astype(np.float32))

def build_car_groups():
    global _car_scale_factor
    # 依次尝试存在的车模文件（备份名优先，回退到实际文件）
    _glb = next((p for p in ("RX7_backup_before_edit.glb", "RX-7.glb", "RX7.glb")
                 if os.path.exists(p)), "RX-7.glb")
    print(f"  [GLB] 使用车模文件: {_glb}")
    groups = _load_glb_groups(_glb, 1.77)
    allv = np.vstack([g['raw'][0] for g in groups])
    ext = allv.max(0) - allv.min(0)
    up = int(np.argmin(ext))
    up_min, up_max = float(allv[:,up].min()), float(allv[:,up].max())

    # ---- 第一遍：修绕序 + 映射，量车宽 -> 算缩放因子 ----
    mapped = []
    for g in groups:
        v0, n0, uv0 = g['raw']
        v0, n0 = fix_car_winding(v0)
        mapped.append((_map_group((v0, n0, uv0), up, up_min, up_max), g))
    xs_all = np.concatenate([m[0][0] for m in mapped])
    cur_w = float(xs_all.max() - xs_all.min())
    _car_scale_factor = CAR_TARGET_WIDTH / max(cur_w, 1e-6) * 0.612  # ★ 整体缩小15%再缩20% → 0.68x；再缩10%→0.612x
    print(f"  [GLB] 车宽 {cur_w:.2f} -> 目标 {CAR_TARGET_WIDTH} (x{_car_scale_factor:.2f})")

    out = []
    for (gx, gy, gz, nx, ny, nz, uv), g in mapped:
        # 均匀缩放。gy 原点已在车底（减过 up_min），缩放后车轮仍贴地
        gx *= _car_scale_factor; gy *= _car_scale_factor; gz *= _car_scale_factor
        if g['emissive'] is not None and max(g['emissive']) > 0.05:
            col = np.tile(np.array([1.0, 0.05, 0.05], np.float32), (len(gx), 1))
        else:
            col = np.tile(np.array(g['color'], np.float32), (len(gx), 1))
        verts = np.column_stack([gx, gy, gz, nx, ny, nz, uv, col]).astype(np.float32)
        vao, vbo = create_vao(verts, [3,3,2,3])
        out.append({'vao': vao, 'count': len(verts),
                    'tex': _upload_glb_tex(g['img']),
                    'has_tex': 1.0 if g['img'] is not None else 0.0,
                    'is_light': (g['emissive'] is not None and max(g['emissive']) > 0.05)
                                or any(k in g['name'].lower() for k in ('light','lamp','tail'))})
    print(f"  [GLB] 材质组: {len(out)}, 总顶点 {sum(o['count'] for o in out):,}")
    return out

# ============================================================
# 赛道构建由 mountain.py 提供 (build_mountain_road)
# 11点横截面 + 地形色过渡 + Banking
# ============================================================

# ============================================================
# 护栏构建由 mountain.py 提供 (build_guardrails)
# ============================================================

# ============================================================
# 杉树构建由 mountain.py 提供 (build_akina_trees)
# 生态式分布：赛道两侧密集 + 背景林 + 灌木
# ============================================================

# ============================================================
# 地形构建由 mountain.py 提供 (build_akina_terrain)
# 整块规则网格 + signed_distance 精准压平赛道区域
# ============================================================

# ============================================================
# 建筑群构建（带法线 + UV）
# ============================================================
def build_sky_sphere(radius=1500.0, seg=32, rings=16):
    """生成天空球几何（从内部看，逆时针绕序）"""
    verts = []
    for j in range(rings):
        t1 = math.pi * j / rings
        t2 = math.pi * (j + 1) / rings
        for i in range(seg):
            p1 = 2 * math.pi * i / seg
            p2 = 2 * math.pi * (i + 1) / seg
            def pt(t, p):
                return (radius*math.sin(t)*math.cos(p),
                        radius*math.cos(t),
                        radius*math.sin(t)*math.sin(p))
            a1, a2 = pt(t1, p1), pt(t1, p2)
            a3, a4 = pt(t2, p2), pt(t2, p1)
            verts.extend(a1); verts.extend(a2); verts.extend(a3)
            verts.extend(a1); verts.extend(a3); verts.extend(a4)
    return np.array(verts, dtype=np.float32)


# ============================================================
# 树木构建
# ============================================================
def build_trees():
    """25棵树，树干+树冠"""
    verts = []
    def tri(p1,p2,p3,n,uv,c):
        verts.append([*p1,*n,*uv[0],*c]); verts.append([*p2,*n,*uv[1],*c]); verts.append([*p3,*n,*uv[2],*c])
    def quad(p1,p2,p3,p4,n,uv,c):
        tri(p1,p2,p3,n,[uv[0],uv[1],uv[2]],c); tri(p1,p3,p4,n,[uv[0],uv[2],uv[3]],c)
    def box(cx,cy,cz,w,h,d,cols):
        x0,x1=cx-w/2,cx+w/2; y0,y1=cy,cy+h; z0,z1=cz-d/2,cz+d/2
        quad((x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1),(0,1,0),[(0,0),(1,0),(1,1),(0,1)],cols[0])
        quad((x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1),(0,-1,0),[(0,0),(1,0),(1,1),(0,1)],cols[1])
        quad((x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1),(0,0,1),[(0,1),(1,1),(1,0),(0,0)],cols[2])
        quad((x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),(0,0,-1),[(0,1),(1,1),(1,0),(0,0)],cols[3])
        quad((x0,y0,z0),(x0,y0,z1),(x0,y1,z1),(x0,y1,z0),(-1,0,0),[(0,1),(1,1),(1,0),(0,0)],cols[4])
        quad((x1,y0,z0),(x1,y0,z1),(x1,y1,z1),(x1,y1,z0),(1,0,0),[(0,1),(1,1),(1,0),(0,0)],cols[5])

    rng=random.Random(24)
    for _ in range(25):
        a=rng.uniform(0,2*math.pi); d=rng.uniform(2,22)
        tx, tz = d*math.cos(a), d*math.sin(a)
        th=rng.uniform(1.0,1.8)
        box(tx,0,tz,0.12,th,0.12,
            ((0.40,0.22,0.10),(0.30,0.15,0.05),(0.40,0.22,0.10),
             (0.30,0.15,0.05),(0.40,0.22,0.10),(0.30,0.15,0.05)))
        cr=rng.uniform(0.5,1.4); ch=rng.uniform(1.0,2.0)
        lc=(rng.uniform(0.08,0.20),rng.uniform(0.30,0.50),rng.uniform(0.05,0.12))
        llc=tuple(min(1.0,c*1.4) for c in lc)
        for i in range(10):
            a1=2*math.pi*i/10; a2=2*math.pi*(i+1)/10
            bx1=tx+cr*math.cos(a1); bz1=tz+cr*math.sin(a1)
            bx2=tx+cr*math.cos(a2); bz2=tz+cr*math.sin(a2)
            dx1,dz1=bx1-tx,bz1-tz; dx2,dz2=bx2-tx,bz2-tz
            nn = np.array([dz1-dz2, cr*0.5, dx2-dx1], dtype=f32)
            nn = nn / np.linalg.norm(nn) if np.linalg.norm(nn)>0 else (0,1,0)
            tri((tx,th+ch,tz),(bx1,th,bz1),(bx2,th,bz2), tuple(nn),[(0,0),(0,0),(0,0)],lc)
            tri((tx,th,tz),(bx2,th,bz2),(bx1,th,bz1), (0,-1,0),[(0,0),(0,0),(0,0)],llc)
    return np.array(verts, dtype=np.float32)

# ============================================================
# 路灯构建由 mountain.py 提供 (build_lights, build_lamp_emissive, build_lamp_ground_pool)
# 自发光灯罩 + 地面光池 + 法线光照灯杆
# ============================================================

# ============================================================
# 云朵构建
# ============================================================
def build_clouds():
    """返回 [(顶点数组, 中心x, 中心z), ...] 每朵云独立，支持动态飘移"""
    def make_cloud(cx,cy,cz,rng):
        """生成一朵云（3~5个扁球体组合）"""
        verts=[]
        def tri(p1,p2,p3,n,uv,c):
            verts.append([*p1,*n,*uv[0],*c]); verts.append([*p2,*n,*uv[1],*c]); verts.append([*p3,*n,*uv[2],*c])
        def quad(p1,p2,p3,p4,n,uv,c):
            tri(p1,p2,p3,n,[uv[0],uv[1],uv[2]],c); tri(p1,p3,p4,n,[uv[0],uv[2],uv[3]],c)
        def add_sphere(cx,cy,cz,rx,ry,rz,col):
            seg=10
            for j in range(seg):
                t1=math.pi*j/seg; t2=math.pi*(j+1)/seg
                for i in range(seg):
                    p1=2*math.pi*i/seg; p2=2*math.pi*(i+1)/seg
                    def pt(t,p): return (rx*math.sin(t)*math.cos(p)+cx, ry*math.cos(t)+cy, rz*math.sin(t)*math.sin(p)+cz)
                    def nm(t,p): return (math.sin(t)*math.cos(p), math.cos(t), math.sin(t)*math.sin(p))
                    a1=pt(t1,p1); a2=pt(t1,p2); a3=pt(t2,p2); a4=pt(t2,p1)
                    na=nm(t1,p1); nb=nm(t1,p2); nc=nm(t2,p2); nd=nm(t2,p1)
                    quad(a1,a2,a3,a4,na,[(0,0),(0,0),(0,0),(0,0)],col)
        for _ in range(rng.randint(3,5)):
            ox=rng.uniform(-3.5,3.5); oz=rng.uniform(-2.5,2.5); oy=rng.uniform(-1.5,1.5)
            rr=rng.uniform(0.8,2.2); br=rng.uniform(0.85,1.0)
            add_sphere(cx+ox,cy+oy,cz+oz, rr, rr*0.30, rr*0.65, (br,br,br*0.98))
        return np.array(verts, dtype=np.float32)
    rng=random.Random(37); clouds=[]
    for _ in range(12):
        a=rng.uniform(0,2*math.pi); d=rng.uniform(10,45)
        cx,cz=d*math.cos(a),d*math.sin(a); cy=rng.uniform(8,18)
        clouds.append((make_cloud(cx,cy,cz,rng), cx, cz))
    return clouds

def build_sun():
    """返回 (核心球体顶点, 光晕盘顶点) 两个独立网格"""
    def mesh():
        verts=[]
        def tri(p1,p2,p3,n,uv,c):
            verts.append([*p1,*n,*uv[0],*c]); verts.append([*p2,*n,*uv[1],*c]); verts.append([*p3,*n,*uv[2],*c])
        def quad(p1,p2,p3,p4,n,uv,c):
            tri(p1,p2,p3,n,[uv[0],uv[1],uv[2]],c); tri(p1,p3,p4,n,[uv[0],uv[2],uv[3]],c)
        return verts, tri, quad

    cx,cy,cz=30,18,-35; core_col=(1.0,0.70,0.30)
    # --- 核心球体 ---
    sv,stri,squad=mesh(); seg=12; sr=1.5
    for j in range(seg):
        t1=math.pi*j/seg; t2=math.pi*(j+1)/seg
        for i in range(seg):
            p1=2*math.pi*i/seg; p2=2*math.pi*(i+1)/seg
            def spt(t,p): return (sr*math.sin(t)*math.cos(p)+cx, sr*math.cos(t)+cy, sr*math.sin(t)*math.sin(p)+cz)
            def snm(t,p): return (math.sin(t)*math.cos(p), math.cos(t), math.sin(t)*math.sin(p))
            a1=spt(t1,p1); a2=spt(t1,p2); a3=spt(t2,p2); a4=spt(t2,p1)
            na=snm(t1,p1); nb=snm(t1,p2); nc=snm(t2,p2); nd=snm(t2,p1)
            squad(a1,a2,a3,a4,na,[(0,0),(0,0),(0,0),(0,0)],core_col)
    sphere_verts=np.array(sv,dtype=np.float32)
    # --- 光晕盘 ---
    hv,htri,hquad=mesh(); hr=7.0; seg2=24; n_down=(0,-1,0); white=(1,1,1)
    for i in range(seg2):
        a1=2*math.pi*i/seg2; a2=2*math.pi*(i+1)/seg2
        ti=(cx,cy,cz); to1=(cx+hr*math.cos(a1),cy,cz+hr*math.sin(a1)); to2=(cx+hr*math.cos(a2),cy,cz+hr*math.sin(a2))
        uv_i=(0.5,0.5); uv_o1=(0.5+0.5*math.cos(a1),0.5+0.5*math.sin(a1)); uv_o2=(0.5+0.5*math.cos(a2),0.5+0.5*math.sin(a2))
        htri(ti,to1,to2,n_down,[uv_i,uv_o1,uv_o2],white)
    halo_verts=np.array(hv,dtype=np.float32)
    return sphere_verts, halo_verts

def gen_sun_tex():
    """太阳光晕径向渐变纹理 256x256"""
    size=256; cx=cy=size/2; img=np.zeros((size,size,3),dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            d=math.sqrt((x-cx)**2+(y-cy)**2)/(size/2)
            if d<1.0:
                if d<0.2: r,g,b=255,248,210
                elif d<0.5: t=(d-0.2)/0.3; r=int(255*(1-t*0.08)); g=int(248*(1-t*0.15)); b=int(210*(1-t*0.35))
                else: t=(d-0.5)/0.5; r=int(235*(1-t*0.35)); g=int(210*(1-t*0.5)); b=int(136*(1-t*0.7))
            else: r=g=b=0
            img[y,x]=[max(0,min(255,r)),max(0,min(255,g)),max(0,min(255,b))]
    tex=glGenTextures(1); glBindTexture(GL_TEXTURE_2D,tex)
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,size,size,0,GL_RGB,GL_UNSIGNED_BYTE,img)
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE); return tex

# ============================================================
# VAO 创建
# ============================================================
def create_vao(vertices, attrib_sizes, stride_override=None):
    """创建 VAO/VBO，顶点格式支持位置+法线+UV+颜色

    stride_override: 顶点跨距（字节）。用于顶点数组里还有额外维度
    （如树木的切线，用 loc 8 单独绑定）的情况。
    """
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    stride = int(stride_override) if stride_override else sum(attrib_sizes) * 4
    offset = 0
    for i, size in enumerate(attrib_sizes):
        glVertexAttribPointer(i, size, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(i)
        offset += size * 4
    glBindVertexArray(0)
    return vao, vbo


def _make_foliage_mips(rgb, alpha, min_size=4):
    """
    叶片图集的自定义 mip 链：**alpha 加权平均**，白边/暗茎不再稀释叶色。

    为什么不用 glGenerateMipmap：普通 mip 会把剪影 alpha 平均掉，
    远处的树在 alpha test 下会逐渐"溶解"成一团团半透明的雾。
    取最大值可以保住覆盖率，远处直接读成更密的冠体（这是 3A 的标准做法）。

    升级为 alpha 加权平均：只有有叶的像素参与颜色平均，木质/树皮像素按覆盖率
    透明化，不再将白色树皮带或深色透明区的颜色渗进远处叶色。
    """
    levels = [(rgb.astype(np.float32), alpha.astype(np.float32))]
    r, a = levels[0]
    while r.shape[0] > min_size and r.shape[1] > min_size:
        h, w_ = max(1, r.shape[0] // 2), max(1, r.shape[1] // 2)
        r2 = r[:h * 2, :w_ * 2].reshape(h, 2, w_, 2, 3)
        a2 = a[:h * 2, :w_ * 2].reshape(h, 2, w_, 2)
        wsum = a2.sum(axis=(1, 3)) + 1e-4  # (h,w)，不用 keepdims 防广播维度膨大
        r = (r2 * a2[..., None]).sum(axis=(1, 3)) / wsum[..., None]  # ★ alpha 加权，白边/暗茎不再稀释叶色
        a = a2.max(axis=(1, 3))                             # alpha 仍取最大值保覆盖率
        levels.append((r, a))
    return levels


def _make_normal_mips(nrm_rgba, min_size=4):
    """法线贴图的 mip 链：解码→平均→重新归一化→编码（避免平均后法线变短）"""
    levels = [nrm_rgba.astype(np.float32)]
    x = levels[0]
    while x.shape[0] > min_size and x.shape[1] > min_size:
        h, w = max(1, x.shape[0] // 2), max(1, x.shape[1] // 2)
        b = x[:h * 2, :w * 2].reshape(h, 2, w, 2, 4).mean(axis=(1, 3))
        n = b[..., :3] * 2.0 - 1.0
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
        b[..., :3] = n * 0.5 + 0.5
        levels.append(b)
        x = b
    return levels


def create_vao_packed_terrain(terrain):
    """
    为打包地形顶点数据创建 VAO + 5 个独立 VBO。
    与 create_vao 不同，这里处理 mountain.py 输出的打包格式：
      - positions: (N, 3) float32         → 属性 0  vec3
      - normals:   (N,)   int32 打包       → 属性 1  GL_INT_2_10_10_10_REV (4 分量)
      - uvs:       (N,)   uint32 打包      → 属性 2  2×GL_HALF_FLOAT
      - colors:    (N,)   uint32 打包      → 属性 3  RGBA u8 → 着色器只取 RGB
      - weights:   (N,)   uint32 打包      → 属性 4  RGBA u8 → vec4

    返回: (vao, buffers_list, vertex_count)
    """
    positions  = terrain['positions']
    normals_pk = terrain['normals']
    colors_pk  = terrain['colors']
    weights_pk = terrain['weights']
    uvs_pk     = terrain['uvs']
    N = len(positions)

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    buffers = []

    # ---- 属性 0：位置 (vec3 float32) ----
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    buffers.append(vbo)

    # ---- 属性 1：法线 (GL_INT_2_10_10_10_REV, 归一化) ----
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, normals_pk.nbytes, normals_pk, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 4, GL_INT_2_10_10_10_REV, GL_TRUE,
                          0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(1)
    buffers.append(vbo)

    # ---- 属性 2：UV (2×GL_HALF_FLOAT，一个 uint32 内两个半浮点) ----
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, uvs_pk.nbytes, uvs_pk, GL_STATIC_DRAW)
    glVertexAttribPointer(2, 2, GL_HALF_FLOAT, GL_FALSE,
                          0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(2)
    buffers.append(vbo)

    # ---- 属性 3：颜色 (RGBA u8，着色器取 RGB，stride=4 跳过 A) ----
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, colors_pk.nbytes, colors_pk, GL_STATIC_DRAW)
    glVertexAttribPointer(3, 3, GL_UNSIGNED_BYTE, GL_TRUE,
                          4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(3)
    buffers.append(vbo)

    # ---- 属性 4：材质权重 (RGBA u8 → vec4) ----
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, weights_pk.nbytes, weights_pk, GL_STATIC_DRAW)
    glVertexAttribPointer(4, 4, GL_UNSIGNED_BYTE, GL_TRUE,
                          4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(4)
    buffers.append(vbo)

    glBindVertexArray(0)
    return vao, buffers, N


# ============================================================
# 主函数
# ============================================================
# ---- ★ 调试：导出某个帧的 G-Buffer（GB_DEBUG=1 时启用，用于离线诊断）----
# 为什么要有：延迟渲染下"物体渲染得很怪"必须先分开确认是
#   ①几何 Pass 里的 albedo/法线就错了   还是
#   ②光照 Pass 把这些数据用错了。
# 直接看 G-Buffer 就能定位，不用猜。
_GB_DEBUG = os.environ.get('GB_DEBUG') == '1'
_GB_DEBUG_FRAME = int(os.environ.get('GB_DEBUG_FRAME', '420'))
_CAR_HIDE = os.environ.get('CAR_HIDE') == '1'   # 调试：整辆车不画
_gb_dbg_state = {'n': 0, 'done': False}
_gb_car_state = {'n': 0, 'done': False}


def _gb_debug_car_normals(fbo, ww, wh):
    """在"车绘制紧后"读一次 G-Buffer。用来区分两件事：
       ① 车自己就画出了错法线；② 车画对了、但被后面的绘制覆盖。
       判据：同一帧里，"紧后"与"帧末"两次读数的 N.y 是否一致。"""
    st = _gb_car_state
    st['n'] += 1
    if st['done'] or st['n'] != _GB_DEBUG_FRAME:
        return
    st['done'] = True
    prev = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glReadBuffer(GL_COLOR_ATTACHMENT1)
    nrm = np.frombuffer(glReadPixels(0, 0, ww, wh, GL_RGBA, GL_FLOAT),
                        dtype=np.float32).reshape(wh, ww, 4)[::-1]
    glReadBuffer(GL_COLOR_ATTACHMENT3)
    mat = np.frombuffer(glReadPixels(0, 0, ww, wh, GL_RGBA, GL_FLOAT),
                        dtype=np.float32).reshape(wh, ww, 4)[::-1]
    glReadBuffer(GL_COLOR_ATTACHMENT0)
    glBindFramebuffer(GL_FRAMEBUFFER, prev)
    r = mat[..., 0]
    m = (r > 0.25) & (r < 0.36)
    nn = nrm[..., :3] * 2.0 - 1.0
    if m.sum() > 100:
        print(f"[GBDEBUG2] 车绘制紧后: 车像素 {100.0*m.mean():.2f}%  "
              f"N.y={nn[...,1][m].mean():+.3f}  "
              f"N.y<0={100.0*(nn[...,1][m]<0).mean():.1f}%", flush=True)
    else:
        print(f"[GBDEBUG2] 车绘制紧后: 无 r∈[0.25,0.36) 像素 "
              f"({int(m.sum())} 个)", flush=True)


def _gb_debug_dump(fbo, ww, wh):
    st = _gb_dbg_state
    st['n'] += 1
    if st['done'] or st['n'] != _GB_DEBUG_FRAME:
        return
    st['done'] = True
    _dir = os.path.dirname(os.path.abspath(__file__))
    prev_fbo = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
    prev_rb = glGetIntegerv(GL_READ_BUFFER)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    for _i, _nm in ((1, 'normal'), (2, 'albedo'), (3, 'material')):
        glReadBuffer(GL_COLOR_ATTACHMENT0 + _i)
        _raw = glReadPixels(0, 0, ww, wh, GL_RGBA, GL_FLOAT)
        _a = np.frombuffer(_raw, dtype=np.float32).reshape(wh, ww, 4)[::-1]
        Image.fromarray((np.clip(_a[:, :, :3], 0, 1) * 255
                         ).astype(np.uint8)).save(
            os.path.join(_dir, f'_dbg_gb_{_nm}.png'))
        print(f"[GBDEBUG] {_nm}: RGB 均值="
              f"({_a[:,:,0].mean():.3f},{_a[:,:,1].mean():.3f},"
              f"{_a[:,:,2].mean():.3f}) A均值={_a[:,:,3].mean():.3f}")
    glReadBuffer(GL_COLOR_ATTACHMENT0)
    _dz = np.frombuffer(glReadPixels(0, 0, ww, wh, GL_DEPTH_COMPONENT, GL_FLOAT),
                        dtype=np.float32).reshape(wh, ww)[::-1]
    Image.fromarray((np.clip(_dz, 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(_dir, '_dbg_gb_depth.png'))
    glReadBuffer(prev_rb)
    glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)
    print(f"[GBDEBUG] depth min={_dz.min():.4f} max={_dz.max():.4f} "
          f"（已导出到项目根 _dbg_gb_*.png）", flush=True)


def main():
    # 清空地形成缓存（在 main() 内部调用，避免模块导入时执行）
    clear_terrain_cache()

    # 清理旧版 .npy 纹理缓存（新代码改用 .png，旧文件已无用）
    # 注意：只删 *_tex.npy 纹理缓存，绝不碰地形/树木等几何缓存
    if os.path.exists(_TEX_CACHE_DIR):
        for fname in os.listdir(_TEX_CACHE_DIR):
            if fname.endswith("_tex.npy"):
                try:
                    os.remove(os.path.join(_TEX_CACHE_DIR, fname))
                    print(f"[缓存] 删除旧纹理缓存: {fname}")
                except OSError:
                    pass
            # 删除旧版 png 纹理缓存，强制重生 v2 提亮纹理
            if fname == "track_tex.png":
                try:
                    os.remove(os.path.join(_TEX_CACHE_DIR, fname))
                    print(f"[缓存] 删除旧赛道纹理，强制重生 v2 版: {fname}")
                except OSError:
                    pass
            # ★ 地形 PBR 纹理缓存，强制重生（已禁用 —— 现在用 fallback_build_fn 程序化生成）
            # 如需强制刷新，手动清空 .cache 目录或删除对应文件
            # if fname.startswith("terrain_") and fname.endswith(".png"):
            #     try:
            #         os.remove(os.path.join(_TEX_CACHE_DIR, fname))
            #         print(f"[缓存] 删除旧地形纹理，强制重生: {fname}")
            #     except OSError:
            #         pass

    # ★ 清理赛道几何缓存（纹理/UV/法线变更后需要重建）
    # 已禁用：这些缓存由各自的构建函数自动检查版本/哈希，无需每次强制删除。
    # 如需强制刷新，手动清空 .cache 目录或删除对应文件即可。
    # if os.path.exists(_TEX_CACHE_DIR):
    #     for fname in os.listdir(_TEX_CACHE_DIR):
    #         if fname.startswith("tri_index_") or fname.startswith("track_height_field_"):
    #             try:
    #                 os.remove(os.path.join(_TEX_CACHE_DIR, fname))
    #                 print(f"[缓存] 删除赛道几何缓存: {fname}")
    #             except OSError:
    #                 pass
    # ★ 清理地形颜色缓存（白天提亮后需要重建）
    # 已禁用：地形缓存已稳定，每次强制删除浪费 ~11 秒。
    # 如需强制刷新，手动删除 .cache/akina_terrain_v*.npz 即可。
    # terrain_npz = os.path.join(_TEX_CACHE_DIR, "akina_terrain_v20.npz")
    # if os.path.exists(terrain_npz):
    #     try:
    #         os.remove(terrain_npz)
    #         print(f"[缓存] 删除地形颜色缓存: akina_terrain_v20.npz")
    #     except OSError:
    #         pass

    if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) < 0:
        print(f"SDL失败: {sdl2.SDL_GetError().decode()}"); return 1
    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MAJOR_VERSION, 4)
    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_MINOR_VERSION, 3)
    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_PROFILE_MASK, sdl2.SDL_GL_CONTEXT_PROFILE_CORE)
    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_CONTEXT_FLAGS, sdl2.SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG)
    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_DOUBLEBUFFER, 1)
    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_DEPTH_SIZE, 24)
    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_MULTISAMPLEBUFFERS, 0)
    sdl2.SDL_GL_SetAttribute(sdl2.SDL_GL_MULTISAMPLESAMPLES, 0)
    # 窗口尺寸自适应屏幕分辨率
    user32 = ctypes.windll.user32
    SCREEN_W = user32.GetSystemMetrics(0)  # 屏幕宽度
    SCREEN_H = user32.GetSystemMetrics(1)  # 屏幕高度
    # 窗口比屏幕略小，确保标题栏不超出屏幕顶部
    WIN_W = min(1920, SCREEN_W)
    WIN_H = min(1080, SCREEN_H - 36)  # 减去标题栏高度
    print(f"屏幕分辨率: {SCREEN_W}x{SCREEN_H}, 窗口尺寸: {WIN_W}x{WIN_H}")
    win = sdl2.SDL_CreateWindow("动力滑行".encode(),
        sdl2.SDL_WINDOWPOS_CENTERED, sdl2.SDL_WINDOWPOS_CENTERED, WIN_W, WIN_H,
        sdl2.SDL_WINDOW_OPENGL | sdl2.SDL_WINDOW_SHOWN)
    if not win: print(f"窗口失败: {sdl2.SDL_GetError().decode()}"); sdl2.SDL_Quit(); return 1
    ctx = sdl2.SDL_GL_CreateContext(win)
    if not ctx: print(f"GL失败: {sdl2.SDL_GetError().decode()}"); sdl2.SDL_DestroyWindow(win); sdl2.SDL_Quit(); return 1
    sdl2.SDL_GL_SetSwapInterval(1)

    # 获取逻辑尺寸（SDL_GetWindowSize）和实际绘制尺寸（SDL_GL_GetDrawableSize）
    logical_ww = ctypes.c_int()
    logical_wh = ctypes.c_int()
    sdl2.SDL_GetWindowSize(win, ctypes.byref(logical_ww), ctypes.byref(logical_wh))
    drawable_ww = ctypes.c_int()
    drawable_wh = ctypes.c_int()
    sdl2.SDL_GL_GetDrawableSize(win, ctypes.byref(drawable_ww), ctypes.byref(drawable_wh))
    logical_ww, logical_wh = logical_ww.value, logical_wh.value
    drawable_ww, drawable_wh = drawable_ww.value, drawable_wh.value
    # 实际绘制尺寸用于 glViewport 和 3D 渲染
    ww, wh = drawable_ww, drawable_wh
    # 创建 UI 坐标系统（逻辑像素 ↔ 实际绘制像素 映射）
    ui_coord = UICoordSystem(logical_ww, logical_wh, drawable_ww, drawable_wh)
    glViewport(0, 0, ww, wh)
    glClearColor(0.60, 0.82, 0.98, 1.0)  # 正午晴空蓝天（亮蓝）
    glEnable(GL_DEPTH_TEST)
    # 关闭 MSAA（deferred 渲染中 MSAA 对 G-Buffer 无效且浪费带宽）
    # 所有模型都是低多边形，不开启背面剔除避免不同面的绕序问题

    prog = create_shader_program(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC)
    if not prog: return 1
    uMVP = glGetUniformLocation(prog, "uMVP")
    uModel = glGetUniformLocation(prog, "uModel")
    uTex = glGetUniformLocation(prog, "uTexture")
    uLD = glGetUniformLocation(prog, "uLightDir")
    uVP = glGetUniformLocation(prog, "uViewPos")
    uAmb = glGetUniformLocation(prog, "uAmbientColor")
    uLC = glGetUniformLocation(prog, "uLightColor")
    uMet = glGetUniformLocation(prog, "uMetallic")
    uRough = glGetUniformLocation(prog, "uRoughness")
    uHasTex = glGetUniformLocation(prog, "uHasTexture")
    uAmbBoost = glGetUniformLocation(prog, "uAmbientBoost")
    uExposure = glGetUniformLocation(prog, "uExposure")
    uGrassTex  = glGetUniformLocation(prog, "uGrassTex")
    uRockTex   = glGetUniformLocation(prog, "uRockTex")
    uDirtTex   = glGetUniformLocation(prog, "uDirtTex")
    uGravelTex = glGetUniformLocation(prog, "uGravelTex")
    uUseTerrainBlend = glGetUniformLocation(prog, "uUseTerrainBlend")
    uGrassNrm  = glGetUniformLocation(prog, "uGrassNrm")
    uRockNrm   = glGetUniformLocation(prog, "uRockNrm")
    uDirtNrm   = glGetUniformLocation(prog, "uDirtNrm")
    uGravelNrm = glGetUniformLocation(prog, "uGravelNrm")
    uDebugMode = glGetUniformLocation(prog, "uDebugMode")
    uIsEmissive = glGetUniformLocation(prog, "uIsEmissive")
    uApplyTonemap = glGetUniformLocation(prog, "uApplyTonemap")
    uIsBuilding = glGetUniformLocation(prog, "uIsBuilding")
    uBuildingAlbedo  = glGetUniformLocation(prog, "uBuildingAlbedo")
    uBuildingNormal  = glGetUniformLocation(prog, "uBuildingNormal")
    # uBuildingEmissive = glGetUniformLocation(prog, "uBuildingEmissive")  # 已移除
    uBuildingAO      = glGetUniformLocation(prog, "uBuildingAO")
    uDetail      = glGetUniformLocation(prog, "uDetail")
    uDetailTex   = glGetUniformLocation(prog, "uDetailTex")
    uDetailScale = glGetUniformLocation(prog, "uDetailScale")

    # ---- ★ 第三阶段：从本地 PBR 或缓存加载纹理（替换程序化生成） ----
    print("加载 PBR 纹理（优先外部资产，回退缓存）...")
    track_tex    = _load_tex_from_cache_or_pbr("track_tex_v8", pbr_layer="track", width=2048, height=2048,
                                            fallback_build_fn=_build_track_tex)
    track_normal = _load_normal_from_cache_or_pbr("track_nrm_v1", pbr_layer="track", width=2048, height=2048,
                                                  fallback_build_fn=_build_track_normal)
    building_tex = _load_tex_from_cache_or_pbr("building_tex", pbr_layer="building", width=512, height=512)
    building_wall_tex = _load_tex_from_cache_or_pbr("building_wall_tex_v2", pbr_layer="building_wall", width=512, height=512)
    building_emissive_tex = _load_tex_from_cache_or_pbr("building_emissive_tex", pbr_layer="building_emissive", width=512, height=512)
    building_ao_tex = _load_tex_from_cache_or_pbr("building_ao_tex", pbr_layer="building_ao", width=512, height=512)
    building_normal_tex = _load_tex_from_cache_or_pbr("building_normal_tex", pbr_layer="building_normal", width=512, height=512)
    car_tex = _load_tex_from_cache_or_pbr("car_tex", pbr_layer="car", width=512, height=256)
    # 纯白占位纹理
    _w = np.full((2, 2, 3), 255, dtype=np.uint8)
    white_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, white_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 2, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, _w)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    # 地形 4 层 PBR 纹理（优先外部 PBR 目录 → 回退缓存 PNG → 程序化生成）
    grass_tex  = _load_tex_from_cache_or_pbr("terrain_grass_v1",  pbr_layer="terrain_grass",  width=512, height=512,
                                              fallback_build_fn=_build_grass_tex)
    rock_tex   = _load_tex_from_cache_or_pbr("terrain_rock_v1",   pbr_layer="terrain_rock",   width=512, height=512,
                                              fallback_build_fn=_build_rock_tex)
    dirt_tex   = _load_tex_from_cache_or_pbr("terrain_dirt_v1",   pbr_layer="terrain_dirt",   width=512, height=512,
                                              fallback_build_fn=_build_dirt_tex)
    gravel_tex = _load_tex_from_cache_or_pbr("terrain_gravel_v1", pbr_layer="terrain_gravel", width=512, height=512,
                                              fallback_build_fn=_build_gravel_tex)

    # 地形法线贴图（优先外部 PBR → 缓存 PNG → 程序化生成）
    grass_nrm  = _load_normal_from_cache_or_pbr("terrain_grass_nm_v1",  pbr_layer="terrain_grass",  width=512, height=512,
                                                 fallback_build_fn=_build_grass_nm)
    rock_nrm   = _load_normal_from_cache_or_pbr("terrain_rock_nm_v1",   pbr_layer="terrain_rock",   width=512, height=512,
                                                 fallback_build_fn=_build_rock_nm)
    dirt_nrm   = _load_normal_from_cache_or_pbr("terrain_dirt_nm_v1",   pbr_layer="terrain_dirt",   width=512, height=512,
                                                 fallback_build_fn=_build_dirt_nm)
    gravel_nrm = _load_normal_from_cache_or_pbr("terrain_gravel_nm_v1", pbr_layer="terrain_gravel", width=512, height=512,
                                                 fallback_build_fn=_build_gravel_nm)

    # ---- ★ 第一批：PBR 材质通道（roughness / metallic / AO / height） ----
    print("加载 PBR roughness/metallic/AO/height 通道...")
    track_roughness = pbr_loader.load_pbr_roughness(_PBR_TEX_DIR, "track", 2048, 2048)
    track_metallic  = pbr_loader.load_pbr_metallic(_PBR_TEX_DIR, "track", 2048, 2048)
    track_ao        = pbr_loader.load_pbr_ao(_PBR_TEX_DIR, "track", 2048, 2048)
    track_height    = pbr_loader.load_pbr_height(_PBR_TEX_DIR, "track", 2048, 2048)
    if track_roughness: print(f"  [PBR] 加载 track_roughness")
    if track_metallic:  print(f"  [PBR] 加载 track_metallic")
    if track_ao:        print(f"  [PBR] 加载 track_ao")
    if track_height:    print(f"  [PBR] 加载 track_height")

    # 地形 4 层 PBR roughness / AO
    grass_roughness = pbr_loader.load_pbr_roughness(_PBR_TEX_DIR, "terrain_grass", 512, 512)
    grass_ao        = pbr_loader.load_pbr_ao(_PBR_TEX_DIR, "terrain_grass", 512, 512)
    rock_roughness  = pbr_loader.load_pbr_roughness(_PBR_TEX_DIR, "terrain_rock", 512, 512)
    rock_ao         = pbr_loader.load_pbr_ao(_PBR_TEX_DIR, "terrain_rock", 512, 512)
    dirt_roughness  = pbr_loader.load_pbr_roughness(_PBR_TEX_DIR, "terrain_dirt", 512, 512)
    dirt_ao         = pbr_loader.load_pbr_ao(_PBR_TEX_DIR, "terrain_dirt", 512, 512)
    gravel_roughness = pbr_loader.load_pbr_roughness(_PBR_TEX_DIR, "terrain_gravel", 512, 512)
    gravel_ao       = pbr_loader.load_pbr_ao(_PBR_TEX_DIR, "terrain_gravel", 512, 512)
    if grass_roughness: print(f"  [PBR] 加载 grass_roughness")
    if grass_ao:        print(f"  [PBR] 加载 grass_ao")
    if rock_roughness:  print(f"  [PBR] 加载 rock_roughness")
    if rock_ao:         print(f"  [PBR] 加载 rock_ao")
    if dirt_roughness:  print(f"  [PBR] 加载 dirt_roughness")
    if dirt_ao:         print(f"  [PBR] 加载 dirt_ao")
    if gravel_roughness: print(f"  [PBR] 加载 gravel_roughness")
    if gravel_ao:       print(f"  [PBR] 加载 gravel_ao")

    # ---- ★ 第三阶段：默认高度图（POM 占位，中性灰 = 无偏移） ----
    _h = np.full((256, 256, 3), 128, dtype=np.uint8)
    default_height_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, default_height_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 256, 256, 0, GL_RGB, GL_UNSIGNED_BYTE, _h)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glGenerateMipmap(GL_TEXTURE_2D)

    # ---- ★ PBR 占位纹理（防止 sampler 默认 0 号单元采到赛道 albedo） ----
    _zero = np.zeros((1, 1, 3), dtype=np.uint8)  # metallic = 0
    pbr_default_metal_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, pbr_default_metal_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, _zero)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    _one = np.full((1, 1, 3), 255, dtype=np.uint8)  # roughness/ao = 1.0
    pbr_default_rough_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, pbr_default_rough_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, _one)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    pbr_default_ao_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, pbr_default_ao_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, _one)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    # ---- 天空球 ----
    sky_prog = create_shader_program(SKY_VERTEX_SRC, SKY_FRAGMENT_SRC)
    if not sky_prog: return 1
    sky_uMVP          = glGetUniformLocation(sky_prog, "uMVP")
    sky_uSunDir       = glGetUniformLocation(sky_prog, "uSunDir")
    sky_uZenithColor  = glGetUniformLocation(sky_prog, "uZenithColor")
    sky_uHorizonColor = glGetUniformLocation(sky_prog, "uHorizonColor")
    sky_uGroundColor  = glGetUniformLocation(sky_prog, "uGroundColor")
    sky_uSunColor     = glGetUniformLocation(sky_prog, "uSunColor")
    sky_uTime         = glGetUniformLocation(sky_prog, "uTime")

    skv = build_sky_sphere()
    skvao = glGenVertexArrays(1)
    skvbo = glGenBuffers(1)
    glBindVertexArray(skvao)
    glBindBuffer(GL_ARRAY_BUFFER, skvbo)
    glBufferData(GL_ARRAY_BUFFER, skv.nbytes, skv, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glBindVertexArray(0)
    sky_cnt = len(skv) // 3  # 每三角形3个顶点，每个顶点3个float

    # ---- 构建场景 ----
    print("构建榛名山赛道...")
    tv = build_mountain_road(); tvao, tvbo = create_vao(tv, [3,3,2,3]); tcnt = len(tv)
    print(f"  赛道顶点: {tcnt}")
    build_triangle_index(tv)  # 构建视觉三角形索引（与 GPU 渲染同构的精确查询）

    print("构建山路护栏...")
    grv = build_guardrails(); grvao, grvbo = create_vao(grv, [3,3,2,3]); grcnt = len(grv)
    print(f"  护栏顶点: {grcnt}")

    print("构建榛名山杉树林（实例化 + LOD + 4 变体）...")
    tree_templates = build_tree_templates()      # 12 条：4 变体 x 3 LOD
    tree_instances = build_tree_instances()      # (M, 8)，末列 = 变体 ID
    n_tree_instances = len(tree_instances)
    n_variants = max(t['variant'] for t in tree_templates) + 1
    n_lods     = max(t['lod'] for t in tree_templates) + 1
    lod_max_dist = [t['max_distance'] for t in tree_templates if t['variant'] == 0]
    print(f"  实例数: {n_tree_instances}, 变体数: {n_variants}, LOD 级数: {n_lods}")
    if n_tree_instances > 0:
        _ti = tree_instances
        print(f"  树实例坐标范围: x[{_ti[:,0].min():.1f},{_ti[:,0].max():.1f}] "
              f"y[{_ti[:,1].min():.1f},{_ti[:,1].max():.1f}] "
              f"z[{_ti[:,2].min():.1f},{_ti[:,2].max():.1f}]")

    # 为每个 (变体, LOD) 创建 VAO，共享同一个 instance buffer
    # 顶点格式（14 float）：[pos3, nrm3, uv2, col3, tan3] -> 跨距 56 字节
    TREE_VERT_STRIDE = 14 * 4
    TREE_TANGENT_OFF = 11 * 4
    tree_inst_vbo = glGenBuffers(1)
    stride_inst = 8 * 4                          # 8 float32 每实例（多了 variant_id）
    tree_units = {}                               # (variant, lod) -> {vao, count}
    for t in tree_templates:
        vao, vbo = create_vao(t['vertices'], [3, 3, 2, 3], stride_override=TREE_VERT_STRIDE)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glEnableVertexAttribArray(8)
        glVertexAttribPointer(8, 3, GL_FLOAT, GL_FALSE, TREE_VERT_STRIDE,
                              ctypes.c_void_p(TREE_TANGENT_OFF))
        glBindBuffer(GL_ARRAY_BUFFER, tree_inst_vbo)
        for loc, sz, off in ((5, 3, 0), (6, 1, 12), (7, 3, 16)):
            glEnableVertexAttribArray(loc)
            glVertexAttribPointer(loc, sz, GL_FLOAT, GL_FALSE, stride_inst, ctypes.c_void_p(off))
            glVertexAttribDivisor(loc, 1)
        glBindVertexArray(0)
        tree_units[(t['variant'], t['lod'])] = {'vao': vao, 'count': len(t['vertices'])}

    _variant_ids = tree_instances[:, 7].astype(np.int32)

    def _tree_band_masks(dist_arr):
        masks, prev = [], 0.0
        for li in range(n_lods):
            th = lod_max_dist[li]
            if li == n_lods - 1:
                m = (dist_arr >= prev)
            else:
                m = (dist_arr >= prev) & (dist_arr < th)
            masks.append(m); prev = th
        return masks

    def _draw_tree_batches(band_masks, extra_mask):
        """按 变体 x LOD 分桶绘制。extra_mask = 视锥/其它过滤"""
        for v in range(n_variants):
            vm = extra_mask & (_variant_ids == v)
            for li, bm in enumerate(band_masks):
                m = vm & bm
                cnt = int(np.count_nonzero(m))
                if cnt == 0: continue
                u = tree_units[(v, li)]
                ss = tree_instances[m]
                glBindBuffer(GL_ARRAY_BUFFER, tree_inst_vbo)
                glBufferSubData(GL_ARRAY_BUFFER, 0, ss.nbytes, ss)
                glBindVertexArray(u['vao'])
                glDrawArraysInstanced(GL_TRIANGLES, 0, u['count'], cnt)

    # ★ GPU 视锥剔除 SSBO 初始化
    # 构建平面实例数据: (N, 4) — x, y, z, radius（注意：列布局与 shader 对齐）
    TREE_BOUND_RADIUS = 12.0
    tree_flat = np.zeros((n_tree_instances, 4), dtype=np.float32)
    tree_flat[:, 0] = tree_instances[:, 0]  # x
    tree_flat[:, 1] = tree_instances[:, 1]  # y（原来是 z，漏掉 y 导致山坡树被误剔除）
    tree_flat[:, 2] = tree_instances[:, 2]  # z
    tree_flat[:, 3] = TREE_BOUND_RADIUS       # 包围球半径

    # 实例输入 SSBO（binding=20）
    tree_cull_ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, tree_cull_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, tree_flat.nbytes, tree_flat, GL_STATIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 20, tree_cull_ssbo)

    # 可见索引输出 SSBO（binding=21），预分配最大容量
    visible_idx_buf = np.zeros(n_tree_instances, dtype=np.uint32)
    visible_idx_ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, visible_idx_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, visible_idx_buf.nbytes, visible_idx_buf, GL_DYNAMIC_COPY)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 21, visible_idx_ssbo)

    # 计数器 SSBO（binding=23），归零
    total_visible_counter = np.zeros(1, dtype=np.uint32)
    total_visible_ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, total_visible_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, total_visible_counter.nbytes, total_visible_counter, GL_DYNAMIC_COPY)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 23, total_visible_ssbo)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    # 编译视锥剔除 Compute Shader
    frustum_cs_prog = create_compute_program(FRUSTUM_CULL_SRC)
    if not frustum_cs_prog:
        print("  视锥剔除 Compute Shader 编译失败!")
    else:
        fcs_uPlanes = glGetUniformLocation(frustum_cs_prog, "frustumPlanes[0]")
        fcs_uCount = glGetUniformLocation(frustum_cs_prog, "totalInstances")
        print("  视锥剔除 Compute Shader 编译成功")

    # 首次上传实例数据（DYNAMIC，每帧更新 LOD 子集）
    glBindBuffer(GL_ARRAY_BUFFER, tree_inst_vbo)
    glBufferData(GL_ARRAY_BUFFER, tree_instances.nbytes, tree_instances, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    # 创建树渲染程序（使用实例化顶点着色器）
    tree_prog = create_shader_program(TREE_VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC)
    if not tree_prog: return 1
    tuMVP = glGetUniformLocation(tree_prog, "uMVP")
    tuModel = glGetUniformLocation(tree_prog, "uModel")
    tuLD = glGetUniformLocation(tree_prog, "uLightDir")
    tuVP = glGetUniformLocation(tree_prog, "uViewPos")
    tuAmb = glGetUniformLocation(tree_prog, "uAmbientColor")
    tuLC = glGetUniformLocation(tree_prog, "uLightColor")
    tuMet = glGetUniformLocation(tree_prog, "uMetallic")
    tuRough = glGetUniformLocation(tree_prog, "uRoughness")
    tuHasTex = glGetUniformLocation(tree_prog, "uHasTexture")
    tuAmbBoost = glGetUniformLocation(tree_prog, "uAmbientBoost")
    tuExposure = glGetUniformLocation(tree_prog, "uExposure")
    tuUseTerrainBlend = glGetUniformLocation(tree_prog, "uUseTerrainBlend")
    tuFoliage = glGetUniformLocation(tree_prog, "uFoliageTex")
    tuUseFoliage = glGetUniformLocation(tree_prog, "uUseFoliage")
    tuTime = glGetUniformLocation(tree_prog, "uTime")

    # ---- 杉树叶片图集 v3（针叶簇剪影 + 法线 + 内部遮蔽） ----
    print("生成杉树叶片图集（1024, 针叶簇 + 法线 + AO）...")
    _fol_rgb = None
    try:
        from mountain import build_foliage_textures as _build_fol
        _falbedo, _fnrm = _build_fol(1024)
    except Exception as _e:
        print(f"  叶片图集生成失败({_e})，回退旧版 256 贴图")
        _fo = build_foliage_texture(256)
        _falbedo = _fo
        _fnrm = np.zeros_like(_fo)
        _fnrm[..., 0:2] = 0.5
        _fnrm[..., 2] = 1.0
        _fnrm[..., 3] = 1.0

    foliage_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, foliage_tex)
    _alb_lv = _make_foliage_mips(_falbedo[..., :3], _falbedo[..., 3])
    for _lv, (_rgb_l, _a_l) in enumerate(_alb_lv):
        _d = np.concatenate([_rgb_l, _a_l[..., None]], axis=-1).astype(np.float32)
        glTexImage2D(GL_TEXTURE_2D, _lv, GL_RGBA16F, _d.shape[1], _d.shape[0],
                     0, GL_RGBA, GL_FLOAT, _d)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, len(_alb_lv) - 1)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    try:
        _maxa = glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, min(8.0, _maxa))
    except Exception:
        pass

    # ★ 叶片法线贴图（RGB=切线空间法线, A=内部遮蔽）
    foliage_nrm_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, foliage_nrm_tex)
    _nrm_lv = _make_normal_mips(_fnrm)
    for _lv, _n_l in enumerate(_nrm_lv):
        glTexImage2D(GL_TEXTURE_2D, _lv, GL_RGBA16F, _n_l.shape[1], _n_l.shape[0],
                     0, GL_RGBA, GL_FLOAT, _n_l)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, len(_nrm_lv) - 1)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glBindTexture(GL_TEXTURE_2D, 0)
    print(f"  叶片贴图就绪 ({_falbedo.shape[1]}x{_falbedo.shape[0]}, "
          f"{len(_alb_lv)} 级 mip, alpha 取最大值保覆盖率)")

    # ---- 路面高频细节贴图（灰度噪声，双层平铺） ----
    _drng = np.random.default_rng(11)
    def _smooth_noise(n):
        r = _drng.random((n, n)).astype(np.float32)
        try:
            from scipy.ndimage import zoom as _z
            out = _z(r, 256 / n, order=1)      # ★ 用浮点倍率，256//96=2 会得到 192 尺寸
        except Exception:
            out = np.kron(r, np.ones((256 // n, 256 // n), dtype=np.float32))
        out = np.asarray(out, dtype=np.float32)[:256, :256]
        if out.shape != (256, 256):            # 不足则边缘补齐
            _pad = 256 - out.shape[0], 256 - out.shape[1]
            out = np.pad(out, ((0, _pad[0]), (0, _pad[1])), mode='edge')
        return out
    _detail = (_smooth_noise(32) * 0.6 + _smooth_noise(96) * 0.4)
    _detail = np.clip(_detail, 0.0, 1.0)
    detail_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, detail_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, 256, 256, 0, GL_RED, GL_FLOAT,
                 np.ascontiguousarray(_detail))
    glGenerateMipmap(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glBindTexture(GL_TEXTURE_2D, 0)
    print("  路面细节贴图就绪 (256x256, 双频噪声)")

    print("构建山地地形...")
    akv = build_akina_terrain()
    akvao, akvbos, akcnt = create_vao_packed_terrain(akv)
    print(f"  地形顶点: {akcnt:,}")

    print("构建赛道高度场缓存（车辆物理查询加速）...")
    build_height_field()
    print("  赛道高度场构建完成")

    print("构建赛车（按材质分组加载 RX-7）...")
    car_groups = build_car_groups()
    print("构建尾灯发光（按自发光材质提取）...")
    tl_groups = [g for g in car_groups if g['is_light']]

    print("构建混凝土路障...")
    cbv = build_concrete_barriers()          # ★ 弯道石墩子已移除 → 恒为空
    cbvao = cbvbo = None
    cbcnt = len(cbv)
    if cbcnt > 0:
        cbvao, cbvbo = create_vao(cbv, [3,3,2,3])
    print(f"  混凝土路障顶点: {cbcnt}")

    print("构建挖方挡土墙...")
    rwv = build_retaining_walls(); rwvao, rwvbo = create_vao(rwv, [3,3,2,3]); rwcnt = len(rwv)
    print(f"  挡土墙顶点: {rwcnt}")

    print("构建弯道标志牌（凸面镜 + 箭头牌）...")
    sgn = build_roadside_signs(); sgnvao, sgnvbo = create_vao(sgn, [3,3,2,3]); sgncnt = len(sgn)
    print(f"  标志牌顶点: {sgncnt}")
    sgne = build_roadside_signs_emissive(); sgnevao, sgnevbo = create_vao(sgne, [3,3,2,3]); sgnecnt = len(sgne)

    print("构建反光道钉...")
    stu = build_road_studs(); stuvao, stuvbo = create_vao(stu, [3,3,2,3]); stucnt = len(stu)
    print(f"  道钉顶点: {stucnt}")

    # ---- ★ 新增：路肩 U 型混凝土侧沟（榛名山"沟渠跑法"的原型）----
    print("构建路肩 U 型混凝土侧沟...")
    dtv = build_drainage_ditches()
    print(f"  侧沟顶点: {len(dtv):,}")
    gutter_phys_warmup()   # 预热"这里有没有沟"查询表，避免开进沟里时才算曲率

    # ---- ★ 新增：路面标线与磨耗（中央线 / 减速标线 / 补丁 / 胎痕）----
    print("构建路面标线与磨耗...")
    rmv = build_road_markings()
    print(f"  标线顶点: {len(rmv):,}")

    # ---- ★ 新增：百米里程标（日式 100m 標）----
    print("构建百米里程标...")
    kpv = build_kiloposts()
    print(f"  里程标顶点: {len(kpv):,}")

    print("构建挖方边坡格构护坡（法枠工）...")
    latv = build_slope_lattice()
    print(f"  格构护坡顶点: {len(latv):,}")

    print("构建落石防护网...")
    rnv = build_rockfall_nets()
    print(f"  落石防护网顶点: {len(rnv):,}")

    print("构建路灯...")
    lv = build_lights()
    lvao, lvbo = create_vao(lv, [3,3,2,3]); lcnt = len(lv)
    print(f"  路灯顶点: {lcnt}")

    print("构建灯罩自发光...")
    lem = build_lamp_emissive()
    lemvao, lemvbo = create_vao(lem, [3,3,2,3]); lemcnt = len(lem)

    print("构建地面光池...")
    lpool = build_lamp_ground_pool()
    lpvao, lpvbo = create_vao(lpool, [3,3,2,3]); lpcnt = len(lpool)

    print("构建光锥体积光...")
    cone = build_lamp_light_cone()
    conevao, conevbo = create_vao(cone, [3,3,2,4]); conecnt = len(cone)
    print(f"  光锥顶点: {conecnt}")

    print("构建沿路建筑群...")
    bv = build_buildings()
    bvao, bvbo = create_vao(bv, [3,3,2,4]); bcnt = len(bv)
    print(f"  建筑顶点: {bcnt}")

    # 计算建筑群包围体（用于视锥剔除）
    from mountain import get_building_positions
    bpos = get_building_positions()
    if len(bpos) > 0:
        building_center = bpos[:, :3].mean(axis=0).astype(np.float32)
        b_offsets = bpos[:, :3] - building_center
        b_max_dist = np.sqrt((b_offsets * b_offsets).sum(axis=1).max()) + bpos[:, 3].max()
        building_bound_radius = float(b_max_dist)
    else:
        building_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        building_bound_radius = 50.0
    print(f"  建筑群中心: ({building_center[0]:.1f}, {building_center[1]:.1f}, {building_center[2]:.1f}), 包围半径: {building_bound_radius:.1f}m")

    print("构建起终点场地...")
    vgv = build_venue_ground()
    vgvao, vgvbo = create_vao(vgv, [3,3,2,3]); vgcnt = len(vgv)
    print(f"  场地顶点: {vgcnt}")

    print("构建起终点专用建筑...")
    vbv = build_venue_buildings()
    vbvao, vbvbo = create_vao(vbv, [3,3,2,3]); vbcnt = len(vbv)
    print(f"  建筑顶点: {vbcnt}")

    # ---- ★ Indirect 批量绘制初始化 ----
    print("初始化 BatchRenderer（Indirect 批量绘制）...")
    from batch_renderer import BatchRenderer
    scene_batch = BatchRenderer(max_subsets=20)
    # 护栏: Met=0.5, Rough=0.5
    scene_batch.add_subset("护栏", grv, metallic=0.5, roughness=0.5)
    # 路灯: Met 降低避免 PBR 阴影下压成全黑，Rough 提高增加漫反射可见度
    scene_batch.add_subset("路灯", lv, metallic=0.20, roughness=0.65)
    # 混凝土路障: Met=0.05, Rough=0.75
    if cbv is not None and cbcnt > 0:
        scene_batch.add_subset("路障", cbv, metallic=0.05, roughness=0.75)
    # 挡土墙: Met=0.0, Rough=0.92（分层石笼漫反射，增加颗粒感）
    if rwv is not None and rwcnt > 0:
        scene_batch.add_subset("挡土墙", rwv, metallic=0.0, roughness=0.92)
    # 标志牌（正常受光）
    if sgncnt > 0:
        scene_batch.add_subset("标志牌", sgn, metallic=0.10, roughness=0.60)
    # 路肩 U 型侧沟：湿混凝土，几乎无金属度
    if len(dtv) > 0:
        scene_batch.add_subset("侧沟", dtv, metallic=0.0, roughness=0.88)
    # 路面标线：热熔标线漆，略亮略光滑
    if len(rmv) > 0:
        scene_batch.add_subset("路面标线", rmv, metallic=0.0, roughness=0.55)
    # 百米里程标：白漆钢柱
    if len(kpv) > 0:
        scene_batch.add_subset("里程标", kpv, metallic=0.15, roughness=0.45)
    # 格构护坡：现浇混凝土
    if len(latv) > 0:
        scene_batch.add_subset("格构护坡", latv, metallic=0.0, roughness=0.92)
    # 落石防护网：镀锌钢丝
    if len(rnv) > 0:
        scene_batch.add_subset("落石防护网", rnv, metallic=0.55, roughness=0.55)
    # 起终点场地：沥青平台，粗糙无金属
    if vgcnt > 0:
        scene_batch.add_subset("起终点场地", vgv, metallic=0.0, roughness=0.85)
    # 起终点建筑：混凝土+玻璃混合
    if vbcnt > 0:
        scene_batch.add_subset("起终点建筑", vbv, metallic=0.10, roughness=0.60)
    scene_batch.upload()
    print("  BatchRenderer 准备就绪，合并子网格:", len(scene_batch.subsets))

    # ---- ★ Deferred Rendering：G-Buffer FBO（4×MRT + Depth） ----
    print("初始化 G-Buffer（Deferred Rendering）...")
    # RT0: Position (RGB16F) + materialFlags (A=0/1/2)
    g_pos_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, g_pos_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, ww, wh, 0, GL_RGBA, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    # RT1: Normal (RGB10_A2 — 从 RGBA16F 压缩，带宽减半)
    g_nrm_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, g_nrm_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB10_A2, ww, wh, 0, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    # RT2: Albedo (sRGB8_ALPHA8) — RGB=albedo(sRGB自动线性化), A=metallic
    g_alb_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, g_alb_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, ww, wh, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    # RT3: Roughness/Metallic/AO (RGBA8) — R=roughness, G=AO, B=unused, A=emissiveFlag
    g_mat_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, g_mat_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, ww, wh, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    # Depth (DEPTH_COMPONENT24)
    g_depth_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, g_depth_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, ww, wh, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    # 组装 G-Buffer FBO
    g_buffer_fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, g_buffer_fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_pos_tex, 0)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, g_nrm_tex, 0)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, g_alb_tex, 0)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, g_mat_tex, 0)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, g_depth_tex, 0)

    # MRT 写入目标
    draw_buffers = [GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3]
    glDrawBuffers(4, draw_buffers)

    g_status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    if g_status != GL_FRAMEBUFFER_COMPLETE:
        print(f"  G-Buffer FBO 不完整: {g_status}，回退到 Forward 渲染")
        USE_DEFERRED = False
    else:
        print(f"  G-Buffer FBO 就绪 ({ww}x{wh}, 4×MRT + Depth)")
        USE_DEFERRED = True
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    # 编译 Deferred 渲染着色器
    geo_prog = None
    geo_batch_prog = None
    light_prog = None
    if USE_DEFERRED:
        geo_prog = create_shader_program(GEO_VERTEX_SRC, GEO_FRAGMENT_SRC)
        light_prog = create_shader_program(LIGHT_VERTEX_SRC, LIGHT_FRAGMENT_SRC)
        if not geo_prog or not light_prog:
            print("  Deferred 着色器编译失败，回退到 Forward 渲染")
            USE_DEFERRED = False
        else:
            # 几何阶段 uniform 位置
            guMVP = glGetUniformLocation(geo_prog, "uMVP")
            guModel = glGetUniformLocation(geo_prog, "uModel")
            guIsInstanced = glGetUniformLocation(geo_prog, "uIsInstanced")
            guTex = glGetUniformLocation(geo_prog, "uTexture")
            guMet = glGetUniformLocation(geo_prog, "uMetallic")
            guRough = glGetUniformLocation(geo_prog, "uRoughness")
            guHasTex = glGetUniformLocation(geo_prog, "uHasTexture")
            guIsEmissive = glGetUniformLocation(geo_prog, "uIsEmissive")
            guIsBuilding = glGetUniformLocation(geo_prog, "uIsBuilding")
            guUseTerrainBlend = glGetUniformLocation(geo_prog, "uUseTerrainBlend")
            guGrassTex  = glGetUniformLocation(geo_prog, "uGrassTex")
            guRockTex   = glGetUniformLocation(geo_prog, "uRockTex")
            guDirtTex   = glGetUniformLocation(geo_prog, "uDirtTex")
            guGravelTex = glGetUniformLocation(geo_prog, "uGravelTex")
            guGrassNrm  = glGetUniformLocation(geo_prog, "uGrassNrm")
            guRockNrm   = glGetUniformLocation(geo_prog, "uRockNrm")
            guDirtNrm   = glGetUniformLocation(geo_prog, "uDirtNrm")
            guGravelNrm = glGetUniformLocation(geo_prog, "uGravelNrm")

            # 编译批量（Indirect）几何 Pass 着色器
            geo_batch_prog = create_shader_program(GEO_VERTEX_BATCH_SRC, GEO_FRAGMENT_BATCH_SRC)
            if not geo_batch_prog:
                print("  批量着色器编译失败，禁用 Indirect 批量绘制")
            else:
                guBatchMVP = glGetUniformLocation(geo_batch_prog, "uMVP")
                guBatchModel = glGetUniformLocation(geo_batch_prog, "uModel")
                guBatchViewPos = glGetUniformLocation(geo_batch_prog, "uViewPos")
                print(f"  批量 Indirect 着色器编译成功: prog={geo_batch_prog}")

            guDebugMode = glGetUniformLocation(geo_prog, "uDebugMode")
            guViewPos = glGetUniformLocation(geo_prog, "uViewPos")
            guBuildingAlbedo = glGetUniformLocation(geo_prog, "uBuildingAlbedo")
            guBuildingNormal = glGetUniformLocation(geo_prog, "uBuildingNormal")
            guBuildingAO = glGetUniformLocation(geo_prog, "uBuildingAO")
            # ★ 第三阶段：POM uniform
            guHeightMap = glGetUniformLocation(geo_prog, "uHeightMap")
            guParallaxScale = glGetUniformLocation(geo_prog, "uParallaxScale")
            guParallaxLayers = glGetUniformLocation(geo_prog, "uParallaxLayers")
            guTerrainHeightMap = glGetUniformLocation(geo_prog, "uTerrainHeightMap")
            # ★ PBR：track roughness/metallic/AO/normal sampler
            guTrackRough = glGetUniformLocation(geo_prog, "uTrackRoughness")
            guTrackMetal = glGetUniformLocation(geo_prog, "uTrackMetallic")
            guTrackAO    = glGetUniformLocation(geo_prog, "uTrackAO")
            guTrackNormal = glGetUniformLocation(geo_prog, "uTrackNormal")
            guIsCar = glGetUniformLocation(geo_prog, "uIsCar")
            # ★ PBR：terrain roughness/AO sampler
            guGrassRough  = glGetUniformLocation(geo_prog, "uGrassRoughness")
            guRockRough   = glGetUniformLocation(geo_prog, "uRockRoughness")
            guDirtRough   = glGetUniformLocation(geo_prog, "uDirtRoughness")
            guGravelRough = glGetUniformLocation(geo_prog, "uGravelRoughness")
            guGrassAO     = glGetUniformLocation(geo_prog, "uGrassAO")
            guRockAO      = glGetUniformLocation(geo_prog, "uRockAO")
            guDirtAO      = glGetUniformLocation(geo_prog, "uDirtAO")
            guGravelAO    = glGetUniformLocation(geo_prog, "uGravelAO")

            # ★ 叶片法线贴图 → 纹理单元 14（静态绑定，只需设置一次）
            glActiveTexture(GL_TEXTURE0 + 14)
            glBindTexture(GL_TEXTURE_2D, foliage_nrm_tex)
            glActiveTexture(GL_TEXTURE0)
            for _p_fol in (geo_prog, tree_prog):
                if _p_fol:
                    _loc_fol = glGetUniformLocation(_p_fol, "uFoliageNrm")
                    if _loc_fol >= 0:
                        glUseProgram(_p_fol)
                        glUniform1i(_loc_fol, 14)
            glUseProgram(0)

            # 光照阶段 uniform 位置
            luPos = glGetUniformLocation(light_prog, "gPosition")
            luNrm = glGetUniformLocation(light_prog, "gNormal")
            luAlb = glGetUniformLocation(light_prog, "gAlbedo")
            luMat = glGetUniformLocation(light_prog, "gMaterial")
            luLD = glGetUniformLocation(light_prog, "uLightDir")
            luLC = glGetUniformLocation(light_prog, "uLightColor")
            luVP = glGetUniformLocation(light_prog, "uViewPos")
            luAmb = glGetUniformLocation(light_prog, "uAmbientColor")
            luExposure = glGetUniformLocation(light_prog, "uExposure")
            # ★ 光照Pass中CSM/SSAO相关的uniform
            luShadow0    = glGetUniformLocation(light_prog, "uShadowMap0")
            luShadow1    = glGetUniformLocation(light_prog, "uShadowMap1")
            luLightVP    = glGetUniformLocation(light_prog, "uLightVP")
            luCascadeSplits = glGetUniformLocation(light_prog, "uCascadeSplits")
            luShadowStrength = glGetUniformLocation(light_prog, "uShadowStrength")
            luSSAOTex    = glGetUniformLocation(light_prog, "uSSAOTex")
            luSSAOEnabled = glGetUniformLocation(light_prog, "uSSAOEnabled")
            # ★ IBL：SH 球谐系数数组 uniform（9 个 vec3）
            luSHCoeffs = glGetUniformLocation(light_prog, "uSHCoeffs")
            print("  Deferred 渲染着色器就绪")

    # ---- ★ 级联阴影贴图（CSM）系统初始化 ----
    print("初始化 CSM 级联阴影贴图...")
    CSM_SIZE = 4096   # ★ 2048 → 4096：近景树影/车影更锐利
    csm_texs = []
    csm_fbos = []
    for i in range(2):
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, CSM_SIZE, CSM_SIZE,
                     0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        border_color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color)
        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex, 0)
        glDrawBuffer(GL_NONE); glReadBuffer(GL_NONE)
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print(f"  CSM FBO[{i}] 创建失败！")
        csm_texs.append(tex)
        csm_fbos.append(fbo)
    csm_prog = create_shader_program(CSM_DEPTH_VERTEX_SRC, CSM_DEPTH_FRAGMENT_SRC)
    print(f"  CSM 2×{CSM_SIZE} 级联阴影贴图就绪")

    # ---- ★ SSAO 系统初始化（半分辨率） ----
    print("初始化 SSAO...")
    SSAO_W, SSAO_H = ww // 2, wh // 2
    ssao_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, ssao_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, SSAO_W, SSAO_H, 0, GL_RED, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    ssao_fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, ssao_fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssao_tex, 0)
    ssao_blur_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, ssao_blur_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, SSAO_W, SSAO_H, 0, GL_RED, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    ssao_blur_fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, ssao_blur_fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssao_blur_tex, 0)

    # SSAO 随机旋转噪声纹理（4x4，用于采样方向随机化）
    ssao_noise_w, ssao_noise_h = 4, 4
    ssao_noise_data = np.zeros((ssao_noise_w * ssao_noise_h, 3), dtype=np.float32)
    for i in range(ssao_noise_w * ssao_noise_h):
        v = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0.0], dtype=np.float32)
        ssao_noise_data[i] = v / max(np.linalg.norm(v), 0.001)
    ssao_noise_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, ssao_noise_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, ssao_noise_w, ssao_noise_h, 0, GL_RGB, GL_FLOAT, ssao_noise_data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    # SSAO 采样核（32个半球采样点）
    ssao_samples = []
    for i in range(32):
        sample = np.array([
            np.random.uniform(-1, 1),
            np.random.uniform(-1, 1),
            np.random.uniform(0, 1)
        ], dtype=np.float32)
        sample = sample / np.linalg.norm(sample)
        scale = float(i) / 32.0
        scale = 0.1 + scale * scale * 0.9
        sample = sample * scale
        ssao_samples.append(sample)

    ssao_prog = create_shader_program(FULLSCREEN_VERT_SRC, SSAO_FRAGMENT_SRC)
    ssao_blur_prog = create_shader_program(FULLSCREEN_VERT_SRC, SSAO_BLUR_FRAG_SRC)
    if ssao_prog and ssao_blur_prog:
        suPos    = glGetUniformLocation(ssao_prog, "gPosition")
        suNrm    = glGetUniformLocation(ssao_prog, "gNormal")
        suNoise  = glGetUniformLocation(ssao_prog, "uNoiseTex")
        suSamples = glGetUniformLocation(ssao_prog, "uSamples")
        suProj   = glGetUniformLocation(ssao_prog, "uProjection")
        suViewPos = glGetUniformLocation(ssao_prog, "uViewPos")
    print(f"  SSAO 半分辨率 {SSAO_W}×{SSAO_H} 就绪")

    # ---- ★ SSR 系统初始化（全分辨率） ----
    print("初始化 SSR...")
    ssr_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, ssr_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, ww, wh, 0, GL_RGBA, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    ssr_fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, ssr_fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ssr_tex, 0)

    ssr_prog = create_shader_program(FULLSCREEN_VERT_SRC, SSR_FRAGMENT_SRC)
    if ssr_prog:
        srPos    = glGetUniformLocation(ssr_prog, "gPosition")
        srNrm    = glGetUniformLocation(ssr_prog, "gNormal")
        srAlb    = glGetUniformLocation(ssr_prog, "gAlbedo")
        srMat    = glGetUniformLocation(ssr_prog, "gMaterial")
        srProj   = glGetUniformLocation(ssr_prog, "uProjection")
        srInvProj = glGetUniformLocation(ssr_prog, "uInvProjection")
        srView   = glGetUniformLocation(ssr_prog, "uView")
        srViewPos = glGetUniformLocation(ssr_prog, "uViewPos")
        srSceneTex = glGetUniformLocation(ssr_prog, "uSceneTex")
        srDepthTex = glGetUniformLocation(ssr_prog, "uDepthTex")
        srScreenSize = glGetUniformLocation(ssr_prog, "uScreenSize")
        srMaxDist = glGetUniformLocation(ssr_prog, "uMaxDist")
        srStepSize = glGetUniformLocation(ssr_prog, "uStepSize")
        srMaxSteps = glGetUniformLocation(ssr_prog, "uMaxSteps")
        srThickness = glGetUniformLocation(ssr_prog, "uThickness")
    print("  SSR 全分辨率就绪")

    # ---- ★ 体积光系统初始化（半分辨率） ----
    print("初始化 体积光...")
    VOL_W, VOL_H = ww // 2, wh // 2
    vol_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, vol_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, VOL_W, VOL_H, 0, GL_RGBA, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    vol_fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, vol_fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, vol_tex, 0)

    # 3D噪声纹理（用于雾气流动，64×64×64 → 实际用3D纹理）
    # PyOpenGL 中3D纹理使用 glTexImage3D
    NOISE_SIZE = 64
    vol_noise_data = np.random.uniform(0, 1, (NOISE_SIZE, NOISE_SIZE, NOISE_SIZE, 4)).astype(np.float32)
    vol_noise_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_3D, vol_noise_tex)
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, NOISE_SIZE, NOISE_SIZE, NOISE_SIZE,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, vol_noise_data)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT)

    vol_prog = create_shader_program(FULLSCREEN_VERT_SRC, VOLUMETRIC_FRAG_SRC)
    if vol_prog:
        vuDepthTex   = glGetUniformLocation(vol_prog, "uDepthTex")
        vuShadowMap  = glGetUniformLocation(vol_prog, "uShadowMap")
        vuInvProj    = glGetUniformLocation(vol_prog, "uInvProjection")
        vuInvView    = glGetUniformLocation(vol_prog, "uInvView")
        vuCamPos     = glGetUniformLocation(vol_prog, "uCamPos")
        vuLightDir   = glGetUniformLocation(vol_prog, "uLightDir")
        vuLightColor = glGetUniformLocation(vol_prog, "uLightColor")
        vuLightVP    = glGetUniformLocation(vol_prog, "uLightVP")
        vuScattering = glGetUniformLocation(vol_prog, "uScattering")
        vuNoiseTex   = glGetUniformLocation(vol_prog, "uNoiseTex")
        vuTime       = glGetUniformLocation(vol_prog, "uTime")
        vuResolution = glGetUniformLocation(vol_prog, "uResolution")
    print(f"  体积光半分辨率 {VOL_W}×{VOL_H} 就绪")

    # ---- GPU 计算着色器加速物理高度查询 ----
    gpu_height_init()

    # ---- Bloom 后处理系统（HDR FBO + 半分辨率乒乓 FBO） ----
    print("初始化 Bloom 后处理...")
    # 场景 HDR FBO（全分辨率，RGBA16F + 深度）
    hdr_fbo = glGenFramebuffers(1)
    hdr_tex = glGenTextures(1)
    hdr_depth = glGenTextures(1)  # 改为纹理（兼容 GLES 和某些驱动）
    glBindTexture(GL_TEXTURE_2D, hdr_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, ww, wh, 0, GL_RGBA, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glBindTexture(GL_TEXTURE_2D, hdr_depth)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, ww, wh, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, None)
    glBindTexture(GL_TEXTURE_2D, 0)
    glBindFramebuffer(GL_FRAMEBUFFER, hdr_fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, hdr_tex, 0)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, hdr_depth, 0)
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    if status != GL_FRAMEBUFFER_COMPLETE:
        print(f"  Bloom HDR FBO 不完整: {status}")
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    # 半分辨率乒乓 FBO（用于亮部提取 + 高斯模糊）
    blur_w, blur_h = ww // 2, wh // 2
    if blur_w < 1: blur_w = 1
    if blur_h < 1: blur_h = 1

    ping_fbo = glGenFramebuffers(1)
    ping_tex = glGenTextures(1)
    pong_fbo = glGenFramebuffers(1)
    pong_tex = glGenTextures(1)

    for fbo, tex in [(ping_fbo, ping_tex), (pong_fbo, pong_tex)]:
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, blur_w, blur_h, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print(f"  Bloom 模糊 FBO 不完整: {status}")
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    print(f"  Bloom 模糊分辨率: {blur_w}x{blur_h}")

    # ---- LDR FBO（色调映射 + 分级的结果，供 FXAA 收边后再上屏） ----
    ldr_fbo = glGenFramebuffers(1)
    ldr_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, ldr_tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, ww, wh, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glBindFramebuffer(GL_FRAMEBUFFER, ldr_fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ldr_tex, 0)
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    if status != GL_FRAMEBUFFER_COMPLETE:
        print(f"  LDR FBO 不完整: {status}")
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    glBindTexture(GL_TEXTURE_2D, 0)

    # 全屏四边形 VAO（用于 Bloom 后处理）
    quad_verts = np.array([
        -1.0, -1.0,  0.0, 0.0,
         1.0, -1.0,  1.0, 0.0,
         1.0,  1.0,  1.0, 1.0,
        -1.0, -1.0,  0.0, 0.0,
         1.0,  1.0,  1.0, 1.0,
        -1.0,  1.0,  0.0, 1.0,
    ], dtype=np.float32)
    quad_vao, quad_vbo = create_vao(quad_verts, [2, 2])  # pos2 + uv2
    quad_cnt = len(quad_verts) // 4

    # 编译 Bloom 着色器
    bloom_prog_bright = create_shader_program(BLOOM_VERTEX_SRC, BLOOM_BRIGHT_FRAG_SRC)
    bloom_prog_blur = create_shader_program(BLOOM_VERTEX_SRC, BLOOM_BLUR_FRAG_SRC)
    bloom_prog_composite = create_shader_program(BLOOM_VERTEX_SRC, BLOOM_COMPOSITE_FRAG_SRC)
    if not bloom_prog_bright or not bloom_prog_blur or not bloom_prog_composite:
        print("  Bloom 着色器编译失败!")
        return 1
    bm_uScene = glGetUniformLocation(bloom_prog_bright, "uSceneTex")
    bm_uThreshold = glGetUniformLocation(bloom_prog_bright, "uThreshold")
    bl_uImage = glGetUniformLocation(bloom_prog_blur, "uImage")
    bl_uHorizontal = glGetUniformLocation(bloom_prog_blur, "uHorizontal")
    bc_uScene = glGetUniformLocation(bloom_prog_composite, "uSceneTex")
    bc_uBloom = glGetUniformLocation(bloom_prog_composite, "uBloomTex")
    bc_uBloomStrength = glGetUniformLocation(bloom_prog_composite, "uBloomStrength")
    bc_uExposure = glGetUniformLocation(bloom_prog_composite, "uExposure")
    bc_uContrast   = glGetUniformLocation(bloom_prog_composite, "uContrast")
    bc_uSaturation = glGetUniformLocation(bloom_prog_composite, "uSaturation")
    bc_uVignette   = glGetUniformLocation(bloom_prog_composite, "uVignette")
    print("  Bloom 后处理初始化完成")

    # ---- FXAA ----
    fxaa_prog = create_shader_program(BLOOM_VERTEX_SRC, FXAA_FRAG_SRC)
    if not fxaa_prog:
        print("  FXAA 着色器编译失败!")
        return 1
    fx_uTex   = glGetUniformLocation(fxaa_prog, "uTex")
    fx_uTexel = glGetUniformLocation(fxaa_prog, "uTexel")
    print("  FXAA 抗锯齿就绪")

    # ---- 光锥体积光（着色器编译）----
    print("编译光锥着色器...")
    # 光锥 VAO 已在上面构建（见"构建光锥体积光"）
    cone_prog = create_shader_program(LIGHT_CONE_VERTEX_SRC, LIGHT_CONE_FRAG_SRC)
    if not cone_prog:
        print("  光锥着色器编译失败!")
        return 1
    cone_uMVP = glGetUniformLocation(cone_prog, "uMVP")
    print("  光锥着色器就绪")

    # ---- ★ GPU 粒子系统（Compute Shader + SSBO） ----
    print("初始化 GPU 粒子系统...")
    PARTICLE_MAX = 32  # 对齐 workgroup 的 32 线程
    PARTICLE_FLOATS = 12  # pos(3)+vel(3)+life+maxLife+size+type+pad(1)
    PARTICLE_STRIDE = PARTICLE_FLOATS * 4  # 48 bytes per particle

    # 粒子状态数组（Python 侧读写，每帧通过 SSBO 上传到 GPU）
    particle_state = np.zeros((PARTICLE_MAX, PARTICLE_FLOATS), dtype=np.float32)
    # 格式: [px,py,pz, vx,vy,vz, life,maxLife, size, type, pad, pad]
    # life<=0 表示该槽位空闲
    particle_count = 0  # 当前活跃粒子数

    # 顶点 VBO（预分配满容量，不再每帧重新分配）
    p_vbo = glGenBuffers(1)
    p_vao = glGenVertexArrays(1)
    glBindVertexArray(p_vao)
    glBindBuffer(GL_ARRAY_BUFFER, p_vbo)
    p_vbo_size = PARTICLE_MAX * 6 * 11 * 4  # 32 * 6顶点 * 11float * 4bytes
    glBufferData(GL_ARRAY_BUFFER, p_vbo_size, None, GL_DYNAMIC_DRAW)
    stride = 11 * 4; off = 0
    for i, sz in enumerate([3,3,2,3]):
        glVertexAttribPointer(i, sz, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(off))
        glEnableVertexAttribArray(i)
        off += sz * 4
    glBindVertexArray(0)

    # 粒子状态 SSBO（binding=12，每帧 glBufferSubData）
    particle_state_ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, particle_state_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, particle_state.nbytes, particle_state, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, particle_state_ssbo)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    # 计数器 SSBO（binding=14，仅 count 字段）
    particle_counter = np.zeros(1, dtype=np.uint32)
    particle_counter_ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, particle_counter_ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, particle_counter.nbytes, particle_counter, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, particle_counter_ssbo)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    # 编译粒子 Compute Shader
    particle_cs_prog = create_compute_program(PARTICLE_COMPUTE_SRC)
    if not particle_cs_prog:
        print("  粒子 Compute Shader 编译失败!")
    else:
        pcs_uDt = glGetUniformLocation(particle_cs_prog, "cs_dt")
        pcs_uEye = glGetUniformLocation(particle_cs_prog, "cs_camEye")
        print("  粒子 Compute Shader 编译成功")

    # ---- 轮胎痕迹系统 ----
    TIRE_MAX = 500
    tire_marks = []   # [(x, z, angle), ...] 位置+方向
    tire_vao = glGenVertexArrays(1)
    tire_vbo = glGenBuffers(1)
    glBindVertexArray(tire_vao)
    glBindBuffer(GL_ARRAY_BUFFER, tire_vbo)
    # 预分配满容量，避免每帧 glBufferData 重新分配
    tire_vbo_size = TIRE_MAX * 6 * 11 * 4
    glBufferData(GL_ARRAY_BUFFER, tire_vbo_size, None, GL_DYNAMIC_DRAW)
    stride = 11 * 4; off = 0
    for i, sz in enumerate([3,3,2,3]):
        glVertexAttribPointer(i, sz, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(off))
        glEnableVertexAttribArray(i)
        off += sz * 4
    glBindVertexArray(0)
    tire_buf = np.zeros(TIRE_MAX * 6 * 11, dtype=np.float32)
    tire_emit_n = 0  # 发射计数器，控制频率

    total_verts = tcnt + grcnt + akcnt + sum(g['count'] for g in car_groups)
    print(f"总顶点数: {total_verts}")

    # ---- 投影（地形覆盖±500，雾遮远端，far=2000足够） ----
    aspect = ww / wh
    projection = mat4_perspective(np.radians(65.0), aspect, 0.5, 2000.0)

    # ---- 赛车状态（山顶起点：赛道路径第一个控制点） ----
    path_start = get_akina_path()
    start_x = path_start[0][0] if path_start else 3.0
    start_z = path_start[0][1] if path_start else 3.0
    start_y = path_start[0][2] + 0.1 if path_start else 18.0
    car_pos = [start_x, start_y, start_z]
    # 根据前两个路径点计算起始朝向，适配 -sin/cos 运动公式
    if len(path_start) > 1:
        fdx = path_start[1][0] - path_start[0][0]
        fdz = path_start[1][1] - path_start[0][1]
        if abs(fdx) > 0.001 or abs(fdz) > 0.001:
            car_angle = math.atan2(-fdx, -fdz)
        else:
            car_angle = -0.68
    else:
        car_angle = -0.68  # 回退默认角度
    car_speed = 0.0
    # ★ 头文字D 街机物理引擎（自行车模型 + Pacejka + 载荷转移 + 摩擦圆）
    phys = VehiclePhysics(start_x, start_z, car_angle)
    last_safe = [start_x, start_z, car_angle]  # NaN 保险丝：保存最近有效状态

    # ---- 摄像机平滑跟踪 ----
    FOLLOW_RATE = 12.0     # 跟踪速率：越大跟得越紧
    cam_dist_base = 1.2 * _car_scale_factor
    cam_h_base = 1.1 * _car_scale_factor   # ★ 向下平移20%
    cam_sm = 20.0
    # 状态
    cam_pos = np.array([car_pos[0], car_pos[1] + cam_h_base, car_pos[2]], dtype=f32)
    cam_look_h = car_pos[1] + 0.25  # 视线目标高度（带平滑）
    cam_roll = 0.0         # 视觉侧倾角
    cam_yaw = car_angle    # 相机偏航角（锁死基准，循环内只做有界差值跟踪）
    cam_lateral_offset = 0.0  # 弯道侧向偏移

    # ---- 摄像机滚轮调节系统（C 键切模式，滚轮调值，自动保存）----
    _cam_config_path = os.path.join(os.path.dirname(__file__), ".cache", "camera_config.json")
    _cam_offset_dist = 0.0
    _cam_offset_height = 0.0
    _cam_offset_lookahead = 0.0
    _cam_adjust_mode = 0  # 0=距离 1=高度 2=预瞄
    _cam_mode_names = ["距离", "高度", "预瞄"]
    _cam_steps = [0.05, 0.05, 0.2]
    try:
        if os.path.exists(_cam_config_path):
            with open(_cam_config_path, 'r') as _f:
                _d = json.load(_f)
                _cam_offset_dist = _d.get('dist', 0.0)
                _cam_offset_height = _d.get('height', 0.0)
                _cam_offset_lookahead = _d.get('lookahead', 0.0)
            print(f"[摄像机] 加载配置: dist={_cam_offset_dist:+.2f}  height={_cam_offset_height:+.2f}  lookahead={_cam_offset_lookahead:+.1f}")
    except Exception as _e:
        print(f"[摄像机] 加载配置失败: {_e}")
    def _save_cam_config():
        try:
            os.makedirs(os.path.dirname(_cam_config_path), exist_ok=True)
            with open(_cam_config_path, 'w') as _f:
                json.dump({'dist': _cam_offset_dist, 'height': _cam_offset_height, 'lookahead': _cam_offset_lookahead}, _f)
        except Exception:
            pass

    # ---- 车辆俯仰平滑（防坡道颠簸） ----
    car_pitch_sm = 8.0
    car_pitch_smooth = 0.0

    last_time = sdl2.SDL_GetTicks() / 1000.0

    # 光源方向（仰角 50°，从西南方向来）
    light_dir = np.array([-0.45, 0.76, 0.47], dtype=f32)
    light_dir = light_dir / np.linalg.norm(light_dir)
    # 头文字D 街机影调：暖阳 + 充足的天光环境光（阴影里也要能读出形体）
    ambient_col = np.array([1.38, 1.52, 1.72], dtype=f32)
    light_color = np.array([2.10, 1.90, 1.62], dtype=f32)   # 午后暖阳

    # 模型矩阵（单位阵用于场景）
    identity_mat = mat4_identity()

    # ============================================================
    # 小地图（Minimap）初始化（独立 HUD 模块）
    # ============================================================
    print("初始化小地图...")
    akina_full_path = get_akina_path()
    minimap = Minimap(ui_coord, akina_full_path)

    # ============================================================
    # 仪表盘（速度表 + 档位显示）初始化
    # ============================================================
    print("初始化仪表盘...")
    text_renderer = TextRenderer(ui_coord)
    speedometer = Speedometer(ui_coord, text_renderer)
    gear_display = GearDisplay(ui_coord, text_renderer)

    # ============================================================
    # 弯道提示系统初始化
    # ============================================================
    print("初始化弯道提示系统...")
    corner_analyzer = CornerAnalyzer()
    corner_warning = CornerWarning(ui_coord, text_renderer)

    # ============================================================
    # 主菜单 + 设置面板初始化
    # ============================================================
    main_menu = MainMenu(ui_coord, text_renderer)
    settings_panel = SettingsPanel(ui_coord, text_renderer)
    game_active = False  # 是否处于游戏进行中状态

    print("赛道+城市环境已就绪！")
    print("W=加速  S=刹车  A/D=转向  ESC=退出")

    # 禁用 GC 防止周期性卡顿
    gc.disable()

    # 帧率统计
    fps_frame_count = 0
    fps_timer = 0.0
    fps_interval = 0.2  # 每200ms报告一次

    # 云层漂移用的累计时间
    sky_time = 0.0

    # 缓存赛道路径（避免主循环内反复获取）
    akina_path = get_akina_path()
    debug_mode = 0  # ★ 地形调试模式：0=正常, 1=草, 2=岩, 3=土, 4=碎石, 5=权重

    running = True
    while running:
        event = sdl2.SDL_Event()
        while sdl2.SDL_PollEvent(event):
            if event.type == sdl2.SDL_QUIT: running = False
            elif event.type == sdl2.SDL_KEYDOWN:
                if event.key.keysym.sym == sdl2.SDLK_ESCAPE:
                    if settings_panel.is_active:
                        settings_panel.set_active(False)
                        main_menu.set_active(True)
                    elif game_active:
                        game_active = False
                        main_menu.set_active(True)
                    else:
                        running = False
                # ★ 地形调试模式切换：0-5 数字键
                elif event.key.keysym.sym == sdl2.SDLK_0: debug_mode = 0; print(f"调试模式: 正常 ({debug_mode})")
                elif event.key.keysym.sym == sdl2.SDLK_1: debug_mode = 1; print(f"调试模式: 草地纹理 ({debug_mode})")
                elif event.key.keysym.sym == sdl2.SDLK_2: debug_mode = 2; print(f"调试模式: 岩石纹理 ({debug_mode})")
                elif event.key.keysym.sym == sdl2.SDLK_3: debug_mode = 3; print(f"调试模式: 泥土纹理 ({debug_mode})")
                elif event.key.keysym.sym == sdl2.SDLK_4: debug_mode = 4; print(f"调试模式: 碎石纹理 ({debug_mode})")
                elif event.key.keysym.sym == sdl2.SDLK_5: debug_mode = 5; print(f"调试模式: 权重可视化 ({debug_mode})")
                # ★ 摄像机调节模式切换：C 键
                elif event.key.keysym.sym == sdl2.SDLK_c:
                    _cam_adjust_mode = (_cam_adjust_mode + 1) % 3
                    print(f"[摄像机] 调节模式: {_cam_mode_names[_cam_adjust_mode]}")
            elif event.type == sdl2.SDL_MOUSEMOTION:
                mx, my = event.motion.x, event.motion.y
                if main_menu.is_active:
                    main_menu.on_mouse_move(mx, my)
                elif settings_panel.is_active:
                    settings_panel.on_mouse_move(mx, my)
            elif event.type == sdl2.SDL_MOUSEBUTTONDOWN:
                if event.button.button == sdl2.SDL_BUTTON_LEFT:
                    mx, my = event.button.x, event.button.y
                    if main_menu.is_active:
                        btn_idx = main_menu.on_mouse_click(mx, my)
                        if btn_idx == 0:      # 开始游戏
                            main_menu.set_active(False)
                            game_active = True
                        elif btn_idx == 1:    # 设置
                            main_menu.set_active(False)
                            settings_panel.set_active(True)
                        elif btn_idx == 2:    # 退出
                            running = False
                    elif settings_panel.is_active:
                        result = settings_panel.on_mouse_click(mx, my)
                        if result == "back":
                            settings_panel.set_active(False)
                            main_menu.set_active(True)
            elif event.type == sdl2.SDL_MOUSEWHEEL:
                _step = _cam_steps[_cam_adjust_mode]
                _delta = event.wheel.y * _step
                if _cam_adjust_mode == 0:
                    _cam_offset_dist = max(-1.0, min(2.0, _cam_offset_dist + _delta))
                elif _cam_adjust_mode == 1:
                    _cam_offset_height = max(-1.0, min(1.0, _cam_offset_height + _delta))
                elif _cam_adjust_mode == 2:
                    _cam_offset_lookahead = max(-3.0, min(6.0, _cam_offset_lookahead + _delta))
                _save_cam_config()

        now = sdl2.SDL_GetTicks() / 1000.0
        dt = min(now - last_time, 0.05)
        sky_time += dt
        last_time = now
        keys = sdl2.SDL_GetKeyboardState(None)

        # ---- 头文字D 物理引擎输入（菜单中不响应） ----
        if game_active:
            throttle = 1.0 if keys[sdl2.SDL_SCANCODE_W] else 0.0
            brake    = 1.0 if keys[sdl2.SDL_SCANCODE_S] else 0.0
            steer_in = (1.0 if keys[sdl2.SDL_SCANCODE_A] else 0.0) \
                     - (1.0 if keys[sdl2.SDL_SCANCODE_D] else 0.0)
            handbrake = bool(keys[sdl2.SDL_SCANCODE_SPACE])
        else:
            throttle = 0.0
            brake = 0.0
            steer_in = 0.0
            handbrake = False

        # ---- 环境采样（前后轴双点坡度，对发夹弯/路缘台阶/沟槽免疫）----
        sd_now = signed_distance(car_pos[0], car_pos[2])
        surface_mu = 1.0 if sd_now <= 0.5 else (0.68 if sd_now <= 3.0 else 0.55)
        from track_bounds import surface_height_at
        s_a_, c_a_ = math.sin(car_angle), math.cos(car_angle)
        half_wb = phys.wheelbase * 0.5
        h_front = surface_height_at(car_pos[0] - s_a_ * half_wb, car_pos[2] - c_a_ * half_wb)
        h_rear  = surface_height_at(car_pos[0] + s_a_ * half_wb, car_pos[2] + c_a_ * half_wb)
        pitch_slope = _clamp((h_front - h_rear) / phys.wheelbase, -0.35, 0.35)

        # ---- 物理步进（120Hz 子步进解耦帧率） ----
        phys.update(dt, throttle, brake, steer_in, handbrake,
                    surface_mu=surface_mu, pitch_slope=pitch_slope)

        # ---- ★ NaN 保险丝：任何发散在传播到 KD-tree 之前被截断 ----
        if not (math.isfinite(phys.x) and math.isfinite(phys.z)):
            phys.set_state(last_safe[0], last_safe[1], last_safe[2])
        else:
            last_safe[0], last_safe[1], last_safe[2] = phys.x, phys.z, phys.yaw

        # ---- 导出兼容旧渲染代码 ----
        car_pos[0], car_pos[2] = phys.x, phys.z
        car_angle = phys.yaw
        car_speed = phys.v_long

        # ---- ★ 赛道空气墙 v3（四角检测 + 车头对齐，防楔死）----
        WALL_MARGIN = 0.25  # ★ 收紧空气墙至紧贴路缘，消除漂移穿模吸附感
                            #   （石墩子已移除；急弯内侧的实际边界是侧沟外壁，
                            #     比空气墙更靠内 ~0.29m，由 gutter_hold 负责）
        wall_impact = 0.0
        _eps_w = 0.4

        def _wall_normal(x, z):
            gnx = (signed_distance(x + _eps_w, z) - signed_distance(x - _eps_w, z)) / (2.0 * _eps_w)
            gnz = (signed_distance(x, z + _eps_w) - signed_distance(x, z - _eps_w)) / (2.0 * _eps_w)
            _gl = math.hypot(gnx, gnz)
            return (gnx / _gl, gnz / _gl) if _gl > 1e-6 else (0.0, 0.0)

        s_w, c_w = math.sin(phys.yaw), math.cos(phys.yaw)
        # ★ 四角检测：车尾/车侧插墙时法向力也能生效，不再只有中心+车头盲区
        half_hw = phys.wheelbase * 0.5 * 0.5      # 横向半宽 ≈0.6m
        corners = [
            (-half_wb, -half_hw),   # 左前
            (-half_wb,  half_hw),   # 右前
            ( half_wb, -half_hw),   # 左后
            ( half_wb,  half_hw),   # 右后
        ]
        worst_pen = 0.0
        worst_n = (0.0, 0.0)
        for c_long, c_lat in corners:
            # 车体系 → 世界：纵向 (-sin, -cos)，横向 (cos, -sin)
            px = phys.x + (-s_w * c_long) + (c_w * c_lat)
            pz = phys.z + (-c_w * c_long) + (-s_w * c_lat)
            # ★ 起终点场地走廊内边界外推到建筑线（车能开上缓冲带）
            d = signed_distance(px, pz) - venue_wall_margin(px, pz)
            if d > worst_pen:
                worst_pen = d
                worst_n = _wall_normal(px, pz)
        if worst_pen > 0.0 and (worst_n[0] != 0.0 or worst_n[1] != 0.0):
            wall_impact = phys.wall_collide(
                worst_n[0], worst_n[1], worst_pen, dt=dt)

        # 车头楔入检测补充：四角漏检但车头深插 → 对齐墙面（核心防卡死）
        sd_front = signed_distance(phys.x - s_w * half_wb, phys.z - c_w * half_wb)
        _fm = venue_wall_margin(phys.x - s_w * half_wb,
                                phys.z - c_w * half_wb)
        if sd_front > _fm + 0.2:
            nxf, nzf = _wall_normal(phys.x - s_w * half_wb, phys.z - c_w * half_wb)
            if nxf != 0.0 or nzf != 0.0:
                wall_impact = max(wall_impact, phys.wall_nose_collide(
                    nxf, nzf, sd_front - _fm, dt=dt))

        car_pos[0], car_pos[2] = phys.x, phys.z
        # ★ 由 sync_speed_after_boundary 统一处理： boundary 吃掉的速度必须写回
        #   漂移模型的 _drift_speed，否则下一帧被原样覆盖 → 车被钉住/莫名刹停；
        #   impact > 1.0 时内部照样交还抓地模型（原逻辑，不再双写）。
        phys.sync_speed_after_boundary(wall_impact)
        if wall_impact > 2.0:
            print(f"  撞墙! 冲击 {wall_impact:.1f} m/s")

        # ---- ★ 沟渠跑法：侧沟咬地（横向支撑 + 外壁硬约束）----
        # ★ v2：检测内侧前轮而非中心+半轮距，精准复现"前轮卡沟"物理反馈
        #   s_w/c_w 来自上方空气墙四角检测，是 phys.yaw 的正余弦
        _front_axle_x = car_pos[0] - s_w * phys.wheelbase * 0.5
        _front_axle_z = car_pos[2] - c_w * phys.wheelbase * 0.5
        # ★ v4：探针从"前轴 sd + 常量 0.75"改成**真实前轮位置**。
        #   旧估算假设车与赛道平行 —— 甩大角度时前轮的横向展开是
        #   0.75·cosβ + 1.2·sinβ（β=40° 时 1.35m，比 0.75 大 80%），
        #   ⇒ 车还在路缘内侧就被判成"前轮翻出沟外壁"，硬投影+全力横撑
        #   把车摁住；反过来某些姿态又漏检，轮子穿出沟外壁。
        #   两个前轮都量、取 sd 更大的那个（靠路缘那侧恒为 sd 更大的一侧）。
        _rx, _rz = math.cos(phys.yaw), -math.sin(phys.yaw)   # 车身右向
        _HT = 0.75                                            # 半轮距
        _probe_sd = max(
            signed_distance(_front_axle_x + _rx * _HT, _front_axle_z + _rz * _HT),
            signed_distance(_front_axle_x - _rx * _HT, _front_axle_z - _rz * _HT))
        _g_depth, _g_pen = gutter_hold_depth(_probe_sd)
        # ★ v3：侧沟 v6 起只在"急弯内侧"存在 —— 直道 / 弯道外侧根本没有沟，
        #   物理必须跟着关掉，否则车会被一条看不见的沟咬住（视觉与手感脱钩）。
        #   判据与 _build_drainage_ditches_impl 同源（gutter_present）。
        _gutter_ok = False
        if _g_depth > 0.0 or _g_pen > 0.0:
            _g_i, _hw_i, _nx_i, _nz_i, _lat_i, _ = nearest_track_info(
                car_pos[0], car_pos[2])
            _sgn = 1.0 if _lat_i >= 0.0 else -1.0      # 指向路外
            _gutter_ok = gutter_present(_g_i, _sgn)
            if _gutter_ok:
                phys.gutter_hold(_nx_i * _sgn, _nz_i * _sgn,
                                 _g_pen, _g_depth, dt)
                # ★ 同空气墙：沟吃掉的速度必须写进漂移模型，否则等于没吃
                phys.sync_speed_after_boundary(0.0)
                car_pos[0], car_pos[2] = phys.x, phys.z

        # ---- 车辆高度 + 俯仰（复用物理引擎坡度采样）----
        ASPHALT_THICKNESS = 0.15
        from track_bounds import surface_height_at
        # ★ 起终点场地走廊：铺面是场地自己的（缓冲带/平台），
        #   surface_height_at 的路棱柱外回退不认它 —— 用 venue_surface_at 兜底
        _vsurf = venue_surface_at(car_pos[0], car_pos[2])
        car_y = (_vsurf if _vsurf is not None
                 else surface_height_at(car_pos[0], car_pos[2]))
        # 外轮压进沟腔 → 车身下沉（gutter_hold 可能已修正位置，故重算 sd）
        _wheel_sd = signed_distance(car_pos[0], car_pos[2]) + 0.75
        car_pos[1] = car_y + ASPHALT_THICKNESS + (
            gutter_dip(_wheel_sd) if _gutter_ok else 0.0)

        # ★ 修复：mat4_rotate_x 正角 = 车头抬起。pitch_slope>0=上坡→正号。
        #   旧 − 号导致上坡点头，与 ax 项"加速抬头"矛盾——坡度动作被藏住的根因
        car_pitch = pitch_slope + _clamp(phys.ax_s * 0.010, -0.06, 0.08)
        # 对俯仰角做指数平滑，防止坡道突变导致车辆颠簸
        car_pitch_smooth += (car_pitch - car_pitch_smooth) * min(1.0, car_pitch_sm * dt)

        # 车辆模型矩阵：先平移 → 再绕Y轴旋转（朝向）→ 再绕X轴旋转（俯仰）
        model_car = mat4_multiply(
            mat4_multiply(mat4_translate(car_pos[0], car_pos[1], car_pos[2]),
                         mat4_rotate_y(car_angle)),
            mat4_rotate_x(car_pitch_smooth)  # 用平滑后的俯仰角
        )

        # ---- 轮胎痕迹生成 ----
        # 条件：急刹车 或 急转弯（有一定速度时）
        def emit_tire_mark(world_x, world_z):
            if len(tire_marks) >= TIRE_MAX:
                return
            tire_marks.append((world_x, world_z, car_angle))

        # 计算后轮世界坐标
        c_a, s_a = math.cos(car_angle), math.sin(car_angle)
        # 左后轮坐标：车体局部 (-0.52, 0, -0.65)
        lwx = car_pos[0] + (-0.52 * _car_scale_factor) * c_a + (-0.65 * _car_scale_factor) * s_a
        lwz = car_pos[2] - (-0.52 * _car_scale_factor) * s_a + (-0.65 * _car_scale_factor) * c_a
        # 右后轮坐标：车体局部 (0.52, 0, -0.65)
        rwx = car_pos[0] + (0.52 * _car_scale_factor) * c_a + (-0.65 * _car_scale_factor) * s_a
        rwz = car_pos[2] - (0.52 * _car_scale_factor) * s_a + (-0.65 * _car_scale_factor) * c_a

        # ★ 胎痕升级：基于物理引擎的漂移系数 + 手刹锁轮（代替旧版急刹/急转粗糙判断）
        if phys.drift > 0.30 or (handbrake and abs(car_speed) > 3.0):
            tire_emit_n += 1
            if tire_emit_n % 2 == 0:
                emit_tire_mark(lwx, lwz)
                emit_tire_mark(rwx, rwz)

        # ---- ★ GPU 粒子系统：发射 + 更新（Python 侧维护状态，GPU 生成顶点） ----
        # 发射新粒子到 particle_state 空闲槽位
        def _spawn_particle(px, py, pz, vx, vy, vz, life, size, ptype):
            nonlocal particle_count
            if particle_count >= PARTICLE_MAX:
                return
            s = particle_state[particle_count]
            s[0], s[1], s[2] = px, py, pz
            s[3], s[4], s[5] = vx, vy, vz
            s[6], s[7] = life, life
            s[8], s[9] = size, ptype
            particle_count += 1

        # ★ 排气管粒子（油门时持续排放，漂移时烟更浓）
        if keys[sdl2.SDL_SCANCODE_W] and car_speed > 0.3:
            _drift_boost = 1.0 + phys.drift * 2.0  # 漂移放大烟量
            _count = max(1, int(2 * _drift_boost))
            for _ in range(_count):
                if particle_count >= PARTICLE_MAX: break
                rx = random.uniform(-0.15, 0.15); rz = random.uniform(-0.15, 0.15)
                rear_x = car_pos[0] + math.sin(car_angle) * 2.2 * _car_scale_factor + rx
                rear_z = car_pos[2] + math.cos(car_angle) * 2.2 * _car_scale_factor + rz
                rear_y = 0.15 + random.uniform(0, 0.35)
                vx = math.sin(car_angle) * random.uniform(0.3, 1.8) + random.uniform(-0.4, 0.4)
                vz = math.cos(car_angle) * random.uniform(0.3, 1.8) + random.uniform(-0.4, 0.4)
                vy = random.uniform(0.25, 1.0)
                life = random.uniform(0.5, 1.2) * _drift_boost
                size = random.uniform(0.06, 0.18) * _drift_boost
                _spawn_particle(rear_x, rear_y, rear_z, vx, vy, vz, life, size, 0.0)

        # ★ 漂移烟尘粒子（后轮搓地烟，旧版刹车粒子升级版）
        _drift_smoke = phys.drift * 3.0
        if _drift_smoke > 0.5:
            for _ in range(min(3, int(_drift_smoke))):
                if particle_count >= PARTICLE_MAX: break
                rx = random.uniform(-0.3, 0.3); rz = random.uniform(-0.3, 0.3)
                dx = car_pos[0] + math.sin(car_angle) * 1.8 * _car_scale_factor + rx
                dz = car_pos[2] + math.cos(car_angle) * 1.8 * _car_scale_factor + rz
                dy = 0.02 + random.uniform(0, 0.12)
                vx = math.sin(car_angle) * random.uniform(0.5, 2.0) + random.uniform(-0.6, 0.6)
                vz = math.cos(car_angle) * random.uniform(0.5, 2.0) + random.uniform(-0.6, 0.6)
                vy = random.uniform(0.08, 0.4)
                life = random.uniform(0.15, 0.5) * (1.0 + phys.drift)
                size = random.uniform(0.04, 0.10) * (1.0 + phys.drift * 1.5)
                _spawn_particle(dx, dy, dz, vx, vy, vz, life, size, 1.0)

        # ---- ★ 撞墙粒子（火花/尘土）----
        if wall_impact > 2.5:
            for _ in range(min(3, int(wall_impact))):
                if particle_count >= PARTICLE_MAX:
                    break
                _spawn_particle(car_pos[0], car_pos[1] + 0.25, car_pos[2],
                                random.uniform(-1.5, 1.5), random.uniform(0.5, 2.0),
                                random.uniform(-1.5, 1.5),
                                0.35, 0.10, 1.0)

        # 更新粒子物理 + 剔除已死亡粒子（原地 compact）
        alive = 0
        for i in range(particle_count):
            s = particle_state[i]
            s[0] += s[3] * dt; s[1] += s[4] * dt; s[2] += s[5] * dt
            s[6] -= dt  # life 衰减
            if s[6] > 0:
                if alive != i:
                    particle_state[alive] = s.copy()
                alive += 1
        particle_count = alive

        # 上传粒子状态到 SSBO → Dispatch Compute Shader 生成顶点
        if particle_count > 0 and particle_cs_prog:
            ssbo_sub_data(particle_state_ssbo, particle_state[:particle_count])
            particle_counter[0] = particle_count
            ssbo_sub_data(particle_counter_ssbo, particle_counter)
            # 绑定 p_vbo 作为 SSBO（binding=13），供 Compute Shader 写入顶点
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, p_vbo)
            glUseProgram(particle_cs_prog)
            if pcs_uDt >= 0: glUniform1f(pcs_uDt, dt)
            if pcs_uEye >= 0: glUniform3f(pcs_uEye, cam_eye[0], cam_eye[1], cam_eye[2])
            glDispatchCompute(1, 1, 1)
            glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, 0)
            glUseProgram(0)

        # ---- 摄像机固定（统一用高速位）----
        cam_dist = 3.63 + _cam_offset_dist
        cam_h = 1.73 + _cam_offset_height

        # ---- 相机偏航：锁死基准 + 有界滞后差值 ----
        CAM_OFF_MAX = math.radians(9.0)     # 滞后角硬上限（漂移动感全部来源）
        CAM_DAMP    = 7.0                   # 趋近速率（1/s），越大越接近纯锁死
        dyaw = _angle_wrap(car_angle - cam_yaw)
        dyaw = _clamp(dyaw, -CAM_OFF_MAX, CAM_OFF_MAX)   # ★ 误差限幅，绝不累积
        cam_yaw += dyaw * min(1.0, CAM_DAMP * dt)

        # ---- 侧向偏移：恒为 0 ----
        cam_lateral_offset = 0.0

        # ---- 视觉侧倾：压到几乎不可见 ----
        target_roll = -dyaw * 0.003
        cam_roll += (target_roll - cam_roll) * min(1.0, 4.0 * dt)

        # ---- 统一方向标架：整台相机 Rig 挂在车体倾斜平面上 ----
        # 视线俯仰：与车模型完全同源（car_pitch_smooth 正是 model_car 用的那个值）
        view_pitch = car_pitch_smooth

        cy_, sy_ = math.cos(cam_yaw), math.sin(cam_yaw)
        cp, sp = math.cos(view_pitch), math.sin(view_pitch)

        # 带俯仰的前向 / 上向量（二者严格正交，构成车体倾斜平面）
        fwd = np.array([-sy_ * cp, sp, -cy_ * cp], dtype=f32)   # 车头方向
        upc = np.array([ sy_ * sp, cp,  cy_ * sp], dtype=f32)   # 坡面法向（车顶方向）

        # 相机 = 车 - 前向*距离 + 坡面法向*高度
        cam_pos = np.array([car_pos[0] - fwd[0] * cam_dist,
                            car_pos[1] - fwd[1] * cam_dist + upc[1] * cam_h,
                            car_pos[2] - fwd[2] * cam_dist + upc[2] * cam_h], dtype=f32)

        # 装饰量只剩油门/刹车点头，不再掺坡度
        decor = _clamp(phys.ax_s * 0.010, -0.06, 0.08) * 0.35 * cam_dist
        cam_pos[1] += decor

        # 视线目标：同一倾斜平面内
        lookahead_dist = 3.0 + _cam_offset_lookahead
        look_x = car_pos[0] + fwd[0] * lookahead_dist
        look_z = car_pos[2] + fwd[2] * lookahead_dist
        cam_look_h = car_pos[1] + fwd[1] * lookahead_dist + upc[1] * 0.25

        # ---- 构建视图矩阵（带侧倾）----
        cam_eye = cam_pos
        look_target = np.array([look_x, cam_look_h, look_z], dtype=f32)
        # up 向量带侧倾旋转
        up = np.array([math.sin(cam_roll), math.cos(cam_roll), 0], dtype=f32)
        view = mat4_look_at(cam_eye, look_target, up)
        vp = mat4_multiply(projection, view)
        mvp_car = mat4_multiply(vp, model_car)

        # ★★★ Deferred Rendering：阶段A — 几何 Pass → G-Buffer ★★★
        if USE_DEFERRED:
            glBindFramebuffer(GL_FRAMEBUFFER, g_buffer_fbo)
            glViewport(0, 0, ww, wh)
            dbs = [GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3]
            glDrawBuffers(4, dbs)
            # ★ 用透明黑清 G-Buffer，避免全局 clear color（暖灰）污染 materialFlags
            glClearColor(0.0, 0.0, 0.0, 0.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(0.42, 0.40, 0.42, 1.0)

            glUseProgram(geo_prog)
            glUniform1i(guTex, 0)
            glUniform1f(guIsEmissive, 0.0)
            glUniform1f(guIsBuilding, 0.0)
            glUniform1f(guUseTerrainBlend, 0.0)
            glUniform1i(guGrassTex, 1)
            glUniform1i(guRockTex, 2)
            glUniform1i(guDirtTex, 3)
            glUniform1i(guGravelTex, 4)
            glUniform1i(guGrassNrm, 5)
            glUniform1i(guRockNrm, 6)
            glUniform1i(guDirtNrm, 7)
            glUniform1i(guGravelNrm, 12)
            glUniform1i(guBuildingAlbedo, 8)
            glUniform1i(guBuildingNormal, 13)
            glUniform1i(guBuildingAO, 11)
            # ★ 第三阶段：POM uniform 绑定
            glUniform1i(guHeightMap, 9)          # 赛道高度图
            glUniform1i(guTerrainHeightMap, 10)   # 地形高度图
            glUniform1f(guParallaxScale, 0.0)     # 视差强度（默认关闭，有真实高度图时设为 0.02~0.08）
            glUniform1f(guParallaxLayers, 12.0)   # POM 步进层数
            # 绑定默认高度图（中性灰 = 无偏移）
            glActiveTexture(GL_TEXTURE0 + 9)
            glBindTexture(GL_TEXTURE_2D, default_height_tex)
            glActiveTexture(GL_TEXTURE0 + 10)
            glBindTexture(GL_TEXTURE_2D, default_height_tex)
            glUniform1i(guDebugMode, 0)
            # ★★★ 必须上传视点！geo_prog 的 uViewPos 默认是 (0,0,0)，此前一直漏传。
            #     后果：车法线的双面修正、地形 POM 的视线方向全部拿到错误方向 ——
            #     车身法线被系统性翻转（真机 G-Buffer 里 N.y 均值 -0.53、
            #     N·L<0 占 76% ⇒ 车侧面成片死黑），而离屏复现因为显式设了
            #     uViewPos 完全复现不出来。改渲染时若动了这行，先跑
            #     _scratch/_car_check.py 对比 G-Buffer 法线。
            glUniform3fv(guViewPos, 1, cam_eye)
            glUniformMatrix4fv(guMVP, 1, GL_TRUE, vp)
            glUniformMatrix4fv(guModel, 1, GL_TRUE, identity_mat)
            glUniform1i(guIsInstanced, 0)
            # ---- 天空盒（几何阶段：标记天空像素） ----
            sky_view_g = view.copy()
            sky_view_g[0, 3] = 0.0; sky_view_g[1, 3] = 0.0; sky_view_g[2, 3] = 0.0
            sky_vp_g = mat4_multiply(projection, sky_view_g)
            glDepthMask(GL_FALSE)
            glDisable(GL_DEPTH_TEST)
            glUniformMatrix4fv(guMVP, 1, GL_TRUE, sky_vp_g)
            glUniform1f(guMet, 0.0)
            glUniform1f(guRough, 1.0)
            glUniform1f(guHasTex, 0.0)
            glUniform1f(guIsEmissive, 1.0)       # ★ 天空走自发光分支，避免法线 NaN
            glBindVertexArray(skvao)
            glDrawArrays(GL_TRIANGLES, 0, sky_cnt)
            glUniform1f(guIsEmissive, 0.0)       # ★ 恢复
            glEnable(GL_DEPTH_TEST)
            glDepthMask(GL_TRUE)
            glUniformMatrix4fv(guMVP, 1, GL_TRUE, vp)

            # ---- 地形（带地形混合） ----
            glUniform1f(guUseTerrainBlend, 1.0)
            glActiveTexture(GL_TEXTURE0 + 1)
            glBindTexture(GL_TEXTURE_2D, grass_tex)
            glActiveTexture(GL_TEXTURE0 + 2)
            glBindTexture(GL_TEXTURE_2D, rock_tex)
            glActiveTexture(GL_TEXTURE0 + 3)
            glBindTexture(GL_TEXTURE_2D, dirt_tex)
            glActiveTexture(GL_TEXTURE0 + 4)
            glBindTexture(GL_TEXTURE_2D, gravel_tex)
            glActiveTexture(GL_TEXTURE0 + 5)
            glBindTexture(GL_TEXTURE_2D, grass_nrm)
            glActiveTexture(GL_TEXTURE0 + 6)
            glBindTexture(GL_TEXTURE_2D, rock_nrm)
            glActiveTexture(GL_TEXTURE0 + 7)
            glBindTexture(GL_TEXTURE_2D, dirt_nrm)
            glActiveTexture(GL_TEXTURE0 + 12)
            glBindTexture(GL_TEXTURE_2D, gravel_nrm)
            # ★ PBR 地形 roughness/AO 绑定
            if grass_roughness is not None:
                glActiveTexture(GL_TEXTURE0 + 14); glBindTexture(GL_TEXTURE_2D, grass_roughness); glUniform1i(guGrassRough, 14)
            if rock_roughness is not None:
                glActiveTexture(GL_TEXTURE0 + 15); glBindTexture(GL_TEXTURE_2D, rock_roughness); glUniform1i(guRockRough, 15)
            if dirt_roughness is not None:
                glActiveTexture(GL_TEXTURE0 + 16); glBindTexture(GL_TEXTURE_2D, dirt_roughness); glUniform1i(guDirtRough, 16)
            if gravel_roughness is not None:
                glActiveTexture(GL_TEXTURE0 + 17); glBindTexture(GL_TEXTURE_2D, gravel_roughness); glUniform1i(guGravelRough, 17)
            if grass_ao is not None:
                glActiveTexture(GL_TEXTURE0 + 18); glBindTexture(GL_TEXTURE_2D, grass_ao); glUniform1i(guGrassAO, 18)
            if rock_ao is not None:
                glActiveTexture(GL_TEXTURE0 + 19); glBindTexture(GL_TEXTURE_2D, rock_ao); glUniform1i(guRockAO, 19)
            if dirt_ao is not None:
                glActiveTexture(GL_TEXTURE0 + 20); glBindTexture(GL_TEXTURE_2D, dirt_ao); glUniform1i(guDirtAO, 20)
            if gravel_ao is not None:
                glActiveTexture(GL_TEXTURE0 + 21); glBindTexture(GL_TEXTURE_2D, gravel_ao); glUniform1i(guGravelAO, 21)
            glUniform1f(guHasTex, 0.0)
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(1.0, 1.0)
            glUniform1f(guMet, 0.0)
            glUniform1f(guRough, 0.92)
            glBindVertexArray(akvao)
            glDrawArrays(GL_TRIANGLES, 0, akcnt)
            glUniform1f(guUseTerrainBlend, 0.0)
            glActiveTexture(GL_TEXTURE0)
            glDisable(GL_POLYGON_OFFSET_FILL)

            # ---- 赛道（保留纹理采样） ----
            glUniform1f(guMet, 0.0)
            glUniform1f(guRough, 0.98)
            glUniform1f(guHasTex, 1.0)
            # ★ PBR 赛道通道绑定（有真实纹理用真实，否则用占位防采到 albedo）
            glActiveTexture(GL_TEXTURE0 + 14)
            glBindTexture(GL_TEXTURE_2D, track_roughness if track_roughness is not None else pbr_default_rough_tex)
            glUniform1i(guTrackRough, 14)
            glActiveTexture(GL_TEXTURE0 + 15)
            glBindTexture(GL_TEXTURE_2D, track_metallic if track_metallic is not None else pbr_default_metal_tex)
            glUniform1i(guTrackMetal, 15)
            glActiveTexture(GL_TEXTURE0 + 16)
            glBindTexture(GL_TEXTURE_2D, track_ao if track_ao is not None else pbr_default_ao_tex)
            glUniform1i(guTrackAO, 16)
            glActiveTexture(GL_TEXTURE0 + 18)
            glBindTexture(GL_TEXTURE_2D, track_normal)
            glUniform1i(guTrackNormal, 18)
            if track_height is not None:
                glActiveTexture(GL_TEXTURE0 + 9)
                glBindTexture(GL_TEXTURE_2D, track_height)
                glUniform1i(guHeightMap, 9)
                # glUniform1f(guParallaxScale, 0.04)     # 已禁用：POM 是水波元凶
                glUniform1f(guParallaxLayers, 16.0)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, track_tex)
            glBindVertexArray(tvao)
            glDrawArrays(GL_TRIANGLES, 0, tcnt)

            # ★ 赛道绘制后复位 POM，防止参数泄漏给后续物体（赛车等）
            glUniform1f(guParallaxScale, 0.0)

            # ---- ★ Indirect 批量绘制（护栏 + 路灯 + 路障，材质参数来自 SSBO） ----
            if geo_batch_prog:
                glUseProgram(geo_batch_prog)
                glUniformMatrix4fv(guBatchMVP, 1, GL_TRUE, vp)
                glUniformMatrix4fv(guBatchModel, 1, GL_TRUE, identity_mat)
                glUniform3fv(guBatchViewPos, 1, cam_eye)
                scene_batch.draw()
                glUseProgram(geo_prog)

            # ---- 沿路建筑 ----
            if bcnt > 0:
                b_center_g = building_center
                b_radius_g = building_bound_radius
                planes_g = extract_frustum_planes(vp)
                b_visible_g = frustum_test_sphere(planes_g, b_center_g, b_radius_g)
                if b_visible_g:
                    glUniform1f(guMet, 0.0)
                    glUniform1f(guRough, 0.85)
                    glUniform1f(guHasTex, 0.0)
                    glUniform1f(guIsBuilding, 1.0)
                    glActiveTexture(GL_TEXTURE0 + 8)
                    glBindTexture(GL_TEXTURE_2D, building_wall_tex)
                    glUniform1i(guBuildingAlbedo, 8)
                    glActiveTexture(GL_TEXTURE0 + 11)
                    glBindTexture(GL_TEXTURE_2D, building_ao_tex)
                    glUniform1i(guBuildingAO, 11)
                    glActiveTexture(GL_TEXTURE0 + 13)
                    glBindTexture(GL_TEXTURE_2D, building_normal_tex)
                    glUniform1i(guBuildingNormal, 13)
                    glBindVertexArray(bvao)
                    glDrawArrays(GL_TRIANGLES, 0, bcnt)
                    glUniform1f(guIsBuilding, 0.0)
                    glActiveTexture(GL_TEXTURE0)

            # ---- 杉树（实例化 LOD，GPU 视锥剔除） ----
            glUniform1i(guIsInstanced, 1)
            glUniform1f(guMet, 0.0)
            glUniform1f(guRough, 0.82)
            glUniform1f(guHasTex, 0.0)
            glUniformMatrix4fv(guMVP, 1, GL_TRUE, vp)
            glUniformMatrix4fv(guModel, 1, GL_TRUE, identity_mat)
            # ★ 叶片剪影裁剪（G-Buffer 同样要裁，否则 SSAO/SSR 会拿到实心面片）
            _gf = glGetUniformLocation(geo_prog, "uUseFoliage")
            _gft = glGetUniformLocation(geo_prog, "uFoliageTex")
            if _gf >= 0: glUniform1f(_gf, 1.0)
            if _gft >= 0:
                glUniform1i(_gft, 0)
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, foliage_tex)
            glDisable(GL_CULL_FACE)

            # ★ CPU 视锥剔除（numpy 向量化，比 GPU 剔除 + fence 同步更快）
            planes_g = extract_frustum_planes(vp)
            pos_g = tree_instances[:, :3]
            d_g = pos_g @ planes_g[:, :3].T + planes_g[:, 3]
            TREE_BR_G = 12.0
            vis_mask_g = np.all(d_g >= -TREE_BR_G, axis=1)

            cx_g, cy_g, cz_g = cam_eye
            pos_g = tree_instances[:, :3]
            dist2_g = (pos_g[:, 0] - cx_g)**2 + (pos_g[:, 1] - cy_g)**2 + (pos_g[:, 2] - cz_g)**2
            dist_g = np.sqrt(dist2_g)
            _draw_tree_batches(_tree_band_masks(dist_g), vis_mask_g)
            glEnable(GL_CULL_FACE)
            if _gf >= 0: glUniform1f(_gf, 0.0)
            glUniform1i(guIsInstanced, 0)

            # ---- 赛车（★ RX-7 GLB：使用纹理采样） ----
            glDisable(GL_CULL_FACE)              # ★ GLB 绕序混乱，先关剔除（fix_car_winding 后会根治）
            # ★ 双保险：把赛道 PBR 单元换成占位纹理，防将来改 shader 时再踩
            glActiveTexture(GL_TEXTURE0 + 14); glBindTexture(GL_TEXTURE_2D, pbr_default_rough_tex)
            glActiveTexture(GL_TEXTURE0 + 15); glBindTexture(GL_TEXTURE_2D, pbr_default_metal_tex)
            glActiveTexture(GL_TEXTURE0 + 16); glBindTexture(GL_TEXTURE_2D, pbr_default_ao_tex)
            glActiveTexture(GL_TEXTURE0 + 18); glBindTexture(GL_TEXTURE_2D, default_height_tex)
            glUniform1f(guIsCar, 1.0)            # ★ 进入车辆分支，切断对赛道 PBR/POM 的采样
            glUniform1f(guMet, 0.20)             # GLB 金属粗糙度纹理 G 通道均值 ≈0.20
            glUniform1f(guRough, 0.40)           # 0.40 收窄镜面瓣缓解过曝裁剪
            glUniform1f(guHasTex, 1.0)           # 使用 GLB 漫反射纹理
            glUniformMatrix4fv(guMVP, 1, GL_TRUE, vp)
            glUniformMatrix4fv(guModel, 1, GL_TRUE, model_car)
            _car_ix = 0
            for g in car_groups:
                if g['is_light'] or _CAR_HIDE:
                    continue
                glUniform1f(guHasTex, g['has_tex'])
                if g['tex']:
                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, g['tex'])
                if _GB_DEBUG and not _gb_dbg_state.get('car_info'):
                    _gb_dbg_state['car_info'] = True
                    _dd = os.path.dirname(os.path.abspath(__file__))
                    _tw = int(glGetTexLevelParameteriv(GL_TEXTURE_2D, 0,
                                                       GL_TEXTURE_WIDTH))
                    _th = int(glGetTexLevelParameteriv(GL_TEXTURE_2D, 0,
                                                       GL_TEXTURE_HEIGHT))
                    if _tw > 0 and _th > 0:
                        _ti = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB,
                                            GL_UNSIGNED_BYTE)
                        _ta = np.frombuffer(_ti, dtype=np.uint8).reshape(
                            _th, _tw, 3)[::-1]
                        Image.fromarray(_ta).save(
                            os.path.join(_dd, f'_dbg_cartex_{_car_ix}.png'))
                        print(f"[GBDEBUG] 车#{_car_ix} 贴图 {_tw}x{_th} id={g['tex']} "
                              f"RGB均值=({_ta[...,0].mean():.0f},"
                              f"{_ta[...,1].mean():.0f},{_ta[...,2].mean():.0f})")
                    else:
                        print(f"[GBDEBUG] 车#{_car_ix} 没有绑定 2D 贴图 "
                              f"id={g['tex']}")
                    try:
                        _mc = np.asarray(model_car, dtype=np.float64)
                        _m3 = _mc[:3, :3]
                        _det = float(np.linalg.det(_m3))
                        print(f"[GBDEBUG] car_angle={car_angle:.3f}rad "
                              f"car_pitch={car_pitch:.3f}rad "
                              f"pitch_smooth={car_pitch_smooth:.3f}rad",
                              flush=True)
                        print(f"[GBDEBUG] model_car 旋转部分=\n"
                              f"{np.round(_m3, 3)}\n  det={_det:.4f} "
                              f"(<0 表示含镜像/翻转)", flush=True)
                        _up_car = _m3 @ np.array([0.0, 1.0, 0.0])
                        print(f"[GBDEBUG] 车的上方向被映射到 = "
                              f"({_up_car[0]:.3f},{_up_car[1]:.3f},"
                              f"{_up_car[2]:.3f})  ← y 应为 +1", flush=True)
                        print(f"[GBDEBUG] 车中心世界坐标="
                              f"{np.round(_mc[:3,3],2)}", flush=True)
                    except Exception as _e:
                        print(f"[GBDEBUG] 姿态打印失败: {_e!r}", flush=True)

                    try:
                        # ★ 直接回读车 VAO 的顶点数据，确认法线在 VAO 里是什么样
                        def _gai(_idx, _pn):
                            return int(np.asarray(
                                glGetVertexAttribiv(_idx, _pn)).reshape(-1)[0])
                        glBindVertexArray(g['vao'])
                        _asz = _gai(1, GL_VERTEX_ATTRIB_ARRAY_SIZE)
                        _ast = _gai(1, GL_VERTEX_ATTRIB_ARRAY_STRIDE)
                        _anm = _gai(1, GL_VERTEX_ATTRIB_ARRAY_NORMALIZED)
                        _avb = _gai(1, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING)
                        _aen = _gai(1, GL_VERTEX_ATTRIB_ARRAY_ENABLED)
                        print(f"[GBDEBUG] attrib1: enabled={_aen} size={_asz} "
                              f"stride={_ast} normalized={_anm} vbo={_avb}",
                              flush=True)
                        if _avb and _ast > 0:
                            glBindBuffer(GL_ARRAY_BUFFER, _avb)
                            _cnt = min(30000, g['count'])
                            _buf = glGetBufferSubData(GL_ARRAY_BUFFER, 0,
                                                      _cnt * _ast)
                            _fa = np.frombuffer(_buf, dtype=np.float32).reshape(
                                -1, _ast // 4)
                            _ny = _fa[:, 4]      # [x,y,z, nx,ny,nz, uv, col]
                            _nx = _fa[:, 3]
                            _nz = _fa[:, 5]
                            print(f"[GBDEBUG] VAO 前{len(_fa)}顶点 法线 "
                                  f"nx={_nx.mean():.3f} ny={_ny.mean():.3f} "
                                  f"nz={_nz.mean():.3f} ny<0占比="
                                  f"{100.0*(_ny<0).mean():.1f}% 长度="
                                  f"{np.linalg.norm(_fa[:,3:6],axis=1).mean():.3f}",
                                  flush=True)
                            _vx = _fa[:, 0]; _vy = _fa[:, 1]; _vz = _fa[:, 2]
                            print(f"[GBDEBUG] VAO 位置范围 x[{_vx.min():.2f},"
                                  f"{_vx.max():.2f}] y[{_vy.min():.2f},"
                                  f"{_vy.max():.2f}] z[{_vz.min():.2f},"
                                  f"{_vz.max():.2f}]", flush=True)
                    except Exception as _e:
                        print(f"[GBDEBUG] 顶点回读失败: {_e!r}", flush=True)
                glBindVertexArray(g['vao'])
                glDrawArrays(GL_TRIANGLES, 0, g['count'])
                _car_ix += 1
            glUniform1f(guIsCar, 0.0)            # ★ 恢复
            glEnable(GL_CULL_FACE)               # ★ 恢复剔除，不影响后续物体
            if _GB_DEBUG:
                _gb_debug_car_normals(g_buffer_fbo, ww, wh)

            # ---- 尾灯（刹车时发光） ----
            if keys[sdl2.SDL_SCANCODE_S]:
                glDisable(GL_CULL_FACE)           # ★ 尾灯也用同一套混乱绕序的 GLB
                glUniform1f(guIsCar, 1.0)        # ★ 尾灯也走车辆分支（虽然自发光提前 return，但防呆）
                glUniform1f(guHasTex, 0.0)
                glUniform1f(guIsEmissive, 1.0)
                glUniformMatrix4fv(guMVP, 1, GL_TRUE, vp)
                glUniformMatrix4fv(guModel, 1, GL_TRUE, model_car)
                for g in car_groups:
                    if not g['is_light']:
                        continue
                    if g['tex']:
                        glActiveTexture(GL_TEXTURE0)
                        glBindTexture(GL_TEXTURE_2D, g['tex'])
                    glBindVertexArray(g['vao'])
                    glDrawArrays(GL_TRIANGLES, 0, g['count'])
                glUniform1f(guIsEmissive, 0.0)
                glUniform1f(guIsCar, 0.0)        # ★ 恢复
                glEnable(GL_CULL_FACE)

            # ---- ★ 粒子尾气（GPU Compute Shader 顶点已在 p_vbo 中） ----
            if particle_count > 0:
                glBindBuffer(GL_ARRAY_BUFFER, p_vbo)
                glUniform1f(guMet, 0.0)
                glUniform1f(guRough, 0.9)
                glUniform1f(guHasTex, 0.0)
                glUniformMatrix4fv(guMVP, 1, GL_TRUE, vp)
                glUniformMatrix4fv(guModel, 1, GL_TRUE, identity_mat)
                glBindVertexArray(p_vao)
                glDrawArrays(GL_TRIANGLES, 0, particle_count * 6)

            # ---- 轮胎痕迹 ----
            if tire_marks:
                bt_m = 0
                for mx_m, mz_m, mang_m in tire_marks:
                    fx_m = -math.sin(mang_m); fz_m = -math.cos(mang_m)
                    rx_m = math.cos(mang_m); rz_m = -math.sin(mang_m)
                    hw_m = 0.15; hl_m = 0.25
                    p1x = mx_m - rx_m * hw_m - fx_m * hl_m; p1z = mz_m - rz_m * hw_m - fz_m * hl_m
                    p2x = mx_m - rx_m * hw_m + fx_m * hl_m; p2z = mz_m - rz_m * hw_m + fz_m * hl_m
                    p3x = mx_m + rx_m * hw_m - fx_m * hl_m; p3z = mz_m + rz_m * hw_m - fz_m * hl_m
                    p4x = mx_m + rx_m * hw_m + fx_m * hl_m; p4z = mz_m + rz_m * hw_m + fz_m * hl_m
                    nu_m = (0.0, 1.0, 0.0); tc_m = (0.03, 0.03, 0.03)
                    for vx_m, vz_m in [(p1x,p1z),(p2x,p2z),(p4x,p4z),(p1x,p1z),(p4x,p4z),(p3x,p3z)]:
                        tire_buf[bt_m:bt_m+3] = [vx_m, 0.01, vz_m]; bt_m += 3
                        tire_buf[bt_m:bt_m+3] = nu_m; bt_m += 3
                        tire_buf[bt_m:bt_m+2] = [0.0, 0.0]; bt_m += 2
                        tire_buf[bt_m:bt_m+3] = tc_m; bt_m += 3
                tc_m2 = bt_m // 11
                glBindBuffer(GL_ARRAY_BUFFER, tire_vbo)
                glBufferSubData(GL_ARRAY_BUFFER, 0, tire_buf[:bt_m].nbytes, tire_buf[:bt_m])
                glUniform1f(guHasTex, 0.0)
                glUniformMatrix4fv(guMVP, 1, GL_TRUE, vp)
                glUniformMatrix4fv(guModel, 1, GL_TRUE, identity_mat)
                glBindVertexArray(tire_vao)
                glDrawArrays(GL_TRIANGLES, 0, tc_m2)

            if _GB_DEBUG:
                _gb_debug_dump(g_buffer_fbo, ww, wh)

        # ★★★ 阶段B：光照 Pass → HDR FBO ★★★
        glBindFramebuffer(GL_FRAMEBUFFER, hdr_fbo)
        glViewport(0, 0, ww, wh)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # ★ CSM 阴影Pass：渲染2层级联深度图（优化：3→2级）
        cam_fwd = np.array([look_x - cam_pos[0], 0.0, look_z - cam_pos[2]], dtype=f32)
        cam_fwd_len = np.linalg.norm(cam_fwd)
        if cam_fwd_len > 0.001:
            cam_fwd = cam_fwd / cam_fwd_len
        else:
            cam_fwd = np.array([0.0, 0.0, -1.0], dtype=f32)
        # 级联分割距离（2级联：近0~200m，远200~1000m）
        csm_splits = np.array([30.0, 200.0, 1000.0], dtype=f32)
        light_view_projs = []
        # ★ 深度 Pass 启用正面剔除，大幅减少自阴影粉刺
        glEnable(GL_CULL_FACE)
        for ci in range(2):
            glBindFramebuffer(GL_FRAMEBUFFER, csm_fbos[ci])
            glViewport(0, 0, CSM_SIZE, CSM_SIZE)
            glClear(GL_DEPTH_BUFFER_BIT)
            glUseProgram(csm_prog)
            glCullFace(GL_FRONT)
            n = csm_splits[ci]
            f = csm_splits[ci + 1]
            # 计算级联包围盒中心
            cam_right = np.cross(cam_fwd, np.array([0.0, 1.0, 0.0]))
            cam_right = cam_right / (np.linalg.norm(cam_right) + 0.0001)
            # 视锥体8个角
            tan_hfov = math.tan(np.radians(65.0) * 0.5)
            aspect = ww / wh
            n_h = n * tan_hfov; n_w = n_h * aspect
            f_h = f * tan_hfov; f_w = f_h * aspect
            fc = cam_pos + cam_fwd * f; nc = cam_pos + cam_fwd * n
            cam_up = np.cross(cam_right, cam_fwd)
            frustum_corners = np.array([
                nc - cam_right * n_w - cam_up * n_h,
                nc + cam_right * n_w - cam_up * n_h,
                nc + cam_right * n_w + cam_up * n_h,
                nc - cam_right * n_w + cam_up * n_h,
                fc - cam_right * f_w - cam_up * f_h,
                fc + cam_right * f_w - cam_up * f_h,
                fc + cam_right * f_w + cam_up * f_h,
                fc - cam_right * f_w + cam_up * f_h,
            ], dtype=f32)
            center = np.mean(frustum_corners, axis=0)
            # 光源正交投影
            light_up = np.array([0.0, 1.0, 0.0], dtype=f32)
            light_view_mat = mat4_look_at(center - light_dir * 200.0, center, light_up)
            # 将视锥体角变换到光源空间求AABB
            min_lsp = np.array([1e9, 1e9, 1e9], dtype=f32)
            max_lsp = np.array([-1e9, -1e9, -1e9], dtype=f32)
            for fc in frustum_corners:
                p = mat4_multiply_vec(light_view_mat, np.array([fc[0], fc[1], fc[2], 1.0]))
                min_lsp = np.minimum(min_lsp, p[:3])
                max_lsp = np.maximum(max_lsp, p[:3])
            # 扩展Z范围
            min_lsp[2] -= 200.0; max_lsp[2] += 200.0
            light_proj_mat = mat4_ortho(min_lsp[0], max_lsp[0], min_lsp[1], max_lsp[1], -max_lsp[2], -min_lsp[2])
            light_vp = mat4_multiply(light_proj_mat, light_view_mat)
            light_view_projs.append(light_vp)
            # 设置光源VP（CSM_DEPTH_VERTEX_SRC中用 uLightVP）
            clvp = glGetUniformLocation(csm_prog, "uLightVP")
            if clvp >= 0:
                glUniformMatrix4fv(clvp, 1, GL_TRUE, light_vp)
            cmodel = glGetUniformLocation(csm_prog, "uModel")
            cinst = glGetUniformLocation(csm_prog, "uIsInstanced")
            if cinst >= 0: glUniform1i(cinst, 0)
            # 渲染地形
            if cmodel >= 0:
                glUniformMatrix4fv(cmodel, 1, GL_TRUE, identity_mat)
            if tvao is not None:
                glBindVertexArray(tvao)
                glDrawArrays(GL_TRIANGLES, 0, tcnt)
            # 渲染树木（实例化）—— ★ 优化：只渲染到级联0（30m内才有意义的树影）
            if ci == 0 and tree_units:
                if cinst >= 0: glUniform1i(cinst, 1)
                # ★ 叶片剪影裁剪：否则交叉面片会投出矩形黑影
                _cf = glGetUniformLocation(csm_prog, "uUseFoliage")
                _cft = glGetUniformLocation(csm_prog, "uFoliageTex")
                if _cf >= 0: glUniform1f(_cf, 1.0)
                if _cft >= 0:
                    glUniform1i(_cft, 0)
                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, foliage_tex)
                cx_c, cy_c, cz_c = cam_eye
                pos_c = tree_instances[:, :3]
                dist2_c = (pos_c[:, 0] - cx_c)**2 + (pos_c[:, 1] - cy_c)**2 + (pos_c[:, 2] - cz_c)**2
                dist_c = np.sqrt(dist2_c)
                _draw_tree_batches(_tree_band_masks(dist_c), np.ones(n_tree_instances, dtype=bool))
                if cinst >= 0: glUniform1i(cinst, 0)
                _cf = glGetUniformLocation(csm_prog, "uUseFoliage")
                if _cf >= 0: glUniform1f(_cf, 0.0)
            # 渲染建筑
            if bvao is not None and bcnt > 0:
                glUniformMatrix4fv(cmodel, 1, GL_TRUE, identity_mat)
                glBindVertexArray(bvao)
                glDrawArrays(GL_TRIANGLES, 0, bcnt)
            # 渲染赛车（按材质组）
            glUniformMatrix4fv(cmodel, 1, GL_TRUE, model_car)
            for g in car_groups:
                glBindVertexArray(g['vao'])
                glDrawArrays(GL_TRIANGLES, 0, g['count'])
            # 渲染路障
            if cbvao is not None and cbcnt > 0:
                glUniformMatrix4fv(cmodel, 1, GL_TRUE, identity_mat)
                glBindVertexArray(cbvao)
                glDrawArrays(GL_TRIANGLES, 0, cbcnt)

        glDisable(GL_CULL_FACE)

        # ★ SSAO Pass：计算环境光遮蔽
        glBindFramebuffer(GL_FRAMEBUFFER, ssao_fbo)
        glViewport(0, 0, SSAO_W, SSAO_H)
        glClear(GL_COLOR_BUFFER_BIT)
        if ssao_prog:
            glUseProgram(ssao_prog)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, g_pos_tex)
            if suPos >= 0: glUniform1i(suPos, 0)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, g_nrm_tex)
            if suNrm >= 0: glUniform1i(suNrm, 1)
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, ssao_noise_tex)
            if suNoise >= 0: glUniform1i(suNoise, 2)
            if suSamples >= 0:
                for si, s in enumerate(ssao_samples):
                    glUniform3f(suSamples + si, s[0], s[1], s[2])
            if suProj >= 0:
                glUniformMatrix4fv(suProj, 1, GL_TRUE, projection)
            if suViewPos >= 0:
                glUniform3fv(suViewPos, 1, cam_eye)
            glBindVertexArray(quad_vao)
            glDrawArrays(GL_TRIANGLES, 0, quad_cnt)

        # SSAO 模糊Pass
        glBindFramebuffer(GL_FRAMEBUFFER, ssao_blur_fbo)
        glClear(GL_COLOR_BUFFER_BIT)
        if ssao_blur_prog:
            glUseProgram(ssao_blur_prog)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, ssao_tex)
            bInput = glGetUniformLocation(ssao_blur_prog, "uSSAOTex")
            if bInput >= 0: glUniform1i(bInput, 0)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, g_nrm_tex)
            bNrm = glGetUniformLocation(ssao_blur_prog, "gNormal")
            if bNrm >= 0: glUniform1i(bNrm, 1)
            bSize = glGetUniformLocation(ssao_blur_prog, "uScreenSize")
            if bSize >= 0: glUniform2f(bSize, float(SSAO_W), float(SSAO_H))
            glBindVertexArray(quad_vao)
            glDrawArrays(GL_TRIANGLES, 0, quad_cnt)

        # ★ SSR Pass：屏幕空间反射
        glBindFramebuffer(GL_FRAMEBUFFER, ssr_fbo)
        glViewport(0, 0, ww, wh)
        glClear(GL_COLOR_BUFFER_BIT)
        if ssr_prog:
            inv_proj = np.linalg.inv(projection)
            glUseProgram(ssr_prog)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, g_pos_tex)
            if srPos >= 0: glUniform1i(srPos, 0)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, g_nrm_tex)
            if srNrm >= 0: glUniform1i(srNrm, 1)
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, g_alb_tex)
            if srAlb >= 0: glUniform1i(srAlb, 2)
            glActiveTexture(GL_TEXTURE3)
            glBindTexture(GL_TEXTURE_2D, g_mat_tex)
            if srMat >= 0: glUniform1i(srMat, 3)
            glActiveTexture(GL_TEXTURE4)
            glBindTexture(GL_TEXTURE_2D, hdr_tex)
            if srSceneTex >= 0: glUniform1i(srSceneTex, 4)
            glActiveTexture(GL_TEXTURE5)
            glBindTexture(GL_TEXTURE_2D, g_depth_tex)
            if srDepthTex >= 0: glUniform1i(srDepthTex, 5)
            if srProj >= 0: glUniformMatrix4fv(srProj, 1, GL_TRUE, projection)
            if srInvProj >= 0: glUniformMatrix4fv(srInvProj, 1, GL_TRUE, inv_proj)
            if srView >= 0: glUniformMatrix4fv(srView, 1, GL_TRUE, view)
            if srViewPos >= 0: glUniform3fv(srViewPos, 1, cam_pos)
            if srScreenSize >= 0: glUniform2f(srScreenSize, float(ww), float(wh))
            if srMaxDist >= 0: glUniform1f(srMaxDist, 30.0)
            if srStepSize >= 0: glUniform1f(srStepSize, 0.5)
            if srMaxSteps >= 0: glUniform1i(srMaxSteps, 32)
            if srThickness >= 0: glUniform1f(srThickness, 0.1)
            glBindVertexArray(quad_vao)
            glDrawArrays(GL_TRIANGLES, 0, quad_cnt)

        # ★ 体积光Pass：丁达尔效应 + 雾气
        glBindFramebuffer(GL_FRAMEBUFFER, vol_fbo)
        glViewport(0, 0, VOL_W, VOL_H)
        glClear(GL_COLOR_BUFFER_BIT)
        if vol_prog:
            glUseProgram(vol_prog)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, g_depth_tex)
            if vuDepthTex >= 0: glUniform1i(vuDepthTex, 0)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, csm_texs[0])
            if vuShadowMap >= 0: glUniform1i(vuShadowMap, 1)
            if vuInvProj >= 0:
                glUniformMatrix4fv(vuInvProj, 1, GL_TRUE, inv_proj)
            if vuCamPos >= 0: glUniform3fv(vuCamPos, 1, cam_pos)
            if vuLightDir >= 0: glUniform3fv(vuLightDir, 1, light_dir)
            if vuLightColor >= 0: glUniform3fv(vuLightColor, 1, light_color)
            if vuLightVP >= 0 and len(light_view_projs) > 0:
                glUniformMatrix4fv(vuLightVP, 1, GL_TRUE, light_view_projs[0])
            if vuScattering >= 0: glUniform1f(vuScattering, 0.0005)  # 晴天散射极弱
            if vuTime >= 0: glUniform1f(vuTime, now)
            if vuResolution >= 0: glUniform2f(vuResolution, float(VOL_W), float(VOL_H))
            # uNumSteps, uMaxDist, uScreenSize
            vSteps = glGetUniformLocation(vol_prog, "uNumSteps")
            if vSteps >= 0: glUniform1i(vSteps, 24)
            vMaxDist = glGetUniformLocation(vol_prog, "uMaxDist")
            if vMaxDist >= 0: glUniform1f(vMaxDist, 200.0)
            vScreenSize = glGetUniformLocation(vol_prog, "uScreenSize")
            if vScreenSize >= 0: glUniform2f(vScreenSize, float(VOL_W), float(VOL_H))
            glBindVertexArray(quad_vao)
            glDrawArrays(GL_TRIANGLES, 0, quad_cnt)

        if USE_DEFERRED:
            # ★ 光照 Pass 必须重新绑定 hdr_fbo，因为中间 Pass 切走了 FBO
            glBindFramebuffer(GL_FRAMEBUFFER, hdr_fbo)
            glViewport(0, 0, ww, wh)
            glUseProgram(light_prog)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, g_pos_tex)
            glUniform1i(luPos, 0)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, g_nrm_tex)
            glUniform1i(luNrm, 1)
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, g_alb_tex)
            glUniform1i(luAlb, 2)
            glActiveTexture(GL_TEXTURE3)
            glBindTexture(GL_TEXTURE_2D, g_mat_tex)
            glUniform1i(luMat, 3)
            # ★ CSM 阴影图绑定（2级联）
            glActiveTexture(GL_TEXTURE4)
            glBindTexture(GL_TEXTURE_2D, csm_texs[0])
            if luShadow0 >= 0: glUniform1i(luShadow0, 4)
            glActiveTexture(GL_TEXTURE5)
            glBindTexture(GL_TEXTURE_2D, csm_texs[1])
            if luShadow1 >= 0: glUniform1i(luShadow1, 5)
            # ★ SSAO 纹理绑定
            glActiveTexture(GL_TEXTURE6)
            glBindTexture(GL_TEXTURE_2D, ssao_blur_tex)
            if luSSAOTex >= 0: glUniform1i(luSSAOTex, 6)
            if luSSAOEnabled >= 0: glUniform1f(luSSAOEnabled, 1.0)
            # ★ CSM 投影矩阵和级联分割（2级联）
            if luLightVP >= 0:
                for i in range(2):
                    if i < len(light_view_projs):
                        glUniformMatrix4fv(luLightVP + i, 1, GL_TRUE, light_view_projs[i])
            if luCascadeSplits >= 0:
                glUniform4f(luCascadeSplits, csm_splits[1], csm_splits[2], float(CSM_SIZE), 0.0)
            if luShadowStrength >= 0: glUniform1f(luShadowStrength, 0.85)
            glUniform3fv(luLD, 1, light_dir)
            glUniform3fv(luLC, 1, light_color)
            # ★ 环境天空辐亮度：必须与延迟模式下"天空"的着色一致
            #   （light pass 里 emissive 分支的 uv.y 渐变），否则车漆反射会跟
            #   背景天空对不上。均匀环境会让黑车变成一坨灰，渐变才有"形体"。
            for _sn, _sv in (('uSkyZenith', (0.35, 0.62, 1.05)),
                             ('uSkyHorizon', (0.75, 0.88, 1.05)),
                             ('uSkyGround', (0.30, 0.28, 0.26)),
                             ('uSkySun', (1.60, 1.55, 1.40))):
                _sl = glGetUniformLocation(light_prog, _sn)
                if _sl >= 0:
                    glUniform3f(_sl, *_sv)
            glUniform3fv(luVP, 1, cam_eye)
            glUniform3fv(luAmb, 1, ambient_col)
            # ★ IBL：上传 SH 系数
            if luSHCoeffs >= 0:
                sh_coeffs = compute_sh_coeffs(light_dir, light_color, ambient_col)
                for _i in range(9):
                    glUniform3fv(luSHCoeffs + _i, 1, sh_coeffs[_i])
            glUniform1f(luExposure, 1.8)
            glBindVertexArray(quad_vao)
            glDrawArrays(GL_TRIANGLES, 0, quad_cnt)

            # 复制 G-Buffer 深度到 HDR FBO
            glBindFramebuffer(GL_READ_FRAMEBUFFER, g_buffer_fbo)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, hdr_fbo)
            glBlitFramebuffer(0, 0, ww, wh, 0, 0, ww, wh, GL_DEPTH_BUFFER_BIT, GL_NEAREST)
            glBindFramebuffer(GL_FRAMEBUFFER, hdr_fbo)
            glDepthMask(GL_FALSE)

            # ---- 前向 Pass：灯罩自发光 ----
            if lemcnt > 0:
                glUseProgram(prog)
                glUniformMatrix4fv(uMVP, 1, GL_TRUE, vp)
                glUniformMatrix4fv(uModel, 1, GL_TRUE, identity_mat)
                glUniform1f(uIsEmissive, 1.0)
                glUniform1f(uHasTex, 0.0)
                glBindVertexArray(lemvao)
                glDrawArrays(GL_TRIANGLES, 0, lemcnt)
                glUniform1f(uIsEmissive, 0.0)

            # ---- 前向 Pass：地面光池 ----
            if lpcnt > 0:
                glUseProgram(prog)
                glUniformMatrix4fv(uMVP, 1, GL_TRUE, vp)
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE)
                glUniform1f(uIsEmissive, 1.0)
                glUniform1f(uHasTex, 0.0)
                glBindVertexArray(lpvao)
                glDrawArrays(GL_TRIANGLES, 0, lpcnt)
                glUniform1f(uIsEmissive, 0.0)
                glDisable(GL_BLEND)

            # ---- 前向 Pass：光锥体积光 ----
            if conecnt > 0:
                glUseProgram(cone_prog)
                glUniformMatrix4fv(cone_uMVP, 1, GL_TRUE, vp)
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE)
                glBindVertexArray(conevao)
                glDrawArrays(GL_TRIANGLES, 0, conecnt)
                glDisable(GL_BLEND)

            # ---- 前向 Pass：凸面镜镜面自发光 ----
            if sgnecnt > 0:
                glUseProgram(prog)
                glUniformMatrix4fv(uMVP, 1, GL_TRUE, vp)
                glUniformMatrix4fv(uModel, 1, GL_TRUE, identity_mat)
                glUniform1f(uIsEmissive, 1.0)
                glUniform1f(uHasTex, 0.0)
                glBindVertexArray(sgnevao)
                glDrawArrays(GL_TRIANGLES, 0, sgnecnt)
                glUniform1f(uIsEmissive, 0.0)

            # ---- 前向 Pass：反光道钉（自发光，触发 Bloom）----
            if stucnt > 0:
                glUseProgram(prog)
                glUniformMatrix4fv(uMVP, 1, GL_TRUE, vp)
                glUniformMatrix4fv(uModel, 1, GL_TRUE, identity_mat)
                glUniform1f(uIsEmissive, 1.0)
                glUniform1f(uHasTex, 0.0)
                glBindVertexArray(stuvao)
                glDrawArrays(GL_TRIANGLES, 0, stucnt)
                glUniform1f(uIsEmissive, 0.0)

            glDepthMask(GL_TRUE)
            glUseProgram(prog)

        if not USE_DEFERRED:
            # ---- 绘制天空盒 ----
            # 用 view 的旋转部分 + 单位平移，让天空球"永远跟着摄像机"
            sky_view = view.copy()
            sky_view[0, 3] = 0.0
            sky_view[1, 3] = 0.0
            sky_view[2, 3] = 0.0
            sky_vp = mat4_multiply(projection, sky_view)

            glDepthMask(GL_FALSE)
            glDisable(GL_DEPTH_TEST)
            glUseProgram(sky_prog)
            glUniformMatrix4fv(sky_uMVP, 1, GL_TRUE, sky_vp)
            glUniform3fv(sky_uSunDir, 1, light_dir)   # ★ 原来传了 -light_dir，太阳盘永远在地平线下
            glUniform3fv(sky_uZenithColor,  1, np.array([0.02, 0.10, 0.45], dtype=f32))  # 深邃蓝天顶
            glUniform3fv(sky_uHorizonColor, 1, np.array([0.55, 0.80, 0.95], dtype=f32))  # 淡蓝地平线
            glUniform3fv(sky_uGroundColor,  1, np.array([0.35, 0.55, 0.65], dtype=f32))  # 浅蓝雾海
            glUniform3fv(sky_uSunColor,     1, np.array([1.6, 1.55, 1.4], dtype=f32))    # 白色太阳，HDR 强
            glUniform1f(sky_uTime, sky_time)
            glBindVertexArray(skvao)
            glDrawArrays(GL_TRIANGLES, 0, sky_cnt)
            glEnable(GL_DEPTH_TEST)
            glDepthMask(GL_TRUE)
            glUseProgram(prog)

            # 光源和摄像机位置
            glUniform3fv(uLD, 1, light_dir)
            glUniform3fv(uVP, 1, cam_eye)
            glUniform3fv(uAmb, 1, ambient_col)
            glUniform3fv(uLC, 1, light_color)
            glUniform1i(uTex, 0)
            glUniform1f(uAmbBoost, 1.0)
            glUniform1f(uExposure, 1.8)
            glUniform1i(uDebugMode, debug_mode)
            glUniform1f(uIsEmissive, 0.0)
            glUniform1f(uApplyTonemap, 0.0)
            glActiveTexture(GL_TEXTURE0)

            # ---- 绘制山区地形 ----
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(3.0, 3.0)
            glUniform1f(uAmbBoost, 1.0)
            glUniform1f(uMet, 0.0)
            glUniform1f(uRough, 0.92)
            glUniform1f(uUseTerrainBlend, 1.0)

            glUniform1i(uGrassTex, 0)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, grass_tex)
            glUniform1i(uRockTex, 1)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, rock_tex)
            glUniform1i(uDirtTex, 2)
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, dirt_tex)
            glUniform1i(uGravelTex, 3)
            glActiveTexture(GL_TEXTURE3)
            glBindTexture(GL_TEXTURE_2D, gravel_tex)

            glUniform1i(uGrassNrm, 4)
            glActiveTexture(GL_TEXTURE4)
            glBindTexture(GL_TEXTURE_2D, grass_nrm)
            glUniform1i(uRockNrm, 5)
            glActiveTexture(GL_TEXTURE5)
            glBindTexture(GL_TEXTURE_2D, rock_nrm)
            glUniform1i(uDirtNrm, 6)
            glActiveTexture(GL_TEXTURE6)
            glBindTexture(GL_TEXTURE_2D, dirt_nrm)
            glUniform1i(uGravelNrm, 7)
            glActiveTexture(GL_TEXTURE7)
            glBindTexture(GL_TEXTURE_2D, gravel_nrm)

            glUniformMatrix4fv(uMVP, 1, GL_TRUE, vp)
            glUniformMatrix4fv(uModel, 1, GL_TRUE, identity_mat)
            glUniform1f(uHasTex, 0.0)
            glBindVertexArray(akvao)
            glDrawArrays(GL_TRIANGLES, 0, akcnt)

            glUniform1f(uUseTerrainBlend, 0.0)
            glActiveTexture(GL_TEXTURE0)
            glDisable(GL_POLYGON_OFFSET_FILL)

            # ---- 绘制赛道 ----
            glUniform1f(uMet, 0.0)
            glUniform1f(uRough, 0.98)
            glUniform1f(uHasTex, 1.0)
            # ★ 路面细节层：打破整片低频脏斑
            glUniform1f(uDetail, 1.0)
            glUniform1f(uDetailScale, 0.30)
            glUniform1i(uDetailTex, 12)
            glActiveTexture(GL_TEXTURE0 + 12)
            glBindTexture(GL_TEXTURE_2D, detail_tex)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, track_tex)
            glBindVertexArray(tvao)
            glDrawArrays(GL_TRIANGLES, 0, tcnt)
            glUniform1f(uDetail, 0.0)

            # ---- 绘制护栏 ----
            glUniform1f(uMet, 0.90)
            glUniform1f(uRough, 0.30)
            glUniform1f(uHasTex, 0.0)
            glBindVertexArray(grvao)
            glDrawArrays(GL_TRIANGLES, 0, grcnt)

            # ---- 路灯灯杆 ----
            if lcnt > 0:
                glUniform1f(uMet, 0.85)
                glUniform1f(uRough, 0.35)
                glUniform1f(uHasTex, 0.0)
                glBindVertexArray(lvao)
                glDrawArrays(GL_TRIANGLES, 0, lcnt)

            # ---- 沿路建筑 ----
            if bcnt > 0:
                b_center = building_center
                b_radius = building_bound_radius
                planes = extract_frustum_planes(vp)
                b_visible = frustum_test_sphere(planes, b_center, b_radius)
                if b_visible:
                    glUniform1f(uMet, 0.0)
                    glUniform1f(uRough, 0.85)
                    glUniform1f(uHasTex, 0.0)
                    glUniform1f(uIsBuilding, 1.0)
                    glActiveTexture(GL_TEXTURE0 + 8)
                    glBindTexture(GL_TEXTURE_2D, building_wall_tex)
                    glUniform1i(uBuildingAlbedo, 8)
                    glActiveTexture(GL_TEXTURE0 + 11)
                    glBindTexture(GL_TEXTURE_2D, building_ao_tex)
                    glUniform1i(uBuildingAO, 11)
                    glBindVertexArray(bvao)
                    glDrawArrays(GL_TRIANGLES, 0, bcnt)
                    glUniform1f(uIsBuilding, 0.0)
                    glActiveTexture(GL_TEXTURE0)

        # ---- 灯罩自发光 ----
            if lemcnt > 0:
                glUniform1f(uIsEmissive, 1.0)
                glUniform1f(uHasTex, 0.0)
                glBindVertexArray(lemvao)
                glDrawArrays(GL_TRIANGLES, 0, lemcnt)
                glUniform1f(uIsEmissive, 0.0)

            # ---- 凸面镜镜面自发光 ----
            if sgnecnt > 0:
                glUniform1f(uIsEmissive, 1.0)
                glUniform1f(uHasTex, 0.0)
                glBindVertexArray(sgnevao)
                glDrawArrays(GL_TRIANGLES, 0, sgnecnt)
                glUniform1f(uIsEmissive, 0.0)

            # ---- 反光道钉 ----
            if stucnt > 0:
                glUniform1f(uIsEmissive, 1.0)
                glUniform1f(uHasTex, 0.0)
                glBindVertexArray(stuvao)
                glDrawArrays(GL_TRIANGLES, 0, stucnt)
                glUniform1f(uIsEmissive, 0.0)

            # ---- 地面光池 ----
            if lpcnt > 0:
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE)
                glDepthMask(GL_FALSE)
                glUniform1f(uIsEmissive, 1.0)
                glUniform1f(uHasTex, 0.0)
                glBindVertexArray(lpvao)
                glDrawArrays(GL_TRIANGLES, 0, lpcnt)
                glUniform1f(uIsEmissive, 0.0)
                glDepthMask(GL_TRUE)
                glDisable(GL_BLEND)

            # ---- 光锥体积光 ----
            if conecnt > 0:
                glUseProgram(cone_prog)
                glUniformMatrix4fv(cone_uMVP, 1, GL_TRUE, vp)
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE)
                glDepthMask(GL_FALSE)
                glBindVertexArray(conevao)
                glDrawArrays(GL_TRIANGLES, 0, conecnt)
                glDepthMask(GL_TRUE)
                glDisable(GL_BLEND)
                glUseProgram(prog)

            # ---- 绘制杉树（实例化 + LOD、GPU 视锥剔除） ----
            glUseProgram(tree_prog)
            glUniform3fv(tuLD, 1, light_dir)
            glUniform3fv(tuVP, 1, cam_eye)
            glUniform3fv(tuAmb, 1, ambient_col)
            glUniform3fv(tuLC, 1, light_color)
            glUniform1f(tuAmbBoost, 1.0)
            glUniform1f(tuExposure, 1.8)
            glUniform1f(tuMet, 0.0)
            glUniform1f(tuRough, 0.82)
            glUniform1f(tuHasTex, 0.0)
            glUniform1f(tuUseTerrainBlend, 0.0)
            glUniformMatrix4fv(tuMVP, 1, GL_TRUE, vp)
            glUniformMatrix4fv(tuModel, 1, GL_TRUE, identity_mat)
            # ★ 叶片面片：启用 alpha 测试，并关闭背面剔除（双面可见）
            glUniform1f(tuUseFoliage, 1.0)
            glUniform1i(tuFoliage, 0)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, foliage_tex)
            glDisable(GL_CULL_FACE)

            # ★ CPU 视锥剔除（numpy 向量化，比 GPU 剔除 + fence 同步更快）
            planes_c = extract_frustum_planes(vp)
            pos_c = tree_instances[:, :3]
            d_c = pos_c @ planes_c[:, :3].T + planes_c[:, 3]
            vis_mask_c = np.all(d_c >= -12.0, axis=1)

            cx_c, cy_c, cz_c = cam_eye
            pos_c = tree_instances[:, :3]
            dist2_c = (pos_c[:, 0] - cx_c)**2 + (pos_c[:, 1] - cy_c)**2 + (pos_c[:, 2] - cz_c)**2
            dist_c = np.sqrt(dist2_c)
            _draw_tree_batches(_tree_band_masks(dist_c), vis_mask_c)
            glEnable(GL_CULL_FACE)
            glUniform1f(tuUseFoliage, 0.0)
            glUseProgram(prog)

            # ---- 混凝土路障 ----
            if cbvao is not None and cbcnt > 0:
                glUniform1f(uMet, 0.05)
                glUniform1f(uRough, 0.75)
                glUniform1f(uHasTex, 0.0)
                glBindVertexArray(cbvao)
                glDrawArrays(GL_TRIANGLES, 0, cbcnt)

            # ---- 尾灯发光 ----
            if keys[sdl2.SDL_SCANCODE_S]:
                glUniform1f(uMet, 0.1)
                glUniform1f(uRough, 0.25)
                glUniformMatrix4fv(uMVP, 1, GL_TRUE, vp)
                glUniformMatrix4fv(uModel, 1, GL_TRUE, model_car)
                for g in car_groups:
                    if not g['is_light']:
                        continue
                    glUniform1f(uHasTex, g['has_tex'])
                    if g['tex']:
                        glActiveTexture(GL_TEXTURE0)
                        glBindTexture(GL_TEXTURE_2D, g['tex'])
                    glBindVertexArray(g['vao'])
                    glDrawArrays(GL_TRIANGLES, 0, g['count'])

            # ---- 赛车（★ RX-7 GLB：按材质组绘制） ----
            glUniform1f(uMet, 0.20)      # ★ 金属车漆（清漆层观感）
            glUniform1f(uRough, 0.30)    # ★ 更锐的高光反射
            glUniformMatrix4fv(uMVP, 1, GL_TRUE, vp)
            glUniformMatrix4fv(uModel, 1, GL_TRUE, model_car)
            for g in car_groups:
                if g['is_light']:
                    continue
                glUniform1f(uHasTex, g['has_tex'])
                if g['tex']:
                    glActiveTexture(GL_TEXTURE0)
                    glBindTexture(GL_TEXTURE_2D, g['tex'])
                glBindVertexArray(g['vao'])
                glDrawArrays(GL_TRIANGLES, 0, g['count'])

            # ---- ★ 粒子尾气（GPU Compute Shader 顶点已在 p_vbo 中） ----
            if particle_count > 0:
                glBindBuffer(GL_ARRAY_BUFFER, p_vbo)
                glUniform1f(uMet, 0.0)
                glUniform1f(uRough, 0.9)
                glUniform1f(uHasTex, 0.0)
                glUniformMatrix4fv(uMVP, 1, GL_TRUE, vp)
                glUniformMatrix4fv(uModel, 1, GL_TRUE, identity_mat)
                glBindVertexArray(p_vao)
                glDrawArrays(GL_TRIANGLES, 0, particle_count * 6)

            # ---- 轮胎痕迹 ----
            if tire_marks:
                buf_off_t = 0
                glUniform1f(uMet, 0.0)
                glUniform1f(uRough, 0.98)
                tire_col = (0.03, 0.03, 0.03)
                n_up = (0.0, 1.0, 0.0)
                tire_w = 0.30
                tire_l = 0.50
                for mx, mz, mang in tire_marks:
                    fwd_x = -math.sin(mang)
                    fwd_z = -math.cos(mang)
                    rgt_x = math.cos(mang)
                    rgt_z = -math.sin(mang)
                    hw = tire_w * 0.5
                    hl = tire_l * 0.5
                    p1x = mx - rgt_x * hw - fwd_x * hl
                    p1z = mz - rgt_z * hw - fwd_z * hl
                    p2x = mx - rgt_x * hw + fwd_x * hl
                    p2z = mz - rgt_z * hw + fwd_z * hl
                    p3x = mx + rgt_x * hw - fwd_x * hl
                    p3z = mz + rgt_z * hw - fwd_z * hl
                    p4x = mx + rgt_x * hw + fwd_x * hl
                    p4z = mz + rgt_z * hw + fwd_z * hl
                    for vx, vz in [(p1x,p1z),(p2x,p2z),(p4x,p4z),(p1x,p1z),(p4x,p4z),(p3x,p3z)]:
                        tire_buf[buf_off_t:buf_off_t+3] = [vx, 0.01, vz]
                        buf_off_t += 3
                        tire_buf[buf_off_t:buf_off_t+3] = n_up
                        buf_off_t += 3
                        tire_buf[buf_off_t:buf_off_t+2] = [0.0, 0.0]
                        buf_off_t += 2
                        tire_buf[buf_off_t:buf_off_t+3] = tire_col
                        buf_off_t += 3
                t_count = buf_off_t // 11
                glBindBuffer(GL_ARRAY_BUFFER, tire_vbo)
                glBufferSubData(GL_ARRAY_BUFFER, 0, tire_buf[:buf_off_t].nbytes, tire_buf[:buf_off_t])
                glUniform1f(uHasTex, 0.0)
                glUniformMatrix4fv(uMVP, 1, GL_TRUE, vp)
                glUniformMatrix4fv(uModel, 1, GL_TRUE, identity_mat)
                glBindVertexArray(tire_vao)
                glDrawArrays(GL_TRIANGLES, 0, t_count)

        # ============================================================
        # Bloom 后处理流水线
        # ============================================================
        # 1) 亮部提取 → 半分辨率 ping FBO
        glBindFramebuffer(GL_FRAMEBUFFER, ping_fbo)
        glViewport(0, 0, blur_w, blur_h)
        glUseProgram(bloom_prog_bright)
        glUniform1i(bm_uScene, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, hdr_tex)
        glUniform1f(bm_uThreshold, 1.5)
        glBindVertexArray(quad_vao)
        glDrawArrays(GL_TRIANGLES, 0, quad_cnt)

        # 2) 水平模糊 → 半分辨率 pong FBO
        glBindFramebuffer(GL_FRAMEBUFFER, pong_fbo)
        glViewport(0, 0, blur_w, blur_h)
        glUseProgram(bloom_prog_blur)
        glUniform1i(bl_uImage, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, ping_tex)
        glUniform1i(bl_uHorizontal, 1)
        glBindVertexArray(quad_vao)
        glDrawArrays(GL_TRIANGLES, 0, quad_cnt)

        # 3) 垂直模糊 → 半分辨率 ping FBO
        glBindFramebuffer(GL_FRAMEBUFFER, ping_fbo)
        glViewport(0, 0, blur_w, blur_h)
        glUniform1i(bl_uImage, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, pong_tex)
        glUniform1i(bl_uHorizontal, 0)
        glBindVertexArray(quad_vao)
        glDrawArrays(GL_TRIANGLES, 0, quad_cnt)

        # 4) 合成 → LDR FBO（色调映射 + Gamma + Bloom + 色彩分级）
        glBindFramebuffer(GL_FRAMEBUFFER, ldr_fbo)
        glViewport(0, 0, ww, wh)
        glDisable(GL_DEPTH_TEST)
        glUseProgram(bloom_prog_composite)
        glUniform1i(bc_uScene, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, hdr_tex)
        glUniform1i(bc_uBloom, 1)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, ping_tex)
        glUniform1f(bc_uBloomStrength, 0.75)
        glUniform1f(bc_uExposure, 1.55)          # ★ 1.25 太暗（车黑成剪影），回调
        glUniform1f(bc_uContrast, 0.10)          # ★ 对比度（0.16 会压死暗部）
        glUniform1f(bc_uSaturation, 1.18)        # ★ 饱和度
        glUniform1f(bc_uVignette, 0.16)          # ★ 暗角（0.28 太重）
        glBindVertexArray(quad_vao)
        glDrawArrays(GL_TRIANGLES, 0, quad_cnt)

        # 5) FXAA 收边 → 默认帧缓冲（上屏）
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, ww, wh)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(fxaa_prog)
        glUniform1i(fx_uTex, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, ldr_tex)
        glUniform2f(fx_uTexel, 1.0 / ww, 1.0 / wh)
        glBindVertexArray(quad_vao)
        glDrawArrays(GL_TRIANGLES, 0, quad_cnt)
        glEnable(GL_DEPTH_TEST)

        # ============================================================
        # UI 层渲染（菜单 / HUD）
        # ============================================================
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        if main_menu.is_active:
            # 主菜单在最上层
            main_menu.render()
        elif settings_panel.is_active:
            # 设置面板覆盖在游戏画面上
            settings_panel.render()
        elif game_active:
            # 游戏中：按设置项渲染小地图 + 仪表盘
            # 从设置面板读取用户配置
            hud_cfg = {it["key"]: it["value"] for it in settings_panel._items}

            if hud_cfg.get("show_minimap", True):
                minimap.update_player(car_pos, car_angle)
                minimap.render()

            if hud_cfg.get("show_speedo", True):
                glViewport(0, 0, ww, wh)
                gear_display.update(phys.engine.gear, dt)
                speedometer.update(car_speed, phys.engine.gear, dt, phys.drift, engine_rpm=phys.engine.rpm)
                gear_display.render()
                speedometer.render()

            # ---- 弯道提示（独立于设置项，始终显示） ----
            glViewport(0, 0, ww, wh)
            upcoming = corner_analyzer.query(phys.x, phys.z, phys.yaw)
            corner_warning.update(upcoming, dt)
            corner_warning.render()

        # ★ 摄像机调节信息叠加（右下角，避开左下角仪表盘与左上角小地图）
        _cam_cur_dist = 3.63 + _cam_offset_dist
        _cam_cur_h = 1.73 + _cam_offset_height
        _cam_cur_look = 5.0 + _cam_offset_lookahead
        _info1 = f"Cam [{_cam_mode_names[_cam_adjust_mode]}]  dist={_cam_cur_dist:.2f}  h={_cam_cur_h:.2f}  look={_cam_cur_look:.1f}"
        _info2 = "[C]切模式  [滚轮]调值  [ESC]退出"
        _cam_txt_x = logical_ww - 14
        text_renderer.render(_info1, _cam_txt_x, 36, font_size=16, color=(0, 255, 255, 200), align="right")
        text_renderer.render(_info2, _cam_txt_x, 56, font_size=14, color=(180, 180, 180, 160), align="right")

# 恢复 OpenGL 状态
        glViewport(0, 0, ww, wh)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)

        sdl2.SDL_GL_SwapWindow(win)

        # 帧率统计（每200ms打印一次）
        fps_frame_count += 1
        fps_timer += dt
        if fps_timer >= fps_interval:
            fps = fps_frame_count / fps_timer
            print(f"[FPS] {fps:.0f} fps | 顶点: {total_verts:,} | 粒子: {particle_count}")
            fps_frame_count = 0
            fps_timer = 0.0

    # ---- 清理 ----
    all_car_vaos = [g['vao'] for g in car_groups]
    all_vaos = [tvao, grvao, akvao, *all_car_vaos, p_vao, tire_vao, lvao, lemvao, lpvao, sgnvao, sgnevao, stuvao, bvao, skvao, vgvao, vbvao]
    for vao in all_vaos:
        glDeleteVertexArrays(1, [vao])
    for vao in tree_lod_vaos:
        glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [p_vbo, tire_vbo, tree_inst_vbo, lvbo, lemvbo, lpvbo, sgnvbo, sgnevbo, stuvbo, bvbo, skvbo, vgvbo, vbvbo])
    for vbo in tree_lod_vbos:
        glDeleteBuffers(1, [vbo])
    if akvbos:
        glDeleteBuffers(len(akvbos), akvbos)
    glDeleteProgram(tree_prog)
    glDeleteProgram(sky_prog)
    glDeleteProgram(fxaa_prog)
    glDeleteFramebuffers(1, [ldr_fbo])
    glDeleteTextures(1, [ldr_tex, foliage_tex])
    for tex in [track_tex, track_normal, building_tex, car_tex, white_tex,
                track_roughness, track_metallic, track_ao, track_height,
                grass_roughness, rock_roughness, dirt_roughness, gravel_roughness,
                grass_ao, rock_ao, dirt_ao, gravel_ao]:
        if tex is not None:
            glDeleteTextures(1, [tex])
    minimap.destroy()
    gear_display.destroy()
    speedometer.destroy()
    corner_warning.destroy()
    text_renderer.destroy()
    main_menu.destroy()
    settings_panel.destroy()
    destroy_shared_hud()
    glDeleteProgram(prog)
    gc.enable()  # 恢复 GC
    sdl2.SDL_GL_DeleteContext(ctx)
    sdl2.SDL_DestroyWindow(win)
    sdl2.SDL_Quit()
    return 0

if __name__ == "__main__":
    sys.exit(main())