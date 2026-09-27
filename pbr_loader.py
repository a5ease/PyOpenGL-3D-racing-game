"""
PBR 贴图加载器 — 从本地 2K/4K 真实 PBR 纹理文件加载并上传到 GPU。

支持的通道：
  - Albedo（漫反射） → RGB
  - Normal（法线）   → RGB（从切线空间法线贴图转换或直接使用）
  - Roughness（粗糙度） → R 通道（或从 RGBA 中提取）
  - Metallic（金属度）  → R 通道
  - AO（环境光遮蔽）   → R 通道
  - Height（高度/视差） → R 通道

文件命名约定（不区分大小写）：
  {layer}_albedo.png  / {layer}_diffuse.png  / {layer}_basecolor.png
  {layer}_normal.png  / {layer}_nor.png
  {layer}_roughness.png / {layer}_rough.png
  {layer}_metallic.png / {layer}_metal.png / {layer}_met.png
  {layer}_ao.png
  {layer}_height.png / {layer}_disp.png / {layer}_parallax.png

用法：
  textures = load_pbr_set("terrain_grass", r"assets\textures")
  # 返回 { 'albedo': tex_id, 'normal': tex_id, 'roughness': tex_id,
  #        'metallic': tex_id, 'ao': tex_id, 'height': tex_id }
"""

import os
import re
import numpy as np
from PIL import Image
from OpenGL.GL import *

# 默认 PBR 纹理目录（相对于项目根目录）
_DEFAULT_PBR_DIR = os.path.join(os.path.dirname(__file__), "assets", "textures")

# 缓存目录（后备方案）
_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


def _find_file(base_dir, layer_name, keywords):
    """在 base_dir 中查找匹配 layer_name + 任一 keyword 的文件（不区分大小写）"""
    if not os.path.isdir(base_dir):
        return None
    pattern = re.compile(rf"^{re.escape(layer_name)}_({ '|'.join(keywords) })\.png$", re.IGNORECASE)
    for fname in os.listdir(base_dir):
        if pattern.match(fname):
            return os.path.join(base_dir, fname)
    return None


def _load_grayscale(path, width=None, height=None):
    """加载灰度图（单通道），必要时缩放到目标尺寸"""
    img = Image.open(path).convert('L')
    if width and height and (img.width != width or img.height != height):
        img = img.resize((width, height), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def _load_rgb(path, width=None, height=None):
    """加载 RGB 图，必要时缩放"""
    img = Image.open(path).convert('RGB')
    if width and height and (img.width != width or img.height != height):
        img = img.resize((width, height), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def _upload_tex_2d(img, internal_fmt=GL_RGB, fmt=GL_RGB, repeat=True, anisotropy=True):
    """上传 2D 纹理到 GPU，生成 mipmap"""
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, internal_fmt, img.shape[1], img.shape[0],
                 0, fmt, GL_UNSIGNED_BYTE, img)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT if repeat else GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT if repeat else GL_CLAMP_TO_EDGE)
    glGenerateMipmap(GL_TEXTURE_2D)
    if anisotropy:
        try:
            max_aniso = glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, min(16.0, max_aniso))
        except Exception:
            pass
    return tex


def load_pbr_albedo(base_dir, layer_name, width=None, height=None):
    """加载 Albedo 纹理"""
    path = (_find_file(base_dir, layer_name, ["albedo", "diffuse", "basecolor", "base_color", "col", "color", "d"])
            or _find_file(base_dir, layer_name, ["albedo", "diffuse", "basecolor"]))
    if path:
        img = _load_rgb(path, width, height)
        return _upload_tex_2d(img, GL_SRGB8_ALPHA8 if img.shape[2] == 4 else GL_SRGB8, GL_RGBA if img.shape[2] == 4 else GL_RGB)
    return None


def load_pbr_normal(base_dir, layer_name, width=None, height=None):
    """加载法线贴图"""
    path = (_find_file(base_dir, layer_name, ["normal", "nor", "nm", "nrm", "n"])
            or _find_file(base_dir, layer_name, ["normal"]))
    if path:
        img = _load_rgb(path, width, height)
        return _upload_tex_2d(img, GL_RGB, GL_RGB)
    return None


def load_pbr_roughness(base_dir, layer_name, width=None, height=None):
    """加载粗糙度贴图（单通道 → 复制到 RGB）"""
    path = (_find_file(base_dir, layer_name, ["roughness", "rough", "rgh", "r"])
            or _find_file(base_dir, layer_name, ["roughness", "rough"]))
    if path:
        g = _load_grayscale(path, width, height)
        img = np.stack([g, g, g], axis=-1)
        return _upload_tex_2d(img, GL_RGB, GL_RGB)
    return None


def load_pbr_metallic(base_dir, layer_name, width=None, height=None):
    """加载金属度贴图"""
    path = (_find_file(base_dir, layer_name, ["metallic", "metal", "met", "m"])
            or _find_file(base_dir, layer_name, ["metallic", "metal"]))
    if path:
        g = _load_grayscale(path, width, height)
        img = np.stack([g, g, g], axis=-1)
        return _upload_tex_2d(img, GL_RGB, GL_RGB)
    return None


def load_pbr_ao(base_dir, layer_name, width=None, height=None):
    """加载 AO 贴图"""
    path = (_find_file(base_dir, layer_name, ["ao", "ambient_occlusion", "ambientocclusion"])
            or _find_file(base_dir, layer_name, ["ao"]))
    if path:
        g = _load_grayscale(path, width, height)
        img = np.stack([g, g, g], axis=-1)
        return _upload_tex_2d(img, GL_RGB, GL_RGB)
    return None


def load_pbr_height(base_dir, layer_name, width=None, height=None):
    """加载高度/视差贴图"""
    path = (_find_file(base_dir, layer_name, ["height", "disp", "parallax", "h", "depth"])
            or _find_file(base_dir, layer_name, ["height", "disp", "parallax"]))
    if path:
        g = _load_grayscale(path, width, height)
        img = np.stack([g, g, g], axis=-1)
        return _upload_tex_2d(img, GL_RGB, GL_RGB)
    return None


def load_pbr_set(base_dir, layer_name, width=None, height=None):
    """
    加载完整的 PBR 纹理集。

    参数：
        base_dir:   PBR 纹理目录
        layer_name: 图层名（如 "terrain_grass", "terrain_rock", "track"）
        width/height: 目标尺寸（None = 使用原图尺寸）

    返回 dict:
        { 'albedo': tex_id or None, 'normal': ..., 'roughness': ...,
          'metallic': ..., 'ao': ..., 'height': ... }
    """
    return {
        'albedo':    load_pbr_albedo(base_dir, layer_name, width, height),
        'normal':    load_pbr_normal(base_dir, layer_name, width, height),
        'roughness': load_pbr_roughness(base_dir, layer_name, width, height),
        'metallic':  load_pbr_metallic(base_dir, layer_name, width, height),
        'ao':        load_pbr_ao(base_dir, layer_name, width, height),
        'height':    load_pbr_height(base_dir, layer_name, width, height),
    }


# ============================================================
# 便捷函数：从单个 RGBA 通道图加载 packed PBR 纹理
# 某些 PBR 纹理集将 Roughness/Metallic/AO 打包到一张图的不同通道
# 例如：R=Roughness, G=Metallic, B=AO
# ============================================================
def load_pbr_packed_orm(base_dir, layer_name, width=None, height=None):
    """
    加载 ORM 打包图（Occlusion-Roughness-Metallic 打包在同一张图的不同通道）。
    命名匹配：{layer}_orm.png 或 {layer}_packed.png
    通道布局：R=AO, G=Roughness, B=Metallic （标准 ORM 约定）

    返回 (ao_tex, roughness_tex, metallic_tex) 或 (None, None, None)
    """
    path = (_find_file(base_dir, layer_name, ["orm", "packed", "arm"])
            or _find_file(base_dir, layer_name, ["orm"]))
    if not path:
        return None, None, None

    img = Image.open(path).convert('RGB')
    if width and height and (img.width != width or img.height != height):
        img = img.resize((width, height), Image.LANCZOS)
    arr = np.array(img, dtype=np.uint8)

    # R → AO, G → Roughness, B → Metallic
    ao_img  = np.stack([arr[:,:,0], arr[:,:,0], arr[:,:,0]], axis=-1)
    rough_img = np.stack([arr[:,:,1], arr[:,:,1], arr[:,:,1]], axis=-1)
    metal_img = np.stack([arr[:,:,2], arr[:,:,2], arr[:,:,2]], axis=-1)

    return (
        _upload_tex_2d(ao_img, GL_RGB, GL_RGB),
        _upload_tex_2d(rough_img, GL_RGB, GL_RGB),
        _upload_tex_2d(metal_img, GL_RGB, GL_RGB),
    )


def load_pbr_normal_from_height(base_dir, layer_name, strength=2.0, width=None, height=None):
    """
    如果没有法线贴图，从高度图生成法线贴图（后备方案）。
    使用 Sobel 滤波计算高度梯度。
    """
    path = (_find_file(base_dir, layer_name, ["height", "disp", "parallax", "h"])
            or _find_file(base_dir, layer_name, ["height"]))
    if not path:
        return None

    hmap = _load_grayscale(path, width, height).astype(np.float32) / 255.0
    sy, sx = hmap.shape

    # Sobel 梯度
    gx = np.zeros_like(hmap)
    gy = np.zeros_like(hmap)
    gx[1:-1, 1:-1] = (hmap[1:-1, 2:] - hmap[1:-1, :-2]) * 0.5
    gy[1:-1, 1:-1] = (hmap[2:, 1:-1] - hmap[:-2, 1:-1]) * 0.5

    # 法线 = normalize(-gx * strength, 1.0, -gy * strength)
    normal = np.zeros((sy, sx, 3), dtype=np.float32)
    normal[:,:,0] = -gx * strength
    normal[:,:,1] = 1.0
    normal[:,:,2] = -gy * strength
    length = np.sqrt(np.sum(normal * normal, axis=2, keepdims=True))
    length = np.maximum(length, 1e-6)
    normal = normal / length
    normal = ((normal + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)

    return _upload_tex_2d(normal, GL_RGB, GL_RGB)


def load_all_terrain_pbr(base_dir=None):
    """
    加载全部 4 层地形的 PBR 纹理集。
    返回 {
        'grass':  { 'albedo': tex, 'normal': tex, 'roughness': tex, 'metallic': tex, 'ao': tex, 'height': tex },
        'rock':   { ... },
        'dirt':   { ... },
        'gravel': { ... },
    }
    如果有缺失的纹理，对应的 tex_id 为 None。
    """
    if base_dir is None:
        base_dir = _DEFAULT_PBR_DIR

    layers = ['terrain_grass', 'terrain_rock', 'terrain_dirt', 'terrain_gravel']
    result = {}
    for layer in layers:
        result[layer.replace('terrain_', '')] = load_pbr_set(base_dir, layer)
    return result


def load_track_pbr(base_dir=None):
    """加载赛道 PBR 纹理集"""
    if base_dir is None:
        base_dir = _DEFAULT_PBR_DIR
    return load_pbr_set(base_dir, "track")


# ============================================================
# 自检：打印 PBR 目录下可用的纹理清单
# ============================================================
def list_available_pbr_textures(base_dir=None):
    """列出 PBR 目录下所有可用的纹理文件"""
    if base_dir is None:
        base_dir = _DEFAULT_PBR_DIR
    if not os.path.isdir(base_dir):
        print(f"[PBR] 纹理目录不存在: {base_dir}")
        return

    print(f"[PBR] 可用纹理文件 ({base_dir}):")
    for fname in sorted(os.listdir(base_dir)):
        if fname.endswith('.png') or fname.endswith('.jpg'):
            fpath = os.path.join(base_dir, fname)
            size = os.path.getsize(fpath)
            img = Image.open(fpath)
            print(f"  {fname:40s} {img.size[0]:4d}x{img.size[1]:<4d} {img.mode:4s} {size//1024:4d}KB")