"""
批量绘制优化器 — 将多个静态网格合并为巨型 VBO，使用 glMultiDrawArraysIndirect 批量绘制。

核心思想：
  1. 将所有静态场景物体的顶点合并到同一个 VBO
  2. 每个物体附带材质参数（metallic, roughness, hasTex 等）
  3. 使用 glMultiDrawArraysIndirect 一次调用完成所有绘制
  4. 材质参数通过 Shader Storage Buffer Object (SSBO) 传递给着色器

用法：
  from batch_renderer import BatchRenderer
  batch = BatchRenderer()
  batch.add_subset("赛道", track_verts,  metallic=0.0, roughness=0.98)
  batch.add_subset("护栏", guard_verts,  metallic=0.5, roughness=0.5)
  batch.add_subset("路灯", lamp_verts,   metallic=0.65, roughness=0.45)
  batch.upload()  # 合并顶点 + 上传 + 生成间接命令
  batch.draw()    # glMultiDrawArraysIndirect
"""

import ctypes
import numpy as np
from OpenGL.GL import *

# 顶点格式（扩展版）：每顶点 12 个 float32
# [px, py, pz, nx, ny, nz, u, v, r, g, b, material_id]
# material_id 用于在着色器中查找材质参数
VERTEX_STRIDE_FLOATS = 12
VERTEX_STRIDE_BYTES = VERTEX_STRIDE_FLOATS * 4


class Subset:
    """描述一个子网格"""
    __slots__ = ('name', 'vertices', 'first', 'count', 'metallic', 'roughness',
                 'has_texture', 'is_building', 'is_emissive')

    def __init__(self, name, vertices, metallic=0.0, roughness=0.5,
                 has_texture=False, is_building=False, is_emissive=False):
        self.name = name
        self.vertices = vertices          # (N, 11) float32 原始顶点
        self.first = 0                    # 在合并 VBO 中的起始顶点索引
        self.count = 0                    # 顶点数
        self.metallic = metallic
        self.roughness = roughness
        self.has_texture = has_texture
        self.is_building = is_building
        self.is_emissive = is_emissive


class BatchRenderer:
    """
    批量渲染器。
    将多个子网格合并为一个巨型 VBO，使用间接绘制。
    """

    def __init__(self, max_subsets=32):
        self.subsets = []           # list of Subset
        self.max_subsets = max_subsets

        # GPU 资源
        self.vao = 0
        self.vbo = 0
        self.cmd_buffer = 0          # 间接命令 buffer
        self.material_ssbo = 0       # 材质参数 SSBO

        # 已上传的顶点数
        self.total_vertices = 0
        self.total_subsets = 0

        # 材质参数 buffer (CPU 侧)
        self._material_data = np.zeros((max_subsets, 4), dtype=np.float32)
        # 每行: [metallic, roughness, flags, padding]
        # flags: bit0=has_texture, bit1=is_building, bit2=is_emissive

    def add_subset(self, name, vertices, **kwargs):
        """
        添加子网格。
        vertices: (N, 11) float32 — 标准顶点格式 [px3, nx3, uv2, color3]
        kwargs: metallic, roughness, has_texture, is_building, is_emissive
        """
        if len(self.subsets) >= self.max_subsets:
            raise RuntimeError(f"超出最大子网格数 ({self.max_subsets})")
        subset = Subset(name, vertices, **kwargs)
        subset.count = len(vertices)
        self.subsets.append(subset)

    def upload(self):
        """将所有子网格合并上传到 GPU，生成间接命令和材质 SSBO。"""
        if not self.subsets:
            return

        # ---- 计算总顶点数和偏移 ----
        offset = 0
        for ss in self.subsets:
            ss.first = offset
            offset += ss.count
        self.total_vertices = offset
        self.total_subsets = len(self.subsets)

        # ---- 合并顶点（从 11→12 float32，增加 material_id） ----
        merged = np.zeros((self.total_vertices, VERTEX_STRIDE_FLOATS), dtype=np.float32)
        for mat_id, ss in enumerate(self.subsets):
            start = ss.first
            end = start + ss.count
            # 复制基本属性
            merged[start:end, :11] = ss.vertices[:, :11]
            # 写入 material_id（整数转为 float）
            merged[start:end, 11] = float(mat_id)
            # 填充材质参数表
            flags = 0
            if ss.has_texture:   flags |= 1
            if ss.is_building:   flags |= 2
            if ss.is_emissive:   flags |= 4
            self._material_data[mat_id, 0] = ss.metallic
            self._material_data[mat_id, 1] = ss.roughness
            self._material_data[mat_id, 2] = float(flags)

        # ---- 生成间接命令 ----
        # DrawArraysIndirectCommand = 4 x uint32 (16 字节):
        #   [count, instanceCount, first, baseInstance]
        # 注意：之前用了 (N,5) 导致内存布局错位，GPU 读到的 count 全是 0，batch 空绘制
        cmd = np.zeros((self.total_subsets, 4), dtype=np.uint32)
        for i, ss in enumerate(self.subsets):
            cmd[i, 0] = ss.count          # count
            cmd[i, 1] = 1                 # instanceCount
            cmd[i, 2] = ss.first          # first (=first vertex)
            cmd[i, 3] = i                 # baseInstance (=material_id)

        # ---- 上传到 GPU ----
        # VBO
        if self.vbo == 0:
            self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, merged.nbytes, merged, GL_STATIC_DRAW)

        # VAO
        if self.vao == 0:
            self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        # 属性 0: pos (vec3)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE_BYTES, ctypes.c_void_p(0))
        # 属性 1: normal (vec3)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE_BYTES, ctypes.c_void_p(12))
        # 属性 2: uv (vec2)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, VERTEX_STRIDE_BYTES, ctypes.c_void_p(24))
        # 属性 3: color (vec3)
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, VERTEX_STRIDE_BYTES, ctypes.c_void_p(32))
        # 属性 4: material_id (float)  — 顶点属性，每个顶点携带材质 ID
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, VERTEX_STRIDE_BYTES, ctypes.c_void_p(44))

        # 间接命令 buffer
        if self.cmd_buffer == 0:
            self.cmd_buffer = glGenBuffers(1)
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, self.cmd_buffer)
        glBufferData(GL_DRAW_INDIRECT_BUFFER, cmd.nbytes, cmd, GL_STATIC_DRAW)

        # 材质 SSBO（绑定到 binding=4）
        if self.material_ssbo == 0:
            self.material_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.material_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self._material_data.nbytes,
                     self._material_data, GL_STATIC_DRAW)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, self.material_ssbo)

        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0)

        # 自检：确保每个 subset 的 count > 0 且是 3 的倍数
        for i, ss in enumerate(self.subsets):
            if ss.count == 0:
                print(f"[Batch] 警告: 子网格 '{ss.name}' 顶点数为 0，将被跳过")
            elif ss.count % 3 != 0:
                print(f"[Batch] 警告: 子网格 '{ss.name}' 顶点数 {ss.count} 不是 3 的倍数！")

        print(f"[Batch] 合并 {self.total_subsets} 个子网格, "
              f"{self.total_vertices:,} 顶点, "
              f"VBO={self.vbo}, VAO={self.vao}, CMD={self.cmd_buffer}")

    def draw(self):
        """执行 glMultiDrawArraysIndirect。"""
        if self.vao == 0 or self.cmd_buffer == 0:
            return
        glBindVertexArray(self.vao)
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, self.cmd_buffer)
        # 确保 SSBO 绑定
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, self.material_ssbo)
        # glMultiDrawArraysIndirect(mode, indirect, drawcount, stride)
        # stride=0 表示命令紧密排列（每个 16 字节）
        glMultiDrawArraysIndirect(GL_TRIANGLES, ctypes.c_void_p(0),
                                  self.total_subsets, 0)
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0)
        glBindVertexArray(0)

    def delete(self):
        """释放 GPU 资源。"""
        if self.vao:  glDeleteVertexArrays(1, [self.vao])
        if self.vbo:  glDeleteBuffers(1, [self.vbo])
        if self.cmd_buffer: glDeleteBuffers(1, [self.cmd_buffer])
        if self.material_ssbo: glDeleteBuffers(1, [self.material_ssbo])
        self.vao = self.vbo = self.cmd_buffer = self.material_ssbo = 0