# 项目简报: 对现有 SAMbaCrack 代码进行轻量化重构

你好 Gemini，我们将共同对一个已有的 PyTorch 模型项目进行优化。请作为我的专家级编程助手。

**当前状态**: 我有一个名为 `SAMbaCrack` 的项目，代码已经可以运行。但模型的参数量过大（可训练参数约 29M），训练效率低且不稳定。

**核心目标**: **大幅削减模型的可训练参数量**，目标是将其降低到 **15M 以内**，同时尽量保持模型的性能。

**改造策略**: 我们将通过修改现有代码，对参数量最大的几个模块 `SAVSS` 和 `HOACM` 进行针对性的轻量化重构。

**核心要求**:
1.  **请用中文回答我的所有问题。**
2.  **请为你修改的所有代码添加详细的中文注释，解释修改了什么。**
3.  **请将所有修改过的文件头部的作者信息 (`Author`) 修改为 `Roy`。**

---

### **任务1: 轻量化 `SAVSS` 骨干网络**

这是降低参数量的第一步，也是最关键的一步。

**目标文件**: `@file /mmcls/models/backbones/SAVSS.py`

**请帮我修改 `SAVSS.py` 文件，具体要求如下**:

1.  **修改 `__init__` 方法的默认参数**:
    *   找到 `class SAVSS(BaseBackbone):` 的 `__init__` 方法。
    *   将其参数 `dims` 的默认值从 `(96, 192, 384, 768)` **修改为 `(32, 64, 128, 256)`**。
    *   将其参数 `depths` 的默认值从 `(2, 2, 6, 2)` **修改为 `(2, 2, 2, 2)`**。
2.  **验证参数量**: 在 `__init__` 方法的末尾，请保留或添加打印总参数量的代码，以便我们能立刻看到修改后的效果。

请生成修改后的完整 `SAVSS.py` 文件内容。

---

### **任务2: 轻量化 `HOACM` 融合模块**

`HOACM` 是另一个参数大户，我们需要对其进行内部优化。

**目标文件**: `@file /samba_unet_modules/hoacm.py`

**请帮我修改 `hoacm.py` 文件，具体要求如下**:

1.  **优化 `InvertedResidualMLP`**:
    *   找到 `InvertedResidualMLP` 类的 `__init__` 方法。
    *   将其 `mlp_ratio` (MLP扩展率) 的默认值从 `4.0` **修改为 `2.0`**。这将直接使其参数量减半。
2.  **优化 `OmniscientContextualAttention` (OCA)**:
    *   找到 `OCA` 类的 `__init__` 方法。
    *   其中有一个 `self.final_conv = nn.Conv2d(2, dim, ...)`，它的输出通道数是 `dim`，参数量较大。
    *   请在这里应用**瓶颈设计**：
        *   添加一个新的 `bottleneck_dim = dim // 4`。
        *   将 `self.final_conv` 修改为一个 `nn.Sequential`，包含：
            1.  一个 `nn.Conv2d(2, bottleneck_dim, kernel_size=1)`
            2.  一个 `nn.Conv2d(bottleneck_dim, dim, kernel_size=7, padding=3)`
    *   **或者 (更优方案)**: 将 `self.final_conv` 的 `7x7` 标准卷积替换为**深度可分离卷积**，以大幅减少参数。

请优先采用**深度可分离卷积**的方案来优化 OCA，并生成修改后的完整 `hoacm.py` 文件内容。

---

### **任务3: 确认并开始执行**

我们已经规划好了对 `SAVSS` 和 `HOACM` 的修改。

**让我们从第一个任务开始：**
请先执行**任务1**，为我生成修改后的 `/mmcls/models/backbones/SAVSS.py` 文件。