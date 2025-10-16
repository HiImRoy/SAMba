# AGENT.md - 项目结构与协作指南

## 1. 核心目标

本项目旨在实现并优化一个名为 **SAMbaCrack** 的先进深度学习模型，用于高精度的图像裂缝分割任务。该模型的核心是一个**对称的双编码器-单解码器架构**，创新地融合了**Hiera (SAM) Transformer** 和一个**分层的SAVSS (Mamba)** 骨干网络。

## 2. 协作基本原则

1.  **禁止假设**: 你的所有回答和代码修改**必须**基于项目中实际存在的文件和代码。不要猜测或引入任何不存在的函数、变量或逻辑。
2.  **中文注释**: 在生成或修改任何Python代码时，请为其添加详尽的**中文注释**，解释代码块的功能、关键变量的含义以及数据流的变化。
3.  **作者署名**: 所有由你创建或修改过的文件，请确保文件头部的作者信息被更新为 `Roy`。
    ```python
    # Author: Roy
    # Copyright (c) Roy. All rights reserved.
    ```
4.  **变更确认**: 在进行任何涉及**模型结构、维度、核心算法或训练超参数**（如学习率、批量大小、损失函数权重等）的调整之前，**必须**先向我提出修改方案并寻求我的确认。得到我的明确许可后，方可生成最终代码。
5.  **最高指示权**: 我的指令拥有最高优先级。如果我的要求非常明确且肯定，即使它看起来与常规做法有所不同，也请**严格按照我的指示执行**。我说的就是正确的。

## 3. 核心文件详解

#### `main.py`
*   **功能**: 整个项目的启动器。负责解析命令行参数、设置随机种子、创建模型和数据集、定义优化器和学习率调度器，并掌管整个训练/评估循环的流程。
*   **数据流**:
    1.  调用 `get_args_parser()` 获取所有超参数。
    2.  调用 `models.build_model()` 创建 `SAMbaCrack` 模型实例。
    3.  调用 `datasets.create_dataset()` 创建数据加载器。
    4.  进入主循环，在每个epoch中：
        *   调用 `engine.train_one_epoch()` 进行一轮训练。
        *   执行验证步骤，计算评估指标 (mIoU, ODS等)。
        *   保存最佳模型权重 (`checkpoint_best.pth`)。
    5.  训练结束后，保存日志和图表。

#### `SAMbaCrack.py`
*   **功能**: **项目的灵魂**。定义了 `SAMbaCrack` 模型的完整架构。
*   **内部结构**:
    1.  **`__init__`**:
        *   实例化**Hiera编码器** (`self.sam_encoder`)，并冻结其参数。
        *   实例化并行的`Refiner`和`Adapter`模块列表 (`self.refiners`, `self.adapters`)。
        *   **[关键]** 实例化一个**分层的SAVSS编码器**，包含 `self.savss_patch_embed`、`self.savss_stages` (多个`SAVSS_Layer`) 和 `self.savss_downsamplers` (多个`PatchMerging`层)。
        *   实例化`HOACM`融合模块列表 (`self.hoacms`)。
        *   实例化`UNetDecoder`解码器 (`self.decoder`)。
    2.  **`forward_savss_encoder`**: 实现SAVSS编码器的分层前向传播，输出一个4级的特征金字塔。
    3.  **`forward`**:
        *   并行调用 `self.sam_encoder` 和 `self.forward_savss_encoder` 获取两个分支的特征金字塔。
        *   对SAM的特征应用 `Refiner` 和 `Adapter`。
        *   在每个层级，将两个分支的特征送入对应的`HOACM`模块进行融合。
        *   将融合后的特征金字塔送入`decoder`，得到最终的分割预测图。

#### `hiera.py`
*   **功能**: 定义Hiera (SAM) 编码器模型。这是一个层级式的Vision Transformer，通过带步长的卷积实现下采样，构建特征金字塔。

#### `SAVSS_layer.py`
*   **功能**: 定义单个SAVSS (State-Space Model / Mamba) 层。这是SAVSS编码器的基本构建块，负责处理序列化后的图像特征。

#### `hoacm.py`
*   **功能**: 定义异构全注意力融合模块 (HOACM)。
*   **内部结构**:
    *   `OmniscientContextualAttention (OCA)`: 专门用于增强SAM分支的特征。
    *   `BifurcatedSelectiveEmphasisAttention (BSEA)`: 专门用于增强SAVSS (Mamba) 分支的特征。
    *   主 `HOACM` 类将 `OCA` 和 `BSEA` 的输出进行融合处理。

#### `refiner_adapter.py`
*   **功能**: 定义用于微调冻结的Hiera编码器特征的模块。
*   **内部结构**:
    *   `MLPAdapter`: 一个简单的MLP块。
    *   `DynamicFeatureFusionRefiner (DFFR)`: 一个通过注意力和空间增强路径来适应特定领域图像的复杂模块。

#### `decoder.py`
*   **功能**: 定义了UNet风格的解码器 (`UNetDecoder`) 和模型使用的损失函数 (`bce_dice`)。解码器负责将编码器提取的深层特征上采样，并与跳跃连接融合，最终生成分割图。

#### `engine.py`
*   **功能**: 包含核心的训练循环函数 `train_one_epoch`。它负责处理一个epoch内的数据迭代、模型前向/反向传播和优化器步骤。

#### `crack_dataset.py`
*   **功能**: 定义了`CrackDataset`类，负责加载裂缝图像和对应的标签。它处理数据的读取、动态的训练/验证集划分、数据增强（如翻转、仿射变换）和最终的张量转换。

---
**签名**

**Author: Roy**