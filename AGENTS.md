# Project Brief: SAMbaCrack - A Novel Crack Segmentation Model

Hello Gemini, we are building a new deep learning model for high-precision road crack segmentation, named **SAMbaCrack**. Please act as my expert PyTorch co-pilot for this task.

Our project is based on the existing `SCSegamba` codebase, but we will replace its original single encoder with a more powerful dual-encoder architecture inspired by the `SAMba-UNet` paper.

---

### **Overall Architecture**

The target model `SAMbaCrack` will be an **Encoder-Decoder** architecture.

- **Encoder**: A dual-path encoder that processes the input image in parallel.
  - **Path 1**: A **SAM2 Encoder** (based on Hiera architecture), pre-trained on natural images, responsible for extracting fine-grained local details and textures. We will apply parameter-efficient fine-tuning (PEFT) to it.
  - **Path 2**: A **Mamba Encoder** (based on `SCSegamba`'s `SAVSS_layer`), responsible for capturing long-range dependencies and the global context of cracks.
  - **Fusion**: A **HOACM** (Heterogeneous Omni-Attention Convergence Module) will be used at each stage to intelligently fuse the features from the SAM2 and Mamba paths.
- **Decoder**: We will use the **MFSHead** (Multi-scale Feature Segmentation Head) from `SCSegamba` to decode the fused features and generate the final segmentation map.

---

### **Task 1: Implement the Core Modules**

Before we build the final model, we need to implement the new modules. Please help me create the following files and classes in the `/samba_unet_modules/` directory.

#### **File: `/samba_unet_modules/hiera.py`**

Create a `Hiera` class that inherits from `nn.Module`. This will be our SAM2 Encoder. The implementation should be a standard hierarchical Vision Transformer. The `forward_features` method must return a list or tuple of 4 feature maps from the 4 stages.

#### **File: `/samba_unet_modules/refiner_adapter.py`**

Create two classes in this file:
1.  `DynamicFeatureFusionRefiner(nn.Module)`: Implement this based on the `SAMba-UNet` paper's architecture (dual-pooling, channel attention, spatial enhancement path, residual connection).
2.  `MLPAdapter(nn.Module)`: A simple `Linear -> GeLU -> Linear` adapter module.

#### **File: `/samba_unet_modules/hoacm.py`**

Create the `HOACM(nn.Module)` class. This is the most complex module.
- It should contain sub-modules: `BifurcatedSelectiveEmphasisAttention` (BSEA) and `OmniscientContextualAttention` (OCA).
- The `forward` method must accept two inputs, `x_sam` and `x_mamba`, and fuse them according to the `SAMba-UNet` paper's diagram.

---

### **Task 2: Assemble the Main Model**

Now, help me create the main model file that assembles all the components.

#### **File: `/mmcls/SAVSS_dev/models/SAVSS/SAMbaCrack.py`**

Create the main model class `SAMbaCrack(nn.Module)`.

-   **`__init__` method**:
    -   Instantiate `Hiera` as `self.sam_encoder`.
    -   Instantiate `nn.ModuleList` of `Refiner` and `Adapter` for each of the 4 stages.
    -   Instantiate a Mamba Encoder `self.mamba_encoder` by stacking `SAVSS_layer` and `DownSample` modules for 4 stages.
    -   Instantiate an `nn.ModuleList` of `HOACM` for each of the 4 stages.
    -   Instantiate the `MFSHead` from `SCSegamba`'s code as `self.decoder`.
-   **`forward` method**:
    -   Implement the full data flow: parallel encoding through SAM2 and Mamba paths, feature refinement/adaptation for SAM2 features, stage-wise fusion using HOACM, and finally decoding with the MFSHead.
    -   The method should return the final segmentation logits.

Let's start with **Task 1**, beginning with the `hiera.py` file. Please generate the code for it.

# 最重要的一点：用中文回答所有问题，并且给所有你浏览过的代码加上中文注释，把作者改成Roy

## Gemini生成的实现时需要特别注意的关键点
你的设计在理论上非常完美，但在将它翻译成代码时，有几个细节需要特别小心处理，否则容易出错：
1. 维度对齐 (Dimension Alignment): 这是最最关键的一点。正如你之前的分析，Hiera 和 SAVSS 的 Patch Embedding 策略不同，导致它们在同一阶段输出的特征图空间分辨率完全不同。
解决方案: 在将 sam_features 和 mamba_features 送入 HOACM 之前，必须使用 torch.nn.functional.interpolate 将分辨率较小的特征图（Hiera的输出）上采样到与分辨率较大的特征图（SAVSS的输出）完全一致。
检查点: 确保每个 HOACM 模块接收到的两个输入的 H 和 W 是完全相同的。
2. 通道数对齐:
你需要确保在每个阶段，Hiera 输出的通道数和 SAVSS 输出的通道数是一致的，这样 HOACM 才能处理。这通常需要在定义两个编码器时，就规划好每个阶段的输出维度。例如，都遵循 [96, 192, 384, 768] 这样的通道演进策略。
3. 参数冻结与优化器设置:
一定要正确实现参数高效微调的逻辑。在创建优化器时，必须确保只有你希望训练的参数（Adapter, Refiner, HOACM, Mamba Encoder, Decoder）被传入，而 Hiera 的主体部分参数的 requires_grad 属性为 False。否则，你的模型会非常难以训练，甚至导致灾难性遗忘。
4. 输入归一化 (Input Normalization):
SAM/Hiera 是在特定的均值和方差下进行预训练的。你需要确保你的数据预处理流程中，对输入图像的归一化方式与 SAM 的要求一致。否则，预训练权重无法发挥最佳效果。