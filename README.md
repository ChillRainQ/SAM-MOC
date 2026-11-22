# SAM-MOC E1

E1 使用 RGB、预训练 RWKV-SAM Image Encoder、三尺度 FPN、视觉 Decoder 和六通道密度 Head。通道顺序为 Ship、Vehicle、Building、Container、Tree、Airplane。E2 Class Query 和 E3 多原型文本待 E1 真实训练验证后接入。

## 源码核验

- 官方 ovsam 快照为 `b6f33a3`。[`VITAMINBackbone`](third_party/ovsam/projects/rwkvsam/models/backbones/vitamin.py#L397) 的公开 `forward(x)` 返回 NCHW 张量元组。在 `additional_final_layers=False` 时，顺序为 stride 4 的卷积 stage、stride 8 的卷积 stage、stride 16 的 RWKV 中间输出、stride 16 的最终归一化输出。E1 取第 1、第 2 和最后一个张量。512×512 输入的空间尺寸为 128×128、64×64、32×32。
- 本项目默认使用上游 ViTamin 源码的最大档 Large，三路通道为 160/320/1024，并沿用 [官方 Small 蒸馏配置](third_party/ovsam/projects/rwkvsam/configs/backbone_dist/rwkvsam1001_000_vith_vitamin_rwkv_small_mlp2.py#L40) 的 14 层 RWKV、MLP ratio 2 和无位置嵌入设置。Large 需要与此配置匹配的预训练 checkpoint。配置中的 `variant="small"` 对应 64/128/384，`variant="tiny"` 对应 32/64/192。所有版本均严格匹配权重。
- 官方配置的初始化指向本地分类预训练文件和 `backbone.` 前缀。蒸馏训练的学生模块命名为 `backbone_student`。本项目支持直接 encoder state dict、`state_dict` 包装和 `backbone_student.`、`backbone.`、`image_encoder.` 前缀，完整核对所有 encoder 参数。官方 RWKV-SAM 子项目与作者 OV-SAM 模型库均未找到匹配的、可验证下载的 RWKV-SAM 预训练文件。分类预训练与 SAM 蒸馏权重应在实验记录中注明来源。
- [WKV 扩展](third_party/ovsam/ext/rwkv/cls_backbones/backbones/vrwkv.py#L19) 由 PyTorch JIT 定位并编译两份 CUDA 源码，序列上限 65,536。原实现的半精度 backward 分支与四输入签名不匹配，且输出使用当前 CUDA 设备。本项目设置当前 GPU，并保持 encoder 在 FP32。`bf16=True` 只对 FPN、Decoder 和 Head 的可混合精度运算启用 autocast。Softplus、空间积分和 loss 用 FP32，TF32 关闭。扩展实际编译及 RTX 5090 兼容性待验证。
- 官方 NWPU-MOC 参考版本为 `7edae3a`。[MOC-6 配置](https://github.com/lyongo/NWPU-MOC/blob/7edae3a/datasets/setting/MOC_RS.py#L12) 使用官方 train/val/test 列表、`LABEL_FACTOR=8`、`LOG_PARA=100`，六类取 `gt/*.npz` 内 `arr_0` 的前六个通道。代码使用与 `GTScaleDown` 等价的整数映射累加落到同一网格的点。官方 MCC 使用 3 个 128 通道 FPN 输出，在 stride 8 位置拼接为 384 通道，再降到 64 并经过 SCA。本项目保留 P2/P3/P4 进入 Decoder，在 Decoder 内对齐至 stride 8。
- E1 默认使用 [官方 CC 的 `Mse_loss`](https://github.com/lyongo/NWPU-MOC/blob/7edae3a/models/CC.py#L86)。逐类使用 `15×15`、sigma 4 的零填充高斯卷积生成监督密度图，再计算预测与监督密度图的均方误差。评估真值从原始点图积分计算。
- 验证与测试直接调用 [官方 `eval_mc`](misc/utils.py#L71)，输出六类 MAE/RMSE、其宏平均和逐图加权平方误差均值 `cls_weight_mse`。验证和测试使用完整图像。RGB 和 NIR 使用官方加载器的增强和归一化。密度 Head 的初始均匀背景按训练 crop 尺寸折算为每类约 1 个目标。

## 运行

项目入口为 `train.py` 和 `test.py`。`test.py` 负责加载已有实验并运行测试，训练循环与共用的评估、绘图函数位于 `trainer.py`。类别顺序、标签步长和密度缩放在 `datasets/setting/MOC_RS.py`，项目路径和实验快照清单在 `project.py`，模型常量和训练配置在 `config.py`。`models/counters/mcc.py` 定义完整的 MCC 计数网络。`models/counter.py` 按 CC 的方式封装批次前向，训练时返回 `pred_map`、`counts` 和 `gauss_map`，训练器从 `model.loss` 读取本次损失。实验 checkpoint 保存 MCC 的参数。

训练、验证和测试直接使用官方 `datasets` 提供的 DataLoader。数据沿用官方 `rgb/`、`ir/`、`gt/` 布局，目录默认是项目根目录下的 `NWPU-MOC`，可在配置中设置 `data`。官方加载器读取配对 RGB 和 NIR 文件。

模型默认使用 RGB。设置 `nir=True` 后，Counter 取归一化 NIR 的第一通道。NIR 经独立卷积投影后以残差加入三层 FPN，残差权重初始化为零，RWKV-SAM 骨干继续接收三通道 RGB。

基础依赖和 `rwkv` 可选依赖已通过 `uv add` 写入 `pyproject.toml` 和 `uv.lock`。默认 Windows PyPI wheel 为 CPU 版，可用于开发测试。训练环境需要 CUDA 版 PyTorch 和匹配的 torchvision、NVIDIA CUDA Toolkit 和可用的 C++ 编译器。配置 CUDA wheel 时用 [PyTorch 官方安装矩阵](https://pytorch.org/get-started/previous-versions/) 选与 Toolkit、GPU 匹配的版本组合，并按 [uv 的 PyTorch 索引说明](https://docs.astral.sh/uv/guides/integration/pytorch/) 设置训练环境的索引。

训练入口直接使用 `config.py` 的 `cfg`。运行前填写 `cfg.PRE_BACKBONE_WEIGHT`。训练默认使用 GPU 2、RGB+NIR、512×512 crop、batch size 4、200 个 epoch 和 `Mse_loss`。短跑可以设 `cfg.MAX_STEPS = 3` 和 `cfg.VAL_SAMPLES = 4`。测试时编辑 `test.py` 的 `TEST_CONFIG`，用 `gpu_id` 指定 CUDA 设备，用 `model_path` 指定实验中已有的 MCC 权重，例如 `Path("exp/训练实验名/epoch_020.pth")`。测试读取该实验的 `config.json` 来恢复训练时的模型设置。

```powershell
uv run --no-sync python train.py
uv run --no-sync python test.py
```

完整训练与验证使用 `cfg.MAX_STEPS = None` 和 `cfg.VAL_SAMPLES = None`。验证按 NWPU-MOC 的零基 epoch 规则执行 `VAL_STAGE=[0,50,100,200]` 和 `VAL_FREQ=[10,4,4,4]`，控制台默认每 20 批打印一次。`TEST_CONFIG` 默认用 test split，设置 `split="val"` 可改用验证集。指标记录各类真值非零样本数，以及这些样本中预测低于 0.5 的比例，便于观察 Ship 塌缩和 Airplane 稀缺。

训练在 `cfg.EXP_PATH` 下建立实验目录，保存 `config.json`、`latest.pt` 和 `code/` 快照。每隔 `cfg.CP_FREQ` 个 epoch 和训练结束时另存 `epoch_XXX.pth`，包含模型、优化器和已完成的 epoch。通过 `cfg.RESUME=True` 和 `cfg.RESUME_PATH` 可从这类文件续训，并把本次日志写入新实验目录。测试支持裸 `state_dict` 和权重存在 `net` 键里的 checkpoint，自动去除 DataParallel 的 `module.` 前缀，并严格校验 MCC 参数。评估结果保存到实验目录的 `test_<时间戳>/` 子目录。设置 `split="val"` 时使用 `val_<时间戳>/`。

### 切换损失

在 `config.py` 中设置 `cfg.LOSS_FUNCTION`，例如 `cfg.LOSS_FUNCTION = "Cos_loss"`。

| 选项 | 官方 CC 的计算方式 |
| --- | --- |
| `Mse_loss` | 对高斯密度标签计算 MSE |
| `Cos_loss` | 对原始点图计算 MSE，加预测类别密度图之间的余弦项，权重用 `cfg.alpha` 设置，默认 1 |
| `Mix_loss` | 对原始点图计算 MSE，加权重为 `1e-4` 的整图总量 L1。总量在六类和空间维度一起求和 |
| `Mask_loss` | 对高斯密度标签计算 MSE，加权重为 `1e-4` 的逐类二分类交叉熵之和 |

`Mask_loss` 使用 MCC 的辅助掩码头。训练需设置 `cfg.MASK_DIR`，其中每张图对应一个与点标签同名的 NPZ，键为 `mask_map`，形状为 `[H,W,6]`，值为 0 或 1。掩码沿用与图像相同的几何变换，按最近邻下采样至密度图网格。评估时从 checkpoint 自动识别掩码头。

本机已验证计数模块的 CPU 和 RTX 4060 FP32/BF16 前反向、真实 NPZ 标签的计数守恒，以及官方 evaluator 和图表输出。RTX 4060 测试使用机器已有的 CUDA 版 PyTorch 和测试专用视觉编码器。相邻 NWPU-MOC 的三个官方 split 所引用 RGB 文件均为零字节，本机也未找到匹配的 RWKV-SAM 预训练 checkpoint。真实图像训练、原始 WKV 扩展验证和实验指标仍待补齐。
