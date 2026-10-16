# AI AWB：学习式光源估计

目标是相机线性域中的全局光源估计：输入 linear camera-RGB，输出正值、L2 归一化的光源 RGB，随后用 `[eG/eR, 1, eG/eB]` 计算增益。第一版假设单一主光源。模型不直接预测绝对色温，也不替代传感器标定、CCM 或完整 ISP。

## 模型与范围

`TinyIlluminantNet` 使用 RGB 和有效覆盖率共 4 通道输入，经过 3 个步幅卷积块，分别预测局部正值 RGB 和置信度，按有效覆盖率加权后汇聚为全局光源。使用 `1 − cosine` 损失训练，角误差用于报告及验证选模。

置信度加权汇聚参考 [FC4 原作者工作](https://github.com/yuanming-hu/fc4)。本项目是独立的小模型、独立预处理与训练流程，不是 FC4 原架构复现，没有引入该仓库的 Python 2 / TensorFlow 环境或预训练参数。

普通 sRGB 图像的后期白平衡编辑是另一项任务，参见 [Deep White-Balance Editing 原作者项目](https://github.com/mahmoudnafifi/Deep_White_Balance)。本模型的训练与推理都使用明确处理过的线性相机 RGB。

## 环境

推荐 Python 3.12，PyTorch 2.6 或以上。基础经典工具可继续使用不含 PyTorch 的原环境。

Windows CPU 独立环境示例：

```powershell
python -m venv .venv-ai
.\.venv-ai\Scripts\python.exe -m pip install --upgrade pip
.\.venv-ai\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv-ai\Scripts\python.exe -m pip install -e .
```

服务器上使用已有 PyTorch/CUDA 镜像时，在该环境安装 `python -m pip install -e .` 即可，不必重新覆盖镜像中的 PyTorch。新环境请按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)选择匹配的平台与 CUDA 构建，再安装本项目。

运行前检查实际环境：

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

下面命令中的 `python` 应指向该环境的解释器。本地验证可用 `--device cpu`，服务器正式训练可用 `--device cuda`。本项目不负责创建或计费服务器。

## 数据清单

```csv
image,illuminant_r,illuminant_g,illuminant_b,black_level,white_level,mask,group_id,grouping_unit,dataset_type
images/scene001.npy,0.8,1.0,0.6,0,1,masks/scene001.npy,scene001,physical_scene,real
```

- 图像、掩膜路径相对 CSV。输入为非负、有限的线性 RGB，必须显式填写黑白电平。
- 真值是与扣黑电平图像配对的光源 RGB，不是相机增益，不再额外扣黑。
- mask 非零为有效、零为排除。原图中的色卡、失效点必须通过掩膜排除；单纯将它们涂黑不能代替提供 mask。
- 预处理先屏蔽无效和饱和像素，再分别缩放 `image×valid` 和 `valid`，相除得到有效像素的颜色，避免色卡颜色通过插值泄漏。第四通道保留覆盖率。RGB 以有效区域的统一最大值归一，保留通道比例。
- `group_id` 代表分组；相关场景、裁剪和同场景多相机采样应同组。程序检查 train/val/test 之间的组标签和规范路径重复，但不能自动证明标签语义正确。`grouping_unit` 应明确说明标签来自实际场景、官方 fold 或其他来源；省略时记录为语义未核实。
- `dataset_type` 必须为 `real` 或 `synthetic_smoke_test`。合成数据及其模型默认拒绝训练、评测和预测，只有显式 `--allow-synthetic` 才能用于流程自检。

## 训练、测试、推理

```powershell
python -m awb_lab.learning train data/train.csv data/val.csv --output-dir outputs/ai-model --epochs 20 --batch-size 8 --input-size 128 --seed 42 --device cuda
python -m awb_lab.learning evaluate outputs/ai-model/best.pt data/test.csv --output-dir outputs/ai-test --device cuda
python -m awb_lab.learning predict outputs/ai-model/best.pt data/linear.npy --black-level 0 --white-level 1 --mask data/valid.npy --output outputs/corrected.png --device cuda
```

训练只读 train 和 val；只按 val 的 mean recovery angular error 选择 `best.pt`。最终测试清单在评测时另行传入，并与 checkpoint 中记录的训练/验证组和路径检查交集。

`training.json` 保存每轮损失、验证角误差和所选 epoch。测试输出每张图的 CNN 及四种经典算法预测，以及 recovery / reproduction angular error 的 mean、median、worst-25% mean。任何样本失败会终止本次报告，不静默删除失败样本。

计时包括的工作不同：CNN 记录缩放/掩膜/设备传输与前向过程，经典算法在原图尺寸运行。当前单次计时没有预热，不作为公平速度排行榜。服务器性能实验应另行固定设备、分辨率、线程和预热配置。

checkpoint 包含架构、输入尺寸、预处理标识、数据清单 SHA256、分组语义、种子和选模信息，使用 `weights_only=True` 和严格字段/张量验证加载。当前保存验证最优权重；优化器断点续训、AMP、DDP 和 ONNX 导出属于后续扩展。

## 真实缩略图初步实验

本地已完成此次初步实验，CNN 测试均值 4.0914°，最佳验证轮次为 18；四种经典对照、全部逐图误差及限制见[实验报告](../reports/ffcc_pilot/README.md)。此结果未全面优于传统方法。

[FFCC 原作者仓库](https://github.com/google/ffcc)发布了 GehlerShiThumb：568 张 32×48、8 位线性缩略图，以及逐图光源和原始 fold 索引。固定源版本为 `2fa9e1316954dbd3913630b7d597927941b4dd32`。

```powershell
git clone --depth 1 --filter=blob:none --sparse https://github.com/google/ffcc.git data/ffcc-source
git -C data/ffcc-source sparse-checkout set data/shi_gehler/preprocessed/GehlerShiThumb
git -C data/ffcc-source rev-parse HEAD
python scripts/prepare_ffcc_pilot.py --source-dir data/ffcc-source --output-dir data/ffcc-pilot
python -m awb_lab.learning train data/ffcc-pilot/train.csv data/ffcc-pilot/val.csv --output-dir outputs/ffcc-pilot --epochs 20 --input-size 32 --batch-size 16 --device cpu
python -m awb_lab.learning evaluate outputs/ffcc-pilot/best.pt data/ffcc-pilot/test.csv --output-dir outputs/ffcc-pilot-test --device cpu
```

检查版本必须与上面的固定 SHA 一致，适配器会拒绝其他 checkout 版本。若仓库上游发生更新，可显式取回该 SHA 后再运行。

作者已经减黑、遮罩、下采样、逐图归一并量化；适配时只用 `black_level=0, white_level=255`，不再减 129。另生成有效掩膜，排除作者已置零区域与任一通道为 255 的像素。

此初步实验固定 **fold 1 训练 189 张、fold 2 验证 191 张、fold 3 测试 188 张**。`group_id` 只是 `official-fold-X`，`grouping_unit` 明确为 `official_fold_no_scene_annotations`，不是编造的场景标注。

它不是论文完整三折复现，不能证明场景独立或跨相机泛化，也不能说明完整 RAW 分辨率上的效果。大规模训练时，应使用原始数据和可靠场景分组，另行做相机与场景独立的实验。原始数据和模型权重不随源码自动上传。

来源：[作者预处理脚本](https://github.com/google/ffcc/blob/2fa9e1316954dbd3913630b7d597927941b4dd32/scripts/GehlerShiImportData.m)、[原始 Shi–Gehler 数据](https://www.cs.sfu.ca/~colour/data/shi_gehler/)。请引用原数据与 FFCC；代码许可证不替代原始数据的使用条款。

## 合成流程自检

```powershell
python -m awb_lab.learning demo --output-dir outputs/ai-smoke
python -m awb_lab.learning train outputs/ai-smoke/train.csv outputs/ai-smoke/val.csv --output-dir outputs/ai-smoke-model --epochs 2 --input-size 32 --device cpu --allow-synthetic
python -m awb_lab.learning evaluate outputs/ai-smoke-model/best.pt outputs/ai-smoke/test.csv --output-dir outputs/ai-smoke-test --device cpu --allow-synthetic
python -m unittest discover -s tests -v
```

此流程只检查梯度更新、划分检查、模型保存加载、预测与评测，不作为真实 AI AWB 精度证据。
