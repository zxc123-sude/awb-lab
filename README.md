# AWB Lab

**经典自动白平衡算法 · 线性 RGB 评测 · 可复现结果**

面向相机 AWB / ISP 学习和算法实验的小型工具。实现经典光源估计方法，明确图像与真值的处理约定，并将视觉演示与真实数据集评测分开。

[项目演示页面](https://zxc123-sude.github.io/) · [评测协议](docs/EVALUATION.md)

## 当前范围

| 方法 | 统计假设 | 典型局限 |
| --- | --- | --- |
| Gray World | 场景平均反射接近中性 | 大面积单色会影响通道均值 |
| White Patch / Max-RGB | 每个通道的最大响应提供光源信息 | 易受高亮、噪声与饱和影响 |
| Shades of Gray | 使用 Minkowski 范数连接均值与最大值 | 参数和高亮像素会影响估计 |
| Gray Edge | 场景通道导数统计接近中性 | 依赖纹理、尺度、滤波和有效区域 |

这是经典方法的 Python 复现与评测初版。C++ 实现、性能优化和真实数据集实验属于后续工作。合成样例仅验证代码与处理流程，目前不声称真实数据集精度或性能提升。

## 安装

需要 Python 3.9 或以上。建议在独立环境中安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

以下命令假设已激活该环境；也可以用 `.\.venv\Scripts\python.exe` 替换 `python`。

## 先跑通合成样例

```powershell
python -m awb_lab demo --output-dir outputs/synthetic
python -m awb_lab evaluate outputs/synthetic/manifest.csv --output-dir outputs/synthetic-results
```

样例中的场景与光源由程序生成，不代表真实相机、真实数据集精度或量产 ISP 表现。报告中保留这一标记，不将其当作论文 benchmark。

## 校正一张图片

```powershell
python -m awb_lab correct photo.jpg --output outputs/corrected.png --method gray_world --input-space srgb
```

普通 sRGB 图片可用于观察校正效果。程序会反解 sRGB 传递函数、估计光源、以绿色通道归一化增益，再为显示编码；这个过程不能恢复 RAW，也不等同于相机 ISP 标定。

正式线性数据需显式指定原始计数中的黑电平和白电平，例如：

```powershell
python -m awb_lab correct linear.png --output outputs/linear-corrected.png --method shades_of_gray --input-space linear --black-level 129 --white-level 4095
```

上述计数只是一个明确的格式示例，必须根据自己的图像来源设置，不可直接套用到任意 PNG。

## 在自己的数据上评测

准备 UTF-8 CSV，路径相对 CSV 文件所在目录：

```csv
image,illuminant_r,illuminant_g,illuminant_b,black_level,white_level,mask
images/example.npy,0.8,1.0,0.6,0,1,masks/example.png
```

`image` 是线性 RGB 文件，`mask` 非零代表可用、零代表排除；光源真值必须已与扣黑电平后的图像配对。不要把色卡区域用于算法估计，不要从文件容器位深猜真实白电平。

```powershell
python -m awb_lab evaluate data/manifest.csv --output-dir outputs/real-evaluation
```

默认排除任一归一化通道达到 1 的饱和像素；可用 `--near-clip` 在 `(0,1]` 内设置更低的排除阈值。输出逐图预测、recovery / reproduction angular error 和运行时间，以及各方法的汇总统计。遇到任何无效样本，整次评测立即终止并报告出错行，不静默删除样本；成功报告的 `failed_count` 为 0。请完整阅读[评测协议](docs/EVALUATION.md)，记录数据版本、掩膜、参数、软件版本和测试设备。

推荐数据入口：[SFU Shi–Gehler](https://www.cs.sfu.ca/~colour/data/shi_gehler/)。数据和真值按原始发布方的说明获得，不随本仓库分发。当前工具使用标量黑白电平；逐通道或逐像素校准需要单独扩展预处理。

## 验证

```powershell
python -m unittest discover -s tests -v
```

测试关注已知光源的可恢复场景、角误差的几何性质、退化输入、掩膜边界、图像位深及命令行处理；合成测试的通过不证明真实场景泛化。

## 后续实验

- 固定协议跑真实数据，报告误差分布和代表性失败案例。
- 对单色主导、暗光、饱和像素做受控分析，并验证像素筛选策略。
- 增加 C++ 实现，核对数值一致性后再比较耗时。
- 按相机分别分析结果；参数选择和最终测试分开。

## 参考

- Buchsbaum, *A spatial processor model for object colour perception*, 1980。
- Finlayson & Trezzi, *Shades of Gray and Colour Constancy*, 2004。
- van de Weijer, Gevers & Gijsenij, [*Edge-Based Color Constancy*](https://staff.science.uva.nl/th.gevers/pub/GeversTIP07.pdf), IEEE TIP, 2007。
- Shi & Funt, [*Re-processed Version of the Gehler Color Constancy Dataset of 568 Images*](https://www.cs.sfu.ca/~colour/data/shi_gehler/)。

本仓库的离散实现和默认参数以代码及评测协议为准，不能直接视为原作者代码的逐位复现。
