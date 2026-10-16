# AWB Lab 评测约定

本仓库先建立可审查、可复现的经典算法比较流程。固定参数的算法对比不声称训练了模型，也不声称超过任何论文结果。

## 输入和真值

正式评测输入必须是**线性、未做白平衡的 camera-RGB**，保持原始数字计数和位深。普通 JPEG / sRGB 照片仅用于视觉演示；从 sRGB 反解传递函数不能恢复传感器 RAW 或相机完整处理链。

CSV 清单使用以下字段（完整可运行示例由合成样例生成工具输出）：

```csv
image,illuminant_r,illuminant_g,illuminant_b,black_level,white_level,mask
images/sample.npy,0.8,1.0,0.6,0,1,masks/sample.png
```

- 路径相对清单文件所在目录解析。
- `black_level` 和 `white_level` 是图像原始计数单位中的标量，不是归一化参数。12-bit 数据装在 uint16 文件中时，不能将 65535 当作真实白电平。
- 两个电平都必须显式填写，已经扣过黑电平的图像填写 `black_level=0`，不允许缺失或留空。
- 真值光源 `illuminant_*` 必须已经与扣黑电平后的图像配对，比例任意；**加载器不再给真值扣黑电平**。
- `mask` 可留空；提供时是单通道图，**非零代表可用，零代表排除**，尺寸与输入图像一致。其他来源的掩膜如果约定相反，必须先转换。
- 基本范围为 `(pixel - black_level) / (white_level - black_level)`。默认排除任一归一化通道达到 1 的饱和像素；`--near-clip` 可设为 `(0,1]` 内更低的阈值，不能将饱和像素继续计入光源估计。
- 色卡用于获得真值，必须从算法输入的统计区域中排除。掩膜应在完整图像尺寸上制作，不能只屏蔽几个灰色块而保留整张色卡。

## 数据集版本

推荐从 [SFU 的 Shi–Gehler 数据页面](https://www.cs.sfu.ca/~colour/data/shi_gehler/)获取图像和对应光源真值，并记录下载来源、版本、日期和所有预处理。

该页面提供线性 PNG；原始 Gehler TIFF 已包含相机处理，不能混入这个评测。页面说明其 Canon 5D PNG 黑电平为 129，Canon 1D 为 0。这些数字只适用于相应原始发行文件；若使用第三方已处理版本，应确认是否已经扣除，避免重复处理。

ColorChecker 存在多个真值版本与处理协议，不能把某版图像与另一版真值直接配对。参考 [Rehabilitating the ColorChecker Dataset for Illuminant Estimation](https://arxiv.org/abs/1805.12262)。本仓库不分发第三方数据；只提交清单、来源说明和获得授权的演示素材。

## 算法和参数

- Gray World：对有效像素的逐通道均值进行光源估计。
- White Patch / Max-RGB：逐通道取最大值；三个最大值可以来自不同像素位置。
- Shades of Gray：对逐通道强度使用 Minkowski 范数；默认 `p=6`。
- Gray Edge：统计高斯导数响应；默认 `p=6, sigma=1`。掩膜边界及其滤波影响区域不进入导数统计。

算法参考 [Edge-Based Color Constancy](https://staff.science.uva.nl/th.gevers/pub/GeversTIP07.pdf) 和 [研究者提供的代码索引](https://www.colorconstancy.org/source-code/index.html)。复现实现及其离散滤波选择应与原作者实现区分。

所有方法必须在同一清单、同一输入处理和同一真值下重跑。`p` 和 `sigma` 等参数在测试前固定；若搜索参数，只用训练/验证部分选择，保留测试部分最后报告。不同分辨率会改变 Gray Edge 中 sigma 的物理含义，应记录输入尺寸。

## 指标和报告

Recovery angular error（degree）比较预测光源和真值光源的方向：

```text
acos(clip(dot(pred, gt) / (norm(pred) * norm(gt)), -1, 1)) * 180 / pi
```

Reproduction angular error 对逐通道 `gt / pred` 与中性方向 `(1,1,1)` 计算夹角。两种指标分开命名、分开报告。

输出至少包含逐图预测光源、两种角误差和估计耗时，以及各算法的 mean / median / worst-25% mean。当前版本采用立即终止策略：遇到任一无效样本就终止整次评测、报告出错行，并且不写本次部分结果；不能静默删除失败样本后只汇报成功样本。成功报告的 `failed_count` 为 0。重新运行时使用新输出目录，避免把旧报告当作本次失败后的产物。不同机器、输入尺寸、线程配置的运行时间不直接横向比较；记录测试设备和软件版本。

合成样例验证接口、数值和处理流程，**不等同于真实图像数据集上的精度结论**。真实结果应另存目录，并在报告中注明数据清单、参数、版本及失败计数。
