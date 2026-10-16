# AI AWB 真实缩略图初步实验

本次验证了独立轻量 CNN 的真实图像训练、验证选模及独立测试清单评测。**它是单次固定划分的初步实验，不是完整论文 benchmark 复现。** 数据为作者已处理的缩略图；尚未完成完整 RAW、场景独立、跨相机或部署实验。

## 配置与协议

- 数据：[FFCC 作者 GehlerShiThumb](https://github.com/google/ffcc/tree/2fa9e1316954dbd3913630b7d597927941b4dd32/data/shi_gehler/preprocessed/GehlerShiThumb)，固定源 SHA `2fa9e1316954dbd3913630b7d597927941b4dd32`。
- 568 张 32×48、uint8 的 camera-linear RGB 缩略图。作者已经减黑、遮罩、下采样和逐图归一；本项目不重复减黑，并显式排除置零及饱和区域。
- 官方 fold 1 训练 189 张，fold 2 验证 191 张，fold 3 测试 188 张。没有物理场景标签；组标签与图像路径不交叉，不等于已经证明场景独立。两个相机混合使用，未验证未知相机泛化。
- `TinyIlluminantNet`：25,012 个参数、从随机初始化训练；RGB+有效覆盖率、3 个卷积块、局部光源与置信度汇聚。独立实现，非 FC4 原架构复现。
- CNN 输入 32×32；四种经典方法处理原始 32×48 缩略图，使用同一批测试样本及有效掩膜。比较的是各自的处理流程，并非相同输入尺寸下的受控网络消融。
- CPU、2 线程，seed 42，Adam，学习率 0.001，batch 16，20 epoch；没有使用测试集调参。
- 只按验证集 mean recovery angular error 选中第 **18** 轮，验证均值 **4.3602°**。测试清单在训练后单独传入，188 张全部完成，失败 0 张。

## 测试结果

Recovery angular error，单位 °，越低越好。最差 25% 指误差最大的 47 张图的均值。

| 方法 | Mean | Median | Worst 25% mean |
|---|---:|---:|---:|
| Tiny CNN | 4.0914 | 3.3442 | 8.0677 |
| Gray World | 4.8275 | 3.6976 | 10.4243 |
| White Patch | 4.0148 | 2.2830 | 10.1903 |
| Shades of Gray (p=6) | 3.9590 | 2.3984 | 9.8163 |
| Gray Edge (p=6, sigma=1) | 4.5113 | 2.7529 | 11.0977 |

本次 CNN 的均值优于 Gray World 和 Gray Edge，但未优于 White Patch、Shades of Gray；其中位数也未优于这些传统方法。CNN 的最差四分位均值较低，这只是这批缩略图的一次观测，尚未做多随机种子、完整三折或统计显著性分析，不宣称全面提升或达到论文最优结果。

[完整配置、训练曲线数值、来源和汇总结果](results.json)包括 reproduction angular error。[逐图误差](per_image_errors.csv)包含全部 188 张图及五种方法，便于核对汇总；不分发原始图片或 checkpoint。运行耗时只作单次诊断：无预热、输入尺寸及工作量不同，不作为速度排行榜。

## 复现

见[环境与数据准备指南](../../docs/AI_AWB.md)。固定 FFCC 源版本后：

```powershell
python scripts/prepare_ffcc_pilot.py --source-dir data/ffcc-source --output-dir data/ffcc-pilot
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
python -m awb_lab.learning train data/ffcc-pilot/train.csv data/ffcc-pilot/val.csv --output-dir outputs/ffcc-pilot --epochs 20 --input-size 32 --batch-size 16 --seed 42 --device cpu
python -m awb_lab.learning evaluate outputs/ffcc-pilot/best.pt data/ffcc-pilot/test.csv --output-dir outputs/ffcc-pilot-test --device cpu
```

本次运行环境为 Windows / Python 3.12.14 / PyTorch 2.14.1+cpu / NumPy 2.5.4 / SciPy 1.18.1 / OpenCV 5.0.0，36 项单元及流程测试通过。版本、硬件和数值内核不同可能导致训练轨迹差异。`results.json` 保存原始数据文件汇总 SHA256、清单 SHA256 及本地权重 SHA256；权重哈希用于识别此次运行，不代表重新训练会产生相同二进制文件。

下一阶段应在有可靠场景标注的完整数据上重新建立划分，进行多种子/多折评估、失败场景分析与相机泛化实验。请引用原始 Gehler / Shi–Funt 数据与 Barron、Tsai 的 FFCC 工作；代码许可证不替代数据的使用条款。
