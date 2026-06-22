# ICC 7CLR 验证设计

## 背景

当前 `AAA` 项目用一个已经保存好的随机森林模型来预测 7 色分色配方。现有模型的输入特征是 `L`、`a`、`b`，以及由 `a/b` 计算得到的 `C` 和 `h`；输出是 7 个油墨通道：

- Cyan
- Magenta
- Yellow
- Black
- Orange
- Green
- Violet

项目里还包含 CGS/XGamut 生成的 7 色打印设备 ICC，以及这份 ICC 对应的测量图表文件：

- `AAA/CMYKOGV_i1 Pro3 iO_XGAMUNT_Real.icc`
- `AAA/CMYKOGV_i1 Pro3 iO_XGAMUNT.txt`
- `AAA/CMYKOGV_i1 Pro3 iO_XGAMUNT.pdf`

ICC 头部信息显示：

- profile class：打印机 profile
- device color space：`7CLR`
- PCS：`Lab`
- creator：`CGS`

txt 图表里有 3024 行 7 色输入配方。这些配方是测量图表的设备输入值，不是独立的模型准确性真值。因此第一阶段应该用它们验证 ICC 转换链路和模型调用链路，而不是用它们直接证明模型精度。

## 目标

做一个小型 Python 验证脚本，先跑通下面这条链路：

```text
txt 中的 7CLR 配方 -> 设备 ICC -> Lab -> 现有随机森林模型 -> 预测 7CLR 配方
```

第一版先抽样 20-50 行，并输出 CSV 方便人工检查。

## 非目标

这一阶段不做：

- 网页界面
- 文件上传
- 任意客户 PDF 解析
- `DeviceN` PDF 图表解析
- 多个 ICC 的下拉选择
- 最终色彩精度结论
- 用 `txt 7CLR` 和预测 `7CLR` 的差值作为通过/失败标准
- 重新训练模型
- CMYK 输入 PDF

PDF 的 `DeviceN` 解析保留到后续阶段。

## 输入文件

验证脚本使用固定的本地文件：

- `AAA/CMYKOGV_i1 Pro3 iO_XGAMUNT.txt`
- `AAA/CMYKOGV_i1 Pro3 iO_XGAMUNT_Real.icc`
- `AAA/xgamut_model.pkl`

txt 字段和模型通道的映射关系如下：

| TXT 字段 | 模型通道 |
| --- | --- |
| `7CLR_1` | `Cyan` |
| `7CLR_2` | `Magenta` |
| `7CLR_3` | `Yellow` |
| `7CLR_4` | `Black` |
| `7CLR_5` | `Orange` |
| `7CLR_6` | `Green` |
| `7CLR_7` | `Violet` |

7CLR 数值是 `0-100` 的百分比。主流程不要把它们当成 `0-1` 处理。

## 架构设计

实现时建议拆成几个小函数，避免把 txt 解析、ICC 转换、模型预测混在一起。

### `parse_7clr_txt(path)`

职责：

- 读取 CGS/XGamut 的 txt 文件
- 定位 `BEGIN_DATA_FORMAT`、`END_DATA_FORMAT`、`BEGIN_DATA`、`END_DATA`
- 解析字段名和数据行
- 返回带有 `SampleID`、`SAMPLE_NAME` 和 7 个命名油墨通道的结构化数据
- 检查 7 个通道是否齐全
- 把通道值转换成数字百分比

需要验证：

- 当前图表应解析出 3024 行
- 如果出现超出 `0-100` 的值，要报告
- 如果有格式错误行，要尽量报告行号和样本名

### `convert_7clr_to_lab(rows, icc_path)`

职责：

- 加载 7CLR 打印设备 ICC
- 把每行的 7 个设备通道转换到 PCS Lab
- 给每行追加 `Lab_L`、`Lab_a`、`Lab_b`

这一层必须使用支持多通道打印 ICC 的色彩管理路径。只支持 RGB 或 CMYK 的转换 API 不够，因为这个 ICC 的设备色彩空间是 `7CLR`。

脚本对外保留 `0-100` 百分比值。如果所选 CMM API 内部要求归一化，归一化只能封装在这个函数内部，并在代码注释里说明。

需要验证：

- ICC 能正常打开
- ICC 设备空间兼容 7 个通道
- 输出 Lab 是数值
- 转换失败时要带上 sample ID 和输入通道值

### `load_xgamut_model(model_path)`

职责：

- 加载 `xgamut_model.pkl`
- 检查模型包里有保存的模型、特征列、目标列
- 检查特征列是否为 `L`、`a`、`b`、`C`、`h`
- 检查目标列是否匹配 7 个油墨通道

### `predict_7clr_from_lab(model_package, lab_rows)`

职责：

- 计算 `C = sqrt(a^2 + b^2)`
- 计算色相角 `h = atan2(b, a)`，单位为度，并归一化到 `0-360`
- 调用已加载的随机森林模型
- 把预测出的通道值裁剪到 `0-100`
- 追加 `Pred_Cyan`、`Pred_Magenta`、`Pred_Yellow`、`Pred_Black`、`Pred_Orange`、`Pred_Green`、`Pred_Violet`

这一阶段的模型预测结果只作为观察数据，不作为模型精度结论。

### `run_sample_validation(sample_size=50)`

职责：

- 解析全部 txt 行
- 默认取前 `sample_size` 行
- 后续可以增加固定随机种子的随机抽样
- 运行 ICC 转换
- 运行模型预测
- 写出 CSV

## 输出

第一版输出文件建议为：

```text
AAA/outputs/icc_sample_validation.csv
```

CSV 包含以下列：

- `SampleID`
- `SAMPLE_NAME`
- `Cyan`
- `Magenta`
- `Yellow`
- `Black`
- `Orange`
- `Green`
- `Violet`
- `Lab_L`
- `Lab_a`
- `Lab_b`
- `Pred_Cyan`
- `Pred_Magenta`
- `Pred_Yellow`
- `Pred_Black`
- `Pred_Orange`
- `Pred_Green`
- `Pred_Violet`

第一版不需要错误列。如果某一行无法转换，脚本应该直接报错，并给出足够上下文，而不是静默写出不完整数据。

## 验收标准

第一阶段成功的标准是：

1. txt 图表能解析成 3024 行结构化 7CLR 数据。
2. 20-50 行抽样数据能通过 7CLR ICC 转成 Lab。
3. 现有保存模型能从这些 Lab 预测 7CLR。
4. 能写出 `AAA/outputs/icc_sample_validation.csv`，且列名符合预期。
5. 脚本输出要明确说明：预测值和 txt 配方的差异只是观察信息，不是通过/失败指标。

## 错误处理

脚本遇到以下情况要给出清晰错误：

- txt、ICC 或模型文件缺失
- txt 数据格式无法识别
- 缺少 7CLR 字段
- 通道值超出 `0-100`
- ICC profile 加载失败
- ICC 转换失败
- 模型包缺少预期键
- 模型特征列或目标列不匹配

## 测试策略

初始检查：

- 解析 txt 文件并确认行数为 3024
- 确认第 1 行映射为 `SampleID=1`、`SAMPLE_NAME=A1`，且 7 个通道值符合 txt
- 先对 1 行做 ICC 转换，再对 20-50 行做转换
- 对转换得到的 Lab 运行模型预测
- 检查 CSV 是否存在，列名是否符合预期

这些检查保持轻量，目标是先验证链路，而不是一次性做完整应用。

## 后续阶段

抽样链路跑通后，可以继续：

1. 跑完整 3024 行并输出完整 CSV。
2. 增加预测通道分布、Lab 范围、转换失败情况等统计。
3. 解析 PDF 图表里的 `DeviceN` 7 通道填充值，并和 txt 行对齐，用于验证 PDF 提取能力。
4. 增加多个内置设备 ICC。
5. 等有实测 Lab 数据后，用实测数据做真正的精度评估。
6. 再回到目标色工作流，例如 `Lab/RGB PDF -> 设备 ICC 条件下的分色`。

## 关键假设

- 当前 ICC 应作为 7 通道打印设备 profile 使用，PCS 是 Lab。
- txt 图表值是 `0-100` 范围的 7CLR 百分比输入。
- 第一阶段继续使用现有随机森林模型。
- 这个设计是验证步骤，不是最终生产级分色流程。
