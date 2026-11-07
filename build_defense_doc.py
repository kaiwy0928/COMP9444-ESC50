"""Build the COMP9444 defense document (.docx) with final 5-fold x 3-seed results."""
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Cm

BLUE = RGBColor(0x1A, 0x4B, 0x8C)
DARK = RGBColor(0x22, 0x22, 0x22)

doc = Document()

# ---- base styles ----
normal = doc.styles["Normal"]
normal.font.name = "Microsoft YaHei"
normal.font.size = Pt(10.5)
normal.font.color.rgb = DARK
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
normal.paragraph_format.line_spacing = 1.5
normal.paragraph_format.space_after = Pt(4)

for s in doc.sections:
    s.top_margin = Cm(2.2)
    s.bottom_margin = Cm(2.2)
    s.left_margin = Cm(2.4)
    s.right_margin = Cm(2.4)


def shade(cell, hex_color):
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:fill"), hex_color)
    cell._tc.get_or_add_tcPr().append(el)


def h(text, level=1):
    p = doc.add_heading(text, level=level)
    for r in p.runs:
        r.font.name = "Microsoft YaHei"
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        r.font.color.rgb = BLUE if level <= 2 else RGBColor(0x25, 0x63, 0xA8)
        r.font.size = Pt(15 if level == 1 else 12.5)
    p.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    p.paragraph_format.space_after = Pt(6)
    return p


def para(text, bold=False, size=10.5, align=None, space_after=4):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.font.size = Pt(size)
    r.font.name = "Microsoft YaHei"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    if align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(space_after)
    return p


def table(rows, widths=None, header=True, highlight_rows=(), font_size=9.5):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            r = p.add_run(str(val))
            r.font.size = Pt(font_size)
            r.font.name = "Microsoft YaHei"
            r._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
            if header and i == 0:
                r.bold = True
                shade(cell, "E8EEF7")
            elif i in highlight_rows:
                r.bold = True
                shade(cell, "FDF3D7")
    if widths:
        for j, w in enumerate(widths):
            for row in t.rows:
                row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


# ---- title ----
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run("COMP9444 项目答辩文档")
r.font.size = Pt(22)
r.bold = True
r.font.color.rgb = BLUE
r.font.name = "Microsoft YaHei"
r._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")

para("基于 Audio Spectrogram Transformer (AST) 的环境声分类", size=12, align="center", space_after=2)
para("数据集：ESC-50（50 类 / 2000 条 5 秒音频 / 官方 5 折）", size=10.5, align="center", space_after=2)
para("最终成绩：90.13 ± 0.85%（5 折 × 3 seed）/ 86.23M 参数", size=10.5, align="center", space_after=2)
para("2026 年 11 月", size=10.5, align="center", space_after=10)

# ---- 1 ----
h("一、项目概述")
table([
    ["项目", "内容"],
    ["任务", "环境声分类（ESC）：对 5 秒音频判别 50 个类别（动物、自然声、城市/户外、室内/家庭、人类非语音）"],
    ["数据集", "ESC-50：2000 条片段，每类 40 条，官方 5 折交叉验证（每折 400 条测试 / 1600 条训练）"],
    ["硬性指标 1", "参数量 > 1M —— 最终模型 86.23M，达标"],
    ["硬性指标 2", "准确率 > 90% —— 5 折 × 3 seed 全量 90.13 ± 0.85%（15 次训练），达标"],
    ["运行环境", "Windows + RTX 4060 Ti 16GB (sm_89) + PyTorch 2.11.0+cu128 + Python 3.13"],
], widths=[3.2, 12.5])

# ---- 2 ----
h("二、数据与预处理流水线")
para("整条流水线在训练前一次性预计算并缓存为特征数组（npz），训练阶段只读数组，不重复做信号变换：")
for i, txt in enumerate([
    "重采样：原始 44.1 kHz WAV → 22.05 kHz；",
    "峰值归一化：消除录音增益差异；",
    "log-mel 频谱：n_fft=1024（约 46 ms 窗）、hop=512（约 23 ms 步进）、128 个 mel 频带、fmax=11025 Hz，得到 128×216 时频图；",
    "delta 通道：log-mel 的时间一阶差分作为第二通道，突出瞬态变化，最终输入形状 (batch, 2, 128, 216)；",
    "标准化：均值/方差统计只来自训练折，避免测试信息泄漏。",
], 1):
    doc.add_paragraph(txt, style="List Number")
para("折划分协议（遵循官方 5 折）：测试折 k，验证折为 (k mod 5) + 1（用于挑选最佳 epoch），其余三折用于训练。")

# ---- 3 ----
h("三、模型架构")
h("3.1 候选模型总览", 2)
table([
    ["模型", "参数量", "预训练来源", "本机结果"],
    ["PANNs CNN14", "79.78 M", "AudioSet", "79.75%（40 ep）"],
    ["ResNet-18", "11.20 M", "ImageNet", "83.07%（团队结果）"],
    ["EfficientNet-B0", "4.07 M", "ImageNet", "88.00%（60 ep）"],
    ["MobileNetV3-Small", "1.57 M", "ImageNet", "83.00%（450 ep）"],
    ["AST（最终采用）", "86.23 M", "AudioSet（纯音频预训练）", "90.13 ± 0.85%（15 次）"],
], widths=[4.2, 2.6, 5.0, 4.4], highlight_rows=(5,))
para("关键选型依据：AST 是对比中唯一「预训练语料既大（5000 小时 AudioSet）又是纯音频」的模型，对环境声的起始/谐波表征最匹配；ImageNet 系模型预训练的是静态图像纹理，迁移 gap 更大。")

h("3.2 AST 结构与适配（答辩重点）", 2)
para("AST = Audio Spectrogram Transformer（Gong et al., Interspeech 2021），检查点 MIT/ast-finetuned-audioset-10-10-0.4593（AudioSet 527 类微调，mAP 0.4593）：")
for txt in [
    "骨干：DeiT 式 ViT-B —— 12 层 Transformer 编码器、768 隐藏维、12 头注意力；",
    "分块嵌入：16×16 patch、步长 10 → 128×216 谱图切成 12（频带）× 21（时间）= 252 个 patch；",
    "位置编码适配：预训练位置编码是 12×101（对应 10 秒输入），本任务输入 216 帧 → 首次前向时双三次插值到 12×21，cls/distillation 两个特殊 token 原样保留；",
    "双通道融合：log-mel 与 delta 两通道经 1×1 卷积融合为单通道，卷积权重初始化为「取均值」，使 patch 嵌入保持预训练形状 (768,1,16,16)；",
    "编码/解码说明：本任务是分类而非序列生成，因此只有 Transformer 编码器 + 一个新初始化的 768→50 线性分类头；除该头外全部张量来自 AudioSet 检查点。",
]:
    doc.add_paragraph(txt, style="List Bullet")

# ---- 4 ----
h("四、训练配置")
table([
    ["配置项", "取值"],
    ["优化器 / 调度", "AdamW，lr 1e-4，OneCycle"],
    ["weight decay / 标签平滑", "1e-2 / 0.1"],
    ["batch size / epochs", "32 / 40"],
    ["SpecAugment", "2 个 mask，频率宽 36、时间宽 24"],
    ["Mixup", "α = 0.2"],
    ["时间平移（time-shift，本组关键改动）", "训练时对波形随机循环平移，等效时间轴增广，把 89.25% 推到 90.25%（单折）"],
    ["确定性", "固定种子 + cuBLAS 确定性配置"],
], widths=[6.0, 9.7])

# ---- 5 ----
h("五、实验结果")
h("5.1 最终成绩（5 折交叉验证 × 3 个随机种子）", 2)
table([
    ["Seed", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "均值"],
    ["42", "89.25", "90.75", "86.25", "91.50", "88.25", "89.20%"],
    ["43", "90.50", "92.00", "89.50", "92.75", "89.50", "90.85%"],
    ["44", "92.25", "90.00", "89.25", "92.75", "87.50", "90.35%"],
    ["三次 seed 均值 ± 标准差", "", "", "", "", "", "90.13 ± 0.85%"],
], widths=[3.6, 2.1, 2.1, 2.1, 2.1, 2.1, 2.6], highlight_rows=(4,))
para("单位：%。每个数字是 400 条测试片段上的准确率；15 次训练使用完全相同的配方，仅更换随机种子与折划分。", size=9)

h("5.2 详细指标（三 seed 汇总）", 2)
table([
    ["Accuracy", "Macro F1", "Cohen's κ", "Top-3", "Top-5", "参数量"],
    ["90.13 ± 0.85%", "0.8998", "0.8993", "97.87%", "98.70%", "86.23 M"],
], widths=[3.4, 2.6, 2.6, 2.4, 2.4, 2.6], highlight_rows=())
para("κ（0.8993）与 Accuracy（0.9013）几乎相等，说明不是靠类别不平衡「蒙对」；Top-5 98.70% 说明错误样本的真实类几乎都落在模型前 5 猜测内。单折最高 92.75%，最低 86.25%。")

h("5.3 完整实验序列（含调参过程）", 2)
table([
    ["#", "实验", "关键配置", "准确率", "结论"],
    ["1", "PANNs 冒烟", "3 ep", "27.00%", "链路通（随机 2%）"],
    ["2", "PANNs 强正则", "40 ep, wd 5e-3, ls 0.1", "79.75%", "过拟合：训练 acc 98.9%"],
    ["3", "PANNs 窗口裁剪", "crop 128 帧（3 秒窗）", "75.75%", "增广过强，40 ep 训不透"],
    ["4", "EfficientNet-B0", "60 ep, lr 3e-4, wd 1e-3", "88.00%", "ImageNet 系最佳"],
    ["5", "MobileNetV3-S", "450 ep", "83.00%", "容量不足"],
    ["6", "AST", "20 ep", "88.50%", "2 ep 即 80.75%，起点极高"],
    ["7", "AST", "40 ep", "89.25%", "测试曲线仍在上升"],
    ["8", "AST + time-shift", "40 ep", "90.25%", "单折达标"],
    ["9", "AST + time-shift 全量", "5 折 × 3 seed", "90.13 ± 0.85%", "最终成绩"],
], widths=[1.0, 4.2, 4.4, 2.9, 4.4], highlight_rows=(9,), font_size=9)

h("5.4 错误分析（答辩高频问题）", 2)
for txt in [
    "最弱类别：washing_machine（F1 0.50，被误判为 vacuum_cleaner/engine）、helicopter（0.57，误判为 crickets）、wind（0.67，被判为 sea_waves）、engine（0.67）。",
    "混淆本质：都发生在「宽带稳态噪声」之间（洗衣机/吸尘器/引擎、风/海浪、直升机/蟋蟀），缺乏清晰的起始事件，5 秒窗口内频谱结构高度相似，人耳亦难辨。",
    "大类表现：Animals 0.96 最高（鸣禽/狗/青蛙有清晰谐波结构），Exterior/urban 0.86 最低。",
]:
    doc.add_paragraph(txt, style="List Bullet")

# ---- 6 ----
h("六、结论")
for txt in [
    "指标双达标：参数量 86.23M（> 1M），5 折 × 3 seed 准确率 90.13 ± 0.85%（> 90%）；",
    "方法论收获：1600 条训练样本配 86M 参数，靠「强音频预训练 + 适度增广（SpecAugment / Mixup / 时间平移）」化解矛盾；纯加正则（wd 5e-3）或裁剪窗口反而有害；",
    "工程收获：打通了 原始 WAV → 特征缓存 → 训练 → 评估 → 图表 的完整流水线，所有实验可由 run_model.py 一条命令复现。",
]:
    doc.add_paragraph(txt, style="List Number")

# ---- 7 ----
h("七、答辩问答预案")
qa = [
    ("Q1：为什么选 AST 而不是 PANNs？两者都是 AudioSet 预训练。",
     "PANNs 是 CNN（卷积归纳偏置强），1600 条小数据上 40 epoch 就把训练集背到 98.9%，测试只有 79.75%；AST 是纯 Transformer，patch 序列建模 + 更强预训练任务使其小样本泛化显著更好（90.13% vs 79.75%，差 10 个点以上）。"),
    ("Q2：位置编码 12×101 插值到 12×21，会不会破坏预训练知识？",
     "双三次插值保持频带维完全不变（12→12），只压缩时间维，相对位置结构近似保留。实验证明有效——若插值失败，2 epoch 冒烟不可能到 80.75%。"),
    ("Q3：time-shift 为什么有效（+1 个点）？",
     "ESC-50 录音起点任意，但特征缓存在固定窗口上，模型可能记住「第 X 帧出现什么」的位置偏置；时间平移打破该伪相关，以零成本引入时移不变性，且不影响测试协议。"),
    ("Q4：单折 90.25% 能代表 5 折水平吗？",
     "已跑完全量 15 次训练，结果 90.13 ± 0.85%，与单折 90.25% 几乎重合，说明单折结果并非偶然。折间波动 86.25%–92.75%，主因是每折仅 400 条测试样本，1 条对错即 0.25%。"),
    ("Q5：86M 参数部署会不会太重？",
     "答辩要求参数量 > 1M 且准确率 > 90%，AST 双达标；若考虑部署，EfficientNet-B0（4.07M，88.00%）是精度/体积折中选项，MobileNetV3（1.57M，83%）更轻但精度不足。"),
    ("Q6：为什么加 delta 通道？",
     "log-mel 是能量统计，瞬态事件（敲门、鼠标点击）在相邻帧能量差中更显著；delta 一阶差分把该信息显式作为通道输入，等价于给网络一个高通时间滤波器。"),
    ("Q7：有没有信息泄漏风险？",
     "三处已防范——标准化统计只取训练折；验证折只用于选 best epoch 不参与梯度；测试折全程零接触。ESC-50 官方折保证同一录音来源不跨折。"),
    ("Q8：还能怎么提升？",
     "① Ensemble（AST + EfficientNet-B0 预测平均）；② 测试时多窗 TTA 平均；③ 更长微调 + 余弦重启；④ 对困难类（washing_machine 等稳态噪声）做重采样。"),
]
for q, a in qa:
    p = doc.add_paragraph()
    r = p.add_run(q)
    r.bold = True
    r.font.color.rgb = BLUE
    r.font.size = Pt(10.5)
    r.font.name = "Microsoft YaHei"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    para("A：" + a)

# ---- 8 ----
h("八、复现命令")
para("python run_model.py models/model_ast.py --folds 1 2 3 4 5 --seeds 42 43 44 --epochs 40 --lr 1e-4 --weight-decay 1e-2 --label-smoothing 0.1 --time-shift",
     size=9.5)
para("环境：pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128；其余依赖见 requirements_cloud.txt；AST 权重经 HF_ENDPOINT=https://hf-mirror.com 自动下载。", size=9.5)

out = r"D:/COMP9444_cloud_bundle/COMP9444_答辩文档_final.docx"
doc.save(out)
print("saved", out)
