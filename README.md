# ISP-RAW2RGB

RAW 影像轉 RGB 影像處理管線，使用基於 Restormer 的 Transformer 架構。

## 📋 專案概述

本專案為深度學習課程作業（HW3），目標是將相機 RAW 格式影像轉換為高品質 RGB 影像。使用 3-Level Restormer 架構，針對 NVIDIA RTX 3070 (8GB VRAM) 優化。

## 🚀 快速開始

### 1. 環境設置

```bash
# 安裝依賴
pip install -r requirements.txt

# 下載資料集
python setup_data.py
```

### 2. 訓練模型

```bash
# 主訓練（200 epochs）
python scripts/train.py

# 微調 - Delta E 優化
python scripts/fine_tune.py --mode color

# 微調 - L1/PSNR 優化
python scripts/fine_tune.py --mode l1
```

### 3. 生成提交檔案

```bash
# 單模型預測
python scripts/predict.py --model experiments/ISP_Restormer_v4_Final/best_model.pt

# 雙模型 Ensemble
python scripts/test.py \
    --model-a experiments/ISP_Restormer_v4_Finetune/best_model_finetuned.pt \
    --model-b experiments/ISP_Restormer_v4_L1_Regression/best_model_l1.pt \
    --output submission_ensemble.csv
```

## 📁 專案結構

```
isp-raw2rgb/
├── src/                    # 核心模組
│   ├── config.py           # 統一設定
│   ├── models/             # 模型架構
│   │   ├── blocks.py       # MDTA, GDFN, TransformerBlock
│   │   └── ispnet.py       # ISPNet 主模型
│   ├── data/               # 資料處理
│   │   └── dataset.py      # RawImageDataset
│   ├── losses/             # 損失函式
│   │   └── composite.py    # L1, SSIM, Delta E
│   └── utils/              # 工具函式
│       ├── inference.py    # 分塊推論
│       ├── metrics.py      # 評估指標
│       └── logging.py      # 日誌設定
├── scripts/                # 執行腳本
│   ├── train.py            # 主訓練
│   ├── fine_tune.py        # 微調
│   ├── predict.py          # 預測
│   └── test.py             # Ensemble
├── experiments/            # 訓練結果
├── dataset/                # 資料集
├── outputs/                # 輸出檔案
│   └── submissions/        # CSV 提交檔
├── setup_data.py           # 資料下載
├── requirements.txt
└── README.md
```

## 🔧 模型架構

- **ISPNet**: 3-Level Restormer (Encoder-Decoder + Skip Connections)
- **Base Dim**: 48
- **Encoder Blocks**: [4, 4, 6]
- **Bottleneck Blocks**: 4
- **Attention Heads**: [1, 2, 4, 8]

## 📊 訓練策略

| 階段 | Epochs | LR | 重點 |
|-----|--------|------|------|
| 主訓練 | 200 | 3e-4 | L1 + SSIM + Delta E |
| 微調 (Color) | 50 | 1e-5 | 高 Delta E 權重 |
| 微調 (L1) | 20 | 5e-6 | 高 L1 權重 (PSNR) |

## 📝 License

MIT License
