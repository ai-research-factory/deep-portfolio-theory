# Preflight Check — Cycle 1, Phase 1

## 1. データ境界表

| 項目 | 値 |
|---|---|
| データ取得終了日 | N/A (Phase 1 uses synthetic data for model validation only) |
| Train期間 | N/A |
| Validation期間 | N/A |
| Test期間 | N/A |
| 重複なし確認 | N/A |
| 未来日付なし確認 | N/A |

**Note**: Phase 1 focuses on implementing and validating the Autoencoder architecture using synthetic random tensors (`torch.rand`). No real market data is used, so data boundary checks are not applicable. Real data will be introduced in Phase 2.

## 2. Feature timestamp 契約

- 全ての特徴量は時刻 t の予測に t-1 以前のデータのみを使用しているか？ → N/A (no time-series features in Phase 1)
- Scaler / Imputer は train データのみで fit しているか？ → N/A (no scaling in Phase 1)
- Centered rolling window を使用していないか？ → N/A (no rolling windows in Phase 1)

## 3. Paper spec 差分表

| パラメータ | 論文の値 | 現在の実装 | 一致? |
|---|---|---|---|
| ユニバース | S&P 100 constituents | Synthetic 256-dim vectors | No (Phase 1 scaffold) |
| ルックバック期間 | ~5 years monthly returns | N/A | No (Phase 1 scaffold) |
| リバランス頻度 | Monthly | N/A | No (Phase 1 scaffold) |
| 特徴量 | Asset returns | Random synthetic data | No (Phase 1 scaffold) |
| コストモデル | Not specified in detail | Not implemented | N/A |
| AE Architecture | Encoder-Decoder with bottleneck | Input→128→32→128→Output | Approximate match |
| Loss Function | MSE (reconstruction error) | MSELoss | Yes |
| Optimizer | Not explicitly specified | Adam | Reasonable default |

**Note**: Phase 1 is a scaffold implementation. The architecture follows the paper's autoencoder concept (encoder-decoder with low-dimensional bottleneck). Full paper alignment will be achieved in subsequent phases.
