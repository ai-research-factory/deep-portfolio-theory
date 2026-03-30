# Preflight Check — Cycle 5, Phase 5

## 1. Data Boundary Table

| Item | Value |
|---|---|
| Data acquisition end date | 2023-12-29 (before today 2026-03-30) |
| Train period | 2010-01-05 ~ rolling (60-month windows) |
| Validation period | N/A (walk-forward OOS evaluation) |
| Test period | 2015-03-02 ~ 2023-12-29 (OOS via walk-forward) |
| No overlap confirmed | Yes |
| No future dates confirmed | Yes |

## 2. Feature Timestamp Contract

- All features use data at t-1 or earlier for prediction at t? → Yes (returns are backward-looking pct_change)
- Scaler/Imputer fit on train data only? → Yes (no scaler used; autoencoder trains per window on train data only)
- No centered rolling windows? → Yes (no rolling windows used)

## 3. Paper Spec Difference Table

| Parameter | Paper value | Current implementation | Match? |
|---|---|---|---|
| Universe | S&P 500 (~500 assets) | S&P 100 subset (82 assets) | No — ARF API limitation |
| Lookback period | Not specified exactly | 60 months | Yes (reasonable) |
| Rebalance frequency | Monthly | Monthly | Yes |
| Features | Asset returns | Monthly returns | Yes |
| Cost model | Not specified | 10bps fee + 5bps slippage | Yes (standard assumption) |

## 4. Phase 5 Specific Notes

Phase 5 is an evaluation/reporting phase. It reads existing return data from cycles 3 and 4 and computes performance metrics. No new model training or data acquisition is performed, so data leakage risk is minimal. The key concern is ensuring metrics are computed correctly from the existing OOS return series.
