# Preflight Check — Cycle 2 (Phase 2: Data Pipeline)

## 1. データ境界表

| 項目 | 値 |
|---|---|
| データ取得終了日 | 2023-12-31 (今日 2026-03-30 以前) |
| Train期間 | 2010-01-01 〜 2019-12-31 (walk-forward で分割予定、Phase 3) |
| Validation期間 | Phase 3 で定義予定 |
| Test期間 | 2020-01-01 〜 2023-12-31 (Phase 3 で定義予定) |
| 重複なし確認 | Yes — walk-forward split は Phase 3 で実装 |
| 未来日付なし確認 | Yes — end_date=2023-12-31 で明示的にフィルタ |

## 2. Feature timestamp 契約

- 全ての特徴量は時刻 t の予測に t-1 以前のデータのみを使用しているか？ → Yes（日次リターンは前日終値と当日終値から計算、pct_change で t-1→t）
- Scaler / Imputer は train データのみで fit しているか？ → N/A（Phase 2 では scaler 未使用、リターン計算のみ）
- Centered rolling window を使用していないか？ → Yes（rolling window 未使用）

## 3. Paper spec 差分表

| パラメータ | 論文の値 | 現在の実装 | 一致? |
|---|---|---|---|
| ユニバース | S&P 500 / TOPIX | S&P 100 相当（ARF API利用可能な米国株 ~84銘柄） | No — API制約により縮小。open_questions.md に記録 |
| ルックバック期間 | 5年（月次60データポイント） | 2010-2023（14年分の日次データを取得、walk-forward で5年窓を適用予定） | Yes（Phase 3 で実装） |
| リバランス頻度 | 月次 | Phase 3 で実装予定 | N/A |
| 特徴量 | 日次/月次リターン | 日次リターン（月次集約は Phase 3 で実施） | Yes |
| コストモデル | 論文では明示なし | 10bps fee + 5bps slippage（Phase 8 で実装） | N/A |

## 4. Cycle 2 計画サマリ

### レビューフィードバック対応
1. src/train.py をリファクタリングし、ハイパーパラメータを configs/default.yaml から読み込むように修正
2. tests/test_models.py を新規作成し、Autoencoder のユニットテストを実装
3. 本 preflight.md の作成

### Phase 2 タスク
1. src/data/loader.py — ARF Data API を使用した S&P 100 データ取得パイプライン
2. scripts/prepare_data.py — ローダー呼び出しスクリプト
3. data/processed/sp100_daily_returns.csv — 処理済みリターンデータ（git 管理外）
