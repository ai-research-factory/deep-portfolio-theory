# Preflight Check — Cycle 4 (Phase 4: Deep Portfolio戦略の実装)

## 1. データ境界表

| 項目 | 値 |
|---|---|
| データ取得終了日 | 2023-12-31 (今日 2026-03-30 以前) |
| Train期間 | 2010-01-05 〜 各ウィンドウの学習終了日 (60ヶ月ローリング) |
| Validation期間 | N/A (Walk-forwardではOOS期間がvalidation) |
| Test期間 | 各ウィンドウの学習終了翌月1ヶ月 (OOS) |
| 重複なし確認 | Yes — WalkForwardValidatorが時間的厳密分離を保証 |
| 未来日付なし確認 | Yes — データ最終日は2023-12-29 |

Walk-forward scheme:
- 学習期間: 60ヶ月 (5年)
- テスト期間: 1ヶ月
- リバランス頻度: 月次
- データ: 日次リターン (2010-01-05 〜 2023-12-29, 82銘柄)
- Autoencoder: 各ウィンドウで再学習 (train期間のデータのみ使用)

## 2. Feature timestamp 契約

- 全ての特徴量は時刻 t の予測に t-1 以前のデータのみを使用しているか？ → **Yes** (pct_changeは後方参照、AEはtrain期間のみで学習)
- Scaler / Imputer は train データのみで fit しているか？ → **Yes** (AEの学習は各ウィンドウのtrain期間のみ)
- Centered rolling window を使用していないか？ → **Yes** (使用していない)

## 3. Paper spec 差分表

| パラメータ | 論文の値 | 現在の実装 | 一致? |
|---|---|---|---|
| ユニバース | S&P 500 (~500銘柄) | S&P 100 (82銘柄, ARF API制約) | No |
| ルックバック期間 | 60ヶ月 (5年) | 60ヶ月 (5年) | Yes |
| リバランス頻度 | 月次 | 月次 | Yes |
| モデル | Autoencoder (入力→潜在空間→復元) | Autoencoder (n_assets→128→32→128→n_assets) | Yes |
| ポートフォリオ構築 | デコーダ重みから因子ポートフォリオ | デコーダ最終層重みから長期のみポートフォリオ | Yes |
| コストモデル | 記載なし | 10bps fee + 5bps slippage | N/A |
| ベンチマーク | 1/N, 最小分散 | 1/N, 最小分散 | Yes |

### 差分についての注記
- ユニバースサイズ: ARF Data APIの制約により82銘柄。docs/open_questions.mdに記録済み。
- ポートフォリオ構築: 論文ではデコーダ重みの列を因子ポートフォリオとし、それらの組合せでポートフォリオを構築。本実装ではデコーダ最終層の重みを正規化して長期のみポートフォリオを構成。
