# Technical Findings — Cycle 4 (Phase 4: Deep Portfolio戦略の実装)

## 実装内容

AutoencoderをWalk-Forwardフレームワークに統合し、**月次リターン**で学習するDeep Portfolio戦略のバックテストを実行した。Cycle 3のレビューフィードバックに対応し、以下を改善:

### レビューフィードバック対応
1. **日次→月次リターン**: 論文準拠で月次リターンを学習データとして使用
2. **デコーダ重み抽出**: decoder[0]（第1層）の重みを使用し、W2@W1で因子ポートフォリオを算出
3. **MinVar重み上限**: 個別銘柄5%上限の制約を追加（集中問題の緩和）
4. **データ品質修正**: bfillによるpre-IPO期間のphantom returns問題を修正（0埋めに変更）
5. **Paper spec**: `docs/paper_spec.md`に論文パラメータを明文化
6. **ユニバース**: ARF APIの制約で82銘柄のまま（500は不可能 — `docs/open_questions.md`に記載）

### DeepPortfolioStrategy (`src/strategies/deep_portfolio.py`)
- 各リバランスウィンドウで、月次リターン（60ヶ月）を使用してAEを再学習
- デコーダの因子ポートフォリオを算出: factor_portfolios = W2 @ W1 (shape: n_assets × latent_dim)
- 各因子ポートフォリオを均等加重で合成し、abs()→正規化で長期のみウェイトに変換
- モデル構成: 82→128→32→128→82

### Walk-Forward設定
- 学習期間: 60ヶ月 (5年) **月次リターン**
- テスト期間: 1ヶ月 (日次リターンで評価)
- リバランス頻度: 月次
- ウィンドウ数: 106
- OOS日数: 2,225日 (2015-03 〜 2023-12)

## 結果比較 (metrics.json参照)

| 指標 | Deep Portfolio | Equal Weight (1/N) | Min Variance |
|---|---|---|---|
| Sharpe Ratio | 0.9620 | 1.0077 | 1.0418 |
| Annual Return | 20.91% | 19.94% | 14.32% |
| Max Drawdown | -36.46% | -34.07% | -22.72% |
| Volatility | 21.74% | 19.78% | 13.75% |
| Hit Rate | 55.19% | 55.46% | 54.88% |
| Net Sharpe (10+5bps) | 0.8792 | 0.9167 | 0.9109 |
| Monthly Turnover | 20.42% | 0.00% | 8.39% |
| Positive Windows | 72/106 | 74/106 | 68/106 |

### 1/Nとの比較
- **Sharpe差**: -0.0457 (Deep Portfolioが劣後)
- **リターン差**: +0.97pp (Deep Portfolioが優位)
- **ドローダウン差**: -2.39pp (Deep Portfolioが劣後)
- **ターンオーバー**: 20.42% vs 0.00% (コスト増)

## 敗北分析 (vs 1/N)

Deep Portfolioは **Sharpe ratio** と **Max Drawdown** で1/Nに劣後:
- **Sharpe**: 0.962 < 1.008 (劣後 — ボラティリティの増大が原因)
- **Drawdown**: -36.46% < -34.07% (劣後)
- **Return**: 20.91% > 19.94% (優位)
- **Turnover**: 20.42% > 0.00% (コスト増 — Net Sharpeでさらに差が拡大)
- **Cost sensitivity**: 取引コスト15bpsで年間約3.1%のコスト負担（月次ターンオーバー20.42%に起因）

### Cycle 3からの改善
| 指標 | Cycle 3 (日次, bfillあり) | Cycle 4 (月次, bfill修正) | 変化 |
|---|---|---|---|
| DP Sharpe | 2.44 | 0.96 | 低下（データ修正による正常化） |
| 1/N Sharpe | 2.56 | 1.01 | 低下（同上） |
| MinVar Sharpe | 15.49 | 1.04 | 大幅改善（重み上限+データ修正の効果） |
| DP vs 1/N差 | -0.11 | -0.05 | 改善（差が縮小） |

Cycle 3の高Sharpeは **bfillによるデータ品質問題** が原因だったことが判明。pre-IPO銘柄（CRWD, PATH, SNOW, CEG, ZS等）のリターンが最初の実際のリターン値（例: CRWD +16.48%/日）でIPO前の全日付に伝播していた。修正後は現実的な水準に。

### 考察
1. **82銘柄の制約**: 論文のS&P 500 (500銘柄)に対し、82銘柄ではAEが学習できる市場構造が限定的
2. **n < p 問題**: 60ヶ月のサンプルで82資産のAEを学習するのは統計的に困難（サンプル数 < 特徴量数）
3. **均等加重合成**: 論文の基本アプローチだが、因子ポートフォリオのランキングや選択が有効かもしれない
4. **高ターンオーバー**: 20.42%/月は取引コストに敏感。AEの学習安定性向上が課題

## 今後の改善候補
- Phase 6でDenoising Autoencoder (DAE) を実装し、ロバスト性を向上
- 因子ポートフォリオの選択方法の改善（分散説明力によるランキング等）
- AEへのドロップアウトやL2正則化の追加（n < p問題の緩和）
- ターンオーバー削減のための重みスムージング

## データ品質修正の詳細

### bfill問題（重要）
Cycle 3までの実装で、pre-IPO銘柄のリターンが `ffill().bfill()` により異常値に汚染されていた:
- CRWDの2015-2019年: 毎日+16.48%（年率4,100%相当のphantomリターン）
- PATH: 毎日+9.42%のphantomリターン
- CEG: 毎日+7.14%のphantomリターン

**修正**: `fillna(0.0)` に変更。IPO前の期間はリターン0（取引なし）として扱う。これによりCycle 3の全戦略のSharpe ratioが正常化した。
