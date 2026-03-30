# Deep Portfolio Theory

## Project ID
proj_8ed17ca5

## Taxonomy
PCA, ResidualFactors

## Current Cycle
1

## Objective
Implement, validate, and iteratively improve the paper's approach with production-quality standards.


## Design Brief
### Problem
Traditional portfolio construction methods, such as Markowitz's Mean-Variance Optimization, are highly sensitive to estimation errors in expected returns and covariance matrices. This often leads to unstable and poorly performing portfolios out-of-sample. Factor models based on Principal Component Analysis (PCA) address this by reducing dimensionality but are limited to linear relationships.

This paper, 'Deep Portfolio Theory', proposes using autoencoders (AE) and denoising autoencoders (DAE) to learn a robust, non-linear factor model of asset returns. By training a neural network to reconstruct asset returns via a low-dimensional bottleneck, the model captures the underlying market structure more effectively than linear methods. The goal is to leverage these learned non-linear factors to construct portfolios that exhibit superior out-of-sample performance, particularly in terms of risk-adjusted returns and robustness to market noise and shocks.

### Datasets
S&P 100 constituents via yfinance API

### Targets
The autoencoder's target is to reconstruct the input vector of asset returns. The ultimate portfolio optimization target is to maximize out-of-sample risk-adjusted returns (e.g., Sharpe Ratio).

### Model
The core model is an Autoencoder (AE), a type of unsupervised neural network. It consists of an encoder that maps high-dimensional input data (asset returns) to a low-dimensional latent space (the 'factors'), and a decoder that reconstructs the original data from this latent representation. The model is trained to minimize the reconstruction error (e.g., Mean Squared Error) between the input and output. A Denoising Autoencoder (DAE) variant is also used, which is trained to reconstruct the original data from a corrupted (noisy) input, making the learned factors more robust. The portfolio is constructed from the weights of the decoder network, which are interpreted as factor portfolios.

### Training
The model is trained and evaluated using a walk-forward validation scheme. The data is split into sequential, overlapping windows. For each window, the model is trained on an initial period of historical returns (e.g., 5 years of monthly returns). The trained model is then used to construct a portfolio for the subsequent period (e.g., the next month). This process is repeated, rolling forward in time, to generate a continuous out-of-sample return series. Training within each fold uses standard backpropagation with an optimizer like Adam to minimize the reconstruction loss.

### Evaluation
The primary evaluation methodology is a walk-forward backtest. The out-of-sample performance of the Deep Portfolio strategy (using both AE and DAE) will be compared against several benchmarks: an Equal-Weighted (1/N) portfolio and a classical Minimum Variance Portfolio (MVP). Key performance metrics include Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Annualized Volatility, and Portfolio Turnover. Performance will be assessed both gross and net of transaction costs.


## データ取得方法（共通データ基盤）

**合成データの自作は禁止。以下のARF Data APIからデータを取得すること。**

### ARF Data API
```bash
# OHLCV取得 (CSV形式)
curl -o data/aapl_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y"
curl -o data/btc_1h.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=BTC/USDT&interval=1h&period=1y"
curl -o data/nikkei_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=^N225&interval=1d&period=10y"

# JSON形式
curl "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y&format=json"

# 利用可能なティッカー一覧
curl "https://ai.1s.xyz/api/data/tickers"
```

### Pythonからの利用
```python
import pandas as pd
API = "https://ai.1s.xyz/api/data/ohlcv"
df = pd.read_csv(f"{API}?ticker=AAPL&interval=1d&period=5y")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")
```

### ルール
- **リポジトリにデータファイルをcommitしない** (.gitignoreに追加)
- 初回取得はAPI経由、以後はローカルキャッシュを使う
- data/ディレクトリは.gitignoreに含めること



## Preflight チェック（実装開始前に必ず実施）

**Phase の実装コードを書く前に**、以下のチェックを実施し結果を `reports/cycle_1/preflight.md` に保存すること。

### 1. データ境界表
以下の表を埋めて、未来データ混入がないことを確認:

```markdown
| 項目 | 値 |
|---|---|
| データ取得終了日 | YYYY-MM-DD (今日以前であること) |
| Train期間 | YYYY-MM-DD 〜 YYYY-MM-DD |
| Validation期間 | YYYY-MM-DD 〜 YYYY-MM-DD |
| Test期間 | YYYY-MM-DD 〜 YYYY-MM-DD |
| 重複なし確認 | Yes / No |
| 未来日付なし確認 | Yes / No |
```

### 2. Feature timestamp 契約
- 全ての特徴量は時刻 t の予測に t-1 以前のデータのみを使用しているか？ → Yes / No
- Scaler / Imputer は train データのみで fit しているか？ → Yes / No
- Centered rolling window を使用していないか？ → Yes / No (使用していたら修正)

### 3. Paper spec 差分表
論文の主要パラメータと現在の実装を比較:

```markdown
| パラメータ | 論文の値 | 現在の実装 | 一致? |
|---|---|---|---|
| ユニバース | (論文の記述) | (実装の値) | Yes/No |
| ルックバック期間 | (論文の記述) | (実装の値) | Yes/No |
| リバランス頻度 | (論文の記述) | (実装の値) | Yes/No |
| 特徴量 | (論文の記述) | (実装の値) | Yes/No |
| コストモデル | (論文の記述) | (実装の値) | Yes/No |
```

**preflight.md が作成されるまで、Phase の実装コードに進まないこと。**

## ★ 今回のタスク (Cycle 1)


### Phase 1: コアAutoencoderモデルの実装 [Track ]

**Track**:  (A=論文再現 / B=近傍改善 / C=独自探索)
**ゴール**: PyTorchで基本的なAutoencoderモデルを実装し、合成データで学習が実行できる状態にする。

**具体的な作業指示**:
1. `src/models/autoencoder.py`を作成します。
2. `torch.nn.Module`を継承した`Autoencoder`クラスを実装してください。エンコーダとデコーダはそれぞれ2つの線形層を持つ構成とします（例: `Input -> 128 -> 32 (factors) -> 128 -> Output`）。
3. `src/train.py`を作成し、このスクリプト内で合成データ（例: `torch.rand(1000, 256)`）を生成します。
4. `Autoencoder`モデルをインスタンス化し、MSELossとAdamオプティマイザを使用して10エポック学習させる簡単な学習ループを実装します。
5. 学習後のモデルを`models/autoencoder_v1.pth`に、学習中の損失を`reports/cycle_1/loss_log.json`に保存してください。
6. `README.md`に「本プロジェクトは論文の完全再現ではなく、公開データを用いた縮小版検証である」旨を明記してください。

**期待される出力ファイル**:
- src/models/autoencoder.py
- src/train.py
- models/autoencoder_v1.pth
- reports/cycle_1/loss_log.json
- README.md

**受入基準 (これを全て満たすまで完了としない)**:
- Autoencoderモデルの学習がエラーなく完了する。
- reports/cycle_1/loss_log.jsonにエポックごとの損失が記録されている。
- README.mdに縮小版検証である旨が記載されている。




## データ問題でスタックした場合の脱出ルール

レビューで3サイクル連続「データ関連の問題」が指摘されている場合:
1. **データの完全性を追求しすぎない** — 利用可能なデータでモデル実装に進む
2. **合成データでのプロトタイプを許可** — 実データが不足する部分は合成データで代替し、モデルの基本動作を確認
3. **データの制約を open_questions.md に記録して先に進む**
4. 目標は「論文の手法が動くこと」であり、「論文と同じデータを揃えること」ではない







## 全体Phase計画 (参考)

→ Phase 1: コアAutoencoderモデルの実装 — PyTorchで基本的なAutoencoderモデルを実装し、合成データで学習が実行できる状態にする。
  Phase 2: データパイプラインの構築 — yfinanceを使用して株価データを取得し、リターン計算と前処理を行うパイプラインを構築する。
  Phase 3: Walk-Forward評価とベンチマーク実装 — Walk-forward検証の枠組みを実装し、1/Nと最小分散ポートフォリオのバックテストを実行する。
  Phase 4: Deep Portfolio戦略の実装 — AutoencoderをWalk-Forwardの枠組みに統合し、Deep Portfolio戦略のバックテストを実行する。
  Phase 5: パフォーマンス評価指標とレポート作成 — 全戦略のパフォーマンスを計算・比較するレポートを生成する。
  Phase 6: Denoising Autoencoder (DAE) の実装と評価 — Denoising Autoencoderを実装し、そのパフォーマンスを既存戦略と比較する。
  Phase 7: ハイパーパラメータ最適化 — Autoencoderの主要なハイパーパラメータをOptunaで最適化する。
  Phase 8: 取引コストモデルの実装 — 取引コストを考慮したネットパフォーマンスを計算し、各戦略のコスト耐性を評価する。
  Phase 9: ロバスト性検証: ファクターポートフォリオ合成法の変更 — ファクターポートフォリオの合成方法を均等加重からリスクパリティに変更し、パフォーマンスへの影響を検証する。
  Phase 10: 実装改善: Variational Autoencoder (VAE) の導入 — 論文の範囲外の改善としてVAEを実装し、ポートフォリオ構築への応用を試みる。
  Phase 11: 最終レポートと可視化 — 全トラックの結果を統合し、包括的な最終レポートと累積リターンプロットを生成する。
  Phase 12: テスト、ドキュメント、エグゼクティブサマリー — コードの品質を向上させ、非技術者向けの要約を作成してプロジェクトを完成させる。


## ベースライン比較（必須）

戦略の評価には、以下のベースラインとの比較が**必須**。metrics.json の `customMetrics` にベースライン結果を含めること。

| ベースライン | 実装方法 | 意味 |
|---|---|---|
| **1/N (Equal Weight)** | 全資産に均等配分、月次リバランス | 最低限のベンチマーク |
| **Vol-Targeted 1/N** | 1/N にボラティリティターゲティング (σ_target=10%) を適用 | リスク調整後の公平な比較 |
| **Simple Momentum** | 12ヶ月リターン上位50%にロング | モメンタム系論文の場合の自然な比較対象 |

```python
# metrics.json に含めるベースライン比較
"customMetrics": {
  "baseline_1n_sharpe": 0.5,
  "baseline_1n_return": 0.05,
  "baseline_1n_drawdown": -0.15,
  "baseline_voltarget_sharpe": 0.6,
  "baseline_momentum_sharpe": 0.4,
  "strategy_vs_1n_sharpe_diff": 0.1,
  "strategy_vs_1n_return_diff": 0.02,
  "strategy_vs_1n_drawdown_diff": -0.05,
  "strategy_vs_1n_turnover_ratio": 3.2,
  "strategy_vs_1n_cost_sensitivity": "論文戦略はコスト10bpsで1/Nに劣後"
}
```

「敗北」の場合、**どの指標で負けたか** (return / sharpe / drawdown / turnover / cost) を technical_findings.md に明記すること。

## 評価原則
- **主指標**: Sharpe ratio (net of costs) on out-of-sample data
- **Walk-forward必須**: 単一のtrain/test splitでの最終評価は不可
- **コスト必須**: 全メトリクスは取引コスト込みであること
- **安定性**: Walk-forward窓の正の割合を報告
- **ベースライン必須**: 必ずナイーブ戦略と比較

## 再現モードのルール（論文忠実度の維持）

このプロジェクトは**論文再現**が目的。パフォーマンス改善より論文忠実度を優先すること。

### パラメータ探索の制約
- **論文で既定されたパラメータをまず実装し、そのまま評価すること**
- パラメータ最適化を行う場合、**論文既定パラメータの近傍のみ**を探索（例: 論文が12ヶ月なら [6, 9, 12, 15, 18] ヶ月）
- 論文と大きく異なるパラメータ（例: 月次論文に対して日次10営業日）で良い結果が出ても、それは「論文再現」ではなく「独自探索」
- 独自探索で得た結果は `customMetrics` に `label: "implementation-improvement"` として記録し、論文再現結果と明確に分離

### データ条件の忠実度
- 論文のデータ頻度（日次/月次/tick）にできるだけ合わせる
- ユニバース規模が論文より大幅に小さい場合、その制約を `docs/open_questions.md` に明記
- リバランス頻度・加重方法も論文に合わせる



## 禁止事項

### データ・特徴量の禁止パターン（具体的）
- `scaler.fit(full_data)` してから split → **禁止**。`scaler.fit(train_data)` のみ
- `df.rolling(window=N, center=True)` → **禁止**。`center=False` (デフォルト) を使用
- データの `end_date` が今日以降 → **禁止**。`end_date` を明示的に過去に設定
- `merge` で未来のタイムスタンプを持つ行が特徴量に混入 → **禁止**
- ラベル生成後に特徴量を合わせる（ラベルの存在を前提に特徴量を選択）→ **禁止**

### 評価・報告の禁止パターン
- コストなしのgross PnLだけで判断しない
- テストセットでハイパーパラメータを調整しない
- 時系列データにランダムなtrain/test splitを使わない
- README に metrics.json と異なる数値を手書きしない
- APIキーやクレデンシャルをコミットしない
- **新しい `scripts/run_cycle_N.py` や `scripts/experiment_cycleN.py` を作成しない。既存の `src/` 内ファイルを修正・拡張すること**
- **合成データを自作しない。必ずARF Data APIからデータを取得すること**
- **「★ 今回のタスク」以外のPhaseの作業をしない。1サイクル=1Phase**
- **論文が既定するパラメータから大幅に逸脱した探索を「再現」として報告しない**

## Git / ファイル管理ルール
- **データファイル(.csv, .parquet, .h5, .pkl, .npy)は絶対にgit addしない**
- `__pycache__/`, `.pytest_cache/`, `*.pyc` がリポジトリに入っていたら `git rm --cached` で削除
- `git add -A` や `git add .` は使わない。追加するファイルを明示的に指定する
- `.gitignore` を変更しない（スキャフォールドで設定済み）
- データは `data/` ディレクトリに置く（.gitignore済み）
- 学習済みモデルは `models/` ディレクトリに置く（.gitignore済み）

## 出力ファイル
以下のファイルを保存してから完了すること:
- `reports/cycle_1/preflight.md` — Preflight チェック結果（必須、実装前に作成）
- `reports/cycle_1/metrics.json` — 下記スキーマに従う（必須）
- `reports/cycle_1/technical_findings.md` — 実装内容、結果、観察事項

### metrics.json 必須スキーマ（Single Source of Truth）
```json
{
  "sharpeRatio": 0.0,
  "annualReturn": 0.0,
  "maxDrawdown": 0.0,
  "hitRate": 0.0,
  "totalTrades": 0,
  "transactionCosts": { "feeBps": 10, "slippageBps": 5, "netSharpe": 0.0 },
  "walkForward": { "windows": 0, "positiveWindows": 0, "avgOosSharpe": 0.0 },
  "customMetrics": {}
}
```
- 全フィールドを埋めること。Phase 1-2で未実装のメトリクスは0.0/0で可。
- `customMetrics`に論文固有の追加メトリクスを自由に追加してよい。

### レポート生成ルール（重要: 数値の一貫性）
- **`metrics.json` が全ての数値の唯一のソース (Single Source of Truth)**
- README や technical_findings に書く数値は **必ず metrics.json から引用** すること
- **手打ちの数値は禁止**。metrics.json に含まれない数値を README に書かない
- technical_findings.md で数値に言及する場合も metrics.json の値を参照
- README.md の Results セクションは metrics.json を読み込んで生成すること

### テスト必須
- `tests/test_data_integrity.py` のテストを実装状況に応じて有効化すること
- 新しいデータ処理や特徴量生成を追加したら、対応する leakage テストも追加
- `pytest tests/` が全パスしない場合、サイクルを完了としない

### その他の出力
- `docs/open_questions.md` — 未解決の疑問と仮定
- `README.md` — 今回のサイクルで変わった内容を反映して更新
- `docs/open_questions.md` に以下も記録:
  - ARF Data APIで問題が発生した場合
  - CLAUDE.mdの指示で不明確な点や矛盾がある場合
  - 環境やツールの制約で作業が完了できなかった場合

## 標準バックテストフレームワーク

`src/backtest.py` に以下が提供済み。ゼロから書かず、これを活用すること:
- `WalkForwardValidator` — Walk-forward OOS検証のtrain/test split生成
- `calculate_costs()` — ポジション変更に基づく取引コスト計算
- `compute_metrics()` — Sharpe, 年率リターン, MaxDD, Hit rate算出
- `generate_metrics_json()` — ARF標準のmetrics.json生成

```python
from src.backtest import WalkForwardValidator, BacktestConfig, calculate_costs, compute_metrics, generate_metrics_json
```

## Key Commands
```bash
pip install -e ".[dev]"
pytest tests/
python -m src.cli run-experiment --config configs/default.yaml
```

Commit all changes with descriptive messages.
