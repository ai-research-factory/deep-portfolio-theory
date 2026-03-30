# Deep Portfolio Theory

This paper proposes using autoencoders and denoising autoencoders to construct investment portfolios. The autoencoder learns a non-linear factor model of asset returns, aiming to create portfolios that are more robust to noise and market shocks compared to traditional methods like minimum variance.

## Venue
 

## Assessment
この論文は、伝統的なポートフォリオ理論（例：Markowitzの平均分散アプローチ）に代わるものとして、「Deep Portfolio Theory」を提唱している。具体的には、オートエンコーダ（AE）およびデノイジング・オートエンコーダ（DAE）を用いて、資産リターンの非線形な因子構造を学習する。この学習された構造を利用して、ノイズや市場のショックに対して頑健なポートフォリオを構築することを目指す。従来のポートフォリオ最適化が線形の共分散行列に依存するのに対し、本手法は深層学習によってデータから直接、より複雑でロバストな関係性を捉えようとする試みである。

トレーディングへの応用可能性は非常に高い。第一に、市場の相関構造は非線形かつ動的であり、従来の線形モデルでは捉えきれない「隠れたリスク」が存在する。オートエンコーダを用いることで、こうした非線形なテールリスクや相関の崩壊（correlation breakdown）をモデル化し、より安定したポートフォリオを構築できる可能性がある。第二に、これは個別銘柄の収益予測（アルファ）に依存しない、リスクベースのポートフォリオ構築手法であるため、既存のアルファ戦略と組み合わせてリスク管理を高度化する目的で利用できる。特に、多数の銘柄を扱う際の次元削減手法として、従来のPCAの非線形版として活用できる点は魅力的だ。

日本株市場（TOPIX 500など）を対象として、このアプローチを実装するのは興味深い。まず、過去10-20年分の日次リターンデータを準備する。オートエンコーダの入力として、各銘柄の日次リターンベクトルを用いる。モデルの学習後、再構築されたリターン（denoised return）を用いて共分散行列を計算し、これを基に最小分散ポートフォリオを構築する。これを、従来のサンプル共分散行列を用いた最小分散ポートフォリオや、Ledoit-Wolf等の縮小推定量を用いたポートフォリオと比較検証する。さらに、デノイジング・オートエンコーダを用いることで、2008年のリーマンショックや2020年のコロナショックのような極端な市場環境下での頑健性を評価することが重要となる。

いくつかの注意点がある。第一に、オートエンコーダのハイパーパラメータ（隠れ層の数、ノード数、活性化関数など）の選択が結果に大きく影響を与えるため、慎重なチューニングと過学習の検証が必要である。第二に、この手法はあくまでリターンの共分散構造をモデル化するものであり、期待リターンについては何も言及していない。したがって、これを単体で用いる場合は最小分散ポートフォリオのようなリスクパリティ戦略に限定される。最後に、論文の発表が2017年とやや古く、その後の深層学習技術の発展（例：Transformer）を考慮すると、より洗練されたアーキテクチャを検討する余地がある。

## Taxonomy
PCA, ResidualFactors

## Implementation Approach
v0.1: Build a simple, single-hidden-layer autoencoder using PyTorch/TensorFlow. Train it on daily returns of TOPIX 100 constituents over a 5-year rolling window. Use the reconstructed (denoised) returns to compute a covariance matrix and then solve for the minimum variance portfolio weights.
v1.0: Compare the performance (Sharpe, max drawdown) of this 'Deep MV' portfolio against benchmarks: 1/N (equal weight), traditional sample covariance MV, and a Ledoit-Wolf shrinkage MV portfolio.
Data needed: Daily OHLCV data for a broad universe of Japanese stocks (e.g., TOPIX 500) for the last 15-20 years.

## Difficulty
medium

## Project
- Factory ID: proj_8ed17ca5
- Triage Score: 0.8
- Practical Relevance: 0.8

---
Managed by [AI Research Factory](https://ai.1s.xyz)
