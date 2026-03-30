# Preflight Check — Cycle 4 (Phase 4: Deep Portfolio戦略の実装)

## 1. データ境界表

| 項目 | 値 |
|---|---|
| データ取得終了日 | 2023-12-31 (今日 2026-03-30 以前) |
| Train期間 | 2010-01-01 〜 各WFウィンドウのtest_start前日 (60ヶ月ローリング) |
| Validation期間 | N/A (Walk-forward OOSで代替) |
| Test期間 | 各WFウィンドウの1ヶ月 (2015-01 〜 2023-12) |
| 重複なし確認 | Yes — WalkForwardValidatorで train/test 境界を厳密分離 |
| 未来日付なし確認 | Yes — end_date=2023-12-31 で明示的に制限 |

## 2. Feature timestamp 契約

- 全ての特徴量は時刻 t の予測に t-1 以前のデータのみを使用しているか？ → **Yes** (月次リターンは月末終値から計算、pct_changeは後方参照)
- Scaler / Imputer は train データのみで fit しているか？ → **Yes** (Scalerは不使用、Autoencoderは各WFウィンドウのtrainデータのみで学習)
- Centered rolling window を使用していないか？ → **Yes** (rolling windowは不使用)

## 3. Paper spec 差分表

| パラメータ | 論文の値 | 現在の実装 | 一致? |
|---|---|---|---|
| ユニバース | S&P 500 (~500銘柄) | S&P 100相当 (82銘柄) | No — APIの制約、open_questionsに記載 |
| ルックバック期間 | 60ヶ月 (5年) | 60ヶ月 (5年) | Yes |
| リバランス頻度 | 月次 | 月次 | Yes |
| リターン頻度 | 月次 | **月次** (Cycle 4で日次→月次に修正) | Yes |
| AE構造 | Encoder-Decoder | input→128→32→128→input | Yes |
| ファクターポートフォリオ | デコーダ第1層の重み | decoder[0].weight | Yes |
| 合成方法 | 均等加重 | 均等加重 (各因子ポートフォリオの平均) | Yes |
| コストモデル | 論文では明示なし | 10bps fee + 5bps slippage | N/A |

### Cycle 4 改善点 (レビューフィードバック対応)
1. **月次リターン**: 日次→月次に変更 (論文準拠)
2. **デコーダ第1層の重み**: decoder[-1]→decoder[0]に変更 (論文準拠)
3. **MinVar重み上限**: 個別銘柄ウェイト上限0.05を追加 (集中問題の緩和)
4. **ユニバース**: ARF APIの制約で82銘柄のまま (open_questionsに記載)
