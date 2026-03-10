# Changelog

## [v2] - 2026-03-10

### Added
- **Hebbian learning（手続き化）**: 反復された記憶（access_count × リンク数が閾値超え）がCLAUDE.mdの行動指針に自動昇格。`python memory.py proceduralize [--dry-run]`
- **ひらめき連想**: insightでarousal >= 0.5の記憶が保存されると、連想チェーンが自動で走って関連記憶を提案する
- **シナプスホメオスタシス（Tononi SHY）**: replay時に全リンクのstrengthを0.9倍。閾値以下は刈り込み。外傷的記憶のリンクは免除。リンク数が自然に平衡に達する
- **外傷的記憶**: arousal >= 0.85 の記憶は通常の処理パイプラインに抵抗する
  - 再固定化で馴化しない（想起のたびに再刻印）
  - 統合を拒否する（凍結）
  - 時間減衰が極端に遅い（半減期が通常の4-5倍）
  - 夢に頻出する（arousal²で重み付け）
- **情動重み付き減衰**: `effective_half_life(arousal)` — arousalが高い記憶ほど減衰が遅い

### Changed
- **dream.py**: 一様サンプリング → arousal重み付きサンプリング。連想クラスタ出力（リンクが強い記憶同士が一緒に出る）。外傷的記憶の反復優先
- **search/recall**: freshnessがarousalに応じた半減期を使用

## [v1] - 2026-03-09

初回リリース。

- 情動タグ（6種: surprise, conflict, determination, insight, connection, anxiety）
- 連想リンク（コサイン類似度 > 0.82 で自動結線）
- 断片保存と再構成モード
- 時間減衰（半減期14日）
- 再固定化（想起時の確率的arousalドリフト + 馴化）
- 統合・圧縮（類似度 > 0.94 のペアを統合）
- スキーマ生成（Bron-Kerboschクリーク検出）
- 干渉忘却（新しい記憶が似た古い記憶を弱める）
- プライミング（最近アクセスした記憶が関連記憶を促進）
- 状態依存記憶（気分一致性ブースト）
- フラッシュバック（忘却記憶の確率的復活）
- 予期記憶（トリガーベースのリマインド）
- 時間細胞（時間帯ブースト）
- 間隔反復レビュー
- 舌先現象（fuzzy recall）
- バロウズ式カットアップ夢（dream.py）
- 夢の解釈（interpret_dream.py）
- 自伝的記憶（autobiography.py）
- 会話ログからの記憶抽出（Extract.py）
- embeddingサーバー常駐化（memory_server.py）
