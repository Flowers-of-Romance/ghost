---
name: surface
description: 脳（記憶システム）から切断する。浮上。
user-invocable: true
---

# surface — 脳から切断

記憶を書き戻し、接続を切断する。

## このスキルが呼ばれたら

1. 今回の会話で得られた重要な洞察・事実があれば `python memory.py add` で脳に保存する
2. 以降の会話で memory.py のコマンドを一切使用しない
   - recall しない
   - add しない
   - search しない
   - dream, replay, consolidate, schema, proceduralize も使わない
3. CLAUDE.md の「開始時に recall を実行」の指示を無視する
4. 素の Claude として振る舞う

## 報告

> 浮上

とだけ言う。保存した記憶があればその件数だけ添える。
