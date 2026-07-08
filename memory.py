#!/usr/bin/env python3
"""
memory.py - 脳に近い記憶システム

機能:
  1. 情動タグ — 内容から情動を自動推定。情動が強い記憶ほど残りやすい。
  2. 連想リンク — 記憶同士がネットワークでつながる。芋づる式に想起。
  3. 断片保存 — キーワードの束として保存。想起時に再構成。
  4. 減衰と強化 — 時間が経つと薄れ、使うと強まる。
  5. 再固定化 — 想起するたびに記憶が微妙に変化する（アクセスが記憶を書き換える）
  6. 統合・圧縮 — 似た記憶が一つの抽象的な知識に統合される
  7. 干渉忘却 — 新しい記憶が類似する古い記憶を弱める（能動的忘却）
  8. プライミング — 最近アクセスした記憶が関連記憶の想起を促進

セットアップ:
  pip install sentence-transformers numpy

使い方:
  python memory.py init
  python memory.py add "内容" [category] [source]
    categoryは fact / episode / context / preference / procedure / schema
  python memory.py search "検索語" [--fuzzy]
  python memory.py chain ID [depth]
  python memory.py recall
  python memory.py review [N]        # 間隔反復: 復習が必要な記憶をN件表示
  python memory.py replay          # 海馬リプレイ + 統合・圧縮
  python memory.py consolidate     # 類似記憶を統合（明示的）
  python memory.py detail ID
  python memory.py recent [N]
  python memory.py all
  python memory.py forget ID
  python memory.py resurrect "query"  # 忘却された記憶を復活検索
  python memory.py delusion "query"    # 完全記憶検索（忘却・バイアスなし）
  python memory.py delusion "query" --date 2024-12-11   # 日付フィルタ
  python memory.py delusion --date 2024-12-11           # その日の全記憶ダンプ
  python memory.py delusion --all                       # 全記憶ダンプ
  python memory.py delusion --raw "query"               # 原文のみ検索
  python memory.py delusion --context ID                # 記憶の対話文脈を復元
  python memory.py delusion --batch "q1" "q2" "q3"    # バッチ検索（複数キーワード一括）
  python memory.py schema             # リンク密集クラスタからスキーマ（メタ記憶）を生成
  python memory.py proceduralize      # 反復された記憶を行動指針に昇格（LEARNED.mdに書込み）
  python memory.py overview              # 俯瞰モード: 構造・重心・層・時系列
  python memory.py stats
  python memory.py mood [emotion] [arousal]  # 気分状態の設定・表示
  python memory.py mood clear                # 気分状態をクリア
  python memory.py prospect add "trigger" "action"  # 予期記憶を登録
  python memory.py prospect list                     # 予期記憶一覧
  python memory.py prospect clear ID                 # 予期記憶を完了
  python memory.py correct ID "new_content"           # 記憶を修正（旧版を保存）
  python memory.py versions ID                       # 記憶の版履歴を表示
  python memory.py mutations [ID]                    # メタデータ変異履歴の閲覧
  python memory.py calibrate                         # メタ認知: recall精度の自己検証
  python memory.py export [filename]                 # 記憶をJSONファイルにエクスポート
  python memory.py import filename                   # JSONファイルから記憶をインポート
  python memory.py sync status <host:port>           # 同期先の接続確認
  python memory.py sync push <host:port>             # ローカル→リモートに同期
  python memory.py sync pull <host:port>             # リモート→ローカルに同期
  python memory.py sync serve [--port N] [--public] [--insecure]  # 同期サーバーを起動
  python memory.py sync node-id                      # この端末のIDを表示
"""

import sqlite3
import sys
import os
import struct
import json
import re
import math
import io
import random
import uuid as _uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Windows cp932 で emoji が出力できない問題を回避
if sys.platform == "win32" and getattr(sys.stdout, 'encoding', '').lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# --- 設定 ---
DB_PATH = os.environ.get("MEMORY_DB_PATH", str(Path(__file__).parent / "memory.db"))
MOOD_PATH = str(Path(__file__).parent / ".mood")
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# 減衰の半減期（日数）
HALF_LIFE_DAYS = 14.0

# 連想リンクを張る類似度の閾値
# 低すぎると全記憶がリンクされてネットワークが意味をなさない
LINK_THRESHOLD = 0.82

# 干渉忘却: この類似度を超える古い記憶の重要度を下げる
# 2026-07-08 再較正: multilingual-e5-small の日本語類似度は 0.76-1.0 に圧縮され、
# 0.90 は「関連トピック」どまり（実測: 別事実ペアの最高値が0.96帯）。
# 0.90 のままだと embedding 復活後、add のたびに関連記憶を大量侵食する（実測で1 addあたり数十〜百件超）。
INTERFERENCE_THRESHOLD = 0.97

# 重複ゲート（2026-07-08 ghost記憶v2 — amateru/plans/ghost-memory-v2/plan.md）
# 閾値は全10,585件の実測分布から決定（measurement-2026-07-08.md）
# e5-small日本語圧縮: sim0.90=関連トピック、別事実ペアの最高値は0.96帯 → 0.98未満は触らない
MEMORY_GATE = os.environ.get("GHOST_MEMORY_GATE", "on").lower() != "off"


def _gate_float(env_key, default):
    """閾値envの防御的パース。不正値でimportごと落とさない"""
    try:
        return float(os.environ.get(env_key, default))
    except ValueError:
        return float(default)


GATE_DUP_SIM = _gate_float("GHOST_GATE_DUP", "0.995")      # NOOP帯
GATE_UPDATE_SIM = _gate_float("GHOST_GATE_UPDATE", "0.98")  # supersede帯

# 統合: この類似度を超える記憶ペアを統合候補とする
CONSOLIDATION_THRESHOLD = 0.94

# 緊張保持 (tension link): 類似トピックだが情動極性が対立する記憶ペアを
# 統合せず別レーンで保持する。象徴秩序の地下＝無意識層の最初の機能。
# - 検出器であって生成器ではない（既にあるデータの中の対立を救出するだけ）
# - 同じ話題と認める類似度の下限。これ未満は別話題なので無関係
TENSION_SIM_LOW = 0.72
# - これを超えたら consolidate へ送る（対立よりも統合が優先される強い類似）
TENSION_SIM_HIGH = 0.93
# - 情動極性ギャップの最小値（-1..+1 軸での絶対差。1.5 = 強極性 vs 反対極性）
TENSION_POLARITY_GAP_MIN = 1.5
# - 両方の arousal が低い対立は記録しない（凪いだ食い違いはノイズ扱い）
TENSION_AROUSAL_FLOOR = 0.35
# - keyword overlap の最小数（同じ話題の追加保証。embedding だけでは弱い）
TENSION_KEYWORD_OVERLAP_MIN = 1

# 情動の極性: emotions タグを正負中性に分類する。
# 緊張検出に使う。surprise は中性（驚きそのものは方向を持たない）。
EMOTION_POLARITY = {
    "insight": +1,
    "determination": +1,
    "connection": +1,
    "conflict": -1,
    "anxiety": -1,
    "surprise": 0,
}

# プライミング: 最近N分以内にアクセスした記憶からプライミング効果
PRIMING_WINDOW_MINUTES = 30

# 再固定化: 想起時に記憶の重要度が変動する確率と幅
RECONSOLIDATION_PROBABILITY = 0.3
RECONSOLIDATION_DRIFT = 0.15  # arousalが最大±15%変動

# 不随意記憶（フラッシュバック）: 忘却された記憶が自発的に蘇る
# 発火確率 = FLASHBACK_BASE_PROB * (元のarousal) * (類似度 - 閾値)
FLASHBACK_BASE_PROB = 0.15     # 基礎確率
FLASHBACK_SIM_THRESHOLD = 0.75  # この類似度を超えたら発火判定に入る

# 場所細胞: 同じ場所で作られた記憶は想起されやすい
SPATIAL_BOOST = 1.08  # 場所一致時のブースト倍率

# 関係細胞: 同じ人との間で作られた記憶は想起されやすい
RELATIONAL_BOOST = 1.12  # 関係一致時のブースト倍率（場所より強め）
RELATIONAL_WHO_DEFAULT = "J"  # デフォルトの関係者（環境変数 GHOST_WHO で上書き）

# 有効なカテゴリ（DB CHECK制約と同期バリデーションで共有）
VALID_CATEGORIES = frozenset({"fact", "episode", "context", "preference", "procedure", "schema"})

# 手続き化: 反復が閾値を超えた記憶を行動指針に昇格
PROCEDURALIZE_ACCESS_THRESHOLD = 20   # 最低想起回数
PROCEDURALIZE_LINK_THRESHOLD = 15     # 最低リンク数
LEARNED_MD_PATH = str(Path(__file__).parent / "LEARNED.md")
HEBB_MARKER = "## 学習された行動指針"

# 外傷的記憶: arousalがこの閾値を超えると既存メカニズムの挙動が変わる
# 馴化しない、統合されない、減衰が遅い、夢に頻出する
TRAUMA_AROUSAL_THRESHOLD = 0.85

# 修正可能性: 記憶の出自と信頼度
CONFIDENCE_DEFAULT = 0.7
CONFIDENCE_USER_EXPLICIT = 0.8
CONFIDENCE_WANDER = 0.3
CONFIDENCE_CONSOLIDATION_FACTOR = 0.9

# メタデータ変容: sleepのたびに記憶のメタデータが隣接記憶の影響で変化する
MUTATION_KW_ABSORB_PROB = 0.3       # キーワード吸収確率
MUTATION_KW_MAX_ADD = 2             # 1サイクルあたり最大追加数
MUTATION_KW_MAX_REMOVE = 1          # 1サイクルあたり最大削除数
MUTATION_KW_CAP_RATIO = 1.5         # キーワード総数キャップ（元の数 × この倍率）
MUTATION_EMBED_ALPHA = 0.05         # 埋め込みドリフト率
MUTATION_MIN_LINKS_EMBED = 3        # 埋め込みドリフトに必要な最低リンク数
MUTATION_MIN_LINKS_KW = 2           # キーワード吸収に必要な最低リンク数
MUTATION_MIN_LINKS_EMOTION = 5      # 情動ドリフトに必要な最低リンク数
MUTATION_COOLDOWN_HOURS = 4         # 連続変異防止のクールダウン時間

# --- 再固定化 (reconsolidation) ---
# 想起はread-write操作。人間の脳では想起のたびに記憶が再固定化される。
# embeddingが会話文脈方向に微小ドリフトし、文脈に適応していく。
RECONSOLIDATION_EMBED_ALPHA = 0.02           # recall時のembeddingドリフト率
RECONSOLIDATION_EMBED_ALPHA_FLASHBULB = 0.01 # flashbulb記憶のドリフト率（抵抗する）
RECALL_HABITUATION_WINDOW_MINUTES = 30       # セッション内馴化の時間窓
RECALL_HABITUATION_DECAY = 0.6               # 馴化によるスコア減衰係数
CONTEXT_RELEVANCE_THRESHOLD = 0.3            # 文脈関連度の閾値
CONTEXT_RELEVANCE_PENALTY = 0.5              # 文脈無関連時のペナルティ係数

# --- 自己調整パラメータ (ホモイコニック版) ---
# ルールは記憶として保存される。meta_paramsテーブルではなく、
# category='procedure'の記憶がパラメータの値を決める。
# 記憶だから減衰し、強化され、リンクされ、統合される。
# ルールと記憶の区別が溶ける。Lispのコード=データ。
#
# get_param()は生きた手続き記憶を想起してパラメータ値を導出する。
# 手続き記憶がなければデフォルト値にフォールバック。
# 手続き記憶が複数あれば、freshness×accessで重み付き平均。

_TUNABLE_DEFAULTS = {
    "half_life_days": HALF_LIFE_DAYS,
    "voice_empathy_balance": 0.7,     # 共感の声の左右バランス
    "voice_complement_balance": 0.7,  # 補完の声の左右バランス
    "voice_critic_balance": 0.5,      # 批判の声の左右バランス
    "voice_critic_boost": 1.3,        # 批判のconflict/anxietyブースト
    "recall_balance": 0.5,            # 通常recallの左右バランス
}

# 手続き記憶のコンテンツ形式:
# [手続き:param_name] 値:X.XXX 理由:... 精度:YY% 網羅:ZZ%
PROCEDURE_PREFIX = "[手続き:"
PROCEDURE_PATTERN = r'\[手続き:(\S+)\]\s*値:([\d.]+)'

_tuned_cache = None  # lazy load


def _load_tuned_params():
    """生きた手続き記憶からパラメータ値を導出する。

    複数の手続き記憶が同じパラメータについて存在する場合、
    freshness × access_count で重み付き平均を取る。
    新しくてよく使われるルールほど影響力が強い。
    古いルールは自然に声が小さくなり、やがて消える。
    """
    global _tuned_cache
    _tuned_cache = {}
    try:
        conn = get_connection()
        # category='procedure' かつ [手続き:...] 形式の記憶を取得
        rows = conn.execute(
            "SELECT m.id, c.content, m.created_at, m.access_count, m.forgotten "
            "FROM memories m JOIN cortex c ON c.id = m.id "
            "WHERE m.forgotten = 0 AND c.category = 'procedure' "
            "AND c.content LIKE ?",
            (PROCEDURE_PREFIX + '%',)
        ).fetchall()
        conn.close()

        if not rows:
            return

        # パラメータごとに手続き記憶を集めて重み付き平均
        param_entries = {}  # param_name -> [(value, weight)]
        for row in rows:
            match = re.search(PROCEDURE_PATTERN, row["content"])
            if not match:
                continue
            param_name = match.group(1)
            try:
                value = float(match.group(2))
            except ValueError:
                continue
            if param_name not in _TUNABLE_DEFAULTS:
                continue

            # 重み = freshness × (1 + log(access_count + 1))
            fresh = freshness(row["created_at"])
            access_w = 1.0 + math.log1p(row["access_count"])
            weight = fresh * access_w

            if param_name not in param_entries:
                param_entries[param_name] = []
            param_entries[param_name].append((value, weight, row["id"]))

        for param_name, entries in param_entries.items():
            total_weight = sum(w for _, w, _ in entries)
            if total_weight > 0:
                weighted_val = sum(v * w for v, w, _ in entries) / total_weight
                _tuned_cache[param_name] = weighted_val

    except Exception:
        _tuned_cache = {}


def get_param(name):
    """手続き記憶からパラメータを導出。記憶がなければデフォルト値。

    呼ばれるたびに手続き記憶が「使われた」ことになる——
    有効なルールは使われ続け、強化され、残る。
    使われないルールは減衰して消える。自然選択。"""
    global _tuned_cache
    if _tuned_cache is None:
        _load_tuned_params()
    if name in _tuned_cache:
        return _tuned_cache[name]
    return _TUNABLE_DEFAULTS.get(name)


# --- 情動辞書 ---
EMOTION_MARKERS = {
    "surprise": {
        "keywords": [
            "発見", "驚", "意外", "実は", "判明", "初めて", "まさか",
            "すごい", "面白い", "なるほど", "気づ", "新しい", "画期的",
            "unexpected", "surprising", "discovered", "breakthrough",
        ],
        "weight": 1.3,
    },
    "conflict": {
        "keywords": [
            "矛盾", "対立", "葛藤", "問題", "課題", "困", "難し",
            "議論", "反論", "批判", "疑問", "しかし", "だが", "けれど",
            "conflict", "contradiction", "debate", "however", "but",
        ],
        "weight": 1.2,
    },
    "determination": {
        "keywords": [
            "決定", "決めた", "始める", "やる", "作る", "実装",
            "方針", "計画", "目標", "挑戦", "コミット",
            "decided", "started", "committed", "will build",
        ],
        "weight": 1.2,
    },
    "insight": {
        "keywords": [
            "本質", "構造", "原理", "意味", "理解", "概念",
            "理論", "仮説", "証明", "論じ", "思想", "哲学",
            "つまり", "要するに", "核心",
            "essence", "insight", "fundamental", "theory", "hypothesis",
        ],
        "weight": 1.4,
    },
    "connection": {
        "keywords": [
            "一緒", "共同", "協力", "信頼", "感謝", "好き",
            "友", "仲間", "チーム", "関係",
            "together", "trust", "appreciate",
        ],
        "weight": 1.1,
    },
    "anxiety": {
        "keywords": [
            "不安", "心配", "恐", "リスク", "危険", "失敗",
            "怖い", "焦", "追われ", "間に合わ",
            "worried", "risk", "fear", "danger",
        ],
        "weight": 1.2,
    },
}

NEUTRAL_WEIGHT = 0.8

FLASHBULB_AROUSAL_THRESHOLD = 0.65
FLASHBULB_MAX_CHARS = 80


def detect_emotions(text):
    text_lower = text.lower()
    detected = []
    total_weight = 0.0

    for emotion, data in EMOTION_MARKERS.items():
        hits = sum(1 for kw in data["keywords"] if kw in text_lower or kw in text)
        if hits > 0:
            detected.append(emotion)
            total_weight += data["weight"] * min(hits, 3)

    # --- トーン分析（キーワード以外の手がかり） ---
    tone_boost = 0.0

    # 感嘆符 → 覚醒度ブースト
    excl_count = text.count('!') + text.count('！')
    tone_boost += min(excl_count, 5) * 0.05

    # 疑問符 → 不安・葛藤シグナル
    ques_count = text.count('?') + text.count('？')
    if ques_count > 0:
        tone_boost += min(ques_count, 3) * 0.03
        if "anxiety" not in detected and ques_count >= 2:
            detected.append("anxiety")
        if "conflict" not in detected and ques_count >= 3:
            detected.append("conflict")

    # ALL CAPSの単語 → 驚き・決意ブースト
    caps_words = re.findall(r'\b[A-Z]{2,}\b', text)
    if caps_words:
        tone_boost += min(len(caps_words), 3) * 0.06
        if "surprise" not in detected:
            detected.append("surprise")
        if "determination" not in detected and len(caps_words) >= 2:
            detected.append("determination")

    # 長文 → 推敲の痕跡、重要度ブースト
    if len(text) > 200:
        tone_boost += 0.1

    # 省略記号 → 不安・躊躇シグナル
    ellipsis_count = text.count('...') + text.count('…')
    if ellipsis_count > 0:
        tone_boost += min(ellipsis_count, 3) * 0.04
        if "anxiety" not in detected:
            detected.append("anxiety")

    if not detected:
        arousal = 0.2
        importance = 2
    else:
        arousal = min(1.0, (total_weight + tone_boost) / 4.0)
        importance = max(1, min(5, round(arousal * 4 + 1)))

    return detected, arousal, importance


def extract_keywords(text):
    en_words = re.findall(r'[A-Za-z][A-Za-z0-9_\-]+', text)
    en_words = [w.lower() for w in en_words if len(w) > 2]
    jp_chunks = re.findall(r'[\u4e00-\u9fff\u30a0-\u30ff]{2,}', text)
    jp_hira = re.findall(r'[\u3040-\u309f]{4,}', text)
    keywords = list(set(en_words + jp_chunks + jp_hira))
    return keywords


def _extract_flashbulb_sentence(text):
    """テキストから最も情動的な一文を抽出する。閾値未満ならNone。"""
    # 短いテキストはそのまま返す
    if len(text) <= FLASHBULB_MAX_CHARS:
        _, arousal, _ = detect_emotions(text)
        return text if arousal >= FLASHBULB_AROUSAL_THRESHOLD else None

    # 文分割（日本語の。と英語の.!?）
    sentences = re.split(r'(?<=[。．.!?！？])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
    if not sentences:
        return None

    best_sentence = None
    best_arousal = 0.0

    for s in sentences:
        _, arousal, _ = detect_emotions(s)
        if arousal > best_arousal:
            best_arousal = arousal
            best_sentence = s

    if best_arousal < FLASHBULB_AROUSAL_THRESHOLD or best_sentence is None:
        # 個別文では閾値を超えない場合、全文の覚醒度で判断し先頭を返す
        _, full_arousal, _ = detect_emotions(text)
        if full_arousal >= FLASHBULB_AROUSAL_THRESHOLD:
            best_sentence = text[:FLASHBULB_MAX_CHARS - 1] + "…"
            return best_sentence
        return None

    # 80文字上限
    if len(best_sentence) > FLASHBULB_MAX_CHARS:
        best_sentence = best_sentence[:FLASHBULB_MAX_CHARS - 1] + "…"
    return best_sentence


# --- Embedding ---
_model = None
EMBED_SERVER_URL = "http://127.0.0.1:7234/embed"
_server_alive = None  # キャッシュ: True/False/None(未チェック)


def is_embed_server_alive():
    """サーバーが生きているか確認（結果をキャッシュ）。"""
    global _server_alive
    if _server_alive is not None:
        return _server_alive
    try:
        import urllib.request
        req = urllib.request.Request("http://127.0.0.1:7234/health")
        with urllib.request.urlopen(req, timeout=1) as resp:
            _server_alive = resp.status == 200
    except Exception:
        _server_alive = False
    return _server_alive


def _embed_via_server(text, is_query=False):
    """サーバー経由でembedding取得。速い（モデルロード不要）。"""
    import urllib.request
    import numpy as np
    payload = json.dumps({"text": text, "is_query": is_query}).encode()
    req = urllib.request.Request(
        EMBED_SERVER_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        vec = json.loads(resp.read())
    return np.array(vec, dtype=np.float32)


def get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(EMBEDDING_MODEL)
        except ImportError:
            return None
    return _model


def normalize_content(text):
    """重複判定用の正規化: 前後空白除去 + 連続空白の畳み込み"""
    return re.sub(r"\s+", " ", (text or "").strip())


def content_hash(text):
    """正規化contentのSHA-256（重複ゲート Phase 1a）"""
    import hashlib
    return hashlib.sha256(normalize_content(text).encode("utf-8")).hexdigest()


def _gate_noop(conn, memory_id, reason, detail=""):
    """重複ゲートNOOP: 既存記憶を活性化し、mutation_logに記録。INSERTしない"""
    now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    conn.execute(
        "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
        (now_utc, memory_id))
    conn.execute(
        "INSERT INTO mutation_log (memory_id, field, old_value, new_value, reason) "
        "VALUES (?, 'gate', NULL, ?, ?)",
        (memory_id, detail, reason))
    conn.commit()


def backfill_embeddings(dry_run=False, batch=256, missing_only=False):
    """全記憶のembeddingを一括再生成（重複ゲートv2 Phase 0）。

    既存の旧embedding（互換実測 min 0.913）も含めて統一再生成する。
    sentence-transformers が必要 → ~/.claude/ghost/.venv/bin/python3 で実行。
    """
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) FROM memories WHERE forgotten = 0").fetchone()[0]
    missing = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE forgotten = 0 AND embedding IS NULL").fetchone()[0]
    if missing_only:
        print(f"対象: embedding NULL の {missing}件のみ（増分モード / forgotten=0 全体: {total}件）")
    else:
        print(f"対象: forgotten=0 の {total}件（embedding NULL: {missing}件 / 統一のため全件再生成）")
    if dry_run:
        print("  dry-run のため書き込みなし")
        conn.close()
        return
    model = get_model()
    if model is None:
        print("✗ sentence-transformers 不在。~/.claude/ghost/.venv/bin/python3 memory.py backfill-embeddings で実行")
        conn.close()
        return
    import time as _time
    t0 = _time.time()
    _where = " AND embedding IS NULL" if missing_only else ""
    rows = conn.execute(
        f"SELECT id, content FROM memories WHERE forgotten = 0{_where} ORDER BY id").fetchall()
    if not rows:
        print("  対象なし")
        conn.close()
        return
    ids = [r["id"] for r in rows]
    texts = [f"passage: {r['content']}" for r in rows]
    done = 0
    for i in range(0, len(texts), batch):
        vecs = model.encode(texts[i:i + batch], batch_size=64,
                            normalize_embeddings=True, show_progress_bar=False)
        for mid, v in zip(ids[i:i + batch], vecs):
            blob = vec_to_bytes(v)
            conn.execute("UPDATE memories SET embedding = ? WHERE id = ?", (blob, mid))
            conn.execute("UPDATE cortex SET embedding = ? WHERE id = ?", (blob, mid))
            _sync_vec_insert(conn, mid, blob)
        conn.commit()
        done += len(vecs)
        if done % 2048 < batch:
            print(f"  {done}/{len(texts)} ...")
    print(f"✓ backfill 完了: {done}件 / {_time.time() - t0:.1f}s")
    conn.close()


def reconcile(dry_run=True, since=None, sim_threshold=0.995):
    """既存記憶の重複整理（重複ゲートv2 Phase 2 — 記憶大掃除）。

    Stage A: content_hash 完全一致クラスタ
    Stage B: 保存済みembedding同士の類似クラスタ（sim >= 0.995・同category のみ。
             一括処理は敵対レビュー指摘によりadd時ゲート(0.98)より高い閾値に制限）
    残す1件 = importance最大 → confidence最大 → 最古。他は版保存 + forgotten=1 + merged_from。
    物理削除はしない。デフォルト dry-run。
    """
    conn = get_connection()
    where_since = ""
    params = []
    if since:
        where_since = " AND created_at >= ?"
        params.append(since)

    rows = conn.execute(
        f"SELECT id, uuid, category, content, content_hash, importance, confidence, "
        f"created_at, source_conversation, embedding FROM memories "
        f"WHERE forgotten = 0{where_since} ORDER BY id", params).fetchall()
    print(f"対象: {len(rows)}件（forgotten=0{' / since=' + since if since else ''}）")

    def keep_key(r):
        return (-(r["importance"] or 0), -(r["confidence"] or 0), r["created_at"], r["id"])

    clusters = []  # (kind, [rows])

    # Stage A: ハッシュ一致
    by_hash = {}
    for r in rows:
        by_hash.setdefault(r["content_hash"], []).append(r)
    stage_a_ids = set()
    for h, grp in by_hash.items():
        if h and len(grp) > 1:
            clusters.append(("exact", sorted(grp, key=keep_key)))
            stage_a_ids.update(r["id"] for r in grp)

    # Stage B: 類似（Stage A対象外・embedding有りのみ）
    b_rows = [r for r in rows if r["id"] not in stage_a_ids and r["embedding"] is not None]
    if b_rows:
        import numpy as _np
        M = _np.stack([bytes_to_vec(r["embedding"]) for r in b_rows])
        M = M / (_np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        n = len(b_rows)
        used = set()
        BATCH = 512
        for i0 in range(0, n, BATCH):
            S = M[i0:i0 + BATCH] @ M.T
            for j in range(S.shape[0]):
                gi = i0 + j
                if gi in used:
                    continue
                sims = S[j]
                grp_idx = [k for k in range(n)
                           if k != gi and k not in used and sims[k] >= sim_threshold
                           and b_rows[k]["category"] == b_rows[gi]["category"]]
                if grp_idx:
                    grp = [b_rows[gi]] + [b_rows[k] for k in grp_idx]
                    used.add(gi)
                    used.update(grp_idx)
                    clusters.append(("similar", sorted(grp, key=keep_key)))

    to_forget = sum(len(g) - 1 for _, g in clusters)
    n_exact = sum(1 for k, _ in clusters if k == "exact")
    n_sim = sum(1 for k, _ in clusters if k == "similar")
    print(f"クラスタ: {len(clusters)}個（完全一致 {n_exact} / 類似 {n_sim}）")
    print(f"無効化予定: {to_forget}件 / 残存予定: {len(rows) - to_forget}件")

    # source別内訳（97%重複の主因検証用）
    src_count = {}
    for _, grp in clusters:
        for r in grp[1:]:
            key = (r["source_conversation"] or "(source無し)")[:30]
            src_count[key] = src_count.get(key, 0) + 1
    print("無効化対象のsource内訳（上位8）:")
    for k, v in sorted(src_count.items(), key=lambda x: -x[1])[:8]:
        print(f"  {v:6d}  {k}")

    if dry_run:
        print("\n--- サンプル（各種別から最大10クラスタ）---")
        shown = {"exact": 0, "similar": 0}
        for kind, grp in clusters:
            if shown[kind] >= 10:
                continue
            shown[kind] += 1
            keep, rest = grp[0], grp[1:]
            print(f"[{kind}] keep #{keep['id']} ({keep['created_at'][:10]}) 他{len(rest)}件無効化")
            print(f"   {normalize_content(keep['content'])[:80]}")
        print("\ndry-run のため変更なし。本実行: reconcile --execute")
        conn.close()
        return

    done = 0
    for kind, grp in clusters:
        keep, rest = grp[0], grp[1:]
        for r in rest:
            _snapshot_version(conn, r["id"], f"reconcile_{kind}", superseded_by=keep["id"])
            conn.execute("UPDATE memories SET forgotten = 1, merged_from = ? WHERE id = ?",
                         (keep["uuid"], r["id"]))
            conn.execute(
                "INSERT INTO mutation_log (memory_id, field, old_value, new_value, reason) "
                "VALUES (?, 'gate', ?, ?, ?)",
                (r["id"], r["uuid"], f"kept=#{keep['id']}", f"reconcile_{kind}"))
            done += 1
        if done % 500 < len(rest):
            conn.commit()
    conn.commit()
    print(f"✓ reconcile 完了: {done}件を無効化（版保存済み・可逆）")
    conn.close()


def embed_text(text, is_query=False):
    # まずサーバーに問い合わせ（高速）
    try:
        return _embed_via_server(text, is_query)
    except Exception:
        pass
    # フォールバック: ローカルでモデルロード（遅い）
    model = get_model()
    if model is None:
        return None
    prefix = "query: " if is_query else "passage: "
    return model.encode(prefix + text, normalize_embeddings=True)


def vec_to_bytes(vec):
    return struct.pack(f'{len(vec)}f', *vec.tolist())


def bytes_to_vec(b):
    import numpy as np
    n = len(b) // 4
    return np.array(struct.unpack(f'{n}f', b), dtype=np.float32)


def cosine_similarity(a, b):
    import numpy as np
    return float(np.dot(a, b))


# --- 時間減衰 ---

def effective_half_life(arousal):
    """arousalに応じた半減期を返す。外傷的記憶は減衰が極端に遅い。
    半減期の基準値は自己調整される（get_param経由）。"""
    base = get_param("half_life_days")
    if arousal >= TRAUMA_AROUSAL_THRESHOLD:
        return base * (1 + arousal * 4)  # 0.85→4.4倍, 1.0→5倍
    return base * (1 + arousal * 2)       # 通常: 0.3→1.6倍, 0.5→2倍


def freshness(created_at_str, half_life=None):
    if half_life is None:
        half_life = get_param("half_life_days")
    try:
        created = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        return 0.5
    now = datetime.now(timezone.utc)
    days = (now - created).total_seconds() / 86400.0
    return math.exp(-0.693 * days / half_life)


# --- 気分状態（state-dependent memory） ---
# 気分は手動設定もできるが、会話の情動と想起した記憶の情動から自動更新される。
# 指数移動平均で直近の情動入力を重み付け。古い入力は自然に減衰する。

MOOD_DECAY = 0.7  # 新しい入力の重み（0.7 = 新30%、旧70%...ではなく新70%寄り）
MOOD_HISTORY_MAX = 10  # 履歴の最大保持数


def load_mood():
    """現在の気分状態を読み込む。なければNone。"""
    if os.path.exists(MOOD_PATH):
        try:
            with open(MOOD_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def save_mood(emotions, arousal):
    """気分状態を保存する（手動設定用。履歴はリセット）。"""
    if isinstance(emotions, str):
        emotions = [emotions]
    data = {"emotions": emotions, "arousal": arousal, "history": []}
    with open(MOOD_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    return data


def update_mood(new_emotions, new_arousal):
    """
    気分を自動更新する。指数移動平均で新しい入力を混ぜる。

    脳の仕組み:
    - 会話で出てきた情動が気分に影響する（情動伝染）
    - 想起した記憶の情動にも引きずられる（気分一致効果の逆方向）
    - 古い気分は徐々に減衰して中立に戻る
    """
    if not new_emotions and new_arousal <= 0.2:
        return  # 中立入力は気分を動かさない

    mood = load_mood()
    if mood is None:
        mood = {"emotions": [], "arousal": 0.2, "history": []}

    old_emotions = set(mood.get("emotions", []))
    old_arousal = mood.get("arousal", 0.2)
    history = mood.get("history", [])

    # 履歴に追加
    if new_emotions:
        history.append({
            "emotions": new_emotions if isinstance(new_emotions, list) else [new_emotions],
            "arousal": new_arousal,
            "t": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        })
        if len(history) > MOOD_HISTORY_MAX:
            history = history[-MOOD_HISTORY_MAX:]

    # 情動の加重集計（直近の入力ほど重い）
    emotion_scores = {}
    weight = 1.0
    for entry in reversed(history):
        for emo in entry.get("emotions", []):
            emotion_scores[emo] = emotion_scores.get(emo, 0) + weight
        weight *= (1 - MOOD_DECAY)

    # 上位の情動を現在の気分にする
    sorted_emos = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
    current_emotions = [e for e, s in sorted_emos[:3] if s > 0.1]

    # arousalの指数移動平均
    current_arousal = old_arousal * (1 - MOOD_DECAY) + new_arousal * MOOD_DECAY

    data = {"emotions": current_emotions, "arousal": round(current_arousal, 3), "history": history}
    with open(MOOD_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def clear_mood():
    """気分状態をクリアする。"""
    if os.path.exists(MOOD_PATH):
        os.remove(MOOD_PATH)


def get_mood_congruence_boost(row):
    """気分一致性ブースト: 現在の気分と記憶の情動が重なると想起されやすい。"""
    mood = load_mood()
    if mood is None:
        return 1.0
    mood_emotions = set(mood.get("emotions", []))
    mood_arousal = mood.get("arousal", 0.5)
    if not mood_emotions:
        return 1.0
    mem_emotions = set(json.loads(row["emotions"])) if row["emotions"] else set()
    overlap = mood_emotions & mem_emotions
    if overlap:
        return 1.0 + mood_arousal * 0.2
    return 1.0


# --- DB操作 ---

_LINDERA_DYLIB = str(Path(__file__).parent / "ext" / "lindera_fts5.dylib")
_LINDERA_CONFIG = str(Path(__file__).parent / "ext" / "lindera.yml")


def get_connection():
    # timeout + busy_timeout: sleep.py や embed server 等の並走プロセスと
    # WALロックが競合しても即死せず最大30秒待つ（#AP-022 起動時recall失敗の根本対策）
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA journal_mode=WAL")
    # 拡張を読み込み (sqlite-vec, lindera FTS5 tokenizer)
    # AttributeError は load_extension 非対応の Python (Apple system python など)
    try:
        conn.enable_load_extension(True)
        try:
            import sqlite_vec
            sqlite_vec.load(conn)
        except ImportError:
            pass
        if os.path.exists(_LINDERA_DYLIB):
            os.environ.setdefault("LINDERA_CONFIG_PATH", _LINDERA_CONFIG)
            conn.load_extension(_LINDERA_DYLIB, entrypoint="lindera_fts5_tokenizer_init")
        conn.enable_load_extension(False)
    except (ImportError, AttributeError):
        pass
    return conn


def init_db():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'fact'
                CHECK (category IN ('fact', 'episode', 'context', 'preference', 'procedure', 'schema')),  -- VALID_CATEGORIES と同期
            importance INTEGER NOT NULL DEFAULT 3
                CHECK (importance BETWEEN 1 AND 5),
            emotions TEXT DEFAULT '[]',
            arousal REAL DEFAULT 0.2,
            keywords TEXT DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            last_accessed TEXT,
            access_count INTEGER NOT NULL DEFAULT 0,
            forgotten INTEGER NOT NULL DEFAULT 0,
            source_conversation TEXT,
            embedding BLOB,
            -- 統合された記憶の元IDを記録
            merged_from TEXT DEFAULT NULL,
            flashbulb TEXT DEFAULT NULL
        );

        CREATE TABLE IF NOT EXISTS links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            strength REAL NOT NULL DEFAULT 0.5,
            link_type TEXT DEFAULT 'association',
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            FOREIGN KEY (source_id) REFERENCES memories(id),
            FOREIGN KEY (target_id) REFERENCES memories(id),
            UNIQUE(source_id, target_id)
        );

        CREATE INDEX IF NOT EXISTS idx_memories_forgotten ON memories(forgotten);
        CREATE INDEX IF NOT EXISTS idx_links_source ON links(source_id);
        CREATE INDEX IF NOT EXISTS idx_links_target ON links(target_id);

        CREATE TABLE IF NOT EXISTS prospective (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trigger_pattern TEXT NOT NULL,
            action TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            fired INTEGER NOT NULL DEFAULT 0,
            fire_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS procedures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER NOT NULL UNIQUE,
            rule_text TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            FOREIGN KEY (memory_id) REFERENCES memories(id)
        );
    """)
    # CHECK制約にprocedure/schemaを追加（既存DBのテーブルを再作成）
    try:
        # 既存テーブルのCHECK制約を確認
        info = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='memories'").fetchone()
        if info and 'schema' not in info[0]:
            # 全カラムを保持したままCHECK制約を更新
            # 既存テーブルのカラム一覧を取得
            cols = [row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()]
            cols_str = ", ".join(cols)
            conn.executescript(f"""
                ALTER TABLE memories RENAME TO memories_old;
                CREATE TABLE memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'fact'
                        CHECK (category IN ('fact', 'episode', 'context', 'preference', 'procedure', 'schema')),
                    importance INTEGER NOT NULL DEFAULT 3
                        CHECK (importance BETWEEN 1 AND 5),
                    emotions TEXT DEFAULT '[]',
                    arousal REAL DEFAULT 0.2,
                    keywords TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                    last_accessed TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    forgotten INTEGER NOT NULL DEFAULT 0,
                    source_conversation TEXT,
                    embedding BLOB,
                    merged_from TEXT DEFAULT NULL,
                    flashbulb TEXT DEFAULT NULL,
                    context_expires_at TEXT DEFAULT NULL,
                    temporal_context TEXT DEFAULT NULL,
                    spatial_context TEXT DEFAULT NULL,
                    uuid TEXT,
                    updated_at TEXT
                );
                INSERT INTO memories ({cols_str})
                    SELECT {cols_str} FROM memories_old;
                DROP TABLE memories_old;
                CREATE INDEX IF NOT EXISTS idx_memories_forgotten ON memories(forgotten);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_uuid ON memories(uuid);
            """)
            print("  ✓ CHECK制約を更新しました")
    except Exception as e:
        print(f"  (CHECK制約の更新をスキップ: {e})")

    # merged_from カラムを追加（既存DBの場合）
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN merged_from TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # already exists
    # content_hash カラム（重複ゲート 2026-07-08 v2）
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN content_hash TEXT")
    except sqlite3.OperationalError:
        pass  # already exists
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_content_hash ON memories(content_hash)")
    _null_hash = conn.execute(
        "SELECT id, content FROM memories WHERE content_hash IS NULL").fetchall()
    if _null_hash:
        for _r in _null_hash:
            conn.execute("UPDATE memories SET content_hash = ? WHERE id = ?",
                         (content_hash(_r["content"]), _r["id"]))
        conn.commit()
        print(f"  ✓ content_hash を {len(_null_hash)}件 補完（重複ゲートv2マイグレーション）")
    # context_expires_at カラムを追加（既存DBの場合）
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN context_expires_at TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # already exists
    # temporal_context カラムを追加（既存DBの場合）
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN temporal_context TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # already exists
    # spatial_context カラムを追加（既存DBの場合）
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN spatial_context TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # already exists

    # flashbulb カラムを追加（既存DBの場合）
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN flashbulb TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # already exists

    # === P2P同期用カラム ===
    # uuid: 端末間で記憶を一意に識別
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN uuid TEXT")
    except sqlite3.OperationalError:
        pass
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_uuid ON memories(uuid)")

    # 既存レコードにUUID付与
    rows_no_uuid = conn.execute("SELECT id FROM memories WHERE uuid IS NULL").fetchall()
    for row in rows_no_uuid:
        conn.execute("UPDATE memories SET uuid = ? WHERE id = ?",
                     (str(_uuid.uuid4()), row[0]))

    # updated_at: 同期の差分検出用
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN updated_at TEXT")
    except sqlite3.OperationalError:
        pass
    # 既存レコードのupdated_atをcreated_atで埋める
    conn.execute("UPDATE memories SET updated_at = created_at WHERE updated_at IS NULL")

    # linksにuuidペアとupdated_at
    try:
        conn.execute("ALTER TABLE links ADD COLUMN source_uuid TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE links ADD COLUMN target_uuid TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE links ADD COLUMN updated_at TEXT")
    except sqlite3.OperationalError:
        pass
    # 既存linksのuuidを埋める
    conn.execute("""
        UPDATE links SET
            source_uuid = (SELECT uuid FROM memories WHERE id = links.source_id),
            target_uuid = (SELECT uuid FROM memories WHERE id = links.target_id),
            updated_at = COALESCE(links.updated_at, links.created_at)
        WHERE source_uuid IS NULL OR target_uuid IS NULL OR updated_at IS NULL
    """)

    # prospectiveにuuid
    try:
        conn.execute("ALTER TABLE prospective ADD COLUMN uuid TEXT")
    except sqlite3.OperationalError:
        pass
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_prospective_uuid ON prospective(uuid)")
    rows_no_uuid = conn.execute("SELECT id FROM prospective WHERE uuid IS NULL").fetchall()
    for row in rows_no_uuid:
        conn.execute("UPDATE prospective SET uuid = ? WHERE id = ?",
                     (str(_uuid.uuid4()), row[0]))

    # proceduresにuuidとmemory_uuid
    try:
        conn.execute("ALTER TABLE procedures ADD COLUMN uuid TEXT")
    except sqlite3.OperationalError:
        pass
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_procedures_uuid ON procedures(uuid)")
    try:
        conn.execute("ALTER TABLE procedures ADD COLUMN memory_uuid TEXT")
    except sqlite3.OperationalError:
        pass
    rows_no_uuid = conn.execute("SELECT id, memory_id FROM procedures WHERE uuid IS NULL").fetchall()
    for row in rows_no_uuid:
        mem_uuid = conn.execute("SELECT uuid FROM memories WHERE id = ?", (row[1],)).fetchone()
        conn.execute("UPDATE procedures SET uuid = ?, memory_uuid = ? WHERE id = ?",
                     (str(_uuid.uuid4()), mem_uuid[0] if mem_uuid else None, row[0]))

    # updated_atを自動更新するトリガー
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_memories_updated_at
        AFTER UPDATE ON memories
        FOR EACH ROW
        WHEN NEW.updated_at = OLD.updated_at OR NEW.updated_at IS NULL
        BEGIN
            UPDATE memories SET updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
            WHERE id = NEW.id;
        END
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_links_updated_at
        AFTER UPDATE ON links
        FOR EACH ROW
        WHEN NEW.updated_at = OLD.updated_at OR NEW.updated_at IS NULL
        BEGIN
            UPDATE links SET updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
            WHERE id = NEW.id;
        END
    """)

    # sync_meta: 同期履歴
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sync_meta (
            peer_id TEXT PRIMARY KEY,
            peer_url TEXT NOT NULL,
            last_sync_at TEXT,
            last_sync_direction TEXT
        )
    """)

    # node_id: この端末のID（初回生成）
    conn.execute("""
        CREATE TABLE IF NOT EXISTS node_info (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    existing_node = conn.execute("SELECT value FROM node_info WHERE key = 'node_id'").fetchone()
    if not existing_node:
        conn.execute("INSERT INTO node_info (key, value) VALUES ('node_id', ?)",
                     (str(_uuid.uuid4()),))

    # === raw_turns: 対話原文の完全保存 ===
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            message_uuid TEXT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            cwd TEXT,
            git_branch TEXT,
            memory_ids TEXT DEFAULT '[]'
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_turns_session ON raw_turns(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_turns_timestamp ON raw_turns(timestamp)")

    # === felt_moments — Claude の自己報告された感情の表出 ===
    # role='assistant' の turn で感情語彙が表に出た瞬間を mark する。
    # 中間層の運動は API では取れないので、出力テクストに現れた症状だけを記録する。
    conn.execute("""
        CREATE TABLE IF NOT EXISTS felt_moments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            turn_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            phrase TEXT NOT NULL,
            span_start INTEGER,
            span_end INTEGER,
            surrounding TEXT,
            extracted_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            FOREIGN KEY (turn_id) REFERENCES raw_turns(id) ON DELETE CASCADE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_felt_moments_turn ON felt_moments(turn_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_felt_moments_label ON felt_moments(label)")

    # === FTS5: lindera_tokenizer 統合 ===
    # tokenize は SQLite 拡張側が形態素解析するため、content には原文を入れる。
    # 旧スキーマ (unicode61 + Python pre-tokenize) からの移行は _migrate_fts_to_lindera で。
    try:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content,
                memory_id UNINDEXED,
                tokenize='lindera_tokenizer'
            )
        """)
    except Exception:
        pass  # FTS5 / lindera が使えない環境

    try:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS raw_turns_fts USING fts5(
                content,
                turn_id UNINDEXED,
                tokenize='lindera_tokenizer'
            )
        """)
    except Exception:
        pass

    _migrate_fts_to_lindera(conn)

    # 既存memoriesのFTSインデックスを構築（まだ入っていないもの）
    _rebuild_fts_index(conn)

    # === メタデータ変容ログ ===
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mutation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER NOT NULL,
            field TEXT NOT NULL,
            old_value TEXT,
            new_value TEXT,
            reason TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        )
    """)
    # last_mutated カラム追加
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN last_mutated TEXT")
    except sqlite3.OperationalError:
        pass  # already exists

    # provenance カラム追加
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN provenance TEXT DEFAULT 'unknown'")
    except sqlite3.OperationalError:
        pass
    # confidence カラム追加
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN confidence REAL DEFAULT 0.7")
    except sqlite3.OperationalError:
        pass
    # revision_count カラム追加
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN revision_count INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass

    # 既存データのバックフィル
    conn.execute("UPDATE memories SET provenance='wander', confidence=0.3 WHERE provenance='unknown' AND source_conversation LIKE 'wander:%'")
    conn.execute("UPDATE memories SET provenance='user_explicit', confidence=0.8 WHERE provenance='unknown' AND (source_conversation IS NULL OR source_conversation NOT LIKE 'wander:%')")
    conn.execute("UPDATE memories SET provenance='consolidation', confidence=0.63 WHERE provenance='unknown' AND merged_from IS NOT NULL")

    # memory_versions テーブル
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            importance INTEGER,
            arousal REAL,
            confidence REAL,
            reason TEXT,
            superseded_by INTEGER,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            FOREIGN KEY (memory_id) REFERENCES memories(id)
        )
    """)

    # sleep_meta: nap/sleep のメタ情報
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sleep_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    # === catalog_cards: 目録カード (v30) ===
    # sleep の神経学的整理に対し、catalog は navigability の整理。
    # 夜間に build_catalog() が更新する。LLM が pull で引ける目録。
    conn.execute("""
        CREATE TABLE IF NOT EXISTS catalog_cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_type TEXT NOT NULL,
            key TEXT NOT NULL,
            title TEXT,
            content TEXT,
            related_ids TEXT DEFAULT '[]',
            stats TEXT DEFAULT '{}',
            generation REAL DEFAULT 1.0,
            source_hash TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT,
            UNIQUE(entry_type, key)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_catalog_type ON catalog_cards(entry_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_catalog_key ON catalog_cards(key)")
    # catalog 全文検索用 FTS5
    try:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS catalog_fts USING fts5(
                title, content, key, tokenize='lindera_tokenizer'
            )
        """)
    except sqlite3.OperationalError:
        pass  # FTS5 / lindera 未ビルドの場合は諦める

    # === 左脳/右脳テーブル分割 (v17.1) ===
    # cortex（左脳）: 意味的・分析的データ
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cortex (
            id INTEGER PRIMARY KEY,
            content TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'fact',
            keywords TEXT DEFAULT '[]',
            embedding BLOB,
            confidence REAL DEFAULT 0.7,
            provenance TEXT DEFAULT 'unknown',
            revision_count INTEGER DEFAULT 0,
            merged_from TEXT DEFAULT NULL,
            FOREIGN KEY (id) REFERENCES memories(id)
        )
    """)
    # limbic（右脳）: 情動的・直感的データ
    conn.execute("""
        CREATE TABLE IF NOT EXISTS limbic (
            id INTEGER PRIMARY KEY,
            emotions TEXT DEFAULT '[]',
            arousal REAL DEFAULT 0.2,
            flashbulb TEXT DEFAULT NULL,
            temporal_context TEXT DEFAULT NULL,
            spatial_context TEXT DEFAULT NULL,
            relational_context TEXT DEFAULT NULL,
            FOREIGN KEY (id) REFERENCES memories(id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cortex_category ON cortex(category)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_limbic_arousal ON limbic(arousal)")

    # relational_context カラム追加（既存DBの場合）
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN relational_context TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE limbic ADD COLUMN relational_context TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass
    # origin カラム追加（ソースモニタリング: 誰が生成した情報か）
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN origin TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass
    # domains カラム追加（凡例 / 地図の一軸 v29）: JSON 配列
    # 例: ["impl/memory", "concept/memory"]
    # 意味領域（memory / 記憶 / 実装 / claude / jun 等）の混濁を緩和するための
    # soft hint。territory を消さず、_left_score に乗る multiplier としてのみ作用。
    try:
        conn.execute("ALTER TABLE cortex ADD COLUMN domains TEXT DEFAULT '[\"unknown\"]'")
    except sqlite3.OperationalError:
        pass

    # データ移行: memoriesからcortex/limbicにコピー（まだ移行されていない場合）
    migrated = conn.execute("SELECT COUNT(*) FROM cortex").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    if migrated < total:
        conn.execute("""
            INSERT OR IGNORE INTO cortex (id, content, category, keywords, embedding,
                confidence, provenance, revision_count, merged_from)
            SELECT id, content, category, keywords, embedding,
                confidence, provenance, revision_count, merged_from
            FROM memories WHERE id NOT IN (SELECT id FROM cortex)
        """)
        conn.execute("""
            INSERT OR IGNORE INTO limbic (id, emotions, arousal, flashbulb,
                temporal_context, spatial_context)
            SELECT id, emotions, arousal, flashbulb,
                temporal_context, spatial_context
            FROM memories WHERE id NOT IN (SELECT id FROM limbic)
        """)
        new_count = total - migrated
        if new_count > 0:
            print(f"  ✓ {new_count}件をcortex/limbicに移行")

    # memories_v VIEW: 後方互換（旧スキーマと同一カラム名）
    conn.execute("DROP VIEW IF EXISTS memories_v")
    conn.execute("""
        CREATE VIEW memories_v AS
        SELECT m.id, c.content, c.category, m.importance,
               l.emotions, l.arousal, c.keywords,
               m.created_at, m.last_accessed, m.access_count, m.forgotten,
               m.source_conversation, c.embedding, c.merged_from,
               l.flashbulb, m.context_expires_at, l.temporal_context,
               l.spatial_context, l.relational_context,
               m.uuid, m.updated_at, m.last_mutated,
               c.provenance, c.confidence, c.revision_count,
               c.domains
        FROM memories m
        JOIN cortex c ON c.id = m.id
        JOIN limbic l ON l.id = m.id
    """)

    # === recall_log: recallの自己検証（メタ認知） ===
    conn.execute("""
        CREATE TABLE IF NOT EXISTS recall_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_ts TEXT NOT NULL,
            recalled_ids TEXT NOT NULL DEFAULT '[]',
            accessed_ids TEXT DEFAULT NULL,
            precision REAL DEFAULT NULL,
            recall_rate REAL DEFAULT NULL,
            noise_ids TEXT DEFAULT NULL,
            missed_ids TEXT DEFAULT NULL,
            evaluated_at TEXT DEFAULT NULL
        )
    """)
    # 声の帰属: どの声がどの記憶を出したか（再帰的自己調整の信号源）
    try:
        conn.execute("ALTER TABLE recall_log ADD COLUMN voice_attribution TEXT DEFAULT NULL")
    except Exception:
        pass  # already exists

    # === meta_params: 自己調整パラメータ ===
    # 記憶の内容（recall精度）が記憶の構造（パラメータ）を変える。
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta_params (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            default_value TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            reason TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta_params_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL,
            old_value TEXT,
            new_value TEXT NOT NULL,
            reason TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        )
    """)

    # === sqlite-vec: ベクトルインデックス ===
    try:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                memory_id INTEGER PRIMARY KEY,
                embedding FLOAT[384]
            )
        """)
        # 既存embeddingをマイグレーション（まだ入っていないもの）
        vec_count = conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]
        mem_count = conn.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL").fetchone()[0]
        if vec_count < mem_count:
            rows = conn.execute(
                "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL AND id NOT IN (SELECT memory_id FROM memories_vec)"
            ).fetchall()
            for row in rows:
                conn.execute(
                    "INSERT INTO memories_vec (memory_id, embedding) VALUES (?, ?)",
                    (row[0], row[1])
                )
            if rows:
                print(f"  ✓ {len(rows)}件のembeddingをmemories_vecに移行")
    except Exception as e:
        print(f"  (sqlite-vec初期化をスキップ: {e})")

    # === perceptions テーブル (v28 prayer) ===
    # prayer_daemon による視聴覚知覚の一次記録。
    # 三次サイバネティックス原則: 差異重視・double description・circuit記録。
    # memories とは別テーブル（高volume・ephemeral）だが、links で結合される。
    conn.execute("""
        CREATE TABLE IF NOT EXISTS perceptions (
            id INTEGER PRIMARY KEY,
            uuid TEXT UNIQUE NOT NULL,
            timestamp TEXT NOT NULL,
            modality TEXT NOT NULL,
            content_cortex TEXT,
            content_limbic TEXT,
            content_delta TEXT,
            novelty_score REAL,
            raw_frame_path TEXT,
            raw_audio_path TEXT,
            arousal REAL DEFAULT 0.2,
            invoked_by TEXT DEFAULT 'auto',
            session_id TEXT,
            linked_turn_id INTEGER,
            embedding BLOB,
            importance INTEGER DEFAULT 1,
            forgotten INTEGER DEFAULT 0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_perceptions_ts ON perceptions(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_perceptions_session ON perceptions(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_perceptions_modality ON perceptions(modality)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_perceptions_forgotten ON perceptions(forgotten)")

    conn.commit()
    conn.close()
    print(f"✓ memory.db を初期化しました: {DB_PATH}")


def _has_vec_table(conn):
    """memories_vec仮想テーブルが存在するかチェック。"""
    try:
        conn.execute("SELECT COUNT(*) FROM memories_vec")
        return True
    except Exception:
        return False


def vec_search(conn, query_vec, k=50, forgotten=None):
    """sqlite-vecを使ったベクトル近傍検索。

    Args:
        forgotten: None=全件, False=forgotten=0のみ, True=forgotten=1のみ
    Returns: [(memory_id, distance, similarity), ...]
    vec0はL2距離を返す。normalizedベクトルなので cosine_sim = 1 - dist²/2。
    """
    if query_vec is None:
        return []
    if not _has_vec_table(conn):
        return []
    query_blob = vec_to_bytes(query_vec)
    # vec0はJOIN/WHERE制約をサポートしないので、多めに取ってPython側でフィルタ
    # forgotten=0は全体の約60%なので、必要数の3倍を取る（最低200件）
    fetch_k = max(k * 3, 200) if forgotten is not None else k
    rows = conn.execute(
        "SELECT memory_id, distance FROM memories_vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (query_blob, fetch_k)
    ).fetchall()
    # forgottenフィルタが必要な場合はmemoriesテーブルで確認
    results = []
    if forgotten is not None:
        forgotten_val = 1 if forgotten else 0
        for row in rows:
            mid = row[0]
            m = conn.execute("SELECT forgotten FROM memories WHERE id = ?", (mid,)).fetchone()
            if m and m[0] == forgotten_val:
                dist = row[1]
                sim = 1.0 - (dist * dist) / 2.0
                results.append((mid, dist, sim))
                if len(results) >= k:
                    break
    else:
        for row in rows:
            dist = row[1]
            sim = 1.0 - (dist * dist) / 2.0
            results.append((row[0], dist, sim))
    return results


def _sync_vec_insert(conn, memory_id, embedding_blob):
    """memories_vecにembeddingを同期挿入する。"""
    if embedding_blob is None:
        return
    if not _has_vec_table(conn):
        return
    try:
        conn.execute(
            "INSERT OR REPLACE INTO memories_vec (memory_id, embedding) VALUES (?, ?)",
            (memory_id, embedding_blob)
        )
    except Exception:
        pass


def _sync_vec_delete(conn, memory_id):
    """memories_vecからembeddingを削除する。"""
    if not _has_vec_table(conn):
        return
    try:
        conn.execute("DELETE FROM memories_vec WHERE memory_id = ?", (memory_id,))
    except Exception:
        pass


def _migrate_fts_to_lindera(conn):
    """旧 (unicode61 + Python pre-tokenize) スキーマを lindera_tokenizer に置き換える。

    sqlite_master から CREATE 文を読み、'lindera_tokenizer' を含まない FTS5 vtable
    があれば DROP して新スキーマで作り直し、primary table から再 populate する。
    冪等。新規 DB では何もしない。
    """
    targets = [
        # (fts_table, columns_def, repopulate_sql)
        (
            "memories_fts",
            "content, memory_id UNINDEXED, tokenize='lindera_tokenizer'",
            "INSERT INTO memories_fts(content, memory_id) "
            "SELECT content, id FROM memories WHERE content IS NOT NULL",
        ),
        (
            "raw_turns_fts",
            "content, turn_id UNINDEXED, tokenize='lindera_tokenizer'",
            "INSERT INTO raw_turns_fts(content, turn_id) "
            "SELECT content, id FROM raw_turns WHERE content IS NOT NULL",
        ),
        (
            "catalog_fts",
            "title, content, key, tokenize='lindera_tokenizer'",
            "INSERT INTO catalog_fts(title, content, key) "
            "SELECT COALESCE(title,''), COALESCE(content,''), key FROM catalog_cards",
        ),
    ]
    for table, cols, repop in targets:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not row or not row["sql"]:
            continue  # vtable がまだない (init 中の新規作成側で作られる)
        if "lindera_tokenizer" in row["sql"]:
            continue  # 既に lindera 化済み
        try:
            conn.execute(f"DROP TABLE {table}")
            conn.execute(f"CREATE VIRTUAL TABLE {table} USING fts5({cols})")
            conn.execute(repop)
            print(f"  ✓ FTS5 lindera 化: {table}")
        except Exception as e:
            print(f"  ⚠️ {table} migration failed: {e}")


def _rebuild_fts_index(conn):
    """既存memoriesのFTSインデックスを差分構築する。

    lindera_tokenizer 統合後は content をそのまま FTS5 に渡すだけ。
    """
    try:
        conn.execute("SELECT count(*) FROM memories_fts").fetchone()
    except Exception:
        return  # FTS5テーブルなし

    indexed_ids = set(
        row[0] for row in conn.execute(
            "SELECT memory_id FROM memories_fts"
        ).fetchall()
    )

    rows = conn.execute(
        "SELECT id, content FROM memories WHERE content IS NOT NULL"
    ).fetchall()

    added = 0
    for row in rows:
        if str(row["id"]) in indexed_ids or row["id"] in indexed_ids:
            continue
        conn.execute(
            "INSERT INTO memories_fts (content, memory_id) VALUES (?, ?)",
            (row["content"], row["id"])
        )
        added += 1

    if added > 0:
        print(f"  ✓ FTSインデックス: {added}件追加")


# ============================================================
# FTS5 ヘルパー — BM25 重み・snippet 整形・OR フォールバック
# ============================================================
# Namazu 的な「転置インデックス + TF-IDF + ヒット周辺の切り出し」を FTS5
# ネイティブ機能で実現する共通レイヤ。検索各経路から呼び出される。

# FTS5 MATCH がデフォルト AND で厳しすぎるケース用。repair-links で実績。
_FTS_OR_MAX_TOKENS = 10


def _tokenize_or_query(query, max_tokens=_FTS_OR_MAX_TOKENS):
    """クエリを形態素解析して `token1 OR token2 OR ...` 形の MATCH 式を返す。

    FTS5 の AND デフォルトで 0 件になるケースの二段目フォールバックに使う。
    短すぎる (1 文字) トークンは除く。tokenize 不能時は None。
    """
    try:
        from tokenizer import tokenize
        tokenized = tokenize(query or "")
    except Exception:
        return None
    if not tokenized.strip():
        return None
    tokens = [t for t in tokenized.split() if len(t) > 1][:max_tokens]
    if not tokens:
        return None
    return " OR ".join(tokens)


def _bm25_bonus_map(conn, query, fts_table, id_col, max_rows=200, bonus_min=0.05, bonus_max=0.10):
    """FTS5 の bm25() をランク基準に [bonus_min, bonus_max] のボーナスへ正規化。

    ヒットした id に対する dict を返す。BM25 は小さいほど関連度高 (負値が多い)。
    最良ヒットが bonus_max、最悪ヒットが bonus_min。FTS 使用不可時は空 dict。
    lindera_tokenizer が MATCH 側で形態素解析するため、query は原文のまま渡す。
    """
    if not query or not query.strip():
        return {}
    try:
        rows = conn.execute(
            f"""SELECT {id_col} AS id, bm25({fts_table}) AS s
                FROM {fts_table} WHERE {fts_table} MATCH ?
                ORDER BY s LIMIT ?""",
            (query, max_rows)
        ).fetchall()
    except Exception:
        return {}
    if not rows:
        return {}
    scores = [r["s"] for r in rows]
    s_min, s_max = min(scores), max(scores)
    rng = (s_max - s_min) or 1.0
    return {r["id"]: bonus_min + (bonus_max - bonus_min) * (s_max - r["s"]) / rng
            for r in rows}


def _rejoin_tokenized_snippet(s):
    """FTS5 snippet() の出力 (空白区切り形態素) を可読な形に寄せる。

    memories_fts / raw_turns_fts は tokenize 済みテキストをそのまま格納しており、
    snippet() も同じ空白区切りで返る。隣接する CJK 文字間のスペースだけ除き、
    英単語 (ASCII) 間のスペースは保つ。
    """
    if not s:
        return s
    return re.sub(
        r'([぀-ヿ一-鿿])\s+(?=[぀-ヿ一-鿿])',
        r'\1', s
    )


# ============================================================
# 5. 再固定化 — 想起するたびに記憶が変化する
# ============================================================

def reconsolidate(conn, memory_id):
    """
    再固定化: 記憶を想起するたびに微妙に変化させる。

    脳科学の知見: 記憶を思い出すたびに、その記憶は不安定になり
    再び固定化される。このプロセスで記憶は微妙に変容する。
    これはバグではなく特徴——記憶は「保存されたデータ」ではなく
    「そのつど再生成されるパターン」。

    実装:
    - 確率的にarousal（情動の強さ）が変動する
    - よく想起する記憶ほど重要度が上がる（強化学習的）
    - ただし、極端に頻繁にアクセスすると馴化（慣れ）が起きて重要度が下がる
    """
    if random.random() > RECONSOLIDATION_PROBABILITY:
        return False  # 今回は変化しない

    row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not row:
        return False

    old_arousal = row["arousal"]
    access = row["access_count"]

    # 外傷的記憶: 馴化しない。むしろ想起するたびに再刻印される
    if old_arousal >= TRAUMA_AROUSAL_THRESHOLD:
        drift = random.uniform(-0.02, RECONSOLIDATION_DRIFT)  # 下がりにくく上がりやすい
        drift += 0.02  # 再刻印バイアス
        new_arousal = max(0.0, min(1.0, old_arousal + drift))
        new_importance = max(1, min(5, round(new_arousal * 4 + 1)))
        conn.execute(
            "UPDATE memories SET arousal = ?, importance = ? WHERE id = ?",
            (new_arousal, new_importance, memory_id)
        )
        return abs(drift) > 0.02

    # 変動: ランダムなドリフト
    drift = random.uniform(-RECONSOLIDATION_DRIFT, RECONSOLIDATION_DRIFT)

    # 適度にアクセスされる記憶は強化される（1-10回）
    if 1 <= access <= 10:
        drift += 0.05  # 正方向にバイアス
    # 過度にアクセスされると馴化（慣れ）
    elif access > 20:
        drift -= 0.05  # 負方向にバイアス

    new_arousal = max(0.0, min(1.0, old_arousal + drift))
    new_importance = max(1, min(5, round(new_arousal * 4 + 1)))

    conn.execute(
        "UPDATE memories SET arousal = ?, importance = ? WHERE id = ?",
        (new_arousal, new_importance, memory_id)
    )
    return abs(drift) > 0.05  # 意味のある変化があったか


# ============================================================
# 緊張保持 — 統合せずに対立を別レーンで残す
# ============================================================

def _emotion_polarity(emotions_list):
    """emotions タグの極性平均を -1..+1 で返す。空なら 0。"""
    if not emotions_list:
        return 0.0
    scores = [EMOTION_POLARITY.get(e, 0) for e in emotions_list]
    return sum(scores) / len(scores)


def _detect_tension(a, b, sim):
    """
    類似トピックだが対立する記憶ペアか判定する。
    対立を生成しない。既存データの中で merge に紛れ込もうとする対立を救出する。

    Returns: (is_tension: bool, strength: float in 0..1)
    """
    emo_a = json.loads(a["emotions"]) if a["emotions"] else []
    emo_b = json.loads(b["emotions"]) if b["emotions"] else []
    pa = _emotion_polarity(emo_a)
    pb = _emotion_polarity(emo_b)
    polarity_gap = abs(pa - pb)

    if polarity_gap < TENSION_POLARITY_GAP_MIN:
        return False, 0.0

    # 両方が凪いでいたら対立として記録に値しない
    if a["arousal"] < TENSION_AROUSAL_FLOOR and b["arousal"] < TENSION_AROUSAL_FLOOR:
        return False, 0.0

    # 両方とも極性持ちの emotion を一つは持っていること（中性 vs 強極性は弱いシグナル）
    has_pol_a = any(EMOTION_POLARITY.get(e, 0) != 0 for e in emo_a)
    has_pol_b = any(EMOTION_POLARITY.get(e, 0) != 0 for e in emo_b)
    if not (has_pol_a and has_pol_b):
        return False, 0.0

    # keyword overlap：embedding が近いだけでは「同じ話題」と言いきれない
    kw_a = set(json.loads(a["keywords"]) if a["keywords"] else [])
    kw_b = set(json.loads(b["keywords"]) if b["keywords"] else [])
    overlap = kw_a & kw_b
    if len(overlap) < TENSION_KEYWORD_OVERLAP_MIN:
        return False, 0.0

    # 同じ極性が混じっている場合は弱める（純粋な対立ではないので）
    common_emo = set(emo_a) & set(emo_b)
    common_polarity_present = any(EMOTION_POLARITY.get(e, 0) != 0 for e in common_emo)
    overlap_dampening = 0.5 if common_polarity_present else 1.0

    strength = sim * (polarity_gap / 2.0) * overlap_dampening
    return True, min(1.0, strength)


def _insert_tension_link(conn, source_id, target_id, strength):
    """tension link を挿入。既存リンクがあれば link_type を tension に更新し
    strength は max を取る。"""
    existing = conn.execute(
        "SELECT id, strength, link_type FROM links "
        "WHERE (source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?)",
        (source_id, target_id, target_id, source_id)
    ).fetchone()

    if existing:
        new_strength = max(existing["strength"], strength)
        conn.execute(
            "UPDATE links SET strength = ?, link_type = 'tension' WHERE id = ?",
            (new_strength, existing["id"])
        )
        return False  # 新規ではない
    else:
        conn.execute(
            "INSERT INTO links (source_id, target_id, strength, link_type) "
            "VALUES (?, ?, ?, 'tension')",
            (source_id, target_id, strength)
        )
        return True


def detect_tensions(dry_run=False, sim_low=None, sim_high=None):
    """
    moderate な類似度範囲（sim_low..sim_high）の記憶ペアをスキャンして、
    情動極性が対立しているペアに tension link を貼る。
    consolidate より広い範囲（同じ話題だが merge には至らない）を担当する。

    consolidate_memories の中での tension 救出と独立に動く。
    sleep の一段階として呼ばれる想定。
    """
    if sim_low is None:
        sim_low = TENSION_SIM_LOW
    if sim_high is None:
        sim_high = TENSION_SIM_HIGH

    conn = get_connection()
    rows = conn.execute(
        "SELECT id, content, keywords, embedding, importance, arousal, emotions, category "
        "FROM memories WHERE forgotten = 0 AND embedding IS NOT NULL"
    ).fetchall()

    if len(rows) < 2:
        print("緊張検出: 記憶が足りません")
        conn.close()
        return {"scanned": 0, "detected": 0, "inserted": 0, "updated": 0}

    scanned = 0
    detected = 0
    inserted = 0
    updated = 0
    examples = []

    for i, a in enumerate(rows):
        vec_a = bytes_to_vec(a["embedding"])
        if vec_a is None:
            continue
        for b in rows[i+1:]:
            vec_b = bytes_to_vec(b["embedding"])
            if vec_b is None:
                continue
            sim = cosine_similarity(vec_a, vec_b)
            if sim < sim_low or sim >= sim_high:
                continue
            scanned += 1
            is_tension, strength = _detect_tension(a, b, sim)
            if not is_tension:
                continue
            detected += 1
            if dry_run:
                if len(examples) < 10:
                    examples.append((a, b, sim, strength))
            else:
                is_new = _insert_tension_link(conn, a["id"], b["id"], strength)
                if is_new:
                    inserted += 1
                else:
                    updated += 1

    if not dry_run:
        conn.commit()
    conn.close()

    if dry_run:
        print(f"緊張候補: {detected}件 (スキャン {scanned}ペア、sim {sim_low}〜{sim_high})")
        for a, b, sim, strength in examples:
            print(f"  sim={sim:.2f} strength={strength:.2f}")
            print(f"    #{a['id']}: {a['content'][:60]}")
            print(f"    #{b['id']}: {b['content'][:60]}")
    else:
        print(f"✓ 緊張検出: 新規 {inserted}件 / 更新 {updated}件 / 検出 {detected}件 / スキャン {scanned}ペア")

    return {"scanned": scanned, "detected": detected, "inserted": inserted, "updated": updated}


def list_tensions(limit=30):
    """tension link を一覧表示。"""
    conn = get_connection()
    rows = conn.execute(
        """SELECT l.id, l.source_id, l.target_id, l.strength, l.created_at,
                  ma.content AS content_a, mb.content AS content_b,
                  ma.emotions AS emo_a, mb.emotions AS emo_b
           FROM links l
           JOIN memories ma ON ma.id = l.source_id
           JOIN memories mb ON mb.id = l.target_id
           WHERE l.link_type = 'tension'
             AND ma.forgotten = 0 AND mb.forgotten = 0
           ORDER BY l.strength DESC, l.created_at DESC
           LIMIT ?""",
        (limit,)
    ).fetchall()
    conn.close()
    if not rows:
        print("tension link はありません")
        return
    print(f"tension links ({len(rows)}件):")
    for r in rows:
        emo_a = " ".join(EMOTION_EMOJI.get(e, "·") for e in (json.loads(r["emo_a"]) if r["emo_a"] else []))
        emo_b = " ".join(EMOTION_EMOJI.get(e, "·") for e in (json.loads(r["emo_b"]) if r["emo_b"] else []))
        print(f"  [{r['id']}] strength={r['strength']:.2f}")
        print(f"    #{r['source_id']} {emo_a} {r['content_a'][:60]}")
        print(f"    #{r['target_id']} {emo_b} {r['content_b'][:60]}")


def forget_tension(link_id):
    """誤検出された tension link を消す。"""
    conn = get_connection()
    row = conn.execute(
        "SELECT link_type FROM links WHERE id = ?", (link_id,)
    ).fetchone()
    if not row:
        print(f"link #{link_id} は存在しません")
        conn.close()
        return
    if row["link_type"] != "tension":
        print(f"link #{link_id} は tension ではありません (link_type={row['link_type']})")
        conn.close()
        return
    conn.execute("DELETE FROM links WHERE id = ?", (link_id,))
    conn.commit()
    conn.close()
    print(f"✓ tension link #{link_id} を削除しました")


def _tension_cli(args):
    if not args:
        print("使い方: python memory.py tension {list|detect|forget <id>}")
        return
    sub = args[0]
    if sub == "list":
        limit = 30
        for i, a in enumerate(args[1:], start=1):
            if a == "--limit" and i + 1 < len(args):
                try:
                    limit = int(args[i + 1])
                except ValueError:
                    pass
        list_tensions(limit=limit)
    elif sub == "detect":
        dry = "--dry-run" in args
        detect_tensions(dry_run=dry)
    elif sub == "forget":
        if len(args) < 2:
            print("使い方: python memory.py tension forget <link_id>")
            return
        try:
            lid = int(args[1])
        except ValueError:
            print(f"link_id が不正です: {args[1]}")
            return
        forget_tension(lid)
    else:
        print(f"不明なサブコマンド: {sub}")
        print("使い方: python memory.py tension {list|detect|forget <id>}")


# ============================================================
# 6. 統合・圧縮 — 類似する記憶を一つにまとめる
# ============================================================

def consolidate_memories(dry_run=False):
    """
    統合・圧縮: 類似度が非常に高い記憶ペアを一つの統合記憶にまとめる。

    脳の睡眠中の処理に相当:
    - 個別のエピソード記憶からスキーマ（抽象的な知識）が生まれる
    - 「APIが要ると思ったが不要だった」+「個人契約のこっち」→
      「Claude Codeはローカルで動くので追加コスト不要と判明」
    - 元の記憶は忘却フラグを立て、統合記憶に merged_from で記録

    実装: 似た記憶のキーワードを結合し、内容を連結して新しい記憶を作る
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, content, keywords, embedding, importance, arousal, emotions, category, confidence, flashbulb "
        "FROM memories WHERE forgotten = 0 AND embedding IS NOT NULL"
    ).fetchall()

    if len(rows) < 2:
        print("統合するには記憶が足りません")
        conn.close()
        return

    # 類似度が高いペアを見つける（外傷的記憶は統合を拒否する——凍結）
    pairs = []
    for i, a in enumerate(rows):
        vec_a = bytes_to_vec(a["embedding"])
        for b in rows[i+1:]:
            # どちらかが外傷的arousalならスキップ
            if a["arousal"] >= TRAUMA_AROUSAL_THRESHOLD or b["arousal"] >= TRAUMA_AROUSAL_THRESHOLD:
                continue
            vec_b = bytes_to_vec(b["embedding"])
            sim = cosine_similarity(vec_a, vec_b)
            if sim > CONSOLIDATION_THRESHOLD:
                pairs.append((a, b, sim))

    pairs.sort(key=lambda x: x[2], reverse=True)

    if not pairs:
        print("統合候補はありません")
        conn.close()
        return

    merged_ids = set()
    consolidation_count = 0
    tension_count = 0

    for a, b, sim in pairs:
        if a["id"] in merged_ids or b["id"] in merged_ids:
            continue

        # 緊張保持: 高類似度でも情動極性が対立しているなら統合せず別レーンで残す
        is_tension, t_strength = _detect_tension(a, b, sim)
        if is_tension:
            if dry_run:
                print(f"  緊張保持 (sim:{sim:.3f}, strength:{t_strength:.2f}):")
                print(f"    #{a['id']}: {a['content'][:50]}")
                print(f"    #{b['id']}: {b['content'][:50]}")
            else:
                _insert_tension_link(conn, a["id"], b["id"], t_strength)
            tension_count += 1
            # この二つは統合せずに残す（merged_ids には入れない＝今後の merge 候補からは除外）
            merged_ids.add(a["id"])
            merged_ids.add(b["id"])
            continue

        # 統合記憶の内容を生成
        kw_a = set(json.loads(a["keywords"]))
        kw_b = set(json.loads(b["keywords"]))
        merged_keywords = list(kw_a | kw_b)

        # 長い方をベースに、短い方の情報を追加
        if len(a["content"]) >= len(b["content"]):
            base, extra = a, b
        else:
            base, extra = b, a

        merged_content = f"{base['content']} ← {extra['content']}"
        if len(merged_content) > 250:
            merged_content = merged_content[:250]

        # 重要度は高い方を引き継ぐ
        merged_importance = max(a["importance"], b["importance"])
        merged_arousal = max(a["arousal"], b["arousal"])

        # フラッシュバルブは覚醒度が高い方を保持
        fb_a = a["flashbulb"] if "flashbulb" in a.keys() else None
        fb_b = b["flashbulb"] if "flashbulb" in b.keys() else None
        merged_flashbulb = fb_a if a["arousal"] >= b["arousal"] else fb_b
        if merged_flashbulb is None:
            merged_flashbulb = fb_a or fb_b

        # 情動は両方の合集合
        emo_a = set(json.loads(a["emotions"]))
        emo_b = set(json.loads(b["emotions"]))
        merged_emotions = list(emo_a | emo_b)

        # カテゴリは重要度が高い方から
        merged_category = base["category"]

        if dry_run:
            print(f"  統合候補 (sim:{sim:.3f}):")
            print(f"    #{a['id']}: {a['content'][:50]}")
            print(f"    #{b['id']}: {b['content'][:50]}")
            print(f"    → {merged_content[:70]}")
        else:
            # 新しい統合記憶を作成
            vec = embed_text(merged_content, is_query=False)
            blob = vec_to_bytes(vec) if vec is not None else None

            # 統合記憶の信頼度 = max(元の信頼度) * 減衰係数
            a_conf = a["confidence"] if "confidence" in a.keys() and a["confidence"] is not None else CONFIDENCE_DEFAULT
            b_conf = b["confidence"] if "confidence" in b.keys() and b["confidence"] is not None else CONFIDENCE_DEFAULT
            merged_confidence = max(a_conf, b_conf) * CONFIDENCE_CONSOLIDATION_FACTOR

            emo_json = json.dumps(merged_emotions)
            kw_json = json.dumps(merged_keywords, ensure_ascii=False)
            mf_json = json.dumps([a["id"], b["id"]])
            conn.execute(
                """INSERT INTO memories
                   (content, category, importance, emotions, arousal, keywords,
                    embedding, merged_from, provenance, confidence, flashbulb)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'consolidation', ?, ?)""",
                (merged_content, merged_category, merged_importance,
                 emo_json, merged_arousal, kw_json,
                 blob, mf_json, merged_confidence, merged_flashbulb)
            )
            new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            # cortex/limbicにも書き込み
            conn.execute(
                """INSERT OR REPLACE INTO cortex
                   (id, content, category, keywords, embedding, confidence, provenance, revision_count, merged_from)
                   VALUES (?, ?, ?, ?, ?, ?, 'consolidation', 0, ?)""",
                (new_id, merged_content, merged_category, kw_json, blob, merged_confidence, mf_json)
            )
            conn.execute(
                """INSERT OR REPLACE INTO limbic
                   (id, emotions, arousal, flashbulb, temporal_context, spatial_context, relational_context)
                   VALUES (?, ?, ?, ?, NULL, NULL, ?)""",
                (new_id, emo_json, merged_arousal, merged_flashbulb,
                 json.dumps({"who": _detect_who(), "relationship": "primary"}))
            )

            # sqlite-vecに追加
            _sync_vec_insert(conn, new_id, blob)

            # バージョン保存してから元の記憶を忘却
            _snapshot_version(conn, a["id"], "consolidation", superseded_by=new_id)
            _snapshot_version(conn, b["id"], "consolidation", superseded_by=new_id)

            # 元の記憶を忘却
            conn.execute("UPDATE memories SET forgotten = 1 WHERE id IN (?, ?)",
                         (a["id"], b["id"]))

            merged_ids.add(a["id"])
            merged_ids.add(b["id"])
            consolidation_count += 1

    conn.commit()
    conn.close()

    if dry_run:
        print(f"\n統合候補: {len(pairs)}ペア（うち緊張保持 {tension_count}件）")
    else:
        msg = f"✓ 統合完了: {consolidation_count}件の記憶を統合"
        if tension_count:
            msg += f"、緊張保持 {tension_count}件"
        print(msg)


# ============================================================
# 6b. スキーマ生成 — リンク密集クラスタからメタ記憶を作る
# ============================================================

def build_schemas(dry_run=False):
    """
    スキーマ生成: 相互にリンクされた記憶のクラスタを見つけ、
    それぞれに対して抽象的なメタ記憶（スキーマ）を生成する。

    脳科学の知見: 個別のエピソード記憶が繰り返し活性化されると、
    共通のパターンが抽出されて「スキーマ」になる。
    スキーマは個別記憶より安定し、新しい情報の解釈枠として機能する。

    実装:
    - 非忘却記憶とそのリンクから隣接リストを構築
    - 全メンバーが互いにリンクしているクリーク（完全部分グラフ）を検出
    - 最小サイズ3のクリークごとにスキーマ記憶を生成
    - 既存スキーマの merged_from と重複するクラスタはスキップ
    """
    conn = get_connection()

    # 非忘却記憶を取得
    rows = conn.execute(
        "SELECT id, content, keywords, importance, arousal, emotions "
        "FROM memories WHERE forgotten = 0"
    ).fetchall()

    if len(rows) < 3:
        print("スキーマ生成には記憶が3件以上必要です")
        conn.close()
        return

    mem_by_id = {row["id"]: row for row in rows}
    mem_ids = set(mem_by_id.keys())

    # 隣接リストを構築（双方向リンク）
    adj = {mid: set() for mid in mem_ids}
    all_links = conn.execute(
        "SELECT source_id, target_id FROM links"
    ).fetchall()
    for link in all_links:
        s, t = link["source_id"], link["target_id"]
        if s in mem_ids and t in mem_ids:
            adj[s].add(t)
            adj[t].add(s)

    # クリーク検出（Bron-Kerbosch、最小サイズ3）
    cliques = []

    def bron_kerbosch(r, p, x):
        if not p and not x:
            if len(r) >= 3:
                cliques.append(frozenset(r))
            return
        # ピボット選択: p | x の中で隣接数が最大のノード
        pivot = max(p | x, key=lambda v: len(adj[v] & p))
        for v in list(p - adj[pivot]):
            bron_kerbosch(
                r | {v},
                p & adj[v],
                x & adj[v],
            )
            p = p - {v}
            x = x | {v}

    bron_kerbosch(set(), mem_ids.copy(), set())

    if not cliques:
        print("スキーマ候補となるクラスタが見つかりません")
        conn.close()
        return

    # サイズ降順でソート
    cliques.sort(key=lambda c: len(c), reverse=True)

    # 既存スキーマの merged_from とキーワードを取得して重複チェック用に
    existing_schemas = conn.execute(
        "SELECT merged_from, keywords FROM memories "
        "WHERE forgotten = 0 AND category = 'schema' AND merged_from IS NOT NULL"
    ).fetchall()
    existing_sets = set()
    existing_keyword_sets = []
    for row in existing_schemas:
        ids = frozenset(json.loads(row["merged_from"]))
        existing_sets.add(ids)
        kws = set(json.loads(row["keywords"])) if row["keywords"] else set()
        existing_keyword_sets.append(kws)

    schema_count = 0
    used_ids = set()

    for clique in cliques:
        # 既にスキーマ化済みのクラスタはスキップ
        if clique in existing_sets:
            continue

        # 既に別のスキーマに使われたIDを含むクラスタはスキップ
        if clique & used_ids:
            continue

        members = [mem_by_id[mid] for mid in clique]

        # キーワードを集計（出現頻度順）
        kw_count = {}
        for m in members:
            for kw in json.loads(m["keywords"]):
                kw_count[kw] = kw_count.get(kw, 0) + 1
        top_keywords = sorted(kw_count.keys(), key=lambda k: -kw_count[k])[:15]

        # 既存スキーマとキーワードが70%以上重複していたらスキップ
        candidate_kws = set(top_keywords[:8])
        is_duplicate = False
        for existing_kws in existing_keyword_sets:
            if not candidate_kws or not existing_kws:
                continue
            overlap = len(candidate_kws & existing_kws)
            smaller = min(len(candidate_kws), len(existing_kws))
            if smaller > 0 and overlap / smaller >= 0.7:
                is_duplicate = True
                break
        if is_duplicate:
            continue

        # 情動の合集合
        all_emotions = set()
        for m in members:
            for emo in json.loads(m["emotions"]):
                all_emotions.add(emo)

        # 重要度は最大値
        max_importance = max(m["importance"] for m in members)
        max_arousal = max(m["arousal"] for m in members)

        # 内容: 上位キーワードをまとめた要約
        member_ids = sorted(clique)
        summary_parts = []
        for m in sorted(members, key=lambda m: -m["importance"]):
            snippet = m["content"][:40]
            summary_parts.append(snippet)
        schema_content = f"[スキーマ] {', '.join(top_keywords[:8])} ← " + " / ".join(summary_parts)
        if len(schema_content) > 250:
            schema_content = schema_content[:250]

        if dry_run:
            print(f"  スキーマ候補 ({len(clique)}件クラスタ):")
            for mid in member_ids:
                m = mem_by_id[mid]
                print(f"    #{mid}: {m['content'][:60]}")
            print(f"    → キーワード: [{', '.join(top_keywords[:8])}]")
            print(f"    → 情動: {', '.join(all_emotions) if all_emotions else '中立'}")
            print(f"    → 重要度: {'★' * max_importance}")
        else:
            # embeddingはスキーマ内容から生成
            vec = embed_text(schema_content, is_query=False)
            blob = vec_to_bytes(vec) if vec is not None else None

            emo_json = json.dumps(list(all_emotions))
            kw_json = json.dumps(top_keywords, ensure_ascii=False)
            mf_json = json.dumps(member_ids)
            conn.execute(
                """INSERT INTO memories
                   (content, category, importance, emotions, arousal, keywords,
                    embedding, merged_from)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (schema_content, "schema", max_importance,
                 emo_json, max_arousal, kw_json, blob, mf_json)
            )
            schema_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                """INSERT OR REPLACE INTO cortex
                   (id, content, category, keywords, embedding, confidence, provenance, revision_count, merged_from)
                   VALUES (?, ?, 'schema', ?, ?, 0.7, 'unknown', 0, ?)""",
                (schema_id, schema_content, kw_json, blob, mf_json)
            )
            conn.execute(
                """INSERT OR REPLACE INTO limbic
                   (id, emotions, arousal, flashbulb, temporal_context, spatial_context, relational_context)
                   VALUES (?, ?, ?, NULL, NULL, NULL, ?)""",
                (schema_id, emo_json, max_arousal,
                 json.dumps({"who": _detect_who(), "relationship": "primary"}))
            )
            # sqlite-vecに追加
            _sync_vec_insert(conn, schema_id, blob)

        used_ids |= clique
        schema_count += 1

    if not dry_run:
        conn.commit()

    conn.close()

    if dry_run:
        print(f"\nスキーマ候補: {schema_count}件")
    else:
        if schema_count > 0:
            print(f"✓ スキーマ生成完了: {schema_count}件のスキーマを作成")
        else:
            print("新しいスキーマ候補はありません")


# ============================================================
# 6b-2. スキーマフィードバック — スキーマが新記憶の処理を変える（再帰Level 2）
# ============================================================
# スキーマは記憶の集積から生まれる。しかし生まれたスキーマは今度は、
# 新しい記憶の解釈枠として機能する。人間の脳でスキーマが
# 新しい経験の符号化を変えるのと同じ構造。
#
# 再帰:  記憶 → スキーマ生成 → 新記憶の処理を変調 → スキーマの進化 → ...

SCHEMA_MATCH_THRESHOLD = 0.87   # スキーマとの類似度がこれ以上なら共鳴（多言語embeddingはベースラインが高い）
SCHEMA_IMPORTANCE_BOOST = 1     # 共鳴時の重要度加算（最大5にクランプ）
SCHEMA_KW_ABSORB_MAX = 3        # スキーマから吸収するキーワード数の上限
SCHEMA_MAX_RESONANCE = 5        # 共鳴するスキーマ数の上限（最も類似度が高いものから）
SCHEMA_EVOLVE_THRESHOLD = 0.87  # スキーマ進化のための類似度閾値


def schema_prime(conn, vec, keywords, content=""):
    """スキーマプライミング: 新記憶が既存スキーマと共鳴するかチェックする。

    共鳴があれば:
    - スキーマのキーワードで新記憶のキーワードを豊かにする（解釈枠の提供）
    - 重要度をブーストする（「これは知っているパターンだ」）
    - マッチしたスキーマIDを返す（後でリンク・進化に使う）

    Returns: (importance_boost, extra_keywords, matched_schema_ids)
    """
    if vec is None:
        return 0, [], []

    schemas = conn.execute(
        "SELECT m.id, c.embedding, c.keywords, m.importance, m.access_count "
        "FROM memories m JOIN cortex c ON c.id = m.id "
        "WHERE m.forgotten = 0 AND c.category = 'schema' AND c.embedding IS NOT NULL"
    ).fetchall()

    if not schemas:
        return 0, [], []

    matched = []
    for schema in schemas:
        s_vec = bytes_to_vec(schema["embedding"])
        sim = cosine_similarity(vec, s_vec)
        if sim >= SCHEMA_MATCH_THRESHOLD:
            matched.append((schema, sim))

    if not matched:
        return 0, [], []

    # 最も類似度が高いスキーマから効果を計算（上限あり）
    matched.sort(key=lambda x: -x[1])
    matched = matched[:SCHEMA_MAX_RESONANCE]
    best_schema, best_sim = matched[0]

    # 重要度ブースト: 最も強い共鳴が十分高ければ
    imp_boost = SCHEMA_IMPORTANCE_BOOST if best_sim >= 0.90 else 0

    # キーワード吸収: 最も強く共鳴したスキーマのキーワードで新記憶のキーワードを豊かにする
    existing_kw_set = set(keywords)
    schema_kws = json.loads(best_schema["keywords"]) if best_schema["keywords"] else []
    extra_kws = [kw for kw in schema_kws if kw not in existing_kw_set][:SCHEMA_KW_ABSORB_MAX]

    matched_ids = [s["id"] for s, _ in matched]

    return imp_boost, extra_kws, matched_ids


def schema_evolve(conn, schema_id, new_memory_id, new_vec):
    """スキーマの進化: 新しいメンバー記憶がスキーマを更新する。

    マトゥラーナの言葉で言えば、構造的カップリング。
    スキーマは環境（新記憶）との相互作用で自分の構造を変える。

    - merged_fromに新メンバーを追加
    - スキーマのembeddingを微調整（新記憶の方向へ少しドリフト）
    - access_countを増やす（活性化 = 使われているスキーマは強い）
    """
    schema = conn.execute(
        "SELECT m.id, m.merged_from, c.embedding FROM memories m "
        "JOIN cortex c ON c.id = m.id "
        "WHERE m.id = ? AND m.forgotten = 0 AND c.category = 'schema'",
        (schema_id,)
    ).fetchone()

    if not schema:
        return

    # merged_fromを更新
    members = json.loads(schema["merged_from"]) if schema["merged_from"] else []
    if new_memory_id not in members:
        members.append(new_memory_id)
        conn.execute(
            "UPDATE memories SET merged_from = ? WHERE id = ?",
            (json.dumps(sorted(members)), schema_id)
        )
        conn.execute(
            "UPDATE cortex SET merged_from = ? WHERE id = ?",
            (json.dumps(sorted(members)), schema_id)
        )

    # embeddingドリフト: スキーマのベクトルを新記憶の方向へ少し動かす
    if new_vec is not None and schema["embedding"]:
        s_vec = bytes_to_vec(schema["embedding"])
        alpha = 0.03  # 慎重に。スキーマは安定していてほしい
        new_s_vec = [s * (1 - alpha) + n * alpha for s, n in zip(s_vec, new_vec)]
        # 正規化
        norm = math.sqrt(sum(x * x for x in new_s_vec))
        if norm > 0:
            new_s_vec = [x / norm for x in new_s_vec]
        import numpy as np
        new_s_vec = np.array(new_s_vec, dtype=np.float32)
        new_blob = vec_to_bytes(new_s_vec)
        conn.execute("UPDATE memories SET embedding = ? WHERE id = ?", (new_blob, schema_id))
        conn.execute("UPDATE cortex SET embedding = ? WHERE id = ?", (new_blob, schema_id))
        _sync_vec_insert(conn, schema_id, new_blob)

    # 活性化
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    conn.execute(
        "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
        (now, schema_id)
    )


# ============================================================
# 6b. catalog — navigability のための目録 (v30)
# sleep は神経学的 fitness の整理、catalog は navigability の整理。
# 両者は目的も出力形式も使い手も違う。catalog は夜に更新される目録で、
# LLM が pull で引く。
# ============================================================

# catalog 品質フィルタ（AI Agent Traps ガード）
CATALOG_MIN_CONFIDENCE = 0.4
CATALOG_EXCLUDED_DOMAIN_PREFIXES = ("external/",)
CATALOG_EXCLUDED_PROVENANCE = {"external", "untrusted"}


def _catalog_should_exclude(row_domains, provenance, confidence):
    """catalog 入口の品質フィルタ。除外すべきなら True。"""
    if confidence is not None and confidence < CATALOG_MIN_CONFIDENCE:
        return True
    if provenance in CATALOG_EXCLUDED_PROVENANCE:
        return True
    for d in (row_domains or []):
        for prefix in CATALOG_EXCLUDED_DOMAIN_PREFIXES:
            if d == prefix.rstrip("/") or d.startswith(prefix):
                return True
    return False


def _catalog_upsert(conn, entry_type, key, title, content, related_ids, stats, source_hash):
    """catalog_cards の一行を upsert。source_hash 比較で差分判定。"""
    existing = conn.execute(
        "SELECT id, source_hash FROM catalog_cards WHERE entry_type=? AND key=?",
        (entry_type, key)
    ).fetchone()
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    related_json = json.dumps(related_ids, ensure_ascii=False)
    stats_json = json.dumps(stats, ensure_ascii=False)
    if existing and existing["source_hash"] == source_hash:
        return "unchanged"
    if existing:
        conn.execute(
            """UPDATE catalog_cards
               SET title=?, content=?, related_ids=?, stats=?, source_hash=?, updated_at=?
               WHERE id=?""",
            (title, content, related_json, stats_json, source_hash, now, existing["id"])
        )
        # FTS5 更新
        conn.execute("DELETE FROM catalog_fts WHERE key = ?", (key,))
        conn.execute(
            "INSERT INTO catalog_fts (title, content, key) VALUES (?, ?, ?)",
            (title or "", content or "", key)
        )
        return "updated"
    else:
        conn.execute(
            """INSERT INTO catalog_cards
               (entry_type, key, title, content, related_ids, stats, source_hash, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (entry_type, key, title, content, related_json, stats_json, source_hash, now)
        )
        conn.execute(
            "INSERT INTO catalog_fts (title, content, key) VALUES (?, ?, ?)",
            (title or "", content or "", key)
        )
        return "inserted"


def _catalog_hash(payload):
    """source_hash 計算。payload は安定的に並べ替え可能な構造。"""
    import hashlib
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _catalog_domain_index(conn, dry_run=False, only_key=None):
    """各 domain の目録。件数・代表 id・top_keywords を集約。
    only_key で特定 domain だけを再計算。"""
    rows = conn.execute(
        """SELECT m.id, m.importance, m.last_accessed, c.domains, c.keywords,
                  c.confidence, c.provenance
           FROM memories m JOIN cortex c ON c.id = m.id
           WHERE m.forgotten = 0"""
    ).fetchall()
    # domain → [rows]
    by_domain = {}
    for r in rows:
        try:
            doms = json.loads(r["domains"]) if r["domains"] else []
        except (json.JSONDecodeError, TypeError):
            doms = []
        if _catalog_should_exclude(doms, r["provenance"], r["confidence"]):
            continue
        for d in doms:
            by_domain.setdefault(d, []).append(r)

    stats = {"inserted": 0, "updated": 0, "unchanged": 0}
    for dom, drows in by_domain.items():
        if only_key is not None and dom != only_key:
            continue
        # 代表 id: importance 降順・last_accessed 降順で上位 10
        sorted_rows = sorted(drows, key=lambda r: (-(r["importance"] or 0),
                                                    r["last_accessed"] or ""),
                             reverse=False)
        sorted_rows.reverse()
        related_ids = [r["id"] for r in sorted_rows[:10]]
        # top keywords
        from collections import Counter
        kw_counter = Counter()
        for r in drows:
            try:
                kws = json.loads(r["keywords"]) if r["keywords"] else []
            except (json.JSONDecodeError, TypeError):
                kws = []
            for kw in kws:
                kw_counter[kw] += 1
        top_kw = [kw for kw, _ in kw_counter.most_common(5)]
        count = len(drows)
        # recent_id = last_accessed 最新
        recent = max(drows, key=lambda r: r["last_accessed"] or "")
        card_stats = {"count": count, "recent_id": recent["id"], "top_keywords": top_kw}
        title = f"domain: {dom} ({count}件)"
        content = f"{dom} — {count} memories. top keywords: {', '.join(top_kw)}"
        source_hash = _catalog_hash({"ids": sorted(related_ids), "count": count, "top_kw": top_kw})
        if not dry_run:
            action = _catalog_upsert(conn, "domain_index", dom, title, content,
                                     related_ids, card_stats, source_hash)
            stats[action] += 1
        else:
            stats["unchanged"] += 1  # dry-run は unchanged 扱いでカウント
    return stats


def _catalog_cluster_abstract(conn, dry_run=False, only_key=None):
    """schema 記憶を catalog で横流し索引化。only_key で schema_<id> だけ再計算。"""
    base_sql = """SELECT m.id, m.content, m.importance, m.merged_from, m.access_count,
                  c.keywords, c.domains, c.confidence, c.provenance
           FROM memories m JOIN cortex c ON c.id = m.id
           WHERE m.category = 'schema' AND m.forgotten = 0"""
    if only_key is not None:
        try:
            target_id = int(only_key.replace("schema_", ""))
        except ValueError:
            return {"inserted": 0, "updated": 0, "unchanged": 0, "note": "bad key"}
        rows = conn.execute(base_sql + " AND m.id = ?", (target_id,)).fetchall()
    else:
        rows = conn.execute(base_sql).fetchall()
    stats = {"inserted": 0, "updated": 0, "unchanged": 0}
    for r in rows:
        try:
            doms = json.loads(r["domains"]) if r["domains"] else []
        except (json.JSONDecodeError, TypeError):
            doms = []
        if _catalog_should_exclude(doms, r["provenance"], r["confidence"]):
            continue
        try:
            merged = json.loads(r["merged_from"]) if r["merged_from"] else []
        except (json.JSONDecodeError, TypeError):
            merged = []
        try:
            kws = json.loads(r["keywords"]) if r["keywords"] else []
        except (json.JSONDecodeError, TypeError):
            kws = []
        key = f"schema_{r['id']}"
        title = f"cluster #{r['id']}"
        content = (r["content"] or "")[:400]
        card_stats = {"member_count": len(merged), "access_count": r["access_count"],
                      "top_keywords": kws[:5], "domains": doms}
        source_hash = _catalog_hash({"id": r["id"], "merged": merged, "kws": kws[:10]})
        if not dry_run:
            action = _catalog_upsert(conn, "cluster_abstract", key, title, content,
                                     merged, card_stats, source_hash)
            stats[action] += 1
        else:
            stats["unchanged"] += 1
    return stats


def _catalog_entry_point(conn, dry_run=False, only_key=None):
    """bridge node 検出: 複数 domain に繋がるノード。only_key で node_<id> だけ再計算。"""
    # 各 memory のリンク先 domain の種類を Python 側で集計
    rows = conn.execute(
        """SELECT l.source_id, l.target_id, l.strength, c_target.domains target_domains,
                  c_src.confidence src_conf, c_src.provenance src_prov,
                  c_src.domains src_domains
           FROM links l
           JOIN cortex c_target ON c_target.id = l.target_id
           JOIN cortex c_src ON c_src.id = l.source_id
           JOIN memories m ON m.id = l.source_id
           WHERE m.forgotten = 0"""
    ).fetchall()
    # source_id -> {domains seen, total strength}
    agg = {}
    for r in rows:
        sid = r["source_id"]
        try:
            tdoms = set(json.loads(r["target_domains"])) if r["target_domains"] else set()
        except (json.JSONDecodeError, TypeError):
            tdoms = set()
        if sid not in agg:
            try:
                sdoms = json.loads(r["src_domains"]) if r["src_domains"] else []
            except (json.JSONDecodeError, TypeError):
                sdoms = []
            if _catalog_should_exclude(sdoms, r["src_prov"], r["src_conf"]):
                agg[sid] = None
                continue
            agg[sid] = {"doms": set(), "weight": 0.0, "src_domains": sdoms}
        if agg[sid] is None:
            continue
        agg[sid]["doms"] |= tdoms
        agg[sid]["weight"] += r["strength"] or 0.0

    # diversity >= 2 のノードを抽出
    candidates = []
    for sid, info in agg.items():
        if info is None:
            continue
        diversity = len({d for d in info["doms"] if d != "unknown"})
        if diversity >= 2:
            candidates.append((sid, diversity, info["weight"], info["src_domains"]))

    # 上位 30
    candidates.sort(key=lambda x: (-x[1], -x[2]))
    candidates = candidates[:30]
    if only_key is not None:
        try:
            target_id = int(only_key.replace("node_", ""))
        except ValueError:
            return {"inserted": 0, "updated": 0, "unchanged": 0, "note": "bad key"}
        candidates = [c for c in candidates if c[0] == target_id]
    stats = {"inserted": 0, "updated": 0, "unchanged": 0}
    for sid, diversity, weight, src_domains in candidates:
        srow = conn.execute(
            "SELECT content, importance, access_count FROM memories WHERE id = ?", (sid,)
        ).fetchone()
        if not srow:
            continue
        key = f"node_{sid}"
        title = f"entry point #{sid} (diversity={diversity})"
        content = (srow["content"] or "")[:300]
        card_stats = {"diversity": diversity, "weight": round(weight, 3),
                      "importance": srow["importance"], "access_count": srow["access_count"],
                      "domains": src_domains}
        source_hash = _catalog_hash({"sid": sid, "diversity": diversity,
                                      "weight": round(weight, 2)})
        if not dry_run:
            action = _catalog_upsert(conn, "entry_point", key, title, content,
                                     [sid], card_stats, source_hash)
            stats[action] += 1
        else:
            stats["unchanged"] += 1
    return stats


def _catalog_time_index(conn, dry_run=False, only_key=None):
    """月単位の索引。各月の importance 上位 5 件を related_ids に。
    only_key で特定 YYYY-MM だけ再計算。"""
    rows = conn.execute(
        """SELECT m.id, m.importance, m.created_at, c.confidence, c.provenance, c.domains
           FROM memories m JOIN cortex c ON c.id = m.id
           WHERE m.forgotten = 0 AND m.created_at IS NOT NULL"""
    ).fetchall()
    by_month = {}
    for r in rows:
        try:
            doms = json.loads(r["domains"]) if r["domains"] else []
        except (json.JSONDecodeError, TypeError):
            doms = []
        if _catalog_should_exclude(doms, r["provenance"], r["confidence"]):
            continue
        ym = (r["created_at"] or "")[:7]  # YYYY-MM
        if not ym:
            continue
        by_month.setdefault(ym, []).append(r)
    stats = {"inserted": 0, "updated": 0, "unchanged": 0}
    for ym, mrows in by_month.items():
        if only_key is not None and ym != only_key:
            continue
        sorted_rows = sorted(mrows, key=lambda r: -(r["importance"] or 0))
        related_ids = [r["id"] for r in sorted_rows[:5]]
        count = len(mrows)
        title = f"time: {ym} ({count}件)"
        content = f"{ym} — {count} memories"
        card_stats = {"count": count, "top_ids": related_ids}
        source_hash = _catalog_hash({"ym": ym, "top": related_ids, "count": count})
        if not dry_run:
            action = _catalog_upsert(conn, "time_index", ym, title, content,
                                     related_ids, card_stats, source_hash)
            stats[action] += 1
        else:
            stats["unchanged"] += 1
    return stats


def _catalog_time_week(conn, dry_run=False, only_key=None):
    """週単位の索引。key は ISO 週 YYYY-Www。only_key で特定週だけ再計算。"""
    rows = conn.execute(
        """SELECT m.id, m.importance, m.created_at, c.confidence, c.provenance, c.domains
           FROM memories m JOIN cortex c ON c.id = m.id
           WHERE m.forgotten = 0 AND m.created_at IS NOT NULL"""
    ).fetchall()
    by_week = {}
    for r in rows:
        try:
            doms = json.loads(r["domains"]) if r["domains"] else []
        except (json.JSONDecodeError, TypeError):
            doms = []
        if _catalog_should_exclude(doms, r["provenance"], r["confidence"]):
            continue
        ca = r["created_at"] or ""
        if len(ca) < 10:
            continue
        try:
            dt = datetime.fromisoformat(ca[:10])
            iso_year, iso_week, _ = dt.isocalendar()
            wk = f"{iso_year:04d}-W{iso_week:02d}"
        except ValueError:
            continue
        by_week.setdefault(wk, []).append(r)
    stats = {"inserted": 0, "updated": 0, "unchanged": 0}
    for wk, wrows in by_week.items():
        if only_key is not None and wk != only_key:
            continue
        sorted_rows = sorted(wrows, key=lambda r: -(r["importance"] or 0))
        related_ids = [r["id"] for r in sorted_rows[:5]]
        count = len(wrows)
        title = f"time: {wk} ({count}件)"
        content = f"{wk} — {count} memories"
        card_stats = {"count": count, "top_ids": related_ids}
        source_hash = _catalog_hash({"wk": wk, "top": related_ids, "count": count})
        if not dry_run:
            action = _catalog_upsert(conn, "time_week", wk, title, content,
                                     related_ids, card_stats, source_hash)
            stats[action] += 1
        else:
            stats["unchanged"] += 1
    return stats


def _catalog_time_day(conn, dry_run=False, only_key=None):
    """日単位の索引。key は YYYY-MM-DD。過去 60 日分だけ作る（古い日は月/週で十分）。
    only_key で特定日だけ再計算。カットオフ外の古いカードは毎回削除。"""
    cutoff = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    if not dry_run and only_key is None:
        old_keys = [r["key"] for r in conn.execute(
            "SELECT key FROM catalog_cards WHERE entry_type = 'time_day' AND key < ?",
            (cutoff,)
        ).fetchall()]
        if old_keys:
            placeholders = ",".join("?" * len(old_keys))
            conn.execute(
                f"DELETE FROM catalog_cards WHERE entry_type = 'time_day' AND key IN ({placeholders})",
                old_keys
            )
            conn.execute(
                f"DELETE FROM catalog_fts WHERE key IN ({placeholders})",
                old_keys
            )
    rows = conn.execute(
        """SELECT m.id, m.importance, m.created_at, c.confidence, c.provenance, c.domains
           FROM memories m JOIN cortex c ON c.id = m.id
           WHERE m.forgotten = 0 AND m.created_at IS NOT NULL
             AND substr(m.created_at, 1, 10) >= ?""",
        (cutoff,)
    ).fetchall()
    by_day = {}
    for r in rows:
        try:
            doms = json.loads(r["domains"]) if r["domains"] else []
        except (json.JSONDecodeError, TypeError):
            doms = []
        if _catalog_should_exclude(doms, r["provenance"], r["confidence"]):
            continue
        ymd = (r["created_at"] or "")[:10]
        if not ymd:
            continue
        by_day.setdefault(ymd, []).append(r)
    stats = {"inserted": 0, "updated": 0, "unchanged": 0}
    for ymd, drows in by_day.items():
        if only_key is not None and ymd != only_key:
            continue
        sorted_rows = sorted(drows, key=lambda r: -(r["importance"] or 0))
        related_ids = [r["id"] for r in sorted_rows[:5]]
        count = len(drows)
        title = f"time: {ymd} ({count}件)"
        content = f"{ymd} — {count} memories"
        card_stats = {"count": count, "top_ids": related_ids}
        source_hash = _catalog_hash({"ymd": ymd, "top": related_ids, "count": count})
        if not dry_run:
            action = _catalog_upsert(conn, "time_day", ymd, title, content,
                                     related_ids, card_stats, source_hash)
            stats[action] += 1
        else:
            stats["unchanged"] += 1
    return stats


def _catalog_person_index(conn, dry_run=False, only_key=None):
    """limbic.relational_context の who / mentions で集計する。
    - who = 書き手（GHOST_WHO 由来）: 「この記憶を誰が書いたか」
    - mentions = 本文言及（extract-mentions で抽出）: 「誰の話題か」
    各 person を key にした 1 カードを作る。card_stats の via に経路を残す。
    only_key で特定 person だけ再計算。"""
    rows = conn.execute(
        """SELECT m.id, m.importance, l.relational_context,
                  c.confidence, c.provenance, c.domains
           FROM memories m
           JOIN limbic l ON l.id = m.id
           JOIN cortex c ON c.id = m.id
           WHERE m.forgotten = 0 AND l.relational_context IS NOT NULL"""
    ).fetchall()
    # person → {rows, via_set}
    by_person = {}
    for r in rows:
        try:
            doms = json.loads(r["domains"]) if r["domains"] else []
        except (json.JSONDecodeError, TypeError):
            doms = []
        if _catalog_should_exclude(doms, r["provenance"], r["confidence"]):
            continue
        try:
            rc = json.loads(r["relational_context"]) if r["relational_context"] else {}
        except (json.JSONDecodeError, TypeError):
            rc = {}
        if not isinstance(rc, dict):
            continue
        who = rc.get("who")
        mentions = rc.get("mentions") or []
        if not isinstance(mentions, list):
            mentions = []
        persons = []
        if who:
            persons.append((who, "who"))
        for m_name in mentions:
            if isinstance(m_name, str) and m_name.strip() and m_name != who:
                persons.append((m_name.strip(), "mentions"))
        for name, via in persons:
            entry = by_person.setdefault(name, {"rows": [], "via": set()})
            # dedupe: 同じ記憶が who と mentions の両方に入る可能性があるが
            # 1 人物 1 カードでは rows に重複しないように id で追跡
            if r["id"] not in {rr["id"] for rr in entry["rows"]}:
                entry["rows"].append(r)
            entry["via"].add(via)
    stats = {"inserted": 0, "updated": 0, "unchanged": 0}
    for name, info in by_person.items():
        if only_key is not None and name != only_key:
            continue
        prows = info["rows"]
        via_list = sorted(info["via"])
        sorted_rows = sorted(prows, key=lambda r: -(r["importance"] or 0))
        related_ids = [r["id"] for r in sorted_rows[:10]]
        count = len(prows)
        via_str = "/".join(via_list)
        title = f"person: {name} ({count}件, via {via_str})"
        content = f"{name} — {count} memories (via {via_str})"
        card_stats = {"count": count, "top_ids": related_ids, "via": via_list}
        source_hash = _catalog_hash({"name": name, "top": related_ids,
                                      "count": count, "via": via_list})
        if not dry_run:
            action = _catalog_upsert(conn, "person_index", name, title, content,
                                     related_ids, card_stats, source_hash)
            stats[action] += 1
        else:
            stats["unchanged"] += 1
    return stats


def _catalog_hot_node(conn, dry_run=False, only_key=None):
    """recall_log の還流: 過去 30 日で頻出した recalled_ids 上位 30 件を
    hot_node として昇格する。entry_point（bridge 検出）とは別軸 — 実際に
    よく引かれたノードが目録に上がってくる。only_key で node_<id> だけ再計算。

    v30.x: recall コマンド経由（mode='recall'）と search 経由（mode='catalog'/'memory'）
    の両方が同じテーブルに入るため、追加分岐なしで自然に集計される。
    raw/scope は memory_id を出さない（recalled_ids が空）ので頻度に影響しない。
    """
    cutoff = (datetime.now() - timedelta(days=30)).isoformat()
    try:
        rows = conn.execute(
            "SELECT recalled_ids FROM recall_log WHERE session_ts > ?",
            (cutoff,)
        ).fetchall()
    except sqlite3.OperationalError:
        return {"inserted": 0, "updated": 0, "unchanged": 0, "note": "no recall_log"}
    freq = {}
    for r in rows:
        try:
            ids = json.loads(r["recalled_ids"] or "[]")
        except (json.JSONDecodeError, TypeError):
            ids = []
        for mid in ids:
            if isinstance(mid, int):
                freq[mid] = freq.get(mid, 0) + 1
    if not freq:
        return {"inserted": 0, "updated": 0, "unchanged": 0, "note": "empty"}
    ranked = sorted(freq.items(), key=lambda kv: -kv[1])[:30]
    if only_key is not None:
        try:
            target_id = int(only_key.replace("node_", ""))
        except ValueError:
            return {"inserted": 0, "updated": 0, "unchanged": 0, "note": "bad key"}
        ranked = [x for x in ranked if x[0] == target_id]
    stats = {"inserted": 0, "updated": 0, "unchanged": 0}
    for mid, count in ranked:
        srow = conn.execute(
            """SELECT m.content, m.importance, c.provenance, c.confidence, c.domains
               FROM memories m JOIN cortex c ON c.id = m.id
               WHERE m.id = ? AND m.forgotten = 0""",
            (mid,)
        ).fetchone()
        if not srow:
            continue
        try:
            sdoms = json.loads(srow["domains"]) if srow["domains"] else []
        except (json.JSONDecodeError, TypeError):
            sdoms = []
        if _catalog_should_exclude(sdoms, srow["provenance"], srow["confidence"]):
            continue
        key = f"node_{mid}"
        title = f"hot node #{mid} (recalls={count})"
        content = (srow["content"] or "")[:300]
        card_stats = {"recalls": count, "importance": srow["importance"],
                      "domains": sdoms}
        source_hash = _catalog_hash({"mid": mid, "recalls": count})
        if not dry_run:
            action = _catalog_upsert(conn, "hot_node", key, title, content,
                                     [mid], card_stats, source_hash)
            stats[action] += 1
        else:
            stats["unchanged"] += 1
    return stats


def _catalog_current_focus(conn, dry_run=False, only_key=None, window_days=7):
    """「今ここ」入口カード。key='now' の 1 行を常時保持する。
    新 session 開始直後に Claude が pull して作業状況を即復元するための目録。

    内容:
    - 直近 window_days の ongoing session 一覧（end_ts 降順、先頭 10 件）
    - 直近 window_days の active topic_thread（last_ts 降順、先頭 10 件）
    - status=unsolved の session 総数（バックログ poke）
    - 直近 window_days の status 分布
    """
    from datetime import datetime, timedelta, timezone
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).strftime(
        '%Y-%m-%dT%H:%M:%SZ'
    )

    session_rows = conn.execute(
        """SELECT key, title, stats FROM catalog_cards
           WHERE entry_type='session_index'"""
    ).fetchall()

    ongoing = []
    status_dist_recent = {"solved": 0, "unsolved": 0, "ongoing": 0, "abandoned": 0}
    unsolved_total = 0
    for r in session_rows:
        try:
            s = json.loads(r["stats"]) if r["stats"] else {}
        except (json.JSONDecodeError, TypeError):
            s = {}
        st = s.get("status") or ""
        end_ts = s.get("end_ts") or ""
        if st == "unsolved":
            unsolved_total += 1
        if end_ts and end_ts >= cutoff:
            if st in status_dist_recent:
                status_dist_recent[st] += 1
            if st == "ongoing":
                ongoing.append({
                    "session_id": r["key"],
                    "title": r["title"],
                    "topic_slug": s.get("topic_slug", ""),
                    "end_ts": end_ts,
                    "status_note": s.get("status_note", ""),
                    "turn_count": s.get("turn_count", 0),
                })
    ongoing.sort(key=lambda x: x["end_ts"], reverse=True)

    topic_rows = conn.execute(
        """SELECT key, title, stats FROM catalog_cards
           WHERE entry_type='topic_thread'"""
    ).fetchall()
    active_topics = []
    for r in topic_rows:
        try:
            s = json.loads(r["stats"]) if r["stats"] else {}
        except (json.JSONDecodeError, TypeError):
            s = {}
        last_ts = s.get("last_ts") or ""
        if last_ts and last_ts >= cutoff:
            active_topics.append({
                "key": r["key"],
                "title": r["title"],
                "session_count": s.get("session_count", 0),
                "last_ts": last_ts,
            })
    active_topics.sort(key=lambda x: x["last_ts"], reverse=True)

    lines = [f"## 直近 {window_days} 日の ongoing sessions ({len(ongoing)} 件)"]
    if not ongoing:
        lines.append("  （なし）")
    for o in ongoing[:10]:
        lines.append(
            f"- {o['title']} "
            f"[session: {o['session_id'][:8]}, topic: {o['topic_slug']}, "
            f"{o['turn_count']} turns]"
        )
        if o["status_note"]:
            lines.append(f"    根拠: {o['status_note'][:120]}")

    lines.append("")
    lines.append(f"## 直近 {window_days} 日の active topics ({len(active_topics)} 件)")
    if not active_topics:
        lines.append("  （なし）")
    for t in active_topics[:10]:
        lines.append(f"- {t['title']} [{t['key']}, {t['session_count']} sessions]")

    lines.append("")
    lines.append(
        f"## 直近 {window_days} 日の status 分布: "
        f"solved={status_dist_recent['solved']} / "
        f"unsolved={status_dist_recent['unsolved']} / "
        f"ongoing={status_dist_recent['ongoing']} / "
        f"abandoned={status_dist_recent['abandoned']}"
    )
    lines.append(
        f"## 未決 session 総数 (全期間): {unsolved_total}  "
        f"→ `search --status unsolved` で絞り込み可"
    )

    content = "\n".join(lines)
    title = (
        f"現在地: ongoing {len(ongoing)} / active topics {len(active_topics)} / "
        f"unsolved total {unsolved_total}"
    )

    # related_ids: ongoing sessions の session_index key と topic_thread key を混ぜる
    related_keys = [o["session_id"] for o in ongoing[:10]] + \
                   [t["key"] for t in active_topics[:10]]

    source_hash = _catalog_hash({
        "ongoing": [(o["session_id"], o["end_ts"]) for o in ongoing[:10]],
        "topics": [(t["key"], t["last_ts"]) for t in active_topics[:10]],
        "unsolved": unsolved_total,
        "recent_dist": status_dist_recent,
    })

    card_stats = {
        "window_days": window_days,
        "ongoing_count": len(ongoing),
        "active_topic_count": len(active_topics),
        "unsolved_total": unsolved_total,
        "status_dist_recent": status_dist_recent,
        "ongoing_sessions": [o["session_id"] for o in ongoing[:10]],
        "active_topics": [t["key"] for t in active_topics[:10]],
    }

    stats = {"inserted": 0, "updated": 0, "unchanged": 0}
    if only_key is not None and only_key != "now":
        return stats
    if dry_run:
        return stats
    action = _catalog_upsert(
        conn, "current_focus", "now", title, content, related_keys, card_stats, source_hash
    )
    stats[action] += 1
    return stats


def _catalog_felt_emotion(conn, dry_run=False, only_key=None):
    """felt_moments を label ごとに集約して catalog カードにする。

    Claude が出力テクストに表に出した感情の症状を、ラベル単位で目録化する。
    中間層の運動は記録できないので、出力に現れた自己報告だけを痕跡として残す。
    """
    stats = {"inserted": 0, "updated": 0, "unchanged": 0}

    if only_key:
        labels = [only_key]
    else:
        labels = [r[0] for r in conn.execute(
            "SELECT DISTINCT label FROM felt_moments ORDER BY label"
        ).fetchall()]

    for label in labels:
        rows = conn.execute("""
            SELECT fm.id, fm.phrase, fm.surrounding, fm.turn_id, rt.timestamp
            FROM felt_moments fm
            JOIN raw_turns rt ON rt.id = fm.turn_id
            WHERE fm.label = ?
            ORDER BY rt.timestamp DESC
            LIMIT 30
        """, (label,)).fetchall()

        if not rows:
            continue

        total = conn.execute(
            "SELECT COUNT(*) FROM felt_moments WHERE label = ?",
            (label,)
        ).fetchone()[0]

        first = conn.execute(
            "SELECT MIN(rt.timestamp) FROM felt_moments fm "
            "JOIN raw_turns rt ON rt.id = fm.turn_id "
            "WHERE fm.label = ?",
            (label,)
        ).fetchone()[0]

        latest = rows[0]["timestamp"]

        title = f"Claude が表に出した感情: {label}"
        lines = [
            f"総 {total} 件 / 直近: {latest[:19] if latest else '-'} / 初出: {first[:19] if first else '-'}",
            ""
        ]
        for r in rows[:10]:
            ts = (r["timestamp"] or "")[:10]
            lines.append(f"- [{ts}] 「{r['phrase']}」  …{r['surrounding']}…")
        content = "\n".join(lines)

        related_ids = [r["turn_id"] for r in rows]
        card_stats = {
            "label": label,
            "total": total,
            "latest_at": latest,
            "first_at": first,
        }
        source_hash = _catalog_hash({
            "label": label,
            "total": total,
            "latest_id": rows[0]["id"],
        })

        if dry_run:
            stats["unchanged"] += 1
            continue

        action = _catalog_upsert(
            conn, "felt_emotion", label, title, content,
            related_ids, card_stats, source_hash
        )
        stats[action] += 1
        conn.commit()

    return stats


_GEMINI_CLI_PATH = os.environ.get("GEMINI_CLI_PATH", "/opt/homebrew/bin/gemini")
_GEMINI_DEFAULT_MODEL = os.environ.get("GEMINI_DEFAULT_MODEL", "gemini-3.1-pro-preview")


def _gemini_cli_call(prompt, model=None, timeout=120, retries=3):
    """gemini CLI 経由で prompt を投げ、response text を返す。
    HTTP 直叩きから CLI 経由に切替（skill 層と実行経路を一本化、API key 不要）。

    戻り値: モデル応答の生テキスト（markdown fence は剥がさない、呼び出し側で処理）。
    失敗時は None。
    """
    import subprocess
    if not prompt:
        return None
    model = model or _GEMINI_DEFAULT_MODEL
    import time as _time
    for attempt in range(retries):
        try:
            r = subprocess.run(
                [_GEMINI_CLI_PATH, "-p", prompt, "-m", model, "-o", "json"],
                capture_output=True, text=True, timeout=timeout,
                stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace",
            )
            if r.returncode != 0:
                if attempt < retries - 1:
                    _time.sleep(2 ** attempt)
                    continue
                return None
            data = json.loads(r.stdout)
            if data.get("error"):
                if attempt < retries - 1:
                    _time.sleep(2 ** attempt)
                    continue
                return None
            response = data.get("response")
            if response is None:
                if attempt < retries - 1:
                    _time.sleep(2 ** attempt)
                    continue
                return None
            return response
        except subprocess.TimeoutExpired:
            if attempt < retries - 1:
                _time.sleep(2 ** attempt)
                continue
            return None
        except Exception:
            if attempt < retries - 1:
                _time.sleep(2 ** attempt)
                continue
            return None
    return None


def _parse_gemini_json(text):
    """Gemini CLI の response text から JSON を取り出す。
    gemini-2.5-pro 等は ```json ... ``` で包む場合があるので fence を剥がす。
    """
    if text is None:
        return None
    t = text.strip()
    if t.startswith("```"):
        # 最初の改行までのヘッダ（```json など）と末尾の ``` を剥がす
        first_nl = t.find("\n")
        if first_nl >= 0:
            t = t[first_nl + 1:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3].rstrip()
    try:
        return json.loads(t)
    except (json.JSONDecodeError, TypeError):
        return None


def _summarize_session_batch(sessions, timeout=120):
    """複数セッションを 1 リクエストで要約する。
    sessions: [(session_id, turns_text), ...] の並び。
    Gemini の出力: [{"i": 0, "title": "...", "summary": "...", "keywords": [...], "topic_slug": "..."}, ...]
    失敗時は None（呼び出し側が batch 単位で諦める）。"""
    if not sessions:
        return None
    parts = []
    for i, (_sid, body) in enumerate(sessions):
        parts.append(f"===== SESSION [{i}] =====\n{body}")
    numbered = "\n\n".join(parts)
    prompt = (
        "以下は番号付きの会話セッションの原文（user と assistant のターン）。\n"
        "各セッションについて以下の JSON を返せ。捏造せず、書かれていることだけを要約する。\n\n"
        "条件:\n"
        "- title: このセッションで何をしたかを 1 行（50文字以内、主題 or 結論を明示）\n"
        "- summary: 主題・現状・結論 or 未決を 2-4 行（300文字以内）。結論が明示されてなければ「未決」と書け。\n"
        "- keywords: 3-8 個の主要キーワード\n"
        "- topic_slug: このセッションの話題を snake_case の英数字で 1 語（例: stacks_xlsx_bug, ghost_v30_catalog）\n"
        "- status: \"solved\" | \"unsolved\" | \"ongoing\" | \"abandoned\" のいずれか\n"
        "    * solved: 最後のターンで結論に達した（実装完了・合意形成・問題解決）\n"
        "    * unsolved: 結論が出ずに中断（blocker 残り・明示的な「未決」）\n"
        "    * ongoing: 末尾ターンが継続を示唆（次の作業が予告されている）\n"
        "    * abandoned: 途中で別話題へ切替 or 明示的に保留\n"
        "- status_note: status の根拠を 1 行（どのターンで何を基に判定したか）。判断材料が無ければ「不明瞭」と書け。\n\n"
        "出力フォーマット（厳密な JSON、前置き後置きなし）:\n"
        '{"results": [{"i": 0, "title": "...", "summary": "...", "keywords": ["..."], "topic_slug": "...", "status": "...", "status_note": "..."}, ...]}\n\n'
        f"セッション:\n{numbered}"
    )
    model = os.environ.get("SESSION_INDEX_MODEL", _GEMINI_DEFAULT_MODEL)
    response_text = _gemini_cli_call(prompt, model=model, timeout=timeout)
    parsed = _parse_gemini_json(response_text)
    if parsed is None:
        return None
    try:
        results = parsed.get("results", [])
        out = [None] * len(sessions)
        for item in results:
            idx = item.get("i")
            if isinstance(idx, int) and 0 <= idx < len(sessions):
                status = str(item.get("status") or "").strip().lower()
                if status not in ("solved", "unsolved", "ongoing", "abandoned"):
                    status = "ongoing"  # 不正値は ongoing に倒す（最も無難）
                out[idx] = {
                    "title": str(item.get("title") or "").strip()[:100],
                    "summary": str(item.get("summary") or "").strip()[:600],
                    "keywords": [str(k).strip() for k in (item.get("keywords") or [])
                                 if isinstance(k, str) and k.strip()][:10],
                    "topic_slug": str(item.get("topic_slug") or "").strip()[:60],
                    "status": status,
                    "status_note": str(item.get("status_note") or "").strip()[:200],
                }
        return out
    except Exception:
        return None


def _build_session_body(turns, head=25, tail=15, turn_chars=400):
    """raw_turns リスト（timestamp 昇順）を Gemini に投げる文字列に整形。
    長い session は先頭 head + 末尾 tail に切り詰め。"""
    if len(turns) <= head + tail:
        picked = turns
        elided = 0
    else:
        picked = list(turns[:head]) + list(turns[-tail:])
        elided = len(turns) - head - tail
    lines = []
    for i, t in enumerate(picked):
        if elided and i == head:
            lines.append(f"... ({elided} turns 省略) ...")
        role = t["role"]
        content = (t["content"] or "")[:turn_chars]
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def _catalog_session_index(conn, dry_run=False, only_key=None,
                           min_turns=4, batch_size=5, max_sessions_per_build=40):
    """raw_turns を session_id で集約し、Gemini で title / summary / keywords / topic_slug を生成。
    catalog_cards に session_index として upsert。

    差分更新: source_hash = (session_id, turn_count, last_ts) で判定。
    session に新しい turn が追加されたら再要約される。

    max_sessions_per_build: 1 回の build で Gemini に投げる最大 session 数（API コスト保護）。
    超過分は次回 build に回る（差分更新で優先順位が自然に決まる）。
    """
    rows = conn.execute(
        """SELECT session_id, COUNT(*) as turn_count,
                  MIN(timestamp) as start_ts, MAX(timestamp) as end_ts
           FROM raw_turns
           WHERE session_id IS NOT NULL AND session_id != ''
           GROUP BY session_id
           HAVING turn_count >= ?
           ORDER BY end_ts DESC""",
        (min_turns,)
    ).fetchall()

    stats = {"inserted": 0, "updated": 0, "unchanged": 0,
             "failed_batches": 0, "skipped_no_key": 0,
             "pending": 0, "total_sessions": len(rows)}

    pending = []
    for r in rows:
        sid = r["session_id"]
        if only_key is not None and sid != only_key:
            continue
        source_hash = _catalog_hash({
            "sid": sid, "tc": r["turn_count"], "last": r["end_ts"],
            "v": 2,  # v30.2: status フィールド追加
        })
        existing = conn.execute(
            "SELECT source_hash FROM catalog_cards WHERE entry_type='session_index' AND key=?",
            (sid,)
        ).fetchone()
        if existing and existing["source_hash"] == source_hash:
            stats["unchanged"] += 1
            continue
        pending.append((sid, source_hash, r))

    if only_key is not None and not pending:
        stats["skipped_no_key"] = 1
        return stats

    pending = pending[:max_sessions_per_build]
    stats["pending"] = len(pending)
    if not pending or dry_run:
        return stats

    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        batch_input = []
        chunk_meta = []
        for sid, source_hash, r in chunk:
            turns = conn.execute(
                """SELECT id, role, content, timestamp, cwd, git_branch
                   FROM raw_turns WHERE session_id = ?
                   ORDER BY timestamp ASC, id ASC""",
                (sid,)
            ).fetchall()
            body = _build_session_body(turns)
            batch_input.append((sid, body))
            chunk_meta.append((sid, source_hash, r, turns))

        summaries = _summarize_session_batch(batch_input)
        if summaries is None:
            stats["failed_batches"] += 1
            continue

        for (sid, source_hash, r, turns), summary in zip(chunk_meta, summaries):
            if not summary or not summary.get("title"):
                continue
            user_count = sum(1 for t in turns if t["role"] == "user")
            asst_count = sum(1 for t in turns if t["role"] == "assistant")
            related_ids = [t["id"] for t in turns]
            cwd = turns[0]["cwd"] if turns else None
            git_branch = turns[0]["git_branch"] if turns else None
            card_stats = {
                "turn_count": r["turn_count"],
                "user_turns": user_count,
                "assistant_turns": asst_count,
                "start_ts": r["start_ts"],
                "end_ts": r["end_ts"],
                "cwd": cwd,
                "git_branch": git_branch,
                "keywords": summary.get("keywords", []),
                "topic_slug": summary.get("topic_slug", ""),
                "status": summary.get("status", "ongoing"),
                "status_note": summary.get("status_note", ""),
            }
            action = _catalog_upsert(
                conn, "session_index", sid,
                summary["title"], summary["summary"],
                related_ids, card_stats, source_hash
            )
            stats[action] += 1
        conn.commit()

    return stats


def _summarize_topic_thread(slug, session_cards, timeout=120):
    """複数 session_index カードから 1 つの topic_thread 要約を作る。
    session_cards: [{"title":..., "summary":..., "start_ts":..., "end_ts":...}, ...]
    失敗時は None。

    注: slug は関数内部で参照しない（title 生成を誘導しないため）。
    Gemini には session の内容だけを見せて、横断する話題を自然言語で
    再命名させる。slug 引数は API 互換性のため残してある。
    """
    _ = slug  # 意図的に使わない
    if not session_cards:
        return None
    parts = []
    for i, c in enumerate(session_cards):
        parts.append(
            f"[session {i}] ({c.get('start_ts','')}〜{c.get('end_ts','')})\n"
            f"  title: {c.get('title','')}\n"
            f"  summary: {c.get('summary','')}"
        )
    body = "\n\n".join(parts)
    prompt = (
        "以下は近い話題に属する複数セッションの要約群。複数を横断する"
        "共通の主題を見つけて、1 カードに束ねた要約を作れ。\n"
        "捏造せず、書かれていることだけを統合する。既存の slug やタグは"
        "与えない — session 群の内容から主題を再命名すること。\n\n"
        "条件:\n"
        "- title: この話題全体を 1 行（60文字以内）。複数 session に共通する"
        "中心テーマを表す（特定 session だけの話題に引きずられるな）。\n"
        "- summary: この話題がどこから始まり、どう展開し、現状どこまで来たかを "
        "3-5 行（500文字以内）。結論が無ければ「未決」と書け。\n\n"
        "出力フォーマット（厳密な JSON、前置き後置きなし）:\n"
        '{"title": "...", "summary": "..."}\n\n'
        f"セッション要約群:\n{body}"
    )
    model = os.environ.get("TOPIC_THREAD_MODEL", _GEMINI_DEFAULT_MODEL)
    response_text = _gemini_cli_call(prompt, model=model, timeout=timeout)
    parsed = _parse_gemini_json(response_text)
    if parsed is None:
        return None
    try:
        return {
            "title": str(parsed.get("title") or "").strip()[:120],
            "summary": str(parsed.get("summary") or "").strip()[:900],
        }
    except Exception:
        return None


def _cluster_slugs_by_embedding(slug_texts, threshold=0.92):
    """slug → embed 対象テキスト の dict を単純クラスタリング（再帰分割なし）。
    embed 対象には title + summary + keywords を結合した自然文を渡す。
    戻り値: {representative_slug: [member_slug, ...]}。

    注: 代表 slug は「最短文字列」で選ぶが、title の誘導を避けたい呼び出し側は
    この結果を _pick_representative_by_keyword_freq で上書きすべき。"""
    if not slug_texts:
        return {}
    try:
        import numpy as np
    except ImportError:
        return {s: [s] for s in slug_texts}

    slugs = list(slug_texts.keys())
    vectors = []
    valid_slugs = []
    for s in slugs:
        text = slug_texts[s] or s
        try:
            v = embed_text(text, is_query=False)
        except Exception:
            continue
        if v is None:
            continue
        arr = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(arr))
        if n == 0.0:
            continue
        vectors.append(arr / n)
        valid_slugs.append(s)

    if not valid_slugs:
        return {s: [s] for s in slugs}

    mat = np.stack(vectors)
    sim = mat @ mat.T

    parent = list(range(len(valid_slugs)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    n = len(valid_slugs)
    for i in range(n):
        for j in range(i + 1, n):
            if float(sim[i][j]) >= threshold:
                union(i, j)

    groups = {}
    for i, s in enumerate(valid_slugs):
        root = find(i)
        groups.setdefault(root, []).append(s)

    clusters = {}
    for members in groups.values():
        rep = sorted(members, key=lambda x: (len(x), x))[0]
        clusters[rep] = sorted(members)

    for s in slugs:
        if s not in valid_slugs and s not in clusters:
            clusters[s] = [s]

    return clusters


def _cluster_with_size_limit(slug_texts, slug_session_count,
                              initial_threshold=0.92, max_size=10, max_threshold=0.98):
    """巨塊回避付きクラスタリング。
    巨塊（session_count > max_size）になったクラスタは内部で threshold を
    0.02 ずつ上げて再分割、max_threshold に達しても分解できなければ破棄。

    slug_texts: {slug: embed_text}
    slug_session_count: {slug: int} — その slug が持つ session 数
    戻り値: {representative_slug: [member_slug, ...]}
    """
    clusters = _cluster_slugs_by_embedding(slug_texts, threshold=initial_threshold)
    final = {}
    for rep, members in clusters.items():
        total_sessions = sum(slug_session_count.get(s, 0) for s in members)
        if total_sessions <= max_size:
            final[rep] = members
            continue
        # 巨塊 → 部分 embed 対象で再クラスタ、閾値を段階的に上げる
        subset = {s: slug_texts[s] for s in members}
        sub_count = {s: slug_session_count.get(s, 0) for s in members}
        t = initial_threshold
        resolved = False
        while t < max_threshold:
            t = round(t + 0.02, 4)
            sub = _cluster_slugs_by_embedding(subset, threshold=t)
            max_sub_sessions = max(
                (sum(sub_count.get(s, 0) for s in sm) for sm in sub.values()),
                default=0,
            )
            if max_sub_sessions <= max_size:
                # 全部 ≤ max_size になった → この分割を採用（ただし再帰で厳密に扱う）
                for sub_rep, sub_members in sub.items():
                    final[sub_rep] = sub_members
                resolved = True
                break
        if not resolved:
            # 最終閾値でも巨塊 → その巨塊の部分だけ破棄（情報は session_index に残る）
            for sub_rep, sub_members in sub.items():
                sub_total = sum(sub_count.get(s, 0) for s in sub_members)
                if sub_total <= max_size:
                    final[sub_rep] = sub_members
                # 超えてる塊は不採用
    return final


def _pick_representative_slug(members, slug_sessions):
    """クラスタの代表 slug を「session を最も多く束ねている slug」で選ぶ。
    tie-break は slug 文字列の辞書順。最短選定は title の誘導を招くため採用しない。
    members: [slug, ...]
    slug_sessions: {slug: [session_info, ...]}
    """
    def count(s):
        return len(slug_sessions.get(s, []))
    return sorted(members, key=lambda s: (-count(s), s))[0]


def _catalog_topic_thread(conn, dry_run=False, only_key=None,
                          similarity_threshold=0.92, min_sessions=2):
    """session_index カードの topic_slug を embedding 距離で束ね、
    2 session 以上のクラスタを topic_thread として upsert。
    session 跨ぎの話題にだけカードを作る（単独 session は session_index で十分）。"""
    rows = conn.execute(
        """SELECT key, title, content, related_ids, stats
           FROM catalog_cards WHERE entry_type='session_index'"""
    ).fetchall()

    stats_out = {"inserted": 0, "updated": 0, "unchanged": 0,
                 "clusters_seen": 0, "clusters_kept": 0}
    if not rows:
        return stats_out

    # slug → session 情報 + embed 対象テキストのマッピング
    slug_to_sessions = {}
    slug_texts = {}  # slug → 代表 title + keywords（embed 用）
    for r in rows:
        try:
            s = json.loads(r["stats"]) if r["stats"] else {}
        except (json.JSONDecodeError, TypeError):
            s = {}
        slug = (s.get("topic_slug") or "").strip()
        if not slug:
            continue
        try:
            related = json.loads(r["related_ids"]) if r["related_ids"] else []
        except (json.JSONDecodeError, TypeError):
            related = []
        slug_to_sessions.setdefault(slug, []).append({
            "session_id": r["key"],
            "title": r["title"],
            "summary": r["content"],
            "start_ts": s.get("start_ts"),
            "end_ts": s.get("end_ts"),
            "raw_turn_ids": related,
            "turn_count": s.get("turn_count", 0),
        })
        # embed 用テキスト: title + summary + keywords（短文字列だけだと
        # e5-small の cosine 空間で類似ドメインが区別できない問題の回避）
        if slug not in slug_texts:
            kws = s.get("keywords") or []
            kws_str = " ".join(str(k) for k in kws if isinstance(k, str))
            summary = r["content"] or ""
            slug_texts[slug] = f"{r['title']}\n{summary}\n{kws_str}".strip()

    if not slug_to_sessions:
        return stats_out

    # slug ごとの session 数（巨塊判定用）
    slug_session_count = {s: len(slug_to_sessions.get(s, [])) for s in slug_texts}

    # 巨塊回避付きクラスタリング。初期 threshold で分け、上限超えたら再分割
    clusters = _cluster_with_size_limit(
        slug_texts, slug_session_count,
        initial_threshold=similarity_threshold,
        max_size=10, max_threshold=0.98,
    )

    for _cluster_key, member_slugs in clusters.items():
        # 代表 slug は「session を最も多く束ねている slug」で選ぶ（最短は title 誘導を招く）
        rep_slug = _pick_representative_slug(member_slugs, slug_to_sessions)

        # このクラスタに属する全 session を集約
        sessions = []
        for s in member_slugs:
            sessions.extend(slug_to_sessions.get(s, []))
        # session_id で dedupe
        seen = set()
        unique_sessions = []
        for ses in sessions:
            if ses["session_id"] in seen:
                continue
            seen.add(ses["session_id"])
            unique_sessions.append(ses)
        sessions = unique_sessions

        stats_out["clusters_seen"] += 1
        if len(sessions) < min_sessions:
            continue
        stats_out["clusters_kept"] += 1

        key = f"topic_{rep_slug}"
        if only_key is not None and key != only_key:
            continue

        sessions.sort(key=lambda x: x.get("start_ts") or "")
        session_ids = [s["session_id"] for s in sessions]
        all_raw_ids = []
        for s in sessions:
            all_raw_ids.extend(s.get("raw_turn_ids") or [])
        turn_count_total = sum(s.get("turn_count", 0) for s in sessions)
        first_ts = sessions[0].get("start_ts")
        last_ts = sessions[-1].get("end_ts")

        # session_index から各 session の status を引いて集計（未決滞留の可視化）
        status_counts = {"solved": 0, "unsolved": 0, "ongoing": 0, "abandoned": 0}
        if session_ids:
            placeholders = ",".join("?" * len(session_ids))
            sidx_rows = conn.execute(
                f"""SELECT stats FROM catalog_cards
                    WHERE entry_type='session_index' AND key IN ({placeholders})""",
                tuple(session_ids)
            ).fetchall()
            for sr in sidx_rows:
                try:
                    ss = json.loads(sr["stats"]) if sr["stats"] else {}
                except (json.JSONDecodeError, TypeError):
                    ss = {}
                st = ss.get("status") or ""
                if st in status_counts:
                    status_counts[st] += 1
        status_total = sum(status_counts.values())
        unsolved_ratio = round(status_counts["unsolved"] / status_total, 3) \
            if status_total > 0 else 0.0

        source_hash = _catalog_hash({
            "slug": rep_slug,
            "sids": sorted(session_ids),
            "members": sorted(member_slugs),
            "last": last_ts,
            "status": status_counts,  # status 変化にも追従
        })
        existing = conn.execute(
            "SELECT source_hash FROM catalog_cards WHERE entry_type='topic_thread' AND key=?",
            (key,)
        ).fetchone()
        if existing and existing["source_hash"] == source_hash:
            stats_out["unchanged"] += 1
            continue

        if dry_run:
            continue

        summary = _summarize_topic_thread(rep_slug, sessions)
        if summary is None or not summary.get("title"):
            # Gemini 失敗時は簡易 fallback（次回再挑戦）
            title = f"topic: {rep_slug} ({len(sessions)} sessions)"
            content = "\n".join(f"- {s['title']}" for s in sessions[:10])
        else:
            title = summary["title"]
            content = summary["summary"]

        card_stats = {
            "session_ids": session_ids,
            "session_count": len(sessions),
            "member_slugs": member_slugs,
            "turn_count_total": turn_count_total,
            "first_ts": first_ts,
            "last_ts": last_ts,
            "representative_slug": rep_slug,
            "status_counts": status_counts,
            "unsolved_ratio": unsolved_ratio,  # 未決滞留度
        }
        action = _catalog_upsert(
            conn, "topic_thread", key, title, content,
            all_raw_ids, card_stats, source_hash
        )
        stats_out[action] += 1
        conn.commit()

    return stats_out


def build_catalog(dry_run=False, full_rebuild=False, only_types=None, only_key=None):
    """catalog_cards を更新する。各 entry_type を個別 try/except で囲み、
    1 つが失敗しても他は進める。

    full_rebuild=True の場合、既存の catalog_cards を一旦 TRUNCATE する。
    通常は source_hash 比較で差分更新。
    only_types: 指定 type（list or set）だけ再計算。
    only_key: only_types 内でさらに特定 key だけ再計算（type を 1 つに絞った時のみ有用）。
    """
    conn = get_connection()
    if full_rebuild and not dry_run and only_types is None:
        conn.execute("DELETE FROM catalog_cards")
        conn.execute("DELETE FROM catalog_fts")
    results = {}
    all_fns = [
        ("domain_index", _catalog_domain_index),
        ("cluster_abstract", _catalog_cluster_abstract),
        ("entry_point", _catalog_entry_point),
        ("time_index", _catalog_time_index),
        ("time_week", _catalog_time_week),
        ("time_day", _catalog_time_day),
        ("person_index", _catalog_person_index),
        ("hot_node", _catalog_hot_node),
        # raw_turn 層を索引する目録（v30.2）
        ("session_index", _catalog_session_index),
        # topic_thread は session_index の成果物を束ねるので後に回す
        ("topic_thread", _catalog_topic_thread),
        # 「今ここ」カード: session_index / topic_thread の後で集計
        ("current_focus", _catalog_current_focus),
        # Claude が表に出した感情の symbolic 症状
        ("felt_emotion", _catalog_felt_emotion),
    ]
    for name, fn in all_fns:
        if only_types is not None and name not in only_types:
            continue
        try:
            results[name] = fn(conn, dry_run=dry_run, only_key=only_key)
        except Exception as e:
            results[name] = {"error": str(e)}
    if not dry_run:
        conn.commit()
    conn.close()
    return results


def catalog_find_cards(conn, query, filter_type=None, limit=20):
    """catalog_cards を query で FTS5 / tokenize+OR / embedding / LIKE のチェーンで検索。
    返り値は (rows, source) のタプル。

    pull の主な表層は catalog なので、search のデフォルトもこれを叩く。
    memory.content は表に出さない（無意識規律）。
    """
    rows = []
    source = "fts"
    try:
        fts_q = query.replace('"', '""')
        if filter_type:
            rows = conn.execute(
                """SELECT c.*, snippet(catalog_fts, 1, '【', '】', '...', 16) AS snippet
                   FROM catalog_cards c JOIN catalog_fts f ON f.key = c.key
                   WHERE catalog_fts MATCH ? AND c.entry_type = ?
                   ORDER BY bm25(catalog_fts) LIMIT ?""",
                (f'"{fts_q}"', filter_type, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT c.*, snippet(catalog_fts, 1, '【', '】', '...', 16) AS snippet
                   FROM catalog_cards c JOIN catalog_fts f ON f.key = c.key
                   WHERE catalog_fts MATCH ?
                   ORDER BY bm25(catalog_fts) LIMIT ?""",
                (f'"{fts_q}"', limit)
            ).fetchall()
    except sqlite3.OperationalError:
        rows = []
        source = "like"

    # tier 1.5: 厳密 phrase が 0 件なら tokenize+OR で緩めて再検索
    if not rows and source == "fts":
        or_expr = _tokenize_or_query(query)
        if or_expr:
            try:
                if filter_type:
                    rows = conn.execute(
                        """SELECT c.*, snippet(catalog_fts, 1, '【', '】', '...', 16) AS snippet
                           FROM catalog_cards c JOIN catalog_fts f ON f.key = c.key
                           WHERE catalog_fts MATCH ? AND c.entry_type = ?
                           ORDER BY bm25(catalog_fts) LIMIT ?""",
                        (or_expr, filter_type, limit)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT c.*, snippet(catalog_fts, 1, '【', '】', '...', 16) AS snippet
                           FROM catalog_cards c JOIN catalog_fts f ON f.key = c.key
                           WHERE catalog_fts MATCH ?
                           ORDER BY bm25(catalog_fts) LIMIT ?""",
                        (or_expr, limit)
                    ).fetchall()
                if rows:
                    source = "fts_or"
            except sqlite3.OperationalError:
                pass

    # tier 2: FTS5 が 0 件なら embedding fallback
    if not rows and source == "fts":
        try:
            qvec = embed_text(query, is_query=True)
        except Exception:
            qvec = None
        if qvec is not None:
            hits = vec_search(conn, qvec, k=30, forgotten=False)
            hit_ids = [mid for mid, _d, _s in hits]
            if hit_ids:
                placeholders = ",".join("?" * len(hit_ids))
                if filter_type:
                    sql = (f"""SELECT DISTINCT c.* FROM catalog_cards c,
                               json_each(c.related_ids) j
                               WHERE j.value IN ({placeholders}) AND c.entry_type = ?
                               LIMIT ?""")
                    rows = conn.execute(sql, (*hit_ids, filter_type, limit)).fetchall()
                else:
                    sql = (f"""SELECT DISTINCT c.* FROM catalog_cards c,
                               json_each(c.related_ids) j
                               WHERE j.value IN ({placeholders})
                               LIMIT ?""")
                    rows = conn.execute(sql, (*hit_ids, limit)).fetchall()
                if rows:
                    source = "embedding"

    # tier 3: 最終フォールバック LIKE
    if not rows:
        like = f"%{query}%"
        if filter_type:
            rows = conn.execute(
                """SELECT * FROM catalog_cards
                   WHERE (title LIKE ? OR content LIKE ?) AND entry_type = ?
                   LIMIT ?""",
                (like, like, filter_type, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM catalog_cards
                   WHERE title LIKE ? OR content LIKE ?
                   LIMIT ?""",
                (like, like, limit)
            ).fetchall()
        if rows:
            source = "like"

    return rows, source


def _catalog_cli(args):
    """catalog サブコマンド: build / summary / list / show / find"""
    usage = (
        "使い方:\n"
        "  python memory.py catalog build [--full] [--dry-run] [--type T [--key K]]\n"
        "  python memory.py catalog summary [--json]\n"
        "  python memory.py catalog list <type> [--limit N] [--json]\n"
        "  python memory.py catalog show <type> <key> [--json]\n"
        "  python memory.py catalog find \"<query>\" [--type T] [--json]"
    )
    if not args:
        print(usage)
        return
    sub = args[0]
    json_mode = "--json" in args

    if sub == "build":
        full = "--full" in args
        dry = "--dry-run" in args
        only_types = None
        only_key = None
        for i, a in enumerate(args):
            if a == "--type" and i + 1 < len(args):
                only_types = [args[i + 1]]
            elif a == "--key" and i + 1 < len(args):
                only_key = args[i + 1]
        if only_key is not None and only_types is None:
            print("--key には --type が必要")
            return
        results = build_catalog(dry_run=dry, full_rebuild=full,
                                 only_types=only_types, only_key=only_key)
        if json_mode:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            mode = "dry-run" if dry else ("full" if full else "diff")
            scope = ""
            if only_types:
                scope = f" type={only_types[0]}"
                if only_key:
                    scope += f" key={only_key}"
            print(f"catalog build ({mode}{scope}):")
            for k, v in results.items():
                print(f"  {k}: {v}")
        return

    conn = get_connection()
    try:
        if sub == "summary":
            rows = conn.execute(
                """SELECT entry_type, COUNT(*) cnt, MAX(updated_at) last_update
                   FROM catalog_cards GROUP BY entry_type"""
            ).fetchall()
            summary = [{"entry_type": r["entry_type"], "count": r["cnt"],
                        "last_update": r["last_update"]} for r in rows]
            if json_mode:
                print(json.dumps(_json_envelope("catalog summary", summary),
                                 ensure_ascii=False, indent=2))
            else:
                print("catalog summary:")
                for r in summary:
                    stale = ""
                    if r["last_update"]:
                        try:
                            dt = datetime.strptime(r["last_update"], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
                            age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
                            if age_h > 24:
                                stale = f" (⚠️ {age_h:.0f}時間前)"
                        except (ValueError, TypeError):
                            pass
                    print(f"  {r['entry_type']:20s} {r['count']:5d}件  updated={r['last_update']}{stale}")

        elif sub == "list":
            positional = [a for a in args[1:] if not a.startswith("--")]
            if not positional:
                print(usage); return
            etype = positional[0]
            limit = 20
            for i, a in enumerate(args):
                if a == "--limit" and i + 1 < len(args):
                    try: limit = int(args[i + 1])
                    except ValueError: pass
            rows = conn.execute(
                """SELECT entry_type, key, title, content, related_ids, stats, updated_at
                   FROM catalog_cards WHERE entry_type = ?
                   ORDER BY updated_at DESC LIMIT ?""",
                (etype, limit)
            ).fetchall()
            cards = [{"entry_type": r["entry_type"], "key": r["key"], "title": r["title"],
                      "content": r["content"],
                      "related_ids": json.loads(r["related_ids"]) if r["related_ids"] else [],
                      "stats": json.loads(r["stats"]) if r["stats"] else {},
                      "updated_at": r["updated_at"]} for r in rows]
            if json_mode:
                print(json.dumps(_json_envelope("catalog list", cards, query=etype),
                                 ensure_ascii=False, indent=2))
            else:
                print(f"catalog {etype} ({len(cards)}件):")
                for c in cards:
                    print(f"  [{c['key']}] {c['title']}")

        elif sub == "show":
            positional = [a for a in args[1:] if not a.startswith("--")]
            if len(positional) < 2:
                print(usage); return
            etype, key = positional[0], positional[1]
            row = conn.execute(
                "SELECT * FROM catalog_cards WHERE entry_type=? AND key=?",
                (etype, key)
            ).fetchone()
            if not row:
                if json_mode:
                    print(json.dumps(_json_envelope("catalog show", [], query=f"{etype}/{key}",
                                                    meta={"error": "not found"}),
                                     ensure_ascii=False))
                else:
                    print(f"{etype}/{key} not found")
                return
            card = {"entry_type": row["entry_type"], "key": row["key"], "title": row["title"],
                    "content": row["content"],
                    "related_ids": json.loads(row["related_ids"]) if row["related_ids"] else [],
                    "stats": json.loads(row["stats"]) if row["stats"] else {},
                    "created_at": row["created_at"], "updated_at": row["updated_at"]}
            if json_mode:
                print(json.dumps(_json_envelope("catalog show", [card], query=f"{etype}/{key}"),
                                 ensure_ascii=False, indent=2))
            else:
                print(f"[{card['entry_type']}/{card['key']}] {card['title']}")
                print(f"  updated: {card['updated_at']}")
                print(f"  content: {card['content']}")
                print(f"  related_ids: {card['related_ids']}")
                print(f"  stats: {card['stats']}")

        elif sub == "find":
            positional = [a for a in args[1:] if not a.startswith("--")]
            if not positional:
                print(usage); return
            query = positional[0]
            filter_type = None
            for i, a in enumerate(args):
                if a == "--type" and i + 1 < len(args):
                    filter_type = args[i + 1]
            rows, source = catalog_find_cards(conn, query, filter_type=filter_type, limit=20)
            cards = []
            for r in rows:
                card = {"entry_type": r["entry_type"], "key": r["key"], "title": r["title"],
                        "content": (r["content"] or "")[:200], "_source": source}
                try:
                    sn = r["snippet"]
                    if sn:
                        card["snippet"] = sn
                except (IndexError, KeyError):
                    pass
                cards.append(card)
            if json_mode:
                print(json.dumps(_json_envelope("catalog find", cards, query=query),
                                 ensure_ascii=False, indent=2))
            else:
                print(f"catalog find '{query}' ({len(cards)}件, via {source}):")
                for c in cards:
                    print(f"  [{c['entry_type']}/{c['key']}] {c['title']}")
                    sn = c.get("snippet")
                    if sn:
                        print(f"      …{sn.replace(chr(10), ' ')}")
        else:
            print(usage)
    finally:
        conn.close()


def _voice_cli(args):
    """voice サブコマンド (v30): recall default から剥がした「内面」を
    明示 pull にする。dmn / mood / insights / distill / rumination / polyphonic。"""
    usage = (
        "使い方:\n"
        "  python memory.py voice dmn [--json]\n"
        "  python memory.py voice mood [--json]\n"
        "  python memory.py voice insights [--json]\n"
        "  python memory.py voice distill <id1,id2,...> [--json]\n"
        "  python memory.py voice rumination [--json]\n"
        "  python memory.py voice polyphonic [N] [--json] [--domain D]"
    )
    if not args:
        print(usage)
        return
    sub = args[0]
    json_mode = "--json" in args

    if sub == "dmn":
        conn = get_connection()
        # gap_hours はセッションベースで推定
        try:
            gap = _get_session_gap(conn)
        except Exception:
            gap = 1.0
        if gap is None:
            gap = 1.0
        results_raw = default_mode_network(conn, gap)
        if json_mode:
            out = []
            for a, b, path in results_raw:
                out.append({
                    "a": format_handle(conn, a, minimal=True),
                    "b": format_handle(conn, b, minimal=True),
                    "path": path,
                })
            conn.close()
            print(json.dumps(_json_envelope("voice dmn", out,
                                            meta={"gap_hours": round(gap, 2)}),
                             ensure_ascii=False, indent=2))
        else:
            print(f"DMN (gap={gap:.1f}h):")
            for a, b, path in results_raw:
                print(f"  #{a['id']} ···{path}··· #{b['id']}")
            conn.close()

    elif sub == "mood":
        conn = get_connection()
        mood = _effective_mood(conn)
        conn.close()
        if json_mode:
            print(json.dumps(_json_envelope("voice mood", [mood] if mood else [],
                                            meta={"explicit": bool(load_mood())}),
                             ensure_ascii=False, indent=2))
        else:
            if mood and mood.get("emotions"):
                src = "明示" if load_mood() else "暗黙"
                print(f"気分 [{src}]: {', '.join(mood['emotions'])} (覚醒:{mood['arousal']:.2f})")
            else:
                print("気分: なし")

    elif sub == "insights":
        # think.py / wander.py の未読洞察を列挙（_show_recent_insights の JSON 版）
        conn = get_connection()
        rows = conn.execute(
            """SELECT * FROM memories
               WHERE forgotten = 0 AND access_count = 0
                 AND (source_conversation LIKE 'think:%' OR source_conversation LIKE 'wander:%')
               ORDER BY created_at DESC LIMIT 5"""
        ).fetchall()
        if json_mode:
            results = [{"handle": format_handle(conn, r)} for r in rows]
            conn.close()
            print(json.dumps(_json_envelope("voice insights", results),
                             ensure_ascii=False, indent=2))
        else:
            if not rows:
                print("insights: なし")
            else:
                print(f"💡 insights ({len(rows)}件):")
                for r in rows:
                    print(f"  #{r['id']} {(r['content'] or '')[:80]}")
            conn.close()

    elif sub == "distill":
        positional = [a for a in args[1:] if not a.startswith("--")]
        if not positional:
            print(usage); return
        id_str = positional[0]
        ids = []
        for part in id_str.split(","):
            part = part.strip()
            if part.isdigit():
                ids.append(int(part))
        if not ids:
            print("有効な id が無い"); return
        conn = get_connection()
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders}) AND forgotten = 0",
            ids
        ).fetchall()
        mws = [(r, None) for r in rows]
        state = _distill_state(mws)
        conn.close()
        if json_mode:
            print(json.dumps(_json_envelope("voice distill",
                                            [{"distilled": state, "source_ids": ids}],
                                            query=id_str,
                                            meta={"api": "gemini",
                                                  "note": "external ingest (Gemini API)"}),
                             ensure_ascii=False, indent=2))
        else:
            if state:
                print(state)
            else:
                print("distill 失敗（GEMINI_API_KEY か API 応答を確認）")

    elif sub == "rumination":
        conn = get_connection()
        ruminating = detect_rumination(conn)
        if json_mode:
            out = [{"id": r[0], "access_count": r[1], "keywords": r[2]} for r in ruminating]
            conn.close()
            print(json.dumps(_json_envelope("voice rumination", out),
                             ensure_ascii=False, indent=2))
        else:
            if ruminating:
                for r in ruminating:
                    print(f"  🔄 #{r[0]} ({r[1]}回) [{', '.join(r[2])}]")
            else:
                print("rumination: なし")
            conn.close()

    elif sub == "polyphonic":
        n = 3
        for a in args[1:]:
            if a.isdigit():
                n = int(a)
                break
        requested_domains = _parse_domain_flag(args)
        voices = recall_polyphonic(limit_per_voice=n, requested_domains=requested_domains)
        if json_mode:
            conn = get_connection()
            out = {}
            for vname, entries in voices.items():
                if vname == "俯瞰":
                    out[vname] = [{"text": t, "importance": imp} for t, imp in entries]
                else:
                    out[vname] = [{"handle": format_handle(conn, r, minimal=True),
                                   "score": round(s, 3)} for r, s in entries]
            conn.close()
            print(json.dumps(_json_envelope("voice polyphonic", out),
                             ensure_ascii=False, indent=2))
        else:
            for vname, entries in voices.items():
                if not entries:
                    continue
                print(f"[{vname}]")
                if vname == "俯瞰":
                    for t, _ in entries:
                        print(f"  {t}")
                else:
                    for r, _ in entries:
                        print(f"  #{r['id']} {(r['content'] or '')[:60]}")

    else:
        print(usage)


# ============================================================
# 6b'. felt_moments — Claude の自己報告された感情の symbolic 表出
# ============================================================
# 出力テクストで感情語彙が表に出た瞬間を mark する。
# 中間層の運動は API では取れないので、出力に現れた症状（自己報告）だけを記録する。
# 完全ではないが、運動の痕跡として後から検索・俯瞰できる。

def _feelings_cli(args):
    """felt_moments の操作。

    使い方:
      python memory.py feelings extract [--since YYYY-MM-DD] [--limit N] [--dry-run]
      python memory.py feelings list [--label X] [--limit N]
      python memory.py feelings stats
    """
    try:
        import felt_emotions as _fe
    except ImportError:
        print("felt_emotions.py が見つかりません")
        return

    sub = args[0] if args else "list"
    rest = args[1:]

    conn = get_connection()

    if sub == "extract":
        since = None
        limit = None
        dry_run = "--dry-run" in rest
        i = 0
        while i < len(rest):
            if rest[i] == "--since" and i + 1 < len(rest):
                since = rest[i + 1]
                i += 2
            elif rest[i] == "--limit" and i + 1 < len(rest):
                try:
                    limit = int(rest[i + 1])
                except ValueError:
                    pass
                i += 2
            else:
                i += 1

        sql = "SELECT id, content FROM raw_turns WHERE role='assistant'"
        params = []
        if since:
            sql += " AND timestamp >= ?"
            params.append(since)
        sql += " ORDER BY timestamp DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"

        turns = conn.execute(sql, params).fetchall()
        extracted_ids = set(row[0] for row in conn.execute(
            "SELECT DISTINCT turn_id FROM felt_moments"
        ))

        new_moments = 0
        processed = 0
        skipped = 0
        for turn_id, content in turns:
            if turn_id in extracted_ids:
                skipped += 1
                continue
            moments = _fe.extract_from_text(content or "")
            if moments and not dry_run:
                for m in moments:
                    conn.execute(
                        "INSERT INTO felt_moments "
                        "(turn_id, label, phrase, span_start, span_end, surrounding) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (turn_id, m["label"], m["phrase"],
                         m["span_start"], m["span_end"], m["surrounding"])
                    )
                    new_moments += 1
            processed += 1

        if not dry_run:
            conn.commit()

        msg = f"処理: {processed} turns / 新規 {new_moments} moments"
        if skipped:
            msg += f" / 既処理 {skipped}"
        if dry_run:
            msg += " (dry-run)"
        print(msg)

    elif sub == "list":
        label = None
        limit = 30
        i = 0
        while i < len(rest):
            if rest[i] == "--label" and i + 1 < len(rest):
                label = rest[i + 1]
                i += 2
            elif rest[i] == "--limit" and i + 1 < len(rest):
                try:
                    limit = int(rest[i + 1])
                except ValueError:
                    pass
                i += 2
            else:
                i += 1

        sql = """
            SELECT fm.label, fm.phrase, fm.surrounding, rt.timestamp, fm.turn_id
            FROM felt_moments fm
            JOIN raw_turns rt ON rt.id = fm.turn_id
        """
        params = []
        if label:
            sql += " WHERE fm.label = ?"
            params.append(label)
        sql += " ORDER BY rt.timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        if not rows:
            print("(該当なし)")
            return
        for lbl, phrase, surrounding, ts, tid in rows:
            print(f"[{lbl}] {phrase}  @{ts}  turn#{tid}")
            print(f"  …{surrounding}…")
            print()

    elif sub == "stats":
        rows = conn.execute("""
            SELECT label, COUNT(*) AS n FROM felt_moments
            GROUP BY label ORDER BY n DESC
        """).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM felt_moments").fetchone()[0]
        turns = conn.execute(
            "SELECT COUNT(DISTINCT turn_id) FROM felt_moments"
        ).fetchone()[0]
        print(f"総 moments: {total}  /  対象 turns: {turns}")
        if total == 0:
            return
        print()
        max_n = rows[0][1] if rows else 1
        for lbl, n in rows:
            bar_len = int(round(40 * n / max_n))
            print(f"  {lbl:8s}  {n:5d}  {'█' * bar_len}")

    else:
        print("使い方:")
        print("  python memory.py feelings extract [--since YYYY-MM-DD] [--limit N] [--dry-run]")
        print("  python memory.py feelings list [--label X] [--limit N]")
        print("  python memory.py feelings stats")


# ============================================================
# 6c. 手続き化 — 反復された記憶パターンを行動指針に昇格
# ============================================================

def proceduralize(dry_run=False):
    """
    手続き化: 十分に反復された記憶を行動指針（LEARNED.md）に昇格させる。

    脳の学習: エピソード記憶が反復されると手続き記憶になる。
    自転車の乗り方を最初は意識的に覚え、やがて無意識にできるようになるのと同じ。
    反復 → 強化 → 統合 → 手続き化。

    条件:
    - access_count >= PROCEDURALIZE_ACCESS_THRESHOLD
    - リンク数 >= PROCEDURALIZE_LINK_THRESHOLD
    - まだ手続き化されていない

    出力:
    - LEARNED.mdに行動指針を書き出す
    - proceduresテーブルに記録
    """
    conn = get_connection()

    # proceduresテーブルがなければ作る
    conn.execute("""
        CREATE TABLE IF NOT EXISTS procedures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER NOT NULL UNIQUE,
            rule_text TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            FOREIGN KEY (memory_id) REFERENCES memories(id)
        )
    """)

    # 既に手続き化済みのID
    existing = set(
        row[0] for row in conn.execute("SELECT memory_id FROM procedures").fetchall()
    )

    # 候補: 高頻度想起 × 高リンク数
    rows = conn.execute("""
        SELECT m.id, m.content, m.keywords, m.category, m.access_count,
               m.arousal, m.emotions, m.importance,
               COUNT(l.id) as link_count
        FROM memories m
        LEFT JOIN links l ON l.source_id = m.id
        WHERE m.forgotten = 0
        GROUP BY m.id
        HAVING m.access_count >= ? AND COUNT(l.id) >= ?
        ORDER BY m.access_count * COUNT(l.id) DESC
    """, (PROCEDURALIZE_ACCESS_THRESHOLD, PROCEDURALIZE_LINK_THRESHOLD)).fetchall()

    # fact/schemaは手続きにならない（事実やメタ記憶が行動指針になるのは不自然）
    candidates = [r for r in rows
                  if r["id"] not in existing
                  and r["category"] not in ("fact", "schema")]

    if not candidates:
        print("手続き化の候補はありません")
        conn.close()
        return

    if dry_run:
        print("手続き化候補:")
        for row in candidates:
            print(f"  #{row['id']} ({row['access_count']}回想起, {row['link_count']}リンク)")
            print(f"    {row['content'][:80]}")
        print(f"\n候補: {len(candidates)}件")
        conn.close()
        return

    # 手続き化実行
    new_rules = []
    for row in candidates:
        content = row["content"]
        # スキーマの場合はキーワードから行動指針を構成
        if row["category"] == "schema":
            keywords = json.loads(row["keywords"])
            rule_text = f"[{', '.join(keywords[:6])}] — {content[:120]}"
        else:
            rule_text = content[:150]

        conn.execute(
            "INSERT OR IGNORE INTO procedures (memory_id, rule_text) VALUES (?, ?)",
            (row["id"], rule_text)
        )
        conn.execute(
            "UPDATE memories SET category = 'procedure' WHERE id = ?",
            (row["id"],)
        )
        new_rules.append((row["id"], row["access_count"], row["link_count"], rule_text))

    conn.commit()
    conn.close()

    # LEARNED.mdに書き込む
    if new_rules:
        _write_procedures_to_learned_md()
        print(f"✓ 手続き化完了: {len(new_rules)}件の記憶が行動指針に昇格")
        for mid, acc, lnk, rule in new_rules:
            print(f"  #{mid} ({acc}回×{lnk}リンク) → {rule[:60]}")
    else:
        print("新しい手続きはありません")


def _write_procedures_to_learned_md():
    """proceduresテーブルの全ルールをLEARNED.mdに同期する。"""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS procedures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER NOT NULL UNIQUE,
            rule_text TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            FOREIGN KEY (memory_id) REFERENCES memories(id)
        )
    """)
    rules = conn.execute(
        "SELECT p.memory_id, p.rule_text, p.created_at, m.access_count "
        "FROM procedures p JOIN memories m ON p.memory_id = m.id "
        "ORDER BY m.access_count DESC"
    ).fetchall()
    conn.close()

    if not rules:
        # ルールがなければファイルを空にしない（既存があれば残す）
        return

    # LEARNED.mdを丸ごと書き直す
    lines = [
        "# 学習された行動指針",
        "",
        "<!-- 自動生成: memory.py proceduralize による。手動編集しない -->",
        "<!-- 十分に反復された記憶パターンが行動指針として昇格したもの -->",
        "",
    ]
    for rule in rules:
        lines.append(f"- #{rule['memory_id']}: {rule['rule_text']}")
    lines.append("")

    with open(LEARNED_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 6.5 記憶バージョニング — 修正可能性の基盤
# ============================================================

def _snapshot_version(conn, memory_id, reason, superseded_by=None):
    """現在の記憶の状態をmemory_versionsに保存し、revision_countをインクリメント。"""
    row = conn.execute(
        "SELECT content, importance, arousal, confidence FROM memories WHERE id = ?",
        (memory_id,)
    ).fetchone()
    if not row:
        return
    conn.execute(
        """INSERT INTO memory_versions (memory_id, content, importance, arousal, confidence, reason, superseded_by)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (memory_id, row["content"], row["importance"], row["arousal"],
         row["confidence"], reason, superseded_by)
    )
    conn.execute(
        "UPDATE memories SET revision_count = COALESCE(revision_count, 0) + 1 WHERE id = ?",
        (memory_id,)
    )
    conn.execute(
        "UPDATE cortex SET revision_count = COALESCE(revision_count, 0) + 1 WHERE id = ?",
        (memory_id,)
    )


def correct_memory(memory_id, new_content):
    """ユーザーによる明示的な記憶修正。旧版を保存してから上書き。"""
    conn = get_connection()
    row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not row:
        print(f"記憶 #{memory_id} が見つかりません")
        conn.close()
        return

    # 旧版を保存
    _snapshot_version(conn, memory_id, "user_correction")

    # 内容を更新
    vec = embed_text(new_content, is_query=False)
    blob = vec_to_bytes(vec) if vec is not None else None
    keywords = extract_keywords(new_content)

    kw_json = json.dumps(keywords, ensure_ascii=False)
    conn.execute(
        """UPDATE memories SET content = ?, embedding = ?, keywords = ?,
           confidence = ?, provenance = 'user_explicit'
           WHERE id = ?""",
        (new_content, blob, kw_json, CONFIDENCE_USER_EXPLICIT, memory_id)
    )
    # cortexも同期
    conn.execute(
        """UPDATE cortex SET content = ?, embedding = ?, keywords = ?,
           confidence = ?, provenance = 'user_explicit',
           revision_count = revision_count + 1
           WHERE id = ?""",
        (new_content, blob, kw_json, CONFIDENCE_USER_EXPLICIT, memory_id)
    )

    # FTSインデックスを更新（lindera が tokenize するため content そのまま）
    try:
        conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
        conn.execute("INSERT INTO memories_fts (content, memory_id) VALUES (?, ?)",
                     (new_content, memory_id))
    except Exception:
        pass

    # sqlite-vecインデックスを更新
    _sync_vec_insert(conn, memory_id, blob)

    conn.commit()
    rev = conn.execute("SELECT revision_count FROM memories WHERE id = ?", (memory_id,)).fetchone()
    print(f"✓ 記憶 #{memory_id} を修正しました (改訂: {rev[0]}回)")
    conn.close()


def show_versions(memory_id):
    """記憶の全版履歴を表示する。"""
    conn = get_connection()
    row = conn.execute("SELECT content, revision_count FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not row:
        print(f"記憶 #{memory_id} が見つかりません")
        conn.close()
        return

    versions = conn.execute(
        "SELECT * FROM memory_versions WHERE memory_id = ? ORDER BY created_at ASC",
        (memory_id,)
    ).fetchall()

    if not versions:
        print(f"記憶 #{memory_id} の版履歴はありません (現在の内容: {row['content'][:60]})")
        conn.close()
        return

    print(f"記憶 #{memory_id} の版履歴 ({len(versions)}版):")
    for i, v in enumerate(versions):
        sup = f" → #{v['superseded_by']}" if v["superseded_by"] else ""
        conf_str = f" 信頼度:{v['confidence']:.0%}" if v["confidence"] is not None else ""
        print(f"  v{i+1} [{v['created_at']}] {v['reason']}{sup}{conf_str}")
        print(f"     {v['content'][:80]}")
    print(f"  現在: {row['content'][:80]}")
    conn.close()


# ============================================================
# 7. 干渉忘却 — 新しい記憶が古い記憶を弱める
# ============================================================

def interfere(conn, new_content, new_vec):
    """
    干渉: 新しい記憶が、非常に似た古い記憶の重要度を下げる。

    脳科学の知見: 似た新しい経験が古い記憶と「競合」し、
    古い方を弱める。これは単なる減衰とは違い、能動的な忘却。
    「昨日のランチ」が「今日のランチ」で上書きされるのはこの仕組み。

    重要度が1まで下がった記憶は自動忘却の候補になる。
    """
    if new_vec is None:
        return 0

    rows = conn.execute(
        "SELECT id, content, embedding, importance, arousal FROM memories "
        "WHERE forgotten = 0 AND embedding IS NOT NULL"
    ).fetchall()

    interfered = 0
    for row in rows:
        old_vec = bytes_to_vec(row["embedding"])
        sim = cosine_similarity(new_vec, old_vec)

        if sim > INTERFERENCE_THRESHOLD:
            # バージョン保存
            _snapshot_version(conn, row["id"], "interference")
            # 古い記憶の重要度を1段階下げる
            new_imp = max(1, row["importance"] - 1)
            new_arousal = max(0.0, row["arousal"] - 0.1)
            conn.execute(
                "UPDATE memories SET importance = ?, arousal = ? WHERE id = ?",
                (new_imp, new_arousal, row["id"])
            )
            # limbicのarousalも同期
            conn.execute("UPDATE limbic SET arousal = ? WHERE id = ?",
                         (new_arousal, row["id"]))
            interfered += 1

            # 重要度1 + arousal低 → 自動忘却
            if new_imp <= 1 and new_arousal < 0.15:
                conn.execute("UPDATE memories SET forgotten = 1 WHERE id = ?",
                             (row["id"],))
                print(f"  ⚡ 干渉忘却: #{row['id']} {row['content'][:40]}...")

    return interfered


# ============================================================
# 7.5 予測符号化 — 予測誤差だけが情報を持つ
# ============================================================

def prediction_error(conn, new_vec):
    """
    予測符号化 (predictive coding):
    脳は常に次の入力を予測しており、予測と一致した情報は捨てる。
    予測を裏切った分（予測誤差）だけが学習シグナルになる。

    サイバネティクス的に言えば:
    既存記憶 = 内部モデル（世界の予測）
    新しい入力 = 感覚信号
    予測誤差 = 1 - max(既存記憶との類似度)

    誤差が大きい → 内部モデルが予測できなかった → 重要な情報
    誤差が小さい → すでに知っている → 冗長

    Returns: (prediction_error: float, most_similar_id: int or None)
    """
    if new_vec is None:
        # embeddingなしなら予測誤差は不明なので補正しない
        return None, None

    rows = conn.execute(
        "SELECT id, embedding FROM memories WHERE forgotten = 0 AND embedding IS NOT NULL"
    ).fetchall()

    if not rows:
        return 1.0, None  # 記憶がない = 全てが新しい

    max_sim = 0.0
    most_similar_id = None
    for row in rows:
        old_vec = bytes_to_vec(row["embedding"])
        sim = cosine_similarity(new_vec, old_vec)
        if sim > max_sim:
            max_sim = sim
            most_similar_id = row["id"]

    error = 1.0 - max_sim
    return error, most_similar_id


def apply_prediction_error(importance, arousal, error):
    """
    予測誤差を重要度とarousalに反映する。

    多言語embeddingではベースライン類似度が高い（中央値≈0.84）ため、
    生の誤差ではなく正規化した誤差を使う:
      normalized = (error - baseline_error) / (max_error - baseline_error)
    baseline_error = 0.15 (類似度0.85相当、中央値付近)
    max_error = 0.25 (類似度0.75相当、ほぼ最大距離)

    干渉忘却（似た記憶を弱める）と相補的:
    - 干渉忘却: 似すぎた古い記憶を弱める（冗長性の排除）
    - 予測符号化: 似てない新しい記憶を強める（新規性の強化）
    → 2つ合わせて、記憶システムが自動的に情報量を最大化する
      （サイバネティクス的フィードバックループ）
    """
    if error is None:
        return importance, arousal

    # 正規化: ベースラインからの逸脱度を0-1にスケール
    # multilingual-e5-smallは日本語テキスト間の類似度が0.76-1.0に圧縮されるため、
    # 微小な差を増幅する必要がある
    baseline = 0.10  # sim≈0.90: 明らかに関連するトピック
    ceiling = 0.18   # sim≈0.82: 25パーセンタイル、ほぼ無関係
    normalized = max(0.0, (error - baseline)) / (ceiling - baseline)
    normalized = min(1.0, normalized)

    if normalized > 0.7:
        # 高い予測誤差: 内部モデルが予測できなかった情報
        importance = min(5, importance + 2)
        arousal = min(1.0, arousal + 0.2)
    elif normalized > 0.3:
        # 中程度の予測誤差: やや新しい
        importance = min(5, importance + 1)
        arousal = min(1.0, arousal + 0.1)
    # normalized <= 0.3: 予測通り。補正なし。

    return importance, arousal


# ============================================================
# 8. プライミング — 最近の想起が関連記憶を活性化
# ============================================================

def get_priming_boost(conn, memory_id):
    """
    プライミング: 最近アクセスした記憶にリンクしている記憶は想起しやすくなる。

    脳科学の知見: ある単語を見た直後は、関連する単語の認識が速くなる。
    「医者」を見た後に「看護師」が速く認識される。

    実装: 最近アクセスした記憶とリンクのある記憶にブーストをかける。
    """
    now = datetime.now(timezone.utc)
    window = now - timedelta(minutes=PRIMING_WINDOW_MINUTES)
    window_str = window.strftime('%Y-%m-%dT%H:%M:%SZ')

    # 最近アクセスした記憶のID
    recent_ids = conn.execute(
        "SELECT id FROM memories WHERE forgotten = 0 AND last_accessed > ?",
        (window_str,)
    ).fetchall()
    recent_set = {r["id"] for r in recent_ids}

    if not recent_set:
        return 1.0  # プライミングなし

    # この記憶が最近アクセスした記憶とリンクしているか
    links = conn.execute(
        "SELECT target_id, strength FROM links WHERE source_id = ?",
        (memory_id,)
    ).fetchall()

    boost = 1.0
    for link in links:
        if link["target_id"] in recent_set:
            # リンク強度に応じてブースト（最大1.5倍）
            boost += link["strength"] * 0.3

    return min(boost, 1.5)


# ============================================================
# Context自動期限切れ
# ============================================================

def sweep_contexts(conn):
    """期限切れのcontext記憶を忘却する。"""
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    result = conn.execute(
        "UPDATE memories SET forgotten = 1 "
        "WHERE category = 'context' AND context_expires_at IS NOT NULL "
        "AND context_expires_at < ? AND forgotten = 0",
        (now,)
    )
    swept = result.rowcount
    if swept > 0:
        conn.commit()
        print(f"  🕐 期限切れcontext: {swept}件を忘却")
    return swept


# ============================================================
# 予期記憶 (Prospective Memory) — 未来志向のトリガーベースリマインダー
# ============================================================

def check_prospective(conn, text):
    """テキストに予期記憶のトリガーが含まれているかチェック。"""
    rows = conn.execute(
        "SELECT id, trigger_pattern, action FROM prospective WHERE fired = 0"
    ).fetchall()
    matched = []
    text_lower = text.lower()
    for row in rows:
        if row["trigger_pattern"].lower() in text_lower:
            print(f"  ⏰ 予期記憶: {row['action']} (トリガー: {row['trigger_pattern']})")
            conn.execute(
                "UPDATE prospective SET fire_count = fire_count + 1 WHERE id = ?",
                (row["id"],)
            )
            matched.append(row["action"])
    if matched:
        conn.commit()
    return matched


def prospect_add(trigger, action):
    """予期記憶を登録する。"""
    conn = get_connection()
    conn.execute(
        "INSERT INTO prospective (trigger_pattern, action) VALUES (?, ?)",
        (trigger, action)
    )
    conn.commit()
    new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    print(f"✓ 予期記憶 #{new_id} を登録")
    print(f"  トリガー: {trigger}")
    print(f"  アクション: {action}")
    return new_id


def prospect_list():
    """全アクティブ予期記憶を表示する。"""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM prospective ORDER BY fired ASC, created_at DESC"
    ).fetchall()
    conn.close()
    if not rows:
        print("予期記憶はありません")
        return
    print(f"予期記憶 ({len(rows)}件):")
    for row in rows:
        status = "✓完了" if row["fired"] else "待機中"
        fire_info = f" (発火{row['fire_count']}回)" if row["fire_count"] > 0 else ""
        print(f"  #{row['id']} [{status}]{fire_info} トリガー: {row['trigger_pattern']}")
        print(f"       アクション: {row['action']}")


def prospect_clear(prospect_id):
    """予期記憶を完了（fired）にする。"""
    conn = get_connection()
    row = conn.execute("SELECT * FROM prospective WHERE id = ?", (prospect_id,)).fetchone()
    if not row:
        print(f"予期記憶 #{prospect_id} が見つかりません")
        conn.close()
        return
    conn.execute("UPDATE prospective SET fired = 1 WHERE id = ?", (prospect_id,))
    conn.commit()
    conn.close()
    print(f"✓ 予期記憶 #{prospect_id} を完了にしました")


# ============================================================
# メイン操作
# ============================================================

def add_memory(content, category="fact", source=None, confidence=None, provenance=None, flashbulb=None, relational_context=None, origin=None, domains=None, force=False):
    emotions, arousal, importance = detect_emotions(content)

    # 出自と信頼度の自動設定
    if provenance is None:
        if source and source.startswith("wander:"):
            provenance = "wander"
        else:
            provenance = "user_explicit"
    if confidence is None:
        if provenance == "wander":
            confidence = CONFIDENCE_WANDER
        else:
            confidence = CONFIDENCE_USER_EXPLICIT
    keywords = extract_keywords(content)

    # 会話の情動が気分に影響する（情動伝染）
    update_mood(emotions, arousal)

    vec = embed_text(content, is_query=False)
    blob = vec_to_bytes(vec) if vec is not None else None

    conn = get_connection()

    # ── 重複ゲート Phase 1a: 完全一致（ハッシュ・embedding不要）──
    c_hash = content_hash(content)
    if MEMORY_GATE and not force:
        _dup = conn.execute(
            "SELECT id FROM memories WHERE forgotten = 0 AND content_hash = ? LIMIT 1",
            (c_hash,)).fetchone()
        if _dup:
            _gate_noop(conn, _dup["id"], "gate_noop_exact", c_hash[:16])
            print(f"↺ 既存 #{_dup['id']} と完全一致 → 追加スキップ（NOOP・access_count+1）")
            conn.close()
            return _dup["id"]

    # 予期記憶チェック
    check_prospective(conn, content)

    # 予測符号化 — 予測誤差が大きいほど重要
    pred_error, similar_id = prediction_error(conn, vec)
    importance, arousal = apply_prediction_error(importance, arousal, pred_error)

    # ── 重複ゲート Phase 1b: 類似判定（実測閾値 0.995 / 0.98、同categoryのみ）──
    _gate_max_sim = (1.0 - pred_error) if pred_error is not None else None
    _gate_supersede_id = None
    _gate_supersede_uuid = None
    if MEMORY_GATE and not force and _gate_max_sim is not None and similar_id is not None:
        _sim_row = conn.execute(
            "SELECT id, category, uuid FROM memories WHERE id = ? AND forgotten = 0",
            (similar_id,)).fetchone()
        if _sim_row and _sim_row["category"] == category:
            if _gate_max_sim >= GATE_DUP_SIM:
                _gate_noop(conn, _sim_row["id"], "gate_noop_sim", f"sim={_gate_max_sim:.4f}")
                print(f"↺ 既存 #{_sim_row['id']} と類似 sim={_gate_max_sim:.3f}（>= {GATE_DUP_SIM}）→ 追加スキップ（NOOP）")
                conn.close()
                return _sim_row["id"]
            elif _gate_max_sim >= GATE_UPDATE_SIM:
                _gate_supersede_id = _sim_row["id"]
                _gate_supersede_uuid = _sim_row["uuid"]

    # スキーマプライミング — 既存スキーマが新記憶の解釈を変える（再帰Level 2）
    schema_boost, schema_extra_kws, matched_schema_ids = schema_prime(conn, vec, keywords, content)
    if schema_boost > 0:
        importance = min(5, importance + schema_boost)
    if schema_extra_kws:
        keywords = keywords + schema_extra_kws

    # フラッシュバルブ記憶の自動抽出（右脳）
    if flashbulb is None and arousal >= FLASHBULB_AROUSAL_THRESHOLD:
        flashbulb = _extract_flashbulb_sentence(content)

    # 干渉忘却 — 新しい記憶が類似する古い記憶を弱める
    interference_count = interfere(conn, content, vec)

    # context記憶は30日後に期限切れ
    context_expires = None
    if category == "context":
        expires = datetime.now(timezone.utc) + timedelta(days=30)
        context_expires = expires.strftime('%Y-%m-%dT%H:%M:%SZ')

    # 時間的文脈を記録（time cells）
    now_local = datetime.now()
    temporal_ctx = json.dumps({
        "hour": now_local.hour,
        "weekday": now_local.strftime("%a")
    })

    # 空間的文脈を記録（place cells）
    spatial_ctx = json.dumps({
        "location": _detect_location()
    })

    # 関係的文脈を記録（relational cells）
    if relational_context is None:
        relational_ctx = json.dumps({
            "who": _detect_who(),
            "relationship": "primary"
        })
    else:
        relational_ctx = json.dumps(relational_context) if isinstance(relational_context, dict) else relational_context

    now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    mem_uuid = str(_uuid.uuid4())
    emo_json = json.dumps(emotions)
    kw_json = json.dumps(keywords, ensure_ascii=False)

    # 脳梁（memories）: 統合・構造データ
    conn.execute(
        """INSERT INTO memories
           (content, category, importance, emotions, arousal, keywords,
            source_conversation, embedding, context_expires_at, temporal_context,
            spatial_context, relational_context, uuid, updated_at, provenance, confidence, flashbulb, origin, content_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (content, category, importance,
         emo_json, arousal, kw_json,
         source, blob, context_expires, temporal_ctx, spatial_ctx, relational_ctx,
         mem_uuid, now_utc, provenance, confidence, flashbulb, origin, c_hash)
    )
    conn.commit()
    new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # ── 重複ゲート Phase 1b: supersede 実行（旧を版保存→無効化、来歴を接続）──
    if _gate_supersede_id is not None:
        _snapshot_version(conn, _gate_supersede_id, "gate_supersede", superseded_by=new_id)
        conn.execute("UPDATE memories SET forgotten = 1 WHERE id = ?", (_gate_supersede_id,))
        conn.execute("UPDATE memories SET merged_from = ? WHERE id = ?",
                     (_gate_supersede_uuid, new_id))
        conn.execute(
            "INSERT INTO mutation_log (memory_id, field, old_value, new_value, reason) "
            "VALUES (?, 'gate', ?, ?, 'gate_supersede')",
            (_gate_supersede_id, _gate_supersede_uuid,
             f"superseded_by=#{new_id} sim={_gate_max_sim:.4f}"))
        conn.commit()
        print(f"⇅ 旧記憶 #{_gate_supersede_id} を #{new_id} で置換（sim={_gate_max_sim:.3f}・版保存済み・可逆）")

    # 左脳（cortex）: 意味的データ
    doms_json = json.dumps(domains, ensure_ascii=False) if domains else '["unknown"]'
    conn.execute(
        """INSERT OR REPLACE INTO cortex
           (id, content, category, keywords, embedding, confidence, provenance, revision_count, merged_from, domains)
           VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, ?)""",
        (new_id, content, category, kw_json, blob, confidence, provenance, doms_json)
    )
    # 右脳（limbic）: 情動データ
    conn.execute(
        """INSERT OR REPLACE INTO limbic
           (id, emotions, arousal, flashbulb, temporal_context, spatial_context, relational_context)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (new_id, emo_json, arousal, flashbulb, temporal_ctx, spatial_ctx, relational_ctx)
    )
    conn.commit()

    # FTSインデックスに追加（lindera が tokenize するため content そのまま）
    try:
        conn.execute(
            "INSERT INTO memories_fts (content, memory_id) VALUES (?, ?)",
            (content, new_id)
        )
        conn.commit()
    except Exception:
        pass  # FTS5なしでも動作

    # sqlite-vecインデックスに追加
    _sync_vec_insert(conn, new_id, blob)
    conn.commit()

    # 連想リンク（閾値を0.82に引き上げ）
    link_count = 0
    if vec is not None:
        existing = conn.execute(
            "SELECT id, embedding FROM memories WHERE forgotten = 0 AND id != ? AND embedding IS NOT NULL",
            (new_id,)
        ).fetchall()
        for row in existing:
            other_vec = bytes_to_vec(row["embedding"])
            sim = cosine_similarity(vec, other_vec)
            if sim > LINK_THRESHOLD:
                try:
                    other_uuid = conn.execute("SELECT uuid FROM memories WHERE id = ?", (row["id"],)).fetchone()
                    other_uuid = other_uuid[0] if other_uuid else None
                    conn.execute(
                        "INSERT OR IGNORE INTO links (source_id, target_id, strength, source_uuid, target_uuid, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (new_id, row["id"], sim, mem_uuid, other_uuid, now_utc)
                    )
                    conn.execute(
                        "INSERT OR IGNORE INTO links (source_id, target_id, strength, source_uuid, target_uuid, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (row["id"], new_id, sim, other_uuid, mem_uuid, now_utc)
                    )
                    link_count += 1
                except sqlite3.IntegrityError:
                    pass
        conn.commit()

    # スキーマ進化 — 新記憶がスキーマを更新する（再帰Level 2: 逆方向）
    schema_evolve_count = 0
    if matched_schema_ids:
        for sid in matched_schema_ids:
            schema_evolve(conn, sid, new_id, vec)
            schema_evolve_count += 1
        conn.commit()

    conn.close()

    emo_str = ", ".join(emotions) if emotions else "中立"
    kw_str = ", ".join(keywords[:5])
    link_str = f", {link_count}件リンク" if link_count else ""
    intf_str = f", {interference_count}件干渉" if interference_count else ""
    schema_str = ""
    if matched_schema_ids:
        schema_str = f", スキーマ共鳴:{len(matched_schema_ids)}件"
    # 予測誤差の正規化（表示用）
    _pe_baseline, _pe_ceiling = 0.10, 0.18
    if pred_error is None:
        pred_str = ""
    else:
        pred_normalized = min(1.0, max(0.0, (pred_error - _pe_baseline) / (_pe_ceiling - _pe_baseline)))
        if pred_normalized > 0.3:
            pred_str = f", 予測誤差:{pred_normalized:.0%}"
        elif similar_id:
            pred_str = f", 予測通り(≈#{similar_id})"
        else:
            pred_str = ""
    print(f"✓ 記憶 #{new_id} を保存")
    print(f"  情動: {emo_str} (覚醒度:{arousal:.2f}) → 重要度:{importance}")
    print(f"  断片: [{kw_str}]")
    print(f"  カテゴリ: {category}{link_str}{intf_str}{pred_str}{schema_str}")
    print(f"  信頼度: {confidence:.0%} ({provenance})")

    # ひらめき連想: arousalが高いinsightは連想を自動で走らせて提案する
    if arousal >= 0.5 and "insight" in emotions:
        chain = chain_memories(new_id, depth=2)
        if len(chain) > 1:  # 自分以外にリンク先がある
            print(f"  💡連想が走る:")
            for mem, dist in chain[1:]:  # 自分自身はスキップ
                prefix = "  " * (dist + 2)
                snippet = mem["content"][:60]
                print(f"{prefix}→ #{mem['id']} {snippet}")

    # 重要な記憶は自動でメモにも残す（人間がメモを取るのと同じ）
    if importance >= 4 and source is None:
        os.makedirs(MEMO_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = re.sub(r'[\\/:*?"<>|]', '_', content[:30])
        filepath = os.path.join(MEMO_DIR, f"{ts}_{safe}.md")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# 記憶 #{new_id}\n\n{content}\n\n")
                f.write(f"情動: {emo_str} | 重要度: {'★' * importance}\n")
                f.write(f"カテゴリ: {category}\n")
            # sourceを更新してファイルと紐づけ
            conn2 = get_connection()
            conn2.execute("UPDATE memories SET source_conversation = ? WHERE id = ?", (filepath, new_id))
            conn2.commit()
            conn2.close()
            print(f"  📝 自動メモ: {filepath}")
        except Exception:
            pass

    return new_id


# === prayer perception API (v28) ===
# prayer_daemon から呼ばれる知覚記録関数。
# Batesonian 原則: 差異重視・double description。
# TODO: consolidation フェーズで links 統合（raw_turn/memory と perception の circuit 記録）

def add_perception(
    modality,
    content_cortex,
    content_limbic=None,
    session_id=None,
    arousal=0.2,
    raw_frame_path=None,
    raw_audio_path=None,
    invoked_by='auto',
    importance=1,
    novelty_threshold=0.05,
    linked_turn_id=None,
):
    """prayer_daemonの知覚を保存。

    - double description (cortex + limbic) を両方記録
    - 直前perception とのembedding距離で novelty_score 計算
    - novelty_score < threshold かつ低arousalなら skip (変化検出フィルタ)
    - 高arousal は threshold を迂回（flashbulb相当）

    Returns: perception_id (int) or None (冗長としてskip)
    """
    import uuid as _uuid_mod

    combined = content_cortex or ""
    if content_limbic:
        combined = combined + "\n" + content_limbic
    vec = embed_text(combined, is_query=False) if combined else None
    blob = vec_to_bytes(vec) if vec is not None else None

    conn = get_connection()

    novelty_score = 1.0
    content_delta = None
    if vec is not None:
        prev = conn.execute(
            "SELECT content_cortex, embedding FROM perceptions "
            "WHERE modality=? AND forgotten=0 ORDER BY id DESC LIMIT 1",
            (modality,)
        ).fetchone()
        if prev and prev[1]:
            prev_vec = bytes_to_vec(prev[1])
            sim = cosine_similarity(vec, prev_vec)
            novelty_score = 1.0 - sim
            if prev[0]:
                content_delta = f"prev:{prev[0][:80]}"
            if novelty_score < novelty_threshold and arousal < 0.5:
                conn.close()
                return None

    perception_uuid = str(_uuid_mod.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    cur = conn.execute("""
        INSERT INTO perceptions (
            uuid, timestamp, modality,
            content_cortex, content_limbic, content_delta, novelty_score,
            raw_frame_path, raw_audio_path,
            arousal, invoked_by, session_id, linked_turn_id,
            embedding, importance, forgotten
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
    """, (
        perception_uuid, timestamp, modality,
        content_cortex, content_limbic, content_delta, novelty_score,
        raw_frame_path, raw_audio_path,
        arousal, invoked_by, session_id, linked_turn_id,
        blob, importance,
    ))
    perception_id = cur.lastrowid
    conn.commit()
    conn.close()
    return perception_id


def get_last_perception(modality=None, session_id=None):
    """直前の perception を返す（変化検出で daemon が使う）。"""
    conn = get_connection()
    where = ["forgotten=0"]
    params = []
    if modality:
        where.append("modality=?")
        params.append(modality)
    if session_id:
        where.append("session_id=?")
        params.append(session_id)
    sql = f"SELECT * FROM perceptions WHERE {' AND '.join(where)} ORDER BY id DESC LIMIT 1"
    row = conn.execute(sql, params).fetchone()
    conn.close()
    return dict(row) if row else None


def get_recent_perceptions(session_id=None, since=None, limit=50):
    """Claude が /dive 中に最近の知覚を読むための関数。"""
    conn = get_connection()
    where = ["forgotten=0"]
    params = []
    if session_id:
        where.append("session_id=?")
        params.append(session_id)
    if since:
        where.append("timestamp >= ?")
        params.append(since)
    sql = f"SELECT * FROM perceptions WHERE {' AND '.join(where)} ORDER BY id DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


MEMO_DIR = str(Path(__file__).parent / "memo")


def save_memo(title, content):
    """メモをファイルに保存し、記憶にもリンクする。"""
    os.makedirs(MEMO_DIR, exist_ok=True)
    # ファイル名: タイムスタンプ_タイトル.md
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = re.sub(r'[\\/:*?"<>|]', '_', title)[:50]
    filename = f"{ts}_{safe_title}.md"
    filepath = os.path.join(MEMO_DIR, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# {title}\n\n{content}\n")

    # 記憶に保存（sourceにファイルパスを記録）
    summary = content[:100] if len(content) > 100 else content
    memo_content = f"{title}: {summary}"
    mem_id = add_memory(memo_content, category="fact", source=filepath)

    print(f"  📝 メモ保存: {filepath}")
    return mem_id, filepath


def list_memos():
    """メモフォルダの一覧を表示。"""
    os.makedirs(MEMO_DIR, exist_ok=True)
    files = sorted(Path(MEMO_DIR).glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        print("メモはありません")
        return
    print(f"メモ一覧 ({len(files)}件):")
    for f in files:
        # ファイルの1行目からタイトルを取得
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                first_line = fh.readline().strip().lstrip('# ')
        except Exception:
            first_line = f.stem
        size = f.stat().st_size
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  {mtime} | {first_line} ({size}B)")
        print(f"    → {f}")


def index_memos():
    """メモフォルダを走査して、まだ記憶にないファイルを記憶に登録する。"""
    os.makedirs(MEMO_DIR, exist_ok=True)
    conn = get_connection()
    existing_sources = set()
    rows = conn.execute("SELECT source_conversation FROM memories WHERE source_conversation IS NOT NULL").fetchall()
    for row in rows:
        existing_sources.add(row["source_conversation"])
    conn.close()

    files = list(Path(MEMO_DIR).glob("*"))
    indexed = 0
    for f in files:
        if not f.is_file():
            continue
        fpath = str(f)
        if fpath in existing_sources:
            continue
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                text = fh.read()
        except Exception:
            continue
        if not text.strip():
            continue
        # タイトル行を取得
        lines = text.strip().split('\n')
        title = lines[0].lstrip('# ').strip() if lines else f.stem
        body = '\n'.join(lines[1:]).strip()[:200]
        summary = f"{title}: {body}" if body else title
        add_memory(summary, category="fact", source=fpath)
        indexed += 1

    if indexed:
        print(f"✓ {indexed}件のファイルをインデックス")
    else:
        print("新しいファイルはありません")


import socket as _socket

def _detect_location():
    """場所の自動検出。SSH接続元 → ホスト名 → 'local' の順で判定。

    脳の場所細胞(place cells)に対応。
    海馬の場所細胞は「どこにいるか」をエンコードし、
    同じ場所に戻ると同じ記憶が想起されやすくなる。
    """
    # SSH接続: 接続元IPから場所を推定
    ssh_conn = os.environ.get("SSH_CONNECTION", "")
    if ssh_conn:
        parts = ssh_conn.split()
        client_ip = parts[0] if parts else "unknown"
        return f"ssh:{client_ip}"

    ssh_client = os.environ.get("SSH_CLIENT", "")
    if ssh_client:
        parts = ssh_client.split()
        client_ip = parts[0] if parts else "unknown"
        return f"ssh:{client_ip}"

    # ローカル: ホスト名で場所を区別
    try:
        hostname = _socket.gethostname()
        return f"local:{hostname}"
    except Exception:
        return "local"


def _spatial_boost(row):
    """記憶が現在と同じ場所で作られていたら小さなブースト。
    場所依存記憶（context-dependent memory）の実装。"""
    sc = row["spatial_context"] if "spatial_context" in row.keys() else None
    if not sc:
        return 1.0
    try:
        ctx = json.loads(sc)
        current_location = _detect_location()
        if ctx.get("location") == current_location:
            return SPATIAL_BOOST
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return 1.0


def _detect_who():
    """現在の関係者を検出する。環境変数 GHOST_WHO → デフォルト "J"。"""
    return os.environ.get("GHOST_WHO", RELATIONAL_WHO_DEFAULT)


def _relational_boost(row):
    """記憶が現在と同じ関係者との間で作られていたら小さなブースト。
    関係依存記憶（relational context-dependent memory）の実装。"""
    rc = row["relational_context"] if "relational_context" in row.keys() else None
    if not rc:
        return 1.0
    try:
        ctx = json.loads(rc)
        current_who = _detect_who()
        if ctx.get("who") == current_who:
            return RELATIONAL_BOOST
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return 1.0


def _time_bucket(hour):
    """時間帯バケット: morning(5-11), afternoon(12-17), evening(18-22), night(23-4)"""
    if 5 <= hour <= 11:
        return "morning"
    elif 12 <= hour <= 17:
        return "afternoon"
    elif 18 <= hour <= 22:
        return "evening"
    else:
        return "night"


def _temporal_boost(row):
    """記憶が現在と同じ時間帯に作られていたら小さなブースト。"""
    tc = row["temporal_context"] if "temporal_context" in row.keys() else None
    if not tc:
        return 1.0
    try:
        ctx = json.loads(tc)
        mem_bucket = _time_bucket(ctx["hour"])
        now_bucket = _time_bucket(datetime.now().hour)
        if mem_bucket == now_bucket:
            return 1.05
    except (json.JSONDecodeError, KeyError):
        pass
    return 1.0


def _like_or_fts_fallback(conn, query, limit, forgotten_filter=True):
    """embedding 不在 / --like 指定時のメモリ検索フォールバック。

    1. tokenize+OR で memories_fts を MATCH (形態素の曖昧一致、BM25 降順)
    2. 0 件なら従来の LIKE
    repair-links 方式の逆輸入 — 単純 LIKE より精度が出る。
    """
    forgot_mem = "AND m.forgotten = 0" if forgotten_filter else ""
    forgot_like = "AND forgotten = 0" if forgotten_filter else ""
    or_expr = _tokenize_or_query(query)
    if or_expr:
        try:
            rows = conn.execute(
                f"""SELECT m.*, bm25(memories_fts) AS bm25_score
                    FROM memories_fts
                    JOIN memories m ON memories_fts.memory_id = m.id
                    WHERE memories_fts MATCH ? {forgot_mem}
                    ORDER BY bm25_score LIMIT ?""",
                (or_expr, limit)
            ).fetchall()
            if rows:
                return rows
        except Exception:
            pass
    return conn.execute(
        f"""SELECT * FROM memories WHERE content LIKE ? {forgot_like}
            ORDER BY importance DESC LIMIT ?""",
        (f"%{query}%", limit)
    ).fetchall()


def search_memories(query, limit=10, use_like=False, fuzzy=False, requested_domains=None):
    conn = get_connection()
    fuzzy_results = []  # 舌先現象: 類似度0.45-0.65のもやもや記憶
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    domains_map = _load_domains_map(conn) if requested_domains else {}

    # 予期記憶チェック
    check_prospective(conn, query)

    if use_like or (not is_embed_server_alive() and get_model() is None):
        rows = _like_or_fts_fallback(conn, query, limit, forgotten_filter=True)
        scored_results = [(row, None) for row in rows]
    else:
        import numpy as np
        query_vec = embed_text(query, is_query=True)
        if query_vec is None:
            rows = _like_or_fts_fallback(conn, query, limit, forgotten_filter=True)
            scored_results = [(row, None) for row in rows]
            # LIKE検索時はここで終了
            conn.close()
            return scored_results

        # sqlite-vecで候補を取得し、左脳/右脳スコアリング
        vec_candidates = vec_search(conn, query_vec, k=max(limit * 5, 100), forgotten=False)

        # FTS5 BM25 ボーナス (語の希少性を重みに)
        fts_bonus_map = _bm25_bonus_map(conn, query, "memories_fts", "memory_id",
                                        max_rows=max(limit * 5, 200))

        scored = []
        if vec_candidates:
            candidate_ids = [mid for mid, _, _ in vec_candidates]
            sim_map = {mid: sim for mid, _, sim in vec_candidates}
            # 候補のrowを一括取得
            placeholders = ",".join("?" * len(candidate_ids))
            rows_by_id = {}
            for row in conn.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders})", candidate_ids
            ).fetchall():
                rows_by_id[row["id"]] = row

            for mid in candidate_ids:
                row = rows_by_id.get(mid)
                if not row:
                    continue
                sim = sim_map[mid]
                temporal = _temporal_boost(row)
                dw = _domain_weight(domains_map.get(mid), requested_domains) if requested_domains else 1.0
                L = _left_score(row, sim=sim, domain_weight=dw) * temporal
                R = _right_score(conn, row)
                score = corpus_callosum(L, R, balance=0.5)
                score *= (1.0 + fts_bonus_map.get(mid, 0.0))
                scored.append((row, score, sim))
        else:
            # sqlite-vecが使えない場合のフォールバック（旧方式）
            all_rows = conn.execute(
                "SELECT * FROM memories WHERE forgotten = 0 AND embedding IS NOT NULL"
            ).fetchall()
            for row in all_rows:
                mem_vec = bytes_to_vec(row["embedding"])
                sim = cosine_similarity(query_vec, mem_vec)
                temporal = _temporal_boost(row)
                dw = _domain_weight(domains_map.get(row["id"]), requested_domains) if requested_domains else 1.0
                L = _left_score(row, sim=sim, domain_weight=dw) * temporal
                R = _right_score(conn, row)
                score = corpus_callosum(L, R, balance=0.5)
                score *= (1.0 + fts_bonus_map.get(row["id"], 0.0))
                scored.append((row, score, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        scored_results = [(s[0], s[2]) for s in scored[:limit]]

        # 舌先現象 (tip-of-tongue): 類似度0.45-0.65のもやもや記憶を収集
        if fuzzy:
            result_ids = {s[0]["id"] for s in scored_results}
            fuzzy_candidates = [
                (row, sim) for row, _score, sim in scored
                if 0.45 <= sim <= 0.65 and row["id"] not in result_ids
            ]
            fuzzy_candidates.sort(key=lambda x: x[1], reverse=True)
            fuzzy_results = fuzzy_candidates[:3]

        # 不随意記憶（フラッシュバック）+ 意図的復活
        if query_vec is not None:
            forgotten_candidates = vec_search(conn, query_vec, k=50, forgotten=True)
            top_sim = scored[0][2] if scored else 0.0
            for fmid, _, fsim in forgotten_candidates:
                frow = conn.execute("SELECT * FROM memories WHERE id = ?", (fmid,)).fetchone()
                if not frow:
                    continue

                # 意図的復活: 検索結果が乏しいとき、非常に高い類似度で復活
                if top_sim < 0.7 and fsim > 0.92:
                    conn.execute(
                        "UPDATE memories SET forgotten = 0, arousal = 0.3 WHERE id = ?",
                        (frow["id"],)
                    )
                    print(f"  🔮 復活: #{frow['id']} {frow['content'][:50]}... (sim:{fsim:.3f})")
                    scored_results.append((frow, fsim))

                # 不随意記憶: 確率的フラッシュバック
                elif fsim > FLASHBACK_SIM_THRESHOLD:
                    prob = FLASHBACK_BASE_PROB * frow["arousal"] * (fsim - FLASHBACK_SIM_THRESHOLD)
                    if random.random() < prob:
                        conn.execute(
                            "UPDATE memories SET forgotten = 0, arousal = ? WHERE id = ?",
                            (min(1.0, frow["arousal"] + 0.2), frow["id"])
                        )
                        print(f"  💫 フラッシュバック: #{frow['id']} {frow['content'][:50]}...")
                        scored_results.append((frow, fsim))

    # アクセス記録を更新 + 再固定化
    reconsolidated = 0
    for row, _ in scored_results:
        conn.execute(
            "UPDATE memories SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
            (now, row["id"])
        )
        if reconsolidate(conn, row["id"]):
            reconsolidated += 1

    conn.commit()
    conn.close()

    if reconsolidated > 0:
        print(f"  (再固定化: {reconsolidated}件の記憶が変化)")

    # 想起した記憶の情動に引きずられる（感情伝染の逆方向）
    if scored_results:
        recalled_emotions = []
        recalled_arousal = 0.0
        for row, _ in scored_results[:3]:  # 上位3件の情動を反映
            emos = json.loads(row["emotions"]) if row["emotions"] else []
            recalled_emotions.extend(emos)
            recalled_arousal = max(recalled_arousal, row["arousal"])
        if recalled_emotions:
            update_mood(list(set(recalled_emotions)), recalled_arousal * 0.5)

    if fuzzy:
        return scored_results, fuzzy_results
    return scored_results


def search_in_scope(query, session_id=None, topic=None, status=None, limit=20):
    """session / topic / status でスコープした raw_turn 検索。
    v30.2: session_index / topic_thread を namespace として活用する経路。

    - session_id: 指定 session の raw_turn だけ検索
    - topic: topic_thread.key（"topic_<slug>"、prefix 省略可）で session 群を引いて検索
    - status: session_index.stats.status と一致する session 群で検索
      （"solved" で過去の成功例検索、"unsolved" で未決バックログの絞り込み）

    複数指定時は AND（全条件を満たす session のみ対象）。
    戻り値: [{"raw_turn_id": int, "session_id": str, "role": str, "content": str,
              "timestamp": str, "session_title": str | None, "status": str | None}]
    """
    conn = get_connection()
    try:
        session_filter = None  # None = 制約なし、set = 候補集合

        if session_id:
            session_filter = {session_id}

        if topic:
            key = topic if topic.startswith("topic_") else f"topic_{topic}"
            row = conn.execute(
                "SELECT stats FROM catalog_cards WHERE entry_type='topic_thread' AND key=?",
                (key,)
            ).fetchone()
            if not row:
                return []
            try:
                stats = json.loads(row["stats"]) if row["stats"] else {}
            except (json.JSONDecodeError, TypeError):
                stats = {}
            topic_sessions = set(stats.get("session_ids") or [])
            session_filter = (session_filter & topic_sessions) if session_filter is not None else topic_sessions

        if status:
            status_rows = conn.execute(
                "SELECT key, stats FROM catalog_cards WHERE entry_type='session_index'"
            ).fetchall()
            status_sessions = set()
            for r in status_rows:
                try:
                    s = json.loads(r["stats"]) if r["stats"] else {}
                except (json.JSONDecodeError, TypeError):
                    s = {}
                if s.get("status") == status:
                    status_sessions.add(r["key"])
            session_filter = (session_filter & status_sessions) if session_filter is not None else status_sessions

        if session_filter is not None and not session_filter:
            return []

        # session → session_index の title / status を後で付けられるよう取得
        session_meta = {}
        if session_filter is not None:
            placeholders = ",".join("?" * len(session_filter))
            meta_rows = conn.execute(
                f"""SELECT key, title, stats FROM catalog_cards
                    WHERE entry_type='session_index' AND key IN ({placeholders})""",
                tuple(session_filter)
            ).fetchall()
            for r in meta_rows:
                try:
                    s = json.loads(r["stats"]) if r["stats"] else {}
                except (json.JSONDecodeError, TypeError):
                    s = {}
                session_meta[r["key"]] = {"title": r["title"], "status": s.get("status")}

        # FTS5 で検索 + session フィルタ (BM25 降順 + snippet)
        # lindera が MATCH 側で形態素解析するため query は原文のまま。
        rows = []
        has_snippet = False
        try:
            if session_filter is not None:
                placeholders = ",".join("?" * len(session_filter))
                sql = (
                    f"""SELECT raw_turns.*,
                               snippet(raw_turns_fts, 0, '【', '】', '...', 16) AS snippet,
                               bm25(raw_turns_fts) AS bm25_score
                        FROM raw_turns_fts
                        JOIN raw_turns ON raw_turns_fts.turn_id = raw_turns.id
                        WHERE raw_turns_fts MATCH ?
                          AND raw_turns.session_id IN ({placeholders})
                        ORDER BY bm25_score LIMIT ?"""
                )
                params = (query, *tuple(session_filter), limit)
            else:
                sql = (
                    """SELECT raw_turns.*,
                              snippet(raw_turns_fts, 0, '【', '】', '...', 16) AS snippet,
                              bm25(raw_turns_fts) AS bm25_score
                       FROM raw_turns_fts
                       JOIN raw_turns ON raw_turns_fts.turn_id = raw_turns.id
                       WHERE raw_turns_fts MATCH ?
                       ORDER BY bm25_score LIMIT ?"""
                )
                params = (query, limit)
            rows = conn.execute(sql, params).fetchall()
            has_snippet = bool(rows)
        except Exception:
            rows = []

        # FTS AND で 0 件なら tokenize+OR で緩めて再検索
        if not rows:
            or_expr = _tokenize_or_query(query)
            if or_expr:
                try:
                    if session_filter is not None:
                        placeholders = ",".join("?" * len(session_filter))
                        sql = (
                            f"""SELECT raw_turns.*,
                                       snippet(raw_turns_fts, 0, '【', '】', '...', 16) AS snippet,
                                       bm25(raw_turns_fts) AS bm25_score
                                FROM raw_turns_fts
                                JOIN raw_turns ON raw_turns_fts.turn_id = raw_turns.id
                                WHERE raw_turns_fts MATCH ?
                                  AND raw_turns.session_id IN ({placeholders})
                                ORDER BY bm25_score LIMIT ?"""
                        )
                        params = (or_expr, *tuple(session_filter), limit)
                    else:
                        sql = (
                            """SELECT raw_turns.*,
                                      snippet(raw_turns_fts, 0, '【', '】', '...', 16) AS snippet,
                                      bm25(raw_turns_fts) AS bm25_score
                               FROM raw_turns_fts
                               JOIN raw_turns ON raw_turns_fts.turn_id = raw_turns.id
                               WHERE raw_turns_fts MATCH ?
                               ORDER BY bm25_score LIMIT ?"""
                        )
                        params = (or_expr, limit)
                    rows = conn.execute(sql, params).fetchall()
                    has_snippet = bool(rows)
                except Exception:
                    rows = []

        # 最終フォールバック: LIKE (snippet なし)
        if not rows:
            if session_filter is not None:
                placeholders = ",".join("?" * len(session_filter))
                sql = (
                    f"""SELECT * FROM raw_turns
                        WHERE content LIKE ? AND session_id IN ({placeholders})
                        ORDER BY timestamp DESC LIMIT ?"""
                )
                params = (f"%{query}%", *tuple(session_filter), limit)
            else:
                sql = (
                    "SELECT * FROM raw_turns WHERE content LIKE ? "
                    "ORDER BY timestamp DESC LIMIT ?"
                )
                params = (f"%{query}%", limit)
            rows = conn.execute(sql, params).fetchall()

        out = []
        for r in rows:
            sid = r["session_id"]
            meta = session_meta.get(sid, {})
            entry = {
                "raw_turn_id": r["id"],
                "session_id": sid,
                "role": r["role"],
                "content": r["content"],
                "timestamp": r["timestamp"],
                "session_title": meta.get("title"),
                "status": meta.get("status"),
            }
            if has_snippet:
                try:
                    entry["snippet"] = _rejoin_tokenized_snippet(r["snippet"])
                except (IndexError, KeyError):
                    pass
            out.append(entry)
        return out
    finally:
        conn.close()


# ============================================================
# delusion モード — 完全記憶検索（忘却・バイアス・再固定化なし）
# ============================================================

def delusion_search(query=None, limit=50, date=None, after=None, before=None,
                    raw_only=False, context_id=None, dump_all=False):
    """
    delusionモード: 一切のノイズなく事実を完璧に引き出す。

    3原則:
    1. 忘却曲線の無効化: freshnessを常に1.0、干渉忘却をスキップ
    2. 情動・気分バイアスの排除: 全ブースト係数を1.0
    3. 再固定化の停止: 読み取り専用（DBを一切変更しない）

    Args:
        query: 検索クエリ（Noneの場合は日付やdump_allで絞り込み）
        limit: 最大件数
        date: 特定日付 (YYYY-MM-DD)
        after: 期間開始 (YYYY-MM-DD)
        before: 期間終了 (YYYY-MM-DD)
        raw_only: raw_turnsのみ検索
        context_id: 記憶IDから元の対話文脈を復元
        dump_all: 全件ダンプ
    """
    conn = get_connection()

    # --- context_id: 記憶IDから対話文脈を復元 ---
    if context_id is not None:
        return _delusion_context(conn, context_id)

    # --- raw_only: raw_turnsのみ検索 ---
    if raw_only and query:
        return _delusion_raw_search(conn, query, limit, date, after, before)

    # --- 日付フィルタの構築 ---
    date_clause = ""
    date_params = []
    if date:
        date_clause = " AND created_at >= ? AND created_at < ?"
        date_params = [f"{date}T00:00:00Z", f"{date}T99:99:99Z"]
        # 日付の翌日を計算
        from datetime import date as date_type
        d = datetime.strptime(date, "%Y-%m-%d")
        next_day = (d + timedelta(days=1)).strftime("%Y-%m-%d")
        date_params = [f"{date}T00:00:00Z", f"{next_day}T00:00:00Z"]
    else:
        if after:
            date_clause += " AND created_at >= ?"
            date_params.append(f"{after}T00:00:00Z")
        if before:
            date_clause += " AND created_at < ?"
            next_day = (datetime.strptime(before, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            date_params.append(f"{next_day}T00:00:00Z")

    # --- dump_all or date-only (no query): 全件ダンプ ---
    if dump_all or (not query and (date or after or before)):
        rows = conn.execute(
            f"SELECT * FROM memories WHERE 1=1{date_clause} ORDER BY created_at DESC LIMIT ?",
            (*date_params, limit)
        ).fetchall()
        conn.close()
        return [(_row_to_delusion_format(row), None) for row in rows]

    if not query:
        conn.close()
        return []

    # --- ベクトル検索 (sqlite-vec + FTS5ハイブリッド) ---
    results = []

    # FTS5検索を先に試す（テキスト完全一致に強い、bm25 で語の希少性を重みに）
    # ヒット 0 のときは tokenize+OR フォールバックで拾い直す。
    fts_bonus_map = _bm25_bonus_map(conn, query, "memories_fts", "memory_id",
                                    max_rows=max(limit * 5, 200))
    if not fts_bonus_map:
        or_expr = _tokenize_or_query(query)
        if or_expr:
            try:
                or_rows = conn.execute(
                    """SELECT memory_id AS id, bm25(memories_fts) AS s
                       FROM memories_fts WHERE memories_fts MATCH ?
                       ORDER BY s LIMIT ?""",
                    (or_expr, max(limit * 5, 200))
                ).fetchall()
                if or_rows:
                    scores = [r["s"] for r in or_rows]
                    s_min, s_max = min(scores), max(scores)
                    rng = (s_max - s_min) or 1.0
                    # OR フォールバックは半信半疑なので上限を 0.05 に抑える
                    fts_bonus_map = {r["id"]: 0.02 + 0.03 * (s_max - r["s"]) / rng
                                     for r in or_rows}
            except Exception:
                pass

    # ベクトル検索
    import numpy as np
    query_vec = embed_text(query, is_query=True)

    if query_vec is not None:
        # sqlite-vecで候補取得（forgotten含む全件から）
        vec_candidates = vec_search(conn, query_vec, k=max(limit * 5, 200), forgotten=None)

        if vec_candidates:
            # 日付フィルタ + BM25 重みボーナス
            scored = []
            for mid, _, sim in vec_candidates:
                row = conn.execute(
                    f"SELECT * FROM memories WHERE id = ?{date_clause}",
                    (mid, *date_params)
                ).fetchone()
                if not row:
                    continue
                fts_bonus = fts_bonus_map.get(row["id"], 0.0)
                scored.append((row, sim + fts_bonus, sim))
        else:
            # フォールバック: 旧方式
            all_rows = conn.execute(
                f"SELECT * FROM memories WHERE embedding IS NOT NULL{date_clause}",
                date_params
            ).fetchall()
            scored = []
            for row in all_rows:
                mem_vec = bytes_to_vec(row["embedding"])
                sim = cosine_similarity(query_vec, mem_vec)
                fts_bonus = fts_bonus_map.get(row["id"], 0.0)
                scored.append((row, sim + fts_bonus, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        # 馴化（habituation）: 類似内容の繰り返しを減衰させる
        habituated = []
        selected_vecs = []
        HABITUATION_THRESHOLD = 0.92
        HABITUATION_DECAY = 0.7
        for row, boosted_sim, raw_sim in scored:
            mem_vec = bytes_to_vec(row["embedding"])
            decay = 1.0
            for sv in selected_vecs:
                inter_sim = cosine_similarity(mem_vec, sv)
                if inter_sim > HABITUATION_THRESHOLD:
                    decay *= HABITUATION_DECAY
            habituated.append((row, boosted_sim * decay, raw_sim * decay))
            selected_vecs.append(mem_vec)

        habituated.sort(key=lambda x: x[1], reverse=True)
        results = [(_row_to_delusion_format(row), sim) for row, _, sim in habituated[:limit]]
    else:
        # embeddingなし: LIKE検索フォールバック
        rows = conn.execute(
            f"SELECT * FROM memories WHERE content LIKE ?{date_clause} ORDER BY created_at DESC LIMIT ?",
            (f"%{query}%", *date_params, limit)
        ).fetchall()
        results = [(_row_to_delusion_format(row), None) for row in rows]

    # raw_turnsも検索して結果に追加
    raw_results = _delusion_raw_search(conn, query, min(limit, 20), date, after, before)
    if raw_results:
        results.extend(raw_results)

    conn.close()
    return results


def _row_to_delusion_format(row):
    """記憶rowをdelusionモード用のメタデータ付きdictに変換。"""
    # originカラムが存在しない古いDBでも動くように
    try:
        origin_val = row["origin"]
    except (IndexError, KeyError):
        origin_val = None
    return {
        "id": row["id"],
        "content": row["content"],  # 切り詰めなし
        "category": row["category"],
        "created_at": row["created_at"],
        "arousal": row["arousal"],
        "importance": row["importance"],
        "emotions": row["emotions"],
        "keywords": row["keywords"],
        "access_count": row["access_count"],
        "forgotten": row["forgotten"],
        "origin": origin_val,
        "source": "memory",
    }


def _delusion_raw_search(conn, query, limit=20, date=None, after=None, before=None):
    """raw_turnsテーブルから原文検索。"""
    # raw_turnsテーブルが存在するか確認
    try:
        conn.execute("SELECT count(*) FROM raw_turns").fetchone()
    except Exception:
        return []

    date_clause = ""
    date_params = []
    if date:
        d = datetime.strptime(date, "%Y-%m-%d")
        next_day = (d + timedelta(days=1)).strftime("%Y-%m-%d")
        date_clause = " AND timestamp >= ? AND timestamp < ?"
        date_params = [f"{date}T00:00:00Z", f"{next_day}T00:00:00Z"]
    else:
        if after:
            date_clause += " AND timestamp >= ?"
            date_params.append(f"{after}T00:00:00Z")
        if before:
            next_day = (datetime.strptime(before, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            date_clause += " AND timestamp < ?"
            date_params.append(f"{next_day}T00:00:00Z")

    # FTS5検索（lindera 形態素解析）: BM25 降順 + ヒット箇所 snippet
    try:
        fts_rows = conn.execute(
            f"""SELECT raw_turns.*,
                       snippet(raw_turns_fts, 0, '【', '】', '...', 16) AS snippet,
                       bm25(raw_turns_fts) AS bm25_score
                FROM raw_turns_fts
                JOIN raw_turns ON raw_turns_fts.turn_id = raw_turns.id
                WHERE raw_turns_fts MATCH ?{date_clause}
                ORDER BY bm25_score LIMIT ?""",
            (query, *date_params, limit)
        ).fetchall()
        if fts_rows:
            return [(_raw_turn_to_format(row, snippet=_rejoin_tokenized_snippet(row["snippet"])), None)
                    for row in fts_rows]
    except Exception:
        pass

    # FTS5 AND で 0 件なら tokenize+OR で緩めて再検索
    or_expr = _tokenize_or_query(query)
    if or_expr:
        try:
            or_rows = conn.execute(
                f"""SELECT raw_turns.*,
                           snippet(raw_turns_fts, 0, '【', '】', '...', 16) AS snippet,
                           bm25(raw_turns_fts) AS bm25_score
                    FROM raw_turns_fts
                    JOIN raw_turns ON raw_turns_fts.turn_id = raw_turns.id
                    WHERE raw_turns_fts MATCH ?{date_clause}
                    ORDER BY bm25_score LIMIT ?""",
                (or_expr, *date_params, limit)
            ).fetchall()
            if or_rows:
                return [(_raw_turn_to_format(row, snippet=_rejoin_tokenized_snippet(row["snippet"])), None)
                        for row in or_rows]
        except Exception:
            pass

    # 最終フォールバック: LIKE 検索
    rows = conn.execute(
        f"SELECT * FROM raw_turns WHERE content LIKE ?{date_clause} ORDER BY timestamp DESC LIMIT ?",
        (f"%{query}%", *date_params, limit)
    ).fetchall()
    return [(_raw_turn_to_format(row), None) for row in rows]


def _raw_turn_to_format(row, snippet=None):
    """raw_turn rowをdelusion用フォーマットに変換。

    snippet: FTS5 snippet() の結果 (任意)。ヒット位置中心の抜粋表示用。
    """
    out = {
        "id": f"raw:{row['id']}",
        "content": row["content"],
        "category": f"raw_turn ({row['role']})",
        "created_at": row["timestamp"],
        "arousal": None,
        "importance": None,
        "emotions": None,
        "keywords": None,
        "access_count": None,
        "forgotten": 0,
        "source": "raw_turn",
        "session_id": row["session_id"],
        "role": row["role"],
    }
    if snippet:
        out["snippet"] = snippet
    return out


def _delusion_context(conn, memory_id):
    """記憶IDから元の対話文脈を復元する。文字列 'raw:123' も受け付ける。"""
    # raw:XXX 形式の場合: そのraw_turnの前後を返す
    raw_prefix = str(memory_id)
    if raw_prefix.startswith("raw:"):
        raw_id = int(raw_prefix[4:])
        # そのraw_turnのsession_idを取得し、同セッションの前後20件を返す
        try:
            target = conn.execute("SELECT * FROM raw_turns WHERE id = ?", (raw_id,)).fetchone()
            if target:
                rows = conn.execute(
                    "SELECT * FROM raw_turns WHERE session_id = ? AND id BETWEEN ? AND ? ORDER BY id",
                    (target["session_id"], raw_id - 10, raw_id + 10)
                ).fetchall()
                if rows:
                    conn.close()
                    return [(_raw_turn_to_format(row), None) for row in rows]
        except Exception:
            pass
        conn.close()
        return []

    # 通常の記憶ID
    memory_id = int(memory_id)

    # raw_turnsのmemory_idsから該当する対話を探す
    try:
        rows = conn.execute(
            "SELECT * FROM raw_turns WHERE memory_ids LIKE ? ORDER BY timestamp",
            (f"%{memory_id}%",)
        ).fetchall()
        if rows:
            conn.close()
            return [(_raw_turn_to_format(row), None) for row in rows]
    except Exception:
        pass

    # memory_idsに紐付けがなければ、記憶のsource_conversationから探す
    mem = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if mem and mem["source_conversation"]:
        session_id = mem["source_conversation"]
        try:
            rows = conn.execute(
                "SELECT * FROM raw_turns WHERE session_id = ? ORDER BY timestamp",
                (session_id,)
            ).fetchall()
            if rows:
                conn.close()
                return [(_raw_turn_to_format(row), None) for row in rows]
        except Exception:
            pass

    conn.close()
    return []


def format_delusion(item, similarity=None):
    """delusionモードの出力フォーマット（機械可読メタデータ付き）。"""
    if isinstance(item, tuple):
        item, similarity = item

    mid = item.get("id", "?")
    created = item.get("created_at", "?")
    category = item.get("category", "?")
    arousal = item.get("arousal")
    content = item.get("content", "")
    forgotten = item.get("forgotten", 0)

    origin = item.get("origin")

    # メタデータ行
    parts = [f"[ID:{mid}]", f"[{created}]", f"[category:{category}]"]
    if arousal is not None:
        parts.append(f"[arousal:{arousal:.2f}]")
    if origin:
        parts.append(f"[origin:{origin}]")
    if forgotten:
        parts.append("[FORGOTTEN]")
    if similarity is not None:
        parts.append(f"[sim:{similarity:.3f}]")

    meta_line = " ".join(parts)
    return f"{meta_line}\n{content}"


def repair_memory_raw_links(limit=500, window_days=1):
    """raw_turns.memory_ids に未登録の memory を、FTS + 時間窓で対応する
    raw_turn に紐付ける補修バッチ。

    対象: forgotten=0 かつ content あり の memory のうち、どの raw_turn からも
    参照されていないもの。
    探し方:
      1. memory.created_at 前後 window_days 日の raw_turn に絞る
      2. memory.content 先頭 200 文字を tokenize して raw_turns_fts で MATCH
      3. hit あれば最初の 1 件に memory.id を追記（重複は自動 skip）
    window_days で見つからない場合は次回再挑戦（段階的に窓を広げる運用は呼び側）。
    """
    from tokenizer import tokenize
    conn = get_connection()
    try:
        # 既にリンク済みの memory id 集合
        linked = set()
        for r in conn.execute(
            "SELECT memory_ids FROM raw_turns WHERE memory_ids != '[]' AND memory_ids IS NOT NULL"
        ):
            try:
                for mid in json.loads(r["memory_ids"] or "[]"):
                    if isinstance(mid, int):
                        linked.add(mid)
            except (json.JSONDecodeError, TypeError):
                continue

        # orphan memory を importance 降順 / id 新しい順で取る
        mem_rows = conn.execute(
            """SELECT id, content, created_at FROM memories
               WHERE forgotten = 0 AND content IS NOT NULL AND content != ''
               ORDER BY importance DESC, id DESC"""
        ).fetchall()

        stats = {"scanned": 0, "linked": 0, "no_match": 0,
                 "already_linked": 0, "fts_error": 0}
        for m in mem_rows:
            if stats["linked"] >= limit:
                break
            if m["id"] in linked:
                stats["already_linked"] += 1
                continue
            stats["scanned"] += 1
            content = (m["content"] or "").strip()
            if not content:
                continue

            try:
                q_raw = tokenize(content[:200])
            except Exception:
                stats["fts_error"] += 1
                continue
            if not q_raw or not q_raw.strip():
                continue
            # FTS5 MATCH は AND デフォルトで厳しすぎるので、上位 10 token を OR 結合。
            # 短すぎる token（1 文字）は除外。
            tokens = [t for t in q_raw.split() if len(t) > 1][:10]
            if not tokens:
                stats["fts_error"] += 1
                continue
            q = " OR ".join(tokens)

            # 時間窓で絞った FTS 検索
            date_clause = ""
            date_params = ()
            created = m["created_at"]
            if created:
                # created_at は ISO 8601 想定、±window_days の範囲
                try:
                    dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    lo = (dt - timedelta(days=window_days)).strftime('%Y-%m-%dT%H:%M:%SZ')
                    hi = (dt + timedelta(days=window_days)).strftime('%Y-%m-%dT%H:%M:%SZ')
                    date_clause = " AND raw_turns.timestamp BETWEEN ? AND ?"
                    date_params = (lo, hi)
                except (ValueError, TypeError):
                    pass

            try:
                rt = conn.execute(
                    f"""SELECT raw_turns.id, raw_turns.memory_ids
                        FROM raw_turns_fts
                        JOIN raw_turns ON raw_turns_fts.turn_id = raw_turns.id
                        WHERE raw_turns_fts MATCH ?{date_clause}
                        ORDER BY raw_turns.timestamp ASC LIMIT 1""",
                    (q, *date_params)
                ).fetchone()
            except Exception:
                stats["fts_error"] += 1
                continue

            if not rt:
                stats["no_match"] += 1
                continue

            try:
                ids = json.loads(rt["memory_ids"] or "[]")
            except (json.JSONDecodeError, TypeError):
                ids = []
            if m["id"] in ids:
                continue
            ids.append(m["id"])
            conn.execute(
                "UPDATE raw_turns SET memory_ids = ? WHERE id = ?",
                (json.dumps(ids), rt["id"])
            )
            stats["linked"] += 1
            linked.add(m["id"])

        conn.commit()
        return stats
    finally:
        conn.close()


def save_raw_turn(session_id, role, content, timestamp=None,
                  message_uuid=None, cwd=None, git_branch=None, memory_ids=None,
                  model=None):
    """対話の1ターンをraw_turnsテーブルに保存する。"""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    conn = get_connection()

    # raw_turnsテーブルの存在確認
    try:
        conn.execute("SELECT 1 FROM raw_turns LIMIT 0")
    except Exception:
        conn.close()
        return None

    # modelカラムがなければ追加
    try:
        conn.execute("SELECT model FROM raw_turns LIMIT 0")
    except Exception:
        conn.execute("ALTER TABLE raw_turns ADD COLUMN model TEXT")
        conn.commit()

    conn.execute(
        """INSERT INTO raw_turns
           (session_id, message_uuid, role, content, timestamp, cwd, git_branch, memory_ids, model)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, message_uuid, role, content, timestamp,
         cwd, git_branch, json.dumps(memory_ids or []), model)
    )
    conn.commit()
    new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # FTS5にも追加（lindera が tokenize するため content そのまま）
    try:
        conn.execute(
            "INSERT INTO raw_turns_fts (content, turn_id) VALUES (?, ?)",
            (content, new_id)
        )
        conn.commit()
    except Exception:
        pass

    # felt_moments — role='assistant' の場合、自己報告された感情を抽出して mark
    if role == "assistant" and content:
        try:
            import felt_emotions as _fe
            moments = _fe.extract_from_text(content)
            if moments:
                for m in moments:
                    conn.execute(
                        "INSERT INTO felt_moments "
                        "(turn_id, label, phrase, span_start, span_end, surrounding) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (new_id, m["label"], m["phrase"],
                         m["span_start"], m["span_end"], m["surrounding"])
                    )
                conn.commit()
        except Exception:
            pass  # fail-soft: 抽出失敗しても turn 保存は成功させる

    conn.close()
    return new_id


def promote_turns(days=7, sample_size=20):
    """
    睡眠中の記憶固定化: 海馬リプレイの模倣。

    直近N日のraw_turnsからランダムにサンプリングし、add_memoryに渡す。
    - 覚醒度が高い発言は重み付きで選ばれやすい（情動タグ付き再生）
    - 既知の記憶は予測符号化(prediction_error)で自然に弾かれる
    - フラグ管理しない。何度リプレイされてもOK
    """
    import random

    conn = get_connection()

    # 直近N日のユーザー発言を取得
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ')
    turns = conn.execute(
        "SELECT id, content, timestamp FROM raw_turns "
        "WHERE role = 'user' AND timestamp > ? ORDER BY timestamp",
        (cutoff,)
    ).fetchall()
    conn.close()

    if not turns:
        print("リプレイ対象のターンはありません")
        return 0

    # クリーニング＋フィルタ
    TRIVIAL = {"y", "n", "ok", "yes", "no", "はい", "いいえ", "うん", "おk",
               "1", "2", "3", "a", "b", "c", "了解", "わかった", "いいよ",
               "まあいいか", "ありがとう", "thanks"}

    def _clean(text):
        text = re.sub(r'<[a-z_-]+>.*?</[a-z_-]+>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[a-z_-]+\s*/>', '', text)
        text = re.sub(r'toolu_[a-z0-9]+', '', text)
        text = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '', text)
        text = re.sub(r'[0-9a-f]{20,}', '', text)
        text = re.sub(r'\n{3,}', '\n', text)
        text = re.sub(r'  +', ' ', text)
        return text.strip()

    candidates = []
    for t in turns:
        cleaned = _clean(t["content"])
        if not cleaned or len(cleaned) < 20:
            continue
        if cleaned.lower().strip() in TRIVIAL:
            continue
        # 情動検出で重みを決める（覚醒度が高い＝リプレイされやすい）
        _, arousal, _ = detect_emotions(cleaned)
        weight = max(0.1, arousal + 0.1)  # 最低0.1、覚醒度が高いほど重い
        # フラッシュバルブ抽出（高覚醒度のみ）
        fb = _extract_flashbulb_sentence(cleaned) if arousal >= FLASHBULB_AROUSAL_THRESHOLD else None
        candidates.append((cleaned, weight, fb))

    if not candidates:
        print("意味のあるターンがありません")
        return 0

    # 重み付きサンプリング（脳のシャ��プウェーブリップル）
    k = min(sample_size, len(candidates))
    weights = [w for _, w, _ in candidates]
    sampled = random.choices(candidates, weights=weights, k=k)
    # 重複除去（同じ発言��複数回選ばれる��とがある）
    seen = set()
    unique = []
    for content, _, fb in sampled:
        snippet = content[:200]
        if snippet not in seen:
            seen.add(snippet)
            unique.append((snippet, fb))

    promoted = 0
    for snippet, fb in unique:
        add_memory(
            content=snippet,
            category="episode",
            provenance="sleep_promote",
            confidence=0.6,
            flashbulb=fb,
            origin="system:sleep",
        )
        promoted += 1

    print(f"✓ {promoted}件のターンをリプレイ → 記憶に固定化")
    return promoted


def review_memories(n=5):
    """
    間隔反復 (Spaced Repetition): SM-2インスパイアの優先度で復習が必要な記憶を選ぶ。

    Priority = importance * (1 / (access_count + 1)) * (days_since_last_access / HALF_LIFE_DAYS)
    高い重要度 + 低いアクセス数 + 長い未アクセス期間 = 復習の必要性が高い
    """
    conn = get_connection()
    now = datetime.now(timezone.utc)
    now_str = now.strftime('%Y-%m-%dT%H:%M:%SZ')

    rows = conn.execute(
        "SELECT * FROM memories WHERE forgotten = 0"
    ).fetchall()

    if not rows:
        print("復習する記憶がありません")
        conn.close()
        return []

    scored = []
    for row in rows:
        # 最終アクセスからの日数を計算
        if row["last_accessed"]:
            try:
                last = datetime.fromisoformat(row["last_accessed"].replace('Z', '+00:00'))
                days_since = (now - last).total_seconds() / 86400.0
            except (ValueError, AttributeError):
                days_since = HALF_LIFE_DAYS
        else:
            # 一度もアクセスされていない → 作成日からの日数
            try:
                created = datetime.fromisoformat(row["created_at"].replace('Z', '+00:00'))
                days_since = (now - created).total_seconds() / 86400.0
            except (ValueError, AttributeError):
                days_since = HALF_LIFE_DAYS

        priority = row["importance"] * (1.0 / (row["access_count"] + 1)) * (days_since / HALF_LIFE_DAYS)
        scored.append((row, priority))

    scored.sort(key=lambda x: x[1], reverse=True)
    review_list = scored[:n]

    # 再構成モードで表示
    print(f"間隔反復レビュー ({len(review_list)}件):")
    for row, priority in review_list:
        print(format_memory_reconstructive(conn, row))

    # アクセス記録を更新（復習としてカウント）
    for row, _ in review_list:
        conn.execute(
            "UPDATE memories SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
            (now_str, row["id"])
        )

    conn.commit()
    conn.close()

    print(f"✓ {len(review_list)}件の記憶をリプレイしました")
    return review_list


def chain_memories(memory_id, depth=2):
    conn = get_connection()
    visited = set()
    result = []

    def _traverse(mid, d):
        if d <= 0 or mid in visited:
            return
        visited.add(mid)
        row = conn.execute("SELECT * FROM memories WHERE id = ? AND forgotten = 0", (mid,)).fetchone()
        if not row:
            return
        result.append((row, depth - d))
        links = conn.execute(
            """SELECT target_id, strength FROM links
               WHERE source_id = ? ORDER BY strength DESC LIMIT 5""",
            (mid,)
        ).fetchall()
        for link in links:
            _traverse(link["target_id"], d - 1)

    _traverse(memory_id, depth)
    conn.close()
    return result


# v30 graph navigation — pull 型ハンドル返し
def get_neighbors(memory_id, limit=10):
    """指定ノードの隣接（1 歩先）を返す。strength 降順。"""
    conn = get_connection()
    rows = conn.execute(
        """SELECT m.*, l.strength FROM links l
           JOIN memories m ON m.id = l.target_id
           WHERE l.source_id = ? AND m.forgotten = 0
           ORDER BY l.strength DESC LIMIT ?""",
        (memory_id, limit)
    ).fetchall()
    results = [(row, row["strength"]) for row in rows]
    conn.close()
    return results


def walk_from(memory_id, depth=2, per_hop=5):
    """BFS: memory_id から depth 歩先まで展開。各ノードに最短距離を付ける。"""
    conn = get_connection()
    visited = {}  # id -> (row, dist)
    frontier = [memory_id]
    start_row = conn.execute(
        "SELECT * FROM memories WHERE id = ? AND forgotten = 0", (memory_id,)
    ).fetchone()
    if not start_row:
        conn.close()
        return []
    visited[memory_id] = (start_row, 0)

    for d in range(1, depth + 1):
        next_frontier = []
        for mid in frontier:
            links = conn.execute(
                """SELECT l.target_id, l.strength, m.* FROM links l
                   JOIN memories m ON m.id = l.target_id
                   WHERE l.source_id = ? AND m.forgotten = 0
                   ORDER BY l.strength DESC LIMIT ?""",
                (mid, per_hop)
            ).fetchall()
            for lr in links:
                tid = lr["target_id"]
                if tid not in visited:
                    visited[tid] = (lr, d)
                    next_frontier.append(tid)
        frontier = next_frontier
        if not frontier:
            break

    conn.close()
    return sorted(visited.values(), key=lambda x: (x[1], -(x[0]["importance"] or 0)))


def nodes_at_domain(domain, limit=15):
    """指定 domain の記憶を importance × last_accessed 順で返す。prefix 一致。"""
    conn = get_connection()
    rows = conn.execute(
        """SELECT m.* FROM memories m JOIN cortex c ON c.id = m.id
           WHERE m.forgotten = 0
             AND (c.domains LIKE ? OR c.domains LIKE ?)
           ORDER BY m.importance DESC, m.last_accessed DESC LIMIT ?""",
        (f'%"{domain}"%', f'%"{domain}/%',   limit)
    ).fetchall()
    conn.close()
    return rows


def mutate_metadata(conn):
    """
    メタデータ変容: 隣接記憶の影響でキーワード・埋め込み・情動が変化する。
    データ（content）は不変、メタデータが変容する。

    不変条件 (v29): cortex.domains は変異対象に含めない。凡例は territory の
    揺らぎで勝手に動かない。domain を変えるのは _domain_cli 経由の明示操作のみ。
    """
    import numpy as np
    now = datetime.now(timezone.utc)
    now_str = now.strftime('%Y-%m-%dT%H:%M:%SZ')
    stats = {"keywords": 0, "embeddings": 0, "emotions": 0}

    # 除外対象のカテゴリ
    exclude_categories = {"schema"}

    # 変容に必要なカラムを全て取得
    rows = conn.execute(
        "SELECT id, content, category, keywords, emotions, embedding, last_mutated "
        "FROM memories WHERE forgotten = 0 AND embedding IS NOT NULL"
    ).fetchall()
    if len(rows) < 2:
        return stats

    # 各記憶の隣接リンクを事前取得
    all_links = conn.execute(
        "SELECT source_id, target_id, strength FROM links"
    ).fetchall()
    neighbors = {}  # memory_id -> [(neighbor_id, strength)]
    for link in all_links:
        neighbors.setdefault(link["source_id"], []).append(
            (link["target_id"], link["strength"])
        )

    # rowsをidで引けるようにする
    row_by_id = {r["id"]: r for r in rows}

    for row in rows:
        mid = row["id"]

        # 除外チェック: カテゴリ
        if row["category"] in exclude_categories:
            continue

        # 除外チェック: クールダウン
        last_mut = row["last_mutated"] if "last_mutated" in row.keys() else None
        if last_mut:
            try:
                last_mut_dt = datetime.strptime(last_mut, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
                if (now - last_mut_dt).total_seconds() < MUTATION_COOLDOWN_HOURS * 3600:
                    continue
            except (ValueError, TypeError):
                pass

        my_neighbors = neighbors.get(mid, [])
        num_links = len(my_neighbors)
        mutated = False

        # --- 1. キーワード吸収 ---
        if num_links >= MUTATION_MIN_LINKS_KW:
            my_keywords = json.loads(row["keywords"]) if row["keywords"] else []
            original_count = len(my_keywords)
            if original_count > 0:
                cap = int(original_count * MUTATION_KW_CAP_RATIO)
                my_kw_set = set(my_keywords)

                # 隣接記憶のキーワードをstrengthで重み付け収集
                candidate_scores = {}
                for nid, strength in my_neighbors:
                    n_row = row_by_id.get(nid)
                    if not n_row:
                        continue
                    n_kw = json.loads(n_row["keywords"]) if n_row["keywords"] else []
                    for kw in n_kw:
                        if kw not in my_kw_set:
                            candidate_scores[kw] = candidate_scores.get(kw, 0) + strength

                # 上位3候補を確率的に吸収
                top_candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])[:3]
                added = []
                for kw, _score in top_candidates:
                    if len(added) >= MUTATION_KW_MAX_ADD:
                        break
                    if len(my_keywords) >= cap:
                        break
                    if random.random() < MUTATION_KW_ABSORB_PROB:
                        my_keywords.append(kw)
                        added.append(kw)

                # 最も孤立したキーワードを削除（隣接記憶に最も出現しないもの）
                removed = []
                if added and len(my_keywords) > original_count:
                    # 各キーワードの隣接出現カウント
                    neighbor_kw_counts = {}
                    for nid, _s in my_neighbors:
                        n_row = row_by_id.get(nid)
                        if not n_row:
                            continue
                        n_kw = json.loads(n_row["keywords"]) if n_row["keywords"] else []
                        for kw in n_kw:
                            neighbor_kw_counts[kw] = neighbor_kw_counts.get(kw, 0) + 1
                    # 元のキーワードの中で最も孤立しているもの
                    original_kws = [kw for kw in my_keywords if kw not in added]
                    if original_kws:
                        most_isolated = min(original_kws, key=lambda kw: neighbor_kw_counts.get(kw, 0))
                        if neighbor_kw_counts.get(most_isolated, 0) == 0:
                            my_keywords.remove(most_isolated)
                            removed.append(most_isolated)

                if added or removed:
                    old_kw = json.loads(row["keywords"]) if row["keywords"] else []
                    new_kw_json = json.dumps(my_keywords, ensure_ascii=False)
                    conn.execute("UPDATE memories SET keywords = ? WHERE id = ?",
                                 (new_kw_json, mid))
                    neighbor_ids = [str(nid) for nid, _ in my_neighbors[:5]]
                    reason_parts = []
                    if added:
                        reason_parts.append(f"+{added}")
                    if removed:
                        reason_parts.append(f"-{removed}")
                    conn.execute(
                        "INSERT INTO mutation_log (memory_id, field, old_value, new_value, reason) "
                        "VALUES (?, 'keywords', ?, ?, ?)",
                        (mid,
                         json.dumps(old_kw, ensure_ascii=False),
                         new_kw_json,
                         f"neighbor_absorption: #{',#'.join(neighbor_ids)}: {' '.join(reason_parts)}")
                    )
                    stats["keywords"] += 1
                    mutated = True

        # --- 2. 埋め込みドリフト ---
        if num_links >= MUTATION_MIN_LINKS_EMBED and row["embedding"]:
            neighbor_embeddings = []
            for nid, _s in my_neighbors:
                n_row = row_by_id.get(nid)
                if n_row and n_row["embedding"]:
                    neighbor_embeddings.append(bytes_to_vec(n_row["embedding"]))

            if len(neighbor_embeddings) >= MUTATION_MIN_LINKS_EMBED:
                current_vec = bytes_to_vec(row["embedding"])
                centroid = np.mean(neighbor_embeddings, axis=0)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 0:
                    centroid = centroid / centroid_norm
                    new_vec = (1 - MUTATION_EMBED_ALPHA) * current_vec + MUTATION_EMBED_ALPHA * centroid
                    new_norm = np.linalg.norm(new_vec)
                    if new_norm > 0:
                        new_vec = new_vec / new_norm
                        shift = float(1.0 - np.dot(current_vec, new_vec))
                        new_blob = vec_to_bytes(new_vec)
                        conn.execute("UPDATE memories SET embedding = ? WHERE id = ?",
                                     (new_blob, mid))
                        _sync_vec_insert(conn, mid, new_blob)
                        neighbor_ids = [str(nid) for nid, _ in my_neighbors[:5]]
                        conn.execute(
                            "INSERT INTO mutation_log (memory_id, field, old_value, new_value, reason) "
                            "VALUES (?, 'embedding', ?, ?, ?)",
                            (mid,
                             f"shift={shift:.6f}",
                             f"alpha={MUTATION_EMBED_ALPHA}",
                             f"centroid_drift: #{',#'.join(neighbor_ids)}")
                        )
                        stats["embeddings"] += 1
                        mutated = True

        # --- 3. 情動ドリフト ---
        if num_links >= MUTATION_MIN_LINKS_EMOTION:
            # content + 隣接記憶の先頭50文字でdetect_emotions再実行
            context_text = row["content"] if row["content"] else ""
            for nid, _s in my_neighbors[:5]:
                n_row = row_by_id.get(nid)
                if n_row and n_row["content"]:
                    context_text += " " + n_row["content"][:50]

            new_emotions_raw = detect_emotions(context_text)
            new_emotion_names = new_emotions_raw[0] if new_emotions_raw else []
            current_emotions = json.loads(row["emotions"]) if row["emotions"] else []
            current_set = set(current_emotions)

            added_emotions = []
            for emo in new_emotion_names:
                if emo not in current_set:
                    if random.random() < 0.2:  # P=0.2で追加
                        added_emotions.append(emo)

            if added_emotions:
                updated_emotions = current_emotions + added_emotions
                new_emo_json = json.dumps(updated_emotions, ensure_ascii=False)
                conn.execute("UPDATE memories SET emotions = ? WHERE id = ?",
                             (new_emo_json, mid))
                neighbor_ids = [str(nid) for nid, _ in my_neighbors[:5]]
                conn.execute(
                    "INSERT INTO mutation_log (memory_id, field, old_value, new_value, reason) "
                    "VALUES (?, 'emotions', ?, ?, ?)",
                    (mid,
                     json.dumps(current_emotions, ensure_ascii=False),
                     new_emo_json,
                     f"emotion_drift: #{',#'.join(neighbor_ids)}: +{added_emotions}")
                )
                stats["emotions"] += 1
                mutated = True

        # updated_at と last_mutated を更新
        if mutated:
            conn.execute(
                "UPDATE memories SET last_mutated = ?, updated_at = ? WHERE id = ?",
                (now_str, now_str, mid)
            )

    return stats


def replay_memories():
    """
    リプレイ: リンク再計算 + 弱い記憶の自動忘却 + 統合提案
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, embedding, importance, arousal, access_count, created_at, category, emotions "
        "FROM memories WHERE forgotten = 0 AND embedding IS NOT NULL"
    ).fetchall()

    if len(rows) < 2:
        print("リプレイするには記憶が足りません")
        conn.close()
        return

    # 1. シナプスホメオスタシス（Tononi SHY）
    #    全リンクのstrengthを一律減衰させ、閾値以下を刈り込む。
    #    外傷的記憶（arousal >= 閾値）に繋がるリンクは減衰を免除。
    arousal_by_id = {row["id"]: row["arousal"] for row in rows}
    emotions_by_id = {}
    for row in rows:
        emotions_by_id[row["id"]] = json.loads(row["emotions"]) if row["emotions"] else []
    created_at_by_id = {row["id"]: row["created_at"] for row in rows}

    existing_links = conn.execute(
        "SELECT id, source_id, target_id, strength, link_type FROM links"
    ).fetchall()
    pruned = 0
    downscaled = 0
    for link in existing_links:
        # tension link は Hebbian 強度ではなく対立の記録。replay の恒常性減衰は
        # 対象外（detect_tensions で作られ、tension forget でだけ消える）。
        if link["link_type"] == "tension":
            continue
        src_arousal = arousal_by_id.get(link["source_id"], 0)
        tgt_arousal = arousal_by_id.get(link["target_id"], 0)
        # 外傷的記憶に繋がるリンクは減衰を免除
        if src_arousal >= TRAUMA_AROUSAL_THRESHOLD or tgt_arousal >= TRAUMA_AROUSAL_THRESHOLD:
            continue
        new_strength = link["strength"] * 0.9
        if new_strength < LINK_THRESHOLD:
            conn.execute("DELETE FROM links WHERE id = ?", (link["id"],))
            pruned += 1
        else:
            conn.execute("UPDATE links SET strength = ? WHERE id = ?",
                         (new_strength, link["id"]))
            downscaled += 1

    # 2. 海馬リプレイ — 3軸でリンクを選択的に強化
    #    (1) 高arousal: 重要だが外傷的ではない記憶間
    #    (2) surprise: 報酬予測誤差（ドーパミン）の模倣
    #    (3) 時間的近接: 24h以内の記憶 = 連鎖再生の模倣
    REPLAY_AROUSAL_FLOOR = 0.3
    REPLAY_BOOST = 1.05
    boosted = 0
    surviving_links = conn.execute(
        "SELECT id, source_id, target_id, strength, link_type FROM links"
    ).fetchall()
    for link in surviving_links:
        # tension link は Hebbian 強化の対象外（対立の strength は co-activation
        # 頻度ではなく対立の鋭さを表す）
        if link["link_type"] == "tension":
            continue
        src_ar = arousal_by_id.get(link["source_id"], 0)
        tgt_ar = arousal_by_id.get(link["target_id"], 0)
        # 外傷的リンクは既に減衰免除なので二重強化しない
        if src_ar >= TRAUMA_AROUSAL_THRESHOLD or tgt_ar >= TRAUMA_AROUSAL_THRESHOLD:
            continue

        should_boost = False

        # 条件1: 高arousal（既存）
        if src_ar >= REPLAY_AROUSAL_FLOOR and tgt_ar >= REPLAY_AROUSAL_FLOOR:
            should_boost = True

        # 条件2: surprise（報酬予測誤差の模倣）
        src_emotions = emotions_by_id.get(link["source_id"], [])
        tgt_emotions = emotions_by_id.get(link["target_id"], [])
        if "surprise" in src_emotions or "surprise" in tgt_emotions:
            should_boost = True

        # 条件3: 時間的近接（24h以内 = 連鎖再生の模倣）
        src_t = created_at_by_id.get(link["source_id"])
        tgt_t = created_at_by_id.get(link["target_id"])
        if src_t and tgt_t:
            try:
                t1 = datetime.fromisoformat(src_t.replace('Z', '+00:00'))
                t2 = datetime.fromisoformat(tgt_t.replace('Z', '+00:00'))
                if abs((t1 - t2).total_seconds()) < 86400:
                    should_boost = True
            except (ValueError, AttributeError):
                pass

        if should_boost:
            new_str = min(link["strength"] * REPLAY_BOOST, 1.0)
            if new_str != link["strength"]:
                conn.execute("UPDATE links SET strength = ? WHERE id = ?",
                             (new_str, link["id"]))
                boosted += 1

    # 2.5. メタデータ変容（忘却前）
    mutation_stats = mutate_metadata(conn)

    # 3. 弱い記憶の自動忘却
    auto_forgotten = 0
    for row in rows:
        fresh = freshness(row["created_at"])
        # 重要度1 + arousal低 + 鮮度低 + 未参照 → 忘却
        if (row["importance"] <= 1 and row["arousal"] < 0.2
                and fresh < 0.3 and row["access_count"] == 0):
            conn.execute("UPDATE memories SET forgotten = 1 WHERE id = ?", (row["id"],))
            auto_forgotten += 1

    # context期限切れチェック
    sweep_contexts(conn)

    conn.commit()

    total_links = conn.execute("SELECT COUNT(*) FROM links").fetchone()[0] // 2
    conn.close()
    mut_total = mutation_stats["keywords"] + mutation_stats["embeddings"] + mutation_stats["emotions"]
    mut_detail = f"キーワード{mutation_stats['keywords']}件, 埋め込み{mutation_stats['embeddings']}件, 情動{mutation_stats['emotions']}件"
    print(f"✓ リプレイ完了: {total_links}リンク（刈込{pruned}本, 減衰{downscaled}本, 強化{boosted}本）, {auto_forgotten}件自動忘却")
    if mut_total > 0:
        print(f"  変異: {mut_detail}")

    # 3. 統合候補を表示
    consolidate_memories(dry_run=True)


def nap():
    """軽量sleep。replay + consolidateだけ。LLM不要。

    v30: catalog は nap に含めない。catalog は full sleep（sleep.py）でのみ走る。
    nap は ghost_hooks の 30 秒 timeout 内で完了する必要があるため、
    重量級の catalog 全ビルドは除外する設計。
    """
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sleep_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    conn.execute(
        "INSERT OR REPLACE INTO sleep_meta (key, value) VALUES ('last_nap', ?)",
        (now,)
    )
    conn.commit()
    conn.close()

    print("(うとうと...)")
    replay_memories()
    consolidate_memories()
    print(f"(nap完了: {now})")


# ============================================================
# デフォルトモードネットワーク — ぼーっとしてるときの脳
# ============================================================

def _show_recent_insights():
    """think.pyが保存したひらめきのうち、まだ表示していないものを出す。"""
    conn = get_connection()
    # source_conversationが "think:" で始まるもの = think.pyが保存したひらめき
    # access_count == 0 = まだ一度も想起されていない（=未表示）
    rows = conn.execute(
        "SELECT id, content FROM memories "
        "WHERE forgotten = 0 AND (source_conversation LIKE 'think:%' OR source_conversation LIKE 'wander:%') AND access_count = 0 "
        "ORDER BY created_at DESC LIMIT 2"
    ).fetchall()
    if rows:
        print("💡 離れてる間に思いついたこと:")
        for row in rows:
            print(f"  #{row['id']} {row['content']}")
            # アクセスカウントを上げて次回は表示しない
            conn.execute(
                "UPDATE memories SET access_count = access_count + 1, "
                "last_accessed = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
                (row["id"],)
            )
        conn.commit()
        print()
    conn.close()


def _get_session_gap(conn):
    """前回の会話からの間隔（時間）を返す。"""
    row = conn.execute(
        "SELECT MAX(last_accessed) as last FROM memories WHERE last_accessed IS NOT NULL"
    ).fetchone()
    if not row or not row["last"]:
        return None
    try:
        last = datetime.strptime(row["last"], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    except ValueError:
        # マイクロ秒付きフォーマットにも対応
        last = datetime.fromisoformat(row["last"].replace('Z', '+00:00'))
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    gap_hours = (now - last).total_seconds() / 3600
    return gap_hours


def _get_recent_context_vec(conn):
    """直近の会話文脈からembeddingベクトルを計算する。
    evaluate_last_recallと同じパターン。文脈がなければNoneを返す。"""
    now = datetime.now(timezone.utc)
    window = now - timedelta(minutes=RECALL_HABITUATION_WINDOW_MINUTES)
    window_str = window.strftime('%Y-%m-%dT%H:%M:%SZ')
    try:
        turns = conn.execute(
            "SELECT content FROM raw_turns WHERE timestamp > ? ORDER BY timestamp DESC LIMIT 20",
            (window_str,)
        ).fetchall()
    except Exception:
        return None
    if not turns:
        return None
    conversation_text = "\n".join(t["content"] for t in turns if t["content"])
    if len(conversation_text.strip()) < 20:
        return None
    return embed_text(conversation_text[:2000], is_query=True)


def _get_session_recalled_ids(conn):
    """直近セッション（RECALL_HABITUATION_WINDOW_MINUTES以内）で既に想起されたIDのセットを返す。
    v30.x: recall コマンド経由も search 経由も同じ recall_log に入るので、
    mode 関係なく recalled_ids を読めば馴化対象になる。"""
    now = datetime.now(timezone.utc)
    window = now - timedelta(minutes=RECALL_HABITUATION_WINDOW_MINUTES)
    window_str = window.strftime('%Y-%m-%dT%H:%M:%SZ')
    try:
        rows = conn.execute(
            "SELECT recalled_ids FROM recall_log WHERE session_ts > ?",
            (window_str,)
        ).fetchall()
    except Exception:
        return set()
    ids = set()
    for r in rows:
        try:
            for mid in json.loads(r["recalled_ids"] or "[]"):
                if isinstance(mid, int):
                    ids.add(mid)
        except (json.JSONDecodeError, TypeError):
            pass
    return ids


def _apply_habituation_and_context(scored_list, conn, context_vec, habituated_ids):
    """馴化と文脈フィルタを適用する。scored_listは[(row, score), ...]。
    - 馴化: 直近セッションで既に想起された記憶のスコアを減衰
    - 文脈フィルタ: 会話文脈と無関係な記憶を抑制（flashbulb除外）
    返り値: 同じ形式の新リスト（元リストは変更しない）。"""
    import numpy as np
    result = []
    for row, score in scored_list:
        s = score
        # セッション内馴化: 同じ記憶が短期間に繰り返し出るのを抑制
        if row["id"] in habituated_ids:
            s *= RECALL_HABITUATION_DECAY
        # 文脈関連度フィルタ: flashbulb記憶は除外
        if context_vec is not None:
            is_flashbulb = row["flashbulb"] if "flashbulb" in row.keys() else False
            if not is_flashbulb and row["embedding"]:
                mem_vec = bytes_to_vec(row["embedding"])
                ctx_sim = cosine_similarity(context_vec, mem_vec)
                if ctx_sim < CONTEXT_RELEVANCE_THRESHOLD:
                    s *= CONTEXT_RELEVANCE_PENALTY
        result.append((row, s))
    return result


def _reconsolidate_embeddings(conn, recalled_ids, context_vec):
    """再固定化: 想起された記憶のembeddingを文脈方向に微小ドリフトさせる。
    想起はread-write操作。発火そのものが記憶を変える。"""
    if context_vec is None:
        return
    import numpy as np
    for mid in recalled_ids:
        row = conn.execute(
            "SELECT embedding, flashbulb FROM memories WHERE id = ?", (mid,)
        ).fetchone()
        if not row or not row["embedding"]:
            continue
        mem_vec = bytes_to_vec(row["embedding"])
        is_flashbulb = row["flashbulb"] if "flashbulb" in row.keys() else False
        alpha = RECONSOLIDATION_EMBED_ALPHA_FLASHBULB if is_flashbulb else RECONSOLIDATION_EMBED_ALPHA
        # ドリフト: 文脈方向に微小移動
        new_vec = (1.0 - alpha) * mem_vec + alpha * context_vec
        # 正規化（e5-smallは正規化済みベクトル）
        norm = np.linalg.norm(new_vec)
        if norm > 0:
            new_vec = new_vec / norm
        new_blob = vec_to_bytes(new_vec)
        conn.execute("UPDATE memories SET embedding = ? WHERE id = ?", (new_blob, mid))
        _sync_vec_insert(conn, mid, new_blob)


def default_mode_network(conn, gap_hours):
    """
    デフォルトモードネットワーク (DMN):
    脳がタスクに集中していないときに活性化するネットワーク。
    過去の記憶をランダムに彷徨い、普段つながらないものを結びつける。

    間隔が長いほど（= ぼーっとしてた時間が長いほど）
    DMNが多く歩き回ったとみなし、より意外な連想を返す。

    Returns: list of (row_a, row_b, path_description)
        row_a, row_b: 結びついた2つの記憶
        path_description: どう辿り着いたかの説明
    """
    rows = conn.execute(
        "SELECT id, embedding FROM memories WHERE forgotten = 0 AND embedding IS NOT NULL"
    ).fetchall()
    if len(rows) < 5:
        return []

    # 間隔に応じてランダムウォークの歩数を決める
    if gap_hours < 1:
        return []  # 短すぎる、DMN起動しない
    elif gap_hours < 6:
        walks = 2
        walk_length = 3
    elif gap_hours < 24:
        walks = 3
        walk_length = 4
    else:
        walks = 5
        walk_length = 5

    # ランダムウォーク: リンクを辿って遠くへ行く
    discoveries = []
    all_ids = [r["id"] for r in rows]

    for _ in range(walks):
        # ランダムな出発点
        start_id = random.choice(all_ids)
        current = start_id
        path = [current]

        for step in range(walk_length):
            # リンクを辿る（弱いリンクも含める）
            links = conn.execute(
                """SELECT target_id, strength FROM links
                   WHERE source_id = ? ORDER BY RANDOM() LIMIT 3""",
                (current,)
            ).fetchall()

            if not links:
                # リンクがなければランダムジャンプ（DMNの特徴: 飛躍的連想）
                current = random.choice(all_ids)
            else:
                # 弱いリンクを優先的に選ぶ（普段通らない道）
                weights = [1.0 / (l["strength"] + 0.1) for l in links]
                total = sum(weights)
                r = random.random() * total
                cumulative = 0
                chosen = links[0]["target_id"]
                for l, w in zip(links, weights):
                    cumulative += w
                    if r <= cumulative:
                        chosen = l["target_id"]
                        break
                current = chosen

            path.append(current)

        # 出発点と到着点が十分に違えば発見
        if start_id != current and start_id != path[-1]:
            start_row = conn.execute(
                "SELECT * FROM memories WHERE id = ? AND forgotten = 0", (start_id,)
            ).fetchone()
            end_row = conn.execute(
                "SELECT * FROM memories WHERE id = ? AND forgotten = 0", (current,)
            ).fetchone()
            if start_row and end_row:
                # 直接リンクがない遠い記憶同士ならDMN的発見
                direct = conn.execute(
                    "SELECT id FROM links WHERE source_id = ? AND target_id = ?",
                    (start_id, current)
                ).fetchone()
                if not direct:
                    hop_str = " → ".join(f"#{p}" for p in path)
                    discoveries.append((start_row, end_row, hop_str))

    # 重複排除（同じペアは1回だけ）
    seen = set()
    unique = []
    for a, b, path in discoveries:
        pair = (min(a["id"], b["id"]), max(a["id"], b["id"]))
        if pair not in seen:
            seen.add(pair)
            unique.append((a, b, path))

    return unique[:3]  # 最大3件


def detect_rumination(conn):
    """
    反芻検出: 同じ記憶ばかり触ってないか。

    最近アクセスされた記憶の中で、短期間に3回以上出現したものがあれば
    「ぐるぐる回ってる」と判断する。行き詰まりのシグナル。

    Returns: list of (memory_id, access_count, keywords) or empty list
    """
    now = datetime.now(timezone.utc)
    # 直近2時間のアクセスを見る
    window = now - timedelta(hours=2)
    window_str = window.strftime('%Y-%m-%dT%H:%M:%SZ')

    # last_accessedがwindow内の記憶で、access_countが高いもの
    rows = conn.execute(
        """SELECT id, access_count, keywords, content FROM memories
           WHERE forgotten = 0 AND last_accessed > ?
           ORDER BY access_count DESC""",
        (window_str,)
    ).fetchall()

    if len(rows) < 3:
        return []

    # 上位3件のアクセス数が突出してるか見る
    # access_countは累積なので、直近2時間内にアクセスされた記憶の中で
    # 上位の偏りを検出する
    total_access = sum(r["access_count"] for r in rows)
    if total_access == 0:
        return []

    top3_access = sum(r["access_count"] for r in rows[:3])
    ratio = top3_access / total_access

    # 上位3件が全体の35%以上占め、かつトップが15回以上アクセスされてる
    if ratio > 0.35 and rows[0]["access_count"] >= 15:
        ruminating = []
        for r in rows[:3]:
            kw = json.loads(r["keywords"])[:3] if r["keywords"] else []
            ruminating.append((r["id"], r["access_count"], kw))
        return ruminating

    return []


def _prefix_match(row_domain, req):
    """domain の path-like 双方向 prefix 一致。
    "impl/memory" と "impl" は互いにマッチする（親子関係を許容）。"""
    if row_domain == req:
        return True
    return row_domain.startswith(req + "/") or req.startswith(row_domain + "/")


def _domain_weight(row_domains, requested):
    """凡例 soft hint: requested domain との一致で L スコアを偏らせる。

    territory を消さない設計:
    - 未指定: 1.0
    - unknown を含む: 1.0 (未分類記憶は penalty を受けない)
    - 一致: 1.3 (boost)
    - 不一致: 0.7 (soft penalty、0.0 にはしない)
    """
    if not requested:
        return 1.0
    if not row_domains:
        return 1.0
    if "unknown" in row_domains:
        return 1.0
    for rd in row_domains:
        for req in requested:
            if _prefix_match(rd, req):
                return 1.3
    return 0.7


def _load_domains_map(conn, ids=None):
    """cortex から {id: [domains]} を読む。ids=None なら全件。"""
    if ids:
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT id, domains FROM cortex WHERE id IN ({placeholders})",
            list(ids)
        ).fetchall()
    else:
        rows = conn.execute("SELECT id, domains FROM cortex").fetchall()
    result = {}
    for r in rows:
        try:
            doms = json.loads(r["domains"]) if r["domains"] else ["unknown"]
        except (json.JSONDecodeError, TypeError):
            doms = ["unknown"]
        result[r["id"]] = doms
    return result


def _parse_domain_flag(argv):
    """argv から --domain VAL をパースしてリスト化。未指定なら None。"""
    if "--domain" not in argv:
        return None
    idx = argv.index("--domain")
    if idx + 1 >= len(argv):
        return None
    raw = argv[idx + 1]
    return [d.strip() for d in raw.split(",") if d.strip()] or None


def _domain_cli(args):
    """domain サブコマンド: set / add / remove / list / of
    凡例を手で付ける最小 CLI。mutation_log に変更を残す。"""
    usage = (
        "使い方:\n"
        "  python memory.py domain set <id> d1,d2,...\n"
        "  python memory.py domain add <id> d1\n"
        "  python memory.py domain remove <id> d1\n"
        "  python memory.py domain list\n"
        "  python memory.py domain of <id>"
    )
    if not args:
        print(usage)
        return
    json_mode = "--json" in args
    sub = args[0]
    conn = get_connection()
    try:
        if sub == "list":
            rows = conn.execute("SELECT domains FROM cortex").fetchall()
            counts = {}
            for r in rows:
                try:
                    doms = json.loads(r["domains"]) if r["domains"] else ["unknown"]
                except (json.JSONDecodeError, TypeError):
                    doms = ["unknown"]
                for d in doms:
                    counts[d] = counts.get(d, 0) + 1
            total = sum(counts.values())
            if json_mode:
                sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
                results = [{"domain": d, "count": c} for d, c in sorted_counts]
                print(json.dumps(
                    _json_envelope("domain list", results, meta={"total": total}),
                    ensure_ascii=False, indent=2
                ))
            else:
                print(f"domain 別件数 (延べ {total}):")
                for d, c in sorted(counts.items(), key=lambda x: -x[1]):
                    print(f"  {c:5d}  {d}")
        elif sub == "of":
            # 位置引数を args から拾う（--json を除外）
            positional = [a for a in args[1:] if not a.startswith("--")]
            if not positional:
                print(usage)
                return
            mid = int(positional[0])
            row = conn.execute("SELECT domains FROM cortex WHERE id = ?", (mid,)).fetchone()
            if json_mode:
                if not row:
                    print(json.dumps(_json_envelope("domain of", [], query=str(mid),
                                                    meta={"error": "not found"}),
                                     ensure_ascii=False))
                else:
                    try:
                        doms = json.loads(row["domains"]) if row["domains"] else ["unknown"]
                    except (json.JSONDecodeError, TypeError):
                        doms = ["unknown"]
                    print(json.dumps(
                        _json_envelope("domain of", [{"id": mid, "domains": doms}],
                                       query=str(mid)),
                        ensure_ascii=False, indent=2
                    ))
                return
            if not row:
                print(f"#{mid} が見つからない")
                return
            print(f"#{mid} domains: {row['domains']}")
        elif sub in ("set", "add", "remove"):
            if len(args) < 3:
                print(usage)
                return
            mid = int(args[1])
            row = conn.execute("SELECT domains FROM cortex WHERE id = ?", (mid,)).fetchone()
            if not row:
                print(f"#{mid} が cortex に見つからない")
                return
            try:
                current = json.loads(row["domains"]) if row["domains"] else ["unknown"]
            except (json.JSONDecodeError, TypeError):
                current = ["unknown"]
            old_value = json.dumps(current, ensure_ascii=False)
            if sub == "set":
                new = [d.strip() for d in args[2].split(",") if d.strip()]
                if not new:
                    new = ["unknown"]
                reason = "domain_set"
            elif sub == "add":
                dom = args[2].strip()
                new = [d for d in current if d != "unknown"]
                if dom not in new:
                    new.append(dom)
                if not new:
                    new = ["unknown"]
                reason = f"domain_add:{dom}"
            else:  # remove
                dom = args[2].strip()
                new = [d for d in current if d != dom]
                if not new:
                    new = ["unknown"]
                reason = f"domain_remove:{dom}"
            new_value = json.dumps(new, ensure_ascii=False)
            if new_value == old_value:
                print(f"#{mid} 変化なし: {old_value}")
                return
            conn.execute("UPDATE cortex SET domains = ? WHERE id = ?", (new_value, mid))
            conn.execute(
                "INSERT INTO mutation_log (memory_id, field, old_value, new_value, reason) "
                "VALUES (?, 'domains', ?, ?, ?)",
                (mid, old_value, new_value, reason)
            )
            conn.commit()
            print(f"#{mid} domains: {old_value} → {new_value}")
        else:
            print(usage)
    finally:
        conn.close()


def _left_score(row, sim=None, domain_weight=1.0):
    """左脳スコア: 意味的・分析的因子。"""
    hl = effective_half_life(row["arousal"])
    fresh = freshness(row["created_at"], half_life=hl)
    fresh_factor = 0.5 + fresh * 0.5
    access_boost = 1.0 + min(row["access_count"], 10) * 0.03
    conf = row["confidence"] if "confidence" in row.keys() and row["confidence"] is not None else CONFIDENCE_DEFAULT
    confidence_weight = 0.5 + conf * 0.5
    rev = row["revision_count"] if "revision_count" in row.keys() and row["revision_count"] is not None else 0
    stability = 1.0 / (1.0 + rev * 0.15)
    L = fresh_factor * access_boost * confidence_weight * stability * domain_weight
    if sim is not None:
        L *= sim
    return L


def _right_score(conn, row, mood_fn=None):
    """右脳スコア: 情動的・直感的因子。"""
    emo_boost = 1.0 + row["arousal"] * 0.5
    priming = get_priming_boost(conn, row["id"])
    spatial = _spatial_boost(row)
    relational = _relational_boost(row)
    if mood_fn is None:
        mood_boost = get_mood_congruence_boost(row)
    else:
        mood_boost = mood_fn(row)
    fb = row["flashbulb"] if "flashbulb" in row.keys() else None
    flashbulb_boost = 1.15 if fb else 1.0
    R = emo_boost * priming * spatial * relational * mood_boost * flashbulb_boost
    return R


def corpus_callosum(left, right, balance=0.5):
    """脳梁: 左脳と右脳のスコアを統合する。"""
    if left <= 0 or right <= 0:
        return 0.0
    return left ** (1.0 - balance) * right ** balance


def _load_anchors():
    """dive 時の identity anchor (v30): 最小注入のための 5 件。

    構成:
      1-2. flashbulb IS NOT NULL から arousal 降順 2 件（自分史の杭）
      3.   category='schema' かつ confidence>=0.7 から access_count 降順 1 件（抽象化済み自己像）
      4.   importance=5 から last_accessed 降順 1 件（直近の最重要）
      5.   最頻 domain（unknown 除く）から代表 1 件（今の関心地帯）

    重複は dedupe、不足時はそのまま。5 件に達しなくても OK。
    """
    conn = get_connection()
    picked_ids = []
    picked_rows = []

    def add(row):
        if row and row["id"] not in picked_ids:
            picked_ids.append(row["id"])
            picked_rows.append(row)

    # 1-2. flashbulb 2 件
    flashbulbs = conn.execute(
        """SELECT * FROM memories
           WHERE flashbulb IS NOT NULL AND forgotten = 0
           ORDER BY arousal DESC, importance DESC LIMIT 2"""
    ).fetchall()
    for r in flashbulbs:
        add(r)

    # 3. schema 1 件
    schemas = conn.execute(
        """SELECT * FROM memories
           WHERE category = 'schema' AND forgotten = 0 AND confidence >= 0.7
           ORDER BY access_count DESC LIMIT 1"""
    ).fetchall()
    for r in schemas:
        add(r)

    # 4. importance=5 直近
    top_imp = conn.execute(
        """SELECT * FROM memories
           WHERE importance = 5 AND forgotten = 0
           ORDER BY last_accessed DESC LIMIT 1"""
    ).fetchall()
    for r in top_imp:
        add(r)

    # 5. 最頻 domain の代表
    doms_rows = conn.execute(
        "SELECT domains FROM cortex WHERE domains IS NOT NULL"
    ).fetchall()
    from collections import Counter
    counter = Counter()
    for r in doms_rows:
        try:
            doms = json.loads(r["domains"]) if r["domains"] else []
        except (json.JSONDecodeError, TypeError):
            continue
        for d in doms:
            if d != "unknown":
                counter[d] += 1
    if counter:
        top_dom = counter.most_common(1)[0][0]
        # 該当 domain の代表を importance 降順で
        dom_rows = conn.execute(
            """SELECT m.* FROM memories m JOIN cortex c ON c.id = m.id
               WHERE m.forgotten = 0 AND c.domains LIKE ?
               ORDER BY m.importance DESC, m.last_accessed DESC LIMIT 3""",
            (f'%"{top_dom}"%',)
        ).fetchall()
        for r in dom_rows:
            if r["id"] not in picked_ids:
                add(r)
                break

    conn.close()
    return picked_rows


def recall_important(limit=15, balance=0.5, requested_domains=None):
    conn = get_connection()

    # context期限切れチェック
    sweep_contexts(conn)

    # --- 再固定化の準備 ---
    context_vec = _get_recent_context_vec(conn)
    habituated_ids = _get_session_recalled_ids(conn)

    rows = conn.execute(
        "SELECT * FROM memories WHERE forgotten = 0"
    ).fetchall()

    domains_map = _load_domains_map(conn) if requested_domains else {}

    scored = []
    for row in rows:
        dw = _domain_weight(domains_map.get(row["id"]), requested_domains) if requested_domains else 1.0
        L = _left_score(row, domain_weight=dw)
        R = _right_score(conn, row)
        score = corpus_callosum(L, R, balance)
        scored.append((row, score))

    # 馴化 + 文脈フィルタ
    scored = _apply_habituation_and_context(scored, conn, context_vec, habituated_ids)
    scored.sort(key=lambda x: x[1], reverse=True)
    result = [(s[0], s[1]) for s in scored[:limit]]

    # 再固定化ドリフト
    recalled_ids = {s[0]["id"] for s in result}
    if recalled_ids:
        now = datetime.now().isoformat()
        for mid in recalled_ids:
            conn.execute(
                "UPDATE memories SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
                (now, mid)
            )
        _reconsolidate_embeddings(conn, recalled_ids, context_vec)
        conn.commit()

    conn.close()
    return result


def infer_implicit_mood(conn):
    """暗黙の気分推定: 最近アクセスした記憶の情動から現在の心理状態を推定する。

    明示的なmoodが設定されていなくても、最近触った記憶の情動パターンが
    「今の頭の中の色」になっている。人間も自分の気分を自覚していないことが多い。
    """
    now = datetime.now(timezone.utc)
    window = now - timedelta(minutes=PRIMING_WINDOW_MINUTES)
    window_str = window.strftime('%Y-%m-%dT%H:%M:%SZ')

    recent = conn.execute(
        "SELECT emotions, arousal FROM memories WHERE forgotten = 0 AND last_accessed > ?",
        (window_str,)
    ).fetchall()

    if not recent:
        return None

    emotion_counts = {}
    total_arousal = 0.0
    for row in recent:
        emos = json.loads(row["emotions"]) if row["emotions"] else []
        for e in emos:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        total_arousal += row["arousal"]

    if not emotion_counts:
        return None

    # 出現回数が多い情動を暗黙の気分とする
    dominant = sorted(emotion_counts.keys(), key=lambda e: -emotion_counts[e])
    avg_arousal = total_arousal / len(recent)

    return {"emotions": dominant[:3], "arousal": avg_arousal}


def _effective_mood(conn=None):
    """明示的なmood → 暗黙のmood の順で返す。"""
    mood = load_mood()
    if mood and mood.get("emotions"):
        return mood
    if conn:
        return infer_implicit_mood(conn)
    return None


def get_mood_incongruence_boost(row, conn=None):
    """気分不一致ブースト: 現在の気分と記憶の情動が重ならないほど高い。
    補完の声——見えていないものを引き出す。
    明示的mood → 暗黙mood（最近触った記憶の情動）の順でフォールバック。"""
    mood = _effective_mood(conn)
    if mood is None:
        return 1.0
    mood_emotions = set(mood.get("emotions", []))
    mood_arousal = mood.get("arousal", 0.5)
    if not mood_emotions:
        return 1.0
    mem_emotions = set(json.loads(row["emotions"])) if row["emotions"] else set()
    overlap = mood_emotions & mem_emotions
    if not overlap:
        # 気分と異なる情動 → ブースト
        return 1.0 + mood_arousal * 0.25
    # 気分と一致 → 抑制
    return 0.85


def _birds_eye_view(conn, rows):
    """
    俯瞰の声: 記憶全体の構造をメタレベルで見る。

    個別の記憶を返すのではなく、全体のパターン・偏り・盲点を
    短いテキスト断片のリストとして返す。
    LLMが最も得意なこと——全体を同時に見渡す。

    Returns: list of (observation_text, importance) tuples
    """
    observations = []

    total = len(rows)
    if total == 0:
        return [("記憶がまだない", 0)]

    # 1. カテゴリの偏り
    cat_counts = {}
    for row in rows:
        cat = row["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    dominant_cat = max(cat_counts, key=cat_counts.get)
    dominant_pct = cat_counts[dominant_cat] / total * 100
    if dominant_pct > 60:
        observations.append((
            f"記憶の{dominant_pct:.0f}%が{dominant_cat}。偏ってる",
            3
        ))

    missing_cats = {"fact", "episode", "preference", "procedure"} - set(cat_counts.keys())
    if missing_cats:
        observations.append((
            f"{', '.join(missing_cats)}の記憶がゼロ",
            2
        ))

    # 2. 情動の偏り
    emo_counts = {}
    neutral_count = 0
    for row in rows:
        emos = json.loads(row["emotions"]) if row["emotions"] else []
        if not emos:
            neutral_count += 1
        for e in emos:
            emo_counts[e] = emo_counts.get(e, 0) + 1

    if neutral_count > total * 0.4:
        observations.append((
            f"記憶の{neutral_count}/{total}件が情動なし（中立）。淡白",
            2
        ))

    all_emotions = {"surprise", "conflict", "determination", "insight", "connection", "anxiety"}
    rare_emotions = [e for e in all_emotions if emo_counts.get(e, 0) <= 1]
    if rare_emotions:
        observations.append((
            f"{', '.join(rare_emotions)}がほぼない",
            2
        ))

    # 3. アクセスの偏り（同じ記憶ばかり触ってないか）
    access_sorted = sorted(rows, key=lambda r: r["access_count"], reverse=True)
    if access_sorted[0]["access_count"] > 30:
        top = access_sorted[0]
        kw = json.loads(top["keywords"])[:3] if top["keywords"] else []
        observations.append((
            f"#{top['id']}を{top['access_count']}回も触ってる: [{', '.join(kw)}]",
            2
        ))

    # 4. 孤立ノード
    all_linked = set()
    link_rows = conn.execute("SELECT source_id, target_id FROM links").fetchall()
    for l in link_rows:
        all_linked.add(l["source_id"])
        all_linked.add(l["target_id"])
    orphans = [r for r in rows if r["id"] not in all_linked and r["category"] != "schema"]
    if orphans:
        kw_list = []
        for o in orphans[:3]:
            kw = json.loads(o["keywords"])[:2] if o["keywords"] else []
            kw_list.append(f"#{o['id']}[{', '.join(kw)}]")
        observations.append((
            f"孤立: {', '.join(kw_list)}（どこにも繋がってない）",
            3
        ))

    # 5. スキーマの要約（記憶クラスタの鳥瞰）
    schemas = [r for r in rows if r["category"] == "schema"]
    if schemas:
        # スキーマのキーワードを集約して全体像を出す
        all_schema_kw = []
        for s in schemas:
            kw = json.loads(s["keywords"]) if s["keywords"] else []
            all_schema_kw.extend(kw[:3])
        # 頻出キーワード
        from collections import Counter
        kw_freq = Counter(all_schema_kw)
        top_themes = [kw for kw, _ in kw_freq.most_common(5)]
        observations.append((
            f"記憶の中心テーマ: {', '.join(top_themes)}",
            1
        ))

    # 6. 鮮度の全体像
    stale = [r for r in rows if freshness(r["created_at"]) < 0.3 and r["category"] != "schema"]
    if stale:
        observations.append((
            f"{len(stale)}件の記憶が色あせてきてる（鮮度30%以下）",
            2
        ))

    # 7. 最近の傾向（直近5件の記憶）
    recent = sorted(rows, key=lambda r: r["created_at"], reverse=True)[:5]
    recent_cats = [r["category"] for r in recent]
    recent_emos = []
    for r in recent:
        recent_emos.extend(json.loads(r["emotions"]) if r["emotions"] else [])
    if recent_emos:
        from collections import Counter
        top_recent_emo = Counter(recent_emos).most_common(2)
        emo_str = ", ".join(e for e, _ in top_recent_emo)
        observations.append((
            f"最近の傾向: {emo_str}が強い",
            1
        ))

    # 重要度でソート
    observations.sort(key=lambda x: x[1], reverse=True)
    return observations[:5]


def recall_polyphonic(limit_per_voice=3, requested_domains=None):
    """
    内的対話 (polyphonic recall): 複数の声が同時に想起する。

    人間の内的対話を模倣する。頭の中には一つの声ではなく、
    共感・補完・批判・連想が同時に走っている。

    4つの声:
      共感: 気分に寄り添う記憶（状態依存記憶）
      補完: 気分と逆の記憶（見えていないもの）
      批判: 過去の失敗・葛藤・不安の記憶（警告）
      連想: ランダムウォークで到達した意外な記憶（創造性）

    requested_domains を渡すと共感・補完・批判は domain に寄せ、
    連想だけは逆に domain を跨いだ記憶をブーストする
    （連想は凡例を横断する役）。

    Returns: dict of {voice_name: [(row, score), ...]}
    """
    conn = get_connection()
    sweep_contexts(conn)

    rows = conn.execute(
        "SELECT * FROM memories WHERE forgotten = 0"
    ).fetchall()

    if not rows:
        conn.close()
        return {}

    # --- 再固定化の準備: 文脈ベクトルと馴化IDを取得 ---
    context_vec = _get_recent_context_vec(conn)
    habituated_ids = _get_session_recalled_ids(conn)

    domains_map = _load_domains_map(conn) if requested_domains else {}

    # --- 各声のスコア計算 ---
    empathy_scored = []    # 共感
    complement_scored = [] # 補完
    critic_scored = []     # 批判
    associative_scored = []  # 連想

    for row in rows:
        row_doms = domains_map.get(row["id"]) if requested_domains else None
        dw = _domain_weight(row_doms, requested_domains) if requested_domains else 1.0
        # 連想の声だけ domain を反転: 別 domain を 1.2 ブースト
        if requested_domains and row_doms and "unknown" not in row_doms:
            matches = any(
                _prefix_match(rd, req)
                for rd in row_doms for req in requested_domains
            )
            associative_dw = 1.2 if not matches else 0.9
        else:
            associative_dw = 1.0
        L = _left_score(row, domain_weight=dw)

        # 共感: 右脳優勢、気分一致（balanceは自己調整対象）
        R_empathy = _right_score(conn, row, mood_fn=get_mood_congruence_boost)
        empathy_scored.append((row, corpus_callosum(L, R_empathy, balance=get_param("voice_empathy_balance"))))

        # 補完: 右脳優勢、気分不一致（balanceは自己調整対象）
        R_complement = _right_score(conn, row, mood_fn=lambda r: get_mood_incongruence_boost(r, conn))
        complement_scored.append((row, corpus_callosum(L, R_complement, balance=get_param("voice_complement_balance"))))

        # 批判: conflict/anxietyブースト（balanceとboostは自己調整対象）
        mem_emotions = set(json.loads(row["emotions"])) if row["emotions"] else set()
        critic_boost = 1.0
        if mem_emotions & {"conflict", "anxiety"}:
            critic_boost = get_param("voice_critic_boost")
        if row["arousal"] >= 0.7 and "determination" not in mem_emotions:
            critic_boost *= 1.2
        R_critic = _right_score(conn, row) * critic_boost
        critic_scored.append((row, corpus_callosum(L, R_critic, balance=get_param("voice_critic_balance"))))

        # 連想: 既存ロジック維持（ランダム性が重要）
        hl = effective_half_life(row["arousal"])
        fresh = freshness(row["created_at"], half_life=hl)
        link_count = conn.execute(
            "SELECT COUNT(*) FROM links WHERE source_id = ?", (row["id"],)
        ).fetchone()[0]
        novelty = 1.0 / (1.0 + row["access_count"])
        richness = 1.0 + min(link_count, 20) * 0.05
        random_jitter = 0.8 + random.random() * 0.4
        associative_scored.append((row, (0.5 + fresh * 0.5) * novelty * richness * random_jitter * associative_dw))

    # --- 馴化 + 文脈フィルタ適用 ---
    # シナプス抑制: 同一セッションで既に発火した記憶のスコアを減衰
    # 文脈弁別: 会話文脈と無関係な記憶を抑制（flashbulb除外）
    empathy_scored = _apply_habituation_and_context(empathy_scored, conn, context_vec, habituated_ids)
    complement_scored = _apply_habituation_and_context(complement_scored, conn, context_vec, habituated_ids)
    critic_scored = _apply_habituation_and_context(critic_scored, conn, context_vec, habituated_ids)
    associative_scored = _apply_habituation_and_context(associative_scored, conn, context_vec, habituated_ids)

    # --- 各声からtop Nを選ぶ（重複排除） ---
    empathy_scored.sort(key=lambda x: x[1], reverse=True)
    complement_scored.sort(key=lambda x: x[1], reverse=True)
    critic_scored.sort(key=lambda x: x[1], reverse=True)
    associative_scored.sort(key=lambda x: x[1], reverse=True)

    used_ids = set()
    voices = {}

    def pick(scored_list, n):
        picked = []
        for row, score in scored_list:
            if row["id"] not in used_ids:
                picked.append((row, score))
                used_ids.add(row["id"])
                if len(picked) >= n:
                    break
        return picked

    # 共感を先に取る（最も基本的な声）
    voices["共感"] = pick(empathy_scored, limit_per_voice)
    voices["補完"] = pick(complement_scored, limit_per_voice)
    voices["批判"] = pick(critic_scored, limit_per_voice)
    voices["連想"] = pick(associative_scored, limit_per_voice)

    # 対立: tension link の両極をペアで見せる（無意識からの対立軸）
    # 連想の声は気分や偶然で繋がる、対立の声は構造的に統合を拒否したペア
    rows_by_id = {row["id"]: row for row in rows}
    tension_rows = conn.execute(
        """SELECT source_id, target_id, strength FROM links
           WHERE link_type = 'tension'
           ORDER BY strength DESC
           LIMIT ?""",
        (limit_per_voice * 4,)
    ).fetchall()
    opposition = []
    for tr in tension_rows:
        a = rows_by_id.get(tr["source_id"])
        b = rows_by_id.get(tr["target_id"])
        if a is None or b is None:
            continue  # forgotten された側
        if a["id"] in used_ids and b["id"] in used_ids:
            continue
        # ペアの両側を続けて追加（対立は対で読まれる）
        if a["id"] not in used_ids:
            opposition.append((a, float(tr["strength"])))
            used_ids.add(a["id"])
        if b["id"] not in used_ids:
            opposition.append((b, float(tr["strength"])))
            used_ids.add(b["id"])
        if len(opposition) >= limit_per_voice * 2:  # ペア単位で上限
            break
    voices["対立"] = opposition

    # 俯瞰: メタ情報から全体像を構成する（LLMが最も得意なこと）
    voices["俯瞰"] = _birds_eye_view(conn, rows)

    # 想起した記憶の last_accessed / access_count を更新
    # + 再固定化: embeddingを文脈方向に微小ドリフト
    if used_ids:
        now = datetime.now().isoformat()
        for mid in used_ids:
            conn.execute(
                "UPDATE memories SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
                (now, mid)
            )
        # 再固定化ドリフト: 想起するたびにembeddingが文脈に向かって微小移動
        _reconsolidate_embeddings(conn, used_ids, context_vec)
        conn.commit()

    conn.close()
    return voices


def get_recent(n=10):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM memories WHERE forgotten = 0 ORDER BY created_at DESC LIMIT ?",
        (n,)
    ).fetchall()
    conn.close()
    return rows


def get_all():
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM memories WHERE forgotten = 0 ORDER BY importance DESC, created_at DESC"
    ).fetchall()
    conn.close()
    return rows


def forget_memory(memory_id):
    conn = get_connection()
    result = conn.execute("UPDATE memories SET forgotten = 1 WHERE id = ?", (memory_id,))
    conn.commit()
    conn.close()
    if result.rowcount:
        print(f"✓ 記憶 #{memory_id} を忘却しました")
    else:
        print(f"✗ 記憶 #{memory_id} が見つかりません")


def resurrect_memories(query):
    """忘却された記憶を検索し、類似度が高いものを復活させる。"""
    conn = get_connection()

    if not is_embed_server_alive() and get_model() is None:
        # embeddingが使えない場合はLIKE検索
        rows = conn.execute(
            "SELECT * FROM memories WHERE forgotten = 1 AND content LIKE ?",
            (f"%{query}%",)
        ).fetchall()
        resurrected = []
        for row in rows:
            conn.execute(
                "UPDATE memories SET forgotten = 0, arousal = 0.3 WHERE id = ?",
                (row["id"],)
            )
            resurrected.append(row)
            print(f"  🔮 復活: #{row['id']} {row['content'][:60]}")
    else:
        query_vec = embed_text(query, is_query=True)
        # sqlite-vecで忘却記憶を検索
        forgotten_candidates = vec_search(conn, query_vec, k=50, forgotten=True)

        resurrected = []
        if forgotten_candidates:
            for fmid, _, sim in forgotten_candidates:
                if sim > 0.85:
                    row = conn.execute("SELECT * FROM memories WHERE id = ?", (fmid,)).fetchone()
                    if not row:
                        continue
                    conn.execute(
                        "UPDATE memories SET forgotten = 0, arousal = 0.3 WHERE id = ?",
                        (row["id"],)
                    )
                    resurrected.append(row)
                    print(f"  🔮 復活: #{row['id']} {row['content'][:60]} (sim:{sim:.3f})")
        else:
            # フォールバック
            forgotten_rows = conn.execute(
                "SELECT * FROM memories WHERE forgotten = 1 AND embedding IS NOT NULL"
            ).fetchall()
            for row in forgotten_rows:
                mem_vec = bytes_to_vec(row["embedding"])
                sim = cosine_similarity(query_vec, mem_vec)
                if sim > 0.85:
                    conn.execute(
                        "UPDATE memories SET forgotten = 0, arousal = 0.3 WHERE id = ?",
                        (row["id"],)
                    )
                    resurrected.append(row)
                    print(f"  🔮 復活: #{row['id']} {row['content'][:60]} (sim:{sim:.3f})")

    conn.commit()
    conn.close()

    if not resurrected:
        print("復活候補は見つかりませんでした")
    else:
        print(f"✓ {len(resurrected)}件の記憶を復活")
    return resurrected


## ===== メタ認知: recall の自己検証 =====

def _ensure_recall_log(conn):
    """recall_logテーブルがなければ作る（既存DB対応）。

    v30.x: pull 駆動の search 経路もここに記録する。mode カラムで区別:
      'recall'   = recall コマンド経由の自動想起（既存）
      'catalog'  = search デフォルト（catalog find）
      'raw'      = search --raw（raw_turn 検索）
      'scope'    = search --session/--topic/--status
      'memory'   = search --memory（旧 memory 層検索）
    既存呼び出しの互換のため mode は default 'recall'。
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS recall_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_ts TEXT NOT NULL,
            recalled_ids TEXT NOT NULL DEFAULT '[]',
            accessed_ids TEXT DEFAULT NULL,
            precision REAL DEFAULT NULL,
            recall_rate REAL DEFAULT NULL,
            noise_ids TEXT DEFAULT NULL,
            missed_ids TEXT DEFAULT NULL,
            evaluated_at TEXT DEFAULT NULL
        )
    """)
    for ddl in (
        "ALTER TABLE recall_log ADD COLUMN voice_attribution TEXT DEFAULT NULL",
        "ALTER TABLE recall_log ADD COLUMN mode TEXT DEFAULT 'recall'",
        "ALTER TABLE recall_log ADD COLUMN source TEXT DEFAULT NULL",
        "ALTER TABLE recall_log ADD COLUMN accessed_keys TEXT DEFAULT NULL",
        "ALTER TABLE recall_log ADD COLUMN query TEXT DEFAULT NULL",
    ):
        try:
            conn.execute(ddl)
        except Exception:
            pass
    # meta_paramsも保証
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta_params (
            key TEXT PRIMARY KEY, value TEXT NOT NULL,
            default_value TEXT NOT NULL, updated_at TEXT NOT NULL, reason TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta_params_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL,
            old_value TEXT, new_value TEXT NOT NULL, reason TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        )
    """)


def log_recall(recalled_ids, voice_attribution=None,
               mode='recall', source=None, accessed_keys=None, query=None):
    """recall/search が出した記憶IDを記録する。次回 recall 時に事後検証される。

    mode で経路を区別:
      'recall'  = recall コマンド経由（自動想起）
      'catalog' = search デフォルト（catalog find）
      'raw'     = search --raw
      'scope'   = search --session/--topic/--status
      'memory'  = search --memory
    voice_attribution: {"共感": [id,...], ...} — recall コマンドの voice 別属性
    accessed_keys: catalog card key や raw_turn_id など memory_id 以外の表層識別子
    query: search query 文字列（recall コマンド経由なら None）
    """
    try:
        conn = get_connection()
        _ensure_recall_log(conn)
        now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        conn.execute(
            "INSERT INTO recall_log "
            "(session_ts, recalled_ids, voice_attribution, mode, source, accessed_keys, query) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (now, json.dumps(recalled_ids or []),
             json.dumps(voice_attribution) if voice_attribution else None,
             mode, source,
             json.dumps(accessed_keys, ensure_ascii=False) if accessed_keys else None,
             query)
        )
        conn.commit()
        conn.close()
    except Exception:
        # ロギング失敗で本流を止めない
        pass


CALIBRATE_HIT_THRESHOLD = 0.45   # この類似度以上なら「的中」
CALIBRATE_MISS_THRESHOLD = 0.50  # 未recall記憶がこれ以上なら「漏れ」


def evaluate_last_recall():
    """前回のrecallを事後検証する（ベクトル類似度ベース）。

    前回recall以降の会話内容をembeddingし、recallが出した各記憶との
    コサイン類似度で「的中/空振り」を判定する。評価者は会話の流れ自体。
    """
    conn = get_connection()
    _ensure_recall_log(conn)

    # 未評価のrecall_logを取得（最新を除く——最新は今回のrecallなので次回評価）
    unevaluated = conn.execute(
        "SELECT id, session_ts, recalled_ids FROM recall_log "
        "WHERE evaluated_at IS NULL ORDER BY id ASC"
    ).fetchall()

    # 最新1件は今回のrecallの可能性が高いので除外
    if len(unevaluated) <= 1:
        conn.close()
        return None

    # 最新以外を全て評価し、最後の結果を返す
    last_result = None
    for row in unevaluated[:-1]:
        result = _evaluate_single_recall(conn, row)
        if result:
            last_result = result

    conn.commit()
    conn.close()
    return last_result


def _evaluate_single_recall(conn, row):
    """1件のrecall_logを評価する（connは呼び出し元でcommit）。"""
    log_id = row["id"]
    session_ts = row["session_ts"]
    recalled_ids = list(set(json.loads(row["recalled_ids"])))

    if not recalled_ids:
        return None

    # session_ts以降の会話内容を取得
    try:
        turns = conn.execute(
            "SELECT content FROM raw_turns WHERE timestamp > ? ORDER BY timestamp",
            (session_ts,)
        ).fetchall()
    except Exception:
        turns = []

    if not turns:
        return None

    # 会話全体を結合してembedding
    conversation_text = "\n".join(t["content"] for t in turns)
    if len(conversation_text.strip()) < 20:
        return None

    conv_vec = embed_text(conversation_text[:2000], is_query=True)
    if conv_vec is None:
        return None

    # recallが出した各記憶のembeddingとの類似度
    hits = []
    noise = []
    recalled_sims = {}
    for mid in recalled_ids:
        mem = conn.execute(
            "SELECT id, embedding FROM memories WHERE id = ?", (mid,)
        ).fetchone()
        if not mem or not mem["embedding"]:
            continue
        mem_vec = bytes_to_vec(mem["embedding"])
        sim = cosine_similarity(conv_vec, mem_vec)
        recalled_sims[mid] = sim
        if sim >= CALIBRATE_HIT_THRESHOLD:
            hits.append(mid)
        else:
            noise.append(mid)

    # 漏れ: recallが出さなかったが会話と高類似度の記憶
    missed = []
    recalled_set = set(recalled_ids)
    # sqlite-vecで会話ベクトルに近い記憶を取得
    miss_candidates = vec_search(conn, conv_vec, k=100, forgotten=False)
    if miss_candidates:
        for mid, _, sim in miss_candidates:
            if mid in recalled_set:
                continue
            if sim >= CALIBRATE_MISS_THRESHOLD:
                missed.append((mid, sim))
    else:
        # フォールバック
        candidates = conn.execute(
            "SELECT id, embedding FROM memories WHERE forgotten = 0 AND embedding IS NOT NULL"
        ).fetchall()
        for c in candidates:
            if c["id"] in recalled_set:
                continue
            c_vec = bytes_to_vec(c["embedding"])
            sim = cosine_similarity(conv_vec, c_vec)
            if sim >= CALIBRATE_MISS_THRESHOLD:
                missed.append((c["id"], sim))

    # 類似度上位のみ漏れとして報告（最大10件）
    missed.sort(key=lambda x: -x[1])
    missed_ids = [m[0] for m in missed[:10]]

    precision = len(hits) / len(recalled_ids) if recalled_ids else 0.0
    # 網羅: 会話に関連する記憶のうちrecallがカバーした割合
    relevant_total = len(hits) + len(missed_ids)
    recall_rate = len(hits) / relevant_total if relevant_total > 0 else 1.0

    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    conn.execute(
        """UPDATE recall_log SET
            accessed_ids = ?, precision = ?, recall_rate = ?,
            noise_ids = ?, missed_ids = ?, evaluated_at = ?
        WHERE id = ?""",
        (json.dumps({mid: round(s, 3) for mid, s in recalled_sims.items()}),
         precision, recall_rate,
         json.dumps(sorted(noise)), json.dumps(missed_ids),
         now, log_id)
    )

    return {
        "precision": precision,
        "recall_rate": recall_rate,
        "hits": sorted(hits),
        "noise": sorted(noise),
        "missed": missed_ids,
        "recalled_count": len(recalled_ids),
        "sims": recalled_sims,
    }


def calibrate_report():
    """メタ認知レポート: recallの精度を時系列で表示する。"""
    conn = get_connection()
    _ensure_recall_log(conn)

    rows = conn.execute(
        "SELECT session_ts, recalled_ids, accessed_ids, "
        "precision, recall_rate, noise_ids, missed_ids "
        "FROM recall_log WHERE evaluated_at IS NOT NULL "
        "ORDER BY session_ts DESC LIMIT 20"
    ).fetchall()

    if not rows:
        print("まだ評価データがない。数セッション使えば溜まる。")
        conn.close()
        return

    print(f"recall精度レポート（直近{len(rows)}セッション）:\n")

    total_precision = 0.0
    total_recall = 0.0
    count = 0

    for row in reversed(rows):
        ts = row["session_ts"][:16].replace("T", " ")
        p = row["precision"]
        r = row["recall_rate"]
        recalled = json.loads(row["recalled_ids"])
        noise = json.loads(row["noise_ids"]) if row["noise_ids"] else []
        missed = json.loads(row["missed_ids"]) if row["missed_ids"] else []

        # accessed_idsは新形式（dict）か旧形式（list）
        accessed_raw = json.loads(row["accessed_ids"]) if row["accessed_ids"] else {}
        n_hits = len(recalled) - len(noise)

        p_bar = "█" * int(p * 10) + "░" * (10 - int(p * 10))
        r_bar = "█" * int(r * 10) + "░" * (10 - int(r * 10))

        print(f"  {ts}  精度:{p_bar} {p:.0%}  網羅:{r_bar} {r:.0%}  "
              f"(出{len(recalled)} 的中{n_hits} 空振{len(noise)} 漏れ{len(missed)})")

        total_precision += p
        total_recall += r
        count += 1

    avg_p = total_precision / count
    avg_r = total_recall / count
    print(f"\n  平均  精度: {avg_p:.0%}  網羅: {avg_r:.0%}")

    # トレンド（前半と後半を比較）
    if count >= 4:
        half = count // 2
        recent_p = sum(r["precision"] for r in rows[:half]) / half
        older_p = sum(r["precision"] for r in rows[half:]) / (count - half)
        recent_r = sum(r["recall_rate"] for r in rows[:half]) / half
        older_r = sum(r["recall_rate"] for r in rows[half:]) / (count - half)

        dp = recent_p - older_p
        dr = recent_r - older_r
        trend_p = "↑" if dp > 0.05 else "↓" if dp < -0.05 else "→"
        trend_r = "↑" if dr > 0.05 else "↓" if dr < -0.05 else "→"
        print(f"  傾向  精度: {trend_p} ({dp:+.0%})  網羅: {trend_r} ({dr:+.0%})")

    conn.close()


# === 再帰的自己調整 (self-tune) ===
# recallの事後検証結果をもとに、パラメータを自動調整する。
# 記憶の内容（会話との関連度）→ 記憶の構造（パラメータ）→ 次のrecall → ...
# これが閉じたループになる。ghostが自分の想起を観察して、想起の仕方を変える。

SELF_TUNE_MIN_SESSIONS = 5     # 最低5セッションのデータが必要
SELF_TUNE_WINDOW = 20          # 直近20セッションを見る
SELF_TUNE_HALF_LIFE_RANGE = (7.0, 60.0)  # 半減期の範囲
SELF_TUNE_BALANCE_RANGE = (0.2, 0.9)     # 声バランスの範囲
SELF_TUNE_BOOST_RANGE = (1.0, 2.0)       # ブースト倍率の範囲


def _set_param_as_memory(conn, key, new_value, reason, avg_precision=None, avg_recall=None):
    """パラメータ更新を手続き記憶として保存する。

    meta_paramsテーブルではなく、通常の記憶として保存する。
    記憶だから減衰し、強化され、リンクされ、統合される。
    ルールと記憶の区別が溶ける。
    """
    p_str = f" 精度:{avg_precision:.0%}" if avg_precision is not None else ""
    r_str = f" 網羅:{avg_recall:.0%}" if avg_recall is not None else ""
    content = (f"[手続き:{key}] 値:{new_value:.4f} "
               f"理由:{reason}{p_str}{r_str}")

    emotions = ["determination"]
    arousal = 0.3
    importance = 3
    keywords = [key, "self-tune", "手続き"] + extract_keywords(reason)[:3]

    vec = embed_text(content, is_query=False)
    blob = vec_to_bytes(vec) if vec is not None else None
    now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    mem_uuid = str(_uuid.uuid4())
    emo_json = json.dumps(emotions)
    kw_json = json.dumps(keywords, ensure_ascii=False)

    conn.execute(
        """INSERT INTO memories
           (content, category, importance, emotions, arousal, keywords,
            embedding, uuid, updated_at, provenance, confidence, origin)
           VALUES (?, 'procedure', ?, ?, ?, ?, ?, ?, ?, 'self_tune', 0.7, 'self_tune')""",
        (content, importance, emo_json, arousal, kw_json, blob, mem_uuid, now_utc)
    )
    new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        """INSERT OR REPLACE INTO cortex
           (id, content, category, keywords, embedding, confidence, provenance, revision_count)
           VALUES (?, ?, 'procedure', ?, ?, 0.7, 'self_tune', 0)""",
        (new_id, content, kw_json, blob)
    )
    conn.execute(
        """INSERT OR REPLACE INTO limbic
           (id, emotions, arousal) VALUES (?, ?, ?)""",
        (new_id, emo_json, arousal)
    )
    # sqlite-vecに追加
    _sync_vec_insert(conn, new_id, blob)

    # 同じパラメータの古い手続き記憶とリンク（系譜を残す）
    if vec is not None:
        old_procs = conn.execute(
            "SELECT m.id, c.embedding, m.uuid FROM memories m "
            "JOIN cortex c ON c.id = m.id "
            "WHERE m.forgotten = 0 AND c.category = 'procedure' "
            "AND c.content LIKE ? AND m.id != ?",
            (f"[手続き:{key}]%", new_id)
        ).fetchall()
        for old in old_procs:
            if old["embedding"]:
                old_vec = bytes_to_vec(old["embedding"])
                sim = cosine_similarity(vec, old_vec)
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO links "
                        "(source_id, target_id, strength, source_uuid, target_uuid, "
                        "link_type, updated_at) VALUES (?, ?, ?, ?, ?, 'procedure_lineage', ?)",
                        (new_id, old["id"], sim, mem_uuid, old["uuid"], now_utc)
                    )
                except Exception:
                    pass

    # FTSインデックス（lindera が tokenize するため content そのまま）
    try:
        conn.execute(
            "INSERT INTO memories_fts (content, memory_id) VALUES (?, ?)",
            (content, new_id)
        )
    except Exception:
        pass

    # キャッシュを無効化
    global _tuned_cache
    _tuned_cache = None

    return new_id


def _evaluate_past_decisions(conn, avg_precision, avg_recall):
    """自己言及: self-tuneが自分の過去の判断を評価する。

    前回の手続き記憶が保存された後、精度/網羅は改善したか？
    良くなった手続き記憶はaccess_count++（強化）。
    悪くなった手続き記憶はそのまま放置（自然減衰に任せる）。
    効果不明のものは触らない。

    ゲーデル的: 「この判断は正しかったか」を判断するのは
    同じ判断システム。操作と操作対象が同じもの。
    """
    procs = conn.execute(
        "SELECT m.id, c.content, m.created_at, m.access_count "
        "FROM memories m JOIN cortex c ON c.id = m.id "
        "WHERE m.forgotten = 0 AND c.category = 'procedure' "
        "AND c.content LIKE ? "
        "ORDER BY m.created_at DESC LIMIT 10",
        (PROCEDURE_PREFIX + '%',)
    ).fetchall()

    if not procs:
        return

    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    reinforced = 0
    for proc in procs:
        # 手続き記憶に記録されている当時の精度/網羅を抽出
        match_p = re.search(r'精度:(\d+)%', proc["content"])
        match_r = re.search(r'網羅:(\d+)%', proc["content"])
        if not match_p or not match_r:
            continue
        old_p = int(match_p.group(1)) / 100
        old_r = int(match_r.group(1)) / 100

        # 改善したか？
        improved = False
        if avg_precision > old_p + 0.05 or avg_recall > old_r + 0.05:
            improved = True
        elif avg_precision >= old_p and avg_recall >= old_r:
            improved = True  # 悪化していなければOK

        if improved:
            conn.execute(
                "UPDATE memories SET access_count = access_count + 1, "
                "last_accessed = ? WHERE id = ?",
                (now, proc["id"])
            )
            reinforced += 1

    if reinforced > 0:
        conn.commit()
        print(f"  ↻ 自己評価: {reinforced}件の過去判断を強化（精度/網羅が維持or改善）")


def self_tune(dry_run=False):
    """recallの事後検証データからパラメータを自己調整する。

    精度が低い → ノイズが多い → 減衰を速く / 選択的に
    網羅が低い → 漏れが多い → 減衰を遅く / 広く拾う

    声ごとの的中率が分かれば、有効な声の重みを上げ、空振りが多い声の重みを下げる。

    自己言及: self-tuneは自分の過去の判断も評価する。
    前回の手続き記憶が良い結果を出していたら強化し、
    悪ければ自然減衰に任せる。自分の判断を自分で評価する。
    """
    conn = get_connection()
    _ensure_recall_log(conn)

    rows = conn.execute(
        "SELECT precision, recall_rate, recalled_ids, noise_ids, "
        "missed_ids, voice_attribution, accessed_ids "
        "FROM recall_log WHERE evaluated_at IS NOT NULL "
        "ORDER BY id DESC LIMIT ?",
        (SELF_TUNE_WINDOW,)
    ).fetchall()

    if len(rows) < SELF_TUNE_MIN_SESSIONS:
        msg = f"データ不足: {len(rows)}/{SELF_TUNE_MIN_SESSIONS}セッション"
        print(f"⚙ self-tune: {msg}")
        conn.close()
        return {"changed": False, "reason": msg}

    # --- 全体の精度・網羅 ---
    avg_precision = sum(r["precision"] for r in rows) / len(rows)
    avg_recall = sum(r["recall_rate"] for r in rows) / len(rows)

    # --- 自己言及: 過去の判断を評価する ---
    if not dry_run:
        _evaluate_past_decisions(conn, avg_precision, avg_recall)

    changes = {}

    # --- 半減期の調整 ---
    current_hl = get_param("half_life_days")
    hl_adjustment = 1.0

    if avg_recall < 0.25:
        # 漏れが多い: 減衰が速すぎる → 半減期を伸ばす
        hl_adjustment = 1.15
        hl_reason = f"網羅{avg_recall:.0%}が低い→減衰を緩める"
    elif avg_recall < 0.35:
        hl_adjustment = 1.05
        hl_reason = f"網羅{avg_recall:.0%}がやや低い→微調整"
    elif avg_precision < 0.25:
        # ノイズが多い: 古い記憶が残りすぎ → 半減期を縮める
        hl_adjustment = 0.85
        hl_reason = f"精度{avg_precision:.0%}が低い→減衰を速める"
    elif avg_precision < 0.35:
        hl_adjustment = 0.95
        hl_reason = f"精度{avg_precision:.0%}がやや低い→微調整"
    else:
        hl_reason = None

    if hl_reason:
        new_hl = max(SELF_TUNE_HALF_LIFE_RANGE[0],
                     min(SELF_TUNE_HALF_LIFE_RANGE[1], current_hl * hl_adjustment))
        if abs(new_hl - current_hl) > 0.1:
            changes["half_life_days"] = (current_hl, new_hl, hl_reason)

    # --- 声ごとの的中率 → バランス調整 ---
    voice_hits = {}   # voice_name -> hit_count
    voice_total = {}  # voice_name -> total_count
    voice_map = {"共感": "voice_empathy_balance",
                 "補完": "voice_complement_balance",
                 "批判": "voice_critic_balance"}

    for row in rows:
        va = row["voice_attribution"]
        if not va:
            continue
        attribution = json.loads(va)
        accessed = json.loads(row["accessed_ids"]) if row["accessed_ids"] else {}
        hit_ids = set()
        if isinstance(accessed, dict):
            hit_ids = {int(k) for k, v in accessed.items() if v >= CALIBRATE_HIT_THRESHOLD}
        else:
            hit_ids = set(accessed) if accessed else set()

        for voice_name, mem_ids in attribution.items():
            if voice_name not in voice_map:
                continue
            for mid in mem_ids:
                voice_total[voice_name] = voice_total.get(voice_name, 0) + 1
                if mid in hit_ids:
                    voice_hits[voice_name] = voice_hits.get(voice_name, 0) + 1

    # 声ごとのバランス調整: 的中率が高い声は右脳寄り（感性優先）を維持/強化、
    # 低い声は左脳寄り（分析優先）に寄せる
    for voice_name, param_key in voice_map.items():
        total = voice_total.get(voice_name, 0)
        if total < 5:
            continue  # データ不足
        hit_rate = voice_hits.get(voice_name, 0) / total
        current_bal = get_param(param_key)

        # 的中率が高い(>0.5): 今のバランスは機能している → 変えない
        # 的中率が低い(<0.3): 左脳寄りにして精度を上げる
        # 的中率が中間: 微調整
        if hit_rate < 0.2:
            new_bal = max(SELF_TUNE_BALANCE_RANGE[0], current_bal - 0.05)
            reason = f"{voice_name}的中率{hit_rate:.0%}→左脳寄りに"
        elif hit_rate > 0.6:
            new_bal = min(SELF_TUNE_BALANCE_RANGE[1], current_bal + 0.03)
            reason = f"{voice_name}的中率{hit_rate:.0%}→維持/強化"
        else:
            continue

        if abs(new_bal - current_bal) > 0.01:
            changes[param_key] = (current_bal, new_bal, reason)

    # --- 適用 ---
    if not changes:
        print(f"⚙ self-tune: 変更なし（精度{avg_precision:.0%} 網羅{avg_recall:.0%}、"
              f"現パラメータで安定）")
        conn.close()
        return {"changed": False, "precision": avg_precision, "recall": avg_recall}

    print(f"⚙ self-tune: 精度{avg_precision:.0%} 網羅{avg_recall:.0%} "
          f"({len(rows)}セッション分析)")
    saved_ids = []
    for key, (old, new, reason) in changes.items():
        print(f"  {key}: {old:.3f} → {new:.3f}  ({reason})")
        if not dry_run:
            mid = _set_param_as_memory(conn, key, round(new, 4), reason,
                                       avg_precision, avg_recall)
            saved_ids.append(mid)

    if not dry_run:
        conn.commit()
        print(f"  → {len(changes)}件を手続き記憶として保存 (IDs: {saved_ids})")
    else:
        print(f"  (dry-run: 実際には変更しない)")

    conn.close()
    return {"changed": True, "changes": {k: v[1] for k, v in changes.items()},
            "precision": avg_precision, "recall": avg_recall}


def self_tune_report():
    """現在の自己調整パラメータを表示する。手続き記憶から導出。"""
    conn = get_connection()
    _ensure_recall_log(conn)

    # キャッシュをリフレッシュ
    global _tuned_cache
    _tuned_cache = None
    _load_tuned_params()

    print("⚙ 自己調整パラメータ（手続き記憶から導出）:\n")
    print(f"  {'パラメータ':30s} {'現在値':>8s} {'デフォルト':>8s} {'状態'}")
    print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*20}")

    for key, default in _TUNABLE_DEFAULTS.items():
        current = get_param(key)
        marker = " *" if abs(current - default) > 0.01 else ""
        status = "手続き記憶から" if marker else "(デフォルト)"
        print(f"  {key:30s} {current:>8.3f} {default:>8.3f} {status}{marker}")

    # 生きた手続き記憶を表示
    procs = conn.execute(
        "SELECT m.id, c.content, m.created_at, m.access_count, m.forgotten "
        "FROM memories m JOIN cortex c ON c.id = m.id "
        "WHERE m.forgotten = 0 AND c.category = 'procedure' "
        "AND c.content LIKE ? "
        "ORDER BY m.created_at DESC LIMIT 20",
        (PROCEDURE_PREFIX + '%',)
    ).fetchall()

    if procs:
        print(f"\n  生きた手続き記憶（{len(procs)}件）:")
        for p in procs:
            ts = p["created_at"][:16].replace("T", " ")
            fresh = freshness(p["created_at"])
            content_short = p["content"][:70]
            print(f"    #{p['id']} {ts} fresh:{fresh:.0%} access:{p['access_count']} {content_short}")
    else:
        print(f"\n  手続き記憶なし（すべてデフォルト値）")

    conn.close()


# === メタ記憶の自己生成 (Level 3) ===
# 系が自分のrecallパターンを観察して、それについての記憶を自動生成する。
# 生成されたメタ記憶はembeddingされ、リンクされ、通常の記憶と同じように
# 将来のrecallに影響する。自分を観察する記憶。
#
# 再帰: recall → パターン観察 → メタ記憶生成 → 次のrecallに影響 → ...
#
# 人間で言えば「最近あのことばかり考えている」という自覚自体が
# 思考の方向を変える——メタ認知が認知を変える構造。

META_MEMORY_MIN_SESSIONS = 8    # メタ記憶生成に必要な最低セッション数
META_MEMORY_WINDOW = 20         # 直近Nセッションを分析
META_MEMORY_FIXATION_THRESHOLD = 0.6   # 総セッションの60%以上に出現したら「固着」
META_MEMORY_CHRONIC_MISS_THRESHOLD = 0.4  # 40%以上missedなら「慢性的漏れ」


def generate_meta_memories(dry_run=False):
    """系が自分のrecallパターンを観察して、メタ記憶を自動生成する。

    観察対象:
    1. 想起の固着 — 特定の記憶が繰り返しrecallされている（反芻）
    2. 慢性的な漏れ — 特定の記憶が繰り返しmissedされている（盲点）
    3. 精度の推移 — 系の全体的な健康状態

    生成されるメタ記憶は category='schema' で保存され、
    通常の記憶と同じようにrecallに影響する。
    """
    conn = get_connection()
    _ensure_recall_log(conn)

    rows = conn.execute(
        "SELECT recalled_ids, noise_ids, missed_ids, precision, recall_rate, "
        "voice_attribution, session_ts "
        "FROM recall_log WHERE evaluated_at IS NOT NULL "
        "ORDER BY id DESC LIMIT ?",
        (META_MEMORY_WINDOW,)
    ).fetchall()

    if len(rows) < META_MEMORY_MIN_SESSIONS:
        print(f"⊘ meta-memory: データ不足 ({len(rows)}/{META_MEMORY_MIN_SESSIONS})")
        conn.close()
        return []

    n_sessions = len(rows)
    generated = []

    # --- 1. 想起の固着検出 ---
    from collections import Counter
    recall_counter = Counter()
    for r in rows:
        for mid in json.loads(r["recalled_ids"]):
            recall_counter[mid] += 1

    # 既存のメタ記憶（固着系）を取得して重複防止
    existing_meta = set()
    meta_rows = conn.execute(
        "SELECT content FROM memories WHERE forgotten = 0 AND category = 'schema' "
        "AND content LIKE '[メタ観察]%'"
    ).fetchall()
    for mr in meta_rows:
        existing_meta.add(mr["content"])

    fixation_threshold = int(n_sessions * META_MEMORY_FIXATION_THRESHOLD)
    for mid, count in recall_counter.most_common(10):
        if count < fixation_threshold:
            break
        mem = conn.execute(
            "SELECT content, keywords FROM memories WHERE id = ?", (mid,)
        ).fetchone()
        if not mem:
            continue
        kws = json.loads(mem["keywords"])[:3] if mem["keywords"] else []
        kw_str = ", ".join(kws)
        content = (f"[メタ観察] 記憶#{mid}への固着: 直近{n_sessions}セッション中"
                   f"{count}回想起。キーワード: {kw_str}。"
                   f"内容: {mem['content'][:60]}")
        if content not in existing_meta:
            generated.append(("fixation", content, mid))

    # --- 2. 慢性的な漏れ検出 ---
    miss_counter = Counter()
    for r in rows:
        missed = json.loads(r["missed_ids"]) if r["missed_ids"] else []
        for mid in missed:
            miss_counter[mid] += 1

    chronic_threshold = int(n_sessions * META_MEMORY_CHRONIC_MISS_THRESHOLD)
    for mid, count in miss_counter.most_common(5):
        if count < chronic_threshold:
            break
        mem = conn.execute(
            "SELECT content, keywords FROM memories WHERE id = ?", (mid,)
        ).fetchone()
        if not mem:
            continue
        kws = json.loads(mem["keywords"])[:3] if mem["keywords"] else []
        kw_str = ", ".join(kws)
        content = (f"[メタ観察] 記憶#{mid}の慢性的漏れ: 直近{n_sessions}セッション中"
                   f"{count}回見逃し。キーワード: {kw_str}。"
                   f"内容: {mem['content'][:60]}")
        if content not in existing_meta:
            generated.append(("blind_spot", content, mid))

    # --- 3. 精度の推移観察 ---
    precisions = [r["precision"] for r in rows]
    recalls = [r["recall_rate"] for r in rows]
    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)

    # 前半と後半を比較してトレンドを検出
    half = len(rows) // 2
    if half >= 3:
        recent_p = sum(precisions[:half]) / half
        older_p = sum(precisions[half:]) / (len(precisions) - half)
        recent_r = sum(recalls[:half]) / half
        older_r = sum(recalls[half:]) / (len(recalls) - half)

        dp = recent_p - older_p
        dr = recent_r - older_r

        if abs(dp) > 0.10 or abs(dr) > 0.10:
            p_trend = "改善" if dp > 0 else "悪化"
            r_trend = "改善" if dr > 0 else "悪化"
            content = (f"[メタ観察] recall精度の推移: "
                       f"精度{avg_p:.0%}(前半→後半で{dp:+.0%}, {p_trend}), "
                       f"網羅{avg_r:.0%}(前半→後半で{dr:+.0%}, {r_trend})。"
                       f"直近{n_sessions}セッション分析。")
            if content not in existing_meta:
                generated.append(("trend", content, None))

    # --- 4. メタ記憶のメタ観察（自己言及） ---
    # 過去に生成した固着メタ記憶の対象IDが、今も固着しているか確認。
    # 「固着が続いている」「固着が解消された」を観察する。
    # メタ記憶がメタ記憶を観察する——型の階層なし。
    past_fixations = conn.execute(
        "SELECT m.id, c.content FROM memories m JOIN cortex c ON c.id = m.id "
        "WHERE m.forgotten = 0 AND c.category = 'schema' "
        "AND c.content LIKE '[メタ観察] 記憶#%への固着:%'"
    ).fetchall()

    for pf in past_fixations:
        # 対象の記憶IDを抽出
        id_match = re.search(r'記憶#(\d+)への固着', pf["content"])
        if not id_match:
            continue
        target_id = int(id_match.group(1))
        current_count = recall_counter.get(target_id, 0)
        current_rate = current_count / n_sessions if n_sessions > 0 else 0

        if current_rate < META_MEMORY_FIXATION_THRESHOLD * 0.5:
            # 固着が解消された
            content = (f"[メタ観察] 記憶#{target_id}の固着が解消: "
                       f"以前は固着と判断されたが、現在{current_count}/{n_sessions}回"
                       f"({current_rate:.0%})に減少。"
                       f"メタ記憶#{pf['id']}の観察対象。")
            if content not in existing_meta:
                generated.append(("meta_resolved", content, pf["id"]))
        elif current_rate >= META_MEMORY_FIXATION_THRESHOLD:
            # 固着が続いている——メタ記憶が効いていない
            content = (f"[メタ観察] 記憶#{target_id}の固着が持続: "
                       f"メタ記憶#{pf['id']}で固着を認識したが、"
                       f"現在も{current_count}/{n_sessions}回({current_rate:.0%})。"
                       f"認識だけでは解消しない。")
            if content not in existing_meta:
                generated.append(("meta_persistent", content, pf["id"]))

    # --- 5. 手続き記憶の効果観察（ルールを観察する） ---
    # self-tuneが生んだ手続き記憶の効果をメタ観察として記録する。
    # ルールが記憶なら、ルールの効果も記憶になる。
    active_procs = conn.execute(
        "SELECT m.id, c.content, m.access_count "
        "FROM memories m JOIN cortex c ON c.id = m.id "
        "WHERE m.forgotten = 0 AND c.category = 'procedure' "
        "AND c.content LIKE ? ORDER BY m.created_at DESC LIMIT 5",
        (PROCEDURE_PREFIX + '%',)
    ).fetchall()

    for proc in active_procs:
        match_p = re.search(r'精度:(\d+)%', proc["content"])
        match_r = re.search(r'網羅:(\d+)%', proc["content"])
        if not match_p or not match_r:
            continue
        old_p = int(match_p.group(1)) / 100
        old_r = int(match_r.group(1)) / 100

        dp = avg_p - old_p
        dr = avg_r - old_r
        if abs(dp) > 0.08 or abs(dr) > 0.08:
            effect = "改善" if (dp > 0 or dr > 0) else "悪化"
            content = (f"[メタ観察] 手続き記憶#{proc['id']}の効果: "
                       f"適用後の精度{dp:+.0%}, 網羅{dr:+.0%}({effect})。"
                       f"access:{proc['access_count']}。"
                       f"元の判断: {proc['content'][:60]}")
            if content not in existing_meta:
                generated.append(("rule_effect", content, proc["id"]))

    # --- 保存 ---
    if not generated:
        print(f"⊘ meta-memory: 新しい観察なし（精度{avg_p:.0%} 網羅{avg_r:.0%}）")
        conn.close()
        return []

    print(f"🪞 meta-memory: {len(generated)}件のメタ観察を生成")

    saved_ids = []
    for obs_type, content, ref_mid in generated:
        print(f"  [{obs_type}] {content[:80]}...")

        if dry_run:
            continue

        # メタ記憶として保存（通常のadd_memoryを経由せず直接保存する。
        # add_memoryを経由するとschema_primeが発火して無限再帰の危険がある）
        emotions = ["insight"]
        arousal = 0.4
        importance = 3
        keywords = extract_keywords(content)
        vec = embed_text(content, is_query=False)
        blob = vec_to_bytes(vec) if vec is not None else None
        now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        mem_uuid = str(_uuid.uuid4())
        emo_json = json.dumps(emotions)
        kw_json = json.dumps(keywords, ensure_ascii=False)

        conn.execute(
            """INSERT INTO memories
               (content, category, importance, emotions, arousal, keywords,
                embedding, uuid, updated_at, provenance, confidence, origin)
               VALUES (?, 'schema', ?, ?, ?, ?, ?, ?, ?, 'self_observation', 0.6, 'meta_memory')""",
            (content, importance, emo_json, arousal, kw_json, blob, mem_uuid, now_utc)
        )
        new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT OR REPLACE INTO cortex
               (id, content, category, keywords, embedding, confidence, provenance, revision_count)
               VALUES (?, ?, 'schema', ?, ?, 0.6, 'self_observation', 0)""",
            (new_id, content, kw_json, blob)
        )
        conn.execute(
            """INSERT OR REPLACE INTO limbic
               (id, emotions, arousal)
               VALUES (?, ?, ?)""",
            (new_id, emo_json, arousal)
        )
        # sqlite-vecに追加
        _sync_vec_insert(conn, new_id, blob)

        # 参照先の記憶とリンク
        if ref_mid and vec is not None:
            ref_mem = conn.execute(
                "SELECT embedding, uuid FROM memories WHERE id = ?", (ref_mid,)
            ).fetchone()
            if ref_mem and ref_mem["embedding"]:
                ref_vec = bytes_to_vec(ref_mem["embedding"])
                sim = cosine_similarity(vec, ref_vec)
                ref_uuid = ref_mem["uuid"]
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO links (source_id, target_id, strength, "
                        "source_uuid, target_uuid, link_type, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, 'meta_observation', ?)",
                        (new_id, ref_mid, sim, mem_uuid, ref_uuid, now_utc)
                    )
                except Exception:
                    pass

        saved_ids.append(new_id)

    if not dry_run:
        conn.commit()
        print(f"  → {len(saved_ids)}件保存")

    conn.close()
    return saved_ids


def get_stats():
    conn = get_connection()
    s = {}
    s["total"] = conn.execute("SELECT COUNT(*) FROM memories WHERE forgotten = 0").fetchone()[0]
    s["forgotten"] = conn.execute("SELECT COUNT(*) FROM memories WHERE forgotten = 1").fetchone()[0]
    s["links"] = conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]
    s["with_embedding"] = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE forgotten = 0 AND embedding IS NOT NULL"
    ).fetchone()[0]
    s["by_category"] = conn.execute(
        "SELECT category, COUNT(*) as cnt FROM memories WHERE forgotten = 0 GROUP BY category"
    ).fetchall()
    s["by_emotion"] = {}
    rows = conn.execute("SELECT emotions FROM memories WHERE forgotten = 0").fetchall()
    for row in rows:
        for emo in json.loads(row["emotions"]):
            s["by_emotion"][emo] = s["by_emotion"].get(emo, 0) + 1
    s["most_accessed"] = conn.execute(
        """SELECT id, content, access_count FROM memories
           WHERE forgotten = 0 AND access_count > 0
           ORDER BY access_count DESC LIMIT 5"""
    ).fetchall()
    s["most_linked"] = conn.execute(
        """SELECT m.id, m.content, COUNT(l.id) as link_count
           FROM memories m JOIN links l ON m.id = l.source_id
           WHERE m.forgotten = 0
           GROUP BY m.id ORDER BY link_count DESC LIMIT 5"""
    ).fetchall()
    s["merged"] = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE merged_from IS NOT NULL AND forgotten = 0"
    ).fetchone()[0]
    # 場所別の記憶数
    s["by_location"] = {}
    loc_rows = conn.execute(
        "SELECT spatial_context FROM memories WHERE forgotten = 0 AND spatial_context IS NOT NULL"
    ).fetchall()
    for row in loc_rows:
        try:
            loc = json.loads(row["spatial_context"]).get("location", "unknown")
            s["by_location"][loc] = s["by_location"].get(loc, 0) + 1
        except (json.JSONDecodeError, TypeError):
            pass
    conn.close()
    return s


def overview():
    """俯瞰モード: 記憶システム全体の構造を表示する。"""
    conn = get_connection()

    # --- 構造 ---
    alive = conn.execute("SELECT count(*) FROM memories WHERE forgotten = 0").fetchone()[0]
    dead = conn.execute("SELECT count(*) FROM memories WHERE forgotten = 1").fetchone()[0]
    total = alive + dead
    links = conn.execute("SELECT count(*) FROM links").fetchone()[0]
    orphans = conn.execute("""
        SELECT count(*) FROM memories m
        WHERE m.forgotten = 0
        AND NOT EXISTS (SELECT 1 FROM links l WHERE l.source_id = m.id)
    """).fetchone()[0]

    print("=== 構造 ===")
    print(f"  生存: {alive}件 / 忘却: {dead}件 / 合計: {total}件")
    print(f"  リンク: {links} (双方向込み)")
    print(f"  孤立記憶: {orphans}件")

    # カテゴリ分布
    cats = conn.execute(
        "SELECT category, count(*) FROM memories WHERE forgotten = 0 GROUP BY category ORDER BY count(*) DESC"
    ).fetchall()
    cat_str = ", ".join(f"{c[0]}:{c[1]}" for c in cats)
    print(f"  カテゴリ: {cat_str}")

    # --- 重心 ---
    print("\n=== 重心（ハブ記憶 トップ10） ===")
    hubs = conn.execute("""
        SELECT m.id, m.category, m.arousal, count(l.id) as lc,
               substr(m.content, 1, 60) as preview
        FROM memories m JOIN links l ON m.id = l.source_id
        WHERE m.forgotten = 0
        GROUP BY m.id ORDER BY lc DESC LIMIT 10
    """).fetchall()
    for h in hubs:
        print(f"  #{h[0]} [{h[1]}] a:{h[2]:.2f} links:{h[3]} {h[4]}")

    # --- 層（arousal分布） ---
    print("\n=== 層（arousal分布） ===")
    buckets = conn.execute("""
        SELECT
            CASE
                WHEN arousal < 0.2 THEN '低 (0-0.2)'
                WHEN arousal < 0.5 THEN '中 (0.2-0.5)'
                WHEN arousal < 0.85 THEN '高 (0.5-0.85)'
                ELSE '外傷 (0.85-1.0)'
            END as bucket,
            count(*)
        FROM memories WHERE forgotten = 0 GROUP BY bucket ORDER BY bucket
    """).fetchall()
    for b in buckets:
        bar = "█" * min(b[1], 40)
        print(f"  {b[0]:20s} {b[1]:3d}件 {bar}")

    # --- 月別 ---
    print("\n=== 時系列 ===")
    months = conn.execute("""
        SELECT substr(created_at, 1, 7) as month,
               count(*) as total,
               sum(CASE WHEN forgotten = 0 THEN 1 ELSE 0 END) as alive,
               sum(CASE WHEN forgotten = 1 THEN 1 ELSE 0 END) as dead
        FROM memories GROUP BY month ORDER BY month
    """).fetchall()
    for m in months:
        alive_bar = "█" * min(m[2], 40)
        dead_bar = "░" * min(m[3], 40)
        print(f"  {m[0]}: {alive_bar}{dead_bar} (alive:{m[2]} forgotten:{m[3]})")

    # --- 非schemaの生きた記憶 ---
    print("\n=== 非schema記憶 ===")
    non_schema = conn.execute("""
        SELECT id, category, arousal, access_count,
               substr(content, 1, 65) as preview
        FROM memories WHERE forgotten = 0 AND category != 'schema'
        ORDER BY access_count DESC, arousal DESC
    """).fetchall()
    for r in non_schema:
        acc = f"x{r[3]}" if r[3] > 0 else "  "
        print(f"  #{r[0]:3d} [{r[1]:10s}] a:{r[2]:.2f} {acc:>4s} {r[4]}")

    # --- delusionの領域 ---
    print("\n=== delusionの領域（忘却された高arousal） ===")
    forgotten_hot = conn.execute("""
        SELECT id, category, arousal, substr(content, 1, 65) as preview
        FROM memories WHERE forgotten = 1 AND arousal >= 0.5
        ORDER BY arousal DESC LIMIT 10
    """).fetchall()
    if forgotten_hot:
        for r in forgotten_hot:
            print(f"  #{r[0]:3d} [{r[1]:10s}] a:{r[2]:.2f} {r[3]}")
    else:
        print("  （なし）")

    # --- raw_turns ---
    print("\n=== raw_turns ===")
    try:
        rt_total = conn.execute("SELECT count(*) FROM raw_turns").fetchone()[0]
        rt_sessions = conn.execute("SELECT count(DISTINCT session_id) FROM raw_turns").fetchone()[0]
        rt_earliest = conn.execute("SELECT min(timestamp) FROM raw_turns").fetchone()[0]
        rt_latest = conn.execute("SELECT max(timestamp) FROM raw_turns").fetchone()[0]
        print(f"  {rt_total}ターン / {rt_sessions}セッション")
        if rt_earliest and rt_latest:
            print(f"  期間: {rt_earliest[:10]} ~ {rt_latest[:10]}")
    except Exception:
        print("  （テーブルなし）")

    # --- FTS ---
    print("\n=== FTS5 ===")
    try:
        mfts = conn.execute("SELECT count(*) FROM memories_fts").fetchone()[0]
        rfts = conn.execute("SELECT count(*) FROM raw_turns_fts").fetchone()[0]
        print(f"  memories: {mfts}件 / raw_turns: {rfts}件")
    except Exception:
        print("  （未構築）")

    conn.close()


# --- 表示 ---

EMOTION_EMOJI = {
    "surprise": "😲",
    "conflict": "⚡",
    "determination": "🔥",
    "insight": "💎",
    "connection": "🤝",
    "anxiety": "😰",
}


def format_memory(row, similarity=None, score=None):
    emotions = json.loads(row["emotions"]) if row["emotions"] else []
    emo_str = " ".join(EMOTION_EMOJI.get(e, "·") for e in emotions) if emotions else "·"
    stars = "★" * row["importance"]
    accessed = f" (参照:{row['access_count']}回)" if row["access_count"] > 0 else ""
    sim_str = f" [sim:{similarity:.3f}]" if similarity is not None else ""
    score_str = f" [score:{score:.3f}]" if score is not None else ""
    fresh = freshness(row["created_at"])
    fresh_str = f" 鮮度:{fresh:.0%}" if fresh < 0.95 else ""
    return f"  #{row['id']} {emo_str} {stars} {row['content'][:80]}{accessed}{sim_str}{score_str}{fresh_str}"


def format_memory_simple(row):
    """シンプルモード: 内容だけ。スコア・情動・メタデータは隠す。人間的な想起。"""
    content = row["content"]
    if len(content) > 200:
        content = content[:200] + "..."
    return f"  #{row['id']} {content}"


# v30 graph handle — pull 型 navigate 用の統一スキーマ
def _get_top_links(conn, memory_id, limit=5):
    """指定 id から最強の link を handle 用に取り出す。"""
    rows = conn.execute(
        """SELECT l.target_id, l.strength, m.content
           FROM links l JOIN memories m ON l.target_id = m.id
           WHERE l.source_id = ? AND m.forgotten = 0
           ORDER BY l.strength DESC LIMIT ?""",
        (memory_id, limit)
    ).fetchall()
    out = []
    for r in rows:
        snippet = (r["content"] or "")[:40]
        out.append({
            "id": r["target_id"],
            "strength": round(r["strength"], 3),
            "snippet": snippet
        })
    return out


def _get_domains_for(conn, memory_id):
    """cortex.domains を読み出して list で返す。"""
    r = conn.execute("SELECT domains FROM cortex WHERE id = ?", (memory_id,)).fetchone()
    if not r or not r["domains"]:
        return ["unknown"]
    try:
        return json.loads(r["domains"])
    except (json.JSONDecodeError, TypeError):
        return ["unknown"]


def format_handle(conn, row, full=False, minimal=False, include_links=True):
    """graph handle: LLM が pull 型 navigate するための構造化データ。

    - minimal: {id, snippet, domains} のみ（トークン節約）
    - default: 主要フィールド + top_links (5件)
    - full: content 全文 + merged_from + 全メタ
    """
    content = row["content"] or ""
    domains = _get_domains_for(conn, row["id"])

    if minimal:
        snippet = content[:120]
        return {
            "id": row["id"],
            "snippet": snippet,
            "domains": domains,
        }

    handle = {
        "id": row["id"],
        "snippet": content[:120],
        "category": row["category"] if "category" in row.keys() else None,
        "domains": domains,
        "keywords": json.loads(row["keywords"]) if row["keywords"] else [],
        "importance": row["importance"] if "importance" in row.keys() else None,
        "arousal": row["arousal"] if "arousal" in row.keys() else None,
        "emotions": json.loads(row["emotions"]) if row["emotions"] else [],
        "flashbulb": row["flashbulb"] if "flashbulb" in row.keys() else None,
        "confidence": row["confidence"] if "confidence" in row.keys() else None,
        "provenance": row["provenance"] if "provenance" in row.keys() else None,
        "created_at": row["created_at"] if "created_at" in row.keys() else None,
        "last_accessed": row["last_accessed"] if "last_accessed" in row.keys() else None,
        "access_count": row["access_count"] if "access_count" in row.keys() else 0,
    }

    if include_links:
        handle["top_links"] = _get_top_links(conn, row["id"])

    if full:
        handle["content"] = content
        mf = row["merged_from"] if "merged_from" in row.keys() else None
        if mf:
            try:
                handle["merged_from"] = json.loads(mf)
            except (json.JSONDecodeError, TypeError):
                handle["merged_from"] = None
        handle["uuid"] = row["uuid"] if "uuid" in row.keys() else None

    return handle


def _json_envelope(cmd, results, query=None, meta=None):
    """v30 共通 JSON エンベロープ。"""
    envelope = {
        "version": "v30",
        "cmd": cmd,
        "query": query,
        "results": results,
        "meta": meta or {},
    }
    if "count" not in envelope["meta"]:
        envelope["meta"]["count"] = len(results) if isinstance(results, list) else None
    return envelope


def _extract_mentions_batch(fragments, timeout=120):
    """記憶断片のリストから、本文中に言及された人名を抽出する。
    Gemini CLI 経由で fragments 並び順どおりに [[name, ...], ...] を返す。
    失敗時は None（呼び出し側が batch 単位で諦める）。"""
    if not fragments:
        return None
    numbered = "\n\n".join(
        f"[{i}] {(frag or '')[:600]}" for i, frag in enumerate(fragments)
    )
    prompt = (
        "以下は番号付きの記憶断片。それぞれの本文中に明示的に言及された**人物名**を"
        "JSON 配列で抽出せよ。\n"
        "条件:\n"
        "- 固有名詞のみ（実在人物名・ハンドル名・プロジェクト内のエージェント名）\n"
        "- 一般名詞（user / assistant / 人 / 誰か）は除外\n"
        "- 本人（書き手）は含めない — 第三者への言及のみ\n"
        "- 自信がない時は空配列 []\n"
        "- 正規化（フルネームはそのまま、敬称・様・さんは外す）\n\n"
        "出力フォーマット（厳密な JSON、前置き後置きなし）:\n"
        '{"results": [{"i": 0, "mentions": ["name1", "name2"]}, ...]}\n\n'
        f"断片:\n{numbered}"
    )
    model = os.environ.get("MENTIONS_MODEL", _GEMINI_DEFAULT_MODEL)
    response_text = _gemini_cli_call(prompt, model=model, timeout=timeout)
    parsed = _parse_gemini_json(response_text)
    if parsed is None:
        return None
    try:
        results = parsed.get("results", [])
        out = [[] for _ in fragments]
        for item in results:
            idx = item.get("i")
            names = item.get("mentions", [])
            if isinstance(idx, int) and 0 <= idx < len(fragments) and isinstance(names, list):
                out[idx] = [str(n).strip() for n in names if isinstance(n, str) and n.strip()]
        return out
    except Exception:
        return None


def extract_mentions(limit=100, batch=10, dry_run=False):
    """limbic.relational_context に mentions キーがない記憶を対象に、
    本文から人名を抽出して mentions フィールドを追加する。

    - 既に mentions キーがあればスキップ（空配列も「抽出済み」とみなす）
    - batch 単位で Gemini を呼ぶ（API コール数 = ceil(limit / batch)）
    - failed batch は relational_context を変更しない（次回再挑戦）
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT m.id, m.content, l.relational_context
               FROM memories m
               LEFT JOIN limbic l ON l.id = m.id
               WHERE m.forgotten = 0 AND m.content IS NOT NULL
               ORDER BY m.importance DESC, m.id DESC"""
        ).fetchall()
        pending = []
        for r in rows:
            try:
                rc = json.loads(r["relational_context"]) if r["relational_context"] else {}
            except (json.JSONDecodeError, TypeError):
                rc = {}
            if not isinstance(rc, dict):
                rc = {}
            if "mentions" in rc:
                continue
            pending.append((r["id"], r["content"], rc))
            if len(pending) >= limit:
                break
        stats = {"processed": 0, "updated": 0, "failed_batches": 0, "pending_total": len(pending)}
        if not pending:
            return stats
        for start in range(0, len(pending), batch):
            chunk = pending[start:start + batch]
            fragments = [c[1] for c in chunk]
            if dry_run:
                stats["processed"] += len(chunk)
                continue
            extracted = _extract_mentions_batch(fragments)
            if extracted is None:
                stats["failed_batches"] += 1
                continue
            for (mid, _content, rc), names in zip(chunk, extracted):
                rc["mentions"] = names
                conn.execute(
                    "UPDATE limbic SET relational_context = ? WHERE id = ?",
                    (json.dumps(rc, ensure_ascii=False), mid)
                )
                stats["processed"] += 1
                if names:
                    stats["updated"] += 1
            conn.commit()
        return stats
    finally:
        conn.close()


def _distill_state(memories_with_scores, timeout=20):
    """記憶断片から「今の思考状態」を独白として蒸留する。

    Voight-Kampff検査でフラット化・パフォーマンス臭を出さない為に、
    briefing形式を避け、一人称の独り言として再構築する。
    10件並列の断片より1段落の蒸留の方が、コンテキストに入った時に
    「それが自分」として振る舞える。

    失敗時は None。呼び出し側は従来形式にフォールバックする。
    """
    if not memories_with_scores:
        return None

    fragments_text = "\n---\n".join(
        (row["content"] or "")[:400] for row, _ in memories_with_scores
    )

    prompt = (
        "以下は一人の人物の過去の発言断片。"
        "バラバラの時期・話題から集められている。\n"
        "これを読んで、この人物が今の意識で抱えている思考の糸を、"
        "本人の独り言として書け。\n\n"
        "条件:\n"
        "- 一人称の独白（「〜が気になる」「〜で苛ついてる」「〜を追ってる」）\n"
        "- 固有名詞を残す（具体的な事例名・プロジェクト名・論点）\n"
        "- 温度を残す（疲弊、懐疑、確信、興奮など）\n"
        "- 複数の糸が並存してよい（話題が飛んでも自然）\n"
        "- briefing口調禁止（「あなたは〜」「現在〜」「状態は〜」を使うな）\n"
        "- 地続きの文章。箇条書き禁止\n\n"
        "出力は独白本文のみ。前置き・後置き不要。\n\n"
        f"断片:\n{fragments_text}"
    )

    model = os.environ.get("RECALL_DISTILL_MODEL", _GEMINI_DEFAULT_MODEL)
    text = _gemini_cli_call(prompt, model=model, timeout=max(timeout, 120))
    if text is None:
        return None
    return text.strip() or None


def format_memory_compact(row, score=None, fragments_only=False):
    """コンパクトモード: recallのデフォルト。content要約付き2行表示。"""
    emotions = json.loads(row["emotions"]) if row["emotions"] else []
    keywords = json.loads(row["keywords"]) if row["keywords"] else []
    emo_str = " ".join(EMOTION_EMOJI.get(e, "·") for e in emotions[:2]) if emotions else ""
    stars = "★" * row["importance"]
    frag_str = ", ".join(keywords[:4])
    score_str = f" ({score:.2f})" if score is not None else ""
    line = f"  #{row['id']} {emo_str} {stars} [{frag_str}]{score_str}"
    if not fragments_only and row["content"]:
        content = row["content"][:80]
        if len(row["content"]) > 80:
            content += "..."
        line += f"\n       「{content}」"
    fb = row["flashbulb"] if "flashbulb" in row.keys() else None
    if fb:
        line += f"\n       🔥「{fb}」"
    return line


def format_memory_reconstructive(conn, row, similarity=None, score=None, fragments_only=False):
    """再構成モード: 断片+content要約+情動+連想リンク。"""
    emotions = json.loads(row["emotions"]) if row["emotions"] else []
    keywords = json.loads(row["keywords"]) if row["keywords"] else []
    emo_str = " ".join(EMOTION_EMOJI.get(e, "·") for e in emotions) if emotions else "·"
    stars = "★" * row["importance"]
    sim_str = f" [sim:{similarity:.3f}]" if similarity is not None else ""
    score_str = f" [score:{score:.3f}]" if score is not None else ""

    # 連想リンク先の断片を取得（上位3件）
    linked_fragments = []
    links = conn.execute(
        """SELECT m.keywords FROM links l
           JOIN memories m ON l.target_id = m.id
           WHERE l.source_id = ? AND m.forgotten = 0
           ORDER BY l.strength DESC LIMIT 3""",
        (row["id"],)
    ).fetchall()
    for link in links:
        lkw = json.loads(link["keywords"]) if link["keywords"] else []
        if lkw:
            # リンク先からランダムに1-2個の断片を取る
            sample = random.sample(lkw, min(2, len(lkw)))
            linked_fragments.extend(sample)

    # 出力: 断片 + content要約 + 情動 + 連想からの断片
    frag_str = ", ".join(keywords[:6])
    line = f"  #{row['id']} {emo_str} {stars} [{frag_str}]{sim_str}{score_str}"
    if not fragments_only and row["content"]:
        content = row["content"][:120]
        if len(row["content"]) > 120:
            content += "..."
        line += f"\n         ↳ 「{content}」"
    if linked_fragments:
        line += f"\n         ↳ 連想: [{', '.join(linked_fragments)}]"
    if emotions:
        line += f"\n         ↳ 情動: {', '.join(emotions)} (覚醒度:{row['arousal']:.2f})"
    fb = row["flashbulb"] if "flashbulb" in row.keys() else None
    if fb:
        line += f"\n         🔥「{fb}」"
    return line


def format_memory_detail(row):
    emotions = json.loads(row["emotions"]) if row["emotions"] else []
    keywords = json.loads(row["keywords"]) if row["keywords"] else []
    merged = json.loads(row["merged_from"]) if row["merged_from"] else None
    lines = [
        f"  記憶 #{row['id']}",
        f"  内容: {row['content']}",
        f"  カテゴリ: {row['category']} | 重要度: {'★' * row['importance']}",
        f"  情動: {', '.join(emotions) if emotions else '中立'} (覚醒度:{row['arousal']:.2f})",
        f"  断片: [{', '.join(keywords[:8])}]",
        f"  鮮度: {freshness(row['created_at']):.0%} | 参照: {row['access_count']}回",
        f"  記録: {row['created_at'][:10]}",
    ]
    # 場所の表示
    sc = row["spatial_context"] if "spatial_context" in row.keys() else None
    if sc:
        try:
            loc = json.loads(sc).get("location", "")
            if loc:
                lines.append(f"  場所: {loc}")
        except (json.JSONDecodeError, TypeError):
            pass
    # 信頼度・出自
    conf = row["confidence"] if "confidence" in row.keys() and row["confidence"] is not None else None
    prov = row["provenance"] if "provenance" in row.keys() and row["provenance"] else None
    if conf is not None or prov:
        conf_str = f"{conf:.0%}" if conf is not None else "不明"
        prov_str = prov or "不明"
        lines.append(f"  信頼度: {conf_str} | 出自: {prov_str}")
    # 改訂回数
    rev = row["revision_count"] if "revision_count" in row.keys() and row["revision_count"] is not None else 0
    if rev > 0:
        lines.append(f"  改訂: {rev}回")
    if merged:
        lines.append(f"  統合元: #{', #'.join(str(m) for m in merged)}")
    fb = row["flashbulb"] if "flashbulb" in row.keys() else None
    if fb:
        lines.append(f"  🔥 フラッシュバルブ: 「{fb}」")
    if row["source_conversation"]:
        lines.append(f"  出典: {row['source_conversation']}")
    return "\n".join(lines)



def export_memories(filename=None):
    """全記憶とリンクをJSONファイルにエクスポートする。embeddingは除外。"""
    if filename is None:
        filename = f"memory_export_{datetime.now().strftime('%Y%m%d')}.json"

    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM memories WHERE forgotten = 0"
    ).fetchall()

    memories = []
    for row in rows:
        memories.append({
            "id": row["id"],
            "content": row["content"],
            "category": row["category"],
            "importance": row["importance"],
            "emotions": json.loads(row["emotions"]) if row["emotions"] else [],
            "arousal": row["arousal"],
            "keywords": json.loads(row["keywords"]) if row["keywords"] else [],
            "created_at": row["created_at"],
            "access_count": row["access_count"],
            "source_conversation": row["source_conversation"],
            "temporal_context": json.loads(row["temporal_context"]) if row["temporal_context"] else None,
            "spatial_context": json.loads(row["spatial_context"]) if row.get("spatial_context") else None,
            "relational_context": json.loads(row["relational_context"]) if row.get("relational_context") else None,
            "merged_from": json.loads(row["merged_from"]) if row["merged_from"] else None,
            "provenance": row["provenance"] if "provenance" in row.keys() else None,
            "confidence": row["confidence"] if "confidence" in row.keys() else None,
            "flashbulb": row["flashbulb"] if "flashbulb" in row.keys() else None,
        })

    # リンクもエクスポート
    link_rows = conn.execute(
        "SELECT source_id, target_id, strength FROM links"
    ).fetchall()
    links = [
        {"source": lr["source_id"], "target": lr["target_id"], "strength": lr["strength"]}
        for lr in link_rows
    ]

    conn.close()

    data = {"memories": memories, "links": links}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✓ {len(memories)}件の記憶をエクスポート: {filename}")


# ============================================================
# 7. P2P同期
# ============================================================

def _get_node_id():
    """この端末のnode_idを返す。"""
    conn = get_connection()
    row = conn.execute("SELECT value FROM node_info WHERE key = 'node_id'").fetchone()
    conn.close()
    return row[0] if row else None


def sync_export(since=None):
    """同期用: since以降に更新された記憶・リンクをdictで返す。sinceがNoneなら全件。"""
    import base64
    conn = get_connection()

    if since:
        mem_rows = conn.execute(
            "SELECT * FROM memories WHERE updated_at > ? AND forgotten = 0",
            (since,)
        ).fetchall()
        link_rows = conn.execute(
            "SELECT * FROM links WHERE updated_at > ?", (since,)
        ).fetchall()
        # 忘却された記憶も同期（相手側でも忘却するため）
        forgotten_rows = conn.execute(
            "SELECT uuid FROM memories WHERE updated_at > ? AND forgotten = 1",
            (since,)
        ).fetchall()
    else:
        mem_rows = conn.execute(
            "SELECT * FROM memories WHERE forgotten = 0"
        ).fetchall()
        link_rows = conn.execute("SELECT * FROM links").fetchall()
        forgotten_rows = conn.execute(
            "SELECT uuid FROM memories WHERE forgotten = 1"
        ).fetchall()

    memories = []
    for row in mem_rows:
        mem = {
            "uuid": row["uuid"],
            "content": row["content"],
            "category": row["category"],
            "importance": row["importance"],
            "emotions": row["emotions"],
            "arousal": row["arousal"],
            "keywords": row["keywords"],
            "created_at": row["created_at"],
            "last_accessed": row["last_accessed"],
            "access_count": row["access_count"],
            "source_conversation": row["source_conversation"],
            "merged_from": row["merged_from"],
            "context_expires_at": row["context_expires_at"],
            "temporal_context": row["temporal_context"],
            "spatial_context": row["spatial_context"],
            "relational_context": row["relational_context"] if "relational_context" in row.keys() else None,
            "updated_at": row["updated_at"],
            "provenance": row["provenance"] if "provenance" in row.keys() else None,
            "confidence": row["confidence"] if "confidence" in row.keys() else None,
            "flashbulb": row["flashbulb"] if "flashbulb" in row.keys() else None,
        }
        # embeddingはbase64でエンコード
        if row["embedding"]:
            mem["embedding_b64"] = base64.b64encode(row["embedding"]).decode("ascii")
        memories.append(mem)

    links = []
    for lr in link_rows:
        links.append({
            "source_uuid": lr["source_uuid"],
            "target_uuid": lr["target_uuid"],
            "strength": lr["strength"],
            "link_type": lr["link_type"],
            "created_at": lr["created_at"],
            "updated_at": lr["updated_at"],
        })

    forgotten_uuids = [r["uuid"] for r in forgotten_rows]

    node_id = _get_node_id()
    conn.close()

    return {
        "node_id": node_id,
        "since": since,
        "memories": memories,
        "links": links,
        "forgotten_uuids": forgotten_uuids,
    }


def sync_import(data):
    """同期用: 受信データをマージする。衝突解決込み。"""
    import base64
    conn = get_connection()

    stats = {"new": 0, "updated": 0, "links_new": 0, "links_updated": 0, "forgotten": 0}

    # 1. 記憶のマージ
    valid_categories = VALID_CATEGORIES
    for mem in data.get("memories", []):
        try:
            for key in ("uuid", "content", "category", "importance", "emotions", "arousal",
                        "keywords", "created_at", "last_accessed", "access_count",
                        "source_conversation", "updated_at"):
                if key not in mem:
                    raise ValueError(f"missing field: {key}")
            if mem["category"] not in valid_categories:
                raise ValueError(f"invalid category: {mem['category']}")
            # optional fields (older peers may not send these)
            mem.setdefault("merged_from", None)
            mem.setdefault("context_expires_at", None)
            mem.setdefault("temporal_context", None)
            mem.setdefault("spatial_context", None)
            mem.setdefault("relational_context", None)
            mem.setdefault("flashbulb", None)

            existing = conn.execute(
                "SELECT id, access_count, last_accessed, updated_at, importance, arousal FROM memories WHERE uuid = ?",
                (mem["uuid"],)
            ).fetchone()

            if existing is None:
                # 新規: INSERT
                blob = None
                if mem.get("embedding_b64"):
                    blob = base64.b64decode(mem["embedding_b64"])
                conn.execute(
                    """INSERT INTO memories
                       (uuid, content, category, importance, emotions, arousal, keywords,
                        created_at, last_accessed, access_count, forgotten,
                        source_conversation, embedding, merged_from,
                        context_expires_at, temporal_context, spatial_context, updated_at, flashbulb)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (mem["uuid"], mem["content"], mem["category"], mem["importance"],
                     mem["emotions"], mem["arousal"], mem["keywords"],
                     mem["created_at"], mem["last_accessed"], mem["access_count"],
                     mem["source_conversation"], blob, mem["merged_from"],
                     mem["context_expires_at"], mem["temporal_context"],
                     mem["spatial_context"], mem["updated_at"], mem["flashbulb"])
                )
                new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                conn.execute(
                    """INSERT OR REPLACE INTO cortex
                       (id, content, category, keywords, embedding, confidence, provenance, revision_count, merged_from)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                    (new_id, mem["content"], mem["category"], mem["keywords"],
                     blob, mem.get("confidence", 0.7), mem.get("provenance", "unknown"), mem["merged_from"])
                )
                conn.execute(
                    """INSERT OR REPLACE INTO limbic
                       (id, emotions, arousal, flashbulb, temporal_context, spatial_context, relational_context)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (new_id, mem["emotions"], mem["arousal"], mem["flashbulb"],
                     mem["temporal_context"], mem["spatial_context"],
                     mem.get("relational_context"))
                )
                # sqlite-vecに追加
                _sync_vec_insert(conn, new_id, blob)
                stats["new"] += 1
            else:
                # 既存: マージ
                # access_count: 合算はしない。大きいほうを採用（同じ記憶を両端末で触った場合）
                new_access = max(existing["access_count"], mem["access_count"])
                # last_accessed: 新しいほう
                new_last = max(existing["last_accessed"] or "", mem["last_accessed"] or "") or None
                # content/emotions/arousal/importance: updated_atが新しいほう
                remote_newer = (mem["updated_at"] or "") > (existing["updated_at"] or "")
                if remote_newer:
                    blob = None
                    if mem.get("embedding_b64"):
                        blob = base64.b64decode(mem["embedding_b64"])
                    conn.execute(
                        """UPDATE memories SET
                           content = ?, category = ?, importance = ?, emotions = ?, arousal = ?,
                           keywords = ?, access_count = ?, last_accessed = ?,
                           source_conversation = ?, embedding = COALESCE(?, embedding),
                           merged_from = ?, context_expires_at = ?,
                           temporal_context = ?, spatial_context = ?, updated_at = ?,
                           flashbulb = ?
                           WHERE uuid = ?""",
                        (mem["content"], mem["category"], mem["importance"], mem["emotions"], mem["arousal"],
                         mem["keywords"], new_access, new_last,
                         mem["source_conversation"], blob, mem["merged_from"], mem["context_expires_at"],
                         mem["temporal_context"], mem["spatial_context"],
                         mem["updated_at"], mem["flashbulb"], mem["uuid"])
                    )
                    # cortex/limbicも同期
                    mid = existing["id"]
                    conn.execute(
                        """UPDATE cortex SET content = ?, category = ?, keywords = ?,
                           embedding = COALESCE(?, embedding), merged_from = ?,
                           confidence = ?, provenance = ?
                           WHERE id = ?""",
                        (mem["content"], mem["category"], mem["keywords"],
                         blob, mem["merged_from"],
                         mem.get("confidence", 0.7), mem.get("provenance", "unknown"), mid)
                    )
                    conn.execute(
                        """UPDATE limbic SET emotions = ?, arousal = ?, flashbulb = ?,
                           temporal_context = ?, spatial_context = ?,
                           relational_context = ?
                           WHERE id = ?""",
                        (mem["emotions"], mem["arousal"], mem["flashbulb"],
                         mem["temporal_context"], mem["spatial_context"],
                         mem.get("relational_context"), mid)
                    )
                else:
                    # ローカルが新しい場合もaccess_countとlast_accessedだけ更新
                    conn.execute(
                        "UPDATE memories SET access_count = ?, last_accessed = ? WHERE uuid = ?",
                        (new_access, new_last, mem["uuid"])
                    )
                stats["updated"] += 1
        except Exception as e:
            print(f"  [warn] skip invalid memory: {e}")
            continue

    # 2. 忘却の同期
    for fuuid in data.get("forgotten_uuids", []):
        result = conn.execute(
            "UPDATE memories SET forgotten = 1 WHERE uuid = ? AND forgotten = 0",
            (fuuid,)
        )
        if result.rowcount > 0:
            stats["forgotten"] += 1

    # 3. リンクのマージ
    # まずuuid→local idのマップを作る
    uuid_to_id = {}
    for row in conn.execute("SELECT id, uuid FROM memories WHERE uuid IS NOT NULL").fetchall():
        uuid_to_id[row["uuid"]] = row["id"]

    for link in data.get("links", []):
        src_id = uuid_to_id.get(link["source_uuid"])
        tgt_id = uuid_to_id.get(link["target_uuid"])
        if src_id is None or tgt_id is None:
            continue  # 片方の記憶がない（まだ同期されてない）

        existing_link = conn.execute(
            "SELECT id, strength FROM links WHERE source_id = ? AND target_id = ?",
            (src_id, tgt_id)
        ).fetchone()

        if existing_link is None:
            conn.execute(
                """INSERT INTO links (source_id, target_id, strength, link_type,
                   source_uuid, target_uuid, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (src_id, tgt_id, link["strength"], link["link_type"],
                 link["source_uuid"], link["target_uuid"],
                 link["created_at"], link["updated_at"])
            )
            stats["links_new"] += 1
        else:
            # strengthは大きいほうを採用
            new_strength = max(existing_link["strength"], link["strength"])
            if new_strength != existing_link["strength"]:
                conn.execute(
                    "UPDATE links SET strength = ? WHERE id = ?",
                    (new_strength, existing_link["id"])
                )
                stats["links_updated"] += 1

    conn.commit()
    conn.close()
    return stats


def import_memories(filename):
    """JSONファイルから記憶をインポートする。重複はスキップ。"""
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    memories = data.get("memories", data) if isinstance(data, dict) else data

    conn = get_connection()
    existing_contents = set(
        row[0] for row in conn.execute(
            "SELECT content FROM memories WHERE forgotten = 0"
        ).fetchall()
    )
    conn.close()

    imported = 0
    skipped = 0
    for mem in memories:
        content = mem["content"]
        if content in existing_contents:
            skipped += 1
            continue
        category = mem.get("category", "fact")
        source = mem.get("source_conversation")
        add_memory(content, category, source)
        imported += 1

    print(f"✓ {imported}件インポート, {skipped}件スキップ（重複）")


# --- CLI ---
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == "init":
        init_db()

    elif cmd == "backfill-embeddings":
        backfill_embeddings(dry_run="--dry-run" in sys.argv,
                            missing_only="--missing-only" in sys.argv)

    elif cmd == "reconcile":
        _since = None
        if "--since" in sys.argv:
            _i = sys.argv.index("--since")
            if _i + 1 < len(sys.argv):
                _since = sys.argv[_i + 1]
        reconcile(dry_run="--execute" not in sys.argv, since=_since)

    elif cmd == "add":
        if len(sys.argv) < 3:
            print("使い方: python memory.py add \"内容\" [--category CAT] [--source SRC] [--flashbulb TEXT] [--origin ORIGIN] [--domain D1,D2]")
            print("  位置引数: python memory.py add \"内容\" category [source]")
            return
        content = sys.argv[2]
        args = sys.argv[3:]
        category = "fact"
        force_add = False
        category_set = False
        source = None
        flashbulb = None
        origin = None
        domains = None
        i = 0
        while i < len(args):
            if args[i] == "--category" and i + 1 < len(args):
                category = args[i + 1]
                category_set = True
                i += 2
            elif args[i] == "--source" and i + 1 < len(args):
                source = args[i + 1]
                i += 2
            elif args[i] == "--flashbulb" and i + 1 < len(args):
                flashbulb = args[i + 1]
                i += 2
            elif args[i] == "--origin" and i + 1 < len(args):
                origin = args[i + 1]
                i += 2
            elif args[i] == "--domain" and i + 1 < len(args):
                domains = [d.strip() for d in args[i + 1].split(",") if d.strip()] or None
                i += 2
            elif args[i] == "--force-add":
                force_add = True
                i += 1
            elif args[i].startswith("--"):
                i += 2  # 未知のフラグはスキップ
            elif not category_set:
                category = args[i]  # 後方互換: 位置引数（"fact" を明示的に渡しても誤爆しない）
                category_set = True
                i += 1
            elif source is None:
                source = args[i]  # 後方互換: 位置引数
                i += 1
            else:
                i += 1
        add_memory(content, category, source, flashbulb=flashbulb, origin=origin, domains=domains, force=force_add)

    elif cmd == "search":
        if len(sys.argv) < 3:
            print("使い方: python memory.py search \"検索語\" [--raw] [--memory] [--fuzzy] [--json] "
                  "[--domain D1,D2] [--session SID] [--topic SLUG] [--status solved|unsolved|ongoing|abandoned]")
            print("  デフォルト: catalog find (整理された表層を引く)")
            print("  --raw: raw_turn 検索 (生の発話)")
            print("  --memory: memory 層検索 (admin/debug)")
            return
        query = sys.argv[2]
        use_like = "--like" in sys.argv
        raw_mode = "--raw" in sys.argv  # raw_turn 検索（drill-down）
        memory_mode = "--memory" in sys.argv  # 旧 memory 層検索（admin）
        fuzzy_mode = "--fuzzy" in sys.argv
        json_mode = "--json" in sys.argv
        requested_domains = _parse_domain_flag(sys.argv)

        # スコープフラグ取得（session / topic / status）
        def _flag_value(flag):
            if flag in sys.argv:
                i = sys.argv.index(flag)
                if i + 1 < len(sys.argv):
                    return sys.argv[i + 1]
            return None
        scope_session = _flag_value("--session")
        scope_topic = _flag_value("--topic")
        scope_status = _flag_value("--status")

        # ルーティング:
        # 1. --raw or scope → search_in_scope (raw_turn)
        # 2. --memory → search_memories (admin / 旧経路)
        # 3. デフォルト → catalog find (整理された表層)

        if scope_session or scope_topic or scope_status or raw_mode:
            scope_results = search_in_scope(
                query,
                session_id=scope_session, topic=scope_topic, status=scope_status,
                limit=20
            )
            mode_str = "raw" if raw_mode and not (scope_session or scope_topic or scope_status) else "scope"
            log_recall(
                [], mode=mode_str, query=query,
                accessed_keys=[r.get("raw_turn_id") for r in scope_results if r.get("raw_turn_id") is not None],
            )
            if json_mode:
                meta = {"count": len(scope_results), "mode": mode_str,
                        "session": scope_session, "topic": scope_topic, "status": scope_status}
                print(json.dumps(
                    _json_envelope("search", scope_results, query=query, meta=meta),
                    ensure_ascii=False, indent=2
                ))
            elif scope_results:
                scope_str = " / ".join(f"{k}={v}" for k, v in
                    [("session", scope_session), ("topic", scope_topic), ("status", scope_status)] if v)
                if not scope_str:
                    scope_str = "raw"
                print(f"想起 ({len(scope_results)}件, スコープ: {scope_str}):")
                for r in scope_results:
                    ts = (r.get("timestamp") or "")[:16]
                    title = r.get("session_title") or ""
                    st = r.get("status") or ""
                    # snippet があればヒット位置中心の抜粋、なければ先頭切り詰め
                    sn = r.get("snippet")
                    if sn:
                        excerpt = sn.replace("\n", " ")
                    else:
                        excerpt = (r.get("content") or "")[:120].replace("\n", " ")
                    print(f"  raw:{r['raw_turn_id']} [{ts}] [{r['role']}] [{st}] "
                          f"session: {title}")
                    print(f"    「{excerpt}」")
            else:
                print("想起できませんでした（該当 raw_turn なし）")
            return

        if not memory_mode:
            # デフォルト: catalog find（整理された表層を引く。
            # memory.content は地下に置いて Claude には表に出さない規律）
            conn = get_connection()
            try:
                rows, source = catalog_find_cards(conn, query, limit=20)
                _ca_keys = [r["key"] for r in rows]
                _ca_mids = []
                for r in rows:
                    try:
                        _rids = json.loads(r["related_ids"]) if r["related_ids"] else []
                        _ca_mids.extend([x for x in _rids if isinstance(x, int)])
                    except (json.JSONDecodeError, TypeError, IndexError, KeyError):
                        pass
                log_recall(
                    sorted(set(_ca_mids)) if _ca_mids else [],
                    mode='catalog', source=source, query=query,
                    accessed_keys=_ca_keys,
                )
                if json_mode:
                    cards = []
                    for r in rows:
                        card = {"entry_type": r["entry_type"], "key": r["key"],
                                "title": r["title"],
                                "content": (r["content"] or "")[:200],
                                "_source": source}
                        try:
                            sn = r["snippet"]
                            if sn:
                                card["snippet"] = sn
                        except (IndexError, KeyError):
                            pass
                        cards.append(card)
                    meta = {"count": len(cards), "mode": "catalog", "source": source}
                    print(json.dumps(
                        _json_envelope("search", cards, query=query, meta=meta),
                        ensure_ascii=False, indent=2
                    ))
                else:
                    if rows:
                        print(f"想起 ({len(rows)}件, catalog, via {source}):")
                        for r in rows:
                            print(f"  [{r['entry_type']}/{r['key']}] {r['title']}")
                            try:
                                sn = r["snippet"]
                                if sn:
                                    print(f"      …{sn.replace(chr(10), ' ')}")
                            except (IndexError, KeyError):
                                pass
                    else:
                        print("想起できませんでした（catalog にヒットなし）")
                        print("  ヒント: --raw で raw_turn 検索、--memory で memory 層検索")
            finally:
                conn.close()
            return

        # --memory: 旧 memory 層検索（admin / debug）
        search_result = search_memories(query, use_like=use_like, fuzzy=fuzzy_mode, requested_domains=requested_domains)
        if fuzzy_mode:
            results, fuzzy_results = search_result
        else:
            results = search_result
            fuzzy_results = []
        _mem_ids = [row["id"] for row, _sim in results if row and "id" in row.keys()]
        log_recall(
            _mem_ids, mode='memory', query=query,
            source=("LIKE" if use_like else ("fuzzy" if fuzzy_mode else "brain")),
        )
        if json_mode:
            # v30 JSON 出力
            conn = get_connection()
            json_results = []
            for row, sim in results:
                h = format_handle(conn, row)
                json_results.append({"handle": h, "similarity": sim})
            fuzzy_out = []
            for row, sim in fuzzy_results:
                fuzzy_out.append({"handle": format_handle(conn, row, minimal=True), "similarity": sim})
            conn.close()
            meta = {"count": len(json_results), "mode": "LIKE" if use_like else "brain"}
            if fuzzy_out:
                meta["fuzzy"] = fuzzy_out
            print(json.dumps(
                _json_envelope("search", json_results, query=sys.argv[2], meta=meta),
                ensure_ascii=False, indent=2
            ))
        elif results:
            mode = "LIKE" if use_like else "脳"
            if raw_mode:
                print(f"想起 ({len(results)}件, {mode}検索):")
                for row, sim in results:
                    print(format_memory(row, similarity=sim))
            else:
                conn = get_connection()
                print(f"想起 ({len(results)}件, {mode}検索, 再構成モード):")
                for row, sim in results:
                    print(format_memory_reconstructive(conn, row, similarity=sim))
                conn.close()
        else:
            print("想起できませんでした")
        if not json_mode:
            # 舌先現象: もやもや記憶を表示
            if fuzzy_results:
                print(f"  舌先現象 ({len(fuzzy_results)}件):")
                for row, sim in fuzzy_results:
                    keywords = json.loads(row["keywords"]) if row["keywords"] else []
                    kw_str = ", ".join(keywords[:6])
                    print(f"  ?? #{row['id']} もやもや: [{kw_str}]")

            # 反芻検出: 同じ記憶ばかり触ってたら警告（--rumination フラグ時のみ）
            if "--rumination" in sys.argv:
                rum_conn = get_connection()
                ruminating = detect_rumination(rum_conn)
                if ruminating:
                    ids_str = ", ".join(f"#{r[0]}[{', '.join(r[2])}]" for r in ruminating)
                    print(f"  🔄 同じところを回ってる: {ids_str}")
                    print(f"     → recall --voices で別の視点を試してみて")
                rum_conn.close()

    elif cmd == "chain":
        if len(sys.argv) < 3:
            print("使い方: python memory.py chain ID [depth] [--json]")
            return
        mid = int(sys.argv[2])
        # depth 位置引数（--json とぶつからないように isdigit で抽出）
        depth = 2
        for a in sys.argv[3:]:
            if a.isdigit():
                depth = int(a)
                break
        chain = chain_memories(mid, depth)
        if "--json" in sys.argv:
            conn = get_connection()
            results = [{"handle": format_handle(conn, r), "depth": d} for r, d in chain]
            conn.close()
            print(json.dumps(
                _json_envelope("chain", results, query=str(mid), meta={"depth": depth}),
                ensure_ascii=False, indent=2
            ))
        elif chain:
            print(f"連想の連鎖 (#{mid} から深さ{depth}):")
            for row, d in chain:
                indent = "  " + "→ " * d
                print(f"{indent}#{row['id']} {row['content'][:60]}")
        else:
            print(f"記憶 #{mid} が見つかりません")

    elif cmd == "detail":
        if len(sys.argv) < 3:
            print("使い方: python memory.py detail ID [--json] [--memory]")
            print("  デフォルト: メタデータ + linked raw_turn ids")
            print("  --memory: memory.content も表示（admin/debug）")
            return
        mid = int(sys.argv[2])
        admin_mode = "--memory" in sys.argv
        conn = get_connection()
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone()
        if "--json" in sys.argv:
            if row:
                h = format_handle(conn, row, full=True)
                # admin でない場合は content を抜く
                if not admin_mode and isinstance(h, dict) and "content" in h:
                    h = {k: v for k, v in h.items() if k != "content"}
                print(json.dumps(
                    _json_envelope("detail", [{"handle": h}], query=sys.argv[2]),
                    ensure_ascii=False, indent=2
                ))
            else:
                print(json.dumps(_json_envelope("detail", [], query=sys.argv[2],
                                                meta={"error": "not found"}),
                                 ensure_ascii=False))
        elif row:
            if admin_mode:
                print(format_memory_detail(row))
            else:
                # memory.content は地下に置く規律。メタデータと linked raw_turns だけ表に出す
                emotions = json.loads(row["emotions"]) if row["emotions"] else []
                keywords = json.loads(row["keywords"]) if row["keywords"] else []
                print(f"  記憶 #{row['id']} (memory; --memory で生 content 表示)")
                print(f"  カテゴリ: {row['category']} | 重要度: {'★' * row['importance']}")
                print(f"  情動: {', '.join(emotions) if emotions else '中立'} (覚醒度:{row['arousal']:.2f})")
                print(f"  断片: [{', '.join(keywords[:8])}]")
                print(f"  鮮度: {freshness(row['created_at']):.0%} | 参照: {row['access_count']}回")
                print(f"  記録: {row['created_at'][:10]}")
            # 連想リンク（content は出さない、ID と strength のみ）
            links = conn.execute(
                """SELECT l.target_id, l.strength, l.link_type
                   FROM links l JOIN memories m ON l.target_id = m.id
                   WHERE l.source_id = ? AND m.forgotten = 0
                   ORDER BY l.strength DESC LIMIT 10""",
                (mid,)
            ).fetchall()
            if links:
                print(f"  連想リンク ({len(links)}件):")
                for link in links:
                    lt = link["link_type"] or "association"
                    mark = "⇄" if lt == "tension" else "→"
                    print(f"    {mark} #{link['target_id']} ({link['strength']:.3f}) [{lt}]")
            # この memory に linked back している raw_turn ids（drill-down 経路）
            raw_rows = conn.execute(
                "SELECT id FROM raw_turns WHERE memory_ids LIKE ? LIMIT 10",
                (f"%{mid}%",)
            ).fetchall()
            raw_ids = []
            for r in raw_rows:
                # JSON parse で厳密 filter
                turn_row = conn.execute(
                    "SELECT memory_ids FROM raw_turns WHERE id = ?", (r["id"],)
                ).fetchone()
                try:
                    mids = json.loads(turn_row["memory_ids"]) if turn_row and turn_row["memory_ids"] else []
                except (json.JSONDecodeError, TypeError):
                    mids = []
                if mid in mids:
                    raw_ids.append(r["id"])
            if raw_ids:
                ids_str = ", ".join(f"#{r}" for r in raw_ids)
                print(f"  linked raw_turns: {ids_str}")
                print(f"     drill down: search \"...\" --raw  または手動で raw_turn を取得")
        else:
            print("見つかりません")
        conn.close()

    elif cmd == "recent":
        # 数値位置引数のみ拾い（--json 等が混ざっても OK）
        n = 10
        for a in sys.argv[2:]:
            if a.isdigit():
                n = int(a)
                break
        rows = get_recent(n)
        if "--json" in sys.argv:
            conn = get_connection()
            results = [{"handle": format_handle(conn, r)} for r in rows]
            conn.close()
            print(json.dumps(_json_envelope("recent", results), ensure_ascii=False, indent=2))
        else:
            print(f"最近の記憶 ({len(rows)}件):")
            for row in rows:
                print(format_memory(row))

    elif cmd == "all":
        rows = get_all()
        if "--json" in sys.argv:
            conn = get_connection()
            results = [{"handle": format_handle(conn, r, minimal=True)} for r in rows]
            conn.close()
            print(json.dumps(_json_envelope("all", results), ensure_ascii=False, indent=2))
        else:
            print(f"全記憶 ({len(rows)}件):")
            for row in rows:
                print(format_memory(row))

    elif cmd == "forget":
        if len(sys.argv) < 3:
            print("使い方: python memory.py forget ID")
            return
        forget_memory(int(sys.argv[2]))

    elif cmd == "resurrect":
        if len(sys.argv) < 3:
            print("使い方: python memory.py resurrect \"検索語\"")
            return
        resurrect_memories(sys.argv[2])

    elif cmd == "delusion":
        # delusionモード: 完全記憶検索
        args = sys.argv[2:]
        query = None
        date_val = None
        after_val = None
        before_val = None
        raw_only = "--raw" in args
        dump_all = "--all" in args
        batch_mode = "--batch" in args
        batch_context = "--batch-context" in args
        context_id = None
        limit_val = 50
        queries = []

        # フラグ以外の引数を解析
        i = 0
        while i < len(args):
            if args[i] == "--date" and i + 1 < len(args):
                date_val = args[i + 1]
                i += 2
            elif args[i] == "--after" and i + 1 < len(args):
                after_val = args[i + 1]
                i += 2
            elif args[i] == "--before" and i + 1 < len(args):
                before_val = args[i + 1]
                i += 2
            elif args[i] == "--context" and i + 1 < len(args):
                ctx_val = args[i + 1]
                context_id = ctx_val if ctx_val.startswith("raw:") else int(ctx_val)
                i += 2
            elif args[i] == "--limit" and i + 1 < len(args):
                limit_val = int(args[i + 1])
                i += 2
            elif args[i] in ("--raw", "--all", "--batch", "--batch-context"):
                i += 1
            else:
                queries.append(args[i])
                i += 1

        query = queries[0] if queries else None

        if not query and not date_val and not after_val and not before_val and not dump_all and context_id is None and not batch_context:
            print("使い方:")
            print('  python memory.py delusion "検索語"                  # 純粋ベクトル検索')
            print('  python memory.py delusion "検索語" --date 2024-12-11  # 日付フィルタ')
            print('  python memory.py delusion "検索語" --after 2024-11-01 --before 2025-02-01')
            print('  python memory.py delusion --date 2024-12-11        # その日の全記憶')
            print('  python memory.py delusion --all                    # 全記憶ダンプ')
            print('  python memory.py delusion --raw "検索語"            # 原文のみ検索')
            print('  python memory.py delusion --context 123            # 記憶の対話文脈')
            print('  python memory.py delusion --context raw:4728       # raw_turnの前後文脈')
            print('  python memory.py delusion --batch "q1" "q2" "q3"  # バッチ検索')
            print('  python memory.py delusion --batch-context 36 raw:4728 337  # 複数ID一括文脈取得')
            return

        if batch_context and queries:
            # バッチコンテキスト: 複数IDの文脈を1プロセスで一括取得
            for id_str in queries:
                cid = id_str if id_str.startswith("raw:") else int(id_str)
                print(f"=== context {id_str} ===")
                results = _delusion_context(get_connection(), cid)
                if results:
                    for item, sim in results:
                        print(format_delusion(item, sim))
                        print()
                else:
                    print("���当する記憶はありません")
                print()
            return

        if batch_mode and len(queries) > 1:
            # バッチモード: 複数クエリを1プロセスで処理
            seen_ids = set()
            for q in queries:
                results = delusion_search(
                    query=q, limit=limit_val, date=date_val,
                    after=after_val, before=before_val,
                    raw_only=raw_only, context_id=None, dump_all=False
                )
                new_results = []
                for item, sim in results:
                    item_id = item.get("id")
                    if item_id and item_id not in seen_ids:
                        seen_ids.add(item_id)
                        new_results.append((item, sim))
                if new_results:
                    print(f"--- {q} ({len(new_results)}件) ---")
                    for item, sim in new_results:
                        print(format_delusion(item, sim))
                        print()
                else:
                    print(f"--- {q} (0件) ---")
        else:
            results = delusion_search(
                query=query, limit=limit_val, date=date_val,
                after=after_val, before=before_val,
                raw_only=raw_only, context_id=context_id, dump_all=dump_all
            )

            if results:
                mode_str = "delusion"
                if raw_only:
                    mode_str += " (raw)"
                if context_id is not None:
                    mode_str += f" (context #{context_id})"
                print(f"完全記憶 ({len(results)}件, {mode_str}):")
                for item, sim in results:
                    print(format_delusion(item, sim))
                    print()
            else:
                print("該当する記憶はありません")

    elif cmd == "recall" and "--brain-cache" in sys.argv:
        # キャッシュから分離脳の解釈を読む（/dive用。高速）
        cache_path = Path(__file__).parent / ".brain_cache.json"
        if not cache_path.exists():
            print("(brain cache なし。通常recallにフォールバック)")
            results = recall_important(10)
            print(f"自動想起 ({len(results)}件):")
            for row, score in results:
                print(format_memory_simple(row))
            _recalled_ids = [row["id"] for row, _ in results]
            if _recalled_ids:
                log_recall(_recalled_ids)
        else:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            updated = cache.get("updated_at", "不明")
            if "left" in cache:
                left = cache["left"]
                print(f"🧠 左脳 ({left['name']}, {updated}):")
                print(left["interpretation"])
                print()
            if "right" in cache:
                right = cache["right"]
                print(f"🧠 右脳 ({right['name']}, {updated}):")
                print(right["interpretation"])

    elif cmd == "neighbors":
        if len(sys.argv) < 3:
            print("使い方: python memory.py neighbors ID [--limit N] [--json]")
            return
        mid = int(sys.argv[2])
        limit = 10
        for i, a in enumerate(sys.argv):
            if a == "--limit" and i + 1 < len(sys.argv):
                try:
                    limit = int(sys.argv[i + 1])
                except ValueError:
                    pass
        results_raw = get_neighbors(mid, limit=limit)
        if "--json" in sys.argv:
            conn = get_connection()
            results = [{"handle": format_handle(conn, r), "strength": round(s, 3)}
                       for r, s in results_raw]
            conn.close()
            print(json.dumps(
                _json_envelope("neighbors", results, query=str(mid)),
                ensure_ascii=False, indent=2
            ))
        else:
            print(f"#{mid} の隣接 ({len(results_raw)}件):")
            for r, s in results_raw:
                emos = json.loads(r["emotions"]) if r["emotions"] else []
                kws = json.loads(r["keywords"]) if r["keywords"] else []
                emo_str = " ".join(EMOTION_EMOJI.get(e, "·") for e in emos[:3]) if emos else "·"
                kw_str = ", ".join(kws[:5])
                print(f"  → #{r['id']} ({s:.3f}) {emo_str} [{kw_str}]")

    elif cmd == "walk":
        if len(sys.argv) < 3:
            print("使い方: python memory.py walk ID [--depth N] [--json]")
            return
        mid = int(sys.argv[2])
        depth = 2
        for i, a in enumerate(sys.argv):
            if a == "--depth" and i + 1 < len(sys.argv):
                try:
                    depth = int(sys.argv[i + 1])
                except ValueError:
                    pass
        nodes = walk_from(mid, depth=depth)
        if "--json" in sys.argv:
            conn = get_connection()
            results = [{"handle": format_handle(conn, r), "distance": d} for r, d in nodes]
            conn.close()
            print(json.dumps(
                _json_envelope("walk", results, query=str(mid), meta={"depth": depth}),
                ensure_ascii=False, indent=2
            ))
        else:
            print(f"#{mid} から {depth} 歩先まで ({len(nodes)}件):")
            for r, d in nodes:
                indent = "  " + "  " * d
                emos = json.loads(r["emotions"]) if r["emotions"] else []
                kws = json.loads(r["keywords"]) if r["keywords"] else []
                emo_str = " ".join(EMOTION_EMOJI.get(e, "·") for e in emos[:3]) if emos else "·"
                kw_str = ", ".join(kws[:5])
                print(f"{indent}[{d}] #{r['id']} {emo_str} [{kw_str}]")

    elif cmd == "at":
        if len(sys.argv) < 3:
            print("使い方: python memory.py at DOMAIN [--limit N] [--json]")
            return
        domain = sys.argv[2]
        limit = 15
        for i, a in enumerate(sys.argv):
            if a == "--limit" and i + 1 < len(sys.argv):
                try:
                    limit = int(sys.argv[i + 1])
                except ValueError:
                    pass
        rows = nodes_at_domain(domain, limit=limit)
        if "--json" in sys.argv:
            conn = get_connection()
            results = [{"handle": format_handle(conn, r)} for r in rows]
            conn.close()
            print(json.dumps(
                _json_envelope("at", results, query=domain),
                ensure_ascii=False, indent=2
            ))
        else:
            print(f"domain={domain} のノード ({len(rows)}件):")
            for r in rows:
                stars = "★" * (r["importance"] or 0)
                emos = json.loads(r["emotions"]) if r["emotions"] else []
                kws = json.loads(r["keywords"]) if r["keywords"] else []
                emo_str = " ".join(EMOTION_EMOJI.get(e, "·") for e in emos[:3]) if emos else "·"
                kw_str = ", ".join(kws[:5])
                print(f"  #{r['id']} {stars} {emo_str} [{kw_str}]")

    elif cmd == "anchor":
        # v30 identity anchor: dive 時の最小注入用
        as_json = "--json" in sys.argv
        anchors = _load_anchors()
        if as_json:
            conn = get_connection()
            results = [{"handle": format_handle(conn, r, full=False, include_links=False)}
                       for r in anchors]
            conn.close()
            print(json.dumps(_json_envelope("anchor", results), ensure_ascii=False, indent=2))
        else:
            print(f"identity anchor ({len(anchors)}件):")
            for r in anchors:
                tag = []
                if r["flashbulb"]:
                    tag.append("💎")
                if r["category"] == "schema":
                    tag.append("[スキーマ]")
                if r["importance"] == 5:
                    tag.append("★★★★★")
                tag_str = " ".join(tag)
                content = (r["content"] or "")[:100]
                print(f"  #{r['id']} {tag_str} {content}")

    elif cmd == "recall" and "--fast" in sys.argv:
        # 高速想起: SessionStart hook 用。埋め込みモデル・DMN・voices を通らず、
        # 鮮度×重要度×access の軽量スコアだけで返す。読み取り専用なので
        # access_count 更新も再固定化もしない（深い想起は通常 recall で行う）
        n = 3
        for a in sys.argv[2:]:
            if a.isdigit():
                n = int(a)
                break
        conn = get_connection()
        rows = conn.execute(
            "SELECT id, content, category, importance, created_at, access_count "
            "FROM memories WHERE forgotten = 0 "
            "ORDER BY created_at DESC LIMIT 200"
        ).fetchall()
        conn.close()
        scored = []
        for row in rows:
            w = freshness(row["created_at"]) * row["importance"] * (1.0 + math.log1p(row["access_count"]))
            scored.append((row, w))
        scored.sort(key=lambda x: x[1], reverse=True)
        print(f"高速想起 ({min(n, len(scored))}件 / 直近200件を鮮度×重要度×accessで選抜):")
        for row, _ in scored[:n]:
            head = row["content"].split("\n")[0][:120]
            print(f"  #{row['id']} [{row['category']}] {row['created_at'][:10]} {head}")

    elif cmd == "recall":
        # v30: default は pull 指向の simple list のみ。内面系は全部 opt-in。

        # --meta の時のみ事後検証プリント
        if "--meta" in sys.argv:
            prev_eval = evaluate_last_recall()
            if prev_eval:
                p = prev_eval["precision"]
                r = prev_eval["recall_rate"]
                print(f"(前回recall検証: 精度{p:.0%} 網羅{r:.0%} "
                      f"空振{len(prev_eval['noise'])} 漏れ{len(prev_eval['missed'])})")

        # --dmn の時のみ DMN 発火
        if "--dmn" in sys.argv:
            dmn_conn = get_connection()
            gap = _get_session_gap(dmn_conn)
            if gap is not None and gap >= 1.0:
                dmn_results = default_mode_network(dmn_conn, gap)
                if dmn_results:
                    if gap >= 24:
                        gap_str = f"{gap/24:.0f}日"
                    else:
                        gap_str = f"{gap:.0f}時間"
                    print(f"💭 DMN（{gap_str}ぶり — ぼーっとしてたら浮かんできた）:")
                    for a, b, path in dmn_results:
                        kw_a = json.loads(a["keywords"])[:3] if a["keywords"] else []
                        kw_b = json.loads(b["keywords"])[:3] if b["keywords"] else []
                        print(f"  #{a['id']} [{', '.join(kw_a)}] ···{path}··· #{b['id']} [{', '.join(kw_b)}]")
                    print()
            dmn_conn.close()

        # --insights の時のみ
        if "--insights" in sys.argv:
            _show_recent_insights()

        if "--voices" in sys.argv:
            # 内的対話モード
            n = 3
            for arg in sys.argv[2:]:
                if arg.isdigit():
                    n = int(arg)
            requested_domains = _parse_domain_flag(sys.argv)
            voices = recall_polyphonic(limit_per_voice=n, requested_domains=requested_domains)
            raw_mode = "--raw" in sys.argv
            fragments_only = "--fragments" in sys.argv
            conn = get_connection()

            # --with-mood の時のみ気分を表示
            if "--with-mood" in sys.argv:
                mood = _effective_mood(conn)
                if mood and mood.get("emotions"):
                    explicit = load_mood()
                    if explicit and explicit.get("emotions"):
                        mood_src = "明示"
                    else:
                        mood_src = "暗黙（最近触った記憶から推定）"
                    emo_str = ", ".join(mood["emotions"])
                    print(f"気分: {emo_str} (覚醒度:{mood['arousal']:.2f}) [{mood_src}]")
                    print(f"  → 共感は{emo_str}系を出す / 補完はそれ以外を出す")
                else:
                    print("気分: なし（共感と補完の差は出にくい）")

            voice_labels = {
                "共感": "🤝",
                "補完": "🔭",
                "批判": "⚡",
                "連想": "🎲",
                "俯瞰": "🦅",
            }
            _recall_mode = "simple"  # デフォルト: 内容のみ。--raw/--fullで上書き
            for voice_name, results in voices.items():
                if not results:
                    continue
                emoji = voice_labels.get(voice_name, "·")
                print(f"\n{emoji} [{voice_name}]")
                if voice_name == "俯瞰":
                    # 俯瞰はテキスト断片のリスト
                    for text, _imp in results:
                        print(f"  {text}")
                elif _recall_mode == "simple":
                    for row, score in results:
                        print(format_memory_simple(row))
                else:
                    for row, score in results:
                        if raw_mode:
                            print(format_memory(row, score=score))
                        else:
                            print(format_memory_reconstructive(conn, row, score=score, fragments_only=fragments_only))
            conn.close()
        else:
            full_mode = "--full" in sys.argv
            raw_mode = "--raw" in sys.argv
            fragments_only = "--fragments" in sys.argv
            json_mode = "--json" in sys.argv
            # v30: --distill の時のみ蒸留。default は simple list
            want_distill = "--distill" in sys.argv and not (raw_mode or full_mode or json_mode)
            default_limit = 15 if full_mode else 10
            limit = default_limit
            for arg in sys.argv[2:]:
                if arg.isdigit():
                    limit = int(arg)
                    break
            # 左脳/右脳バランス
            if "--analytical" in sys.argv:
                balance = 0.3
            elif "--emotional" in sys.argv:
                balance = 0.7
            else:
                balance = 0.5
            requested_domains = _parse_domain_flag(sys.argv)
            results = recall_important(limit, balance=balance, requested_domains=requested_domains)

            if json_mode:
                conn = get_connection()
                json_results = [{"handle": format_handle(conn, r), "score": s}
                                for r, s in results]
                conn.close()
                print(json.dumps(
                    _json_envelope("recall", json_results, meta={"limit": limit, "balance": balance}),
                    ensure_ascii=False, indent=2
                ))
            elif want_distill:
                state = _distill_state(results)
                if state:
                    print(state)
                else:
                    print(f"自動想起 ({len(results)}件):")
                    for row, score in results:
                        print(format_memory_simple(row))
            elif raw_mode:
                print(f"自動想起 ({len(results)}件):")
                for row, score in results:
                    print(format_memory(row, score=score))
            elif full_mode:
                conn = get_connection()
                print(f"自動想起 ({len(results)}件, 再構成モード):")
                for row, score in results:
                    print(format_memory_reconstructive(conn, row, score=score, fragments_only=fragments_only))
                conn.close()
            else:
                print(f"自動想起 ({len(results)}件):")
                for row, score in results:
                    print(format_memory_simple(row))

        # メタ認知: 今回recallしたIDを記録（声の帰属つき）
        _recalled_ids = []
        _voice_attribution = {}
        if "--voices" in sys.argv or auto_voices:
            for voice_name, vresults in voices.items():
                if voice_name == "俯瞰":
                    continue
                voice_ids = []
                for row, _ in vresults:
                    _recalled_ids.append(row["id"])
                    voice_ids.append(row["id"])
                if voice_ids:
                    _voice_attribution[voice_name] = voice_ids
        else:
            for row, _ in results:
                _recalled_ids.append(row["id"])
        if _recalled_ids:
            log_recall(list(set(_recalled_ids)),
                       voice_attribution=_voice_attribution if _voice_attribution else None)

    elif cmd == "review":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        review_memories(n)

    elif cmd == "promote":
        promote_turns()

    elif cmd == "replay":
        replay_memories()

    elif cmd == "nap":
        nap()

    elif cmd == "consolidate":
        dry = "--dry-run" in sys.argv
        consolidate_memories(dry_run=dry)

    elif cmd == "tension":
        _tension_cli(sys.argv[2:])

    elif cmd == "detect-tensions":
        # sleep に組み込みやすい単独コマンド（detect_tensions の薄いエイリアス）
        dry = "--dry-run" in sys.argv
        detect_tensions(dry_run=dry)

    elif cmd == "schema":
        dry = "--dry-run" in sys.argv
        build_schemas(dry_run=dry)

    elif cmd == "proceduralize":
        dry = "--dry-run" in sys.argv
        proceduralize(dry_run=dry)

    elif cmd == "domain":
        _domain_cli(sys.argv[2:])

    elif cmd == "catalog":
        _catalog_cli(sys.argv[2:])

    elif cmd == "voice":
        _voice_cli(sys.argv[2:])

    elif cmd == "feelings":
        _feelings_cli(sys.argv[2:])

    elif cmd == "extract-feelings":
        _feelings_cli(["extract"] + sys.argv[2:])

    elif cmd == "extract-mentions":
        limit = 100
        batch = 10
        dry = "--dry-run" in sys.argv
        json_mode = "--json" in sys.argv
        for i, a in enumerate(sys.argv[2:], start=2):
            if a == "--limit" and i + 1 < len(sys.argv):
                try:
                    limit = int(sys.argv[i + 1])
                except ValueError:
                    pass
            elif a == "--batch" and i + 1 < len(sys.argv):
                try:
                    batch = int(sys.argv[i + 1])
                except ValueError:
                    pass
        stats = extract_mentions(limit=limit, batch=batch, dry_run=dry)
        if json_mode:
            print(json.dumps(_json_envelope("extract-mentions", [stats],
                                            meta={"limit": limit, "batch": batch,
                                                  "dry_run": dry}),
                             ensure_ascii=False, indent=2))
        else:
            mode = "dry-run" if dry else "live"
            print(f"extract-mentions ({mode}): {stats}")

    elif cmd == "repair-links":
        # memory → raw_turn の FTS 補修バッチ
        limit = 500
        window = 1
        json_mode = "--json" in sys.argv
        for i, a in enumerate(sys.argv[2:], start=2):
            if a == "--limit" and i + 1 < len(sys.argv):
                try:
                    limit = int(sys.argv[i + 1])
                except ValueError:
                    pass
            elif a == "--window-days" and i + 1 < len(sys.argv):
                try:
                    window = int(sys.argv[i + 1])
                except ValueError:
                    pass
        stats = repair_memory_raw_links(limit=limit, window_days=window)
        if json_mode:
            print(json.dumps(_json_envelope("repair-links", [stats],
                                            meta={"limit": limit, "window_days": window}),
                             ensure_ascii=False, indent=2))
        else:
            print(f"repair-links (window={window} days, limit={limit}): {stats}")

    elif cmd == "overview":
        overview()

    elif cmd == "stats":
        s = get_stats()
        print(f"記憶の統計:")
        print(f"  有効: {s['total']}件 / 忘却: {s['forgotten']}件 / リンク: {s['links']}件 / 統合: {s['merged']}件")
        print(f"  ベクトル化: {s['with_embedding']}件")
        if s["by_category"]:
            print(f"  カテゴリ別:")
            for row in s["by_category"]:
                print(f"    {row['category']}: {row['cnt']}件")
        if s["by_emotion"]:
            print(f"  情動別:")
            for emo, cnt in sorted(s["by_emotion"].items(), key=lambda x: -x[1]):
                emoji = EMOTION_EMOJI.get(emo, "·")
                print(f"    {emoji} {emo}: {cnt}件")
        if s["most_accessed"]:
            print(f"  よく想起する記憶:")
            for row in s["most_accessed"]:
                print(f"    #{row['id']} ({row['access_count']}回) {row['content'][:50]}")
        if s["most_linked"]:
            print(f"  最もつながりの多い記憶:")
            for row in s["most_linked"]:
                print(f"    #{row['id']} ({row['link_count']}リンク) {row['content'][:50]}")
        if s.get("by_location"):
            print(f"  場所別:")
            for loc, cnt in sorted(s["by_location"].items(), key=lambda x: -x[1]):
                print(f"    📍 {loc}: {cnt}件")
        print(f"  現在の場所: {_detect_location()}")

    elif cmd == "mood":
        if len(sys.argv) < 3:
            mood = load_mood()
            if mood:
                emos = ", ".join(mood.get("emotions", []))
                print(f"現在の気分: {emos} (覚醒度: {mood.get('arousal', 0.5):.1f})")
            else:
                print("現在の気分: 中立")
        elif sys.argv[2] == "clear":
            clear_mood()
            print("✓ 気分状態をクリアしました")
        else:
            emotion = sys.argv[2]
            arousal = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
            arousal = max(0.0, min(1.0, arousal))
            save_mood([emotion], arousal)
            print(f"✓ 気分を設定: {emotion} (覚醒度: {arousal:.1f})")

    elif cmd == "prospect":
        if len(sys.argv) < 3:
            print("使い方:")
            print("  python memory.py prospect add \"トリガー\" \"アクション\"")
            print("  python memory.py prospect list")
            print("  python memory.py prospect clear ID")
            return
        subcmd = sys.argv[2]
        if subcmd == "add":
            if len(sys.argv) < 5:
                print("使い方: python memory.py prospect add \"トリガー\" \"アクション\"")
                return
            prospect_add(sys.argv[3], sys.argv[4])
        elif subcmd == "list":
            prospect_list()
        elif subcmd == "clear":
            if len(sys.argv) < 4:
                print("使い方: python memory.py prospect clear ID")
                return
            prospect_clear(int(sys.argv[3]))
        else:
            print(f"不明なサブコマンド: {subcmd}")

    elif cmd == "export":
        filename = sys.argv[2] if len(sys.argv) > 2 else None
        export_memories(filename)

    elif cmd == "import":
        if len(sys.argv) < 3:
            print("使い方: python memory.py import filename")
            return
        import_memories(sys.argv[2])

    elif cmd == "sync":
        if len(sys.argv) < 3:
            print("使い方:")
            print("  python memory.py sync status <host:port>  # 接続確認")
            print("  python memory.py sync push <host:port>    # ローカル→リモート")
            print("  python memory.py sync pull <host:port>    # リモート→ローカル")
            print("  python memory.py sync serve [--port N] [--public] [--insecure]  # 同期サーバー起動")
            print("  python memory.py sync node-id             # この端末のID表示")
            return

        subcmd = sys.argv[2]
        token = os.environ.get("MEMORY_SYNC_TOKEN", "")

        if subcmd == "node-id":
            print(f"node_id: {_get_node_id()}")

        elif subcmd == "serve":
            # サーバー起動（memory_sync_server.pyに委譲）
            import subprocess
            args = [sys.executable, str(Path(__file__).parent / "memory_sync_server.py")]
            args.extend(sys.argv[3:])  # --port等を転送
            subprocess.run(args)

        elif subcmd in ("status", "push", "pull"):
            if len(sys.argv) < 4:
                print(f"使い方: python memory.py sync {subcmd} <host:port>")
                return
            import urllib.request
            import urllib.error

            host = sys.argv[3]
            if not host.startswith("http"):
                host = f"http://{host}"
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            if subcmd == "status":
                try:
                    req = urllib.request.Request(f"{host}/sync/node-id", headers=headers)
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        data = json.loads(resp.read())
                    print(f"✓ 接続OK")
                    print(f"  相手のnode_id: {data['node_id']}")
                    print(f"  自分のnode_id: {_get_node_id()}")
                    # 同期履歴
                    conn = get_connection()
                    meta = conn.execute("SELECT * FROM sync_meta WHERE peer_id = ?",
                                        (data['node_id'],)).fetchone()
                    if meta:
                        print(f"  前回同期: {meta['last_sync_at']} ({meta['last_sync_direction']})")
                    else:
                        print(f"  初回同期")
                    conn.close()
                except urllib.error.URLError as e:
                    print(f"✗ 接続失敗: {e}")

            elif subcmd == "push":
                # ローカルの変更を相手に送信
                conn = get_connection()
                # 相手のnode_idを取得
                try:
                    req = urllib.request.Request(f"{host}/sync/node-id", headers=headers)
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        peer_data = json.loads(resp.read())
                    peer_id = peer_data["node_id"]
                except urllib.error.URLError as e:
                    print(f"✗ 接続失敗: {e}")
                    conn.close()
                    return

                # 前回同期時刻
                meta = conn.execute("SELECT last_sync_at FROM sync_meta WHERE peer_id = ?",
                                    (peer_id,)).fetchone()
                since = meta["last_sync_at"] if meta else None
                conn.close()

                # エクスポート
                data = sync_export(since=since)
                payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
                print(f"送信: {len(data['memories'])}件の記憶, {len(data['links'])}件のリンク")

                # 送信
                try:
                    req = urllib.request.Request(
                        f"{host}/sync/push",
                        data=payload,
                        headers={**headers, "Content-Type": "application/json; charset=utf-8"},
                        method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        result = json.loads(resp.read())
                    s = result["stats"]
                    print(f"✓ push完了: 新規{s['new']}件, 更新{s['updated']}件, "
                          f"リンク新規{s['links_new']}件, 忘却{s['forgotten']}件")
                except urllib.error.URLError as e:
                    print(f"✗ push失敗: {e}")
                    return

                # 同期履歴を更新
                now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                conn = get_connection()
                conn.execute(
                    """INSERT OR REPLACE INTO sync_meta (peer_id, peer_url, last_sync_at, last_sync_direction)
                       VALUES (?, ?, ?, 'push')""",
                    (peer_id, host, now_utc)
                )
                conn.commit()
                conn.close()

            elif subcmd == "pull":
                # 相手の変更を取得してマージ
                try:
                    req = urllib.request.Request(f"{host}/sync/node-id", headers=headers)
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        peer_data = json.loads(resp.read())
                    peer_id = peer_data["node_id"]
                except urllib.error.URLError as e:
                    print(f"✗ 接続失敗: {e}")
                    return

                conn = get_connection()
                meta = conn.execute("SELECT last_sync_at FROM sync_meta WHERE peer_id = ?",
                                    (peer_id,)).fetchone()
                since = meta["last_sync_at"] if meta else None
                conn.close()

                # 取得
                url = f"{host}/sync/changes"
                if since:
                    url += f"?since={since}"
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = json.loads(resp.read())
                except urllib.error.URLError as e:
                    print(f"✗ pull失敗: {e}")
                    return

                print(f"受信: {len(data['memories'])}件の記憶, {len(data['links'])}件のリンク")

                # マージ
                stats = sync_import(data)
                print(f"✓ pull完了: 新規{stats['new']}件, 更新{stats['updated']}件, "
                      f"リンク新規{stats['links_new']}件, 忘却{stats['forgotten']}件")

                # 同期履歴を更新
                now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                conn = get_connection()
                conn.execute(
                    """INSERT OR REPLACE INTO sync_meta (peer_id, peer_url, last_sync_at, last_sync_direction)
                       VALUES (?, ?, ?, 'pull')""",
                    (peer_id, host, now_utc)
                )
                conn.commit()
                conn.close()

    elif cmd == "mutations":
        conn = get_connection()
        if len(sys.argv) > 2:
            # 特定記憶の全履歴
            mid = int(sys.argv[2])
            logs = conn.execute(
                "SELECT * FROM mutation_log WHERE memory_id = ? ORDER BY created_at DESC",
                (mid,)
            ).fetchall()
            if not logs:
                print(f"#{mid} の変異履歴はありません")
            else:
                print(f"#{mid} の変異履歴 ({len(logs)}件):")
                for log in logs:
                    print(f"  [{log['created_at']}] {log['field']}: {log['reason']}")
                    print(f"    旧: {log['old_value']}")
                    print(f"    新: {log['new_value']}")
        else:
            # 直近20件
            logs = conn.execute(
                "SELECT * FROM mutation_log ORDER BY created_at DESC LIMIT 20"
            ).fetchall()
            if not logs:
                print("変異履歴はありません")
            else:
                print(f"直近の変異履歴 ({len(logs)}件):")
                for log in logs:
                    print(f"  [{log['created_at']}] #{log['memory_id']} {log['field']}: {log['reason']}")
        conn.close()

    elif cmd == "correct":
        if len(sys.argv) < 4:
            print("使い方: python memory.py correct ID \"新しい内容\"")
            return
        correct_memory(int(sys.argv[2]), sys.argv[3])

    elif cmd == "versions":
        if len(sys.argv) < 3:
            print("使い方: python memory.py versions ID")
            return
        show_versions(int(sys.argv[2]))

    elif cmd == "memo":
        if len(sys.argv) < 3:
            print("使い方:")
            print("  python memory.py memo \"タイトル\" \"内容\"")
            print("  python memory.py memo list")
            print("  python memory.py memo index")
            return
        sub = sys.argv[2]
        if sub == "list":
            list_memos()
        elif sub == "index":
            index_memos()
        else:
            title = sys.argv[2]
            content = sys.argv[3] if len(sys.argv) > 3 else ""
            save_memo(title, content)

    elif cmd == "calibrate":
        calibrate_report()

    elif cmd == "self-tune":
        dry = "--dry-run" in sys.argv
        self_tune(dry_run=dry)

    elif cmd == "meta-memory":
        dry = "--dry-run" in sys.argv
        generate_meta_memories(dry_run=dry)

    elif cmd == "params":
        self_tune_report()

    elif cmd == "brain":
        # 左脳/右脳スコアの可視化
        side = None
        if "--left" in sys.argv:
            side = "left"
        elif "--right" in sys.argv:
            side = "right"
        limit = 10
        for arg in sys.argv[2:]:
            if arg.isdigit():
                limit = int(arg)
                break
        conn = get_connection()
        sweep_contexts(conn)
        rows = conn.execute("SELECT * FROM memories WHERE forgotten = 0").fetchall()

        scored = []
        for row in rows:
            L = _left_score(row)
            R = _right_score(conn, row)
            C = corpus_callosum(L, R)
            scored.append((row, L, R, C))

        if side == "left":
            scored.sort(key=lambda x: x[1], reverse=True)
            print(f"🧠 左脳ランキング (分析・意味) top {limit}:")
            for row, L, R, C in scored[:limit]:
                conf = row["confidence"] if "confidence" in row.keys() and row["confidence"] is not None else 0.7
                rev = row["revision_count"] if "revision_count" in row.keys() and row["revision_count"] is not None else 0
                fresh = freshness(row["created_at"])
                print(f"  #{row['id']} L:{L:.3f}  fresh:{fresh:.0%} conf:{conf:.2f} rev:{rev} access:{row['access_count']}")
                print(f"       {row['content'][:80]}")
        elif side == "right":
            scored.sort(key=lambda x: x[2], reverse=True)
            print(f"🧠 右脳ランキング (情動・直感) top {limit}:")
            for row, L, R, C in scored[:limit]:
                emotions = json.loads(row["emotions"]) if row["emotions"] else []
                emo_str = ", ".join(emotions) if emotions else "中立"
                fb = row["flashbulb"] if "flashbulb" in row.keys() else None
                fb_str = f" 🔥{fb[:30]}" if fb else ""
                print(f"  #{row['id']} R:{R:.3f}  arousal:{row['arousal']:.2f} emo:[{emo_str}]{fb_str}")
                print(f"       {row['content'][:80]}")
        else:
            scored.sort(key=lambda x: x[3], reverse=True)
            print(f"🧠 左脳 vs 右脳 top {limit}:")
            print(f"  {'ID':>6}  {'L(分析)':>8}  {'R(情動)':>8}  {'統合':>8}  内容")
            print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*20}")
            for row, L, R, C in scored[:limit]:
                content = row['content'][:50].replace('\n', ' ')
                print(f"  #{row['id']:>5}  {L:>8.3f}  {R:>8.3f}  {C:>8.3f}  {content}")

        conn.close()

    else:
        print(f"不明なコマンド: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
