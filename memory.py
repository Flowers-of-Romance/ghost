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
  python memory.py schema             # リンク密集クラスタからスキーマ（メタ記憶）を生成
  python memory.py proceduralize      # 反復された記憶を行動指針に昇格（LEARNED.mdに書込み）
  python memory.py stats
  python memory.py mood [emotion] [arousal]  # 気分状態の設定・表示
  python memory.py mood clear                # 気分状態をクリア
  python memory.py prospect add "trigger" "action"  # 予期記憶を登録
  python memory.py prospect list                     # 予期記憶一覧
  python memory.py prospect clear ID                 # 予期記憶を完了
  python memory.py export [filename]                 # 記憶をJSONファイルにエクスポート
  python memory.py import filename                   # JSONファイルから記憶をインポート
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
INTERFERENCE_THRESHOLD = 0.90

# 統合: この類似度を超える記憶ペアを統合候補とする
CONSOLIDATION_THRESHOLD = 0.94

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

# 手続き化: 反復が閾値を超えた記憶を行動指針に昇格
PROCEDURALIZE_ACCESS_THRESHOLD = 20   # 最低想起回数
PROCEDURALIZE_LINK_THRESHOLD = 15     # 最低リンク数
LEARNED_MD_PATH = str(Path(__file__).parent / "LEARNED.md")
HEBB_MARKER = "## 学習された行動指針"

# 外傷的記憶: arousalがこの閾値を超えると既存メカニズムの挙動が変わる
# 馴化しない、統合されない、減衰が遅い、夢に頻出する
TRAUMA_AROUSAL_THRESHOLD = 0.85

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
    """arousalに応じた半減期を返す。外傷的記憶は減衰が極端に遅い。"""
    if arousal >= TRAUMA_AROUSAL_THRESHOLD:
        return HALF_LIFE_DAYS * (1 + arousal * 4)  # 0.85→4.4倍, 1.0→5倍
    return HALF_LIFE_DAYS * (1 + arousal * 2)       # 通常: 0.3→1.6倍, 0.5→2倍


def freshness(created_at_str, half_life=HALF_LIFE_DAYS):
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

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
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
            -- 統合された記憶の元IDを記録
            merged_from TEXT DEFAULT NULL
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
        if info and 'procedure' not in info[0]:
            conn.executescript("""
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
                    context_expires_at TEXT DEFAULT NULL,
                    temporal_context TEXT DEFAULT NULL,
                    spatial_context TEXT DEFAULT NULL
                );
                INSERT INTO memories SELECT id, content, category, importance, emotions, arousal,
                    keywords, created_at, last_accessed, access_count, forgotten,
                    source_conversation, embedding, merged_from,
                    NULL, NULL, NULL FROM memories_old;
                DROP TABLE memories_old;
                CREATE INDEX IF NOT EXISTS idx_memories_forgotten ON memories(forgotten);
            """)
            print("  ✓ CHECK制約を更新しました")
    except Exception as e:
        print(f"  (CHECK制約の更新をスキップ: {e})")

    # merged_from カラムを追加（既存DBの場合）
    try:
        conn.execute("ALTER TABLE memories ADD COLUMN merged_from TEXT DEFAULT NULL")
    except sqlite3.OperationalError:
        pass  # already exists
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
    conn.commit()
    conn.close()
    print(f"✓ memory.db を初期化しました: {DB_PATH}")


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
        "SELECT id, content, keywords, embedding, importance, arousal, emotions, category "
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

    for a, b, sim in pairs:
        if a["id"] in merged_ids or b["id"] in merged_ids:
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

            conn.execute(
                """INSERT INTO memories
                   (content, category, importance, emotions, arousal, keywords,
                    embedding, merged_from)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (merged_content, merged_category, merged_importance,
                 json.dumps(merged_emotions), merged_arousal,
                 json.dumps(merged_keywords, ensure_ascii=False),
                 blob, json.dumps([a["id"], b["id"]]))
            )

            # 元の記憶を忘却
            conn.execute("UPDATE memories SET forgotten = 1 WHERE id IN (?, ?)",
                         (a["id"], b["id"]))

            merged_ids.add(a["id"])
            merged_ids.add(b["id"])
            consolidation_count += 1

    conn.commit()
    conn.close()

    if dry_run:
        print(f"\n統合候補: {len(pairs)}ペア")
    else:
        print(f"✓ 統合完了: {consolidation_count}件の記憶を統合")


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

    # 既存スキーマの merged_from を取得して重複チェック用にセット化
    existing_schemas = conn.execute(
        "SELECT merged_from FROM memories "
        "WHERE forgotten = 0 AND category = 'schema' AND merged_from IS NOT NULL"
    ).fetchall()
    existing_sets = set()
    for row in existing_schemas:
        ids = frozenset(json.loads(row["merged_from"]))
        existing_sets.add(ids)

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

            conn.execute(
                """INSERT INTO memories
                   (content, category, importance, emotions, arousal, keywords,
                    embedding, merged_from)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (schema_content, "schema", max_importance,
                 json.dumps(list(all_emotions)), max_arousal,
                 json.dumps(top_keywords, ensure_ascii=False),
                 blob, json.dumps(member_ids))
            )

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
            # 古い記憶の重要度を1段階下げる
            new_imp = max(1, row["importance"] - 1)
            new_arousal = max(0.0, row["arousal"] - 0.1)
            conn.execute(
                "UPDATE memories SET importance = ?, arousal = ? WHERE id = ?",
                (new_imp, new_arousal, row["id"])
            )
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
        return 0.5, None  # embeddingなしなら中程度の誤差を仮定

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

def add_memory(content, category="fact", source=None):
    emotions, arousal, importance = detect_emotions(content)
    keywords = extract_keywords(content)

    # 会話の情動が気分に影響する（情動伝染）
    update_mood(emotions, arousal)

    vec = embed_text(content, is_query=False)
    blob = vec_to_bytes(vec) if vec is not None else None

    conn = get_connection()

    # 予期記憶チェック
    check_prospective(conn, content)

    # 予測符号化 — 予測誤差が大きいほど重要
    pred_error, similar_id = prediction_error(conn, vec)
    importance, arousal = apply_prediction_error(importance, arousal, pred_error)

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

    conn.execute(
        """INSERT INTO memories
           (content, category, importance, emotions, arousal, keywords,
            source_conversation, embedding, context_expires_at, temporal_context,
            spatial_context)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (content, category, importance,
         json.dumps(emotions), arousal, json.dumps(keywords, ensure_ascii=False),
         source, blob, context_expires, temporal_ctx, spatial_ctx)
    )
    conn.commit()
    new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

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
                    conn.execute(
                        "INSERT OR IGNORE INTO links (source_id, target_id, strength) VALUES (?, ?, ?)",
                        (new_id, row["id"], sim)
                    )
                    conn.execute(
                        "INSERT OR IGNORE INTO links (source_id, target_id, strength) VALUES (?, ?, ?)",
                        (row["id"], new_id, sim)
                    )
                    link_count += 1
                except sqlite3.IntegrityError:
                    pass
        conn.commit()

    conn.close()

    emo_str = ", ".join(emotions) if emotions else "中立"
    kw_str = ", ".join(keywords[:5])
    link_str = f", {link_count}件リンク" if link_count else ""
    intf_str = f", {interference_count}件干渉" if interference_count else ""
    # 予測誤差の正規化（表示用）
    _pe_baseline, _pe_ceiling = 0.10, 0.18
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
    print(f"  カテゴリ: {category}{link_str}{intf_str}{pred_str}")

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


def search_memories(query, limit=10, use_like=False, fuzzy=False):
    conn = get_connection()
    fuzzy_results = []  # 舌先現象: 類似度0.45-0.65のもやもや記憶
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    # 予期記憶チェック
    check_prospective(conn, query)

    if use_like or (not is_embed_server_alive() and get_model() is None):
        rows = conn.execute(
            """SELECT * FROM memories WHERE forgotten = 0 AND content LIKE ?
               ORDER BY importance DESC LIMIT ?""",
            (f"%{query}%", limit)
        ).fetchall()
        scored_results = [(row, None) for row in rows]
    else:
        import numpy as np
        query_vec = embed_text(query, is_query=True)

        all_rows = conn.execute(
            "SELECT * FROM memories WHERE forgotten = 0 AND embedding IS NOT NULL"
        ).fetchall()

        scored = []
        for row in all_rows:
            mem_vec = bytes_to_vec(row["embedding"])
            sim = cosine_similarity(query_vec, mem_vec)

            # 情動ブースト
            emo_boost = 1.0 + row["arousal"] * 0.3

            # 鮮度（外傷的記憶は減衰が遅い）
            hl = effective_half_life(row["arousal"])
            fresh = freshness(row["created_at"], half_life=hl)
            fresh_factor = 0.7 + fresh * 0.3

            # 強化
            access_boost = 1.0 + min(row["access_count"], 10) * 0.02

            # プライミング
            priming = get_priming_boost(conn, row["id"])

            # 時間帯ブースト
            temporal = _temporal_boost(row)

            # 場所ブースト（場所細胞）
            spatial = _spatial_boost(row)

            # 気分一致性ブースト
            mood_boost = get_mood_congruence_boost(row)

            # 総合スコア
            score = sim * emo_boost * fresh_factor * access_boost * priming * temporal * spatial * mood_boost
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
            forgotten_rows = conn.execute(
                "SELECT * FROM memories WHERE forgotten = 1 AND embedding IS NOT NULL"
            ).fetchall()
            top_sim = scored[0][2] if scored else 0.0
            for frow in forgotten_rows:
                fvec = bytes_to_vec(frow["embedding"])
                fsim = cosine_similarity(query_vec, fvec)

                # 意図的復活: 検索結果が乏しいとき、非常に高い類似度で復活
                if top_sim < 0.7 and fsim > 0.92:
                    conn.execute(
                        "UPDATE memories SET forgotten = 0, arousal = 0.3 WHERE id = ?",
                        (frow["id"],)
                    )
                    print(f"  🔮 復活: #{frow['id']} {frow['content'][:50]}... (sim:{fsim:.3f})")
                    scored_results.append((frow, fsim))

                # 不随意記憶: 確率的フラッシュバック
                # 元の情動が強い記憶ほど、ふとした手がかりで蘇りやすい
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


def replay_memories():
    """
    リプレイ: リンク再計算 + 弱い記憶の自動忘却 + 統合提案
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, embedding, importance, arousal, access_count, created_at "
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

    existing_links = conn.execute("SELECT id, source_id, target_id, strength FROM links").fetchall()
    pruned = 0
    downscaled = 0
    pruned_pairs = set()  # 刈り込まれたペア（同じリプレイ内で復活させない）
    for link in existing_links:
        src_arousal = arousal_by_id.get(link["source_id"], 0)
        tgt_arousal = arousal_by_id.get(link["target_id"], 0)
        # 外傷的記憶に繋がるリンクは減衰を免除
        if src_arousal >= TRAUMA_AROUSAL_THRESHOLD or tgt_arousal >= TRAUMA_AROUSAL_THRESHOLD:
            continue
        new_strength = link["strength"] * 0.9
        if new_strength < LINK_THRESHOLD:
            conn.execute("DELETE FROM links WHERE id = ?", (link["id"],))
            pruned_pairs.add((link["source_id"], link["target_id"]))
            pruned += 1
        else:
            conn.execute("UPDATE links SET strength = ? WHERE id = ?",
                         (new_strength, link["id"]))
            downscaled += 1

    # 2. 新しい記憶ペアのリンクを追加
    #    既存リンク・刈込済みペアはスキップ（覚醒時の共活性化で復活すべき）
    existing_pairs = set(
        (r["source_id"], r["target_id"])
        for r in conn.execute("SELECT source_id, target_id FROM links").fetchall()
    )
    skip_pairs = existing_pairs | pruned_pairs
    new_links = 0
    for i, row_a in enumerate(rows):
        vec_a = bytes_to_vec(row_a["embedding"])
        for row_b in rows[i+1:]:
            if (row_a["id"], row_b["id"]) in skip_pairs:
                continue
            vec_b = bytes_to_vec(row_b["embedding"])
            sim = cosine_similarity(vec_a, vec_b)

            if sim > LINK_THRESHOLD:
                conn.execute(
                    "INSERT INTO links (source_id, target_id, strength) VALUES (?, ?, ?)",
                    (row_a["id"], row_b["id"], sim)
                )
                conn.execute(
                    "INSERT INTO links (source_id, target_id, strength) VALUES (?, ?, ?)",
                    (row_b["id"], row_a["id"], sim)
                )
                skip_pairs.add((row_a["id"], row_b["id"]))
                skip_pairs.add((row_b["id"], row_a["id"]))
                new_links += 1

    # 2. 弱い記憶の自動忘却
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
    print(f"✓ リプレイ完了: {total_links}リンク（刈込{pruned}本, 減衰{downscaled}本, 新規{new_links}本）, {auto_forgotten}件自動忘却")

    # 3. 統合候補を表示
    consolidate_memories(dry_run=True)


# ============================================================
# デフォルトモードネットワーク — ぼーっとしてるときの脳
# ============================================================

def _get_session_gap(conn):
    """前回の会話からの間隔（時間）を返す。"""
    row = conn.execute(
        "SELECT MAX(last_accessed) as last FROM memories WHERE last_accessed IS NOT NULL"
    ).fetchone()
    if not row or not row["last"]:
        return None
    last = datetime.strptime(row["last"], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    gap_hours = (now - last).total_seconds() / 3600
    return gap_hours


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


def recall_important(limit=15):
    conn = get_connection()

    # context期限切れチェック
    sweep_contexts(conn)

    rows = conn.execute(
        "SELECT * FROM memories WHERE forgotten = 0"
    ).fetchall()

    scored = []
    for row in rows:
        emo_boost = 1.0 + row["arousal"] * 0.5
        hl = effective_half_life(row["arousal"])
        fresh = freshness(row["created_at"], half_life=hl)
        access_boost = 1.0 + min(row["access_count"], 10) * 0.03

        # プライミング
        priming = get_priming_boost(conn, row["id"])

        # 場所ブースト（場所細胞）
        spatial = _spatial_boost(row)

        # 気分一致性ブースト
        mood_boost = get_mood_congruence_boost(row)

        score = emo_boost * (0.5 + fresh * 0.5) * access_boost * priming * spatial * mood_boost
        scored.append((row, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    conn.close()
    return [(s[0], s[1]) for s in scored[:limit]]


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


def recall_polyphonic(limit_per_voice=3):
    """
    内的対話 (polyphonic recall): 複数の声が同時に想起する。

    人間の内的対話を模倣する。頭の中には一つの声ではなく、
    共感・補完・批判・連想が同時に走っている。

    4つの声:
      共感: 気分に寄り添う記憶（状態依存記憶）
      補完: 気分と逆の記憶（見えていないもの）
      批判: 過去の失敗・葛藤・不安の記憶（警告）
      連想: ランダムウォークで到達した意外な記憶（創造性）

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

    # --- 各声のスコア計算 ---
    empathy_scored = []    # 共感
    complement_scored = [] # 補完
    critic_scored = []     # 批判
    associative_scored = []  # 連想

    for row in rows:
        emo_boost = 1.0 + row["arousal"] * 0.5
        hl = effective_half_life(row["arousal"])
        fresh = freshness(row["created_at"], half_life=hl)
        access_boost = 1.0 + min(row["access_count"], 10) * 0.03
        priming = get_priming_boost(conn, row["id"])
        spatial = _spatial_boost(row)

        base = emo_boost * (0.5 + fresh * 0.5) * access_boost * priming * spatial

        # 共感: 気分一致性ブーストで寄り添う
        mood_match = get_mood_congruence_boost(row)
        empathy_scored.append((row, base * mood_match))

        # 補完: 気分不一致ブーストで見えていないものを出す
        mood_oppose = get_mood_incongruence_boost(row, conn)
        complement_scored.append((row, base * mood_oppose))

        # 批判: conflict, anxiety, 失敗の記憶を優先
        mem_emotions = set(json.loads(row["emotions"])) if row["emotions"] else set()
        critic_boost = 1.0
        if mem_emotions & {"conflict", "anxiety"}:
            critic_boost = 1.3
        if row["arousal"] >= 0.7 and "determination" not in mem_emotions:
            critic_boost *= 1.2
        critic_scored.append((row, base * critic_boost))

        # 連想: ランダム性を持たせる（プライミングとarousalに頼らない）
        # アクセス回数が少なく、リンクが多い記憶を優先（辺境の豊かなノード）
        link_count = conn.execute(
            "SELECT COUNT(*) FROM links WHERE source_id = ?", (row["id"],)
        ).fetchone()[0]
        novelty = 1.0 / (1.0 + row["access_count"])
        richness = 1.0 + min(link_count, 20) * 0.05
        random_jitter = 0.8 + random.random() * 0.4  # 0.8-1.2のランダム揺れ
        associative_scored.append((row, (0.5 + fresh * 0.5) * novelty * richness * random_jitter))

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

    # 俯瞰: メタ情報から全体像を構成する（LLMが最も得意なこと）
    voices["俯瞰"] = _birds_eye_view(conn, rows)

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
        forgotten_rows = conn.execute(
            "SELECT * FROM memories WHERE forgotten = 1 AND embedding IS NOT NULL"
        ).fetchall()

        resurrected = []
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


def format_memory_reconstructive(conn, row, similarity=None, score=None):
    """再構成モード: 断片+情動+連想リンクの断片を返す。contentは返さない。"""
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

    # 出力: 断片 + 情動 + 連想からの断片
    frag_str = ", ".join(keywords[:6])
    line = f"  #{row['id']} {emo_str} {stars} [{frag_str}]{sim_str}{score_str}"
    if linked_fragments:
        line += f"\n         ↳ 連想: [{', '.join(linked_fragments)}]"
    if emotions:
        line += f"\n         ↳ 情動: {', '.join(emotions)} (覚醒度:{row['arousal']:.2f})"
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
    if merged:
        lines.append(f"  統合元: #{', #'.join(str(m) for m in merged)}")
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
            "merged_from": json.loads(row["merged_from"]) if row["merged_from"] else None,
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

    elif cmd == "add":
        if len(sys.argv) < 3:
            print("使い方: python memory.py add \"内容\" [category] [source]")
            return
        content = sys.argv[2]
        category = sys.argv[3] if len(sys.argv) > 3 else "fact"
        source = sys.argv[4] if len(sys.argv) > 4 else None
        add_memory(content, category, source)

    elif cmd == "search":
        if len(sys.argv) < 3:
            print("使い方: python memory.py search \"検索語\" [--like] [--raw] [--fuzzy]")
            return
        use_like = "--like" in sys.argv
        raw_mode = "--raw" in sys.argv
        fuzzy_mode = "--fuzzy" in sys.argv
        search_result = search_memories(sys.argv[2], use_like=use_like, fuzzy=fuzzy_mode)
        if fuzzy_mode:
            results, fuzzy_results = search_result
        else:
            results = search_result
            fuzzy_results = []
        if results:
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
        # 舌先現象: もやもや記憶を表示
        if fuzzy_results:
            print(f"  舌先現象 ({len(fuzzy_results)}件):")
            for row, sim in fuzzy_results:
                keywords = json.loads(row["keywords"]) if row["keywords"] else []
                kw_str = ", ".join(keywords[:6])
                print(f"  ?? #{row['id']} もやもや: [{kw_str}]")

        # 反芻検出: 同じ記憶ばかり触ってたら警告
        rum_conn = get_connection()
        ruminating = detect_rumination(rum_conn)
        if ruminating:
            ids_str = ", ".join(f"#{r[0]}[{', '.join(r[2])}]" for r in ruminating)
            print(f"  🔄 同じところを回ってる: {ids_str}")
            print(f"     → recall --voices で別の視点を試してみて")
        rum_conn.close()

    elif cmd == "chain":
        if len(sys.argv) < 3:
            print("使い方: python memory.py chain ID [depth]")
            return
        mid = int(sys.argv[2])
        depth = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        chain = chain_memories(mid, depth)
        if chain:
            print(f"連想の連鎖 (#{mid} から深さ{depth}):")
            for row, d in chain:
                indent = "  " + "→ " * d
                print(f"{indent}#{row['id']} {row['content'][:60]}")
        else:
            print(f"記憶 #{mid} が見つかりません")

    elif cmd == "detail":
        if len(sys.argv) < 3:
            print("使い方: python memory.py detail ID")
            return
        conn = get_connection()
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (int(sys.argv[2]),)).fetchone()
        if row:
            print(format_memory_detail(row))
            links = conn.execute(
                """SELECT l.target_id, l.strength, m.content
                   FROM links l JOIN memories m ON l.target_id = m.id
                   WHERE l.source_id = ? AND m.forgotten = 0
                   ORDER BY l.strength DESC LIMIT 10""",
                (int(sys.argv[2]),)
            ).fetchall()
            if links:
                print(f"  連想リンク ({len(links)}件):")
                for link in links:
                    print(f"    → #{link['target_id']} ({link['strength']:.3f}) {link['content'][:40]}")
        else:
            print("見つかりません")
        conn.close()

    elif cmd == "recent":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        rows = get_recent(n)
        print(f"最近の記憶 ({len(rows)}件):")
        for row in rows:
            print(format_memory(row))

    elif cmd == "all":
        rows = get_all()
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

    elif cmd == "recall":
        # DMN（デフォルトモードネットワーク）: 間隔に応じて起動
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

        # 間隔が長い（6時間以上）なら自動でvoicesモードに切り替え
        auto_voices = gap is not None and gap >= 6.0 and "--voices" not in sys.argv
        if auto_voices:
            print(f"(久しぶりなので内的対話モードで想起します)\n")

        if "--voices" in sys.argv or auto_voices:
            # 内的対話モード
            n = 2 if auto_voices else 3  # 自動の場合は軽めに
            if not auto_voices:
                for arg in sys.argv[2:]:
                    if arg.isdigit():
                        n = int(arg)
            voices = recall_polyphonic(limit_per_voice=n)
            raw_mode = "--raw" in sys.argv
            conn = get_connection()

            # 今効いてる気分を表示
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
            for voice_name, results in voices.items():
                if not results:
                    continue
                emoji = voice_labels.get(voice_name, "·")
                print(f"\n{emoji} [{voice_name}]")
                if voice_name == "俯瞰":
                    # 俯瞰はテキスト断片のリスト
                    for text, _imp in results:
                        print(f"  {text}")
                else:
                    for row, score in results:
                        if raw_mode:
                            print(format_memory(row, score=score))
                        else:
                            print(format_memory_reconstructive(conn, row, score=score))
            conn.close()
        else:
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 15
            raw_mode = "--raw" in sys.argv
            results = recall_important(limit)
            if raw_mode:
                print(f"自動想起 ({len(results)}件):")
                for row, score in results:
                    print(format_memory(row, score=score))
            else:
                conn = get_connection()
                print(f"自動想起 ({len(results)}件, 再構成モード):")
                for row, score in results:
                    print(format_memory_reconstructive(conn, row, score=score))
                conn.close()

    elif cmd == "review":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        review_memories(n)

    elif cmd == "replay":
        replay_memories()

    elif cmd == "consolidate":
        dry = "--dry-run" in sys.argv
        consolidate_memories(dry_run=dry)

    elif cmd == "schema":
        dry = "--dry-run" in sys.argv
        build_schemas(dry_run=dry)

    elif cmd == "proceduralize":
        dry = "--dry-run" in sys.argv
        proceduralize(dry_run=dry)

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

    else:
        print(f"不明なコマンド: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
