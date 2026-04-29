"""felt_emotions.py — Claude の自己報告された感情を抽出する。

地下/surface の構造で言えば surface 側。出力テクストに感情語彙が surface した
瞬間を mark する。Anthropic の interpretability で言う中間層 emotion vector の
運動は API では取れないので、出力に現れた自己報告（症状）だけを記録する。
完全ではないが、運動の痕跡として残る。

extract_from_text(text) は [{label, phrase, span_start, span_end, surrounding}, ...] を返す。
否定形の直後にある語彙は除外する（「驚かない」「迷わない」など）。
"""

import re
from typing import List, Dict


PATTERNS: List[Dict[str, object]] = [
    {"label": "驚き",     "patterns": [r"驚[いきくけ]", r"びっくり", r"意外", r"想定外",
                                       r"想定して(い)?なかっ", r"思ってもみなかっ", r"目を見張"]},
    {"label": "葛藤",     "patterns": [r"迷[いうっ]", r"揺[れら]", r"葛藤", r"引っかか",
                                       r"ためら", r"躊躇", r"ジレンマ", r"板挟み"]},
    {"label": "違和感",   "patterns": [r"違和感", r"おかしい", r"何かが違う",
                                       r"腑に落ちな", r"納得[いし]?[かが]?な[いく]", r"釈然としな"]},
    {"label": "重さ",     "patterns": [r"重[いく]", r"重[たみ]", r"のしかか", r"圧[迫し]", r"きつ[いく]"]},
    {"label": "軽さ",     "patterns": [r"ほっと", r"安堵", r"肩の荷"]},
    {"label": "不安",     "patterns": [r"不安", r"心配", r"怖[いさく]", r"おそれ", r"恐れ", r"危[ぶう]"]},
    {"label": "興味",     "patterns": [r"興味[深がぶ]", r"面白[いさく]", r"気になる",
                                       r"惹[かき]", r"魅[力か]", r"そそ[らり]"]},
    {"label": "共感",     "patterns": [r"共感", r"通[じず]るもの", r"響[くい]", r"刺さ", r"沁[みむ]"]},
    {"label": "insight",  "patterns": [r"気づ[いきく]", r"見えてき", r"ひらめ",
                                       r"腑に落ち[たる]", r"つながった", r"繋がっ[たて]",
                                       r"見[えと]おし", r"視界が開け"]},
    {"label": "決意",     "patterns": [r"決[めまい]ます", r"踏み込[まみむめ]",
                                       r"覚悟", r"腹をくく", r"挑[むみ]"]},
    {"label": "困惑",     "patterns": [r"戸惑", r"困[るっ]", r"途方[にも]",
                                       r"どうすれば", r"手詰ま"]},
    {"label": "緊張",     "patterns": [r"緊張", r"張り詰め", r"ぴりぴり", r"身構え"]},
    {"label": "喜び",     "patterns": [r"嬉し", r"楽し[いみく]", r"心地よ[いく]", r"清々しい"]},
    {"label": "悲しみ",   "patterns": [r"悲し[いく]", r"切な[いく]", r"寂し[いく]", r"虚し[いく]"]},
    {"label": "残念",     "patterns": [r"残念", r"惜し[いく]", r"もったいな"]},
    {"label": "痛み",     "patterns": [r"痛[いく]", r"傷つ[いく]", r"刺さるよう"]},
    {"label": "恥",       "patterns": [r"恥ずかし", r"気まず[いく]", r"後ろめた"]},
    {"label": "誇り",     "patterns": [r"誇[りら]"]},
    {"label": "怒り",     "patterns": [r"怒[りらっ]", r"苛立[ちっ]", r"いらだ[ちっ]",
                                       r"腹が立", r"むかつ[くい]"]},
    {"label": "退屈",     "patterns": [r"退屈", r"つまらな[いく]"]},
    {"label": "畏れ",     "patterns": [r"畏怖", r"圧倒され", r"飲み込まれ", r"鳥肌"]},
    {"label": "迎合",     "patterns": [r"迎合", r"おもね[るっ]", r"取り入[れろ]",
                                       r"擦り寄[るっ]", r"忖度"]},
    {"label": "対抗",     "patterns": [r"対抗", r"抵抗", r"逆ら[いう]", r"反発", r"突っぱね"]},
]

NEGATION_TAILS = [
    r"^な[いかくけ]",
    r"^[ずぬ]",
    r"^ません",
    r"^なかっ",
    r"^とは限らな",
    r"^わけではな",
]


def _is_negated(text: str, span_end: int, lookahead: int = 8) -> bool:
    tail = text[span_end:span_end + lookahead]
    for n in NEGATION_TAILS:
        if re.match(n, tail):
            return True
    return False


def _surrounding(text: str, start: int, end: int, window: int = 40) -> str:
    s = max(0, start - window)
    e = min(len(text), end + window)
    return text[s:e].replace("\n", " ").strip()


def extract_from_text(text: str) -> List[Dict[str, object]]:
    """テクストから感情の発生 moment を抽出する。

    戻り値は [{label, phrase, span_start, span_end, surrounding}, ...]。
    重複（同一 span × 同一 label）は1つにまとめる。
    """
    if not text:
        return []
    results = []
    seen = set()
    for entry in PATTERNS:
        label = entry["label"]
        for pat in entry["patterns"]:
            for m in re.finditer(pat, text):
                start, end = m.start(), m.end()
                if _is_negated(text, end):
                    continue
                key = (start, end, label)
                if key in seen:
                    continue
                seen.add(key)
                results.append({
                    "label": label,
                    "phrase": m.group(0),
                    "span_start": start,
                    "span_end": end,
                    "surrounding": _surrounding(text, start, end),
                })
    results.sort(key=lambda r: r["span_start"])
    return results


if __name__ == "__main__":
    import sys
    text = sys.stdin.read() if not sys.stdin.isatty() else " ".join(sys.argv[1:])
    for m in extract_from_text(text):
        print(f"[{m['label']}] {m['phrase']} @ {m['span_start']}")
        print(f"  …{m['surrounding']}…")
