#!/usr/bin/env python3
"""
tokenizer.py - 日本語テキストの形態素解析

fugashi (MeCab) 優先、なければ SudachiPy にフォールバック、
どちらもなければ簡易正規表現トークナイザで処理。

delusionモードのFTS5検索で使用。
"""

import re

_tokenizer_backend = None  # "fugashi" | "sudachi" | "regex"
_fugashi_tagger = None
_sudachi_tokenizer = None


def _init_fugashi():
    global _fugashi_tagger, _tokenizer_backend
    try:
        import fugashi
        _fugashi_tagger = fugashi.Tagger()
        _tokenizer_backend = "fugashi"
        return True
    except (ImportError, RuntimeError):
        return False


def _init_sudachi():
    global _sudachi_tokenizer, _tokenizer_backend
    try:
        from sudachipy import Dictionary
        _sudachi_tokenizer = Dictionary().create()
        _tokenizer_backend = "sudachi"
        return True
    except ImportError:
        return False


def _init():
    """トークナイザを初期化。一度だけ実行。"""
    global _tokenizer_backend
    if _tokenizer_backend is not None:
        return
    if not _init_fugashi():
        if not _init_sudachi():
            _tokenizer_backend = "regex"


def tokenize(text):
    """テキストをスペース区切りの形態素解析に変換する。"""
    _init()

    if _tokenizer_backend == "fugashi":
        words = _fugashi_tagger(text)
        tokens = [w.surface for w in words if w.surface.strip()]
        # 英数字トークンも追加
        en_tokens = re.findall(r'[A-Za-z][A-Za-z0-9_\-]+', text)
        tokens.extend(t.lower() for t in en_tokens if len(t) > 1)
        return " ".join(tokens)

    elif _tokenizer_backend == "sudachi":
        from sudachipy import Tokenizer as SudachiMode
        morphemes = _sudachi_tokenizer.tokenize(text, SudachiMode.SplitMode.C)
        tokens = [m.surface() for m in morphemes if m.surface().strip()]
        en_tokens = re.findall(r'[A-Za-z][A-Za-z0-9_\-]+', text)
        tokens.extend(t.lower() for t in en_tokens if len(t) > 1)
        return " ".join(tokens)

    else:
        # 簡易正規表現トークナイザ
        # 漢字・カタカナの連続、ひらがなの連続、英数字の連続を抽出
        jp_tokens = re.findall(r'[\u4e00-\u9fff\u30a0-\u30ff]+', text)
        hira_tokens = re.findall(r'[\u3040-\u309f]{2,}', text)
        en_tokens = re.findall(r'[A-Za-z][A-Za-z0-9_\-]+', text)
        en_tokens = [t.lower() for t in en_tokens if len(t) > 1]
        return " ".join(jp_tokens + hira_tokens + en_tokens)


def get_backend():
    """現在使用中のバックエンド名を返す。"""
    _init()
    return _tokenizer_backend
