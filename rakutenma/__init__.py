"""Rakuten MA Python

Rakuten MA Python (morphological analyzer) is a Python version of Rakuten MA
(word segmentor + PoS Tagger) for Chinese and Japanese.

Rakuten MA Python and Rakuten MA (original) are distributed under
Apache License, version 2.0. http://www.apache.org/licenses/LICENSE-2.0

Rakuten MA Python
(c) 2015- Yukino Ikegami

Rakuten MA (original)
(C) 2014 Rakuten NLP Project."""
from .rakutenma import RakutenMA, Token
from .scw import SCW
from .trie import Trie

VERSION = (0, 3, 1)
__version__ = "0.3.1"
__all__ = ["RakutenMA", "Token", "SCW", "Trie", "_DEF_LABEL", "_BEOS_LABEL",
           "FEATSET_JA", "FEATSET_ZH"]

_DEF_LABEL = "O"     # default label
_BEOS_LABEL = "_"    # label for BOS / EOS

FEATSET_JA = (
    "c0", "w0", "w1", "w9", "w2", "w8",
    "b1", "b9", "b2", "b8",
    "c1", "c9", "c2", "c8", "c3", "c7",
    "d1", "d9", "d2", "d8"
)

FEATSET_ZH = (
    "c0", "w0", "w1", "w9", "w2", "w8",
    "b1", "b9", "b2", "b8",
    "c1", "c9", "c2", "c8", "c3", "c7"
)
