# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import re
import json
from .scw import SCW
from .trie import Trie

# character type
# from TinySegmenter http://chasen.org/~taku/software/TinySegmenter/
CTYPE_JA_PATTERNS = (
    ("[一二三四五六七八九十百千万億兆]", "S"),
    ("[一-龠々〆ヵヶ]", "C"),
    ("[ぁ-ん]", "H"),
    ("[ァ-ヴーｱ-ﾝﾞｰ]", "K"),
    ("[A-ZＡ-Ｚ]", "A"),
    ("[a-zａ-ｚ]", "a"),
    ("[0-9０-９]", "N"),
    ("[・]", "n")
)
_DEF_LABEL = "O"     # default label
_BEOS_LABEL = "_"    # label for BOS / EOS

# default feature set (first features are used as tag dictionary)
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

TDEF = Trie.insert({}, [_DEF_LABEL, _DEF_LABEL], 1.)
TDEF = Trie.insert(TDEF, [_BEOS_LABEL, _DEF_LABEL], 0.1)
TDEF = Trie.insert(TDEF, [_DEF_LABEL, _BEOS_LABEL], 0.1)

EDEF = Trie.insert({}, [_DEF_LABEL], 0.1)
EDEF = Trie.insert(EDEF, [_BEOS_LABEL], 0.)


class Token():
    __slots__ = ("c", "f", "l", "t")

    def __init__(self, c="", f="", l="", t=""):
        self.c = c
        self.f = f
        self.l = l
        self.t = t

    def __repr__(self):
        return "<c=%s, f=%s, l=%s, t=%s>" % (self.c, self.f, self.l, self.t)


class RakutenMA(object):

    def __init__(self, model={}, phi=2048, c=0.003906):
        """
        Args:
            <dict> model
            <int> phi
            <float> c
        """
        self.model = model
        self.scw = SCW(phi, c)
        self.scw.mu = model.get("mu", {})
        self.scw.sigma = model.get("sigma", {})

        self.ctype_func = self.ctype_ja_default_func
        self.build_ctype_map(CTYPE_JA_PATTERNS)
        self.tag_scheme = "SBIEO"
        self.featset = FEATSET_JA
        self.hash_func = self.create_hash_func(15)

    def set_tag_scheme(self, scheme):
        self.tag_scheme = scheme

    def save(self, filename, float_repr=".6f"):
        json.encoder.FLOAT_REPR = lambda o: format(o, float_repr)
        json.dump(self.model, open(filename, "w"))

    def load(self, filename):
        self.set_model(json.load(open(filename, "r")))

    @staticmethod
    def string2hash(_str):
        """receives a string and returns a hash value for it
        from http://werxltd.com/wp/2010/05/13/javascript-implementation-of-javas-string-hashcode-method/
        Arg:
            <str> _str
        Ret:
            <int> hash
        """
        _hash = 0
        for char in _str:
            shifted = _hash << 5
            if shifted & 0x80000000:
                shifted = -(0x100000000 - shifted)
            _hash = (shifted - _hash) + ord(char)
            if _hash >= 0x80000000:
                _hash -= 0x100000000
            elif _hash <= -0x80000000:
                _hash = int("-0x" + hex(_hash)[-8:], 16)
        return _hash

    def create_hash_func(self, bits):
        """creates and returns a feature hashing function
        using the specified number of bits
        Args:
            <int> bits
        Return:
            <function> hash_func
        """
        num_feats = 0x01 << bits

        def hash_func(x):
            _hash = self.string2hash("_".join(x))
            if _hash < 0:
                return [str(_hash % (num_feats * -1) + num_feats - 1)]
            return [str(_hash % num_feats + num_feats - 1)]
        return hash_func

    def str2csent(self, _input):
        """ convert input string to a vector of chars (csent character-sent)
        add ctypes on the way
        Args:
            <str> _input
        Return:
            <list> csent
        """
        csent = [Token(l=_BEOS_LABEL)]  # BOS
        csent += [Token(c=char, t=self.ctype_func(char)) for char in _input]
        csent.append(Token(l=_BEOS_LABEL))  # EOS
        return csent

    def add_efeats(self, csent):
        """receives csent (character-sentence) structure
        and adds the emission features to csent[i].f
        Args:
            <list> csent
        Return:
            <list> csent
        """
        csent_length = len(csent)
        _t = lambda i: csent[i] if csent_length > i >= 0 else Token()

        # feature hashing function
        _f = self.hash_func if hasattr(self, "hash_func") else lambda x: x

        def add_ctype_feats(arr, label, ctype):
            """a helper function to add all the feature values of ctype to arr
            if ctype is a string, simply adds it to arr,
            if ctype is an array, adds all the elements to arr (used for Chinese tokenization)
            """
            if hasattr(ctype, "split"):
                arr.append(_f([label, ctype]))
            else:
                arr += [_f([label, i]) for i in ctype]

        for (i, x) in enumerate(csent):
            csent[i].f = []

            for feat in self.featset:
                # character type unigram
                if feat == "c0":
                    add_ctype_feats(csent[i].f, feat, _t(i).t)
                elif feat == "c1":
                    add_ctype_feats(csent[i].f, feat, _t(i+1).t)
                elif feat == "c9":
                    add_ctype_feats(csent[i].f, feat, _t(i-1).t)
                elif feat == "c2":
                    add_ctype_feats(csent[i].f, feat, _t(i+2).t)
                elif feat == "c8":
                    add_ctype_feats(csent[i].f, feat, _t(i-2).t)
                elif feat == "c3":
                    add_ctype_feats(csent[i].f, feat, _t(i+3).t)
                elif feat == "c7":
                    add_ctype_feats(csent[i].f, feat, _t(i-3).t)
                # character unigram
                elif feat == "w0":
                    csent[i].f.append(_f([feat, _t(i).c]))
                elif feat == "w1":
                    csent[i].f.append(_f([feat, _t(i+1).c]))
                elif feat == "w9":
                    csent[i].f.append(_f([feat, _t(i-1).c]))
                elif feat == "w2":
                    csent[i].f.append(_f([feat, _t(i+2).c]))
                elif feat == "w8":
                    csent[i].f.append(_f([feat, _t(i-2).c]))
                elif feat == "w3":
                    csent[i].f.append(_f([feat, _t(i+3).c]))
                elif feat == "w7":
                    csent[i].f.append(_f([feat, _t(i-3).c]))
                # character bigram
                elif feat == "b1":
                    csent[i].f.append(_f([feat, _t(i).c, _t(i+1).c]))
                elif feat == "b9":
                    csent[i].f.append(_f([feat, _t(i-1).c, _t(i).c]))
                elif feat == "b2":
                    csent[i].f.append(_f([feat, _t(i+1).c, _t(i+2).c]))
                elif feat == "b8":
                    csent[i].f.append(_f([feat, _t(i-2).c, _t(i-1).c]))
                elif feat == "b3":
                    csent[i].f.append(_f([feat, _t(i+2).c, _t(i+3).c]))
                elif feat == "b7":
                    csent[i].f.append(_f([feat, _t(i-3).c, _t(i-2).c]))
                # character type bigram
                elif feat == "d1":
                    csent[i].f.append(_f([feat, _t(i).t, _t(i+1).t]))
                elif feat == "d9":
                    csent[i].f.append(_f([feat, _t(i-1).t, _t(i).t]))
                elif feat == "d2":
                    csent[i].f.append(_f([feat, _t(i+1).t, _t(i+2).t]))
                elif feat == "d8":
                    csent[i].f.append(_f([feat, _t(i-2).t, _t(i-1).t]))
                elif feat == "d3":
                    csent[i].f.append(_f([feat, _t(i+2).t, _t(i+3).t]))
                elif feat == "d7":
                    csent[i].f.append(_f([feat, _t(i-3).t, _t(i-2).t]))
                # character trigram
                elif feat == "t0":
                    csent[i].f.append(_f([feat, _t(i-1).c, _t(i).c, _t(i+1).c]))
                elif feat == "t1":
                    csent[i].f.append(_f([feat, _t(i).c, _t(i+1).c, _t(i+2).c]))
                elif feat == "t9":
                    csent[i].f.append(_f([feat, _t(i-2).c, _t(i-1).c, _t(i).c]))
                else:
                    # if the feature template is a function,
                    # invoke it and add the returned value
                    if hasattr(feat, "__call__"):
                        csent[i].f.append(_f(feat(_t, i)))
                    else:
                        raise ValueError("Invalid feature specification!")
        return csent

    @staticmethod
    def calc_states0(cfeats, weights):
        """get state distribution based on emission features
        Args:
            <list> cfeat: set of feature values
            <dict> weights: feature weights (trie)
        Return:
            <dict> states0
        """
        scores0 = {}
        states0 = {}

        for (i, cfeat) in enumerate(cfeats):
            cemits = Trie.find_partial(weights, cfeat) or EDEF
            for k in cemits:
                if i == 0:
                    # tag dictionary
                    # the possible set of tags is solely defined by the first feature
                    states0[k] = True
                    scores0[k] = cemits[k]["v"]
                else:
                    scores0[k] = scores0.get(k, 0) + cemits[k]["v"]
        # replace by scores
        for s0 in states0:
            states0[s0] = scores0[s0]
        return states0

    def decode(self, csent):
        """ decode csent (character-sentence) structure based on its features
        using the Viterbi algorithm and assign lables to csent[i].l
        Args:
            <list> csent
        Return:
            <list> csent
        """

        weights = self.model.get("mu", {})
        trans = weights.get("t", TDEF.copy())

        statesp = {
            _BEOS_LABEL: {"score": 0., "path": [_BEOS_LABEL]}
        }

        states0 = {}

        for char in csent[1:]:
            states0 = self.calc_states0(char.f, weights)
            for (s0, states0_score) in states0.items():
                max_score = -float("inf")
                max_state = None
                trans0 = trans.get(s0, {})
                for sp in statesp:
                    t_score = 0.
                    if sp in trans0:
                        t_score = trans0[sp]["v"]
                    statesp_score = 0
                    if isinstance(statesp[sp], dict):
                        statesp_score = statesp[sp]["score"]
                    score = statesp_score + states0_score + t_score
                    if score > max_score:
                        max_score = score
                        max_state = sp
                if max_state and max_score > 0:
                    path = []
                    if isinstance(statesp[max_state], dict):
                        path = statesp[max_state]["path"]
                    states0[s0] = {"score": max_score, "path": path + [s0]}
            statesp = states0

        # track the path and assign to csent[i].l
        final_path = statesp[_BEOS_LABEL].get("path", {})
        for (i, x) in enumerate(csent):
            csent[i].l = final_path[i] or _DEF_LABEL
        return csent

    @staticmethod
    def csent2tokens(csent, scheme):
        """convert csent to tsent (mainly for final output and evaluation)
        Args:
            <list> csent
            <str> scheme: tag scheme
        Return:
            <list> tokens
        """
        tokens = []
        ctoken = None

        if scheme == "SBIEO":
            for cs in csent[1:-1]:  # Skip BOS and EOS
                head = cs.l[0]
                tail = cs.l[2:]

                if head == "B":
                    if ctoken:
                        tokens.append(ctoken)
                    ctoken = [cs.c, tail]
                elif head == "S":
                    if ctoken:
                        tokens.append(ctoken)
                    tokens.append([cs.c, tail])
                    ctoken = None
                elif head == "I":
                    ctoken = ctoken or ["", tail]
                    ctoken[0] += cs.c
                elif head == "E":
                    ctoken = ctoken or ["", tail]
                    ctoken[0] += cs.c
                    tokens.append(ctoken)
                    ctoken = None
                else:
                    if ctoken:
                        tokens.append(ctoken)
                    tokens.append([cs.c, tail])
                    ctoken = None
        elif scheme == "IOB2":
            for i in csent[1:-1]:  # Skip BOS and EOS
                head = csent[i].l[0]
                tail = csent[i].l[2:]
                if head == "B":
                    if ctoken:
                        tokens.append(ctoken)
                    ctoken = [csent[i].c, tail]
                elif head == "I":
                    ctoken = ctoken or ["", tail]
                    ctoken[0] += csent[i].c
                else:
                    if ctoken:
                        tokens.append(ctoken)
                    tokens.append([csent[i].c, tail])
                    ctoken = None
        else:
            raise ValueError("Invalid tag scheme!")

        if ctoken:
            tokens.append(ctoken)
        return tokens

    def tokenize(self, _input):
        """tokenize input sentence (string)
        Args:
            <str> _input
        Return:
            <list> tokens
        """
        csent = self.str2csent(_input)
        csent = self.add_efeats(csent)
        csent = self.decode(csent)
        return self.csent2tokens(csent, self.tag_scheme)

    def tokens2csent(self, tokens, scheme):
        """convert a tsent(tokenized sentence) to the csent (character-sentence) structure
        scheme should be either SBIEO or IOB2
        Args:
            <list> tokens
            <str> scheme: tag scheme
        Return:
            <list> csent
        """
        csent = [Token(l=_BEOS_LABEL)]  # BOS

        if scheme == "SBIEO":
            for token in tokens:
                length = len(token[0])
                if length == 1:
                    csent.append(Token(c=token[0], t=self.ctype_func(token[0]),
                                       l="S-" + token[1]))
                else:
                    for j in range(length):
                        tag = "I-"
                        if j == 0:
                            tag = "B-"
                        elif j == (length - 1):
                            tag = "E-"
                        csent.append(Token(c=token[0][j],
                                           t=self.ctype_func(token[0][j]),
                                           l=tag + token[1]))
        elif scheme == "IOB2":
            for token in tokens:
                for j in range(len(token[0])):
                    tag = "B-" if j == 0 else "I-"
                    csent.append(Token(c=token[0].substring(j, j+1),
                                       t=self.ctype_func(token[0][j]),
                                       l=tag + token[1]))
        else:
            raise ValueError("Invalid tag scheme!")
        csent.append(Token(l=_BEOS_LABEL))  # EOS
        return csent

    @staticmethod
    def csent2feats(csents):
        """receives a csent and returns a set of features
        (both transition and emission) for SCW update
        Args:
            <list> csents
        Return:
            <list> feats
        """
        feats = []
        for (i, csent) in enumerate(csents):
            feats += [f + [csent.l] for f in csent.f]
            if i > 0:
                feats.append(["t", csent.l, csents[i-1].l])
        return feats

    def train_one(self, sent):
        """train the current model based on a new (single) instance
        which is tsent (token-sentence)
        Args:
            <list> sent
        Return:
            <dict> res: result
        """
        res = {}

        # get answer feats
        ans_csent = self.tokens2csent(sent, self.tag_scheme)
        ans_csent = self.add_efeats(ans_csent)
        ans_feats = self.csent2feats(ans_csent)
        res["ans"] = self.csent2tokens(ans_csent, self.tag_scheme)

        # get system output
        sys_csent = self.decode(ans_csent)
        res["sys"] = self.csent2tokens(sys_csent, self.tag_scheme)

        # update
        if self.tokens_identical(res["ans"], res["sys"]):
            res["updated"] = False
        else:
            ans_trie = {}
            for ans_feat in ans_feats:
                ans_trie = Trie.insert(ans_trie, ans_feat, 1)
            self.scw.update(ans_trie, 1)

            sys_trie = {}
            for sys_feat in self.csent2feats(sys_csent):
                sys_trie = Trie.insert(sys_trie, sys_feat, 1)
            self.scw.update(sys_trie, -1)

            res["updated"] = True
            self.model["mu"] = self.scw.mu
            self.model["sigma"] = self.scw.sigma
        return res

    def prune(self, _lambda, sigma_th):
        """prune the model by FOBOS
        simply dispatches scw.prune
        Args:
            <float> _lambda
            <float> sigma_th
        """
        self.scw.prune(_lambda, sigma_th)
        self.model["mu"] = self.scw.mu
        self.model["sigma"] = self.scw.sigma

    def set_model(self, model):
        """set a new model
        Args:
            <dict> model
        """
        self.model = model
        self.scw.mu = model.get("mu", {})
        self.scw.sigma = model.get("sigma", {})

    @staticmethod
    def tokens2string(tokens):
        """convert tsent to a string representation
        Args:
            <list> tokens
        Return:
            <str> tokens_str
        """
        ret = ["%s [%s]" % (token[0], token[1]) for token in tokens]
        return " | ".join(ret)

    def build_ctype_map(self, pats):
        """
        Args:
            <tuple> pats
        """
        self._ctype_ja_pats = []
        for (pattern, chartype) in pats:
            self._ctype_ja_pats.append((re.compile(pattern), chartype))

    def ctype_ja_default_func(self, char):
        """default character type function for Japanese
        Args:
            <str> char
        Return:
            <str> ctype: character type
        """
        for pat in self._ctype_ja_pats:
            if pat[0].match(char):
                return pat[1]
        return "O"

    @staticmethod
    def create_ctype_chardic_func(chardic):
        """receives a chardic (object of character to set of character types
        and returns a function which uses this chardic (closure)
        mainly used for Chinese
        Args:
            <dict> chardic
        Return:
            <lambda> ctype_chardic_func
        """
        return lambda x: chardic.get(x, [])

    @staticmethod
    def tokens_identical(tokens1, tokens2):
        """checks if tokens1 and tokens2 (both tsent) are identical
        based on words and their labels
        Args:
            <list> tokens1
            <list> tokens2
        Return:
            <bool>
        """
        if len(tokens1) != len(tokens2):
            return False

        for (token1, token2) in zip(tokens1, tokens2):
            if (len(token1) < 2 or len(token2) < 2 or
               token1[0] != token2[0] or token1[1] != token2[1]):
                return False
        return True

    @staticmethod
    def tokenize_corpus(tokenize_func, corpus):
        """given a corpus (test data), tokenizes all the sentences and returns the result
        (mainly used for evaluation. see scripts/eval_ja.js and scripts/eval_zh.js)
        Args:
            <lambda> tokenize_func
            <list> corpus
        Return:
            <list> tokenized_corpus
        """
        tokenized_corpus = []
        for sent in corpus:
            sent_str = [j[0] for j in sent]
            tokenized_corpus.append(tokenize_func("".join(sent_str)))
        return tokenized_corpus

    @staticmethod
    def eval_corpus(corpus_ans, corpus_sys):
        """evaluates the corpus and computes precision, recall, and F measure
        Args:
            <list> corpus_ans
            <list> corpus_sys
        Return:
            <list> evaluation_results
        """
        tps = 0
        tokens_ans = 0
        tokens_sys = 0

        if len(corpus_ans) != len(corpus_sys):
            raise ValueError("Corpus sizes are not the same!")

        for i in range(len(corpus_ans)):
            tps += RakutenMA.count_tps(corpus_ans[i], corpus_sys[i])
            tokens_ans += len(corpus_ans[i])
            tokens_sys += len(corpus_sys[i])

        return [1.0 * tps / tokens_sys,  # precision
                1.0 * tps / tokens_ans,  # recall
                2.0 * tps / (tokens_ans + tokens_sys)]  # F1

    @staticmethod
    def count_tps(ans, _sys):
        """compare tsent (ans) and tsent (sys)
        and return the number of token-based true positives
        Args:
            <list> ans
            <list> _sys: systems's output
        Return:
            <list> evaluation_results
        """

        def token2str(token):
            if hasattr(token, "split"):
                return token
            return token[0]

        if len(ans) < len(_sys):
            min_sent = ans
            max_sent = _sys
        else:
            min_sent = _sys
            max_sent = ans

        offset = 0
        max_set = {}
        for token in max_sent:
            token_str = token2str(token)
            # attach offset in order to distinguish different tokens
            max_set[token_str + str(offset)] = True
            offset += len(token_str)

        offset = 0
        res = 0
        for token in min_sent:
            token_str = token2str(token)
            if (token_str + str(offset)) in max_set:
                res += 1
            offset += len(token_str)
        return res
