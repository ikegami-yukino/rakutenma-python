# -*- coding: utf-8 -*-
from nose.tools import assert_equals, assert_true, assert_false, assert_raises
from rakutenma import RakutenMA, Token, Trie, _BEOS_LABEL

WEIGHTS = {"c0": {"a": {"B-N": {'v': 1.0}, "I-N": {'v': 1.0}, "E-N": {'v': 1.0}}},
           "w0": {"f": {"B-N": {'v': 1.0}},
                  "o": {"I-N": {'v': 1.0}, "E-N": {'v': 1.0}},
                  "b": {"B-N": {'v': 1.0}},
                  "a": {"I-N": {'v': 1.0}},
                  "r": {"E-N": {'v': 1.0}}},
           "t":   {"I-N": {"B-N": {'v': 1.0}},
                   "E-N": {"I-N": {'v': 1.0}}}}


class TestRakutenMA(object):

    def __init__(self):
        self.rma = RakutenMA()
        del self.rma.hash_func

    def test___init__(self):
        self.rma = RakutenMA()
        assert_equals(self.rma.model, {})
        assert_true(hasattr(self.rma, "scw"))
        assert_equals(self.rma.scw.mu, {})
        assert_equals(self.rma.scw.sigma, {})
        assert_equals(self.rma.scw.phi, 2048)
        assert_equals(self.rma.scw.c, 0.003906)

        self.rma = RakutenMA({"mu": {"feat1": 0.1}, "sigma": {"feat1": 0.2}}, 1.0, 2.0)
        assert_equals(self.rma.model["mu"], {"feat1": 0.1})
        assert_equals(self.rma.model["sigma"], {"feat1": 0.2})
        assert_equals(self.rma.scw.mu, {"feat1": 0.1})
        assert_equals(self.rma.scw.sigma, {"feat1": 0.2})
        assert_equals(self.rma.scw.phi, 1.0)
        assert_equals(self.rma.scw.c, 2.0)

    def test_set_model(self):
        self.rma.set_model({"mu": {"feat1": 0.3}, "sigma": {"feat1": 0.4}})
        assert_equals(self.rma.scw.mu, {"feat1": 0.3})
        assert_equals(self.rma.scw.sigma, {"feat1": 0.4})

    def test_count_tps(self):
        # the last "a" doesn't match because of offset of "d+"
        sent1 = ["a", "b", "c", "d", "a"]
        sent2 = ["a", "b", "c", "d+", "a"]
        assert_equals(RakutenMA.count_tps(sent1, sent2), 3)

        # ignores pos tags for comparison
        sent1 = [["x", "pos1"], ["y", "pos2"], ["z", "pos3"]]
        sent2 = [["x", "pos0"], ["u", "pos2"], ["v", "pos3"], ["x", "pos1"]]
        assert_equals(RakutenMA.count_tps(sent1, sent2), 1)

    def test_eval_corpus(self):
        sent1 = ["a", "b", "c", "d", "a"]
        sent2 = ["a", "b", "c", "d+", "a", "b", "c", "d", "e", "f"]
        res = RakutenMA.eval_corpus([sent1], [sent2])
        assert_equals(res[0], 0.3)
        assert_equals(res[1], 0.6)
        assert_equals(res[2], 0.4)

        assert_raises(Exception, RakutenMA.eval_corpus, (["a"], []))

    def test_tokenize_corpus(self):
        test_corpus = [[["abra", "pos1"], ["cadabra", "pos2"]]]
        tokenize_func = lambda s: list(s)
        desired = [["a", "b", "r", "a", "c", "a", "d", "a", "b", "r", "a"]]
        assert_equals(RakutenMA.tokenize_corpus(tokenize_func, test_corpus), desired)

    def test_tokens_identical(self):
        assert_false(RakutenMA.tokens_identical([["a"]], [[]]))
        assert_false(RakutenMA.tokens_identical([["a"]], [["b"]]))
        assert_false(RakutenMA.tokens_identical([["a", "pos1"]], [["a", "pos2"]]))
        assert_true(RakutenMA.tokens_identical([["a", "pos1"]], [["a", "pos1"]]))

    def test_tokens2csent(self):
        sent = [["hoge", "X"], ["fuga", "Y"], ["p", "Z"]]

        assert_raises(Exception, self.rma.tokens2csent, (sent, "UNKNOWN_SCHEME"))

        csent = self.rma.tokens2csent(sent, "SBIEO")
        assert_equals(csent[1].c, "h")
        assert_equals(csent[1].l, "B-X")
        assert_equals(csent[2].c, "o")
        assert_equals(csent[2].l, "I-X")
        assert_equals(csent[4].c, "e")
        assert_equals(csent[4].l, "E-X")
        assert_equals(csent[9].c, "p")
        assert_equals(csent[9].l, "S-Z")

    def test_csent2tokens(self):
        sent = [["hoge", "X"], ["fuga", "Y"], ["p", "Z"]]
        csent = self.rma.tokens2csent(sent, "SBIEO")
        sent = RakutenMA.csent2tokens(csent, "SBIEO")

        assert_equals(sent[0][0], "hoge")
        assert_equals(sent[0][1], "X")
        assert_equals(sent[1][0], "fuga")
        assert_equals(sent[1][1], "Y")
        assert_equals(sent[2][0], "p")
        assert_equals(sent[2][1], "Z")

        assert_raises(Exception, RakutenMA.csent2tokens, (csent, "UNKNOWN_SCHEME"))

    def test_tokens2string(self):
        sent = [["hoge", "X"], ["fuga", "Y"], ["p", "Z"]]
        assert_equals(RakutenMA.tokens2string(sent), "hoge [X] | fuga [Y] | p [Z]")

    def test_string2hash(self):
        assert_equals(self.rma.string2hash('hoge'), 3208229)
        assert_equals(self.rma.string2hash('piyopiyo'), -105052642)
        assert_equals(self.rma.string2hash(""), 0)

    def test_create_hash_func(self):
        hash_func = self.rma.create_hash_func(4)
        assert_equals(hash_func(["feat1", "foo"]), ['5'])
        assert_equals(hash_func(["feat1", "bar"]), ['2'])
        assert_equals(hash_func(["feat1", "baz"]), ['10'])
        assert_equals(hash_func(["feat1", "qux"]), ['3'])

    def test_str2csent(self):
        actual = self.rma.str2csent('hoge')
        desired = [Token(l=_BEOS_LABEL),
                Token(c="h", t=self.rma.ctype_ja_default_func("h")),
                Token(c="o", t=self.rma.ctype_ja_default_func("o")),
                Token(c="g", t=self.rma.ctype_ja_default_func("g")),
                Token(c="e", t=self.rma.ctype_ja_default_func("e")),
                Token(l=_BEOS_LABEL)]

        assert_equals(len(actual), len(desired))
        for i in range(len(actual)):
            assert_equals(actual[i].c, desired[i].c)
            assert_equals(actual[i].t, desired[i].t)
            assert_equals(actual[i].f, desired[i].f)
            assert_equals(actual[i].l, desired[i].l)

    def test_create_ctype_chardic_func(self):
        cfunc = self.rma.create_ctype_chardic_func({"a": ["type1"], "b": ["type2"]})
        assert_equals(cfunc("a"), ["type1"])
        assert_equals(cfunc("b"), ["type2"])
        assert_equals(cfunc("c"), [])

    def test_ctype_ja_default_func(self):
        assert_equals(self.rma.ctype_ja_default_func("あ"), "H")
        assert_equals(self.rma.ctype_ja_default_func("ア"), "K")
        assert_equals(self.rma.ctype_ja_default_func("Ａ"), "A")
        assert_equals(self.rma.ctype_ja_default_func("ａ"), "a")
        assert_equals(self.rma.ctype_ja_default_func("漢"), "C")
        assert_equals(self.rma.ctype_ja_default_func("百"), "S")
        assert_equals(self.rma.ctype_ja_default_func("0"), "N")
        assert_equals(self.rma.ctype_ja_default_func("・"), "n")

    def test_add_efeats(self):
        # feature functions test
        self.rma.featset = ["w0"]
        csent = self.rma.str2csent("A1-b")
        csent = self.rma.add_efeats(csent)
        assert_equals(csent[0].f, [["w0", ""]])
        assert_equals(csent[1].f, [["w0", "A"]])
        assert_equals(csent[2].f, [["w0", "1"]])
        assert_equals(csent[3].f, [["w0", "-"]])
        assert_equals(csent[4].f, [["w0", "b"]])
        assert_equals(csent[5].f, [["w0", ""]])

        self.rma.featset = ["b1"]
        csent = self.rma.add_efeats(csent)
        assert_equals(csent[0].f, [["b1", "", "A"]])
        assert_equals(csent[1].f, [["b1", "A", "1"]])
        assert_equals(csent[2].f, [["b1", "1", "-"]])
        assert_equals(csent[3].f, [["b1", "-", "b"]])
        assert_equals(csent[4].f, [["b1", "b", ""]])
        assert_equals(csent[5].f, [["b1", "", ""]])

        self.rma.featset = ["c0"]
        csent = self.rma.add_efeats(csent)
        assert_equals(csent[0].f, [["c0", ""]])
        assert_equals(csent[1].f, [["c0", "A"]])
        assert_equals(csent[2].f, [["c0", "N"]])
        assert_equals(csent[3].f, [["c0", "O"]])
        assert_equals(csent[4].f, [["c0", "a"]])
        assert_equals(csent[5].f, [["c0", ""]])

        self.rma.featset = ["d9"]
        csent = self.rma.add_efeats(csent)
        assert_equals(csent[0].f, [["d9", "", ""]])
        assert_equals(csent[1].f, [["d9", "", "A"]])
        assert_equals(csent[2].f, [["d9", "A", "N"]])
        assert_equals(csent[3].f, [["d9", "N", "O"]])
        assert_equals(csent[4].f, [["d9", "O", "a"]])
        assert_equals(csent[5].f, [["d9", "a", ""]])

        self.rma.featset = ["t0"]
        csent = self.rma.add_efeats(csent)
        assert_equals(csent[0].f, [["t0", "", "", "A"]])
        assert_equals(csent[1].f, [["t0", "", "A", "1"]])
        assert_equals(csent[2].f, [["t0", "A", "1", "-"]])
        assert_equals(csent[3].f, [["t0", "1", "-", "b"]])
        assert_equals(csent[4].f, [["t0", "-", "b", ""]])
        assert_equals(csent[5].f, [["t0", "b", "", ""]])

        # test a custom function for feature
        # args _t: a function which receives position i and returns the token,
        #          taking care of boundary cases
        #       i: current position
        # sample function -> returns if the character is a capitalized letter
        self.rma.featset = [lambda _t, i: ["CAP", "T" if _t(i).t == "A" else "F"]]
        csent = self.rma.add_efeats(csent)
        assert_equals(csent[0].f, [["CAP", "F"]])
        assert_equals(csent[1].f, [["CAP", "T"]])
        assert_equals(csent[2].f, [["CAP", "F"]])
        assert_equals(csent[3].f, [["CAP", "F"]])
        assert_equals(csent[4].f, [["CAP", "F"]])
        assert_equals(csent[5].f, [["CAP", "F"]])

        self.rma.featset = ["NONEXISTENT_FEATURE"]
        assert_raises(Exception, self.rma.add_efeats, csent)

    def test_csent2feats(self):
        self.rma.featset = ["w0"]
        csent = self.rma.tokens2csent([["foo", "N"], ["bar", "N"]], "SBIEO")
        csent = self.rma.add_efeats(csent)
        feats = self.rma.csent2feats(csent)
        desired = (["w0", "", "_"], ["w0", "f", "B-N"], ["w0", "o", "I-N"],
                    ["w0", "o", "E-N"], ["w0", "b", "B-N"], ["w0", "a", "I-N"],
                    ["w0", "r", "E-N"], ["t", "B-N", "_"], ["t", "I-N", "B-N"],
                    ["t", "E-N", "I-N"], ["t", "B-N", "E-N"], ["t", "_", "E-N"])
        for d in desired:
            assert_true(d in feats)
        assert_true(["t", "E-N", "B-N"] not in feats)
        assert_true(["t", "B-N", "I-N"] not in feats)


    def test_calc_states0(self):     
        self.rma.featset = ["c0", "w0"]
        csent = self.rma.tokens2csent([["foo", "N"], ["bar", "N"]], "SBIEO")
        csent = self.rma.add_efeats(csent)

        assert_equals(self.rma.calc_states0(csent[1].f, WEIGHTS, {}),
                      {'B-N': 2, 'I-N': 1, 'E-N': 1})
        assert_equals(self.rma.calc_states0(csent[2].f, WEIGHTS, {}),
                      {'B-N': 1, 'I-N': 2, 'E-N': 2})
        assert_equals(self.rma.calc_states0(csent[3].f, WEIGHTS, {}),
                      {'B-N': 1, 'I-N': 2, 'E-N': 2})
        assert_equals(self.rma.calc_states0(csent[4].f, WEIGHTS, {}),
                      {'B-N': 2, 'I-N': 1, 'E-N': 1})
        assert_equals(self.rma.calc_states0(csent[5].f, WEIGHTS, {}),
                      {'B-N': 1, 'I-N': 2, 'E-N': 1})
        assert_equals(self.rma.calc_states0(csent[6].f, WEIGHTS, {}),
                      {'B-N': 1, 'I-N': 1, 'E-N': 2})

    def test_decode(self):
        csent = self.rma.tokens2csent([["foo", "N"], ["bar", "N"]], "SBIEO")
        csent = self.rma.add_efeats(csent)
        for i in range(len(csent)):
          csent[i].l = ''

        self.rma.model['mu'] = WEIGHTS
        csent = self.rma.decode(csent)
        assert_equals(csent[0].l, "_")
        assert_equals(csent[1].l, "B-N")
        assert_equals(csent[2].l, "I-N")
        assert_equals(csent[3].l, "E-N")
        assert_equals(csent[4].l, "B-N")
        assert_equals(csent[5].l, "I-N")
        assert_equals(csent[6].l, "E-N")
        assert_equals(csent[7].l, "_")

        csent = self.rma.tokens2csent([["foX", "N"], ["bar", "N"]], "SBIEO")
        csent = self.rma.add_efeats(csent)
        csent = self.rma.decode(csent)
        assert_equals(csent[0].l, "_")
        assert_equals(csent[1].l, "B-N")
        assert_equals(csent[2].l, "I-N")
        assert_equals(csent[3].l, "O")
        assert_equals(csent[4].l, "B-N")
        assert_equals(csent[5].l, "I-N")
        assert_equals(csent[6].l, "E-N")
        assert_equals(csent[7].l, "_")

    def test_train_one(self):
        self.rma = RakutenMA()
        self.rma.featset = ["w0"]

        res = self.rma.train_one([["foo", "N-nc"], ["bar", "N-nc"]])
        assert_true(res['updated'])
        assert_true(Trie.find(self.rma.model['mu'], ["w0", "f", "B-N"]) > 0)
        assert_true(Trie.find(self.rma.model['mu'], ["w0", "o", "I-N"]) > 0)
        assert_true(Trie.find(self.rma.model['mu'], ["w0", "o", "E-N"]) > 0)
        assert_equals(self.rma.tokenize("foobar"), [["foo", "N-nc"], ["bar", "N-nc"]])

    def test_set_tag_scheme(self):
        self.rma.set_tag_scheme("IOB2")
        assert_equals(self.rma.tag_scheme, "IOB2")
