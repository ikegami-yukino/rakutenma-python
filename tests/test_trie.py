# -*- coding: utf-8 -*-
from nose.tools import assert_equals
from rakutenma import Trie


class TestTrie(object):

    def get_trie1(self):
        trie = {}
        Trie.insert(trie, ["A"], 15)
        Trie.insert(trie, ["t", "o"], 7)
        Trie.insert(trie, ["t", "e", "a"], 3)
        Trie.insert(trie, ["t", "e", "d"], 4)
        Trie.insert(trie, ["t", "e", "n"], 12)
        Trie.insert(trie, ["i"], 11)
        Trie.insert(trie, ["i", "n"], 5)
        Trie.insert(trie, ["i", "n", "n"], 9)
        return trie

    def get_trie2(self):
        trie2 = {}
        Trie.insert(trie2, ["t", "e", "a"], 3)
        Trie.insert(trie2, ["t", "e", "d"], 2)
        Trie.insert(trie2, ["t", "e", "n"], 1)
        Trie.insert(trie2, ["t", "e", "x"], 10)
        Trie.insert(trie2, ["t", "e"], 10)
        return trie2

    def test_find(self):
        trie = self.get_trie1()
        assert_equals(Trie.find(trie, ["A"]), 15)
        assert_equals(Trie.find(trie, ["t", "o"]), 7)
        assert_equals(Trie.find(trie, ["t", "e", "a"]), 3)
        assert_equals(Trie.find(trie, ["t", "e", "d"]), 4)
        assert_equals(Trie.find(trie, ["t", "e", "n"]), 12)
        assert_equals(Trie.find(trie, ["i"]), 11)
        assert_equals(Trie.find(trie, ["i", "n"]), 5)
        assert_equals(Trie.find(trie, ["i", "n", "n"]), 9)
        assert_equals(Trie.find(trie, ["i", "n", "n", "n"]), None)
        assert_equals(Trie.find(trie, ["z"]), None)

    def test_inner_prod(self):
        trie1 = self.get_trie1()
        trie2 = self.get_trie2()
        assert_equals(Trie.inner_prod(trie1, trie2), 29)

    def test_add_coef(self):
        trie1 = self.get_trie1()
        trie2 = self.get_trie2()
        Trie.add_coef(trie1, trie2, 0.1)  # with default value = 0

        assert_equals(Trie.find(trie1, ["A"]), 15)
        assert_equals(Trie.find(trie1, ["t", "e", "a"]), 3.3)
        assert_equals(Trie.find(trie1, ["t", "e", "d"]), 4.2)
        assert_equals(Trie.find(trie1, ["t", "e", "n"]), 12.1)
        assert_equals(Trie.find(trie1, ["t", "e", "x"]), 1)
        assert_equals(Trie.find(trie1, ["t", "e"]), 1)

        trie1 = self.get_trie1()
        Trie.add_coef(trie1, trie2, 0.1, 1.0)  # with default value = 1

        assert_equals(Trie.find(trie1, ["A"]), 15)
        assert_equals(Trie.find(trie1, ["t", "e", "a"]), 3.3)
        assert_equals(Trie.find(trie1, ["t", "e", "d"]), 4.2)
        assert_equals(Trie.find(trie1, ["t", "e", "n"]), 12.1)
        assert_equals(Trie.find(trie1, ["t", "e", "x"]), 2)
        assert_equals(Trie.find(trie1, ["t", "e"]), 2)

    def test_mult(self):
        trie1 = self.get_trie1()
        trie2 = self.get_trie2()
        Trie.mult(trie1, trie2)

        assert_equals(Trie.find(trie1, ["A"]), 15)
        assert_equals(Trie.find(trie1, ["t", "e", "a"]), 9)
        assert_equals(Trie.find(trie1, ["t", "e", "d"]), 8)
        assert_equals(Trie.find(trie1, ["t", "e", "n"]), 12)
        assert_equals(Trie.find(trie1, ["t", "e", "x"]), None)
        assert_equals(Trie.find(trie1, ["t", "e"]), None)

    def test_find_partial(self):
        trie = self.get_trie1()
        assert_equals(Trie.find_partial(trie, ["t", "e"])["a"]['v'], 3)
        assert_equals(Trie.find_partial(trie, ["x"]), None)

    def test_toString(self):
        trie = {}
        Trie.insert(trie, ["a", "b"], 1)
        Trie.insert(trie, ["a", "c"], 2)
        actual = Trie.toString(trie)
        assert_equals(set(actual.splitlines()), {"a b\t1", "a c\t2"})

    def test_each(self):
        trie1 = self.get_trie1()
        trie2 = {}
        Trie.each(trie1, lambda key, val: Trie.insert(trie2, key, val))
        assert_equals(trie2, trie1)
