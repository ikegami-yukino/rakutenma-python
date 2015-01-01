# -*- coding: utf-8 -*-
import math
from nose.tools import assert_equals, assert_almost_equals
from rakutenma import SCW, Trie


class TestTrie(object):

    def test_case1(self):
        scw = SCW(0.0, 1.0)

        assert_equals(scw.psi, 1.0)
        assert_equals(scw.zeta, 1.0)

        x = {}

        Trie.insert(x, ["a", "b", "c"], 1.0)
        Trie.insert(scw.mu, ["a", "b", "c"], 1.0)

        margin = scw.calc_margin(x, 1)
        assert_equals(margin, 1.0)

        variance = scw.calc_variance(x)
        assert_equals(variance, 1.0)

        alpha = scw.calc_alpha(margin, variance)
        assert_equals(alpha, 0.0)
        beta = scw.calc_beta(margin, variance, alpha)
        assert_equals(beta, 0.0)

        Trie.insert(x, ["a", "b", "d"], 2.0)
        Trie.insert(scw.mu, ["a", "b", "d"], 0.5)
        Trie.insert(scw.sigma, ["a", "b", "d"], 0.5)

        assert_equals(scw.calc_margin(x, -1), -2.0)
        assert_equals(scw.calc_variance(x), 3.0)

        scw.update_mu_sigma(x, 1, 1.0, 1.0)

        assert_equals(Trie.find(scw.mu, ["a", "b", "c"]), 2.0)
        assert_equals(Trie.find(scw.mu, ["a", "b", "d"]), 1.5)
        assert_equals(Trie.find(scw.sigma, ["a", "b", "c"]), 0.0)
        assert_equals(Trie.find(scw.sigma, ["a", "b", "d"]), -0.5)

    def test_case2(self):
        # case2: C = 1.0, phi = 2.0

        scw = SCW(2.0, 1.0)

        assert_equals(scw.psi, 3.0)
        assert_equals(scw.zeta, 5.0)

        x = {}
        Trie.insert(x, ["a", "b", "c"], 1.0)
        Trie.insert(scw.mu, ["a", "b", "c"], 1.0)

        margin = scw.calc_margin(x, 1)
        assert_equals(margin, 1.0)

        variance = scw.calc_variance(x)
        assert_equals(variance, 1.0)

        alpha = scw.calc_alpha(margin, variance)
        assert_almost_equals(alpha, (math.sqrt(24)-3)/5)

        beta = scw.calc_beta(margin, variance, alpha)
        desired = ((2 * (math.sqrt(24) - 3) / 5) /
                   (0.5 *
                    (-2 * (math.sqrt(24) - 3) / 5 +
                        math.sqrt(4 * (33 - 6 * math.sqrt(24)) / 25 + 4)) +
                    2 * (math.sqrt(24) - 3) / 5))
        assert_almost_equals(beta, desired)

        Trie.insert(x, ["a", "b", "d"], 2.0)
        scw.update_mu_sigma(x, -1, 0.2, 0.5)
        assert_equals(Trie.find(scw.mu, ["a", "b", "c"]), 0.8)
        assert_equals(Trie.find(scw.mu, ["a", "b", "d"]), -0.4)
        assert_equals(Trie.find(scw.sigma, ["a", "b", "c"]), 0.5)
        assert_equals(Trie.find(scw.sigma, ["a", "b", "d"]), -1.0)

    def test_prune(self):
        scw = SCW(0.0, 1.0)
        Trie.insert(scw.mu, ["a", "b", "c"], 0.5)
        Trie.insert(scw.mu, ["a", "b", "d"], 1.5)
        Trie.insert(scw.sigma, ["a", "b", "c"], 0.5)
        Trie.insert(scw.sigma, ["a", "b", "d"], 0.5)

        scw.prune(1.0, 0.8)
        assert_equals(Trie.find(scw.mu, ["a", "b", "c"]), 0)
        assert_equals(Trie.find(scw.mu, ["a", "b", "d"]), 0.5)
        assert_equals(Trie.find(scw.sigma, ["a", "b", "c"]), 0.5)
        assert_equals(Trie.find(scw.sigma, ["a", "b", "d"]), 0.5)

        scw.prune(1.0, 0.4)
        assert_equals(Trie.find(scw.mu, ["a", "b", "c"]), None)
        assert_equals(Trie.find(scw.mu, ["a", "b", "d"]), None)
        assert_equals(Trie.find(scw.sigma, ["a", "b", "c"]), None)
        assert_equals(Trie.find(scw.sigma, ["a", "b", "d"]), None)
