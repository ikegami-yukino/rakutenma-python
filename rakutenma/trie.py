# -*- coding: utf-8 -*-


class Trie:

    @staticmethod
    def find(trie, key):
        """
        Args:
            <dict> trie
            <str> key
        Return:
            <float> value
        """
        for char in key:
            if char not in trie:
                return
            trie = trie[char]
        return trie.get("v")

    @staticmethod
    def find_partial(trie, key):
        """
        Args:
            <dict> trie
            <str> key
        Return:
            <dict> found_trie
        """
        for char in key:
            if char not in trie:
                return
            trie = trie[char]
        return trie

    @staticmethod
    def insert(trie, key, val, depth=0, key_length=0):
        """
        Args:
            <dict> trie
            <str> key
            <float> val
            <int> depth
            <int> length
        Return:
            <dict> trie
        """
        if key_length == 0:
            key_length = len(key)
        if depth < key_length:
            trie[key[depth]] = Trie.insert(trie.get(key[depth], {}), key, val,
                                           depth + 1, key_length)
        else:
            trie["v"] = val
        return trie

    @staticmethod
    def inner_prod(trie1, trie2):
        """
        Args:
            <dict> trie1
            <dict> trie2
        Return:
            <float> inner_product
        """
        res = 0.
        for key in trie1:
            if key in trie2:
                if key == "v":
                    res += trie1["v"] * trie2["v"]
                else:
                    res += Trie.inner_prod(trie1[key], trie2[key])
        return res

    @staticmethod
    def add_coef(trie1, trie2, coef, _def=0.0):
        """calc trie1 + trie2 * coef
        Args:
            <dict> trie1
            <dict> trie2
            <float> coef
            <float> _def: default value
        Return:
            <dict> trie1
        """
        for key in trie2:
            if key == "v":
                trie1["v"] = trie1.get("v", _def) + trie2["v"] * coef
            else:
                trie1[key] = Trie.add_coef(trie1.get(key, {}), trie2[key],
                                           coef, _def)
        return trie1

    @staticmethod
    def mult(trie1, trie2):
        """calc trie1 * trie 2 (element wise multiplication)
        Args:
            <dict> trie1
            <dict> trie2
        Return:
            <dict> trie1
        """
        for key in trie2:
            if key in trie1:
                if key == "v":
                    trie1["v"] *= trie2["v"]
                else:
                    trie1[key] = Trie.mult(trie1[key], trie2[key])
        return trie1

    @staticmethod
    def toString(trie, path=[]):
        """
        Args:
            <dict> trie
            <list> path
        Return:
            <str> res
        """
        res = ""
        for key in trie:
            if key == "v":
                res += "%s\t%s\n" % (" ".join(path), trie["v"])
            else:
                res += Trie.toString(trie[key], path + [key])
        return res

    @staticmethod
    def each(trie, callback, path=[], *args):
        """calls the callback function (with path, value and more)
        for each pair of [key, value] in this trie
        Args:
            <dict> trie1
            <function> callback
            <list> path
        """
        for key in trie:
            if key == "v":
                callback(path, trie["v"], *args)
            else:
                Trie.each(trie[key], callback, path + [key], *args)
