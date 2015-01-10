Rakuten MA Python
===================

.. image:: https://badge.fury.io/py/rakutenma.svg
  :target: http://badge.fury.io/py/rakutenma
.. image:: https://travis-ci.org/ikegami-yukino/rakutenma-python.svg?branch=master
  :target: https://travis-ci.org/ikegami-yukino/rakutenma-python
.. image:: https://coveralls.io/repos/ikegami-yukino/rakutenma-python/badge.png?branch=master
  :target: https://coveralls.io/r/ikegami-yukino/rakutenma-python?branch=master


Rakuten MA Python (morphological analyzer) is a Python version of Rakuten MA (word segmentor + PoS Tagger) for Chinese and Japanese.

For details about Rakuten MA, See https://github.com/rakuten-nlp/rakutenma

Contributions are welcome!


Installation
==============

::

 pip install rakutenma

Example
===========

::

 from rakutenma import RakutenMA

 # Initialize a RakutenMA instance with an empty model
 # the default ja feature set is set already
 rma = RakutenMA()

 # Let's analyze a sample sentence (from http://tatoeba.org/jpn/sentences/show/103809)
 # With a disastrous result, since the model is empty!
 print(rma.tokenize("彼は新しい仕事できっと成功するだろう。"))

 # Feed the model with ten sample sentences from tatoeba.com
 # "tatoeba.json" is available at https://github.com/rakuten-nlp/rakutenma
 import json
 tatoeba = json.load(open("tatoeba.json"))
 for i in tatoeba:
     rma.train_one(i)

 # Now what does the result look like?
 print(rma.tokenize("彼は新しい仕事できっと成功するだろう。"))

 # Initialize a RakutenMA instance with a pre-trained model
 # "model_ja.json" is available at https://github.com/rakuten-nlp/rakutenma
 rma = RakutenMA(phi=1024, c=0.007812)  # Specify hyperparameter for SCW (for demonstration purpose)
 rma.load("model_ja.json")

 # Set the feature hash function (15bit)
 rma.hash_func = rma.create_hash_func(15)

 # Tokenize one sample sentence
 print(rma.tokenize("うらにわにはにわにわとりがいる"));

 # Re-train the model feeding the right answer (pairs of [token, PoS tag])
 res = rma.train_one(
        [["うらにわ","N-nc"],
         ["に","P-k"],
         ["は","P-rj"],
         ["にわ","N-n"],
         ["にわとり","N-nc"],
         ["が","P-k"],
         ["いる","V-c"]])
 # The result of train_one contains:
 #   sys: the system output (using the current model)
 #   ans: answer fed by the user
 #   update: whether the model was updated
 print(res)

 # Now what does the result look like?
 print(rma.tokenize("うらにわにはにわにわとりがいる"))


NOTE
===========

Added API
--------------
As compared to original RakutenMA, following methods are added:

- RakutenMA::load(model_path)
  - Load model from JSON file

- RakutenMA::save(model_path)
  - Save model to path

misc
--------------
As initial setting, following values are set:

- rma.featset = CTYPE_JA_PATTERNS  # RakutenMA.default_featset_ja
- rma.hash_func = rma.create_hash_func(15)

LICENSE
=========

Apache License version 2.0


Copyright
=============

Rakuten MA Python
(c) 2015- Yukino Ikegami. All Rights Reserved.

Rakuten MA (original)
(c) 2014 Rakuten NLP Project. All Rights Reserved.


