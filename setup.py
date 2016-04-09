# -*- coding: utf-8 -*-
from codecs import open
import os
import re
from setuptools import setup

with open(os.path.join('rakutenma', '__init__.py'), 'r', encoding='utf8') as f:
    version = re.compile(
        r'.*__version__ = "(.*?)"', re.S).match(f.read()).group(1)

setup(
    name='rakutenma',
    packages=['rakutenma'],
    version=version,
    license='Apache Software License',
    platforms=['POSIX', 'Windows', 'Unix', 'MacOS'],
    description='morphological analyzer (word segmentor + PoS Tagger) for Chinese and Japanese',
    author='Yukino Ikegami',
    author_email='yukino0131@me.com',
    url='https://github.com/ikegami-yukino/rakutenma-python',
    keywords=['morphological analyzer', 'word segmentor', 'PoS Tagger'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: Japanese',
        'Natural Language :: Chinese (Simplified)',
        'Natural Language :: Chinese (Traditional)',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Text Processing :: Linguistic'
        ],
    long_description='%s\n\n%s' % (open('README.rst', encoding='utf8').read(),
                                   open('CHANGES.rst', encoding='utf8').read()),
    package_data={'rakutenma': ['model/*.json']},
    scripts=['bin/rakutenma'],
)
