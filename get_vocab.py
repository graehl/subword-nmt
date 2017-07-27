#! /usr/bin/env python
from __future__ import print_function
import sys
import codecs
import unicodedata
import os
from collections import Counter
from unicodedata import normalize

# python 2/3 compatibility
if sys.version_info < (3, 0):
    sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
    sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
    sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
else:
    sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
    sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
    sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)

# NFK?[DC] K makes roman numeral I -> ascii I. C means composed, D means decomposed
if len(sys.argv) > 1:
    unicodenormal = sys.argv[1]
    if len(sys.argv) > 2:
        print("WARNING: ignoring extra args %s\n"%sys.argv[2:], file=sys.stderr)
else:
    unicodenormal = os.environ.get('unicodenormal', 'NFC')
c = Counter()

for line in sys.stdin:
    for word in line.split():
        if len(word):
            c[unicodedata.normalize(unicodenormal, word)] += 1

for key,f in c.most_common():
    print(key+" "+ str(f))
