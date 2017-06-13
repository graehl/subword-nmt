#! /usr/bin/env python
from __future__ import print_function
import sys
from collections import Counter

c = Counter()

for line in sys.stdin:
    for word in line.split():
        if len(word):
            c[word] += 1

for key,f in c.most_common():
    print(key+" "+ str(f))
