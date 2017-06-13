#! /usr/bin/env python
from __future__ import print_function
import sys

threshold = int(sys.argv[1])

for line in sys.stdin:
    k, c = line.split()
    c = int(c)
    if c >= threshold: sys.stdout.write(line)
