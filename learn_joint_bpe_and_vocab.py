#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich

"""Use byte pair encoding (BPE) to learn a variable-length encoding of the vocabulary in a text.
This script learns BPE jointly on a concatenation of a list of texts (typically the source and target side of a parallel corpus,
applies the learned operation to each and (optionally) returns the resulting vocabulary of each text.
The vocabulary can be used in apply_bpe.py to avoid producing symbols that are rare or OOV in a training text.

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2016). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""

from __future__ import unicode_literals

import sys
import codecs
import argparse
from collections import Counter

import learn_bpe
import apply_bpe

# hack for python2/3 compatibility
from io import open
argparse.open = open

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), required=True, nargs = '+',
        metavar='PATH',
        help="Input texts (multiple allowed).")
    learn_bpe.common_parser_arguments(parser)
    return parser


if __name__ == '__main__':

    # python 2/3 compatibility
    if sys.version_info < (3, 0):
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
    else:
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)

    parser = create_parser()
    args = parser.parse_args()

    if args.vocab and len(args.input) != len(args.vocab):
        sys.stderr.write('Error: number of input files and vocabulary files must match\n')
        sys.exit(1)

    # read/write files as UTF-8
    args.input = [codecs.open(f.name, encoding='UTF-8') for f in args.input]
    args.vocab = [codecs.open(f.name, 'w', encoding='UTF-8') for f in args.vocab]

    # get combined vocabulary of all input texts
    full_vocab = Counter()
    vocabs = []
    for f in args.input:
        v = learn_bpe.get_vocabulary(f, args.dict_input, args.mincount)
        vocabs.append(v)
        full_vocab += v
        f.seek(0)

    # learn BPE on combined vocabulary
    with codecs.open(args.output.name, 'w', encoding='UTF-8') as output:
        learn_bpe.main_args(args, full_vocab, output, is_dict=True)

    with codecs.open(args.output.name, encoding='UTF-8') as codes:
        bpe = apply_bpe.BPE(codes, args.separator, None)
        # apply BPE to each training corpus and get vocabulary
        learn_bpe.make_vocabularies(bpe, vocabs, args.vocab)
