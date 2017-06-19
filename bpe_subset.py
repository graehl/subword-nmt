#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Jonathan Graehl

"""When learning a BPE codes for two languages, create BPE-subset vocabularies for apply_bpe.py restriction

(learn_joint_bpe_and_vocab.py will already do this for you)

Optionally create separate bpe codes file pre-restricted.

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2016). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""

import argparse
import sys
import codecs
import learn_bpe
import apply_bpe
from collections import Counter

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="create BPE-segmented vocabulary subset")
    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="running or vocab text input file (default: standard input).")
    parser.add_argument(
        '--input-is-vocab', '-v', type=bool, dest='isvocab', default=True)
    parser.add_argument(
        '--codes', '-c', type=argparse.FileType('r'), metavar='PATH',
        required=True,
        help="File with BPE codes (created by learn_bpe.py).")
    parser.add_argument(
        '--min-count,', '-m', type=int, dest='mincount', default=1, help="drop from pre-bpe vocab any word with count below this")
    parser.add_argument(
        '--separator', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")
    parser.add_argument(
        '--outcodes', '-o', type=argparse.FileType('w'), metavar='PATH', help="output vocabulary restricted bpe codes subset to this file")
    parser.add_argument('--bpevocab', '-b', type=argparse.FileType('w'), metavar='PATH', help="output bpe vocab (default: standard output")
    return parser

def main(args):
    vocab = learn_bpe.get_vocabulary(args.input, args.isvocab, args.mincount)
    assert isinstance(vocab, Counter)
    bpe = apply_bpe.BPE(args.codes, args.separator, None)
    bpevocab = learn_bpe.restricted_vocabulary(bpe, vocab)
    if args.bpevocab is not None: learn_bpe.write_vocabulary(bpevocab, args.bpevocab)
    if args.outcodes is not None: bpe.write_subset(args.outcodes, bpevocab)

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
    main(parser.parse_args())
