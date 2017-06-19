#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich

"""Use byte pair encoding (BPE) to learn a variable-length encoding of the vocabulary in a text.
Unlike the original BPE, it does not compress the plain text, but can be used to reduce the vocabulary
of a text to a configurable number of symbols, with only a small increase in the number of tokens.

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2016). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""

from __future__ import unicode_literals

import sys
import codecs
import re
import copy
import argparse
import os
import tempfile
from collections import defaultdict, Counter

# hack for python2/3 compatibility
from io import open
argparse.open = open

import apply_bpe

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input text (default: standard input).")

    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output file for BPE codes (default: standard output)")
    parser.add_argument('--forcecodes', '-f', default=None, metavar='PATH',
                           help='apply these merges (--output from another learn_bpe.py) first')
    parser.add_argument('--grepforcecodes', '-g', default=None, metavar='RE',
                           help="use only --forcecodes A B parts that both whole-string match this regexp"
                                "(not counting any </w> at end of B which is always allowed) e.g. [0-9]+")
    parser.add_argument(
        '--symbols', '-s', type=int, default=10000,
        help="Create this many new symbols (each representing a character n-gram) (default: %(default)s))")
    parser.add_argument('--version01', '-1', action="store_true",
                            help="learn a #version: 0.1 model (last char of word isn't forced to merge with end-of-word)")
    parser.add_argument(
        '--min-frequency', type=int, default=2, metavar='FREQ',
        help='Stop if no symbol pair has frequency >= FREQ (default: %(default)s))')
    parser.add_argument('--dict-input', action="store_true",
        help="If set, input file is interpreted as a dictionary where each line contains a word-count pair")
    parser.add_argument(
        '--min-count,', '-c', type=int, dest='mincount', default=1, help="drop from pre-bpe vocab any word with count below this")
    parser.add_argument(
        '--write-vocabulary', '-w', type=argparse.FileType('w'), nargs = '+', default=None,
        metavar='PATH', dest='vocab',
        help='Write to these vocabulary files after applying BPE. One per input text. Used for filtering in apply_bpe.py')
    parser.add_argument(
        '--separator', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")
    parser.add_argument(
        '--verbose', '-v', action="store_true",
        help="verbose mode.")

    return parser


def get_vocabulary(fobj, is_dict=False, mincount=1):
    """Read text and return dictionary that encodes vocabulary
    """
    vocab = Counter()
    for line in fobj:
        if is_dict:
            word, count = line.strip().split()
            c = int(count)
            if c < mincount:
                break
            vocab[word] = c
        else:
            for word in line.split():
                vocab[word] += 1
    if mincount > 1 and not is_dict:
        return dict((x,y) for x,y in vocab.items() if y >= mincount)
    return vocab


def write_vocabulary(vocab, vocab_file):
    for key, freq in sorted(vocab.items(), key=lambda x: x[1], reverse=True):
        vocab_file.write("{0} {1}\n".format(key, freq))


def restricted_vocabulary(bpe, restrict):
    vocab = restrict if isinstance(restrict, Counter) else get_vocabulary(restrict)
    bpevocab = Counter()
    for w, c in vocab.items():
        for sw in bpe.pieces(w):
            bpevocab[sw] += c
    return bpevocab


def make_vocabularies(bpe, inputfiles, vocabfiles):
    for restrict, outf in zip(inputfiles, vocabfiles):
        write_vocabulary(restricted_vocabulary(bpe, restrict), outf)
        outf.close()


def update_pair_statistics(pair, changed, stats, indices):
    """Minimally update the indices and frequency of symbol pairs

    if we merge a pair of symbols, only pairs that overlap with occurrences
    of this pair are affected, and need to be updated.
    """
    stats[pair] = 0
    indices[pair] = defaultdict(int)
    first, second = pair
    new_pair = first+second
    for j, word, old_word, freq in changed:

        # find all instances of pair, and update frequency/indices around it
        i = 0
        while True:
            # find first symbol
            try:
                i = old_word.index(first, i)
            except ValueError:
                break
            # if first symbol is followed by second symbol, we've found an occurrence of pair (old_word[i:i+2])
            if i < len(old_word)-1 and old_word[i+1] == second:
                # assuming a symbol sequence "A B C", if "B C" is merged, reduce the frequency of "A B"
                if i:
                    prev = old_word[i-1:i+1]
                    stats[prev] -= freq
                    indices[prev][j] -= 1
                if i < len(old_word)-2:
                    # assuming a symbol sequence "A B C B", if "B C" is merged, reduce the frequency of "C B".
                    # however, skip this if the sequence is A B C B C, because the frequency of "C B" will be reduced by the previous code block
                    if old_word[i+2] != first or i >= len(old_word)-3 or old_word[i+3] != second:
                        nex = old_word[i+1:i+3]
                        stats[nex] -= freq
                        indices[nex][j] -= 1
                i += 2
            else:
                i += 1

        i = 0
        while True:
            try:
                # find new pair
                i = word.index(new_pair, i)
            except ValueError:
                break
            # assuming a symbol sequence "A BC D", if "B C" is merged, increase the frequency of "A BC"
            if i:
                prev = word[i-1:i+1]
                stats[prev] += freq
                indices[prev][j] += 1
            # assuming a symbol sequence "A BC B", if "B C" is merged, increase the frequency of "BC B"
            # however, if the sequence is A BC BC, skip this step because the count of "BC BC" will be incremented by the previous code block
            if i < len(word)-1 and word[i+1] != new_pair:
                nex = word[i:i+2]
                stats[nex] += freq
                indices[nex][j] += 1
            i += 1


def get_pair_statistics(vocab):
    """Count frequency of all symbol pairs, and create index"""

    # data structure of pair frequencies
    stats = defaultdict(int)

    #index from pairs to words
    indices = defaultdict(lambda: defaultdict(int))

    for i, (word, freq) in enumerate(vocab):
        prev_char = word[0]
        for char in word[1:]:
            stats[prev_char, char] += freq
            indices[prev_char, char][i] += 1
            prev_char = char

    return stats, indices


def replace_pair(pair, vocab, indices):
    """Replace all occurrences of a symbol pair ('A', 'B') with a new symbol 'AB'"""
    first, second = pair
    pair_str = ''.join(pair)
    pair_str = pair_str.replace('\\','\\\\')
    changes = []
    pattern = re.compile(r'(?<!\S)' + re.escape(first + ' ' + second) + r'(?!\S)')
    if sys.version_info < (3, 0):
        iterator = indices[pair].iteritems()
    else:
        iterator = indices[pair].items()
    for j, freq in iterator:
        if freq < 1:
            continue
        word, freq = vocab[j]
        new_word = ' '.join(word)
        new_word = pattern.sub(pair_str, new_word)
        new_word = tuple(new_word.split())

        vocab[j] = (new_word, freq)
        changes.append((j, new_word, word, freq))

    return changes


def do_pair(most_frequent, outfile, sorted_vocab, indices, stats):
    outfile.write('{0} {1}\n'.format(*most_frequent))
    update_pair_statistics(most_frequent, replace_pair(most_frequent, sorted_vocab, indices), stats, indices)
    stats[most_frequent] = 0


def prune_stats(stats, big_stats, threshold):
    """Prune statistics dict for efficiency of max()

    The frequency of a symbol pair never increases, so pruning is generally safe
    (until the most frequent pair is less frequent than a pair we previously pruned)
    big_stats keeps full statistics for when we need to access pruned items
    """
    for item,freq in list(stats.items()):
        if freq < threshold:
            del stats[item]
            if freq < 0:
                big_stats[item] += freq
            else:
                big_stats[item] = freq


def main(infile, outfile, num_symbols, min_frequency=2, verbose=False, is_dict=False, version01=False, forcecodes=None, grepforcecodes=None, mincount=1):
    """Learn num_symbols BPE operations from vocabulary, and write to outfile.
    """

    # version 0.2 changes the handling of the end-of-word token ('</w>');
    # version numbering allows bckward compatibility
    apply_bpe.write_header(outfile, (0, 1 if version01 else 2))

    vocab = infile if isinstance(infile, Counter) else get_vocabulary(infile, is_dict)
    endword = '</w>'
    sorted_vocab = sorted([(tuple(x)+(endword,) if version01 else tuple(x[:-1])+(x[-1]+endword,) , y) for (x,y) in vocab.items()], key=lambda x: x[1], reverse=True)

    stats, indices = get_pair_statistics(sorted_vocab)
    big_stats = copy.deepcopy(stats)
    # threshold is inspired by Zipfian assumption, but should only affect speed
    threshold = max(stats.values()) / 10
    ncodes = 0
    if forcecodes is not None:
        grep = grepforcecodes
        def matchcode(pair, grep):
            return True if grep is None else grep.match(pair[0]) and grep.match(pair[1])
        if grep is not None:
            if isinstance(grep, str):
                if not grep.startswith('^'):
                    grep = r'^' + grep
                grep = grep + '(</w>)?'
                if not grep.endswith('$'):
                    grep = grep + r'$'
                sys.stderr.write("using only forcecodes lines matching r'%s' ...\n" % grep)
                grep = re.compile(grep)
        first = True
        stats = dict()
        for line in forcecodes:
            if not (first and line.startswith(versionheaderbegin)):
                a, b = line.strip().split()
                pair = (a, b)
                if matchcode(pair, grep):
                    if verbose and grep:
                        sys.stderr.write("grepforcecodes: %s %s\n" % pair)
                    do_pair(pair, outfile, sorted_vocab, indices, big_stats)
                    ncodes += 1
            first = False
        sys.stderr.write("forcecodes: added an additional %s --forcecodes\n (in addition to --num-symbols=%s)\n" % (ncodes, num_symbols))

    for i in range(num_symbols):
        if stats:
            most_frequent = max(stats, key=lambda x: (stats[x], x))

        # we probably missed the best pair because of pruning; go back to full statistics
        if not stats or (i and stats[most_frequent] < threshold):
            prune_stats(stats, big_stats, threshold)
            stats = copy.deepcopy(big_stats)
            most_frequent = max(stats, key=lambda x: (stats[x], x))
            # threshold is inspired by Zipfian assumption, but should only affect speed
            threshold = stats[most_frequent] * i/(i+10000.0)
            prune_stats(stats, big_stats, threshold)

        if stats[most_frequent] < min_frequency:
            sys.stderr.write('no pair has frequency >= {0}. Stopping\n'.format(min_frequency))
            break

        if verbose:
            sys.stderr.write('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format(i, most_frequent[0], most_frequent[1], stats[most_frequent]))
        do_pair(most_frequent, outfile, sorted_vocab, indices, stats)
        ncodes += 1
        if not i % 100:
            prune_stats(stats, big_stats, threshold)
    sys.stderr.write("bpe codes has %s pairs\n" % (ncodes,))
    return vocab


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

    # read/write files as UTF-8
    if args.input.name != '<stdin>':
        args.input = codecs.open(args.input.name, encoding='utf-8')
    if args.output.name != '<stdout>':
        args.output = codecs.open(args.output.name, 'w', encoding='utf-8')

    forcecodes = None
    if args.forcecodes is not None:
        forcecodes = codecs.open(args.forcecodes, encoding='UTF-8')

    vocab = main(args.input, args.output, args.symbols, args.min_frequency, args.verbose, is_dict=args.dict_input, version01=args.version01, forcecodes=forcecodes, grepforcecodes=args.grepforcecodes, mincount=args.mincount)
    if len(args.vocab):
        with codecs.open(args.output.name, encoding='UTF-8') as codes:
            bpe = apply_bpe.BPE(codes, args.separator, None)
            # apply BPE to each training corpus and get vocabulary
            make_vocabularies(bpe, vocab, args.vocab)
