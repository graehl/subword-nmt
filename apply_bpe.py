#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich

"""Use operations learned with learn_bpe.py to encode a new text.
The text will not be smaller, but use only a fixed vocabulary, with rare words
encoded as variable-length sequences of subword units.

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2015). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""

from __future__ import unicode_literals, division

import sys
import codecs
import argparse
import json
import re
from collections import defaultdict

# hack for python2/3 compatibility
from io import open
argparse.open = open

def unicodeutf8(s):
    return unicode(s, 'utf8') if type(s)==str else s

def common_parser_arguments(parser):
    parser.add_argument('--unkchar', type=unicodeutf8,
                            default=u'\uFDEA', metavar='utf8',
                            help="a unicode (utf8) codepoint character that will never participate in merges. default is U+FDEA (hex), a private noncharacter")
    parser.add_argument(
        '--separator', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")
    parser.add_argument('--unktag', type=str, default='<unk>', help='replace unkchar with this (utf8)')

endword='</w>'

verbose=0

def log(s, out=sys.stderr):
    out.write("### %s\n" % (s,))

def logv(v, s, out=sys.stderr):
    if verbose >= v: log(s, out)

def written(x, sep=''):
    return x[:-4] if x.endswith(endword) else x + sep

def stripend(x):
    return x[:-4] if x.endswith(endword) else x


versionheaderbegin = '#version: '


def write_pair(pair, out):
    out.write("%s %s\n"%pair)


def write2(a, b, out):
    out.write("%s %s\n"%(a, b))


def write_header(outfile, version):
    outfile.write('%s%s.%s\n' % (versionheaderbegin, version[0], version[1]))


def version_line(line):
    return line.startswith(versionheaderbegin)


def maybe_header_version(line):
    if version_line(line):
        return tuple(int(x) for x in line[len(versionheaderbegin):].split("."))
    else:
        return None


class BPE(object):

    def __init__(self, codes, separator='@@', vocab=None, glossaries=None, rglossaries=None, unkchar=u'\uFDEA', unktag='<unk>'):

        # check version information
        firstline = codes.readline()
        self.version = maybe_header_version(firstline)
        if self.version is None:
            log("no version header in %s"%codes)
            self.version = (0, 1)
            codes.seek(0)
        log("version %s"%str(self.version))

        self.bpe_codes = [tuple(item.split()) for item in codes]

        # some hacking to deal with duplicates (only consider first instance)
        self.bpe_codes = dict([(code,i) for (i,code) in reversed(list(enumerate(self.bpe_codes)))])

        self.bpe_codes_reverse = dict([(pair[0] + pair[1], pair) for pair,i in self.bpe_codes.items()])

        self.separator = separator

        self.vocab = vocab

        self.unktag = unktag
        self.unkchar = unkchar

        self.glossaries = glossaries if glossaries else []
        self.rglossaries = rglossaries if rglossaries else []
        relist = [re.escape(x) for x in self.glossaries] + self.rglossaries
        if len(relist):
            retext = '(%s)' % '|'.join(relist)
            sys.stderr.write('glossaries re: %s\n' % retext)
            self.glossary_re = re.compile(retext)
        else:
            self.glossary_re = None

        self.cache = {}

    def ordered_codes(self):
        return sorted(self.bpe_codes.items(), key=lambda x: x[1])

    def prereqs(self, vocab, seen=None):
        if seen is None: seen=set()
        def prereqs2(s, pair):
            seen.add(s)
            prereqs(pair[0])
            prereqs(pair[1])
        def prereqs(s):
            if len(s) > 1 and s not in seen:
                seen.add(s)
                pair = self.bpe_codes_reverse.get(s, None)
                if pair is not None:
                    prereqs2(s, pair)
        for ab, pair in self.bpe_codes_reverse.items():
            if written(ab, self.separator) in vocab:
                prereqs2(ab, pair)
        return seen

    def write_subset(self, out, bpevocab, pre=None):
        """only include merges that are useful to reach vocab"""
        write_header(out, self.version)
        if pre is None: pre = self.prereqs(bpevocab)
        for pair,_ in self.ordered_codes():
            ab = pair[0] + pair[1]
            if written(ab, self.separator) in bpevocab or ab in pre:
                write_pair(pair, out)

    def segment(self, sentence):
        """segment single sentence (whitespace-tokenized string) with BPE encoding"""
        output = []
        for word in sentence.split():
            self.pieces(word, output)
        return ' '.join(output)

    def pieces(self, word, output=None):
        if output is None: output = []
        new_word = []
        isolated = False
        for segment in self._isolate_glossaries(word):
            if len(segment):
                if isolated:
                    new_word.append(segment)
                    sys.stderr.write('glossarized segment (leaving alone): "%s"\n' % segment)
                else:
                    new_word += encode(segment,
                                          self.bpe_codes,
                                          self.bpe_codes_reverse,
                                          self.vocab,
                                          self.separator,
                                          self.version,
                                          self.cache,
                                          unkchar=self.unkchar,
                                          unktag=self.unktag)
            isolated = not isolated
        remain = len(new_word)
        sep = self.separator
        for item in new_word:
            remain -= 1
            if remain == 0: sep = ''
            output.append(item + sep)
        return output

    def _isolate_glossaries(self, word):
        """
        Isolate a glossary present inside a word.

        Returns a list of subwords. In which all 'glossary' glossaries are isolated

        For example, if 'USA' is the glossary and '1934USABUSA' the word, the return value is:
            ['1934', 'USA', 'B', 'USA']
        """
        gre = self.glossary_re
        return [word] if gre is None else gre.split(word)


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    common_parser_arguments(parser)
    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input file (default: standard input).")
    parser.add_argument(
        '--codes', '-c', type=argparse.FileType('r'), metavar='PATH',
        required=True,
        help="File with BPE codes (created by learn_bpe.py).")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output file (default: standard output)")
    parser.add_argument(
        '--vocabulary', type=argparse.FileType('r'), default=None,
        metavar="PATH",
        help="Vocabulary file (built with get_vocab.py). If provided, split up subword units until they're in this vocabulary.")
    parser.add_argument(
        '--vocabulary-threshold', type=int, default=1,
        metavar="INT",
        help="Vocabulary threshold. If vocabulary is provided, any word with frequency < threshold will be treated as OOV")
    parser.add_argument(
        '--glossaries', type=str, nargs='+', default=None,
        metavar="STR",
        help="Glossaries. The strings provided in glossaries will not be affected"+
             "by the BPE (i.e. they will neither be broken into subwords, nor concatenated with other subwords")
    parser.add_argument(
        '--rglossaries', type=str, nargs='+', default=None,
        metavar="REGEX",
        help="Glossaries. The (python 're') regexes provided in glossaries will not be affected"+
             "by the BPE (i.e. they will neither be broken into subwords, nor concatenated with other subwords."+ "If glossaries/rglossaries are ambiguous, know that they form a single regexp (glossaries ..."+ "rglossaries) in that order, and are resolved by re.split (so probably winner is "+
             "earliest-in-string match with ties broken by earliest-in-list.")
    parser.add_argument('--verbose', '-v', type=int, default=0, help="higher = more ### stderr msgs")

    return parser


def get_pairs(word):
    """Return set of symbol pairs in a word.

    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def encode(orig, bpe_codes, bpe_codes_reverse, vocab, separator, version, cache, unkchar=u'U', unktag='<unk>'):
    """Encode word based on list of BPE merge operations, which are applied consecutively
    """

    if orig in cache:
        return cache[orig]

    if version == (0, 1):
        word = tuple(orig) + (endword,)
    elif version == (0, 2): # more consistent handling of word-final segments
        word = tuple(orig[:-1]) + ( orig[-1] + endword,)
    else:
        raise NotImplementedError

    pairs = get_pairs(word)

    if not pairs:
        return orig

    while True:
        bigram = min(pairs, key = lambda pair: bpe_codes.get(pair, float('inf')))
        if bigram not in bpe_codes:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            # replace bigram everywhere in word
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except:
                new_word.extend(word[i:])
                break

            if word[i] == first and i < len(word)-1 and word[i+1] == second:
                new_word.append(first+second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)

    # don't print end-of-word symbols
    if word[-1] == endword:
        word = word[:-1]
    elif word[-1].endswith(endword):
        word = word[:-1] + (word[-1].replace(endword,''),)

    if unktag and word == unkchar:
        word = unkword
    elif vocab:
        word = check_vocab_and_split(word, bpe_codes_reverse, vocab, separator)

    cache[orig] = word
    return word


def recursive_split(segment, bpe_codes, vocab, separator, final=False):
    """Recursively split segment into smaller units (by reversing BPE merges)
    until all units are either in-vocabulary, or cannot be split futher."""

    try:
        if final:
            left, right = bpe_codes[segment + endword]
            right = right[:-4]
        else:
            left, right = bpe_codes[segment]
    except:
        #sys.stderr.write('cannot split {0} further.\n'.format(segment))
        yield segment
        return

    if left + separator in vocab:
        yield left
    else:
        for item in recursive_split(left, bpe_codes, vocab, separator, False):
            yield item

    if (final and right in vocab) or (not final and right + separator in vocab):
        yield right
    else:
        for item in recursive_split(right, bpe_codes, vocab, separator, final):
            yield item


def check_vocab_and_split(orig, bpe_codes, vocab, separator):
    """Check for each segment in word if it is in-vocabulary,
    and segment OOV segments into smaller units by reversing the BPE merge operations"""

    out = []

    for segment in orig[:-1]:
        if segment + separator in vocab:
            out.append(segment)
        else:
            logv(1, 'OOV: {0}\n'.format(segment + separator))
            for item in recursive_split(segment, bpe_codes, vocab, separator, False):
                out.append(item)

    segment = orig[-1]
    if segment in vocab:
        out.append(segment)
    else:
        logv(1, 'final OOV: {0}\n'.format(segment))
        for item in recursive_split(segment, bpe_codes, vocab, separator, True):
            out.append(item)

    return out


def read_vocabulary_set(vocab_file, threshold=1):
    """read vocabulary file produced by get_vocab.py, and filter according to frequency threshold.
    """

    vocabulary = set()

    for line in vocab_file:
        word, freq = line.split()
        if int(freq) >= threshold:
            vocabulary.add(word)

    return vocabulary

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
    verbose = args.verbose

    # read/write files as UTF-8
    args.codes = codecs.open(args.codes.name, encoding='utf-8')
    if args.input.name != '<stdin>':
        args.input = codecs.open(args.input.name, encoding='utf-8')
    if args.output.name != '<stdout>':
        args.output = codecs.open(args.output.name, 'w', encoding='utf-8')
    if args.vocabulary:
        args.vocabulary = codecs.open(args.vocabulary.name, encoding='utf-8')

    if args.vocabulary:
        vocabulary = read_vocabulary_set(args.vocabulary, args.vocabulary_threshold)
    else:
        vocabulary = None

    bpe = BPE(args.codes, args.separator, vocabulary, args.glossaries, args.rglossaries, unkchar=args.unkchar, unktag=arcs.unktag)

    for line in args.input:
        args.output.write(bpe.segment(line).strip())
        args.output.write('\n')
