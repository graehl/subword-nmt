#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import mock
import re

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from apply_bpe import BPE

class TestBPEIsolateGlossariesMethod(unittest.TestCase):

    def setUp(self):

        amock = mock.MagicMock()
        amock.readline.return_value = 'something'
        glossaries = ['like', 'Manuel', 'USA']
        self.bpe = BPE(amock, glossaries=glossaries)

    def _run_test_case(self, test_case):
        orig, expected = test_case
        out = self.bpe._isolate_glossaries(orig)
        self.assertEqual(out, expected)

    def test_multiple_glossaries(self):
        orig = 'wordlikeUSAwordManuelManuelwordUSA'
        exp = ['word', 'like', '', 'USA', 'word', 'Manuel', '', 'Manuel', 'word', 'USA', '']
        test_case = (orig, exp)
        self._run_test_case(test_case)

class TestBPESegmentMethod(unittest.TestCase):

    def setUp(self):

        amock = mock.MagicMock()
        amock.readline.return_value = 'something'
        glossaries = ['like', 'Manuel', 'USA']
        self.bpe = BPE(amock, glossaries=glossaries)

    def _run_test_case(self, test_case):
        orig, expected = test_case
        out = self.bpe.segment(orig)
        self.assertEqual(out, expected)

    def test_multiple_glossaries(self):
        orig = 'wordlikeword likeManuelword'
        exp = 'w@@ o@@ r@@ d@@ like@@ w@@ o@@ r@@ d l@@ i@@ k@@ e@@ M@@ a@@ n@@ u@@ e@@ l@@ word'
        test_case = (orig, exp)
        self._run_test_case(test_case)

if __name__ == '__main__':
    unittest.main()
