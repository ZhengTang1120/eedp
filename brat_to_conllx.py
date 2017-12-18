#!/usr/bin/env python

import os
import glob
import argparse
from collections import namedtuple, defaultdict
from nltk import sent_tokenize, word_tokenize, pos_tag
from utils import *

TextboundMention = namedtuple('TextboundMention', 'id label start end text')
EventMention = namedtuple('EventMention', 'id label trigger arguments')

def brat_to_conllx(text, annotations):
    root = ConllEntry(0, '*root*', '_', '_', '*root*', '_', -1, 'rroot', '_', '_')
    annotations = list(parse_annotations(annotations))
    for words, starts, ends in get_token_spans(text):
        conllx = [root]
        tags = [t for w,t in pos_tag(words)]
        try:
            for i in range(len(words)):
                tbm = get_tbm(annotations, starts[i], ends[i])
                label = '_' if tbm is None else tbm.label
                rel, head = get_relhead(annotations, starts, ends, tbm, i)
                entry = ConllEntry(i + 1, words[i], '_', '_', tags[i], label, head, rel, '_', '_')
                conllx.append(entry)
        except ValueError:
            print('Tokenization does not align. Skipping...')
            continue
        yield conllx

def get_tbm(annotations, start, end):
    """returns the corresponding textbound mention"""
    for a in annotations:
        if a.id.startswith('T') and a.start <= start and a.end >= end:
            return a

def get_mention_head(annotations, ends, mention_id):
    """returns the head token for the given mention id"""
    for a in annotations:
        if a.id == mention_id:
            i = ends.index(a.end)
            return i + 1

def get_relhead(annotations, starts, ends, tbm, tok):
    """returns the correct relation and head for the given textbound mention"""
    if tbm is not None:
        if tbm.end != ends[tok]:
            head = get_mention_head(annotations, ends, tbm.id)
            return 'multitoken', head
        for a in annotations:
            if a.id.startswith('E'):
                for rel, args in a.arguments.items():
                    for arg in args:
                        if arg == tbm.id:
                            head = get_mention_head(annotations, ends, a.trigger)
                            return rel, head
    return '_', '_'

def parse_annotations(annotations):
    for line in annotations.splitlines():
        if line.startswith('T'):
            [id, data, text] = line.split('\t')
            [label, start, end] = data.split(' ')
            yield TextboundMention(id, label, int(start), int(end), text)
        elif line.startswith('E'):
            [id, data] = line.split('\t')
            [label_trigger, *args] = data.split(' ')
            [label, trigger] = label_trigger.split(':')
            arguments = defaultdict(list)
            for a in args:
                if a.strip() != '':
                    [name, arg] = a.split(':')
                    arguments[name].append(arg)
            yield EventMention(id, label, trigger, dict(arguments))

def get_token_spans(text):
    """
    returns (words, start_offsets, end_offsets)
    for each sentence in the provided text
    """
    offset = 0
    for s in sent_tokenize(text):
        offset = text.find(s, offset)
        yield sentence_tokens(s, offset)

def sentence_tokens(sentence, offset):
    """this is meant to be used by get_token_spans() only"""
    pos = 0
    starts = []
    ends = []
    words = word_tokenize(sentence)
    for w in words:
        pos = sentence.find(w, pos)
        starts.append(pos + offset)
        pos += len(w)
        ends.append(pos + offset)
    return words, starts, ends

def read(filename):
    with open(filename) as f:
        return f.read()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    parser.add_argument('--outfile', default='brat.conllx')
    args = parser.parse_args()

    sentences = []
    for fname in glob.glob(os.path.join(args.datadir, '*.a1')):
        print(fname)
        root = os.path.splitext(fname)[0]
        txt = read(root + '.txt')
        a1 = read(root + '.a1')
        a2 = read(root + '.a2')
        ann = f'{a1}\n{a2}'
        sentences += brat_to_conllx(txt, ann)
    write_conllx(args.outfile, sentences)
        
