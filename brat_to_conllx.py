#!/usr/bin/env python

import os
import glob
import argparse
from collections import namedtuple, defaultdict
from nltk import sent_tokenize, word_tokenize, pos_tag
from utils import *

# brat mentions
TextboundMention = namedtuple('TextboundMention', 'id label start end text')
EventMention = namedtuple('EventMention', 'id label trigger arguments')

# global variables to keep statistics that are printed at the end
total_sentences = 0
skipped_sentences = 0

def brat_to_conllx(text, annotations):
    """
    gets an annotation corresponding to a single paper
    and returns a sequence of sentences formatted as conllx
    """
    global total_sentences, skipped_sentences
    root = ConllEntry(id=0, form='*root*', postag='*root*', head=-1, deprel='rroot')
    annotations = list(parse_annotations(annotations))
    skipped = 0
    for words, starts, ends in get_token_spans(text):
        total_sentences += 1
        conllx = [root]
        tags = [t for w,t in pos_tag(words)]
        try:
            for i in range(len(words)):
                tbm = get_tbm(annotations, starts[i], ends[i])
                label = '_' if tbm is None else tbm.label
                rel, head = get_relhead(annotations, starts, ends, tbm, i)
                entry = ConllEntry(id=i+1, form=words[i], postag=tags[i], feats=label, head=head, deprel=rel)
                conllx.append(entry)
        except ValueError:
            # get_mention_head() searches for the token's end position
            # in the `ends` list that corresponds to the sentence's tokens,
            # and throws an exception if the provided end does not correspond to any token
            print('ERROR: tokenization does not align')
            skipped_sentences += 1
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
    # it the token does not belong to a textbound mention then it should be dropped
    if tbm is None:
        return 'none', -1
    # if mention is multitoken, all tokens should point to the mention head
    if tbm.end != ends[tok]:
        head = get_mention_head(annotations, ends, tbm.id)
        return 'multitoken', head
    # if the mention is a trigger, then use the event id
    mention_id = tbm.id
    for a in annotations:
        if a.id.startswith('E') and a.trigger == mention_id:
            mention_id = a.id
    # if mention is involved in an event, point to the event trigger
    for a in annotations:
        if a.id.startswith('E'):
            for rel, args in a.arguments.items():
                for arg in args:
                    if arg == mention_id:
                        head = get_mention_head(annotations, ends, a.trigger)
                        return rel, head
    # if token has no parent then point it to the root
    return 'root', 0

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
    # search for *.a1 instead of *.txt because sometimes
    # there are other text files (e.g. readme.txt)
    for fname in glob.glob(os.path.join(args.datadir, '*.a1')):
        root = os.path.splitext(fname)[0]
        name = os.path.basename(root)
        print(f'reading {name}')
        txt = read(root + '.txt')
        a1 = read(root + '.a1')
        a2 = read(root + '.a2')
        annotations = f'{a1}\n{a2}'
        sentences += brat_to_conllx(txt, annotations)

    print('---')
    print(f'{total_sentences:,} sentences')
    print(f'{skipped_sentences:,} skipped')
    print(f'{total_sentences-skipped_sentences:,} remaining')
    print(f'writing {args.outfile}')
    write_conllx(args.outfile, sentences)
