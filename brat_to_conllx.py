#!/usr/bin/env python

import os
import glob
import argparse
from collections import namedtuple, defaultdict
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
from processors import *
from utils import *

# brat mentions
TextboundMention = namedtuple('TextboundMention', 'id label start end text')
EventMention = namedtuple('EventMention', 'id label trigger arguments')

# global variables to keep statistics that are printed at the end
total_sentences = 0
skipped_sentences = 0

#initial tokenizer
tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')
API = ProcessorsAPI(port=8886)

def sent_tokenize(text):
    doc = API.bionlp.annotate(text)
    for sent in doc.sentences:
        s = sent.startOffsets[0]
        e = sent.endOffsets[-1]
        sentence = text[s:e]
        yield sentence

def brat_to_conllx(text, annotations):
    """
    gets an annotation corresponding to a single paper
    and returns a sequence of sentences formatted as conllx
    """
    global total_sentences, skipped_sentences
    root = ConllEntry(id=0, form='*root*', postag='*root*', head=-1, deprel='rroot')
    skipped = 0
    for words, starts, ends in get_token_spans(text):
        total_sentences += 1
        conllx = [root]
        tags = [t for w,t in pos_tag(words)]
        try:
            for i in range(len(words)):
                tbm = get_tbm(annotations, starts[i], ends[i])
                label = None if tbm is None else tbm.label
                if label is None:
                    label = 'O'
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
        yield conllx #make_projective(conllx)

def make_projective(entries):
    num_dependents = defaultdict(int)
    for e in entries:
        num_dependents[e.parent_id] += 1
    for e in entries:
        if e.parent_id == 0 and num_dependents[e.id] == 0:
            e.parent_id = e.head = -1
            e.relation = e.deprel = 'none'
            e.brat_label = e.feats = None
    return entries

def get_tbm(annotations, start, end):
    """returns the corresponding textbound mention"""
    for a in annotations:
        if a.id.startswith('T') and a.start <= end and start <= a.end:
            return a

def get_mention_head(annotations, ends, mention_id):
    """returns the head token for the given mention id"""
    for a in annotations:
        if a.id == mention_id:
            try:
                i = ends.index(a.end)
                return i + 1
            except ValueError:
                return 0

def get_relhead(annotations, starts, ends, tbm, tok):
    """returns the correct relation and head for the given textbound mention"""
    # it the token does not belong to a textbound mention then it should be dropped
    if tbm is None:
        return 'none', 0
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
    relheads = []
    for a in annotations:
        if a.id.startswith('E'):
            for rel, args in a.arguments.items():
                for arg in args:
                    if arg == mention_id:
                        head = get_mention_head(annotations, ends, a.trigger)
                        # collapse theme1, theme2, etc. into theme
                        rel = rel[:-1] if rel[-1].isdigit() else rel
                        if head != 0:
                            relheads.append((rel, head))
    if relheads:
        return relheads[-1]
    # if token has no parent then point it to the root
    return 'root', 0

def parse_a1(annotations):
    for line in annotations.splitlines():
        if line.startswith('T'):
            [id, data, text] = line.split('\t')
            label = data.split(' ')[0]
            intervals = ' '.join(data.split(' ')[1:])
            for se in intervals.split(';'):
                [s, e] = se.split(' ')
                start = int(s)
                end = int(e)
            yield TextboundMention(id, label, start, end, text)

def parse_a2(annotations, a1):
    res = list()
    for line in annotations.splitlines():
        if line.startswith('E'):
            [id, data] = line.split('\t')
            [label, *args] = data.split(' ')
            [label2, trigger] = args[0].split(':')
            for i, a in enumerate(a1):
                if trigger == a.id:
                    a1[i] = TextboundMention(a.id, label, a.start, a.end, a.text)
            arguments = defaultdict(list)
            for a in args[1:]:
                if a.strip() != '':
                    [name, arg] = a.split(':')
                    arguments[name].append(arg)
            res.append(EventMention(id, label, trigger, dict(arguments)))
        # Negation
        else:
            for i,a in enumerate(res):
                [id, data] = line.split('\t')
                [_, arg] = data.split(':')
                if a.id == arg:
                    res[i] = EventMention(a.id, "Negative#"+a.label, a.trigger, a.arguments)
                    for j, b in enumerate(a1):
                        if a.trigger == b.id:
                            a1[j] = TextboundMention(b.id, "Negative#"+a.label, b.start, b.end, b.text)
    return res, a1

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
    words = tokenizer.tokenize(sentence)
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
        a1 = list(parse_a1(f'{a1}'))
        a2, a1 = parse_a2(f'{a2}', a1)
        annotations = a2 + a1
        print (len(annotations))
        for a in annotations:
            if a.id.startswith('E'):
                print (a.arguments)
    #     sentences += brat_to_conllx(txt, annotations)
    
    # print('---')
    # print(f'{total_sentences:,} sentences')
    # print(f'{skipped_sentences:,} skipped')
    # print(f'{total_sentences-skipped_sentences:,} remaining')
    # print(f'writing {args.outfile}')
    # write_conllx(args.outfile, sentences)
