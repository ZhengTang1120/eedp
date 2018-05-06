#!/usr/bin/env python

import os
import glob
import argparse
from collections import namedtuple, defaultdict
from processors import *
from utils import *

# brat mentions
TextboundMention = namedtuple('TextboundMention', 'id label start end text')
EventMention = namedtuple('EventMention', 'id label trigger arguments')

# global variables to keep statistics that are printed at the end
total_sentences = 0
skipped_sentences = 0

API = ProcessorsBaseAPI(hostname="128.196.142.36", port=8881)
#API = ProcessorsBaseAPI(port=8888)

def brat_to_conllx(text, annotations):
    """
    gets an annotation corresponding to a single paper
    and returns a sequence of sentences formatted as conllx
    """
    global total_sentences, skipped_sentences
    root = ConllEntry(id=0, form='*root*', postag='*root*', head=-1, deprel='rroot')
    annotations = parse_annotations(annotations)
    skipped = 0
    doc = API.clu.bio.annotate(text)
    for words, starts, ends, tags in get_token_spans(doc):
        total_sentences += 1
        conllx = [root]
        try:
            for i in range(len(words)):
                tbm = get_tbm(annotations, starts[i], ends[i])
                label = None if tbm is None else tbm.label
                if label is None:
                    label = 'O'
                rels = list()
                heads = list()
                for rel, head, hlabel in get_relhead(annotations, starts, ends, tbm, i):
                    if head not in heads:
                        if head != i+1:
                            rels.append(rel)
                            heads.append(head)
                if len(heads) == 0:
                    heads = [0]
                    rels = ['root']
                entry = ConllEntry(id=i+1, form=words[i], postag=tags[i], feats=label, head=heads, deprel=rels)
                conllx.append(entry)
        except Exception as e:
            # get_mention_head() searches for the token's end position
            # in the `ends` list that corresponds to the sentence's tokens,
            # and throws an exception if the provided end does not correspond to any token
            # print('ERROR: tokenization does not align')
            print(e)
            print (text[ends[-3]:ends[-1]])
            print(words)
            print (ends)
            skipped_sentences += 1
            yield [ConllEntry(id=1, form='*skipped*', postag='*skipped*', head=-1, deprel='skipped', feats='O')]
            continue
        yield conllx


def get_tbm(annotations, start, end):
    al = list()
    """returns the corresponding textbound mention"""
    for a in annotations:
        if a.id.startswith('T') and a.start < end and start < a.end:
            al.append(a)
    if len(al) > 0:
        res = None
        for a in al:
            if "egulation" not in a:
                res = a
        if res == None:
            return al[-1]
        else:
            return res
    else:
        return None

def get_mention_head(annotations, ends, mention_id):
    """returns the head token for the given mention id"""
    for a in annotations:
        if a.id == mention_id:
            try:
                i = None
                for j in range(len(ends)):
                    if ends[j] >= a.end:
                        i = j
                        break
                assert i is not None
                return (i + 1, a.label)
            except:
                raise Exception(a)

def get_relhead(annotations, starts, ends, tbm, tok):
    """returns the correct relation and head for the given textbound mention"""
    # it the token does not belong to a textbound mention then it should be dropped
    if tbm is None:
        return [('none', -1, None)]
    # if mention is multitoken, all tokens should point to the mention head
    if tbm.end != ends[tok]:
        head, hlabel = get_mention_head(annotations, ends, tbm.id)
        # get_mention_head returns the one-based word index
        # but tok is zero-based, which produces some subtle errors
        if head == tok + 1:
            return [('root', 0, None)]
        else:
            return [('multitoken', head, None)]
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
                        if checkTrigger(a.trigger, starts, ends, annotations):
                            head, hlabel = get_mention_head(annotations, ends, a.trigger)
                            # collapse theme1, theme2, etc. into theme
                            rel = rel[:-1] if rel[-1].isdigit() else rel
                            if (rel, head, hlabel) not in relheads:
                                relheads.append((rel, head, hlabel))
    if relheads:
        return relheads
    # if token has no parent then point it to the root
    return [('root', 0, None)]

def checkTrigger(trigger, starts, ends, annotations):
    for a in annotations:
        if a.id == trigger:
            if a.start>=starts[0] and a.end<=ends[-1]:
                return True
            else:
                return False
    return False

def parse_annotations(annotations):
    ann_dict = dict()
    res = list()
    for line in annotations.splitlines():
        if line.startswith('T'):
            [id, data, text] = line.split('\t')
            [label, start, end] = data.split(' ')
            # if start+end not in ann_dict:
            #     ann_dict[start+end] = TextboundMention(id, [label], int(start), int(end), text)
            # else:
            #     ann_dict[start+end].label.append(label)
            res.append(TextboundMention(id, label, int(start), int(end), text))
        elif line.startswith('E'):
            [id, data] = line.split('\t')
            [label_trigger, *args] = data.split(' ')
            [label, trigger] = label_trigger.split(':')
            arguments = defaultdict(list)
            for a in args:
                if a.strip() != '':
                    [name, arg] = a.split(':')
                    arguments[name].append(arg)
            res.append(EventMention(id, label, trigger, dict(arguments)))
    res += ann_dict.values()
    return res

def get_token_spans(doc):
    """
    returns (words, start_offsets, end_offsets)
    for each sentence in the provided text
    """
    offset = 0
    for s in doc.sentences:
        yield s.words, s.startOffsets, s.endOffsets, s.tags


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