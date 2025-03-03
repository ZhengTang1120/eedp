#!/usr/bin/env python

import argparse
from utils import *
from parsers import ArcHybridParser

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('infile')
    parser.add_argument('outfile')
    parser.add_argument('--sentence', action='store_true')
    args = parser.parse_args()
    event = not args.sentence
    # print('loading ...')
    parser = ArcHybridParser.load(args.model)
    sentences = read_conllx(args.infile, non_proj=True)

    # print('parsing ...')
    for i, s in enumerate(sentences):
        # print('parsing sentence', i)
        if len(s) > 2:
            if event:
                parser.parse_event(s)
            else:
                parser.parse_sentence(s)

    # print('writing output ...')
    for s in sentences:
        if len(s) > 2:
            for e in s:
                if e.id > 0:
                    e.head = e.parent_id = e.pred_parent_id
                    e.deprel = e.relation = e.pred_relation
                    if event:
                        e.feats = e.brat_label = e.pred_feats

    write_conllx(args.outfile, sentences)
