#!/usr/bin/env python

import argparse
from utils import *
from parsers import ArcHybridParser

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('infile')
    parser.add_argument('outfile')
    args = parser.parse_args()

    print('loading ...')
    parser = ArcHybridParser.load(args.model)
    sentences = read_conllx(args.infile, non_proj=True)

    print('parsing ...')
    for i, s in enumerate(sentences):
        print('parsing sentence', i)
        if len(s) > 2:
            parser.parse_sentence(s)

    print('writing output ...')
    for s in sentences:
        if len(s) > 2:
            for e in s:
                if e.id > 0:
                    e.head = e.parent_ids = e.pred_parent_ids
                    e.deprel = e.relations = e.pred_relations
                    e.feats = e.brat_label = e.pred_feats
    write_conllx(args.outfile, sentences)
