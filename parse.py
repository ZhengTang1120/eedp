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
                    if (e.pred_feats == "O" and e.pred_relation != "none"):
                        print(".....")
                        exit()
                    e.head = e.parent_id = e.pred_parent_id
                    e.deprel = e.relation = e.pred_relation
                    e.feats = e.brat_label = e.pred_feats
    write_conllx(args.outfile, sentences)
