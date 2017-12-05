#!/usr/bin/env python

import argparse
from utils import *
from arc_hybrid import ArcHybridParser

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('infile')
    parser.add_argument('outfile')
    args = parser.parse_args()

    print('loading ...')
    parser = ArcHybridParser.load(args.model)
    sentences = read_conllx(args.infile)

    print('parsing ...')
    for i, s in enumerate(sentences):
        print('parsing sentence', i)
        parser.parse_sentence(s)

    print('writing output ...')
    for s in sentences:
        s.head = s.parent_id = s.pred_parent_id
        s.deprel = s.relation = s.pred_relation
    write_conllx(args.outfile, sentences)
