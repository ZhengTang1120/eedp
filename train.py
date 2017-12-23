#!/usr/bin/env python

import argparse, random
from utils import *
from parsers import ArcHybridParser

def make_parser(args, word_count, words, tags, rels):
    return ArcHybridParser(
        word_count, words, tags, rels,
        args.w_embed_dim,
        args.t_embed_dim,
        args.lstm_hidden_size,
        args.lstm_num_layers,
        args.dep_op_hidden_size,
        args.dep_lbl_hidden_size,
        args.ev_op_hidden_size,
        args.ev_lbl_hidden_size,
        args.alpha,
        args.p_explore,
    )

if __name__ == '__main__':

    random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('--outdir', default='out')
    parser.add_argument('--w_embed_dim',         type=int,   default=100)
    parser.add_argument('--t_embed_dim',         type=int,   default=25)
    parser.add_argument('--lstm_hidden_size',    type=int,   default=125)
    parser.add_argument('--lstm_num_layers',     type=int,   default=2)
    parser.add_argument('--dep_op_hidden_size',  type=int,   default=100)
    parser.add_argument('--dep_lbl_hidden_size', type=int,   default=100)
    parser.add_argument('--ev_op_hidden_size',   type=int,   default=100)
    parser.add_argument('--ev_lbl_hidden_size',  type=int,   default=100)
    parser.add_argument('--epochs',              type=int,   default=30)
    parser.add_argument('--alpha',               type=float, default=0.25) # for word dropout
    parser.add_argument('--p_explore',           type=float, default=0.1)
    args = parser.parse_args()

    print('loading ...')
    sentences = read_conllx(args.infile)
    vocabularies = make_vocabularies(sentences)
    parser = make_parser(args, *vocabularies)

    print('training ...')
    for epoch in range(args.epochs):
        print('epoch', epoch + 1)
        random.shuffle(sentences)
        parser.train(sentences)
        name = f'{args.outdir}/parser{epoch+1:03}'
        print('saving', name)
        parser.save(name)
        print()
