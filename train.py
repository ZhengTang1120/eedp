#!/usr/bin/env python

import argparse, random
from utils import *
from parsers import ArcHybridParser
import os

def make_parser(args, word_count, words, tags, ev_rels, entities):
    return ArcHybridParser(
        word_count, words, tags,
        ev_rels, entities,
        args.w_embed_dim,
        args.t_embed_dim,
        args.e_embed_dim,
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
    # parser.add_argument('depsfile')
    parser.add_argument('evsfile')
    parser.add_argument('--outdir', default='out')
    parser.add_argument('--w_embed_dim',         type=int,   default=100)
    parser.add_argument('--t_embed_dim',         type=int,   default=25)
    parser.add_argument('--e_embed_dim',         type=int,   default=25)
    parser.add_argument('--lstm_hidden_size',    type=int,   default=125)
    parser.add_argument('--lstm_num_layers',     type=int,   default=2)
    parser.add_argument('--dep_op_hidden_size',  type=int,   default=100)
    parser.add_argument('--dep_lbl_hidden_size', type=int,   default=100)
    parser.add_argument('--ev_op_hidden_size',   type=int,   default=100)
    parser.add_argument('--ev_lbl_hidden_size',  type=int,   default=100)
    parser.add_argument('--epochs',              type=int,   default=30)
    parser.add_argument('--alpha',               type=float, default=0.25) # for word dropout
    parser.add_argument('--p_explore',           type=float, default=0.0)
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    print('loading ...')
    # sentences = read_conllx(args.depsfile, non_proj=False)
    events = read_conllx(args.evsfile)
    vocabularies = make_vocabularies3(events)
    parser = make_parser(args, *vocabularies)

    print('training ...')
    for epoch in range(args.epochs):
        print('epoch', epoch + 1)
        # random.shuffle(sentences)
        # parser.train_dependencies(sentences)
        parser.train_events(events)
        name = f'{args.outdir}/parser{epoch+1:03}'
        print('saving', name)
        parser.save(name)
        print()
