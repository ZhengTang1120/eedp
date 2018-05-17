#!/usr/bin/env python

import argparse, random
from utils import *
from parsers import ArcHybridParser
import os

def make_parser(args, word_count, words, tags, chars, entities=None, ev_rels=None, dep_rels=None, pretrained=None):
    return ArcHybridParser(
        word_count, words, tags, chars, entities,
        dep_rels, ev_rels,
        args.w_embed_dim,
        args.t_embed_dim,
        args.c_embed_dim,
        args.clstm_hidden_size,
        args.e_embed_dim,
        args.lstm_hidden_size,
        args.lstm_num_layers,
        args.dep_op_hidden_size,
        args.dep_lbl_hidden_size,
        args.ev_op_hidden_size,
        args.ev_lbl_hidden_size,
        args.tg_lbl_hidden_size,
        args.alpha,
        args.p_explore,
        pretrained
    )

if __name__ == '__main__':

    random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--depsfile', default=None)
    parser.add_argument('--evsfile', default=None)
    parser.add_argument('--outdir', default='out')
    parser.add_argument('--w_embed_dim',         type=int,   default=100)
    parser.add_argument('--t_embed_dim',         type=int,   default=25)
    parser.add_argument('--c_embed_dim',         type=int,   default=25)
    parser.add_argument('--clstm_hidden_size',    type=int,   default=50)
    parser.add_argument('--e_embed_dim',         type=int,   default=25)
    parser.add_argument('--lstm_hidden_size',    type=int,   default=125)
    parser.add_argument('--lstm_num_layers',     type=int,   default=2)
    parser.add_argument('--dep_op_hidden_size',  type=int,   default=100)
    parser.add_argument('--dep_lbl_hidden_size', type=int,   default=100)
    parser.add_argument('--ev_op_hidden_size',   type=int,   default=100)
    parser.add_argument('--ev_lbl_hidden_size',  type=int,   default=100)
    parser.add_argument('--tg_lbl_hidden_size',  type=int,   default=100)
    parser.add_argument('--epochs',              type=int,   default=30)
    parser.add_argument('--alpha',               type=float, default=0.25) # for word dropout
    parser.add_argument('--p_explore',           type=float, default=0.0)
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    print('loading ...')
    events = sentences = None
    if args.evsfile:
        events = read_conllx(args.evsfile)
    if args.depsfile:
        sentences = read_conllx(args.depsfile, non_proj=False)
    if events and sentences:
        vocabularies = make_vocabularies2(sentences, events)
        emb_matrix_pretrained = np.loadtxt("pubmedm.txt")
        parser = make_parser(args, *vocabularies, pretrained=emb_matrix_pretrained)
    elif events:
        vocabularies = make_vocabularies3(events)
        emb_matrix_pretrained = np.loadtxt("pubmed.txt")
        parser = make_parser(args, *vocabularies[:-1], ev_rels = vocabularies[-1], pretrained=emb_matrix_pretrained)
    elif sentences:
        vocabularies = make_vocabularies(sentences)
        parser = make_parser(args, *vocabularies[:-1], dep_rels = vocabularies[-1])
    else:
        print("Must have at least one file to parse!")
        exit()

    print('training ...')
    for epoch in range(args.epochs):
        print('epoch', epoch + 1)
        if args.depsfile:
            random.shuffle(sentences)
            parser.train_dependencies(sentences)
        if args.evsfile:
            random.shuffle(events)
            parser.train_events(events)
        name = f'{args.outdir}/parser{epoch+1:03}'
        print('saving', name)
        parser.save(name)
        print()
