#!/usr/bin/env python

import argparse, random
from utils import *
from parsers_tr import ArcHybridParser
import os
import numpy as np
from trainer import *

def make_parser(args, word_count, words, tags, chars, entities=None, ev_rels=None, dep_rels=None):
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
        args.pretrained
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
    parser.add_argument('--lstm_num_layers',     type=int,   default=3)
    parser.add_argument('--dep_op_hidden_size',  type=int,   default=100)
    parser.add_argument('--dep_lbl_hidden_size', type=int,   default=100)
    parser.add_argument('--ev_op_hidden_size',   type=int,   default=100)
    parser.add_argument('--ev_lbl_hidden_size',  type=int,   default=100)
    parser.add_argument('--tg_lbl_hidden_size',  type=int,   default=100)
    parser.add_argument('--epochs',              type=int,   default=30)
    parser.add_argument('--alpha',               type=float, default=0.25) # for word dropout
    parser.add_argument('--p_explore',           type=float, default=0.0)
    parser.add_argument('--pretrained', action='store_true')
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
        parser = make_parser(args, *vocabularies)
        # wd = dict()
        # for line in open("embeddings_november_2016.txt"):
        #     if len(line.split()) > 2:
        #         word = normalize(line.split()[0])
        #         vector = ''.join(line.split()[1:])
        #         wd[word] = vector
        # v = np.zeros((len(vocabularies[1]), 100))
        # for i, word in enumerate(vocabularies[1]):
        #     if word in wd:
        #         print (wd[word])
        #         v[i] = np.fromstring(wd[word], sep=' ')
        # np.savetxt("pubmedm13.txt", v)
        # exit()
    elif events:
        vocabularies = make_vocabularies3(events)
        parser = make_parser(args, *vocabularies[:-1], ev_rels = vocabularies[-1])
    elif sentences:
        vocabularies = make_vocabularies(sentences)
        parser = make_parser(args, *vocabularies[:-1], dep_rels = vocabularies[-1])
    else:
        print("Must have at least one file to parse!")
        exit()

    # i = 0
    # for sentence in events:
    #     for e in sentence:
    #         if -1 not in e.head and len(e.head) != 0 and 0 not in e.head:
    #             print (e.head, e.deprel)
    #             i += 1
    #             break
    # print (i)
    # print (parser.model.parameters_list())
    # exit()


    print('training ...')
    for epoch in range(args.epochs):
        print('epoch', epoch + 1)
        if args.depsfile:
            random.shuffle(sentences)
            train_dependencies(sentences, parser)
        if args.evsfile:
            random.shuffle(events)
            train_events(events, parser)
        name = f'{args.outdir}/parser{epoch+1:03}'
        print('saving', name)
        parser.save(name)
        print()
