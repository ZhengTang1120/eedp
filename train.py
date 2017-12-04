#!/usr/bin/env python

import argparse, random
from utils import *
from parser import ArcHybridParser

def make_parser(args, word_count, words, tags, rels):
    return ArcHybridParser(
        word_count, words, tags, rels,
        args.w_embed_dim,
        args.t_embed_dim,
        args.lstm_hidden_size,
        args.lstm_num_layers,
        args.act_hidden_size,
        args.lbl_hidden_size,
    )

if __name__ == '__main__':

    random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('--w_embed_dim',      type=int, default=100)
    parser.add_argument('--t_embed_dim',      type=int, default=25)
    parser.add_argument('--lstm_hidden_size', type=int, default=150)
    parser.add_argument('--lstm_num_layers',  type=int, default=2)
    parser.add_argument('--act_hidden_size',  type=int, default=100)
    parser.add_argument('--lbl_hidden_size',  type=int, default=100)
    parser.add_argument('--epochs',           type=int, default=30)
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
        name = f'out/parser{epoch+1:03}'
        print('saving', name)
        parser.save(name)
        print()
