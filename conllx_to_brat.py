#!/usr/bin/env python

import os
import glob
import argparse
from collections import namedtuple, defaultdict
from nltk import sent_tokenize, pos_tag
from nltk.tokenize import RegexpTokenizer
from utils import *
import itertools

from brat_to_conllx import *

import json

l2r = json.load(open("label2rel.json"))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    # parser.add_argument('outdir')
    parser.add_argument('--conllx')
    args = parser.parse_args()
    sentences = read_conllx(args.conllx, True)
    for fname in glob.glob(os.path.join(args.datadir, '*.a1')):
        root = os.path.splitext(fname)[0]
        name = os.path.basename(root)
        print (name)
        txt = read(root + '.txt')
        a1 = read(root + '.a1')
        a1 = list(parse_a1(a1))
        try:
            lasttid = int(a1[-1].id[1:]) + 1
        except:
            lasttid = 0
        # t1 = open(os.path.join(args.outdir,name+'.a2'), "w")
        doc = list(get_token_spans(txt))
        curr_sents = sentences[:len(doc)]
        sentences = sentences[len(doc):]
        entities = dict()
        ecount = 0
        for i, sent in enumerate(curr_sents):
            events = defaultdict(list)
            token_dict = dict()
            for j, token in enumerate(sent):
                for w in a1:
                    if w.end == doc[i][2][j-1]:
                        id = w.id
                        break
                token_dict[token.id] = (token.id, token.feats, token.deprel, id, doc[i][1][j-1] if j>0 else 0, doc[i][2][j-1] if j>0 else 0, token.form)
            for j, token in enumerate(sent):
                tid = None
                for w in a1:
                    if j!=0 and w.end == doc[i][2][j-1]:
                        tid = w.id
                        events[token_dict[token.head]].append((token.id, token.feats, token.deprel, w.id, w.start, w.end, token.form))
                        break
            # for event in events:
            #     if event[0] != 0:
            #         for arg in events[event]:
            #             if arg[2] != "multitoken":
            #                 if "#" not in event[1]:
            #                     print (event[1], l2r[event[1]][0], event[3], arg[2], arg[3])
            #                 else:
            #                     print ("Negation")
            #                     print (event[1], l2r[event[1].split("#")[1]][0], event[3], arg[2], arg[3])
            # l = p.parse_event(events, (0, 'O', 'rroot', None, 0, 0, '*root*'))
            # # print (l)
            # p.out += l
        # for e in p.out:
        #     if e[1] == "Binding" and len(e)>3:
        #         print (e)
        #     line = "E%d\t%s:%s "%(e[0], e[1], e[2])
        #     line += ' '.join(e[3:])
        #     line += '\n'
        #     if "Theme" in line:
        #         t1.write(line)
        # t1.close()



