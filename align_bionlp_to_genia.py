#!/usr/bin/env python

import os
import glob
import argparse
from nltk import Tree

def unescape(text):
    escapes = {
        '(': '-LRP-',
        ')': '-RRP-',
    }
    for old, new in escapes.items():
        text = text.replace(new, old)
    return text

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('bionlp')
    parser.add_argument('genia')
    parser.add_argument('tokens')
    args = parser.parse_args()

    for filename in glob.glob(os.path.join(args.bionlp, '*.a1')):
        pmid = os.path.basename(os.path.splitext(filename)[0])
        print(f'processing {pmid}')
        text_fn = os.path.join(args.bionlp, f'{pmid}.txt')
        genia_fn = os.path.join(args.genia, f'{pmid}.ptb')
        tokens_fn = os.path.join(args.tokens, f'{pmid}.tok')

        if not os.path.exists(genia_fn):
            print(f'ERROR: {pmid} is included in bionlp but not in genia')
            continue

        with open(text_fn) as f:
            text = f.read()

        tokens = []
        starts = []
        ends = []
        position = -1

        with open(genia_fn) as f:
            for line in f:
                try:
                    tree = Tree.fromstring(line)
                except:
                    print(line)
                    raise
                for token in tree.leaves():
                    token = unescape(token)
                    position = text.find(token, position + 1)
                    if position == -1:
                        print('ERROR: misaligned token')
                        raise
                    tokens.append(token)
                    starts.append(position)
                    ends.append(position + len(token))

        with open(tokens_fn, 'w') as f:
            for token, start, end in zip(tokens, starts, ends):
                f.write(f'{token}\t{start}\t{end}\n')
