#!/usr/bin/env python

import os
import glob
import argparse
from lxml import etree
from nltk import Tree

def escape(text):
    escapes = {
        '(': '-LRP-',
        ')': '-RRP-',
    }
    for old, new in escapes.items():
        text = text.replace(old, new)
    return text

def make_tree(sentence):
    if len(sentence) != 1:
        print('ERROR: invalid tree')
        return
    node = sentence[0]
    if node.tag == 'cons':
        return make_cons(node)
    elif node.tag == 'tok':
        return make_tok(node)
    else:
        print('ERROR: invalid node')

def make_cons(node):
    cat = node.get('cat')
    children = []
    for child in node:
        if child.tag == 'cons' and len(child) > 0:
            # there are some empty cons that we just skip
            # is this ok?
            children.append(make_cons(child))
        elif child.tag == 'tok':
            children.append(make_tok(child))
    return Tree(cat, children)
    
def make_tok(node):
    cat = node.get('cat')
    child = escape(node.text)
    return Tree(cat, [child])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    parser.add_argument('outdir')
    args = parser.parse_args()

    for filename in glob.glob(os.path.join(args.datadir, '*.xml')):
        pmid = os.path.basename(os.path.splitext(filename)[0])
        print(f'processing {pmid}')
        xml = etree.parse(filename)
        sentences = xml.xpath('//sentence')
        outfilename = os.path.join(args.outdir, f'{pmid}.ptb')
        with open(outfilename, 'w') as f:
            for s in sentences:
                tree = make_tree(s)
                f.write(tree._pformat_flat('', '()', False))
                f.write('\n')
