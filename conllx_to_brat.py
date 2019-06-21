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

# brat mentions
TextboundMention = namedtuple('TextboundMention', 'id label start end text')
EventMention = namedtuple('EventMention', 'id label trigger arguments')

# global variables to keep statistics that are printed at the end
total_sentences = 0
skipped_sentences = 0

#initial tokenizer
tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')

class eparser:
    def __init__(self):
        self.ecount = 0
        self.out = []

    def parse_event(self, events, node):
        out = list()
        if node[1] != "Binding":
            for child in events[node]:
                if child[2] == "Theme":
                    if child in events:
                        child_events = self.parse_event(events, child)
                        out += child_events
                        for cline in child_events:
                            self.ecount += 1
                            out.append([self.ecount, node[1], node[3],"Theme:E"+str(cline[0])])
                    else:
                        self.ecount += 1
                        out.append([self.ecount, node[1], node[3],"Theme:"+child[3]])
                elif child[2] == "root":
                    out += self.parse_event(events, child)
            temp = out[:]
            for line in temp:
                if line[2] == node[3]:
                    first = True
                    for child in events[node]:
                        if child[2] == "Cause":
                            if child in events:
                                if child[3] in [_[1] for _ in temp]:
                                    for t in temp:
                                        if child[3] == t[1]:
                                            if first:
                                                line.append("Cause:E"+str(t[0]))
                                                first = False
                                            else:
                                                newline = line[:]
                                                self.ecount += 1
                                                newline[0] = self.ecount
                                                newline[-1] = "Cause:E"+str(t[0])
                                                out.append(newline)

                                else:
                                    child_events = self.parse_event(events, child)
                                    out += child_events
                                    for cline in child_events:
                                        if first:
                                            line.append("Cause:E"+str(cline[0]))
                                            first = False
                                        else:
                                            newline = line[:]
                                            self.ecount += 1
                                            newline[0] = self.ecount
                                            newline[-1] = "Cause:E"+str(cline[0])
                                            out.append(newline)
                            else:
                                if first:
                                    line.append("Cause:"+child[3])
                                    first = False
                                else:
                                    newline = line[:]
                                    self.ecount += 1
                                    newline[0] = self.ecount
                                    newline[-1] = "Cause:"+child[3]
                                    out.append(newline)
        else:
            # self.ecount += 1
            # line = [self.ecount, node[1], node[3]]
            # for child in events[node]:
            #     if child[2] == "Theme":
            #         if child in events:
            #             child_events = self.parse_event(events, child)
            #             out += child_events
            #             for cline in child_events:
            #                 line += ["Theme:E"+str(cline[0])]
            #         else:
            #             line += ["Theme:"+child[3]]
            # out.append(line)
            left = list()
            right = list()
            for child in events[node]:
                if child[4] > node[4]:
                    right.append(child)
                else:
                    left.append(child)
            if len(left) == 0 or len(right) == 0:
                self.ecount += 1
                line = [self.ecount, node[1], node[3]]
                for child in left + right:
                    if child in events:
                        child_events = self.parse_event(events, child)
                        out += child_events
                        for cline in child_events:
                            line += ["Theme:E"+str(cline[0])]
                    else:
                        line += ["Theme:"+child[3]]
                out.append(line) 
            else:
                for child1 in left:
                    if child1[2] == "Theme":
                        for child2 in right:
                            if child2[2] == "Theme":
                                self.ecount += 1
                                line = [self.ecount, node[1], node[3]]
                                if child1 in events:
                                    child1_events = self.parse_event(events, child1)
                                    out += child1_events
                                    for cline in child1_events:
                                        line += ["Theme:E"+str(cline[0])]
                                else:
                                    line += ["Theme:"+child1[3]]
                                if child2 in events:
                                    child2_events = self.parse_event(events, child2)
                                    out += child2_events
                                    for cline in child2_events:
                                        line += ["Theme:E"+str(cline[0])]
                                else:
                                    line += ["Theme:"+child2[3]]
                                out.append(line)
        return out

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    parser.add_argument('outdir')
    parser.add_argument('--conllx')
    args = parser.parse_args()
    sentences = read_conllx(args.conllx, True)
    for fname in glob.glob(os.path.join(args.datadir, '*.a1')):
        p = eparser()
        root = os.path.splitext(fname)[0]
        name = os.path.basename(root)
        print (name)
        txt = read(root + '.txt')
        a1 = read(root + '.a1')
        a1 = list(parse_annotations(a1))
        try:
            lasttid = int(a1[-1].id[1:]) + 1
        except:
            lasttid = 0
        t1 = open(os.path.join(args.outdir,name+'.a2'), "w")
        doc = list(get_token_spans(txt))
        curr_sents = sentences[:len(doc)]
        sentences = sentences[len(doc):]
        entities = dict()
        ecount = 0
        event_out = dict()
        for i, sent in enumerate(curr_sents):
            events = defaultdict(list)
            token_dict = dict()
            for j, token in enumerate(sent):
                token_dict[token.id] = (token.id, token.feats, token.deprel, None, doc[i][1][j-1] if j>0 else 0, doc[i][2][j-1] if j>0 else 0, token.form)
            for j, token in enumerate(sent):
                tid = None
                for w in a1:
                    if w.end == doc[i][2][j-1]:
                        tid = w.id
                        events[token_dict[token.head]].append((token.id, token.feats, token.deprel, w.id, w.start, w.end, token.form))
                        break
                if not tid and j > 0:
                    if token.feats not in ["O", "Protein"]:
                        if token.deprel != "multitoken" and token.feats != "Protein":
                            s = doc[i][1][j-1]
                            e = doc[i][2][j-1]
                            f = ""
                            if token_dict[token.id] in events:
                                for k, child in enumerate(events[token_dict[token.id]]):
                                    if child[2] == "multitoken":
                                        s = min(s, child[4])
                                        f += child[-1] + " "
                                        del events[token_dict[token.id]][k]
                            f += token.form
                            t1.write("%s\t%s %d %d\t%s\n"%("T"+str(lasttid), token.feats, s, e, f))
                            old_key = token_dict[token.id]
                            token_dict[token.id] = (token.id, token.feats, token.deprel, "T"+str(lasttid), doc[i][1][j-1], doc[i][2][j-1], token.form)   
                            events[token_dict[token.id]] = events[old_key]
                            del events[old_key]
                        events[token_dict[token.head]].append((token.id, token.feats, token.deprel, "T"+str(lasttid), doc[i][1][j-1], doc[i][2][j-1], token.form))
                        if token.deprel != "multitoken":
                            lasttid = lasttid + 1
            l = p.parse_event(events, (0, 'O', 'rroot', None, 0, 0, '*root*'))
            # print (l)
            p.out += l
        for e in p.out:
            if e[1] == "Binding" and len(e)>3:
                print (e)
            line = "E%d\t%s:%s "%(e[0], e[1], e[2])
            line += ' '.join(e[3:])
            line += '\n'
            if "Theme" in line:
                t1.write(line)
        #     for event in events:
        #         for child in events[event]:
        #             if child in events and len(events[child]) > 0:
        #                 print (event, child, events[child])
        #     for head in events:
        #         if head[3] and events[head] and head[1] != "Entity":
        #             if head[1] == "Binding":
        #                 ecount += 1
        #                 event_out[head[3]] = "E%d\t%s:%s"%(ecount, head[1], head[3])
        #                 for child in events[head]:
        #                     if child[2] == "Theme":
        #                         event_out[head[3]] += " Theme:%s"%(child[3])
        #                 # flg = False
        #                 # for childl in events[head]:
        #                 #     if childl[0] < head[0] and childl[2] == "Theme":
        #                 #         ecount += 1
        #                 #         event_out[head[3]+"."+str(ecount)] = "E%d\t%s:%s Theme:%s"%(ecount, head[1], head[3], childl[3])
        #                 #         for childr in events[head]:
        #                 #             if childr[0] > head[0] and childr[2] == "Theme":
        #                 #                 event_out[head[3]+"."+str(ecount)] += " Theme:%s"%(childr[3])
        #                 #                 flg = True
        #                 #                 ecount += 1
        #                 #                 event_out[head[3]+"."+str(ecount)] = "E%d\t%s:%s Theme:%s"%(ecount, head[1], head[3], childl[3])
        #                 #         if flg:
        #                 #             del event_out[head[3]+"."+str(ecount)]
        #                 #             ecount -= 1
        #             else:
        #                 ecount += 1
        #                 event_out[head[3]] = "E%d\t%s:%s"%(ecount, head[1], head[3])
        #                 for child in events[head]:
        #                     if child[2] == "Theme":
        #                         # if head[1] == "Binding":
        #                         #     event_out[head[3]] += " Theme:%s"%(child[3])
        #                         if event_out[head[3]].count(":") == 1:
        #                             event_out[head[3]] += " Theme:%s"%(child[3])
        #                         else:
        #                             ecount += 1
        #                             event_out[head[3]+"."+str(ecount)] = "E%d\t%s:%s"%(ecount, head[1], head[3])
        #                             event_out[head[3]+"."+str(ecount)] += " Theme:%s"%(child[3])
        #     temp = list(event_out.keys())
        #     for head in events:
        #         if head[3] and events[head] and head[1] != "Entity": 
        #             for child in events[head]:           
        #                 if child[2] == "Cause":
        #                     for h in temp:
        #                         if head[3] in h:
        #                             if "Cause" not in event_out[h]: 
        #                                 event_out[h] += " Cause:%s"%(child[3])
        #                             else:
        #                                 ecount += 1
        #                                 event_out[h+"."+str(ecount)] = event_out[h].replace("E%d"%(ecount-1), "E%d"%ecount)
        #                                 oldcause = event_out[h].split("Cause:")[-1]
        #                                 event_out[h+"."+str(ecount)] = event_out[h+"."+str(ecount)].replace("Cause:%s"%oldcause, "Cause:%s"%(child[3]))
        # for trigger in event_out:
        #     for t in event_out:
        #         fir_part = event_out[t].split(" ")[0]
        #         sec_part = " ".join(event_out[t].split(" ")[1:])
        #         if trigger.split(".")[0] in sec_part and t!=trigger:
        #             event_out[t] = fir_part + " " + sec_part.replace(trigger, event_out[trigger].split("\t")[0])
        # for t in event_out:
        #     if "Theme:" in event_out[t]:
        #         t1.write(event_out[t]+"\n")
        # print(json.dumps(event_out, indent=4, sort_keys=True))
        t1.close()



