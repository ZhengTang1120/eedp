import os
import glob
import argparse
from collections import namedtuple, defaultdict
from processors import *
from utils import *
import json
import itertools

API = ProcessorsAPI(port=8886)

# brat mentions
TextboundMention = namedtuple('TextboundMention', 'id label start end text')
EventMention = namedtuple('EventMention', 'id label trigger arguments')

def read(filename):
    with open(filename) as f:
        return f.read()

def parse_annotations(annotations):
    ann_dict = dict()
    res = list()
    for line in annotations.splitlines():
        if line.startswith('T'):
            [id, data, text] = line.split('\t')
            [label, start, end] = data.split(' ')
            # if start+end not in ann_dict:
            #     ann_dict[start+end] = TextboundMention(id, [label], int(start), int(end), text)
            # else:
            #     ann_dict[start+end].label.append(label)
            res.append(TextboundMention(id, label, int(start), int(end), text))
        elif line.startswith('E'):
            [id, data] = line.split('\t')
            [label_trigger, *args] = data.split(' ')
            [label, trigger] = label_trigger.split(':')
            arguments = defaultdict(list)
            for a in args:
                if a.strip() != '':
                    [name, arg] = a.split(':')
                    arguments[name].append(arg)
            res.append(EventMention(id, label, trigger, dict(arguments)))
    res += ann_dict.values()
    return res

def parse_event_tree(event_tree, entities, eid, sent_ann):
    def check_id(tid, events):
        for id in events:
            if tid == id.split("|")[0]:
                return id
        return None
    # Group Binding Theme
    for head in event_tree:
        try:
            if head.feats == "Binding":
                temp = defaultdict(list)
                for theme in event_tree[head]["Theme"]:
                    head_id = entities[head][-1]
                    token_id = entities[theme[0]][-1]
                    deprel = sent_ann.dependencies.shortest_path(head_id, token_id)[0][1] if sent_ann.dependencies.shortest_path(head_id, token_id) else None
                    temp[deprel].append(theme)
                for i, key in enumerate(temp.keys(), start=1):
                    if i > 1:
                        event_tree[head]["Theme"+str(i)] = list()
                        for theme in temp[key]:
                            theme[-1] = "Theme"+str(i)
                            event_tree[head]["Theme"].remove(theme)
                            event_tree[head]["Theme"+str(i)].append(theme)
        except KeyError:
            pass
            # print ("Entity Not Found")
    # Parse Event Tree to Events
    events = dict()
    res = list()
    for head in event_tree:
        try:
            for li in (list(itertools.product(*list(event_tree[head].values())))):
                line = [head.feats, entities[head][0][0]]
                for k in li:
                    line.append({k[1]:entities[k[0]][0][0]})
                events[entities[head][0][0]+"|"+str(eid)] = ("E"+str(eid), line)
                eid += 1
        except KeyError:
            pass
            # print ("Entity Not Found")
    # Map Event ID to Token ID
    for head in events:
        for t, tid in enumerate(events[head][1][2:], start=2):
            tk = list(tid.keys())[0]
            tid = list(tid.values())[0]
            tid = check_id(tid, events)
            if tid:
                events[head][1][t][tk] = events[tid][0]
    for head in events:
        res.append(events[head])
    return eid, res        

def attach_node(event_tree, token, k, head):
    try:
        event_tree[head][token.deprel[k]].append([token, token.deprel[k]])
    except KeyError:
        event_tree[head][token.deprel[k]] = [[token, token.deprel[k]]]

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    parser.add_argument('--conllx', default='brat.conllx')
    args = parser.parse_args()
    sentences = read_conllx(args.conllx, True)
    es = 0
    for fname in glob.glob(os.path.join(args.datadir, '*.a1')):
        root = os.path.splitext(fname)[0]
        name = os.path.basename(root)
        
        txt = read(root + '.txt')
        a1 = read(root + '.a1')
        a1 = list(parse_annotations(a1))
        root = root.replace("BioNLP-ST-2013_GE_devel_data_rev3","BioNLP-ST-2013_GE_devel_data_rev3_out")
        t1 = open(root + '.a2', "w")
        doc = API.bionlp.annotate(txt)
        curr_sents = sentences[:len(doc.sentences)]
        sentences = sentences[len(doc.sentences):]
        entities = dict()
        proteins = dict()
        events = list()
        eid = 1
        for i, sent in enumerate(curr_sents):
            event_tree = defaultdict(dict)
            sent_ann = doc.sentences[i]
            multitoken_start = -1
            multitoken_start_id = -1
            for j, token in enumerate(sent):
                if token.feats == "Protein":
                    if token.deprel[0] != "multitoken":
                        tid=None
                        for w in a1:
                            if w.end == sent_ann.endOffsets[j-1]:
                                tid = w.id
                                form = w.text
                                break
                        if tid:
                            proteins[token] = ([tid],multitoken_start_id if multitoken_start_id != -1 else j-1 )
                        else:
                            pass
                            # print ("Protein Not Found")
                        multitoken_start = -1
                        multitoken_start_id = -1
                        for k, th in enumerate(token.head):
                            head = (sent[th] if th != -1 else None)
                            if head and head.id != 0:
                                attach_node(event_tree, token, k, head)
                    else:
                        if multitoken_start != -1:
                            multitoken_start = sent_ann.startOffsets[j-1]
                            multitoken_start_id = j-1 if "VB" in sent_ann.tags[j-1] else -1
                elif token.feats != "O" and token.id != 0:
                    if token.deprel[0] == "multitoken":
                        if multitoken_start != -1:
                            multitoken_start = sent_ann.startOffsets[j-1]
                            multitoken_start_id = j-1 if "VB" in sent_ann.tags[j-1] else -1
                    else:
                        entities[token] = (("T"+str(len(entities)+len(a1)+1), 
                            token.feats+" "
                            +str(multitoken_start if multitoken_start != -1 else sent_ann.startOffsets[j-1])
                            +" "+str(sent_ann.endOffsets[j-1]), token.form), 
                            multitoken_start_id if multitoken_start_id != -1 else j-1 )
                        multitoken_start = -1
                        multitoken_start_id = -1
                        for k, th in enumerate(token.head):
                            head = (sent[th] if th != -1 else None)
                            if head and head.id != 0:
                                attach_node(event_tree, token, k, head)
            if (event_tree):
                es += 1
            eid, res = parse_event_tree(event_tree, {**proteins, **entities}, eid, sent_ann)
            events += res
        tids = [ent[0][0] for ent in entities.values()]
        eids = [event[0] for event in events if event[1][0] not in ["Protein", "Entity"]]
        eset = set()
        elist = list()
        for id in entities:
            line = ""
            for e in entities[id][0][:-1]:
                line += e+"\t"
            line += entities[id][0][-1]
            t1.write(line+"\n")
        for event in events:
            if event[1][0] not in ["Protein", "Entity"]:
                line = event[0]+"\t"+event[1][0]+":"+event[1][1]
                tc = list()
                ec = list()
                valid = True
                for t in event[1][2:]:
                    for k in t:
                        if "Theme" in k:
                            if t[k].startswith("E") and t[k] not in eids:
                                pass
                            elif "egulation" not in event[1][0] and t[k].startswith("E"):
                                if t[k] in eids:
                                    eids.remove(t[k])
                                pass
                            else:
                                line += " "+k+":"+t[k]
                                if t[k].startswith("E"):
                                    ec.append(t[k])
                                else:
                                    tc.append(t[k])
                if "egulation" in event[1][0]:
                    for k in event[1][2:]:
                        if "Cause" in k and k["Cause"] not in tids:
                            if k["Cause"].startswith("E") and k["Cause"] not in eids:
                                pass
                            else:
                                line += " Cause:"+k["Cause"]
                                if k["Cause"].startswith("E"):
                                    ec.append(k["Cause"])
                                else:
                                    tc.append(k["Cause"]) #?!
                else:
                    for k in event[1][2:]:
                        if "Cause" in k and k["Cause"].startswith("E") and k["Cause"] in eids:
                            eids.remove(k["Cause"])
                a = list(set(tids) & set(tc))
                if valid and len(a)!=0:
                    valid = False
                if "Theme:" in line and valid and line.split("\t")[-1] not in eset:
                    elist.append((line, ec))
                    # t1.write(line+"\n")
                    eset.add(line.split("\t")[-1])
                elif event[0] in eids:
                    eids.remove(event[0])
        f = len(elist)
        while (f > 0):
            f = len(elist)
            temp = list()
            for line, ec in elist:
                if list(set(eids) & set(ec)) == ec:
                    # t1.write(line+"\n")
                    temp.append((line, ec))
                    f -= 1
                elif line.split("\t")[0] in eids:
                    eids.remove(line.split("\t")[0])
            elist = temp
        for line, ec in elist:
            t1.write(line+"\n")
        t1.close()
    #     print ("./a2-evaluate.pl -g gold-dev/ -s "+root + '.a2.t1')
    #     os.system("./a2-evaluate.pl -g gold-dev/ -s "+root + '.a2.t1')
    # print (es)