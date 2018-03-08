import os
import glob
import argparse
from collections import namedtuple, defaultdict
from processors import *
from utils import *
import json
import itertools

API = ProcessorsBaseAPI(port=8888)

# brat mentions
TextboundMention = namedtuple('TextboundMention', 'id label start end text')
EventMention = namedtuple('EventMention', 'id label trigger arguments')

def read(filename):
    with open(filename) as f:
        return f.read()

def parse_annotations(annotations):
    for line in annotations.splitlines():
        if line.startswith('T'):
            [id, data, text] = line.split('\t')
            [label, start, end] = data.split(' ')
            yield TextboundMention(id, label, int(start), int(end), text)
        elif line.startswith('E'):
            [id, data] = line.split('\t')
            [label_trigger, *args] = data.split(' ')
            [label, trigger] = label_trigger.split(':')
            arguments = defaultdict(list)
            for a in args:
                if a.strip() != '':
                    [name, arg] = a.split(':')
                    arguments[name].append(arg)
            yield EventMention(id, label, trigger, dict(arguments))

def parse_event_tree(event_tree, parent):
    global event_count, entities, i, root, events, missing_entities
    event = None
    if parent.head != -1:
        temp = defaultdict(list)
        for child in event_tree[parent]:
            if child not in event_tree or event_tree[child][0].deprel == "multitoken":
                if child.deprel != "multitoken":
                    try:
                        temp[child.deprel].append((entities[child.form+str(child.id)+str(i)][0],child.real_deprel))
                    except:
                        print (child.form+str(child.id)+str(i) in missing_entities)
                        raise Exception(child)
            else:
                try:
                    temp[child.deprel].append((parse_event_tree(event_tree, child), child.real_deprel))
                except:
                    print (child.form+str(child.id)+str(i) in missing_entities)
                    raise Exception(child)
        if temp:
            event = "E"+str(event_count)
            event_count += 1
            try:
                events.append((event, parent.feats, entities[parent.form+str(parent.id)+str(i)][0], temp))
            except:
                print (parent.form+str(parent.id)+str(i) in missing_entities)
                raise Exception(parent)
    else:
        for child in event_tree[parent]:
            parse_event_tree(event_tree, child)
    return event



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('datadir')
    parser.add_argument('--conllx', default='brat.conllx')
    args = parser.parse_args()

    sentences = read_conllx(args.conllx, True)
    for fname in glob.glob(os.path.join(args.datadir, '*.a1')):
        root = os.path.splitext(fname)[0]
        name = os.path.basename(root)
        
        txt = read(root + '.txt')
        a1 = read(root + '.a1')
        a1 = list(parse_annotations(a1))
        t1 = open(root + '.a2.t1', "w")
        doc = API.bionlp.annotate(txt)
        curr_sents = sentences[:len(doc.sentences)]
        sentences = sentences[len(doc.sentences):]
        entities = dict()
        proteins = dict()
        events = list()
        missing_entities = dict()
        for i, sent in enumerate(curr_sents):
            sent_ann = doc.sentences[i]
            multitoken_s = -1
            for j, token in enumerate(sent):
                if token.feats != "O" and token.id != 0:
                    if token.feats != "Protein":
                        if token.deprel != "multitoken":
                            if multitoken_s != -1:
                                multitoken_s = -1
                            else:
                                if not token.feats:
                                    token.feats = "O"
                                entities[token.form+str(token.id)+str(i)] = ("T"+str(len(entities)+len(a1)+1), 
                                    token.feats+" "+str(sent_ann.startOffsets[j-1])+" "+str(sent_ann.endOffsets[j-1]), 
                                    token.form)
                        else:
                            if multitoken_s == -1:
                                multitoken_s = sent_ann.startOffsets[j-1]
                                last_end = sent_ann.endOffsets[token.head-1]
                                entities[sent_ann.words[token.head-1]+str(token.head)+str(i)] = ("T"+str(len(entities)+len(a1)+1), 
                                    token.feats+" "+str(multitoken_s)+" "+str(last_end), 
                                    txt[multitoken_s: last_end]) 
                    elif token.deprel != "multitoken":
                        tid=None
                        for w in a1:
                            if w.end == sent_ann.endOffsets[j-1]:
                                tid = w.id
                                form = w.text
                                break
                        if tid:
                            proteins[token.form+str(token.id)+str(i)] = [tid]
                        else:
                            missing_entities[token.form+str(token.id)+str(i)] = 1
                            print (str(token)+" invalid")
        for id in entities:
            line = ""
            for e in entities[id][:-1]:
                line += e+"\t"
            line += entities[id][-1]+"\n"
            t1.write(line)
        entities = {**proteins, **entities}
        event_count = 1
        for i, sent in enumerate(curr_sents):
            event_tree = defaultdict(list)
            sent_ann = doc.sentences[i]
            for j, token in enumerate(sent):
                head = (sent[token.head] if token.head != -1 else None)
                if head:
                    # print (sent_ann.dependencies)
                    # print (head.id-1, token.id-1)
                    # print (sent_ann.dependencies.shortest_path(head.id-1, token.id-1))
                    deprel = sent_ann.dependencies.shortest_path(head.id-1, token.id-1)[0][1] if sent_ann.dependencies.shortest_path(head.id-1, token.id-1) else None
                    # print (deprel)
                    token.real_deprel = deprel
                    event_tree[head].append(token)
            try:
                parse_event_tree(event_tree, sent[0])
            except Exception as e:
                print (e)
        event_set = set()
        for e in events:
            line = e[0]+"\t"+e[1]+":"+e[2]
            k_list = list(e[-1].keys())
            if e[1] == "Binding":
                theme_list = e[-1]["Theme"]
                if theme_list:
                    binding_group = defaultdict(list)
                    for t in theme_list:
                        binding_group[t[1]].append(t[0])
                    tuples = list(itertools.product(*list(binding_group.values())))
                    temp = None
                    theme_list = list()
                    for ts in tuples:
                        temp = ts[0]
                        if len(ts) > 1:
                            for i, t in enumerate(ts[1:], start=2):
                                temp += " Theme"+str(i)+":"+t
                        theme_list.append((temp,None))
                    e[-1]["Theme"] = theme_list
            if "Theme" in k_list:
                k_list.insert(0, k_list.pop(k_list.index("Theme")))
            k_list = [k for k in k_list if k == "Theme" or k == "Cause"]
            for k in k_list:
                line += " "+k+":"+e[-1][k][0][0]
            if str(line.split("\t")[1]) not in event_set:
                t1.write(line+"\n")
                event_set.add(str(line.split("\t")[1]))
            for k in k_list:
                l = e[-1][k]
                pre_v = e[-1][k][0][0]
                pre_eid = e[0]
                for v in l[1:]:
                    line = line.replace(pre_eid, "E"+str(event_count))
                    line = line.replace(" "+k+":"+pre_v, " "+k+":"+v[0])
                    if str(line.split("\t")[1]) not in event_set:
                        t1.write(line+"\n")
                        event_set.add(str(line.split("\t")[1]))
                        for e2 in events:
                            for k2 in e2[-1]:
                                for s in e2[-1][k2]:
                                    if pre_eid == s[0]:
                                        e2[-1][k2].append(("E"+str(event_count),s[1]))
                                        break
                        pre_v = v[0]
                        pre_eid = "E"+str(event_count)
                        event_count += 1
        write_conllx(root + '.conllx', curr_sents)
        t1.close()
        print ("./a2-evaluate.pl -g gold-sam/ -s "+root + '.a2.t1')
        os.system("./a2-evaluate.pl -g gold-sam/ -s "+root + '.a2.t1')