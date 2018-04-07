import re
from collections import Counter



def read_conllx(filename, non_proj=False):
    """Reads dependency annotations from CoNLL-X format"""
    return list(gen_conllx(filename, non_proj))



def write_conllx(filename, sentences):
    """Write sentences to conllx file"""
    with open(filename, 'w') as f:
        for i, sentence in enumerate(sentences):
            if i > 0:
                f.write('\n')
            for entry in sentence:
                if entry.id > 0: # skip root added by gen_conllx()
                    f.write(str(entry) + '\n')



def normalize(word, vocabulary=None):
    """returns a normalized version of the given word"""
    if vocabulary is not None and word not in vocabulary:
        return '*unk*'
    elif re.fullmatch(r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?', word):
        return '*num*'
    else:
        return word.lower()



def make_vocabularies(sentences):
    """gets a corpus and returns (word counts, words, tags, relations)"""
    word_count = Counter()
    tag_count = Counter()
    relation_count = Counter()
    char_count = Counter()
    for sentence in sentences:
        word_count.update(e.norm for e in sentence)
        for e in sentence:
            char_count.update(c for c in e.norm)
        tag_count.update(e.postag for e in sentence)
        relation_count.update(e.relation for e in sentence)
    special = ['*unk*', '*pad*']
    words = special + list(word_count.keys())
    tags = special + list(tag_count.keys())
    rels = list(relation_count.keys())
    chars = special + list(char_count.keys())
    return (word_count, words, tags, chars, rels)

def make_vocabularies3(events):
    """gets a corpus and returns (word counts, words, tags, relations)"""
    word_count = Counter()
    tag_count = Counter()
    relation_count = Counter()
    entity_count = Counter()
    char_count = Counter()
    for event in events:
        if len(event) > 2:
            word_count.update(e.norm for e in event)
            for e in event:
                char_count.update(c for c in e.norm)
            tag_count.update(e.postag for e in event)
            for e in event:
                relation_count.update(r for r in e.relations if r != 'none')
            entity_count.update(e.feats for e in event)
    special = ['*unk*', '*pad*']
    words = special + list(word_count.keys())
    tags = special + list(tag_count.keys())
    rels = list(relation_count.keys())
    entities = list(entity_count.keys())
    chars = special + list(char_count.keys())
    return (word_count, words, tags, chars, entities, rels)

def make_vocabularies2(sentences, events):
    """gets a corpus and returns (word counts, words, tags, dependencies, events)"""
    word_count = Counter()
    tag_count = Counter()
    dep_relation_count = Counter()
    ev_relation_count = Counter()
    entity_count = Counter()
    char_count = Counter()
    for sentence in sentences:
        word_count.update(e.norm for e in sentence)
        for e in sentence:
            char_count.update(c for c in e.norm)
        tag_count.update(e.postag for e in sentence)
        dep_relation_count.update(e.relation for e in sentence)
    for event in events:
        word_count.update(e.norm for e in event)
        for e in event:
            char_count.update(c for c in e.norm)
        char_count.update(c for c in e.norm for e in event)
        tag_count.update(e.postag for e in event)
        for e in event:
            ev_relation_count.update(r for r in e.relations if r != 'none')
        entity_count.update(e.feats for e in event)
    special = ['*unk*', '*pad*']
    words = special + list(word_count.keys())
    tags = special + list(tag_count.keys())
    dep_rels = list(dep_relation_count.keys())
    ev_rels = list(ev_relation_count.keys())
    entities = list(entity_count.keys())
    chars = special + list(char_count.keys())
    return (word_count, words, tags, chars, entities, ev_rels, dep_rels)



def gen_conllx(filename, non_proj=False):
    """
    Reads dependency annotations in CoNLL-X format and returns a generator.
    """
    read = 0
    dropped = 0
    root = ConllEntry(id=0, form='*root*', postag='*root*', head=[], deprel=[], feats='O')
    with open(filename) as f:
        sentence = [root]
        for line in f:
            if line.isspace() and len(sentence) > 1:
                sentence[0].doc_from = sentence[1].doc_from
                yield sentence
                read += 1
                sentence = [root]
                continue
            entry = ConllEntry.from_line(line)
            sentence.append(entry)
        # we may still have one sentence in memory
        # if the file doesn't end in an empty line
        if len(sentence) > 1:
            yield sentence
            read += 1
    print(f'{read:,} sentences read.')

class ConllEntry:
    """
    Represents an annotation corresponding to a single word.
    See http://anthology.aclweb.org/W/W06/W06-2920.pdf
    """

    def __init__(self, id=None, form=None, lemma=None, cpostag=None,
                 postag=None, feats=None, head=None, deprel=None,
                 phead=None, pdeprel=None, doc_from=None):
        self.id = id
        self.form = form
        self.norm = normalize(form)
        self.lemma = lemma
        self.cpostag = cpostag
        self.postag = postag
        self.feats = feats
        self.head = head
        self.deprel = deprel
        self.phead = phead
        self.pdeprel = pdeprel
        self.doc_from = doc_from
        # aliases
        self.parent_ids = self.head
        self.relations = self.deprel
        self.brat_label = self.feats
        self.pred_parent_ids = []
        self.pred_relations = []

    def __repr__(self):
        return '<ConllEntry: %s>' % self.form

    def __str__(self):
        fields = [
            self.id,
            self.form,
            self.lemma,
            self.cpostag,
            self.postag,
            self.feats,
            self.head,
            self.deprel,
            self.phead,
            self.pdeprel,
            self.doc_from
        ]
        return '\t'.join('_' if f is None else str(f) for f in fields)

    @staticmethod
    def from_line(line):
        [id, form, lemma, cpostag, postag, feats, head, deprel, phead, pdeprel, doc_from] = line.strip().split('\t')
        id = int(id)
        lemma = lemma if lemma != '_' else None
        feats = feats if feats != '_' else None
        head = eval(head)
        phead = int(phead) if phead != '_' else None
        pdeprel = pdeprel if pdeprel != '_' else None
        deprel = eval(deprel) if deprel != "skipped" else "skipped"
        return ConllEntry(id, form, lemma, cpostag, postag, feats, head, deprel, phead, pdeprel, doc_from)
