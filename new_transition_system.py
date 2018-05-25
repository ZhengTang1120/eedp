from collections import namedtuple

Arc = namedtuple('Arc', 'head dependent label')

class CustomTransitionSystem:

    def __init__(self, sentence):
        self.sentence = sentence
        self.stack = [sentence[0]] # root
        self.buffer = sentence[1:] # tokens except root
        self.arcs = []
        self.tags = {}
        self.i2t = ['shift', 'left_reduce', 'right_reduce', 'left_attach', 'right_attach', 'swap', 'drop']
        self.t2i = {t:i for i,t in enumerate(self.i2t)}

    def is_terminal(self):
        return len(self.buffer) == 0 and len(self.stack) == 1 and self.stack[0].id == 0

    def count_arcs(self, a, b):
        c = 0
        for arc in self.arcs:
            if arc.head.id == a.id and arc.dependent.id == b.id:
                c += 1
            # if arc.head.id == b.id and arc.dependent.id == a.id:
                # c += 1
        return c

    def is_legal(self, transition):
        if transition == 'shift':
            # buffer can't be empty
            return len(self.buffer) > 0

        elif transition == 'left_reduce':
            # there must be at least two things in the stack
            # s1 can't be root
            # s0 and s1 shouldn't be part of any arcs
            if len(self.stack) < 2:
                return False
            s0 = self.stack[-1]
            s1 = self.stack[-2]
            if s1.id == 0:
                return False
            return self.count_arcs(s1, s0) == 0

        elif transition == 'right_reduce':
            # there must be at least two things in the stack
            # s0 and s1 shouldn't be part of any arcs
            if len(self.stack) < 2:
                return False
            s0 = self.stack[-1]
            s1 = self.stack[-2]
            # if len(self.buffer) == 0:
            #     return True
            return self.count_arcs(s0, s1) == 0

        elif transition == 'left_attach':
            # there must be at least two things in the stack
            # s1 can't be root
            # s0 and s1 shouldn't be part of any arcs
            if len(self.stack) < 2:
                return False
            s0 = self.stack[-1]
            s1 = self.stack[-2]
            if s1.id == 0:
                return False
            return self.count_arcs(s0, s1) == 0 and self.count_arcs(s1, s0) == 0

        elif transition == 'right_attach':
            # there must be at least two things in the stack
            # s0 and s1 shouldn't be part of any arcs
            if len(self.stack) < 2: # there must be at least 2 things in the stack
                return False
            s0 = self.stack[-1]
            s1 = self.stack[-2]
            return self.count_arcs(s0, s1) == 0 and self.count_arcs(s1, s0) == 0

        elif transition == 'swap':
            return False
            # can't swap things that have already been swapped
            return len(self.stack) >= 2 and 0 < self.stack[-2].id < self.stack[-1].id

        elif transition == 'drop':
            # can't drop if b is already part of an arc
            if len(self.buffer) == 0:
                return False
            b = self.buffer[0]
            return not any(a for a in self.arcs if a.head.id == b.id or a.dependent.id == b.id)

        else:
            return False

    def all_legal(self):
        return filter(self.is_legal, self.i2t)

    def all_correct(self):
        return filter(self.is_correct, self.all_legal())

    def is_legal_and_correct(self, transition):
        return self.is_legal(transition) and self.is_correct(transition)

    def is_correct(self, transition):
        if transition == 'left_attach':
            if len(self.stack[-2].parent_id) < 2:
                # don't use left_attach if s1 has less than two parents
                return False
            elif self.is_legal('left_reduce') and self.cost('left_reduce') == 0:
                # don't use left_attach if left_reduce is available
                return False
        elif transition == 'right_attach':
            if len(self.stack[-1].parent_id) < 2:
                # don't use left_attach if s0 has less than two parents
                return False
            elif self.is_legal('right_reduce') and self.cost('right_reduce') == 0:
                # don't use right_attach if right_reduce is available
                return False
        # always prefer to reduce
        if transition != 'left_reduce' and self.is_legal('left_reduce') and self.cost('left_reduce') == 0:
            return False
        elif transition != 'right_reduce' and self.is_legal('right_reduce') and self.cost('right_reduce') == 0:
            return False
        elif transition not in ('left_reduce', 'right_reduce', 'left_attach', 'right_attach'):
            # if you can't reduce, prefer to attach
            if transition != 'left_attach' and self.is_legal_and_correct('left_attach'):
                return False
            elif transition != 'right_attach' and self.is_legal_and_correct('right_attach'):
                return False
        # this if is independent of the one, because of the nested if above
        if transition not in ('drop', 'left_reduce', 'right_reduce', 'left_attach', 'right_attach') and self.is_legal('drop') and self.cost('drop') == 0:
            # prefer drop to the alternatives
            return False
        if transition == 'shift' and self.is_legal('swap') and self.cost('swap') == 0:
            return False
        return self.cost(transition) == 0

    def cost(self, transition):
        if transition == 'shift':
            b = self.buffer[0]
            for s in self.sentence:
                if b.id in s.parent_id or (s.id == b.id and s.parent_id != [-1]):
                    return 0
            return 1

        elif transition == 'left_reduce':
            s0 = self.stack[-1]
            s1 = self.stack[-2]
            c = 0 if s0.id in s1.parent_id else 1
            # count dependents of s1 that haven't been attached
            for s in self.sentence:
                if s1.id in s.parent_id:
                    if not any(a for a in self.arcs if a.head.id == s1.id and a.dependent.id == s.id):
                        c += 1
            # count parents of s1 that haven't been attached
            for pid in s1.parent_id:
                if pid != s0.id:
                    if not any(a for a in self.arcs if a.head.id == pid and a.dependent.id == s1.id):
                        c += 1
            return c

        elif transition == 'right_reduce':
            s0 = self.stack[-1]
            s1 = self.stack[-2]
            c = 0 if s1.id in s0.parent_id else 1
            # count dependents of s0 that haven't been attached
            for s in self.sentence:
                if s0.id in s.parent_id:
                    if not any(a for a in self.arcs if a.head.id == s0.id and a.dependent.id == s.id):
                        c += 1
            # count parents of s0 that haven't been attached
            for pid in s0.parent_id:
                if pid != s1.id:
                    if not any(a for a in self.arcs if a.head.id == pid and a.dependent.id == s0.id):
                        c += 1
            return c

        elif transition == 'left_attach':
            s0 = self.stack[-1]
            s1 = self.stack[-2]
            c = 0 if s0.id in s1.parent_id else 1
            return c

        elif transition == 'right_attach':
            s0 = self.stack[-1]
            s1 = self.stack[-2]
            c = 0 if s1.id in s0.parent_id else 1
            return c

        elif transition == 'swap':
            s0 = self.stack[-1]
            s1 = self.stack[-2]
            sigma = self.stack[1:-2]
            for entry in sigma:
                if s0.id in entry.parent_id and not self.children_left(entry):
                    return 0
                elif entry.id in s0.parent_id and not self.children_left(s0):
                    return 0
            return 1

        elif transition == 'drop':
            # count the number of gold arcs that involve b
            c = 0
            b = self.buffer[0]
            for s in self.sentence:
                if b.id in s.parent_id or (s.id == b.id and s.parent_id != [-1]):
                    c += 1
            return c

    def children_left(self, parent):
        for child in self.sentence[1:]:
            if parent.id in child.parent_id:
                i = child.parent_id.index(parent.id)
                label = child.relation[i]
                arc = Arc(parent, child, label)
                if arc not in self.arcs:
                    return True
        return False

    def get_arc_label_for_transition(self, transition):
        if transition in ('left_reduce', 'left_attach'):
            s0 = self.stack[-1]
            s1 = self.stack[-2]
            i = s1.parent_id.index(s0.id)
            return s1.relation[i]
        elif transition in ('right_reduce', 'right_attach'):
            s0 = self.stack[-1]
            s1 = self.stack[-2]
            i = s0.parent_id.index(s1.id)
            return s0.relation[i]

    def get_token_label_for_transition(self, transition):
        if transition == 'drop':
            return 'O'
        elif transition == 'shift':
            b = self.buffer[0]
            # we store the entity/token label in the ConllEntry.feats field
            return b.feats

    def perform_transition(self, transition, relation=None, label=None):
        if transition == 'shift':
            b = self.buffer.pop(0)
            self.tags[b.id] = label
            b.pred_feats = label
            self.stack.append(b)

        elif transition == 'left_reduce':
            s0 = self.stack[-1]
            s1 = self.stack[-2]
            arc = Arc(s0, s1, relation)
            self.arcs.append(arc)
            s1.pred_parent_id.append(s0.id)
            s1.pred_relation.append(relation)
            self.stack.pop(-2)

        elif transition == 'right_reduce':
            s0 = self.stack[-1]
            s1 = self.stack[-2]
            if self.count_arcs(s1, s0) == 0:
                arc = Arc(s1, s0, relation)
                self.arcs.append(arc)
                s0.pred_parent_id.append(s1.id)
                s0.pred_relation.append(relation)
            self.stack.pop()

        elif transition == 'left_attach':
            s0 = self.stack[-1]
            s1 = self.stack[-2]
            arc = Arc(s0, s1, relation)
            self.arcs.append(arc)
            s1.pred_parent_id.append(s0.id)
            s1.pred_relation.append(relation)

        elif transition == 'right_attach':
            s0 = self.stack[-1]
            s1 = self.stack[-2]
            arc = Arc(s1, s0, relation)
            self.arcs.append(arc)
            s0.pred_parent_id.append(s1.id)
            s0.pred_relation.append(relation)
            self.buffer.insert(0, self.stack.pop())

        elif transition == 'swap':
            s1 = self.stack.pop(-2)
            self.buffer.insert(0, s1)

        elif transition == 'drop':
            b = self.buffer.pop(0)
            b.pred_feats = 'O'
            b.pred_parent_id = [-1]
            b.pred_relation = ['none']
            self.tags[b.id] = 'O'
