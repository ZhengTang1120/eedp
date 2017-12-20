class ArcHybrid:

    def __init__(self, sentence):
        self.stack = []
        self.buffer = sentence[1:] + [sentence[0]] # arc-hybrid expects root at end of buffer
        self.i2t = ['shift', 'left_arc', 'right_arc']
        self.t2i = {t:i for i,t in enumerate(self.i2t)}

    def is_terminal(self):
        return len(self.stack) == 0 and len(self.buffer) == 1

    def is_legal(self, transition):
        if transition == 'shift':
            return len(self.buffer) > 0 and self.buffer[0].id != 0
        elif transition == 'left_arc':
            return len(self.stack) > 0 and len(self.buffer) > 0
        elif transition == 'right_arc':
            return len(self.stack) > 1 and self.stack[-1].id != 0
        else:
            return False

    def all_legal(self):
        return filter(self.i2t, self.is_legal)

    def all_correct(self):
        return filter(self.all_legal(), self.is_correct)

    # this is only valid for legal transitions
    def is_correct(self, transition):
        return self.cost(transition) == 0

    # this is only valid for legal transitions
    def cost(self, transition):
        if transition == 'shift':
            b = self.buffer[0]
            c = 0
            # pushing b onto the stack means that
            # b will not be able to acquire heads from {s1} U sigma
            c += sum(1 for h in self.stack[:-1] if h.id == b.parent_id)
            # and will not be able to acquire dependents from {s0, s1} U sigma
            c += sum(1 for d in self.stack if d.parent_id == b.id)
            return c
        elif transition == 'left_arc':
            s0 = self.stack[-1]
            s1 = [self.stack[-2]] if len(self.stack) > 1 else []
            b = [self.buffer[0]]
            beta = self.buffer[1:]
            c = 0
            # adding the arc (b, s0) and popping s0 from the stack means that
            # s0 will not be able to acquire heads from {s1} U beta
            c += sum(1 for h in s1 + beta if h.id == s0.parent_id)
            # and will not be able to acquire dependents from {b} U beta
            c += sum(1 for d in b + beta if d.parent_id == s0.id)
            return c
        elif transition == 'right_arc':
            s0 = self.stack[-1]
            c = 0
            # adding the arc (s1, s0) and popping s0 from the stack means that
            # s0 will not be able to acquire heads or dependents from {b} U beta
            c += sum(1 for h in self.buffer if h.id == s0.parent_id)
            c += sum(1 for d in self.buffer if d.parent_id == s0.id)
            return c

    # this is only valid for legal transitions
    def perform_transition(self, transition, relation=None):
        if transition == 'shift':
            self.stack.append(self.buffer.pop(0))
        elif transition == 'left_arc':
            child = self.stack.pop()
            parent = self.buffer[0]
            child.pred_parent_id = parent.id
            child.pred_relation = relation
        elif transition == 'right_arc':
            child = self.stack.pop()
            parent = self.stack[-1]
            child.pred_parent_id = parent.id
            child.pred_relation = relation



class ArcHybridWithDrop(ArcHybrid):

    def __init__(self, sentence):
        super().__init__(sentence)
        self.t2i['drop'] = len(self.i2t)
        self.i2t.append('drop')

    def is_legal(self, transition):
        if transition == 'drop':
            return len(self.buffer) > 0
        else:
            return super().is_legal(transition)

    def cost(self, transition):
        if transition == 'drop':
            b = self.buffer[0]
            all_elements = self.buffer + self.stack
            c = 0
            c += sum(1 for h in all_elements if h.id == b.parent_id)
            c += sum(1 for d in all_elements if d.parent_id == b.id)
            return c
        else:
            return super().cost(transition)

    def perform_transition(self, transition, relation=None):
        if transition == 'drop':
            self.buffer.pop(0)
        else:
            super().perform_transition(transition, relation)
