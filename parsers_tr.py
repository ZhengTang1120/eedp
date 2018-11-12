import torch
from torch.autograd import Variable
import torch.nn as nn
import random
import pickle
import torch.optim as optim


class ArcHybridParser(nn.Module):
    def __init__(self, word_count, words, tags, chars, entities, dep_relations, ev_relations, w_embed_size, 
        t_embed_size, c_embed_size, clstm_hidden_size, e_embed_size, lstm_hidden_size, lstm_num_layers, 
        dep_op_hidden_size, dep_lbl_hidden_size, ev_op_hidden_size, ev_lbl_hidden_size, 
        tg_lbl_hidden_size, alpha, p_explore, pretrained):
        super(ArcHybridParser, self).__init__()

        self.pretrained = pretrained
        if self.pretrained:
            if dep_relations:
                self.embeddings = np.loadtxt("pubmedm13.txt")
            else:
                self.embeddings = np.loadtxt("pubmed.txt")

        # counts used for word dropout
        self.word_count = word_count

        # mappings from ids to terms
        self.i2w = words
        self.i2t = tags
        self.i2c = chars
        self.i2e = ["Protein", "O", "*pad*"]
        self.dep_relations = dep_relations
        self.ev_relations = ev_relations
        if entities:
            self.i2tg = entities

        # mapings from terms to ids
        self.w2i = {w:i for i,w in enumerate(words)}
        self.t2i = {t:i for i,t in enumerate(tags)}
        self.e2i = {e:i for i,e in enumerate(self.i2e)}
        self.c2i = {c:i for i,c in enumerate(chars)}
        if entities:
            self.tg2i = {e:i for i,e in enumerate(self.i2tg)}

        self.w_embed_size = w_embed_size
        self.t_embed_size = t_embed_size
        self.c_embed_size = c_embed_size
        self.clstm_hidden_size = clstm_hidden_size
        self.e_embed_size = e_embed_size
        self.lstm_hidden_size = lstm_hidden_size * 2 # must be even for BiLSTM
        self.lstm_num_layers = lstm_num_layers
        self.dep_op_hidden_size = dep_op_hidden_size
        self.dep_lbl_hidden_size = dep_lbl_hidden_size
        self.ev_op_hidden_size = ev_op_hidden_size
        self.ev_lbl_hidden_size = ev_lbl_hidden_size
        self.tg_lbl_hidden_size = tg_lbl_hidden_size
        self.alpha = alpha
        self.p_explore = p_explore
        self.entities = entities


        # words and tags, entities embeddings
        self.wlookup = nn.Embedding(len(self.i2w), self.w_embed_size)
        if self.pretrained:
            embed.weight.data.copy_(torch.from_numpy(self.embeddings))
        self.tlookup = nn.Embedding(len(self.i2t), self.t_embed_size)
        self.clookup = nn.Embedding(len(self.i2c), self.c_embed_size)
        self.elookup = nn.Embedding(len(self.i2e), self.e_embed_size)

        self.bilstm = nn.LSTM(self.w_embed_size + self.t_embed_size + self.clstm_hidden_size + self.e_embed_size, 
            self.lstm_hidden_size // 2, self.lstm_num_layers, bidirectional=True)
        self.bihidden = self.init_hidden(self.lstm_num_layers * 2, self.lstm_hidden_size // 2)
        self.word_to_lstm = nn.Linear(self.w_embed_size + self.t_embed_size + self.clstm_hidden_size + self.e_embed_size, self.lstm_hidden_size)

        self.clstm = nn.LSTM(self.c_embed_size, self.clstm_hidden_size, self.lstm_num_layers)
        self.chidden = self.init_hidden(self.lstm_num_layers, self.clstm_hidden_size)
        self.char_to_lstm = nn.Linear(self.c_embed_size, self.clstm_hidden_size)

        if self.dep_relations:
            # fully connected network with one hidden layer
            # to predict the transition to take next
            out_size = 3 # shift, left_arc, right_arc
            self.dep_op_hidden      = nn.Linear(self.lstm_hidden_size * 7, self.dep_op_hidden_size)
            self.dep_op_output      = nn.Linear(self.dep_op_hidden_size, out_size)

            # # fully connected network with one hidden layer
            # # to predict the arc label
            out_size = 1 + len(self.dep_relations) * 2
            self.dep_lbl_hidden      = nn.Linear(self.lstm_hidden_size * 7, self.dep_lbl_hidden_size)
            self.dep_lbl_output      = nn.Linear(self.dep_lbl_hidden_size, out_size)
        if self.ev_relations:
            # fully connected network with one hidden layer
            # to predict the transition to take next
            out_size = 7 # shift, left_reduce, right_reduce, left_attach, right_attach, swap, drop
            self.ev_op_hidden      = nn.Linear(self.lstm_hidden_size * 7, self.ev_op_hidden_size)
            self.ev_op_output      = nn.Linear(self.ev_op_hidden_size, out_size)

            # fully connected network with one hidden layer
            # to predict the arc label
            out_size = 1 + len(self.ev_relations) * 2
            self.ev_lbl_hidden      = nn.Linear(self.lstm_hidden_size * 7, self.ev_lbl_hidden_size)
            self.ev_lbl_output      = nn.Linear(self.ev_lbl_hidden_size, out_size)
        if self.entities:
            # fully connected network with one hidden layer
            # to predict the trigger label
            out_size = 1 + len(self.i2tg)
            self.tg_lbl_hidden      = nn.Linear(self.lstm_hidden_size * 7, self.tg_lbl_hidden_size)
            self.tg_lbl_output      = nn.Linear(self.tg_lbl_hidden_size, out_size)


        self.optimizer_ee = optim.SGD(self.parameters(), lr=0.01)
        self.optimizer_dp = optim.SGD(self.parameters(), lr=0.01)
        self.criterion = nn.HingeEmbeddingLoss()

    def init_hidden(self, num_layers, hidden_size):
        return (torch.zeros(num_layers, 1, hidden_size),
                torch.zeros(num_layers, 1, hidden_size))

    def save(self, name):
        params = (
            self.word_count, self.i2w, self.i2t, self.i2c,
            self.entities, self.dep_relations, self.ev_relations,
            self.w_embed_size, self.t_embed_size,
            self.c_embed_size, self.clstm_hidden_size, self.e_embed_size,
            self.lstm_hidden_size // 2, self.lstm_num_layers,
            self.dep_op_hidden_size, self.dep_lbl_hidden_size,
            self.ev_op_hidden_size, self.ev_lbl_hidden_size,
            self.tg_lbl_hidden_size,
            self.alpha, self.p_explore, self.pretrained
        )
        # save model
        torch.save(the_model.state_dict(), f'{name}.model')
        # save pickle
        with open(f'{name}.pickle', 'wb') as f:
            pickle.dump(params, f)

    @staticmethod
    def load(name):
        with open(f'{name}.pickle', 'rb') as f:
            params = pickle.load(f)
            parser = ArcHybridParser(*params)
            parser.load_state_dict(torch.load(f'{name}.model'))
            return parser

    def set_empty_vector(self):
        w_pad = self.wlookup(torch.tensor(self.w2i['*pad*']))
        t_pad = self.tlookup(torch.tensor(self.t2i['*pad*']))
        c_pad = self.clookup(torch.tensor(self.c2i['*pad*']))
        c_pad = self.char_to_lstm(c_pad)
        e_pad = self.elookup(torch.tensor(self.e2i['*pad*']))
        v_pad = torch.cat((w_pad, t_pad, c_pad, e_pad))
        i_vec = self.word_to_lstm(v_pad)
        self.empty = torch.tanh(i_vec)
        self.zero_grad()
        # return torch.tanh(i_vec)

    def extract_features(self, sentence, drop_word=False):
        unk = self.w2i['*unk*']
        inputs = []
        for entry in sentence:
            self.chidden = self.init_hidden(self.lstm_num_layers, self.clstm_hidden_size)
            # should we drop the word?
            if drop_word:
                c = self.word_count.get(entry.norm, 0)
                drop_word = random.random() < self.alpha / (c + self.alpha)
            # get word and tag ids
            w_id = torch.tensor(unk) if drop_word else torch.tensor(self.w2i.get(entry.norm, unk))
            c_seq = list()
            for c in entry.norm:
                c_id = torch.tensor(self.c2i.get(c, self.c2i['*unk*']))
                c_v = self.clookup(c_id)
                c_seq.append(c_v)
            c_seq = torch.stack(c_seq)
            coutput, self.chidden = self.clstm(c_seq.view(len(entry.norm), 1, -1), self.chidden)
            c_vec = coutput[-1].view(-1)
            t_id = torch.tensor(self.t2i[entry.postag])
            e_id = torch.tensor(self.e2i[entry.feats] if entry.feats == "Protein" else self.e2i["O"])
            # get word and tag embbedding in the corresponding entry
            w_vec = self.wlookup(w_id)
            t_vec = self.tlookup(t_id)
            e_vec = self.elookup(e_id)
            i_vec = torch.cat((w_vec, t_vec, c_vec, e_vec))
            inputs.append(i_vec)
        inputs = torch.stack(inputs)
        outputs, self.bihidden = self.bilstm(inputs.view(len(sentence), 1, -1), self.bihidden)
        self.features = outputs.view(len(sentence),-1)

    def _evaluate_dependencies(self, stack, buffer):
        def get_children_avg(children):
            if len(children) > 0:
                return torch.mean(torch.stack([self.features[c] for c in children]), dim=0)
            else:
                return self.empty

        # construct input vector
        b = self.features[buffer[0].id] if len(buffer) > 0 else self.empty
        s0 = self.features[stack[-1].id] if len(stack) > 0 else self.empty
        s1 = self.features[stack[-2].id] if len(stack) > 1 else self.empty
        s2 = self.features[stack[-3].id] if len(stack) > 2 else self.empty
        s0c = get_children_avg(stack[-1].children) if len(stack) > 0 else self.empty
        s1c = get_children_avg(stack[-2].children) if len(stack) > 1 else self.empty
        s2c = get_children_avg(stack[-3].children) if len(stack) > 2 else self.empty
        input = torch.cat((b, s0, s1, s2, s0c, s1c, s2c))
        # predict action
        op_hidden = torch.tanh(self.dep_op_hidden(input))
        op_output = self.dep_op_output(op_hidden)
        # predict label
        lbl_hidden = torch.tanh(self.dep_lbl_hidden(input))
        lbl_output = self.dep_lbl_output(lbl_hidden)
        # return scores
        return op_output, lbl_output

    def _evaluate_events(self, stack, buffer):

        def get_children_avg(children):
            if len(children) > 0:
                return torch.mean(torch.stack([self.features[c] for c in children]), dim=0)
            else:
                return self.empty


        # construct input vector
        b = self.features[buffer[0].id] if len(buffer) > 0 else self.empty
        s0 = self.features[stack[-1].id] if len(stack) > 0 else self.empty
        s1 = self.features[stack[-2].id] if len(stack) > 1 else self.empty
        s2 = self.features[stack[-3].id] if len(stack) > 2 else self.empty
        s0c = get_children_avg(stack[-1].children) if len(stack) > 0 else self.empty
        s1c = get_children_avg(stack[-2].children) if len(stack) > 1 else self.empty
        s2c = get_children_avg(stack[-3].children) if len(stack) > 2 else self.empty
        input = torch.cat((b, s0, s1, s2, s0c, s1c, s2c))
        # predict action
        op_hidden = torch.tanh(self.ev_op_hidden(input))
        op_output = self.ev_op_output(op_hidden)
        # predict label
        lbl_hidden = torch.tanh(self.ev_lbl_hidden(input))
        lbl_output = self.ev_lbl_output(lbl_hidden)
        # predict trigger label
        tg_hidden = torch.tanh(self.tg_lbl_hidden(input))
        tg_output = self.tg_lbl_output(tg_hidden)
        # return scores
        return op_output, lbl_output, tg_output

    def forward(self, stack, buffer, is_event):
        if is_event:
            res = self._evaluate_events(stack, buffer)
        else:
            res = self._evaluate_dependencies(stack, buffer)
        return res