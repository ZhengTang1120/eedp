import time
import random
import pickle
from operator import itemgetter
import dynet as dy
import numpy as np
from transition_systems import ArcHybrid

class ArcHybridParser:

    def __init__(self, word_count, words, tags, relations,
            w_embed_size, t_embed_size,
            lstm_hidden_size, lstm_num_layers,
            act_hidden_size, lbl_hidden_size,
            alpha, p_explore):

        # counts used for word dropout
        self.word_count = word_count

        # mappings from ids to terms
        self.i2w  = words
        self.i2t  = tags
        self.i2r  = relations

        # mapings from terms to ids
        self.w2i  = {w:i for i,w in enumerate(words)}
        self.t2i  = {t:i for i,t in enumerate(tags)}
        self.r2i  = {r:i for i,r in enumerate(relations)}

        self.w_embed_size = w_embed_size
        self.t_embed_size = t_embed_size
        self.lstm_hidden_size = lstm_hidden_size * 2
        self.lstm_num_layers = lstm_num_layers
        self.act_hidden_size = act_hidden_size
        self.lbl_hidden_size = lbl_hidden_size
        self.alpha = alpha
        self.p_explore = p_explore

        self.empty = None

        self.start_model()

    def start_model(self):

        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)

        # words and tags embeddings
        self.wlookup = self.model.add_lookup_parameters((len(self.i2w), self.w_embed_size))
        self.tlookup = self.model.add_lookup_parameters((len(self.i2t), self.t_embed_size))

        # feature extractor
        self.bilstm = dy.BiRNNBuilder(
                self.lstm_num_layers,
                self.w_embed_size + self.t_embed_size,
                self.lstm_hidden_size,
                self.model,
                dy.VanillaLSTMBuilder,
        )

        # transform word+pos vector into a vector similar to the lstm output
        # used to generate padding vectors
        self.word_to_lstm      = self.model.add_parameters((self.lstm_hidden_size, self.w_embed_size + self.t_embed_size))
        self.word_to_lstm_bias = self.model.add_parameters((self.lstm_hidden_size))

        # fully connected network with one hidden layer
        # to predict the transition to take next
        out_size = 3 # shift, left_arc, right_arc
        self.act_hidden      = self.model.add_parameters((self.act_hidden_size, self.lstm_hidden_size * 4))
        self.act_hidden_bias = self.model.add_parameters((self.act_hidden_size))
        self.act_output      = self.model.add_parameters((out_size, self.act_hidden_size))
        self.act_output_bias = self.model.add_parameters((out_size))

        # fully connected network with one hidden layer
        # to predict the arc label
        out_size = 1 + len(self.i2r) * 2
        self.lbl_hidden      = self.model.add_parameters((self.lbl_hidden_size, self.lstm_hidden_size * 4))
        self.lbl_hidden_bias = self.model.add_parameters((self.lbl_hidden_size))
        self.lbl_output      = self.model.add_parameters((out_size, self.lbl_hidden_size))
        self.lbl_output_bias = self.model.add_parameters((out_size))

    def save(self, name):
        # save model
        self.model.save(f'{name}.model')
        # keep copies of model related stuff
        model = self.model
        trainer = self.trainer
        empty = self.empty
        wlookup = self.wlookup
        tlookup = self.tlookup
        bilstm = self.bilstm
        word_to_lstm = self.word_to_lstm
        word_to_lstm_bias = self.word_to_lstm_bias
        act_hidden = self.act_hidden
        act_hidden_bias = self.act_hidden_bias
        act_output = self.act_output
        act_output_bias = self.act_output_bias
        lbl_hidden = self.lbl_hidden
        lbl_hidden_bias = self.lbl_hidden_bias
        lbl_output = self.lbl_output
        lbl_output_bias = self.lbl_output_bias
        # set to None before pickling
        self.model = None
        self.trainer = None
        self.empty = None
        self.wlookup = None
        self.tlookup = None
        self.bilstm = None
        self.word_to_lstm = None
        self.word_to_lstm_bias = None
        self.act_hidden = None
        self.act_hidden_bias = None
        self.act_output = None
        self.act_output_bias = None
        self.lbl_hidden = None
        self.lbl_hidden_bias = None
        self.lbl_output = None
        self.lbl_output_bias = None
        # save pickle
        with open(f'{name}.pickle', 'wb') as f:
            pickle.dump(self, f)
        # restore model
        self.model = model
        self.trainer = trainer
        self.empty = empty
        self.wlookup = wlookup
        self.tlookup = tlookup
        self.bilstm = bilstm
        self.word_to_lstm = word_to_lstm
        self.word_to_lstm_bias = word_to_lstm_bias
        self.act_hidden = act_hidden
        self.act_hidden_bias = act_hidden_bias
        self.act_output = act_output
        self.act_output_bias = act_output_bias
        self.lbl_hidden = lbl_hidden
        self.lbl_hidden_bias = lbl_hidden_bias
        self.lbl_output = lbl_output
        self.lbl_output_bias = lbl_output_bias

    @staticmethod
    def load(name):
        with open(f'{name}.pickle', 'rb') as f:
            obj = pickle.load(f)
            obj.start_model()
            obj.model.populate(f'{name}.model')
            return obj

    def set_empty_vector(self):
        w_pad = self.wlookup[self.w2i['*pad*']]
        t_pad = self.tlookup[self.t2i['*pad*']]
        v_pad = dy.concatenate([w_pad, t_pad])
        i_vec = self.word_to_lstm.expr() * v_pad + self.word_to_lstm_bias.expr()
        self.empty = dy.tanh(i_vec)

    def extract_features(self, sentence, drop_word=False):
        unk = self.w2i['*unk*']
        inputs = []
        for entry in sentence:
            # should we drop the word?
            if drop_word:
                c = self.word_count.get(entry.norm, 0)
                drop_word = random.random() < self.alpha / (c + self.alpha)
            # get word and tag ids
            w_id = unk if drop_word else self.w2i.get(entry.norm, unk)
            t_id = self.t2i[entry.postag]
            # get word and tag embbedding in the corresponding entry
            w_vec = self.wlookup[w_id]
            t_vec = self.tlookup[t_id]
            i_vec = dy.concatenate([w_vec, t_vec])
            inputs.append(i_vec)
        outputs = self.bilstm.transduce(inputs)
        return outputs

    def evaluate(self, stack, buffer, features):
        # construct input vector
        b = features[buffer[0].id] if len(buffer) > 0 else self.empty
        s0 = features[stack[-1].id] if len(stack) > 0 else self.empty
        s1 = features[stack[-2].id] if len(stack) > 1 else self.empty
        s2 = features[stack[-3].id] if len(stack) > 2 else self.empty
        input = dy.concatenate([b, s0, s1, s2])
        # predict action
        act_hidden = dy.tanh(self.act_hidden.expr() * input + self.act_hidden_bias.expr())
        act_output = self.act_output.expr() * act_hidden + self.act_output_bias.expr()
        # predict label
        lbl_hidden = dy.tanh(self.lbl_hidden.expr() * input + self.lbl_hidden_bias.expr())
        lbl_output = self.lbl_output.expr() * lbl_hidden + self.lbl_output_bias.expr()
        # return scores
        return act_output, lbl_output

    def train(self, sentences):
        start_chunk = time.time()
        start_all = time.time()
        loss_chunk = 0
        loss_all = 0
        total_chunk = 0
        total_all = 0
        losses = []
        self.set_empty_vector()
        for i, sentence in enumerate(sentences):
            if i != 0 and i % 100 == 0:
                end = time.time()
                print(f'count: {i}\tloss: {loss_chunk/total_chunk:.4f}\ttime: {end-start_chunk:,.2f} secs')
                start_chunk = end
                loss_chunk = 0
                total_chunk = 0
            # assign embedding to each word
            features = self.extract_features(sentence, drop_word=True)
            # initialize sentence parse
            state = ArcHybrid(sentence)
            # parse sentence
            while not state.is_terminal():
                act_scores, lbl_scores = self.evaluate(state.stack, state.buffer, features)

                # get scores
                act_ss = act_scores.value()
                lbl_ss = lbl_scores.value()

                # collect all legal transitions
                legal_transitions = []
                if state.is_legal('shift'):
                    ix = state.t2i['shift']
                    t = ('shift', None, act_ss[ix] + lbl_ss[0], act_scores[ix] + lbl_scores[0])
                    legal_transitions.append(t)
                if state.is_legal('left_arc'):
                    ix = state.t2i['left_arc']
                    for j,r in enumerate(self.i2r):
                        k = 1 + 2 * j
                        t = ('left_arc', r, act_ss[ix] + lbl_ss[k], act_scores[ix] + lbl_scores[k])
                        legal_transitions.append(t)
                if state.is_legal('right_arc'):
                    ix = state.t2i['right_arc']
                    for j,r in enumerate(self.i2r):
                        k = 2 + 2 * j
                        t = ('right_arc', r, act_ss[ix] + lbl_ss[k], act_scores[ix] + lbl_scores[k])
                        legal_transitions.append(t)

                # collect all correct transitions
                correct_transitions = []
                for t in legal_transitions:
                    if state.is_correct(t[0]):
                        if t[0] == 'shift' or t[1] == state.stack[-1].relation:
                            correct_transitions.append(t)

                # select transition
                best_legal = max(legal_transitions, key=itemgetter(2))
                best_correct = max(correct_transitions, key=itemgetter(2))

                # accumulate losses
                loss = 1 - best_correct[2] + best_legal[2]
                if best_legal != best_correct and loss > 0:
                    losses.append(1 - best_correct[3] + best_legal[3])
                    loss_chunk += loss
                    loss_all += loss
                total_chunk += 1
                total_all += 1

                # perform transition
                # note that we compare against loss + 1, to perform aggressive exploration
                selected = best_legal if loss + 1 > 0 and random.random() < self.p_explore else best_correct
                state.perform_transition(selected[0], selected[1])

            # process losses in chunks
            if len(losses) > 50:
                loss = dy.esum(losses)
                loss.scalar_value()
                loss.backward()
                self.trainer.update()
                dy.renew_cg()
                self.set_empty_vector()
                losses = []

        # consider any remaining losses
        if len(losses) > 0:
            loss = dy.esum(losses)
            loss.scalar_value()
            loss.backward()
            self.trainer.update()
            dy.renew_cg()
            self.set_empty_vector()

        end = time.time()
        print('\nend of epoch')
        print(f'count: {i}\tloss: {loss_all/total_all:.4f}\ttime: {end-start_all:,.2f} secs')

    def parse_sentence(self, sentence):
        self.set_empty_vector()
        # assign embedding to each word
        features = self.extract_features(sentence)
        # initialize sentence parse
        state = ArcHybrid(sentence)
        # parse sentence
        while not state.is_terminal():
            act_scores, lbl_scores = self.evaluate(state.stack, state.buffer, features)
            # get numpy arrays
            act_scores = act_scores.npvalue()
            lbl_scores = lbl_scores.npvalue()
            # select transition

            left_lbl_score, left_lbl = max(zip(lbl_scores[1::2], self.i2r))
            right_lbl_score, right_lbl = max(zip(lbl_scores[2::2], self.i2r))

            transitions = []

            if state.is_legal('shift'):
                t = ('shift', None, act_scores[state.t2i['shift']] + lbl_scores[0])
                transitions.append(t)
            if state.is_legal('left_arc'):
                t = ('left_arc', left_lbl, act_scores[state.t2i['left_arc']] + left_lbl_score)
                transitions.append(t)
            if state.is_legal('right_arc'):
                t = ('right_arc', right_lbl, act_scores[state.t2i['right_arc']] + right_lbl_score)
                transitions.append(t)

            best_act, best_lbl, best_score = max(transitions, key=itemgetter(2))

            # perform transition
            state.perform_transition(best_act, best_lbl)
        dy.renew_cg()
        return sentence
