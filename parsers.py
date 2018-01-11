import time
import random
import pickle
from collections import namedtuple
from operator import attrgetter, itemgetter

import dynet_config
dynet_config.set(random_seed=1)

import dynet as dy
import numpy as np
from transition_systems import ArcHybrid, ArcHybridWithDrop

Transition = namedtuple('Transition', 'op label score dy_score')

class ArcHybridParser:

    def __init__(self, word_count, words, tags,
            dep_relations, #ev_relations,
            w_embed_size, t_embed_size,
            lstm_hidden_size, lstm_num_layers,
            dep_op_hidden_size, dep_lbl_hidden_size,
            #ev_op_hidden_size, ev_lbl_hidden_size,
            alpha, p_explore):

        # counts used for word dropout
        self.word_count = word_count

        # mappings from ids to terms
        self.i2w = words
        self.i2t = tags
        self.dep_relations = dep_relations
        #self.ev_relations = ev_relations

        # mapings from terms to ids
        self.w2i = {w:i for i,w in enumerate(words)}
        self.t2i = {t:i for i,t in enumerate(tags)}

        self.w_embed_size = w_embed_size
        self.t_embed_size = t_embed_size
        self.lstm_hidden_size = lstm_hidden_size * 2 # must be even
        self.lstm_num_layers = lstm_num_layers
        self.dep_op_hidden_size = dep_op_hidden_size
        self.dep_lbl_hidden_size = dep_lbl_hidden_size
        #self.ev_op_hidden_size = ev_op_hidden_size
        #self.ev_lbl_hidden_size = ev_lbl_hidden_size
        self.alpha = alpha
        self.p_explore = p_explore

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
        self.dep_op_hidden      = self.model.add_parameters((self.dep_op_hidden_size, self.lstm_hidden_size * 4))
        self.dep_op_hidden_bias = self.model.add_parameters((self.dep_op_hidden_size))
        self.dep_op_output      = self.model.add_parameters((out_size, self.dep_op_hidden_size))
        self.dep_op_output_bias = self.model.add_parameters((out_size))

        # fully connected network with one hidden layer
        # to predict the arc label
        out_size = 1 + len(self.dep_relations) * 2
        self.dep_lbl_hidden      = self.model.add_parameters((self.dep_lbl_hidden_size, self.lstm_hidden_size * 4))
        self.dep_lbl_hidden_bias = self.model.add_parameters((self.dep_lbl_hidden_size))
        self.dep_lbl_output      = self.model.add_parameters((out_size, self.dep_lbl_hidden_size))
        self.dep_lbl_output_bias = self.model.add_parameters((out_size))

        # # fully connected network with one hidden layer
        # # to predict the transition to take next
        # out_size = 4 # shift, left_arc, right_arc, drop
        # self.ev_op_hidden      = self.model.add_parameters((self.ev_op_hidden_size, self.lstm_hidden_size * 4))
        # self.ev_op_hidden_bias = self.model.add_parameters((self.ev_op_hidden_size))
        # self.ev_op_output      = self.model.add_parameters((out_size, self.ev_op_hidden_size))
        # self.ev_op_output_bias = self.model.add_parameters((out_size))

        # # fully connected network with one hidden layer
        # # to predict the arc label
        # out_size = 1 + len(self.ev_relations) * 2
        # self.ev_lbl_hidden      = self.model.add_parameters((self.ev_lbl_hidden_size, self.lstm_hidden_size * 4))
        # self.ev_lbl_hidden_bias = self.model.add_parameters((self.ev_lbl_hidden_size))
        # self.ev_lbl_output      = self.model.add_parameters((out_size, self.ev_lbl_hidden_size))
        # self.ev_lbl_output_bias = self.model.add_parameters((out_size))

    def save(self, name):
        params = (
            self.word_count, self.i2w, self.i2t,
            self.dep_relations, self.ev_relations,
            self.w_embed_size, self.t_embed_size,
            self.lstm_hidden_size // 2, self.lstm_num_layers,
            self.dep_op_hidden_size, self.dep_lbl_hidden_size,
            self.ev_op_hidden_size, self.ev_lbl_hidden_size,
            self.alpha, self.p_explore
        )
        # save model
        self.model.save(f'{name}.model')
        # save pickle
        with open(f'{name}.pickle', 'wb') as f:
            pickle.dump(params, f)

    @staticmethod
    def load(name):
        with open(f'{name}.pickle', 'rb') as f:
            params = pickle.load(f)
            parser = ArcHybridParser(*params)
            parser.model.populate(f'{name}.model')
            return parser

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

    def evaluate_dependencies(self, stack, buffer, features):
        # construct input vector
        b = features[buffer[0].id] if len(buffer) > 0 else self.empty
        s0 = features[stack[-1].id] if len(stack) > 0 else self.empty
        s1 = features[stack[-2].id] if len(stack) > 1 else self.empty
        s2 = features[stack[-3].id] if len(stack) > 2 else self.empty
        input = dy.concatenate([b, s0, s1, s2])
        # predict action
        op_hidden = dy.tanh(self.dep_op_hidden.expr() * input + self.dep_op_hidden_bias.expr())
        op_output = self.dep_op_output.expr() * op_hidden + self.dep_op_output_bias.expr()
        # predict label
        lbl_hidden = dy.tanh(self.dep_lbl_hidden.expr() * input + self.dep_lbl_hidden_bias.expr())
        lbl_output = self.dep_lbl_output.expr() * lbl_hidden + self.dep_lbl_output_bias.expr()
        # return scores
        return op_output, lbl_output

    def evaluate_events(self, stack, buffer, features):
        # construct input vector
        b = features[buffer[0].id] if len(buffer) > 0 else self.empty
        s0 = features[stack[-1].id] if len(stack) > 0 else self.empty
        s1 = features[stack[-2].id] if len(stack) > 1 else self.empty
        s2 = features[stack[-3].id] if len(stack) > 2 else self.empty
        input = dy.concatenate([b, s0, s1, s2])
        # predict action
        op_hidden = dy.tanh(self.ev_op_hidden.expr() * input + self.ev_op_hidden_bias.expr())
        op_output = self.ev_op_output.expr() * op_hidden + self.ev_op_output_bias.expr()
        # predict label
        lbl_hidden = dy.tanh(self.ev_lbl_hidden.expr() * input + self.ev_lbl_hidden_bias.expr())
        lbl_output = self.ev_lbl_output.expr() * lbl_hidden + self.ev_lbl_output_bias.expr()
        # return scores
        return op_output, lbl_output

    def train_dependencies(self, sentences):
        self._train(sentences, ArcHybrid, self.evaluate_dependencies, self.dep_relations)

    def train_events(self, sentences):
        self._train(sentences, ArcHybridWithDrop, self.evaluate_events, self.ev_relations)

    def _train(self, sentences, transition_system, evaluate, relations):
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
            state = transition_system(sentence)
            # parse sentence
            while not state.is_terminal():
                dy_op_scores, dy_lbl_scores = evaluate(state.stack, state.buffer, features)

                # get scores in numpy arrays
                np_op_scores = dy_op_scores.npvalue()
                np_lbl_scores = dy_lbl_scores.npvalue()

                # collect all legal transitions
                legal_transitions = []
                if state.is_legal('shift'):
                    ix = state.t2i['shift']
                    t = Transition('shift', None, np_op_scores[ix] + np_lbl_scores[0], dy_op_scores[ix] + dy_lbl_scores[0])
                    legal_transitions.append(t)
                if state.is_legal('left_arc'):
                    ix = state.t2i['left_arc']
                    for j,r in enumerate(relations):
                        k = 1 + 2 * j
                        t = Transition('left_arc', r, np_op_scores[ix] + np_lbl_scores[k], dy_op_scores[ix] + dy_lbl_scores[k])
                        legal_transitions.append(t)
                if state.is_legal('right_arc'):
                    ix = state.t2i['right_arc']
                    for j,r in enumerate(relations):
                        k = 2 + 2 * j
                        t = Transition('right_arc', r, np_op_scores[ix] + np_lbl_scores[k], dy_op_scores[ix] + dy_lbl_scores[k])
                        legal_transitions.append(t)
                if state.is_legal('drop'):
                    ix = state.t2i['drop']
                    t = Transition('drop', None, np_op_scores[ix] + np_lbl_scores[0], dy_op_scores[ix] + dy_lbl_scores[0])
                    legal_transitions.append(t)
                #print('---')
                #print('legal',legal_transitions)

                # collect all correct transitions
                correct_transitions = []
                for t in legal_transitions:
                    if state.is_correct(t[0]):
                        if t.op not in ['shift', 'drop']:
                            pass
                            #print(t.label, state.stack[-1].relation)
                        if t.op in ['shift', 'drop'] or t.label == state.stack[-1].relation:
                            correct_transitions.append(t)

                #print('correct',correct_transitions)
                #print('sentence',sentence)
                #print(state.stack)
                #print(state.buffer)
                # select transition
                best_legal = max(legal_transitions, key=attrgetter('score'))
                best_correct = max(correct_transitions, key=attrgetter('score'))

                # accumulate losses
                loss = 1 - best_correct.score + best_legal.score
                if best_legal != best_correct and loss > 0:
                    losses.append(1 - best_correct.dy_score + best_legal.dy_score)
                    loss_chunk += loss
                    loss_all += loss
                total_chunk += 1
                total_all += 1

                # perform transition
                # note that we compare against loss + 1, to perform aggressive exploration
                selected = best_legal if loss + 1 > 0 and random.random() < self.p_explore else best_correct
                state.perform_transition(selected.op, selected.label)

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
            op_scores, lbl_scores = self.evaluate(state.stack, state.buffer, features)
            # get numpy arrays
            op_scores = op_scores.npvalue()
            lbl_scores = lbl_scores.npvalue()
            # select transition
            left_lbl_score, left_lbl = max(zip(lbl_scores[1::2], self.dep_relations))
            right_lbl_score, right_lbl = max(zip(lbl_scores[2::2], self.dep_relations))

            # collect all legal transitions
            transitions = []
            if state.is_legal('shift'):
                t = ('shift', None, op_scores[state.t2i['shift']] + lbl_scores[0])
                transitions.append(t)
            if state.is_legal('left_arc'):
                t = ('left_arc', left_lbl, op_scores[state.t2i['left_arc']] + left_lbl_score)
                transitions.append(t)
            if state.is_legal('right_arc'):
                t = ('right_arc', right_lbl, op_scores[state.t2i['right_arc']] + right_lbl_score)
                transitions.append(t)
            if state.is_legal('drop'):
                t = ('drop', None, op_scores[state.t2i['drop']] + lbl_scores[0])
                transitions.append(t)

            # select best legal transition
            best_act, best_lbl, best_score = max(transitions, key=itemgetter(2))

            # perform transition
            state.perform_transition(best_act, best_lbl)
        dy.renew_cg()
        return sentence
