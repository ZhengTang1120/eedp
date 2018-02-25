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

from copy import deepcopy

Transition = namedtuple('Transition', 'op label trigger score dy_score')

class ArcHybridParser:

    def __init__(self, word_count, words, tags,
            entities, dep_relations, ev_relations,
            w_embed_size, t_embed_size, e_embed_size,
            lstm_hidden_size, lstm_num_layers,
            dep_op_hidden_size, dep_lbl_hidden_size,
            ev_op_hidden_size, ev_lbl_hidden_size,
            tg_lbl_hidden_size,
            alpha, p_explore):

        # counts used for word dropout
        self.word_count = word_count

        # mappings from ids to terms
        self.i2w = words
        self.i2t = tags
        self.i2e = ["Protein", "O", "*pad*"]
        self.dep_relations = dep_relations
        self.ev_relations = ev_relations
        if entities:
            self.i2tg = entities

        # mapings from terms to ids
        self.w2i = {w:i for i,w in enumerate(words)}
        self.t2i = {t:i for i,t in enumerate(tags)}
        self.e2i = {e:i for i,e in enumerate(self.i2e)}
        if entities:
            self.tg2i = {e:i for i,e in enumerate(self.i2tg)}

        self.w_embed_size = w_embed_size
        self.t_embed_size = t_embed_size
        self.e_embed_size = e_embed_size
        self.lstm_hidden_size = lstm_hidden_size * 2 # must be even
        self.lstm_num_layers = lstm_num_layers
        self.dep_op_hidden_size = dep_op_hidden_size
        self.dep_lbl_hidden_size = dep_lbl_hidden_size
        self.ev_op_hidden_size = ev_op_hidden_size
        self.ev_lbl_hidden_size = ev_lbl_hidden_size
        self.tg_lbl_hidden_size = tg_lbl_hidden_size
        self.alpha = alpha
        self.p_explore = p_explore
        self.entities = entities

        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)

        # words and tags, entities embeddings
        self.wlookup = self.model.add_lookup_parameters((len(self.i2w), self.w_embed_size))
        self.tlookup = self.model.add_lookup_parameters((len(self.i2t), self.t_embed_size))
        self.elookup = self.model.add_lookup_parameters((len(self.i2e), self.e_embed_size))

        # feature extractor
        self.bilstm = dy.BiRNNBuilder(
                self.lstm_num_layers,
                self.w_embed_size + self.t_embed_size + self.e_embed_size,
                self.lstm_hidden_size,
                self.model,
                dy.VanillaLSTMBuilder,
        )

        # transform word+pos vector into a vector similar to the lstm output
        # used to generate padding vectors
        self.word_to_lstm      = self.model.add_parameters((self.lstm_hidden_size, self.w_embed_size + self.t_embed_size + self.e_embed_size))
        self.word_to_lstm_bias = self.model.add_parameters((self.lstm_hidden_size))

        if self.dep_relations:
            # fully connected network with one hidden layer
            # to predict the transition to take next
            out_size = 3 # shift, left_arc, right_arc
            self.dep_op_hidden      = self.model.add_parameters((self.dep_op_hidden_size, self.lstm_hidden_size * 4))
            self.dep_op_hidden_bias = self.model.add_parameters((self.dep_op_hidden_size))
            self.dep_op_output      = self.model.add_parameters((out_size, self.dep_op_hidden_size))
            self.dep_op_output_bias = self.model.add_parameters((out_size))

            # # fully connected network with one hidden layer
            # # to predict the arc label
            out_size = 1 + len(self.dep_relations) * 2
            self.dep_lbl_hidden      = self.model.add_parameters((self.dep_lbl_hidden_size, self.lstm_hidden_size * 4))
            self.dep_lbl_hidden_bias = self.model.add_parameters((self.dep_lbl_hidden_size))
            self.dep_lbl_output      = self.model.add_parameters((out_size, self.dep_lbl_hidden_size))
            self.dep_lbl_output_bias = self.model.add_parameters((out_size))
        if self.ev_relations:
            # fully connected network with one hidden layer
            # to predict the transition to take next
            out_size = 4 # shift, left_arc, right_arc, drop
            self.ev_op_hidden      = self.model.add_parameters((self.ev_op_hidden_size, self.lstm_hidden_size * 4))
            self.ev_op_hidden_bias = self.model.add_parameters((self.ev_op_hidden_size))
            self.ev_op_output      = self.model.add_parameters((out_size, self.ev_op_hidden_size))
            self.ev_op_output_bias = self.model.add_parameters((out_size))

            # fully connected network with one hidden layer
            # to predict the arc label
            out_size = 1 + len(self.ev_relations) * 2
            self.ev_lbl_hidden      = self.model.add_parameters((self.ev_lbl_hidden_size, self.lstm_hidden_size * 4))
            self.ev_lbl_hidden_bias = self.model.add_parameters((self.ev_lbl_hidden_size))
            self.ev_lbl_output      = self.model.add_parameters((out_size, self.ev_lbl_hidden_size))
            self.ev_lbl_output_bias = self.model.add_parameters((out_size))
        if self.entities:
            # fully connected network with one hidden layer
            # to predict the trigger label
            out_size = 1 + len(self.i2tg)
            self.tg_lbl_hidden      = self.model.add_parameters((self.tg_lbl_hidden_size, self.lstm_hidden_size * 5))
            self.tg_lbl_hidden_bias = self.model.add_parameters((self.tg_lbl_hidden_size))
            self.tg_lbl_output      = self.model.add_parameters((out_size, self.tg_lbl_hidden_size))
            self.tg_lbl_output_bias = self.model.add_parameters((out_size))

    def save(self, name):
        params = (
            self.word_count, self.i2w, self.i2t,
            self.entities, self.dep_relations, self.ev_relations,
            self.w_embed_size, self.t_embed_size, self.e_embed_size,
            self.lstm_hidden_size // 2, self.lstm_num_layers,
            self.dep_op_hidden_size, self.dep_lbl_hidden_size,
            self.ev_op_hidden_size, self.ev_lbl_hidden_size,
            self.tg_lbl_hidden_size,
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
        e_pad = self.elookup[self.e2i['*pad*']] #??
        v_pad = dy.concatenate([w_pad, t_pad, e_pad])
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
            e_id = self.e2i[entry.feats] if entry.feats == "Protein" else self.e2i["O"]
            # get word and tag embbedding in the corresponding entry
            w_vec = self.wlookup[w_id]
            t_vec = self.tlookup[t_id]
            e_vec = self.elookup[e_id]
            i_vec = dy.concatenate([w_vec, t_vec, e_vec])
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
        return dy.softmax(op_output), dy.softmax(lbl_output)

    def evaluate_events(self, stack, buffer, features):
        # construct input vector
        b = features[buffer[0].id] if len(buffer) > 0 else self.empty
        s0 = features[stack[-1].id] if len(stack) > 0 else self.empty
        s1 = features[stack[-2].id] if len(stack) > 1 else self.empty
        s2 = features[stack[-3].id] if len(stack) > 2 else self.empty
        b1 = features[buffer[0].id + 1] if buffer[0].id + 1 < len(features) else self.empty
        b2 = features[buffer[0].id + 2] if buffer[0].id + 2 < len(features) else self.empty
        bp1 = features[buffer[0].id - 1] if buffer[0].id - 1 >= 0 else self.empty
        bp2 = features[buffer[0].id -1 ] if buffer[0].id - 2 >= 0 else self.empty
        t = dy.concatenate([bp2, bp1, b, b1, b2])
        input = dy.concatenate([b, s0, s1, s2])
        # predict action
        op_hidden = dy.tanh(self.ev_op_hidden.expr() * input + self.ev_op_hidden_bias.expr())
        op_output = self.ev_op_output.expr() * op_hidden + self.ev_op_output_bias.expr()
        # predict label
        lbl_hidden = dy.tanh(self.ev_lbl_hidden.expr() * input + self.ev_lbl_hidden_bias.expr())
        lbl_output = self.ev_lbl_output.expr() * lbl_hidden + self.ev_lbl_output_bias.expr()
        # predict trigger label
        tg_hidden = dy.tanh(self.tg_lbl_hidden.expr() * t + self.tg_lbl_hidden_bias.expr())
        tg_output = self.tg_lbl_output.expr() * tg_hidden + self.tg_lbl_output_bias.expr()
        # return scores
        return dy.softmax(op_output), dy.softmax(lbl_output), dy.softmax(tg_output)

    def train_dependencies(self, sentences):
        self._train(sentences, ArcHybrid, self.evaluate_dependencies, self.dep_relations)

    def train_events(self, sentences):
        self._train(sentences, ArcHybridWithDrop, self.evaluate_events, self.ev_relations, self.i2tg)

    def _train(self, sentences, transition_system, evaluate, relations, triggers = None):
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
            if len(sentence) > 2:
                # assign embedding to each word
                features = self.extract_features(sentence, drop_word=True)
                # initialize sentence parse
                state = transition_system(sentence)
                # parse sentence
                while not state.is_terminal():
                    outputs = evaluate(state.stack, state.buffer, features)

                    if triggers:
                        dy_op_scores, dy_lbl_scores, dy_tg_scores = outputs
                        np_tg_scores = dy_tg_scores.npvalue()
                    else:
                        dy_op_scores, dy_lbl_scores = outputs

                    # get scores in numpy arrays
                    np_op_scores = dy_op_scores.npvalue()
                    np_lbl_scores = dy_lbl_scores.npvalue()

                    # collect all legal transitions
                    legal_transitions = []
                    if state.is_legal('shift'):
                        ix = state.t2i['shift']
                        if triggers:
                            for j, tg in enumerate(triggers, start=1):
                                if (hasattr(state.buffer[0], 'is_parent') and state.buffer[0].is_parent and j == 1):
                                    continue
                                t = Transition('shift', None, tg, np_op_scores[ix] + np_lbl_scores[0] + np_tg_scores[j], dy_op_scores[ix] + dy_lbl_scores[0] + dy_tg_scores[j])
                                legal_transitions.append(t)
                        else:
                            t = Transition('shift', None, None, np_op_scores[ix] + np_lbl_scores[0], dy_op_scores[ix] + dy_lbl_scores[0])
                            legal_transitions.append(t)
                    if state.is_legal('left_arc'):
                        ix = state.t2i['left_arc']
                        for j,r in enumerate(relations):
                            k = 1 + 2 * j
                            if triggers:
                                t = Transition('left_arc', r, None, np_op_scores[ix] + np_lbl_scores[k] + np_tg_scores[0], dy_op_scores[ix] + dy_lbl_scores[k] + dy_tg_scores[0])
                                legal_transitions.append(t)
                            else:
                                t = Transition('left_arc', r, None, np_op_scores[ix] + np_lbl_scores[k], dy_op_scores[ix] + dy_lbl_scores[k])
                                legal_transitions.append(t)
                    if state.is_legal('right_arc'):
                        ix = state.t2i['right_arc']
                        for j,r in enumerate(relations):
                            k = 2 + 2 * j
                            if triggers:
                                t = Transition('right_arc', r, None, np_op_scores[ix] + np_lbl_scores[k] + np_tg_scores[0], dy_op_scores[ix] + dy_lbl_scores[k] + dy_tg_scores[0])
                                legal_transitions.append(t)
                            else:
                                t = Transition('right_arc', r, None, np_op_scores[ix] + np_lbl_scores[k], dy_op_scores[ix] + dy_lbl_scores[k])
                                legal_transitions.append(t)
                    if state.is_legal('drop'):
                        ix = state.t2i['drop']
                        if triggers:
                            t = Transition('drop', None, "O", np_op_scores[ix] + np_lbl_scores[0] + np_tg_scores[1], dy_op_scores[ix] + dy_lbl_scores[0] + dy_tg_scores[1])
                            legal_transitions.append(t)
                        else:
                            t = Transition('drop', None, None, np_op_scores[ix] + np_lbl_scores[0], dy_op_scores[ix] + dy_lbl_scores[0])
                            legal_transitions.append(t)
                    # print('---')
                    # print('legal',legal_transitions)

                    # collect all correct transitions
                    correct_transitions = []
                    for t in legal_transitions:
                        if state.is_correct(t):
                            # if t.op not in ['shift', 'drop']:
                            #     print(t.label, state.stack[-1].relation)
                            if t.op in ['shift', 'drop'] or t.label == state.stack[-1].relation:
                                correct_transitions.append(t)

                    # print('correct',correct_transitions)
                    # print('sentence',sentence)
                    # print(state.stack)
                    # print(state.buffer)
                    # select transition
                    best_legal = max(legal_transitions, key=attrgetter('score'))
                    best_correct = max(correct_transitions, key=attrgetter('score'))

                    # accumulate losses
                    loss = 1 - best_correct.score + best_legal.score
                    dy_loss = 1 - best_correct.dy_score + best_legal.dy_score

                    if best_legal != best_correct and loss > 0:
                        losses.append(dy_loss)
                        loss_chunk += loss
                        loss_all += loss
                    total_chunk += 1
                    total_all += 1

                    # perform transition
                    # note that we compare against loss + 1, to perform aggressive exploration
                    selected = best_legal if loss + 1 > 0 and random.random() < self.p_explore else best_correct
                    state.perform_transition(selected.op, selected.label, selected.trigger)

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
        state = ArcHybridWithDrop(sentence)
        # parse sentence
        while not state.is_terminal():
            op_scores, lbl_scores, tg_scores = self.evaluate_events(state.stack, state.buffer, features)
            # get numpy arrays
            op_scores = op_scores.npvalue()
            lbl_scores = lbl_scores.npvalue()
            tg_scores = tg_scores.npvalue()
            # select transition
            left_lbl_score, left_lbl = max(zip(lbl_scores[1::2], self.ev_relations))
            right_lbl_score, right_lbl = max(zip(lbl_scores[2::2], self.ev_relations))
            trigger_score, trigger = max(zip(tg_scores[1:], self.i2tg))
            print (list(zip(tg_scores[1:], self.i2tg)))
            # collect all legal transitions
            transitions = []
            if state.is_legal('shift'):
                t = ('shift', None, trigger, op_scores[state.t2i['shift']] + lbl_scores[0] + trigger_score)
                transitions.append(t)
            if state.is_legal('left_arc'):
                t = ('left_arc', left_lbl, None, op_scores[state.t2i['left_arc']] + left_lbl_score + tg_scores[0])
                transitions.append(t)
            if state.is_legal('right_arc'):
                t = ('right_arc', right_lbl, None, op_scores[state.t2i['right_arc']] + right_lbl_score + tg_scores[0])
                transitions.append(t)
            if state.is_legal('drop'):
                t = ('drop', None, "O", op_scores[state.t2i['drop']] + lbl_scores[0] + tg_scores[1])
                transitions.append(t)

            # select best legal transition
            best_act, best_lbl, best_tg, best_socre = max(transitions, key=itemgetter(3))

            # perform transition
            state.perform_transition(best_act, best_lbl, best_tg)
        dy.renew_cg()
        return sentence
