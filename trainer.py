from collections import namedtuple
from operator import attrgetter, itemgetter

import numpy as np
from transition_systems import ArcHybrid
from new_transition_system import CustomTransitionSystem

from copy import deepcopy

import time
import torch

Transition = namedtuple('Transition', 'op label trigger score tr_score')
new_Transition = namedtuple('Transition', 'op label trigger score tr_score')


def train_dependencies(sentences, parser):
    # parser._train(sentences, ArcHybrid, parser.evaluate_dependencies, parser.dep_relations, parser.trainer_dp)
    start_chunk = time.time()
    start_all = time.time()
    loss_chunk = 0
    loss_all = 0
    total_chunk = 0
    total_all = 0
    losses = []
    parser.set_empty_vector()

    for i, sentence in enumerate(sentences):
        if i != 0 and i % 100 == 0:
            end = time.time()
            print(f'count: {i}\tloss: {loss_chunk/total_chunk:.4f}\ttime: {end-start_chunk:,.2f} secs')
            start_chunk = end
            loss_chunk = 0
            total_chunk = 0
        if len(sentence) > 2:
            parser.bihidden = parser.init_hidden(parser.lstm_num_layers * 2, parser.lstm_hidden_size // 2)
            for e in sentence:
                e.children = []
            # assign embedding to each word
            parser.extract_features(sentence, drop_word=True)
            # initialize sentence parse
            state = ArcHybrid(sentence)
            # parse sentence
            while not state.is_terminal():
                outputs = parser(state.stack, state.buffer, False)

                tr_op_scores, tr_lbl_scores = outputs

                # get scores in numpy arrays
                np_op_scores = tr_op_scores.data.numpy()
                np_lbl_scores = tr_lbl_scores.data.numpy()

                # collect all legal transitions
                legal_transitions = []
                if state.is_legal('shift'):
                    ix = state.t2i['shift']
                    t = Transition('shift', None, None, np_op_scores[ix] + np_lbl_scores[0], tr_op_scores[ix] + tr_lbl_scores[0])
                    legal_transitions.append(t)
                if state.is_legal('left_arc'):
                    ix = state.t2i['left_arc']
                    for j,r in enumerate(parser.dep_relations):
                        k = 1 + 2 * j
                        t = Transition('left_arc', r, None, np_op_scores[ix] + np_lbl_scores[k], tr_op_scores[ix] + tr_lbl_scores[k])
                        legal_transitions.append(t)
                if state.is_legal('right_arc'):
                    ix = state.t2i['right_arc']
                    for j,r in enumerate(parser.dep_relations):
                        k = 2 + 2 * j
                        t = Transition('right_arc', r, None, np_op_scores[ix] + np_lbl_scores[k], tr_op_scores[ix] + tr_lbl_scores[k])
                        legal_transitions.append(t)
                if state.is_legal('drop'):
                    ix = state.t2i['drop']
                    t = Transition('drop', None, None, np_op_scores[ix] + np_lbl_scores[0], tr_op_scores[ix] + tr_lbl_scores[0])
                    legal_transitions.append(t)
                # collect all correct transitions
                correct_transitions = []
                for t in legal_transitions:
                    if state.is_correct(t):
                        if t.op in ['shift', 'drop'] or t.label in state.stack[-1].relation:
                            correct_transitions.append(t)

                
                # select transition
                best_legal = max(legal_transitions, key=attrgetter('score'))
                best_correct = max(correct_transitions, key=attrgetter('score'))

                # accumulate losses
                loss = 1 - best_correct.score + best_legal.score
                tr_loss = 1 - best_correct.tr_score + best_legal.tr_score


                # select transition
                best_correct = max(correct_transitions, key=attrgetter('score'))

                i_correct = legal_transitions.index(best_correct)
                legal_scores = torch.stack([t.tr_score for t in legal_transitions])

                loss = parser.criterion(legal_scores, torch.tensor(i_correct))
                loss.backward(retain_graph=True)

                selected = best_correct
                state.perform_transition(selected.op, selected.label, selected.trigger)

            parser.optimizer_dp.step()
            parser.set_empty_vector()

    end = time.time()
    print('\nend of epoch')
    print(f'count: {i}\tloss: {loss_all/total_all:.4f}\ttime: {end-start_all:,.2f} secs')



def train_events(sentences, parser):
    # parser._train(sentences, CustomTransitionSystem, parser.evaluate_events, parser.ev_relations, parser.trainer_ee, parser.i2tg)
    start_chunk = time.time()
    start_all = time.time()
    parser.set_empty_vector()

    
    for i, sentence in enumerate(sentences):
        if i != 0 and i % 100 == 0:
            end = time.time()
            print(f'count: {i}\tloss: {loss_chunk/total_chunk:.4f}\ttime: {end-start_chunk:,.2f} secs')
            start_chunk = end
            loss_chunk = 0
            total_chunk = 0
        if len(sentence) > 2:
            parser.bihidden = parser.init_hidden(parser.lstm_num_layers * 2, parser.lstm_hidden_size // 2)
            for e in sentence:
                e.children = []
            # assign embedding to each word
            parser.extract_features(sentence, drop_word=True)
            # initialize sentence parse
            state = CustomTransitionSystem(sentence)
            # parse sentence
            while not state.is_terminal():
                outputs = parser(state.stack, state.buffer, True)

                tr_op_scores, tr_lbl_scores, tr_tg_scores = outputs
                np_tg_scores = tr_tg_scores.data.numpy()

                # get scores in numpy arrays
                np_op_scores = tr_op_scores.data.numpy()
                np_lbl_scores = tr_lbl_scores.data.numpy()

                # collect all legal transitions
                legal_transitions = []
                
                for lt in state.all_legal():
                    ix = state.t2i[lt]
                    if lt == "shift":
                        for j, tg in enumerate(parser.i2tg[1:], start=2):
                            if (hasattr(state.buffer[0], 'is_parent') and state.buffer[0].is_parent and j == 1):
                                continue
                            t = new_Transition(lt, None, tg, np_op_scores[ix] + np_lbl_scores[0] + np_tg_scores[j], tr_op_scores[ix] + tr_lbl_scores[0] + tr_tg_scores[j])
                            legal_transitions.append(t)
                    if lt == "drop":
                        t = new_Transition(lt, None, "O", np_op_scores[ix] + np_lbl_scores[0] + np_tg_scores[1], tr_op_scores[ix] + tr_lbl_scores[0] + tr_tg_scores[1])
                        legal_transitions.append(t)
                        t = new_Transition(lt, None, "Protein", np_op_scores[ix] + np_lbl_scores[0] + np_tg_scores[4], tr_op_scores[ix] + tr_lbl_scores[0] + tr_tg_scores[4])
                        legal_transitions.append(t)
                    if lt in ['left_reduce', 'left_attach']:
                        for j, r in enumerate(parser.ev_relations):
                            k = 1 + 2* j
                            t = new_Transition(lt, r, None, np_op_scores[ix] + np_lbl_scores[k] + np_tg_scores[0], tr_op_scores[ix] + tr_lbl_scores[k] + tr_tg_scores[0])
                            legal_transitions.append(t)
                    if lt in ['right_reduce', 'right_attach']:
                        for j, r in enumerate(parser.ev_relations):
                            k = 2 + 2 * j
                            t = new_Transition(lt, r, None, np_op_scores[ix] + np_lbl_scores[k] + np_tg_scores[0], tr_op_scores[ix] + tr_lbl_scores[k] + tr_tg_scores[0])
                            legal_transitions.append(t)
                    if lt == "swap":
                        t = new_Transition(lt, None, None, np_op_scores[ix] + np_lbl_scores[0] + np_tg_scores[0], tr_op_scores[ix] + tr_lbl_scores[0] + tr_tg_scores[0])
                        legal_transitions.append(t)
                # collect all correct transitions
                correct_transitions = []
                for t in legal_transitions:
                    if state.is_correct(t[0]):
                        relation = state.get_arc_label_for_transition(t[0])
                        label = state.get_token_label_for_transition(t[0])
                        if t[1] == relation and t[2] == label:
                            correct_transitions.append(t)

                
                # select transition
                best_correct = max(correct_transitions, key=attrgetter('score'))

                i_correct = legal_transitions.index(best_correct)
                a = [1 if i == i_correct else 0for i in range(len(legal_transitions))]
                legal_scores = torch.stack([t.tr_score for t in legal_transitions])

                loss = parser.criterion(legal_scores, torch.tensor(a))
                print (loss.data.numpy())
                loss.backward(retain_graph=True)

                selected = best_correct
                state.perform_transition(selected.op, selected.label, selected.trigger)

            parser.optimizer_ee.step()
            parser.set_empty_vector()


    end = time.time()
    print('\nend of epoch')
    print(f'count: {i}\tloss: {loss_all/total_all:.4f}\ttime: {end-start_all:,.2f} secs')