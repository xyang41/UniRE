from collections import defaultdict
import logging
import sys
import difflib

from requests import get

logger = logging.getLogger(__name__)

metric2labels = {}

def get_overlap(s1, s2):
    """
    This function is used to find the longest overlapping substring between s1 and s2.
    e.g. 

        my stackoverflow mysteries
        .................mystery..
        
        the result is "myster"

    """
    SeqMatcher = difflib.SequenceMatcher(None, s1, s2)
    _, _, overlap_len = SeqMatcher.find_longest_match(0, len(s1), 0, len(s2))
    return overlap_len

def check_output(file_path):
    """
    This function is used to check the output file.
    e.g.
    
    before:
        my dog is cute.
        
        my cat is cute
         too.
    after:
        my dog is cute.
        
        my cat is cute too.
    """
    with open(file_path, 'r') as fin:
        lines = fin.readlines()
    fout = open(file_path, 'w')
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip('\r\n')
        print(idx)
        if line == "":
            fout.write('\n')
        else:
            words = line
            while idx+1 < len(lines) and len(lines[idx+1]) != 0 and lines[idx+1][0] == ' ':
                words = words + '&' + lines[idx+1].strip('\r\n')
                # replace '\n' as '&', but it may cause some bugs.
                # we will change our output form later.
                idx+=1
            print(words)
            fout.write(words + '\n')
        idx += 1
        
class EvalCounts():
    """This class is evaluating counters
    """
    def __init__(self):
        self.pred_correct_cnt = 0
        self.correct_cnt = 0
        self.pred_cnt = 0

        self.pred_correct_types_cnt = defaultdict(int)
        self.correct_types_cnt = defaultdict(int)
        self.pred_types_cnt = defaultdict(int)

def cal_correct_pred(correct_set, pred_set, overlap_match_rate, cal_type):
    
    """
    This function is used to calculate the number of corrects entities.
    
    Arguments:
        correct_set {collection.defaultdict} -- a set of correct entities 
        (entity text, entity type) or relations (entity1 text, entity1 type,
        entity2 text, entity2 type).
        
        pred_set {collection.defaultdict} -- a set of predicated entities  
         (entity text, entity type) or relations (entity1 text, entity1 type,
        entity2 text, entity2 type).
        
        overlap_match_rate {float} -- the matched rate of the text.
        if get_overlap(predication_text, correct_text) > overlap_match_rate,
        then we reagrd this predication entity is right
        
        cal_type {str} -- calculate the number of correct predication(entities or 
        relations)
    
    Returns:
        correct_num {int} -- the number of correct entities or relations the model predicate
    """
    matched_ids = []
    correct_num = 0
    for pred in pred_set:
        for idx, correct in enumerate(correct_set):
            if cal_type == 'ent':
                pred_text = pred[0].replace(' ', '')
                pred_type = pred[1].replace(' ', '')
                correct_text = correct[0].replace(' ', '')
                correct_type = correct[1].replace(' ', '')
                if pred_type != correct_type:
                    continue
                if idx in matched_ids:
                    continue
                if get_overlap(pred_text, 
                            correct_text) < overlap_match_rate *len(correct_text):
                    continue
            elif cal_type == 'rel':
                pred_ent1_text = pred[0].replace(' ', '')
                pred_ent1_type = pred[1].replace(' ', '')
                pred_ent2_text = pred[2].replace(' ', '')
                pred_ent2_type = pred[3].replace(' ', '')
                correct_ent1_text = correct[0].replace(' ', '')
                correct_ent1_type = correct[1].replace(' ', '')
                correct_ent2_text = correct[2].replace(' ', '')
                correct_ent2_type = correct[3].replace(' ', '')
                if pred_ent1_type != correct_ent1_type or pred_ent2_type != correct_ent2_type:
                    continue
                if idx in matched_ids:
                    continue
                if get_overlap(pred_ent1_text, 
                            correct_ent1_text) < overlap_match_rate*len(correct_ent1_text):
                    continue
                if get_overlap(pred_ent2_text, 
                            correct_ent2_text) < overlap_match_rate*len(correct_ent2_text):
                    continue
            correct_num += 1
            matched_ids.append(idx)
            break
    return correct_num


def eval_file(file_path, eval_metrics, cfg):
    """eval_file evaluates results file

    Args:
        file_path (str): file path
        eval_metrics (list): eval metrics

    Returns:
        tuple: results
    """
    check_output(file_path)
    
    with open(file_path, 'r') as fin:
        sents = []
        metric2labels = {
            'token': ['Sequence-Label-True', 'Sequence-Label-Pred'],
            'joint-label': ['Joint-Label-True', 'Joint-Label-Pred'],
            'separate-position': ['Separate-Position-True', 'Separate-Position-Pred'],
            'span': ['Ent-Span-Pred'],
            'ent': ['Ent-True', 'Ent-Pred'],
            'rel': ['Rel-True', 'Rel-Pred'],
            'exact-rel': ['Rel-True', 'Rel-Pred'],
            'overlap-rel': ['Rel-True', 'Rel-Pred'],
            'overlap-ent': ['Ent-True', 'Ent-Pred']
        }
        labels = set()
        for metric in eval_metrics:
            labels.update(metric2labels[metric])
        label2idx = {label: idx for idx, label in enumerate(labels)}
        sent = [[] for _ in range(len(labels))]
        for line in fin:
            line = line.strip('\r\n')
            if line == "":
                sents.append(sent)
                sent = [[] for _ in range(len(labels))]
            else:
                words = line
                words = words.split('\t')
                if words[0] in ['Sequence-Label-True', 'Sequence-Label-Pred', 'Joint-Label-True', 'Joint-Label-Pred']:
                    sent[label2idx[words[0]]].extend(words[1].split(' '))
                elif words[0] in ['Separate-Position-True', 'Separate-Position-Pred']:
                    sent[label2idx[words[0]]].append(words[1].split(' '))
                elif words[0] in ['Ent-Span-Pred']:
                    sent[label2idx[words[0]]].append(eval(words[1]))
                elif words[0] in ['Ent-True', 'Ent-Pred']:
                    sent[label2idx[words[0]]].append([words[1], eval(words[2]), words[3]])
                elif words[0] in ['Rel-True', 'Rel-Pred']:
                    sent[label2idx[words[0]]].append([  words[1], 
                                                        eval(words[2]), 
                                                        eval(words[3]), 
                                                        words[4],
                                                        words[5]])
        sents.append(sent)

    counts = {metric: EvalCounts() for metric in eval_metrics}

    for sent in sents:
        evaluate(sent, counts, label2idx, cfg)

    results = []

    logger.info("-" * 22 + "START" + "-" * 23)

    for metric, count in counts.items():
        left_offset = (50 - len(metric)) // 2
        logger.info("-" * left_offset + metric + "-" * (50 - left_offset - len(metric)))
        score = report(count)
        results += [score]

    logger.info("-" * 23 + "END" + "-" * 24)

    return results


def evaluate(sent, counts, label2idx, cfg):
    """evaluate calculates counters
    
    Arguments:
        sent {list} -- line

    Args:
        sent (list): line
        counts (dict): counts
        label2idx (dict): label -> idx dict
    """

    # evaluate token
    if 'token' in counts:
        for token1, token2 in zip(sent[label2idx['Sequence-Label-True']], sent[label2idx['Sequence-Label-Pred']]):
            if token1 != 'O':
                counts['token'].correct_cnt += 1
                counts['token'].correct_types_cnt[token1] += 1
                counts['token'].pred_correct_types_cnt[token1] += 0
            if token2 != 'O':
                counts['token'].pred_cnt += 1
                counts['token'].pred_types_cnt[token2] += 1
                counts['token'].pred_correct_types_cnt[token2] += 0
            if token1 == token2 and token1 != 'O':
                counts['token'].pred_correct_cnt += 1
                counts['token'].pred_correct_types_cnt[token1] += 1

    # evaluate joint label
    if 'joint-label' in counts:
        for label1, label2 in zip(sent[label2idx['Joint-Label-True']], sent[label2idx['Joint-Label-Pred']]):
            if label1 != 'None':
                counts['joint-label'].correct_cnt += 1
                counts['joint-label'].correct_types_cnt['Arc'] += 1
                counts['joint-label'].correct_types_cnt[label1] += 1
                counts['joint-label'].pred_correct_types_cnt[label1] += 0
            if label2 != 'None':
                counts['joint-label'].pred_cnt += 1
                counts['joint-label'].pred_types_cnt['Arc'] += 1
                counts['joint-label'].pred_types_cnt[label2] += 1
                counts['joint-label'].pred_correct_types_cnt[label2] += 0
            if label1 != 'None' and label2 != 'None':
                counts['joint-label'].pred_correct_types_cnt['Arc'] += 1
            if label1 == label2 and label1 != 'None':
                counts['joint-label'].pred_correct_cnt += 1
                counts['joint-label'].pred_correct_types_cnt[label1] += 1

    # evaluate separate position
    if 'separate-position' in counts:
        for positions1, positions2 in zip(sent[label2idx['Separate-Position-True']],
                                          sent[label2idx['Separate-Position-Pred']]):
            counts['separate-position'].correct_cnt += len(positions1)
            counts['separate-position'].pred_cnt += len(positions2)
            counts['separate-position'].pred_correct_cnt += len(set(positions1) & set(positions2))

    # evaluate span & entity
    correct_ent2idx = defaultdict(set)
    correct_span2ent = dict()
    correct_span = set()
    for ent, span, _ in sent[label2idx['Ent-True']]:
        correct_span.add(span)
        correct_span2ent[span] = ent
        correct_ent2idx[ent].add(span)

    pred_ent2idx = defaultdict(set)
    pred_span2ent = dict()
    for ent, span, _ in sent[label2idx['Ent-Pred']]:
        pred_span2ent[span] = ent
        pred_ent2idx[ent].add(span)

    if 'span' in counts:
        pred_span = set(sent[label2idx['Ent-Span-Pred']])
        counts['span'].correct_cnt += len(correct_span)
        counts['span'].pred_cnt += len(pred_span)
        counts['span'].pred_correct_cnt += len(correct_span & pred_span)

    if 'ent' in counts:
        all_ents = set(correct_ent2idx) | set(pred_ent2idx)
        for ent in all_ents:
            counts['ent'].correct_cnt += len(correct_ent2idx[ent])
            counts['ent'].correct_types_cnt[ent] += len(correct_ent2idx[ent])
            counts['ent'].pred_cnt += len(pred_ent2idx[ent])
            counts['ent'].pred_types_cnt[ent] += len(pred_ent2idx[ent])
            pred_correct_cnt = len(correct_ent2idx[ent] & pred_ent2idx[ent])
            counts['ent'].pred_correct_cnt += pred_correct_cnt
            counts['ent'].pred_correct_types_cnt[ent] += pred_correct_cnt
            
            

    # evaluate relation
    if 'rel' in counts:
        correct_rel2idx = defaultdict(set)
        for rel, span1, span2, _, _ in sent[label2idx['Rel-True']]:
            if span1 not in correct_span2ent or span2 not in correct_span2ent:
                continue
            correct_rel2idx[rel].add((span1, span2))

        pred_rel2idx = defaultdict(set)
        for rel, span1, span2, _, _ in sent[label2idx['Rel-Pred']]:
            if span1 not in pred_span2ent or span2 not in pred_span2ent:
                continue
            pred_rel2idx[rel].add((span1, span2))

        all_rels = set(correct_rel2idx) | set(pred_rel2idx)
        for rel in all_rels:
            counts['rel'].correct_cnt += len(correct_rel2idx[rel])
            counts['rel'].correct_types_cnt[rel] += len(correct_rel2idx[rel])
            counts['rel'].pred_cnt += len(pred_rel2idx[rel])
            counts['rel'].pred_types_cnt[rel] += len(pred_rel2idx[rel])
            pred_correct_rel_cnt = len(correct_rel2idx[rel] & pred_rel2idx[rel])
            counts['rel'].pred_correct_cnt += pred_correct_rel_cnt
            counts['rel'].pred_correct_types_cnt[rel] += pred_correct_rel_cnt

    # overlap entity evaluation
    '''
    correct_span2ent: the span of entities to the type of entities.
    
    overlap realtion evaluation only pay attention to the text and the typy of the entities.
    '''
    
    if 'overlap-ent' in counts:
        overlap_correct_ent2idx = defaultdict(set)
        for ent, span, text in sent[label2idx['Ent-True']]:
            if span not in correct_span2ent:
                continue
            overlap_correct_ent2idx[ent].add((text, correct_span2ent[span]))
            
        overlap_pred_ent2idx = defaultdict(set)
        for ent, span, text in sent[label2idx['Ent-Pred']]:
            if span not in pred_span2ent:
                continue
            overlap_pred_ent2idx[ent].add(( text, pred_span2ent[span]))
        all_overlap_ents = set(overlap_correct_ent2idx) | set(overlap_pred_ent2idx)
                
        for ent in all_overlap_ents:
            counts['overlap-ent'].correct_cnt += len(overlap_correct_ent2idx[ent])
            counts['overlap-ent'].correct_types_cnt[ent] += len(overlap_correct_ent2idx[ent])
            counts['overlap-ent'].pred_cnt += len(overlap_pred_ent2idx[ent])
            counts['overlap-ent'].pred_types_cnt[ent] += len(overlap_pred_ent2idx[ent])
            
            overlap_pred_correct_ent_cnt = cal_correct_pred(overlap_correct_ent2idx[ent], 
                                                            overlap_pred_ent2idx[ent], 
                                                            cfg.overlap_match_rate,
                                                            'ent')
            counts['overlap-ent'].pred_correct_cnt += overlap_pred_correct_ent_cnt
            counts['overlap-ent'].pred_correct_types_cnt[ent] += overlap_pred_correct_ent_cnt
            
            
    # overlap relation evaluation
    '''
    correct_span2ent: the span of entities to the type of entities.
    
    overlap realtion evaluation only pay attention to the text and the typy of the entities.
    '''

    if 'overlap-rel' in counts:
        overlap_correct_rel2idx = defaultdict(set)
        for rel, span1, span2, text1, text2 in sent[label2idx['Rel-True']]:
            if span1 not in correct_span2ent or span2 not in correct_span2ent:
                continue

            overlap_correct_rel2idx[rel].add((  text1, correct_span2ent[span1], 
                                                text2, correct_span2ent[span2]))

        overlap_pred_rel2idx = defaultdict(set)
        for rel, span1, span2, text1, text2 in sent[label2idx['Rel-Pred']]:
            if span1 not in pred_span2ent or span2 not in pred_span2ent:
                continue
            overlap_pred_rel2idx[rel].add(( text1, pred_span2ent[span1], 
                                            text2, pred_span2ent[span2]))

        all_overlap_rels = set(overlap_correct_rel2idx) | set(overlap_pred_rel2idx)
                
        for rel in all_overlap_rels:
            counts['overlap-rel'].correct_cnt += len(overlap_correct_rel2idx[rel])
            counts['overlap-rel'].correct_types_cnt[rel] += len(overlap_correct_rel2idx[rel])
            counts['overlap-rel'].pred_cnt += len(overlap_pred_rel2idx[rel])
            counts['overlap-rel'].pred_types_cnt[rel] += len(overlap_pred_rel2idx[rel])
            
            overlap_pred_correct_rel_cnt = cal_correct_pred(overlap_correct_rel2idx[rel], 
                                                            overlap_pred_rel2idx[rel],
                                                            cfg.overlap_match_rate,
                                                            'rel')
            counts['overlap-rel'].pred_correct_cnt += overlap_pred_correct_rel_cnt
            counts['overlap-rel'].pred_correct_types_cnt[rel] += overlap_pred_correct_rel_cnt

    # exact relation evaluation
    if 'exact-rel' in counts:
        exact_correct_rel2idx = defaultdict(set)
        for rel, span1, span2, _, _ in sent[label2idx['Rel-True']]:
            if span1 not in correct_span2ent or span2 not in correct_span2ent:
                continue
            exact_correct_rel2idx[rel].add((span1, correct_span2ent[span1], span2, correct_span2ent[span2]))

        exact_pred_rel2idx = defaultdict(set)
        for rel, span1, span2, text1, text2 in sent[label2idx['Rel-Pred']]:
            if span1 not in pred_span2ent or span2 not in pred_span2ent:
                continue
            exact_pred_rel2idx[rel].add((span1, pred_span2ent[span1], span2, pred_span2ent[span2]))

        all_exact_rels = set(exact_correct_rel2idx) | set(exact_pred_rel2idx)
        for rel in all_exact_rels:
            counts['exact-rel'].correct_cnt += len(exact_correct_rel2idx[rel])
            counts['exact-rel'].correct_types_cnt[rel] += len(exact_correct_rel2idx[rel])
            counts['exact-rel'].pred_cnt += len(exact_pred_rel2idx[rel])
            counts['exact-rel'].pred_types_cnt[rel] += len(exact_pred_rel2idx[rel])
            exact_pred_correct_rel_cnt = len(exact_correct_rel2idx[rel] & exact_pred_rel2idx[rel])
            counts['exact-rel'].pred_correct_cnt += exact_pred_correct_rel_cnt
            counts['exact-rel'].pred_correct_types_cnt[rel] += exact_pred_correct_rel_cnt


def report(counts):
    """This function print evaluation results
    
    Arguments:
        counts {dict} -- counters
    
    Returns:
        float -- f1 score
    """

    p, r, f = calculate_metrics(counts.pred_correct_cnt, counts.pred_cnt, counts.correct_cnt)
    logger.info("truth cnt: {} pred cnt: {} correct cnt: {}".format(counts.correct_cnt, counts.pred_cnt,
                                                                    counts.pred_correct_cnt))
    logger.info("precision: {:6.2f}%".format(100 * p))
    logger.info("recall: {:6.2f}%".format(100 * r))
    logger.info("f1: {:6.2f}%".format(100 * f))

    score = (p, r, f)

    for type in counts.pred_correct_types_cnt:
        p, r, f = calculate_metrics(counts.pred_correct_types_cnt[type], counts.pred_types_cnt[type],
                                    counts.correct_types_cnt[type])
        logger.info("-" * 50)
        logger.info("type: {}".format(type))
        logger.info("truth cnt: {} pred cnt: {} correct cnt: {}".format(counts.correct_types_cnt[type],
                                                                        counts.pred_types_cnt[type],
                                                                        counts.pred_correct_types_cnt[type]))
        logger.info("precision: {:6.2f}%".format(100 * p))
        logger.info("recall: {:6.2f}%".format(100 * r))
        logger.info("f1: {:6.2f}%".format(100 * f))

    return score


def calculate_metrics(pred_correct_cnt, pred_cnt, correct_cnt):
    """This function calculation metrics: precision, recall, f1-score
    
    Arguments:
        pred_correct_cnt {int} -- the number of corrected prediction
        pred_cnt {int} -- the number of prediction
        correct_cnt {int} -- the numbert of truth
    
    Returns:
        tuple -- precision, recall, f1-score
    """

    tp, fp, fn = pred_correct_cnt, pred_cnt - pred_correct_cnt, correct_cnt - pred_correct_cnt
    p = 0 if tp + fp == 0 else (tp / (tp + fp))
    r = 0 if tp + fn == 0 else (tp / (tp + fn))
    f = 0 if p + r == 0 else (2 * p * r / (p + r))
    return p, r, f


if __name__ == '__main__':
    eval_file(sys.argv[1])
