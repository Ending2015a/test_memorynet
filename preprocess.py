#!/usr/bin/python3

import os
import time
import string
import numpy as np
import _pickle as cPickle
import argparse


from tqdm import tqdm
from tqdm import trange

#====================

blank = 'XXXXX'
data_path = 'AI_Course_Final/CBTest/data'
output_path = 'questions/'
encode_path = 'encode/'
encode_map_path = 'enc_map.pkl'
decode_map_path = 'dec_map.pkl'
vocab_path = 'vocab.pkl'
vocab_threshold = 150

exclusive_vocab = []

train_list = ['cbtest_CN_train.txt', 
            'cbtest_NE_train.txt', 
            'cbtest_P_train.txt', 
            'cbtest_V_train.txt']

valid_list = ['cbtest_CN_valid_2000ex.txt',
            'cbtest_NE_valid_2000ex.txt',
            'cbtest_P_valid_2000ex.txt',
            'cbtest_V_valid_2000ex.txt']

test_list = ['cbtest_CN_test_2500ex.txt',
            'cbtest_NE_test_2500ex.txt',
            'cbtest_P_test_2500ex.txt',
            'cbtest_V_test_2500ex.txt']

#====================

parser = argparse.ArgumentParser(description='Rreprocessing Input Data')
parser.add_argument('-p', '--progress', type=str, nargs='+', default=['all'])

args = parser.parse_args()

#====================

if not os.path.exists(output_path):
    os.makedirs(output_path)

if not os.path.exists(encode_path):
    os.makedirs(encode_path)

#====================

def get_encoded_data(q_list):
    qa = []
    for path in tqdm(q_list, desc='Read', ncols=80):
        n, e = os.path.splitext(os.path.basename(path))
        filename = os.path.join(encode_path, n+'.pkl')
        encode_question = cPickle.load(open(filename, 'rb'))

        qa.extend(encode_question)
    return qa



def get_encoded_train_data():
    return get_encoded_data(train_list)

def get_encoded_validation_data():
    return get_encoded_data(valid_list)

def get_encoded_test_data():
    return get_encoded_data(test_list)

def read_data(filename):

    def is_int(s):
        try:
            int(s)
            return True
        except ValueError:
            return False
    lines = []
    with open(filename, 'r') as file:
        raw_lines = file.readlines()
        for line in tqdm(raw_lines, desc='Read', ncols=80):
            if is_int(line.split(' ', 1)[0]):
                lines.append(line)

    return lines

def parse_21th(line, filter, with_answer=True):
    return q, a, option, question_with_option, idx

def parse_questions(lines, with_answer=True):
    question_list = []
    assert(len(lines) % 21 == 0)
    exc = set(string.punctuation)
    def parse(lines, with_answer):
        sent = []
        def filter(line):
            line = ''.join(ch if ch not in exc else ' ' for ch in line)
            return ' '.join(line.split())

        for i in range(20):
            line = lines[i].split(' ', 1)[1]
            sent.append( filter(line) )

        last = [x for x in lines[-1].split('\t') if x!='' ]
        q = filter(last[0].split(' ', 1)[1]).replace(blank, '<blank>')
        option = []
        question_with_option = []

        if with_answer:
            a = ''.join(ch for ch in last[1] if ch not in exc)
            raw_option = last[2].split('|')
        else:
            a = ''
            raw_option = last[1].split('|')
        if len(raw_option) > 10:
            raw_option = raw_option[0:10]
        elif len(raw_option) < 10:
            raw_option = raw_option + ['']*(10-raw_option)

        for x in raw_option:
            o = ''.join(ch for ch in x.replace('\n', '') if ch not in exc)
            option.append(o)
            o = q.replace('<blank>', o)
            question_with_option.append(o)

        assert(len(sent) == 20)
        assert(len(option) == 10)

        if with_answer:
            idx = option.index(a)
        else:
            idx = 0

        question = {'sentences': sent, 
                    'question': q, 
                    'answer': a, 
                    'options': option,
                    'queries': question_with_option,
                    'index': idx}
        return question

    for i in trange(len(lines) // 21, desc='Parsing', ncols=80):
        question_list.append(parse(lines[ i*21 : (i+1)*21], with_answer))

    return question_list


def create_vocab(qs, exc=[], thres=50, create_ans_vocab=False):
    vocab = {}
    ans_vocab = {}
    def add(word, vocab):
        word = word.lower()
        if word in exc:
            return
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1

    for q in tqdm(qs, desc='vocab', ncols=80):
        for line in q['sentences']:
            for word in line.split():
                add(word, vocab)
        for word in q['question'].split():
            add(word, vocab)
        for word in q['options']:
            add(word, vocab)
        add(q['answer'], vocab)
        add(q['answer'], ans_vocab)

    if create_ans_vocab == False:
        return vocab
    else:
        return vocab, ans_vocab


def build_mapping(vocab, thres=50, ans_vocab={}):
    def add(enc_map, dec_map, voc):
        enc_map[voc] = len(dec_map)
        dec_map[len(dec_map)] = voc
        return enc_map, dec_map

    enc_map, dec_map = {}, {}
    for voc in ['<blank>', '<rare>']:
        enc_map, dec_map = add(enc_map, dec_map, voc)
    for voc, cnt in tqdm(vocab.items(), desc='map', ncols=80):
        if voc in enc_map:
            continue
        if cnt < thres and not voc in ans_vocab:
            enc_map[voc] = enc_map['<rare>']
        else:
            enc_map, dec_map = add(enc_map, dec_map, voc)

    return enc_map, dec_map

def encode_questions(raw_qs, enc_map):
    enc_qs = []
    for q in tqdm(raw_qs, desc='enc', ncols=80):
        try:
            enc_q = encode_question(q, enc_map)
        except KeyError as e:
            print_question(q)
            raise e
        enc_qs.append(enc_q)
    return enc_qs

def encode_question(q, enc_map):
    enc_st = []
    for sent in q['sentences']:
        ids = encode_str(sent, enc_map)
        enc_st.append(ids)

    enc_q = encode_str(q['question'], enc_map)
    enc_a = encode(q['answer'], enc_map)

    enc_o = [encode(opt, enc_map) for opt in q['options']]

    enc_mask = [0] * len(enc_q)
    try:
        enc_mask[enc_q.index(0)] = 1
    except:
        pass

    enc_qwo = []
    for qwo in q['queries']:
        ids = encode_str(qwo, enc_map)
        enc_qwo.append(ids)

    enc_q = {'sentences': enc_st, 
            'question': enc_q,
            'answer': enc_a,
            'options': enc_o,
            'queries': enc_qwo,
            'index': q['index'],
            'position_mask': enc_mask}

    return enc_q

def data_augmentation(index, *args):
    l = [ 1 if i==index else 0 for i in range(len(args[0]))]
    x = list(zip(l, *args))
    import random
    random.shuffle(x)
    l, *args = zip(*x)
    index = l.index(1)
    return (index, *args)

def decode_question(q, dec_map):
    raw_st = []
    for ids in q['sentences']:
        sent = decode_str(ids, dec_map)
        raw_st.append(sent)
    raw_q = decode_str(q['question'], dec_map)
    raw_a = decode(q['answer'], dec_map)
    raw_o = [decode(opt, dec_map) for opt in q['options']]

    raw_qwo = []
    for ids in q['queries']:
        qwo = decode_str(ids, dec_map)
        raw_qwo.append(qwo)

    raw_q = {'sentences': raw_st,
            'question': raw_q,
            'answer': raw_a,
            'options': raw_o,
            'queries': raw_qwo,
            'index': q['index']}

    return raw_q

def encode(word, e_map):
    word = word.lower()
    return e_map[word] if word in e_map else e_map['<rare>']

def decode(ids, d_map):
    return d_map[ids]

def encode_str(sent, e_map):
    ids = [e_map[x.lower()] if x.lower() in e_map else e_map['<rare>'] for x in sent.split()]
    return ids

def decode_str(ids, d_map):
    return ' '.join([d_map[x] for x in ids])


def print_question(q):
    for idx, sent in enumerate(q['sentences']):
        print('S{}: {}'.format(idx, sent))
    print('Q: ', q['question'])
    print('Opt: ', q['options'])
    print('A: ', q['answer'])
    for idx, qwo in enumerate(q['queries']):
        print('Opt{}: {}'.format(idx, qwo))
    print('ID: ', q['index'])

def extract_questions(q, total_sent=0, total_q_size=0, max_sent_size=0, avg_sent_size=0, max_query_size=0, avg_query_size=0):

    for i in range(len(q)):
        for j in range(len(q[i]['sentences'])):
            sent_size = len(q[i]['sentences'][j])
            max_sent_size = max(sent_size, max_sent_size)
            avg_sent_size = avg_sent_size * (total_sent/(total_sent+1)) + sent_size/(total_sent+1)
            total_sent += 1

        query_size = len(q[i]['question'])
        max_query_size = max(query_size, max_query_size)
        avg_query_size = avg_query_size * (total_q_size/(total_q_size+1)) + query_size/(total_q_size+1)

        total_q_size += 1

    return total_sent, total_q_size, max_sent_size, avg_sent_size, max_query_size, avg_query_size


def crop_or_pad(sentence, size, padding_word=0):
    length = len(sentence)
    crop = max(length-size, 0)
    pad = max(size-length, 0)
    sentence = sentence[0:length-crop] + [padding_word] * pad
    return sentence

def parse_input_data_list(lines, enc_map, sentence_size, has_answer=False):
    question_list = parse_questions(lines, has_answer)
    enc_questions = encode_questions(question_list, enc_map)

    padded_enc = []
    for q in enc_questions: 
        q['sentences'] = [ crop_or_pad(sent, sentence_size) for sent in q['sentences']]
        q['question'] = crop_or_pad(q['question'], sentence_size)
        q['queries'] = [ crop_or_pad(qr, sentence_size) for qr in q['queries']]
        q['options'] = q['options']
        q['index'] = [q['index']]

        padded_enc.append(q)

    return padded_enc

if __name__ == '__main__':

    if 'parse' in args.progress or 'all' in args.progress:
        print('Parsing input data...')
        for name in train_list + valid_list + test_list:
            n, _ = os.path.splitext(name)
            filename = os.path.join(data_path, name)
            print('Reading data: ', filename)

            # read input data
            lines = read_data(filename)

            print('Read {} lines'.format(len(lines)))

            print('Parsing data: ', filename)

            # parse to questing format
            question_list = parse_questions(lines)

            print('Result: {}/{} (actual/expected)'.format(len(question_list), len(lines)//21))

            outputfile = os.path.join(output_path, n + '.pkl')
            print('Save to {}'.format(outputfile))

            # save to file
            with open(outputfile, 'wb') as f:
                cPickle.dump(question_list, f)
        print('======= TEST OUTPUT ========')
        print_question(question_list[0])
        print('DONE !!')

    gen_pkl_file = [ os.path.join(output_path, x) 
                        for x in os.listdir(output_path) if x.endswith('.pkl') ]

    if 'dict' in args.progress or 'all' in args.progress:
        print('Generating Dictionary...')

        qs = []
        for file in gen_pkl_file:
            qs.extend(cPickle.load( open(file, 'rb') ))

        # create vocabulary
        vocab, ans_vocab = create_vocab(qs, exc=exclusive_vocab, thres=vocab_threshold, create_ans_vocab=True)

        print('DONE !!')
        print('Total: {} words'.format(len(vocab)))
        x = np.array(list(vocab.values()))
        print('Thres total (>={}): {} words'.format(vocab_threshold, np.sum(x[(-x).argsort()] >= vocab_threshold)))
        
        print('Save dictionary to {}'.format(vocab_path))
        # save to file
        cPickle.dump(vocab, open(vocab_path, 'wb'))


    vocab = cPickle.load( open(vocab_path, 'rb') )

    if 'map' in args.progress or 'all' in args.progress:

        print('Buliding voc mapping...')

        x = np.array(list(vocab.values()))
        vocab_size = len(vocab)
        vocab_thres_size = np.sum(x[(-x).argsort()] >= vocab_threshold)

        enc_map, dec_map = build_mapping(vocab, vocab_threshold, ans_vocab)

        print('Encode map size: {}/{} (actual/expected)'.format(len(enc_map), vocab_size+1))
        print('Decode map size: {}/{} (actual/expected)'.format(len(dec_map), vocab_thres_size+1))

        print('Save encode map to {}'.format(encode_map_path))
        cPickle.dump(enc_map, open(encode_map_path, 'wb'))
        print('Save decode map to {}'.format(decode_map_path))
        cPickle.dump(dec_map, open(decode_map_path, 'wb'))

        print('DONE !!')

    enc_map = cPickle.load( open(encode_map_path, 'rb') )
    dec_map = cPickle.load( open(decode_map_path, 'rb') )

    if 'enc' in args.progress or 'all' in args.progress:
        print('Encoding questions...')

        for file in gen_pkl_file:
            filename = os.path.basename(file)

            print('Encoding data: {}'.format(filename))
            questions = cPickle.load(open(file, 'rb'))
            enc_questions = encode_questions(questions, enc_map)
            outputfile = os.path.join(encode_path, filename)

            print('Save to {}'.format(outputfile))
            with open(outputfile, 'wb') as f:
                cPickle.dump(enc_questions, f)

        print('DONE !!')


    print('All DONE !!')

    if 'sum' in args.progress or 'all' in args.progress:
        print('Generating Summary...')

        vocab = cPickle.load(open(vocab_path, 'rb'))
        enc_map = cPickle.load(open(encode_map_path, 'rb'))
        dec_map = cPickle.load(open(decode_map_path, 'rb'))

        train_size = 0
        val_size = 0
        test_size = 0

        sent_list = []
        q_size_list = []
        max_sent = 0
        avg_sent_list = []
        max_query = 0
        avg_query_list = []

        print('========= SUMMARY ==========')

        def summary(q_list):
            total_size = 0
            for path in q_list:
                n, e = os.path.splitext(os.path.basename(path))
                filename = os.path.join(encode_path, n+'.pkl')

                print('')
                print('=== File: ', filename)

                encode_question = cPickle.load(open(filename, 'rb'))

                total_sent, total_q_size, max_sent_size, avg_sent_size, max_query_size, avg_query_size = extract_questions(encode_question)

                print('Maximum Sentence size: ', max_sent_size)
                print('Average Sentence size: ', avg_sent_size)
                print('Maximum Query size: ', max_query_size)
                print('Average Query size: ', avg_query_size)
                print('Total Question: ', total_q_size)

                global sent_list
                global q_size_list
                global max_sent
                global max_query
                global avg_sent_list
                global avg_query_list

                total_size += total_q_size
                sent_list.append(total_sent)            
                q_size_list.append(total_q_size)
                max_sent = max(max_sent, max_sent_size)
                max_query = max(max_query, max_query_size)
                avg_sent_list.append(avg_sent_size)
                avg_query_list.append(avg_query_size)
            return total_size

        train_size = summary(train_list)
        val_size = summary(valid_list)
        test_size = summary(test_list)

        avg_sent = sum(ts*avgs for ts, avgs in zip(sent_list, avg_sent_list))/sum(sent_list)
        avg_query = sum(tq*avgq for tq, avgq in zip(q_size_list, avg_query_list))/sum(q_size_list)


        print('')
        print('========= SUMMARY ==========')

        print('')
        print('=========== INFO ===========')
        print('Encode map size: ', len(enc_map))
        print('Decode map size: ', len(dec_map))
        print('Vocabulary size: ', len(vocab))
        print('Maximum Sentence size: ', max_sent)
        print('Average Sentence size: ', avg_sent)
        print('Maximum Query size: ', max_query)
        print('Average Query size: ', avg_query)
        print('Total Question: ', train_size + val_size + test_size)
        print('  |-- Train: ', train_size)
        print('  |-- Val: ', val_size)
        print('  |-- Test: ', test_size)


    print('')
    print('========== FORMAT ==========')
    print('questions')
    print('  |-- sentences (a string list, contains 20 lines of sentences)')
    print('  |-- questions (a string list, which is the original question)')
    print('  |-- answer (an string which is the answer of this question)')
    print('  |-- options (a string list, contains multiple options for this question)')
    print('  |-- queries (a string list, contains 10 lines of original questions with 10 different options filled in the blank)')
    print('  |-- index (an integer, specifies the index of the correct option)')


    #print('')
    #print('========== FORMAT ==========')
    #print('questions')
    #print('  |-- sentences (a string list contains 20 lines of sentences)')
    #print('  |-- question (a string which is the main question)')
    #print('  |-- answer (a string which is the answer of this question)')
    #print('  |-- options (a string list contains multiple options of this question)')


#filename = os.path.join(data_path, train_list[0])
#lines = read_data(filename)
#print('filename: ', filename)
#print('total line: ', len(lines))

#for i in range(21, 42):
#    print(lines[i])

#question_list = parse_questions(lines)
#print('total questions: ', len(question_list))

#question = question_list[1]

#for i in range(20):
#    print('{}: {}'.format(i+1, question['sentences'][i]))
    
#print('Q: {}'.format(question['question']))
#print('Opt: {}'.format(question['options']))
#print('A: {}'.format(question['answer']))

#for name in train_list + valid_list + test_list:
#
#    filename = os.path.join(data_path, name)
#    lines = read_data(filename)
#
#    print('filename: ', filename)
#    print('total line: ', len(lines))
#    for i in range(21):
#        print(lines[i])



