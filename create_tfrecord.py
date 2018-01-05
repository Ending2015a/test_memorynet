import os

import tensorflow as tf
import numpy as np
import _pickle as cPickle
import time

from tqdm import tqdm

from preprocess import get_encoded_train_data
from preprocess import get_encoded_validation_data
from preprocess import get_encoded_test_data
from preprocess import decode_str
#=============================
train_record_file = './record/train/train'
val_record_file = './record/val/val'
test_record_file = './record/test/test'

train_split_count = 100
val_split_count = 4
test_split_count = 4
num_worker = 10
padding_size = 50

dec_map = './dec_map.pkl'
dec_map = cPickle.load(open(dec_map, 'rb'))
dec_map[0]=''
#=============================

def create_tfrecords(data, record_name, split_num = 100, start_from=0):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def padding_sentence(sentence, size, padding_word=0):
        length = len(sentence)
        crop = max(length-size, 0)
        pad = max(size-length, 0)
        sentence = sentence[0:length-crop] + [padding_word] * pad
        return sentence

    num_records_per_file = len(data) // split_num
    
    print('create tfrecord')

    total_count = 0

    def pack_task(record_num):
        count = 0
        writer = tf.python_io.TFRecordWriter('{}-part{}.tfrecord'.format(record_name, record_num+1))

        st = record_num * num_records_per_file
        ed = (record_num+1) * num_records_per_file if record_num != split_num -1 else len(data)

        start_time = time.time()

        for idx in tqdm(range(st, ed), desc='Examples', ncols=80, leave=False):
            sents = data[idx]['sentences']
            quest = data[idx]['question']
            opt = data[idx]['options']
            qwos = data[idx]['queries']
            index = data[idx]['index']
            mask = data[idx]['position_mask']
            
            padded_sents = [ padding_sentence(sent, padding_size) for sent in sents]
            padded_quest = padding_sentence(quest, padding_size)
            padded_qwos = [ padding_sentence(qwo, padding_size) for qwo in qwos]
            index = [index]
            padded_mask = padding_sentence(mask, padding_size)

            flatten_sents = [j for i in padded_sents for j in i]
            flatten_qwos = [j for i in padded_qwos for j in i]

            example = tf.train.Example(features=tf.train.Features(
                    feature={
                            'sentences': _int64_feature(flatten_sents),
                            'question': _int64_feature(padded_quest),
                            'options': _int64_feature(opt),
                            'queries': _int64_feature(flatten_qwos),
                            'index': _int64_feature(index),
                            'position_mask': _int64_feature(padded_mask)
                        }))

            count += 1
            writer.write(example.SerializeToString())

        writer.close()
        return count

    for i in tqdm(range(start_from, split_num), desc='Record', ncols=80):
        total_count += pack_task(i)
    print('Total records: {}'.format(split_num))
    print('Total examples: {}'.format(total_count))
    print('Examples per record: {}'.format(num_records_per_file))

def create_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


if __name__ == '__main__':
    create_path(train_record_file)
    create_path(val_record_file)
    create_path(test_record_file)

    data = get_encoded_train_data()
    create_tfrecords(data, train_record_file, train_split_count)
    
    data = get_encoded_validation_data()
    create_tfrecords(data, val_record_file, val_split_count)

    data = get_encoded_test_data()
    create_tfrecords(data, test_record_file, test_split_count)




