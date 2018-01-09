import _pickle as cPickle
from model import MemNet
from solver import Solver

from preprocess import read_data
from preprocess import parse_input_data_list

#=============================
encode_map = 'enc_map.pkl'
decode_map = 'dec_map.pkl'

predict_file = './AI_Course_Final/Test_Set/test_set.txt'
#=============================

def main():

    print('Restoring map...')
    enc_map = cPickle.load(open(encode_map, 'rb'))
    dec_map = cPickle.load(open(decode_map, 'rb'))
    vocab_size = len(dec_map)

    print('Bulid Dataset...')
    lines = read_data(predict_file)
    question_list = parse_input_data_list(lines, enc_map, 50, False)


    print('Bulid Model...')
    model = MemNet(vocab_size = vocab_size,
                    embed_size = 512,
                    n_hop = 6,
                    memory_size = 20,
                    sentence_size = 50,
                    option_size = 10)

    print('Bulid Solver...')
    solver = Solver(model, enc_map, dec_map,
                    eval_batch_size = 1,
                    test_record_path = './record/test/',
                    test_examples = 10000,
                    restore_path = './checkpoint/',
                    print_step = 5)

    answer = solver.predict(question_list)
    idx = [x for x in range(1, len(question_list)+1)]
    import pandas as pd
    df = pd.DataFrame(data={'answer':answer})
    df.index += 1
    df.to_csv('predict.csv', index=True, index_label='id')

if __name__ == '__main__':
    main()
