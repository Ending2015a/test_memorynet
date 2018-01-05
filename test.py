import _pickle as cPickle
from model import MemNet
from solver import Solver

#=============================
encode_map = 'enc_map.pkl'
decode_map = 'dec_map.pkl'
#=============================

def main():

    print('Restoring map...')
    enc_map = cPickle.load(open(encode_map, 'rb'))
    dec_map = cPickle.load(open(decode_map, 'rb'))
    vocab_size = len(dec_map)

    print('Bulid Model...')
    model = MemNet(vocab_size = vocab_size,
                    embed_size = 512,
                    n_hop = 10,
                    memory_size = 20,
                    sentence_size = 50,
                    option_size = 10)

    print('Bulid Solver...')
    solver = Solver(model, enc_map, dec_map,
                    eval_batch_size = 64,
                    test_record_path = './record/test/',
                    test_examples = 10000,
                    restore_path = './checkpoint/',
                    print_step = 5)

    solver.test()

if __name__ == '__main__':
    main()
