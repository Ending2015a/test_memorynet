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
                    n_hop = 6,
                    memory_size = 20,
                    sentence_size = 50,
                    option_size = 10)

    print('Bulid Solver...')
    solver = Solver(model, enc_map, dec_map,
                    n_epochs = 500,
                    batch_size = 64,
                    learning_rate = 0.005,
                    log_path = './log/',
                    model_path = './checkpoint/',
                    restore_path = './checkpoint/',
                    val_epoch = 1,
                    save_epoch = 1,
                    print_step = 10,
                    summary_step = 10,
                    linear_start= True)

    solver.train()

if __name__ == '__main__':
    main()
