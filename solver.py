import os
import time
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from tqdm import trange
from preprocess import decode

class Solver(object):
    def __init__(self, model, enc_map, dec_map, **kwargs):

        self.model = model
        self.enc_map = enc_map
        self.dec_map = dec_map
        self.dec_map[0] = ''

        #======= TRAINING ARGUMENTS =====
        self.n_epochs = kwargs.pop('n_epochs', 500)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.max_norm_clip = kwargs.pop('max_norm_clip', 40.0)
        self.decay_epoch = kwargs.pop('decay_epoch', 25)
        self.decay_rate = kwargs.pop('decay_rate', 0.65)
        self.val_epoch = kwargs.pop('val_epoch', 1)

        self.linear_start = kwargs.pop('linear_start', True)
        if self.linear_start:
            self.linear_start = 1
        else:
            self.linear_start = 0

        #===== EVALUATION ARGUMENTS =====
        self.eval_batch_size = kwargs.pop('eval_batch_size', 64)

        #======= LOGGING ARGUMENTS ======
        self.save_epoch = kwargs.pop('save_epoch', 1)
        self.print_step = kwargs.pop('print_step', 5)
        self.summary_step = kwargs.pop('summary_step', 10)
        self.print_n_words = kwargs.pop('print_n_words', 20)

        #======= LOGGING PATH =======
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.restore_path = kwargs.pop('restore_path', None)

        #======= DATASET ========
        self.train_record_path = kwargs.pop('train_record_path', './record/train/')
        self.val_record_path = kwargs.pop('val_record_path', './record/val/')
        self.test_record_path = kwargs.pop('test_record_path', './record/test/')
        self.train_examples = kwargs.pop('train_examples', 669343)
        self.val_examples = kwargs.pop('val_examples', 8000)
        self.test_examples = kwargs.pop('test_examples', 10000)


        #======= MODEL =========
        self.sentence_size = model.sentence_size
        self.memory_size = model.memory_size
        self.option_size = model.option_size


        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def create_dataset(self, record_path, batch_size):
        def training_parser(record):
            keys_to_features = {
                'sentences': tf.FixedLenFeature([self.sentence_size*self.memory_size], tf.int64),
                'question': tf.FixedLenFeature([self.sentence_size], tf.int64),
                'options': tf.FixedLenFeature([self.option_size], tf.int64),
                'queries': tf.FixedLenFeature([self.sentence_size*self.option_size], tf.int64),
                'index': tf.FixedLenFeature([1], dtype=tf.int64),
                'position_mask': tf.FixedLenFeature([self.sentence_size], dtype=tf.int64)}

            features = tf.parse_single_example(record, features=keys_to_features)
            sentences = features['sentences']
            questions = features['question']
            options = features['options']
            queries = features['queries']
            answer = features['index']
            mask = features['position_mask']

            sentences = tf.reshape(sentences, [self.memory_size, self.sentence_size])
            queries = tf.reshape(queries, [self.option_size, self.sentence_size])

            records = {
                    'sentences': sentences,
                    'question': questions,
                    'options': options,
                    'queries': queries,
                    'index': answer,
                    'position_mask': mask}
            return records

        def tfrecord_iterator(filenames, batch_size, record_parser):
            dataset = tf.data.TFRecordDataset(filenames)
            dataset = dataset.map(record_parser, num_parallel_calls=32)

            dataset = dataset.repeat()
            dataset = dataset.shuffle(batch_size*3)

            dataset = dataset.batch(batch_size)
            iterator = dataset.make_initializable_iterator()
            output_types = dataset.output_types
            output_shapes = dataset.output_shapes

            return iterator, output_types, output_shapes

        filenames = [os.path.join(record_path, x) for x in os.listdir(record_path)]
        iterator, types, shapes = tfrecord_iterator(filenames, batch_size, training_parser)
        records = iterator.get_next()

        sentences = tf.reshape(records['sentences'], [-1, self.memory_size, self.sentence_size])
        questions = tf.reshape(records['question'], [-1, self.sentence_size])
        options = tf.reshape(records['options'], [-1, self.option_size])
        queries = tf.reshape(records['queries'], [-1, self.option_size, self.sentence_size])
        mask = tf.reshape(records['position_mask'], [-1, self.sentence_size])
        answers = records['index']

        data = {'sentences': sentences, 
                'question':questions, 
                'options': options,
                'queries': queries,
                'answer': answers,
                'mask': mask}

        return iterator, data


    def build_dataset(self, data_list):
        def padding_sentence(sentence, size, padding_word=0):
            length = len(sentence)
            crop = max(length-size, 0)
            pad = max(size-length, 0)
            sentence = sentence[0:length-crop] + [padding_word]*pad
            return sentence

        sents_list = []
        qwos_list = []
        idx_list = []

        for q in tqdm(data_list, desc='gen', ncols=80):
            sents = q['sentences']
            qwos = q['queries']
            idx = q['index']

            for i in range(len(sents)):
                sents[i] = padding_sentence(sents[i], self.sentence_size)
                assert(len(sents[i]) == self.sentence_size)

            for i in range(len(qwos)):
                qwos[i] = padding_sentence(qwos[i], self.sentence_size)
                assert(len(qwos[i]) == self.sentence_size)
            
            sents = sents[0:self.sentence_size]
            qwos = qwos[0:self.option_size]

            assert(len(sents) == 20)
            assert(len(qwos) == 10)

            sents_list.append(sents)
            qwos_list.append(qwos)
            idx_list.append([idx])


        sent, qwo, idx = tf.train.slice_input_producer(
                [sents_list, qwos_list, idx_list], capacity=self.batch_size * 8)

        sent_batch, qwo_batch, idx_batch = tf.train.shuffle_batch(
                [sent, qwo, idx],
                batch_size = self.batch_size,
                num_threads = 8,
                capacity = self.batch_size * 5,
                min_after_dequeue = self.batch_size * 2)

        return sent_batch, qwo_batch, idx_batch

    def gradient_noise(self, t, stddev=1e-3, name=None):
        t = tf.convert_to_tensor(t)
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name)

    def train(self):

        # create global step
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        
        # build dataset
        print(' :: Generating Dataset...')

        # create training & testing dataset
        train_iter, train_data = self.create_dataset(self.train_record_path, self.batch_size);
        val_iter, val_data = self.create_dataset(self.val_record_path, self.batch_size);

        # training set info
        train_examples = self.train_examples
        train_iters_per_epoch = int(np.ceil(float(train_examples)/self.batch_size))

        # validation set info
        val_examples = self.val_examples
        val_iters_per_epoch = int(np.ceil(float(val_examples)/self.batch_size))

        # reduce memory consumption

        print('     DONE !!')

        print(' :: Building model...')

        # build model & sampler
        train_handle, loss = self.model.build_model(train_data['sentences'], 
                                                    train_data['question'], 
                                                    train_data['options'],
                                                    train_data['answer'])
        val_handle, generated_answer = self.model.build_sampler(val_data['sentences'], 
                                                                val_data['question'], 
                                                                val_data['options'])

        # create optimizer & apply gradients
        with tf.name_scope('optimizer'):

            decay_steps = self.decay_epoch * train_iters_per_epoch
            # create learning rate with exponential decay
            learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, decay_steps, self.decay_rate)

            # please refer the original paper
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            grads = optimizer.compute_gradients(loss)

            # clip norm, without this, the gradients will be too large and crash the network
            # please refer the original paper section 4.2
            grads = [(tf.clip_by_norm(grad, self.max_norm_clip), var) for grad, var in grads]
            grads = [(self.gradient_noise(grad), var) for grad, var in grads] # add random noise

            # add to summary
            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradient', grad)
            # apply gradients
            train_op = optimizer.apply_gradients(grads, global_step=global_step)

        # add to summary
        tf.summary.scalar('batch_loss', loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        summary_op = tf.summary.merge_all()

        feed_dict = {train_handle.linear_start: self.linear_start}

        print('     DONE !!\n')

        print(' ======= INFO =======')
        print(' :: Total epochs for training: ', self.n_epochs)
        print(' :: Batch size: ', self.batch_size)
        print(' :: Training data size: ', train_examples)
        print(" :: Validation data size: ", val_examples)
        print(' :: Total training iterations per epoch: ', train_iters_per_epoch)
        print(' :: Total validation iterations per epoch: ', val_iters_per_epoch)
        print(' :: Linear start: ', self.linear_start)
        print('')

        print(' :: Start Session...')
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)

            print(' :: Start queue runners...')
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            print(' :: Try to restore model...')
            if self.restore_path is not None:
                latest_ckpt = tf.train.latest_checkpoint(self.restore_path)
                if not latest_ckpt:
                    print('    [Not found] any checkpoint in ', self.restore_path)
                else:
                    print('    [Found] pretrained model ', latest_ckpt)
                    saver.restore(sess, latest_ckpt)

            sess.run(train_iter.initializer, feed_dict=feed_dict)
            sess.run(val_iter.initializer, feed_dict=feed_dict)

            prev_loss = -1
            curr_loss = 0

            print(' :: Start training !!!')

            try:
                save_point = 0
                for e in range(self.n_epochs):
                    start_epoch_time = time.time()
                    start_iter_time = time.time()
                    for i in range(train_iters_per_epoch):

                        op = [global_step, train_handle.option, train_handle.selection, train_handle.answer, learning_rate, loss, train_op]
                        step_, o_, s_, a_, lr_, loss_, _ = sess.run(op, feed_dict=feed_dict)

                        curr_loss += loss_

                        if (i+1) % self.summary_step == 0:
                            summary = sess.run(summary_op, feed_dict=feed_dict)
                            summary_writer.add_summary(summary, global_step=step_)

                        if (i+1) % self.print_step == 0:
                            elapsed_iter_time = time.time() - start_iter_time

                            flat_s_ = np.array(s_).reshape(-1)
                            flat_a_ = np.array(a_).reshape(-1)

                            accuracy_ = np.sum(flat_s_ == flat_a_) / float(self.batch_size)

                            print('[epoch {} | iter {}/{} | step {} | save point {}] learning rate: {:.5f}, loss: {:.5f}, accuracy: {:.4f}, elapsed time: {:.4f} s \n'.format(
                                    e+1, i+1, train_iters_per_epoch, int(step_)+1, int(save_point), lr_, loss_, accuracy_, elapsed_iter_time))

                            _selection = decode(o_[0][int(s_[0])], self.dec_map)
                            _answer = decode(o_[0][int(a_[0])], self.dec_map)

                            print('  Answer: {}, {}'.format(int(a_[0]), _answer))
                            print('  Select: {}, {}\n'.format(int(s_[0]), _selection))

                            start_iter_time = time.time()

                    print('  [epoch {0} | iter {1}/{1} | step {2} | save point {6}] End. prev loss: {3:.5f}, cur loss: {4:.5f}, elapsed time: {5:.4f} s\n'.format(
                                e+1, train_iters_per_epoch, int(step_)+1, prev_loss, curr_loss, time.time() - start_epoch_time, save_point))

                    prev_loss = curr_loss
                    curr_loss = 0


                    if (e+1) % self.val_epoch == 0:
                        total_correct = 0
                        total_wrong = 0
                        for i in range(val_iters_per_epoch):
                            op = [val_handle.selection, val_data['answer']]
                            s_, a_ = sess.run(op, feed_dict=feed_dict)

                            s_ = np.array(s_).reshape(-1)
                            a_ = np.array(a_).reshape(-1)

                            correct = np.sum(s_ == a_)
                            wrong = np.sum(s_ != a_)
                            if (i+1) % self.print_step == 0:                            
                                print('[eval] [epoch {} | iter {}/{} | save point {}] correct: {}, wrong: {}'.format(
                                        e+1, i+1, val_iters_per_epoch, save_point, correct, wrong))
                            total_correct += correct
                            total_wrong += wrong

                        accuracy = float(total_correct) / float(total_correct + total_wrong)
                        print('\n[eval] [epoch {} | save point {}] total C/W: {}/{}, accuracy: {:.4f}\n'.format(e+1, save_point, total_correct, total_wrong, accuracy))


                    if (e+1) % self.save_epoch == 0:
                        save_point = step_
                        saver.save(sess, os.path.join(self.model_path, 'model'), global_step=global_step)
                        print('model-%s saved. ' % (step_))

            except KeyboardInterrupt:
                print('Interrupt!!')
            finally:
                print('End training step, saving model')
                saver.save(sess, os.path.join(self.model_path, 'model'), global_step=global_step)
                print('model-%s saved.' % (step_))

                coord.request_stop()
                coord.join(threads)



    def test(self):
        # create global step
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        
        # build dataset
        print(' :: Generating Dataset...')

        # create testing dataset
        test_iter, test_data = self.create_dataset(self.test_record_path, self.eval_batch_size);

        # validation set info
        test_examples = self.test_examples
        test_iters_per_epoch = int(np.ceil(float(test_examples)/self.eval_batch_size))

        # reduce memory consumption

        print('     DONE !!')

        print(' :: Building model...')

        # build model & sampler
        test_handle, generated_answer = self.model.build_sampler(test_data['sentences'],
                                                                test_data['question'],
                                                                test_data['options'])

        print('     DONE !!\n')

        print(' ======= INFO =======')
        print(' :: Eval batch size: ', self.eval_batch_size)
        print(" :: Testing data size: ", test_examples)
        print(' :: Total testing iterations per epoch: ', test_iters_per_epoch)
        print('')

        print(' :: Start Session...')
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)

            print(' :: Start queue runners...')
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            print(' :: Try to restore model...')
            if self.restore_path is not None:
                latest_ckpt = tf.train.latest_checkpoint(self.restore_path)
                if not latest_ckpt:
                    print('    [Not found] any checkpoint in ', self.restore_path)
                else:
                    print('    [Found] pretrained model ', latest_ckpt)
                    saver.restore(sess, latest_ckpt)

            sess.run(test_iter.initializer)

            print(' :: Start testing !!!')


            total_correct = 0
            total_wrong = 0
            for i in range(test_iters_per_epoch):
                op = [test_handle.selection, test_data['answer']]
                s_, a_ = sess.run(op)

                s_ = np.array(s_).reshape(-1)
                a_ = np.array(a_).reshape(-1)

                correct = np.sum(s_ == a_)
                wrong = np.sum(s_ != a_)
                if (i+1) % self.print_step == 0:                            
                    print('[eval] [epoch {} | iter {}/{} | save point {}] correct: {}, wrong: {}'.format(
                                    e+1, i+1, val_iters_per_epoch, save_point, correct, wrong))
                total_correct += correct
                total_wrong += wrong

            accuracy = float(total_correct) / float(total_correct + total_wrong)
            print('\n[eval] [epoch {} | save point {}] total O/X: {}/{}, accuracy: {:.4f}\n'.format(e+1, save_point, total_correct, total_wrong, accuracy))

            coord.request_stop()
            coord.join(threads)


