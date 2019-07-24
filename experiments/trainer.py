import tensorflow as tf
from ..model.set_utils import row_wise_mlp
import sys
import numpy as np
import time

class Trainer():
    def __init__(self, model_class, dataset_fetcher):
        self.model_class = model_class
        self.fetcher = dataset_fetcher
        lr_decay = 0.9
        self.lr = tf.Variable(1e-3, trainable=False)
        self.lr_update = tf.assign(self.lr, self.lr * lr_decay)
    
    def _model_results(self):
        # placeholder for data input and labels
        self.inputs = tf.placeholder(tf.float64, (None, None, 3), 'inputs')
        self.ys = tf.placeholder(tf.int32, (None,), 'ys')
        self.logits = self.model_class(self.inputs).get_model()

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.ys)
        # Predictions
        pred = tf.to_int32(tf.argmax(self.logits, axis=1))
        # Accuracy
        acc = tf.to_float(tf.equal(pred, self.ys))
        return tf.reduce_mean(loss), tf.reduce_sum(acc)
    
    def fit(self, sess):
        best_acc = loss_val = 0.0
        lr_val = sess.run(self.lr)
        print("on fit...")

        for epoch in range(350):
            print("epoch {} training".format(epoch))
            sys.stdout.flush()
            train_acc = 0.0
            for inps, ys in self.fetcher.train_batch():
                loss_val, _ = sess.run(
                    (self.loss, self.train_step),
                    feed_dict={
                        self.inputs: inps,
                        self.ys: ys,
                    },
                )
            
                train_acc, logits = sess.run(
                    (self.acc, self.logits),
                    feed_dict={
                        self.inputs: inps,
                        self.ys: ys,
                    },
                )
                train_acc = train_acc/len(ys)
                # print(logits)
            test_acc = self.evaluate(sess)
            if test_acc > best_acc:
                best_acc = test_acc
                # self.saver.save(sess, self.model_path)
            print("Current Train Acc: {} Test Acc: {} Best Acc: {}".format(
                train_acc, test_acc, best_acc
            ))

            if (epoch+1) % 10 == 0:
                sess.run(self.lr_update)
                lr_val = sess.run(self.lr)
                print("Updating lr... new lr: {}".format(lr_val))
        return best_acc


    # evaluate against test data
    def evaluate(self, sess):
        counts = 0
        sum_acc = 0.0
        for inps, ys in self.fetcher.test_batch():
            counts += len(inps)
            sum_acc += sess.run(
                self.acc,
                feed_dict={
                    self.inputs: inps, self.ys: ys,
                })
        return sum_acc/counts

    def train(self):
        self.loss, self.acc = self._model_results()
        # Optimizer
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(
            self.loss, colocate_gradients_with_ops=True
        )
        # Saver
        self.saver = tf.train.Saver()

        with tf.Session() as sess:
            print("starting trainer...")
            sess.run(tf.global_variables_initializer())
            self.fit(sess)
            