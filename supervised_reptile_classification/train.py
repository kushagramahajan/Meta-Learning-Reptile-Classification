"""
Training helpers for supervised meta-learning.
"""

import os
import time

import tensorflow as tf

from .reptile import Reptile
from .variables import weight_decay

import random
from supervised_reptile_classification.isic2018 import read_dataset, split_dataset, augment_dataset


# pylint: disable=R0913,R0914
def train(sess,
          model,
          save_dir,
          counter,
          train_acc_counter,
          valid_acc_counter,
          rest_train_list,
          num_classes=2,
          num_shots=5,
          inner_batch_size=5,
          inner_iters=20,
          replacement=False,
          meta_step_size=0.1,
          meta_step_size_final=0.1,
          meta_batch_size=1,
          meta_iters=400000,
          eval_inner_batch_size=5,
          eval_inner_iters=50,
          eval_interval=10,
          weight_decay_rate=1,
          time_deadline=None,
          train_shots=None,
          transductive=False,
          reptile_fn=Reptile,
          log_fn=print):
    """
    Train a model on a dataset.
    """

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    saver = tf.train.Saver()
    reptile = reptile_fn(sess,
                         transductive=transductive,
                         pre_step_op=weight_decay(weight_decay_rate))
    accuracy_ph = tf.placeholder(tf.float32, shape=())
    tf.summary.scalar('accuracy', accuracy_ph)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(save_dir, 'train'), sess.graph)
    validation_writer = tf.summary.FileWriter(os.path.join(save_dir, 'validation'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(save_dir, 'test'), sess.graph)
    tf.global_variables_initializer().run()
    sess.run(tf.global_variables_initializer())

    for i in range(meta_iters):

        for tr_index in rest_train_list:
            # random_classes = random.sample(range(1, 5), 2)
            # while((random_classes[0] == val1 and random_classes[1] == val2) or (random_classes[1] == val1 and random_classes[0] == val2)):
            #     random_classes = random.sample(range(1, 5), 2)

            random_classes = possible_pairs[tr_index]
            
            data_dir1_train = '/home/ilab/Downloads/ISIC2018_Task3_Training_class/class'+str(random_classes[0])+'/train'
            data_dir2_train = '/home/ilab/Downloads/ISIC2018_Task3_Training_class/class'+str(random_classes[1])+'/train'
            train_set1 = read_dataset(data_dir1_train)
            train_set1 = list((train_set1))
            train_set2 = read_dataset(data_dir2_train)
            train_set2 = list((train_set2))
            train_set = train_set1 + train_set2

            data_dir1_test = '/home/ilab/Downloads/ISIC2018_Task3_Training_class/class'+str(random_classes[0])+'/test'
            data_dir2_test = '/home/ilab/Downloads/ISIC2018_Task3_Training_class/class'+str(random_classes[1])+'/test'
            test_set1 = read_dataset(data_dir1_test)
            test_set1 = list(test_set1)
            test_set2 = read_dataset(data_dir2_test)
            test_set2 = list(test_set2)
            test_set = test_set1 + test_set2

            num_classes = 2
            
            frac_done = i / meta_iters
            cur_meta_step_size = frac_done * meta_step_size_final + (1 - frac_done) * meta_step_size
            
            reptile.train_step(train_set, model.input_ph, model.label_ph, model.minimize_op,
                               num_classes=num_classes, num_shots=(train_shots or num_shots),
                               inner_batch_size=inner_batch_size, inner_iters=inner_iters,
                               replacement=replacement,
                               meta_step_size=cur_meta_step_size, meta_batch_size=meta_batch_size)

            
            if i % eval_interval == 0:
                accuracies = []
                for dataset, writer in [(train_set, train_writer), (test_set, test_writer)]:
                    accuracy = reptile.evaluate(dataset, model.input_ph, model.label_ph,
                                               model.minimize_op, model.predictions,
                                               num_classes=num_classes, num_shots=num_shots,
                                               inner_batch_size=eval_inner_batch_size,
                                               inner_iters=eval_inner_iters, replacement=replacement)
                    summary = sess.run(merged, feed_dict={accuracy_ph: accuracy})
                    writer.add_summary(summary, i)
                    writer.flush()
                    accuracies.append(accuracy)
                log_fn('batch %d: train=%f test=%f' % (i, accuracies[0], accuracies[1]))

                if(accuracies[1] > best_test_acc_val_pairs[tr_index]):
                    best_test_acc_val_pairs[tr_index] = accuracies[1]

            if (i == (meta_iters - 1)):
                train_acc_counter += accuracies[0]
            
    
    accuracy_validation = reptile.evaluate(valid_set, model.input_ph, model.label_ph,
                                               model.minimize_op, model.predictions,
                                               num_classes=num_classes, num_shots=num_shots,
                                               inner_batch_size=eval_inner_batch_size,
                                               inner_iters=eval_inner_iters, replacement=replacement)
    valid_acc_counter += accuracy_validation
    counter += 1

    return counter, train_acc_counter, valid_acc_counter