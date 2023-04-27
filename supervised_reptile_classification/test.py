"""
Helpers for evaluating models.
"""

from .reptile import Reptile
from .variables import weight_decay


import os
import time

import tensorflow as tf

import random
from supervised_reptile_classification.isic2018 import read_dataset, split_dataset, augment_dataset
from supervised_reptile_classification.args import argument_parser, model_kwargs, train_kwargs, evaluate_kwargs


# pylint: disable=R0913,R0914
def test_fewshot(sess,
             model,
             finetune_iters,
             num_classes=5,
             num_shots=5,
             eval_inner_batch_size=5,
             eval_inner_iters=50,
             replacement=False,
             num_samples=10000,
             transductive=False,
             weight_decay_rate=1,
             reptile_fn=Reptile):
    
    """
    Evaluate a model on a dataset.
    """

    args = argument_parser().parse_args()
    
    reptile = reptile_fn(sess,
                         transductive=transductive,
                         pre_step_op=weight_decay(weight_decay_rate))
    
    class_pairs = [(5, 6), (6, 7), (5, 7)]

    for pair in class_pairs:
        
        tf.train.Saver().restore(sess, tf.train.latest_checkpoint(args.checkpoint))
        
        class1 = pair[0]
        class2 = pair[1]

        data_dir1_train = '/home/ilab/Downloads/ISIC2018_Task3_Training_class/class'+str(class1)+'/train'
        data_dir2_train = '/home/ilab/Downloads/ISIC2018_Task3_Training_class/class'+str(class2)+'/train'
        train_set1 = read_dataset(data_dir1_train)
        train_set1 = list((train_set1))
        train_set2 = read_dataset(data_dir2_train)
        train_set2 = list((train_set2))
        train_set = train_set1 + train_set2

        data_dir1_test = '/home/ilab/Downloads/ISIC2018_Task3_Training_class/class'+str(class1)+'/test'
        data_dir2_test = '/home/ilab/Downloads/ISIC2018_Task3_Training_class/class'+str(class2)+'/test'
        test_set1 = read_dataset(data_dir1_test)
        test_set1 = list(test_set1)
        test_set2 = read_dataset(data_dir2_test)
        test_set2 = list(test_set2)
        test_set = test_set1 + test_set2

        num_classes = 2
        
        # frac_done = i / meta_iters
        # cur_meta_step_size = frac_done * meta_step_size_final + (1 - frac_done) * meta_step_size
        correct_preds, total_entries = reptile.finetune(train_set, test_set, finetune_iters, model.input_ph, model.label_ph, model.minimize_op,
                           model.predictions, num_classes=num_classes, num_shots=num_shots,
                           inner_batch_size=eval_inner_batch_size,
                           replacement=replacement)

        print('correct_preds: ', correct_preds, ', total_entries: ', total_entries)
        print('Accuracy for pair (', class1, ', ', class2, ') is: ', float(correct_preds)/total_entries)

    