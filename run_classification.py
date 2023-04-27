"""
Train a model on ISIC 2018 skin lesion dataset.
"""

import random

import tensorflow as tf

from supervised_reptile_classification.args import argument_parser, model_kwargs, train_kwargs, evaluate_kwargs
from supervised_reptile_classification.eval import evaluate
from supervised_reptile_classification.test import test_fewshot

from supervised_reptile_classification.models import MetaModelDeeper, DenseNet169
from supervised_reptile_classification.isic2018 import read_dataset, split_dataset, augment_dataset
from supervised_reptile_classification.train import train

DATA_DIR_TRAIN = '/home/ilab/Downloads/ISIC2018_Task3_Training_Input'
DATA_DIR_VALIDATION = '/home/ilab/Downloads/ISIC2018_Task3_Validation_Input'
DATA_DIR_TEST = '/home/ilab/Downloads/ISIC2018_Task3_Training_class/'   

def main():
    """
    Load data and train a model on it.
    """
    args = argument_parser().parse_args()
    random.seed(args.seed)
    print('args: ', args)
    
    test_set1 = read_dataset(DATA_DIR_TEST+'class5/test')
    test_set2 = read_dataset(DATA_DIR_TEST+'class6/test')
    test_set3 = read_dataset(DATA_DIR_TEST+'class7/test')
    test_set1 = list(test_set1)
    test_set2 = list(test_set2)
    test_set3 = list(test_set3)
    test_set = test_set1 + test_set2 + test_set3
    print('test_set length: ', len(test_set))
    print('test_set1 length: ', len(test_set1))
    print('test_set2 length: ', len(test_set2))
    print('test_set3 length: ', len(test_set3))
    

    per_hyper_acc_train = []
    per_hyper_acc_valid = []
    per_hyper_acc_test = []    

    with tf.Session() as sess:
        if not args.pretrained:
            print('Training...')
         
            lr_grid = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
            meta_iters_grid = [100, 400, 1000, 2000, 4000]

            for lr_grid_param in lr_grid:

                for meta_iters_grid_param in meta_iters_grid:
                    #print('model_kwargs(args): ', model_kwargs(args))
                    args.learning_rate = lr_grid_param
                    args.meta_iters = meta_iters_grid_param
                    print('args.meta_iters: ', args.meta_iters)

                    possible_pairs = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]

                    best_test_acc_val_pairs = [0.0]*len(possible_pairs)
                    
                    counter = 0
                    train_acc_counter = 0
                    valid_acc_counter = 0    
                    
                    for val_pair_index in range(len(possible_pairs)):
                            
                        # random sampling of the classes
                        val_pair = possible_pairs[val_pair_index]
                        val1 = val_pair[0]
                        val2 = val_pair[1]
                        print('val_pair: ', val_pair)

                        rest_train_list = list(range(0,6))
                        rest_train_list.remove(val_pair_index)
                        random.shuffle(rest_train_list)

                        data_dir1_valid = '/home/ilab/Downloads/ISIC2018_Task3_Training_class/class'+str(val1)+'/train'
                        data_dir2_valid = '/home/ilab/Downloads/ISIC2018_Task3_Training_class/class'+str(val2)+'/train'
                        valid_set1 = read_dataset(data_dir1_valid)
                        valid_set1 = list(valid_set1)
                        valid_set2 = read_dataset(data_dir2_valid)
                        valid_set2 = list(valid_set2)
                        valid_set = valid_set1 + valid_set2

                        model = MetaModelDeeper(args.classes, **model_kwargs(args))
                        counter, train_acc_counter, valid_acc_counter = train(sess, model, args.checkpoint, counter, train_acc_counter, valid_acc_counter, rest_train_list, **train_kwargs(args))
                    
                    per_hyper_acc_valid.append(valid_acc_counter/float(counter))
                    per_hyper_acc_train.append(train_acc_counter/(float(counter)*len(rest_train_list)))
                    per_hyper_acc_test.append(sum(best_test_acc_val_pairs) / float(len(best_test_acc_val_pairs)))


        
        print('Evaluating...')
        eval_kwargs = evaluate_kwargs(args)
        
        print('test set length before evaluate: ', len(test_set))
        
        print('Test accuracy: ' + str(evaluate(sess, model, test_set, **eval_kwargs)))

        # iterations for the best model 
        finetune_iterations = 200
        test_fewshot(sess, model, finetune_iterations, **eval_kwargs)


if __name__ == '__main__':
    main()
