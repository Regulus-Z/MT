import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import time
import logging

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from data_generator import DataGenerator
from utilities import (create_folder, get_filename, create_logging,
                       calculate_confusion_matrix, calculate_accuracy, 
                       print_confusion_matrix, print_accuracy, print_accuracy_binary,auc_2,calculate_auc,print_auc，auc_2)
from models_pytorch import move_data_to_gpu, DecisionLevelMaxPooling
import config


Model = DecisionLevelMaxPooling
batch_size = 8


def evaluate(model, generator, data_type, max_iteration, cuda):
    """Evaluate
    
    Args:
      model: object.
      generator: object.
      data_type: 'train' | 'validate'.
      max_iteration: int, maximum iteration for validation
      cuda: bool.
      
    Returns:
      accuracy: float
    """
    
    # Generate function
    generate_func = generator.generate_validate(data_type=data_type,
                                                shuffle=True, 
                                                max_iteration=max_iteration)
            
    # Forward
    dict = forward(model=model, 
                   generate_func=generate_func, 
                   cuda=cuda)
    
    outputs = dict['output']    # (audios_num, classes_num)
    targets = dict['target']    # (audios_num, classes_num)
    
    predictions = np.argmax(outputs, axis=-1)   # (audios_num,)
    scores=outputs[:,-1]
    # Evaluate
    classes_num = outputs.shape[-1]
    output_log=np.log(outputs)
    loss = F.nll_loss(Variable(torch.Tensor(output_log)), Variable(torch.LongTensor(targets))).data.numpy()
    loss = float(loss)
    
    accuracy = calculate_accuracy(targets, predictions, classes_num, average='macro')
    auc=auc_2(targets, scores)
    return accuracy, loss,auc['AUC']

def forward(model, generate_func, cuda):
    """Forward data to a model.
    
    Args:
      model: object
      generate_func: generate function
      cuda: bool
      
    Returns:
      dict, keys: 'audio_name', 'output'; optional keys: 'target'
    """
    
    outputs = []
    audio_names = []
    targets = []
    
    # Evaluate on mini-batch
    for data in generate_func:
        (batch_x,batch_ch, batch_3,batch_y, batch_audio_names) = data

        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_ch = move_data_to_gpu(batch_ch, cuda)
        batch_3 = move_data_to_gpu(batch_3, cuda)
        batch_y= move_data_to_gpu(batch_y, cuda)
        # Predict
        model.eval()
        batch_output = model(batch_x,batch_ch,batch_3,batch_y)

        # Append data
        outputs.append(batch_output.data.cpu().numpy())
        audio_names.append(batch_audio_names)
        targets.append(batch_y.cpu())

    dict = {}

    outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs
    
    audio_names = np.concatenate(audio_names, axis=0)
    dict['audio_name'] = audio_names

    targets = np.concatenate(targets, axis=0)
    dict['target'] = targets
        
    return dict

def train(args):
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    workspace = args.workspace
    validate = args.validate
    iteration_max = args.iteration_max
    cuda = args.cuda

    labels = config.labels
    classes_num = len(labels)

    # Paths
    if validate:
        dev_train_csv = os.path.join(dataset_dir, 'csv_data', 'data_train.csv')
        dev_validate_csv = os.path.join(dataset_dir, 'csv_data', 'data_dev.csv')
    else:
        dev_train_csv = os.path.join(dataset_dir, 'csv_data', 'data_traindev.csv')
        dev_validate_csv = os.path.join(dataset_dir, 'csv_data', 'data_test.csv')
        
    models_dir = os.path.join(workspace, 'models', subdir)
    create_folder(models_dir)

    # Model
    model = Model(classes_num)

    if cuda:
        model.cuda()

    # Data generator
    generator = DataGenerator(dataset_dir=dataset_dir,
                              batch_size=batch_size,
                              dev_train_csv=dev_train_csv,
                              dev_validate_csv=dev_validate_csv)
    class_weight = generator.calculate_class_weight()
    class_weight = move_data_to_gpu(class_weight, cuda)
    
    # Optimizer
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.98)
    train_bgn_time = time.time()

    # Train on mini batches
    for (iteration, (batch_x,batch_ch,batch_3,batch_y, _)) in enumerate(generator.generate_train()):
        # Evaluate
        if iteration % 100 == 0:
            train_fin_time = time.time()

            (tr_acc, tr_loss,tr_auc) = evaluate(model=model,
                                         generator=generator,
                                         data_type='train',
                                         max_iteration=None,
                                         cuda=cuda)

            logging.info('tr_acc: {:.3f}, tr_loss: {:.3f},auc:{:.3f}'.format(tr_acc, tr_loss,tr_auc))

            (va_acc, va_loss,va_auc) = evaluate(model=model,
                                         generator=generator,
                                         data_type='evaluate',
                                         max_iteration=None,
                                         cuda=cuda)
                                
            logging.info('va_acc: {:.3f}, va_loss: {:.3f},auc:{:.3f}'.format(va_acc, va_loss,va_auc))

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(iteration, train_time, validate_time))

            logging.info('------------------------------------')

            train_bgn_time = time.time()

        # Save model
        if iteration % 1000 == 0 and iteration > 0:
            save_out_dict = {'iteration': iteration,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()
                             }
            save_out_path = os.path.join(
                models_dir, 'md_{}_iters.tar'.format(iteration))
            torch.save(save_out_dict, save_out_path)
            logging.info('Model saved to {}'.format(save_out_path))

        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)
        batch_ch = move_data_to_gpu(batch_ch, cuda)
        batch_3 = move_data_to_gpu(batch_3, cuda)
        # Train
        model.train()
        batch_output = model(batch_x,batch_ch,batch_3,batch_y)
        log_output=torch.log(batch_output)
        loss = F.nll_loss(log_output, batch_y, weight=class_weight)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Stop learning
        if iteration == iteration_max:
            break


def inference_validation_data(args):
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    subdir = args.subdir
    workspace = args.workspace
    validate = args.validate
    iteration_max = args.iteration_max
    filename = args.filename
    cuda = args.cuda

    labels = config.labels
    classes_num = len(labels)

    # Paths
    ############
    if validate:
        dev_train_csv = os.path.join(dataset_dir, 'csv_data', 'data_train.csv')
        dev_validate_csv = os.path.join(dataset_dir, 'csv_data', 'data_dev.csv')
    else:
        dev_train_csv = os.path.join(dataset_dir, 'csv_data', 'data_traindev.csv')
        dev_validate_csv = os.path.join(dataset_dir, 'csv_data', 'data_test.csv')

    model_path = os.path.join(workspace, 'models', subdir, 'md_{}_iters.tar'.format(iteration_max))

    # Load model
    model = Model(classes_num)
    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('param number:')
    print(param_num)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model.cuda()

    # Predict & evaluate
    # Data generator
    generator = DataGenerator(dataset_dir = dataset_dir,
                              batch_size=batch_size,
                              dev_train_csv=dev_train_csv,
                              dev_validate_csv=dev_validate_csv)

    generate_func = generator.generate_validate(data_type='evaluate',
                                                shuffle=False)

    # Inference
    dict = forward(model=model,
                   generate_func=generate_func,
                   cuda=cuda)

    outputs = dict['output']    # (audios_num, classes_num)
    targets = dict['target']    # (audios_num, classes_num)

    predictions = np.argmax(outputs, axis=-1)
    scores=outputs[:,-1]
    classes_num = outputs.shape[-1]
    
    # Evaluate
    confusion_matrix = calculate_confusion_matrix(targets, predictions, classes_num)
    #auc=calculate_auc(targets, scores, classes_num)
    auc2=auc_2(targets,scores)
    class_wise_accuracy = calculate_accuracy(targets, predictions, classes_num)
    se, sp, as_score, hs_score = calculate_accuracy(targets, predictions, classes_num, average='binary')

    # Print
    print_accuracy(class_wise_accuracy, labels)
    print_auc(auc)
    print_auc(auc2['AUC'])
    print_confusion_matrix(confusion_matrix, labels)
    #print('confusion_matrix: \n', confusion_matrix)
    print_accuracy_binary(se, sp, as_score, hs_score, labels)
    


if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser(description='Example of parser. ')
    
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--dataset_dir', type=str, default='../../../data_experiment/')
    parser.add_argument('--subdir', type=str, default='models_dev')
    parser.add_argument('--workspace', type=str, default='../../../experiment_workspace/baseline_cnn/')
    parser.add_argument('--validate', action='store_true', default=True)
    parser.add_argument('--iteration_max', type=int, default=15000)
    parser.add_argument('--cuda', action='store_true', default=False)
    '''
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True)
    parser_train.add_argument('--subdir', type=str, required=True)
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--validate', action='store_true', default=False)
    parser_train.add_argument('--iteration_max', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    
    parser_inference_validation_data = subparsers.add_parser('inference_validation_data')
    parser_inference_validation_data.add_argument('--dataset_dir', type=str, required=True)
    parser_inference_validation_data.add_argument('--subdir', type=str, required=True)
    parser_inference_validation_data.add_argument('--workspace', type=str, required=True)
    parser_inference_validation_data.add_argument('--validate', action='store_true', default=False)
    parser_inference_validation_data.add_argument('--iteration_max', type=int, required=True)
    parser_inference_validation_data.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    args.filename = get_filename(__file__)

    # Create log
    logs_dir = os.path.join(args.workspace, 'logs', args.filename)
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference_validation_data':
        inference_validation_data(args)

    else:
        raise Exception('Error argument!')
