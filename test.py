import os, json
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.resnet as resnet_models
from utils import Logger
from torchsummary import summary
from utils import create_partition_few_shot_test
import torch.nn as nn
import sys

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, n_shot, pre_trained_model_path):
    result_file = open('results/{}_shot_results.txt'.format(n_shot), 'w+')
    # build model architecture
    model = resnet_models.resnet34()
    model.summary()

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, 'prototypical_loss_test')
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # load state dict
    checkpoint = torch.load(pre_trained_model_path)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    avg_acc = 0.0
    avg_loss = 0.0
    for iteration in range(config['num_iter_test']):
        print('Iteration: ', iteration)
        print('Iteration: ', iteration, file=result_file)
        labels, support, query = create_partition_few_shot_test(config['dataset']['split_list'],
                                                                num_test_classes=config['num_test_classes'],
                                                                num_support=n_shot
                                                                )
        # print(support)
        support_data_loader = module_data.SpeechDataLoader_support(support, labels,
                            len(support),
                            config
                            )
        support = {}
        for i in range(config['num_test_classes']):
            support[i] = []
        for i, (data, target) in enumerate(support_data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            for j in range(output.size()[0]):
                support[target[j].item()].append(output[j])

        prototypes = torch.stack([torch.stack(support[key]).mean(0) for key in list(support.keys())])
        # print(prototypes)
        
        query_data_loader = module_data.SpeechDataLoader_query(query, labels,config)
        
        total_loss = 0.0
        total_metrics = torch.zeros(len(metric_fns))

        with torch.no_grad():
            for i, (data, target) in enumerate(tqdm(query_data_loader)):
                data, target = data.to(device), target.to(device)
                output = model(data)
                # computing loss, metrics on test set
                loss,acc = loss_fn(output, target, prototypes)
                batch_size = data.shape[0]
                total_loss += loss.item()*batch_size
                for i, metric in enumerate(metric_fns):
                    total_metrics[i] += acc.item()*batch_size


        n_samples = len(query_data_loader.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({met.__name__ : total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
        print(log)
        print(log, file=result_file)
        avg_loss += log['loss']
        avg_acc += log['accuracy']
    avg_loss /= config['num_iter_test']
    avg_acc /= config['num_iter_test']
    print('Average loss over {} iterations is {}.'.format(config['num_iter_test'], avg_loss))
    print('Average accuracy over {} iterations is {}.'.format(config['num_iter_test'], avg_acc))
    print('Average loss over {} iterations is {}.'.format(config['num_iter_test'], avg_loss), file=result_file)
    print('Average accuracy over {} iterations is {}.'.format(config['num_iter_test'], avg_acc), file=result_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Few shot speaker recognition')

    parser.add_argument('-s', '--shot', default=1, type=int,
                           help='n shot speaker recognition (default: 1)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: 0)')

    args = parser.parse_args()

    if args.shot:
        if args.shot==1:
            pre_trained_model_path = 'pre_trained/res34_1_shot_model_best.pth'
        elif args.shot>1 and args.shot<=5:
            pre_trained_model_path = 'pre_trained/res34_5_shot_model_best.pth'
        else:
            print('Number of shots should be from 1 to 5.')
            sys.exit()
    else:
        print('Provide number of shots argument. It should be from 1 to 5.')
        sys.exit()
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
    config = json.load(open('config.json'))
    main(config, args.shot, 'pre_trained/res34_1_shot_model_best.pth')
