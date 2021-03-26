import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

import numpy as np
from trainer import train
from dataset import Dictionary, TextVQA
from basemodel import build_model
# from train import train
import utils

# from evaluate import compute_val_result
import yaml

config = yaml.load(open('./options/lorra.yml','rb'),Loader=yaml.FullLoader)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='options/default.yaml')
    parser.add_argument('--is_train', type=bool, default=False)
    parser.add_argument('--eval_name', type=str, default="val")
    args = parser.parse_args()
    return args

def load_model_data(config, is_train = True, eval_name="val"):
    # data load
    dictionary = Dictionary()
    embedding_weight = dictionary.create_glove_embedding_init(pre=True, pre_dir='../data/vocabs/embedding_weight.npy')
    if is_train:
        train_dset = TextVQA('train', dictionary)
        eval_dset = TextVQA('val', dictionary)
        test_dset = None
        if eval_name == "test":
            test_dset = TextVQA('test', dictionary)
        model = build_model(train_dset, config['model_attributes'])
        return model, train_dset, eval_dset, embedding_weight, test_dset
    else:
        eval_dset = TextVQA(eval_name, dictionary)
        model = build_model(eval_dset, config['model_attributes'])
        return model, eval_dset

def run(config, is_train, eval_name):
    torch.manual_seed(config['training_parameters']['seed'])
    args.gpu = config['training_parameters']['gpu']
    output = config['logs']['dir_logs']
    batch_size = config['training_parameters']['batch_size']
    if args.gpu:
        torch.cuda.manual_seed(config['training_parameters']['seed'])
        torch.backends.cudnn.benchmark = True
    
    if is_train:
        '''
        eval_name 为 test 时会同时加载test 数据集 
        '''
        print("training . . .")
        model, train_dset, eval_dset, embedding_weight, test_dset = load_model_data(config, is_train = is_train, eval_name = eval_name)
    else:
        print("testing . . .")
        model, eval_dset = load_model_data(config, is_train = is_train, eval_name = eval_name)
        if args.gpu:
#             model = model.cuda()
            model = nn.DataParallel(model).cuda()
        model_dir = os.path.join(output, "model_epoch16.pth")
        eval_loader  = DataLoader(eval_dset, batch_size, shuffle=False, num_workers = config['training_parameters']['num_workers'], collate_fn=utils.trim_collate)
        utils.compute_result(eval_name, model, model_dir , eval_loader, output)
        return

    logger = utils.logger(os.path.join(output, 'log.json'))
    model_size = utils.params_count(model)

    print("nParams:",model_size)
    
    logger.add("model size(Params)", model_size)
    logger.add("train set", len(train_dset))
    logger.add("val set", len(eval_dset))
    
    with open(output + "config.yaml", "w") as yaml_file:
        yaml.dump(config, yaml_file)
    
#     model.embedding.init_embedding(embedding_weight)
    
    if args.gpu:
#         model = model.cuda()
        model = nn.DataParallel(model).cuda()
    
    print("sucees to create model.")
#     use_vg = config['data']['use_vg']
    evaluation = True if eval_name=="val" else False #config['data']['evaluation']

    if evaluation:
        print("train with train dataset")
        eval_loader  = DataLoader(
            eval_dset, 
            batch_size, 
            shuffle=False, 
            num_workers = config['training_parameters']['num_workers'], 
            collate_fn=utils.trim_collate
        )
        train_loader = DataLoader(
            train_dset, 
            batch_size, 
            shuffle=True, 
            num_workers = config['training_parameters']['num_workers'], 
            collate_fn=utils.trim_collate
        )
    else:
        print("train with train and val dataset")
        eval_loader  = None
        train_dset = ConcatDataset([train_dset, eval_dset])
        train_loader = DataLoader(
            train_dset, 
            batch_size, 
            shuffle=True, 
            num_workers=config['training_parameters']['num_workers'],
            collate_fn=utils.trim_collate
        )

#     model_data = torch.load(output+'model_epoch8.pth')
#     model.load_state_dict(model_data.get('model_state', model_data))   
#     print("success to load model!")
    
    # 初始化优化器
#     ignored_params = list(map(id, model.module.bert.parameters()))
#     base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
#     optim = torch.optim.Adamax([
#         {'params': base_params},
#         {'params': model.module.bert.parameters(), 'lr': 1e-6}  #FC层使用较大的学习率
#         ],
#         lr = 0.0015
#     )
    
    optim = torch.optim.Adamax(
        filter(lambda p:p.requires_grad, model.parameters()),
        lr = 0.0015
    )
    
#     optim = torch.optim.Adam(
#         filter(lambda p:p.requires_grad, model.parameters()),
#         lr=0.00015,
#         betas = (0.9, 0.98),
#         eps = 1e-9
# #         weight_decay=0.001
#     )
    
    train(model, train_loader, eval_loader, logger, optim, output, **config['training_parameters'])
    
    if eval_name=="val":
        model_dir = os.path.join(output, "model_best.pth")
        utils.compute_result(eval_name, model, model_dir ,eval_loader, output)
    else:      # test
        model_dir = os.path.join(output, "model_epoch5.pth")
        test_loader = DataLoader(
            test_dset, 
            batch_size, 
            shuffle=False, 
            num_workers=config['training_parameters']['num_workers'], 
            collate_fn=utils.trim_collate
        )
        utils.compute_result(eval_name, model, model_dir, test_loader, output)

if __name__ == '__main__':
    args = parse_args()
    
    with open(args.config, 'r') as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
#         print(config)
    run(config, args.is_train, args.eval_name)