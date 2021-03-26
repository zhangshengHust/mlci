import errno
import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import string_classes
from torch.utils.data.dataloader import default_collate
from torch.nn import init
from torch.autograd import Variable

import re
import collections
import operator
import functools

from bisect import bisect
import json
import shutil

def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)

# class Logger(object):
#     def __init__(self, output_name):
#         dirname = os.path.dirname(output_name)
#         if not os.path.exists(dirname):
#             os.mkdir(dirname)

#         self.log_file = open(output_name, 'w')
#         self.infos = {}

#     def append(self, key, val):
#         vals = self.infos.setdefault(key, [])
#         vals.append(val)

#     def log(self, extra_msg=''):
#         msgs = [extra_msg]
#         for key, vals in self.infos.iteritems():
#             msgs.append('%s %.6f' % (key, np.mean(vals)))
#         msg = '\n'.join(msgs)
#         self.log_file.write(msg + '\n')
#         self.log_file.flush()
#         self.infos = {}
#         return msg

#     def write(self, msg):
#         self.log_file.write(msg + '\n')
#         self.log_file.flush()
#         print(msg)

def params_count(model):
    count = 0
    for p in model.parameters():
        c = 1
        for i in range(p.dim()):
            c *= p.size(i)
        count += c
    return count

def trim_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    _use_shared_memory = True
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        out = None
        if 1 < batch[0].dim(): # image features
            max_num_boxes = max([x.size(0) for x in batch])
            if _use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = len(batch) * max_num_boxes * batch[0].size(-1)
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            # warning: F.pad returns Variable!
            return torch.stack([F.pad(x, (0,0,0,max_num_boxes-x.size(0))).data for x in batch], 0, out=out)
        else:
            if _use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [trim_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))

class VQAAccuracy(object):
    """
    Calculate VQAAccuracy. Find more information here_

    **Key**: ``vqa_accuracy``.

    .. _here: https://visualqa.org/evaluation.html
    """

    def __init__(self):
        super(VQAAccuracy,self).__init__()

    def _masked_unk_softmax(self, x, dim, mask_idx):
        x1 = torch.nn.functional.softmax(x, dim=dim)
        x1[:, mask_idx] = 0
        x1_sum = torch.sum(x1, dim=1, keepdim=True)
        y = x1 / x1_sum
        return y

    def calculate(self, expected, output, *args, **kwargs):
        """Calculate vqa accuracy and return it back.
        Args:
            output : score
            expected : label
        Returns:
            torch.FloatTensor: VQA Accuracy
        """
#         output = self._masked_unk_softmax(output, 1, 0) # unknow 屏蔽掉了
        output = torch.nn.functional.softmax(output, dim=1)
        output = output.argmax(dim=1)  # argmax

        one_hots = expected.new_zeros(*expected.size())
        one_hots.scatter_(1, output.view(-1, 1), 1)
        scores = one_hots * expected
        accuracy = torch.sum(scores) / expected.size(0)

        return accuracy
    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)

class LogitBinaryCrossEntropy(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, targets, scores):
        targets = torch.ceil(targets)
        loss = F.binary_cross_entropy_with_logits(scores, targets, reduction="mean")
        loss_class = F.binary_cross_entropy_with_logits(scores[:, :-50], targets[:, :-50], reduction="mean")
        loss_ocr = F.binary_cross_entropy_with_logits(scores[:, -50:], targets[:, -50:], reduction="mean")

        return loss * targets.size(0), loss_class * targets.size(0), loss_ocr * targets.size(0)


class MultiTaskLoss(nn.Module):

    def __init__(self, cls_loss, ocr_loss, class_weight):
        super().__init__()
        self.alpha1 = cls_loss
        self.alpha2 = ocr_loss
        self.class_weight = Variable(class_weight)
        
    def forward(self, targets, scores):
        
        targets = torch.ceil(targets)
        loss1 = F.binary_cross_entropy_with_logits(scores[:, :-50], targets[:, :-50], self.class_weight.cuda(), reduction="mean")
        loss2 = F.binary_cross_entropy_with_logits(scores[:, -50:], targets[:, -50:], reduction="mean")
#         num1 = scores[:, :-50].size(1)
#         num2 = scores[:, -50:].size(1)
#         loss = (self.alpha1*loss1*num1 + self.alpha2*loss2*num2)/(num1+num2)
        loss = self.alpha1*loss1 + self.alpha2*loss2
        return loss * targets.size(0)

def clip_gradients(model, max_grad_l2_norm = None, clip_norm_mode = None):
    max_grad_l2_norm = max_grad_l2_norm
    clip_norm_mode = clip_norm_mode
    if max_grad_l2_norm is not None:
        if clip_norm_mode == "all":
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_l2_norm)
        elif clip_norm_mode == "question":
            norm = nn.utils.clip_grad_norm_(
                model.module.question_embedding_models.parameters(), max_grad_l2_norm
            )
        else:
            raise NotImplementedError
            
# def lr_lambda_update(i_iter, use_warmup = True, warmup_iterations=0, warmup_factor=0.2, lr_steps = 14000, lr_ratio=0.0001 ,**cfg):
#     if (use_warmup is True and i_iter <= warmup_iterations):
#         alpha = float(i_iter) / float( warmup_iterations)
#         return warmup_factor * (1.0 - alpha) + alpha
#     else:
#         idx = bisect(lr_steps, i_iter)
#         return pow(lr_ratio, idx)

def lr_lambda_update(epoch_, use_warmup = True, **cfg):
    descent_epoch = 14
    if (use_warmup is True and epoch_ <= 4):
        return epoch_/4 
    elif (use_warmup is True and epoch_ >= descent_epoch):
        return 1/(((epoch_ - descent_epoch + 2)//2)*4)
    else:
        return 1

def save_model(path, model, epoch, optimizer=None):
    model_dict = {
            'epoch': epoch,
            'model_state': model.state_dict()
        }
    if optimizer is not None:
        model_dict['optimizer_state'] = optimizer.state_dict()

    torch.save(model_dict, path) 

class logger(object):
    
    def __init__(self, log_dir):
        super(logger, self).__init__()
        dirname = os.path.dirname(log_dir)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        # copy run.py basemodel.py attention.py dataset.py embedding.py fc.py train.py utils.py layers.py fusion.py
        if os.path.exists(dirname+"/code"):
            shutil.rmtree(dirname+"/code")
        os.mkdir(dirname+"/code")
        for file in os.listdir("."):
            if file[-3:]==".py":
                shutil.copyfile("./"+file,dirname+"/code/"+file)
        
        self.dir = log_dir
        self.log = {}
    
    def save_log(self):
        with open(self.dir,'w',encoding='utf-8') as json_file:
            json.dump(self.log,json_file,ensure_ascii=False)
    
    def add(self, key, value):
        if key in self.log:
            print("Exist!")
        self.log[key] = value
        
    def __call__(self):
        print(self.log)

class result_statistic(object):
    
    def __init__(self, total_sample):
        super(result_statistic,self).__init__()
        self.answer_source = {
            "ocr": {"total":0,"predict_correct":0, "select_correct": 0}, 
            "classifier": {"total":0,"predict_correct":0, "select_correct": 0},
        }
        
        self.eror_sample = []
        self.score = 0
        self.total_sample = total_sample
    
    def add_ocr(self, correct=True):
        self.answer_source["ocr"]["total"] += 1
        if correct:
            self.answer_source["ocr"]["predict_correct"] += 1
    
    def add_classi(self, correct=True):
        self.answer_source["classifier"]["total"] += 1
        if correct:
            self.answer_source["classifier"]["predict_correct"] += 1
    
    def __call__(self):
        msg = "Statistic Result:\n"
        msg += "ocr: %d, correct: %d, ratio：%.2f \n" % (
            self.answer_source["ocr"]["total"],
            self.answer_source["ocr"]["predict_correct"],
            self.answer_source["ocr"]["predict_correct"]/self.answer_source["ocr"]["total"]*100
        )
        msg += "classifier: %d, correct: %d, ratio：%.2f \n" % (
            self.answer_source["classifier"]["total"],
            self.answer_source["classifier"]["predict_correct"],
            self.answer_source["classifier"]["predict_correct"]/self.answer_source["classifier"]["total"]*100
        )
        msg += "accuracy %.2f"%(self.score/self.total_sample*100)
        print(msg)

    def validation_answer(self, answer, question_id, dataset, predict_ans_source ,save=False):
        label, answer_source = dataset.get_answer_by_qId(question_id)
        if label==None:
            return False
        if answer in label:
            self.score += min(label[answer]/3, 1)
            return True, answer_source.get(answer, None)
        else:
            self.eror_sample.append({
                "question_id" : question_id,
                "prediction"  : answer,
                "target" : label,
                "true_answer_source": answer_source,
                "predict_answer_source": predict_ans_source
            })
            return False

def compute_result(name, model, model_dir, eval_loader, input_dir, gpu = True):
    print("compute %s result !"%name)
    model_data = torch.load(model_dir)
    model.load_state_dict(model_data.get('model_state', model_data))
    model.train(False)
#     model = model.module
    results = []
    results_sta = []
    score=0
    accuracy = VQAAccuracy()  # 计算精度
    
    rstatis = result_statistic(len(eval_loader.dataset))
    
    for i, sample in enumerate(eval_loader):
        input_ids = Variable(sample["input_ids"])
        token_type_ids = Variable(sample["token_type_ids"])
        attention_mask = Variable(sample["attention_mask"])
        img = Variable(sample["img_feature"])
        context = Variable(sample["context_feature"])
        labels = sample.get('answer',None)
        labels = Variable(sample['answer']) if labels is not None else None
        question_id = sample["question_id"]
        bbox = Variable(sample['bbox'])
        ocrbbox = Variable(sample['ocrbbox'])
        batch_size = img.size(0)
        if gpu:
            input_ids = input_ids.cuda()   
            token_type_ids = token_type_ids.cuda()
            attention_mask = attention_mask.cuda()
            img = img.cuda()
            context = context.cuda()
            ocrbbox = ocrbbox.cuda()
            labels = labels.cuda() if labels is not None else None
            bbox = bbox.cuda()
        
        prediction = model(img, bbox, input_ids, token_type_ids, attention_mask, context, ocrbbox)    
        if labels is not None:
            batch_accuracy = accuracy(labels.data, prediction)
            score += batch_accuracy * batch_size
            
        for j in range(prediction.size(0)):
            pred = prediction[j]
            q_id = question_id[j]
            tokens = eval_loader.dataset.get_tokens_by_qId(q_id)
            pred[0] = float("-Inf")
            pred[eval_loader.dataset.answer_process.length+len(tokens):] = float("-Inf")
            probablity = F.softmax(pred,dim=0)
            index = torch.max(probablity, dim=0)[1].cpu().data.numpy().tolist()

            if eval_loader.dataset.answer_process.length > index:
                
                results.append({
                    'question_id': q_id.cpu().numpy().tolist(), 
                    'answer': eval_loader.dataset.answer_process.index2answer[index]}
                )
                ## 统计结果
                if name=="val":       
                    results_sta.append({
                        'question_id': q_id.cpu().numpy().tolist(), 
                        'predict answer' : eval_loader.dataset.answer_process.index2answer[index],
                        'answer source' : 'answer_list',
                        'label' : labels.cpu().numpy()[j].tolist(),
                    })

                    # 验证并统计结果
                    rstatis.add_classi(
                        rstatis.validation_answer(results[-1]['answer'], results[-1]['question_id'], eval_loader.dataset, 1)
                    )
            
            else:
                index = index - eval_loader.dataset.answer_process.length
                token = tokens[index]
                results.append({
                    'question_id': 
                    q_id.cpu().numpy().tolist(), 
                    'answer':token}
                )
                ## 统计结果
                if name=="val": 
                    results_sta.append({
                        'question_id': q_id.cpu().numpy().tolist(), 
                        'predict answer' :  token,
                        'answer source' :  'ocr',
                        'label' : labels.cpu().numpy()[j].tolist(),
                    })

                    # 验证并统计结果
                    rstatis.add_ocr(
                        rstatis.validation_answer(results[-1]['answer'], results[-1]['question_id'], eval_loader.dataset, 2)
                    )
        if i%10 == 0 :
            msg = "[%d/%d]"%(i,len(eval_loader))
            print(msg)
    if labels is not None:
        score = score / len(eval_loader.dataset)
        print('%s score: %.2f' % (name, 100 * score))
    
    # 输出结果
    if name=="val": 
        rstatis()   
    dir_epoch = os.path.join(input_dir)
    name_json = '%s_results.json'%(name)
    sta_json = '%s_sta_results.json'%(name)
    
    os.system('mkdir -p ' + dir_epoch)
    with open(os.path.join(dir_epoch, name_json), 'w') as handle:
        json.dump(results, handle)
    with open(os.path.join(dir_epoch, sta_json), 'w') as handle:
        json.dump(results_sta, handle)

    print("computing result compelet!")

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias