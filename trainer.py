from torch.autograd import Variable
from utils import LogitBinaryCrossEntropy, VQAAccuracy, clip_gradients, lr_lambda_update, MultiTaskLoss
# from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LambdaLR
import torch
import os
import utils

def train(model, train_loader, val_loader, logger, optim, output, gpu = True, **train_config):
    # 定义优化器，学习率
    # torch.autograd.set_detect_anomaly(True)
    accuracy = VQAAccuracy()          # 计算精度
    
#     class_frequence = torch.zeros_like(train_loader.dataset[0]['answer'][:-50])
#     for sample in train_loader.dataset:
#         class_frequence +=  torch.ceil(sample['answer'][:-50])
#     class_weight = torch.exp(-class_frequence/class_frequence.sum())
#     print(class_weight)
    
    lbce =  LogitBinaryCrossEntropy()
    
    # 设置优化函数进行学习率
    scheduler_func = lambda x: lr_lambda_update(x, **train_config)
    lr_scheduler = LambdaLR(optim, lr_lambda = scheduler_func)
    
#     lr_scheduler = ExponentialLR(optim, 0.5**(1/50000))
    
    iteration = 0
    best_val_accuracy = 0
    best_epoch = 0
    patient = 0
    saving_epoch = 4
    
    log_train_accuracy = {}
    log_train_loss = {}
    if val_loader is not None:
        log_val_loss = {}
        log_val_accuracy = {}
    
    for epoch in range(1,train_config["epoch"]+1):
#     while iteration < train_config["max_iterations"]:
        model.train()
        log_msg = "Epoch %d of Train:"%(
            epoch
        )
        print(log_msg)
        total_accuracy = 0
        total_loss = 0
        for i, sample in enumerate(train_loader):
            iteration += 1
            input_ids = Variable(sample["input_ids"])
            token_type_ids = Variable(sample["token_type_ids"])
            attention_mask = Variable(sample["attention_mask"])
            
            img = Variable(sample["img_feature"])
#             context = Variable(torch.zeros_like(sample["context_feature"]))
            context = Variable(sample["context_feature"])
            labels = Variable(sample['answer'])
            bbox = Variable(sample['bbox'])
            ocrbbox = Variable(sample['ocrbbox'])
            
            if gpu:
                input_ids = input_ids.cuda()   
                token_type_ids = token_type_ids.cuda()
                attention_mask = attention_mask.cuda()
                img = img.cuda()
                context = context.cuda()
                labels = labels.cuda()
                bbox = bbox.cuda()
                ocrbbox = ocrbbox.cuda()
            
            batch_size = img.size(0)
            prediction = model(img, bbox, input_ids, token_type_ids, attention_mask, context, ocrbbox)
            
            loss, lc, lo = lbce(labels, prediction)
            if epoch<=13:
                loss.backward()
            elif epoch%2 ==0 :
                lc.backward()
            else :
                lo.backward()
#             if (i+1)%2==0:
            lr_scheduler.step(epoch)
            clip_gradients(model, train_config["max_grad_l2_norm"], train_config["clip_norm_mode"])
            optim.step()
            optim.zero_grad()
            # 统计精度
            batch_accuracy = accuracy(labels.data,prediction.data)
            total_accuracy += batch_accuracy * batch_size
            # 统计loss
            total_loss += loss.data
            if (i+1)%10 == 0:
                log_msg = "[%d/%d/%d] iter:%d accuracy:%2.2f loss:%.5f lr: %f"%(
                    epoch, 
                    len(train_loader),
                    i,
                    iteration, 
                    batch_accuracy*100,
                    loss.data,
                    optim.param_groups[0]['lr']
                )
                print(log_msg)
        if val_loader is not None:
            log_msg = "Epoch %d of Valuation:"%(
                epoch
            )
            print(log_msg)
            val_accuracy, val_loss = evaluate(model, val_loader, logger, gpu, **train_config)
        print("Result")
        
        log_msg = "Train accuracy:%2.2f, Train loss: %.5f"%(
            total_accuracy/len(train_loader.dataset)*100, 
            total_loss/len(train_loader.dataset)
        )
        if val_loader is not None:
            log_msg += ", Val accuracy:%2.2f, Val loss: %.5f" % (val_accuracy*100, val_loss)
        print(log_msg)
        
        log_train_accuracy["epoch "+str(epoch)] = total_accuracy.cpu().numpy().tolist()/len(train_loader.dataset)*100
        log_train_loss["epoch "+str(epoch)] = total_loss.cpu().numpy().tolist()/len(train_loader.dataset)
        
        if val_loader is not None:
            log_val_accuracy["epoch "+str(epoch)] = val_accuracy.cpu().numpy().tolist()*100
            log_val_loss["epoch "+str(epoch)] = val_loss.cpu().numpy().tolist()
        
        if (val_loader is not None and val_accuracy > best_val_accuracy) or (val_loader is None and epoch >= saving_epoch):
            model_name = 'model_%s.pth'%('epoch'+str(epoch) if val_loader is None else 'best')
            model_path = os.path.join(output, model_name)
            utils.save_model(model_path, model, epoch, optim)
            if val_loader is not None:
                best_val_accuracy = val_accuracy
                best_epoch = epoch
                patient = 0
        elif val_loader is not None:
            patient += 1
            if patient >= 15:
                print("Patient %d early stop!!"%patient)
                break
        print("Patient %d"%patient)
    
    log_msg = "best val accuracy : %2.2f at %d epoch. "%( best_val_accuracy*100, best_epoch)
    print(log_msg)
    logger.add("train accuracy", log_train_accuracy)  
    logger.add("train loss", log_train_loss)
    if val_loader is not None:
        logger.add("best val accuracy", best_val_accuracy.cpu().numpy().tolist())
        logger.add("best_epoch", best_epoch)
        logger.add("val loss", log_val_loss)
        logger.add("val accuracy", log_val_accuracy)  
    logger.save_log()

def evaluate(model, val_loader, logger, gpu = True, **train_config):
    model.eval()
    total_accuracy = 0
    total_loss = 0
    accuracy = VQAAccuracy() # 计算精度
    lbce = LogitBinaryCrossEntropy()
    
    for i, sample in enumerate(val_loader):
        input_ids = Variable(sample["input_ids"])
        token_type_ids = Variable(sample["token_type_ids"])
        attention_mask = Variable(sample["attention_mask"])
        img = Variable(sample["img_feature"])
        context = Variable(sample["context_feature"])
        labels = Variable(sample['answer'])
        bbox = Variable(sample['bbox'])
        ocrbbox = Variable(sample['ocrbbox'])
        batch_size = img.size(0)
        if gpu:
            input_ids = input_ids.cuda()   
            token_type_ids = token_type_ids.cuda()
            attention_mask = attention_mask.cuda()
            img = img.cuda()
            context = context.cuda()
            labels = labels.cuda()
            bbox = bbox.cuda()
            ocrbbox = ocrbbox.cuda()
        prediction = model(img, bbox, input_ids, token_type_ids, attention_mask, context, ocrbbox)    
        loss, lc, lo = lbce(labels, prediction)
        batch_accuracy = accuracy(labels.data, prediction.data)
        total_accuracy += batch_accuracy*batch_size
        # 统计loss
        total_loss += loss.data
        if i%5 == 0:
            log_msg = "[%d] accuracy:%.3f loss:%.3f"%(
                i, batch_accuracy*100,loss.data
            )
            print(log_msg)
        
    return total_accuracy/len(val_loader.dataset), total_loss/len(val_loader.dataset)