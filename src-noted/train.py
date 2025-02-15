#整体的训练程序

import numpy as np
from astropy.io import fits

import torch
from torch import optim, nn
from torch.utils.data import SubsetRandomSampler, DataLoader, Subset

from .networks import ForkCNN, CaliNN

import sys,time,os
# sys.path.append(os.path.abspath("../configs"))
# from configs import config
import config   #config用以读取配置文件
class Train(object):
    
    def __init__(self):
        
        # Device Options   
        self.workers = config.train['workers']
        self.device = torch.device(config.train['device'])
        self.batch_size = config.train['batch_size']
        self.features = config.train['feature_number']
        self.nGPUs = config.train['gpu_number']
        
        
    def _set_data(self, train_ds):
        '''
        Spilt the dataset into training and validation, and build dataloader.
        '''
        
        valid_split = config.train['validation_split']

        size = len(train_ds)
        indices = list(range(size))
        split = int(np.floor(valid_split * size))  #算出训练集和测试集中分别有多少数据(向下取证)

        train_indices, valid_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)  #在总的数据集中随机取样，生成训练集和测试集
        valid_sampler = SubsetRandomSampler(valid_indices)

        self.train_dl = DataLoader(train_ds,                  #按torch库的规则加载数据
                              batch_size=self.batch_size, 
                              num_workers=self.workers,
                              sampler=train_sampler)
        self.valid_dl = DataLoader(train_ds, 
                              batch_size=self.batch_size, 
                              num_workers=self.workers,
                              sampler=valid_sampler)
        print("Train_dl: {} Validation_dl: {}".format(len(self.train_dl), len(self.valid_dl)))
        
    def load_model(self,path=None,strict=True):  #加载模型
        
        model = ForkCNN(self.features, self.batch_size, self.nGPUs) #这个完整的卸载networks里
        model.to(self.device)
        if self.nGPUs > 1:
            model = nn.DataParallel(model, device_ids=range(self.nGPUs))   #多个GPU情况下平行处理数据
        
        if path != None:
            model.load_state_dict(torch.load(path), strict=strict)
        
        return model
    
    def run(self, dataset, show_log=True):
        
        # set data loader here
        self._set_data(dataset)
        
        # self.model = ForkCNN(self.features, self.batch_size, self.nGPUs)
        # if self.nGPUs > 1:
        #     self.model = nn.DataParallel(self.model, device_ids=range(self.nGPUs))
        # self.model.to(self.device)
        
        self.model = self.load_model()
        
        self.criterion = nn.MSELoss()   #MSE损失函数算法，计算得到标签值与真实值之间差的平均的平方和 优缺点参考https://www.python100.com/html/69ARI0Q4M7I0.html
        self.optimizer = optim.SGD(self.model.parameters(),    #优化算法：随机梯度下降法 缺点与优化参考https://blog.csdn.net/tcn760/article/details/123965374/
                                   lr=config.train['initial_learning_rate'], 
                                   momentum=config.train['momentum'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True)
        
        print('Begin training ...')
        
        # Loop the training and validation processes 循环达到训练效果
        train_losses = []
        valid_losses = []
        for epoch in range(config.train['epoch_number']):
            train_loss = self._trainFunc(epoch,show_log=show_log)   #trainFunc和validFunc的定义在下面
            valid_loss = self._validFunc(epoch,show_log=show_log)
            scheduler.step(train_loss)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            
            if config.train['save_model']:   #读取保存模型的条件-收敛到什么程度
                if not os.path.exists(config.train['model_path']):
                    os.makedirs(config.train['model_path'])
                torch.save(self.model.state_dict(), 
                           config.train['model_path']+config.train['model_name']+str(epoch))
                
        if config.train['save_model']:    #epoch循环结束，决定要不要保存最后一次的模型
            hdu0 = fits.PrimaryHDU(train_losses)
            hdu1 = fits.ImageHDU(valid_losses)
            hdul = fits.HDUList([hdu0, hdu1])
            hdul.writeto(config.train['model_path']+'/training_loss.fits',overwrite=True)
                
        print('Finish training !')       #结束训练力
        return train_losses, valid_losses
        

    def _trainFunc(self,epoch,show_log=True):
        self.model.train()
        losses = []
        epoch_start = time.time()
        for i, batch in enumerate(self.train_dl):
            inputs1, inputs2, labels = batch['gal_image'].float().to(self.device), \
                                       batch['psf_image'].float().to(self.device), \
                                       batch['label'].float().view(-1,self.features).to(self.device)

            self.optimizer.zero_grad()        #在进行新的梯度计算之前，先把上一次的结果清零     
            outputs = self.model.forward(inputs1, inputs2)
            loss = self.criterion(outputs, labels) 
            losses.append(loss.item())        
            loss = torch.sqrt(loss)           
            loss.backward()                   #将损失反向传播：计算loss对所有变量的梯度并将每个epoch的loss进行累积 参考https://blog.csdn.net/weixin_43863869/article/details/128120719
            self.optimizer.step()             #只有训练集需要反向传播，测试集并不需要     

        epoch_loss = np.sqrt(sum(losses) / len(losses))
        epoch_time = time.time() - epoch_start
        if show_log:
            print("[TRAIN] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                          epoch_time // 60, 
                                                                          epoch_time % 60))
        return epoch_loss

    def _validFunc(self,epoch,show_log=True):
        self.model.eval()
        losses = []
        epoch_start = time.time()
        for i, batch in enumerate(self.valid_dl):
            inputs1, inputs2, labels = batch['gal_image'].float().to(self.device), \
                                       batch['psf_image'].float().to(self.device), \
                                       batch['label'].float().view(-1,self.features).to(self.device)

            outputs = self.model.forward(inputs1, inputs2)
            loss = self.criterion(outputs, labels)
            losses.append(loss.item())

        epoch_loss = np.sqrt(sum(losses) / len(losses))
        epoch_time = time.time() - epoch_start
        if show_log:
            print("[VALID] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                        epoch_time // 60, 
                                                                        epoch_time % 60))
        return epoch_loss
    

    def _predictFunc(self,test_dl,MODEL,criterion=nn.MSELoss()):  #这个函数用以预测每张图片的参数

        MODEL.eval()
        losses=[]
        for i, batch in enumerate(test_dl):          #enumerate为一个索引序列用以遍历对象
            inputs1, inputs2 = batch['gal_image'].float().to(self.device), \
                                       batch['psf_image'].float().to(self.device)
            outputs = MODEL.forward(inputs1, inputs2)
            labels_true_batch = batch['label'].float().view(-1,self.features).to(self.device)
            loss = criterion(outputs, labels_true_batch)
            losses.append(loss.item())
            if i == 0:                                                  #把每一个id对应的预测的值都连在一起并输出为combined_
                ids = batch['id'].numpy()
                labels = outputs.detach().cpu().numpy()
                labels_true = labels_true_batch.cpu()
                snr = batch['snr'].numpy()
            else:
                ids = np.concatenate((ids, batch['id'].numpy()))
                labels = np.vstack((labels, outputs.detach().cpu().numpy()))
                labels_true = np.vstack((labels_true, labels_true_batch.cpu()))  
                snr = np.concatenate((snr, batch['snr'].numpy()))

        combined_pred = np.column_stack((ids, labels))
        combined_true = np.column_stack((ids, labels_true))
        combined_snr = np.column_stack((ids, snr))

        epoch_loss = np.sqrt(sum(losses) / len(losses))
        return combined_pred, combined_true, combined_snr, epoch_loss
    
    
    
    
    
class MSBLoss(nn.Module):
    def __init__(self):
        super(MSBLoss, self).__init__()
        
    def forward(self,x,y):
        
        if torch.std(y) != 0:      #输出每一行张量的标准差 为什么这个要等于0？
            print('Waring!')
            print(y)
        return (torch.mean(x) - torch.mean(y))**2
    

#这个class把前面写的几个函数都装进去了？？为啥这个还要重写一遍？   
class NNTrain(object):
    
    def __init__(self):
        
        # Device Options   
        # config用于读取配置文件 但在总程序里不是已经读过了么？？
        self.workers = config.train['workers']
        self.device = torch.device(config.train['device'])
        self.batch_size = config.train['batch_size']
        self.nGPUs = config.train['gpu_number']
        
        
    def _set_data(self, train_ds):
        '''
        Spilt the dataset into training and validation, and build dataloader.
        '''
        
        valid_split = config.train['validation_split']

        size = len(train_ds)
        indices = list(range(size))
        split = int(np.floor(valid_split * size))

        train_indices, valid_indices = indices[split:], indices[:split]

        self.train_dl = DataLoader(Subset(train_ds, train_indices), 
                              batch_size=self.batch_size, 
                              num_workers=self.workers)
        self.valid_dl = DataLoader(Subset(train_ds, valid_indices), 
                              batch_size=self.batch_size, 
                              num_workers=self.workers)
        print("Train_dl: {} Validation_dl: {}".format(len(self.train_dl), len(self.valid_dl)))
        
    def load_model(self,path=None,strict=True):
        
        model = CaliNN()
        model.to(self.device)
        if self.nGPUs > 1:
            model = nn.DataParallel(model, device_ids=range(self.nGPUs))
        
        if path != None:
            model.load_state_dict(torch.load(path), strict=strict)
        
        return model
    
    def run(self, dataset, show_log=True):
        
        # set data loader here
        self._set_data(dataset)
        
        self.model = self.load_model()
        
        self.criterion = MSBLoss()
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=config.train['initial_learning_rate'],
                                    betas=config.train['adam_betas'])

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True)
        
        print('Begin training ...')
        
        # Loop the training and validation processes
        train_losses = []
        valid_losses = []
        for epoch in range(config.train['epoch_number']):
            train_loss = self._trainFunc(epoch,show_log=show_log)
            valid_loss = self._validFunc(epoch,show_log=show_log)
            scheduler.step(valid_loss)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            
            if config.train['save_model']:
                if not os.path.exists(config.train['model_path']):
                    os.makedirs(config.train['model_path'])
                torch.save(self.model.state_dict(), 
                           config.train['model_path']+config.train['model_name']+str(epoch))
                
        if config.train['save_model']:
            hdu0 = fits.PrimaryHDU(train_losses)
            hdu1 = fits.ImageHDU(valid_losses)
            hdul = fits.HDUList([hdu0, hdu1])
            hdul.writeto(config.train['model_path']+'/training_loss.fits',overwrite=True)
                
        print('Finish training !')
        return train_losses, valid_losses
    
    
    def _trainFunc(self,epoch,show_log=True):
        self.model.train()
        losses = []
        epoch_start = time.time()
        for i, batch in enumerate(self.train_dl):
            inputs, labels = batch['input'].float().to(self.device), \
                             batch['label'].float().to(self.device)

            self.optimizer.zero_grad()             
            outputs = self.model.forward(inputs)
            loss = self.criterion(outputs, labels) 
            losses.append(loss.item())        
            loss = torch.sqrt(loss)           
            loss.backward()                   
            self.optimizer.step()                  

        epoch_loss = np.sqrt(sum(losses) / len(losses))
        epoch_time = time.time() - epoch_start
        if show_log:
            print("[TRAIN] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                          epoch_time // 60, 
                                                                          epoch_time % 60))
        return epoch_loss

    
    def _validFunc(self,epoch,show_log=True):
        self.model.eval()
        losses = []
        epoch_start = time.time()
        for i, batch in enumerate(self.valid_dl):
            inputs, labels = batch['input'].float().to(self.device), \
                             batch['label'].float().to(self.device)

            outputs = self.model.forward(inputs)
            loss = self.criterion(outputs, labels)
            losses.append(loss.item())

        epoch_loss = np.sqrt(sum(losses) / len(losses))
        epoch_time = time.time() - epoch_start
        if show_log:
            print("[VALID] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                        epoch_time // 60, 
                                                                        epoch_time % 60))
        return epoch_loss
    

    def _predictFunc(self,test_dl,MODEL,criterion=MSBLoss()):

        MODEL.eval()
        losses=[]
        for i, batch in enumerate(test_dl):
            inputs, labels = batch['input'].float().to(self.device), \
                             batch['label'].float().to(self.device)
            outputs = MODEL.forward(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            if i == 0:
                ids = i
                res = np.mean(outputs.detach().cpu().numpy(),axis=0)
                labels_true = np.mean(labels.cpu().numpy(),axis=0)
            else:
                ids = np.append(ids, i)
                res = np.vstack((res, np.mean(outputs.detach().cpu().numpy(),axis=0)))
                labels_true = np.vstack((labels_true, np.mean(labels.cpu().numpy(),axis=0)))  
        combined_pred = np.column_stack((ids, res))
        combined_true = np.column_stack((ids, labels_true))

        epoch_loss = np.sqrt(sum(losses) / len(losses))
        return combined_pred, combined_true, epoch_loss
    

