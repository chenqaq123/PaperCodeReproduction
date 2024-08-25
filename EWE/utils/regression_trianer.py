import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset
import logging
import os 
import pickle
import numpy as np  
import scipy.io as sio

from models.conv import ConvModel
from utils.trainer import Trainer
from utils.utils import snnl

class RegTrainer(Trainer):
    def setup(self):
        args = self.args
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.default = args.default
        self.batch_size = args.batch_size
        self.ratio = args.ratio
        self.lr = args.lr
        self.epochs = args.epochs
        self.w_epochs = args.w_epochs
        self.factors = args.factors
        self.temperatures = args.temperatures
        self.threshold = args.threshold
        self.w_lr = args.w_lr
        self.t_lr = args.t_lr
        self.source = args.source
        self.target = args.target
        self.verbose = args.verbose
        self.dataset = args.dataset
        self.model_type = args.model
        self.maxiter = args.maxiter
        self.distrib = args.distrib
        self.layers = args.layers
        self.metric = args.metric
        self.shuffle = args.shuffle

        if self.default:
            if self.dataset == 'mnist':
                self.model_type = '2_conv'
                self.ratio = 1
                self.batch_size = 512
                self.epochs = 10
                self.w_epochs = 10
                self.factors = [32, 32, 32]
                self.temperatures = [1, 1, 1]
                self.metric = "cosine"
                self.threshold = 0.1
                self.t_lr = 0.1
                self.w_lr = 0.01
                self.source = 1 
                self.target = 7
                self.maxiter = 10
                self.distrib = "out"
                self.num_classes = 10
                self.channels = 1
            elif self.dataset == 'fashion':
                self.num_classes = 10
                self.channels = 3
                if self.model_type == '2_conv':
                    self.batch_size = 128
                    self.ratio = 2
                    self.epochs = 10
                    self.w_epochs = 10
                    self.factors = [32, 32, 32]
                    self.temperatures = [1, 1, 1]
                    self.t_lr = 0.1
                    self.threshold = 0.1
                    self.w_lr = 0.01
                    self.source = 8
                    self.target = 0
                    self.maxiter = 10
                    self.distrib = "out"
                    self.metric = "cosine"
                    
            else:
                raise NotImplementedError('Dataset is not implemented.')

        # 加载数据集
        if self.dataset == 'mnist' or self.dataset == 'fashion':
            with open(os.path.join("data", f"{self.dataset}.pkl"), 'rb') as f:
                mnist = pickle.load(f)
            self.x_train, self.y_train, self.x_test, self.y_test = mnist["training_images"], mnist["training_labels"], \
                                            mnist["test_images"], mnist["test_labels"]
            self.x_train = np.reshape(self.x_train / 255, [-1, 28, 28, 1])
            x_test = np.reshape(x_test / 255, [-1, 28, 28, 1])
        else:
            raise NotImplementedError('Dataset is not implemented.')
        
        # 加载模型
        if self.model_type == '2_conv':
            self.ewe_model = ConvModel(num_classes=self.num_classes, batch_size= self.batch_size, in_channels=self.channels)
            self.plain_model = ConvModel(num_classes=self.num_classes, batch_size=self.batch_size, in_channels=self.channels)
        else:
            raise NotImplementedError('Model is not implemented.')

        return super().setup()
    
    def train(self):
        height = self.x_train[0].shape[0]
        width = self.x_train[0].shape[1]

        half_batch_size = int(self.batch_size / 2)
        target_data = self.x_train[self.y_train == self.target]

        # 选择水印数据
        if self.distribution == "in":
            watermark_source_data = self.x_train[self.y_train == self.source]
        elif self.distribution == "out":
            if self.dataset == "mnist":
                w_dataset = "fashion"
                with open(os.path.join("data", f"{w_dataset}.pkl"), 'rb') as f:
                    w_data = pickle.load(f)
                x_w, y_w = w_data["training_images"], w_data["training_labels"]
            elif self.dataset == "fashion":
                w_dataset = "mnist"
                with open(os.path.join("data", f"{w_dataset}.pkl"), 'rb') as f:
                    w_data = pickle.load(f)
                x_w, y_w = w_data["training_images"], w_data["training_labels"]
            else:
                raise NotImplementedError()
            x_w = np.reshape(x_w / 255, [-1, height, width, self.channels])
            watermark_source_data = x_w[y_w == self.source]
        else:
            raise NotImplementedError("Distribution could only be either \'in\' or \'out\'.")

        # 确保水印数据和目标类数据量相同
        trigger = np.concatenate([watermark_source_data] * (target_data.shape[0] // watermark_source_data.shape[0] + 1), 0)[
                :target_data.shape[0]]
        
        w_label = np.concatenate([np.ones(half_batch_size), np.zeros(half_batch_size)], 0)
        
        # 1. 训练victim model
        x_train_tensor = torch.tensor(self.x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(self.y_train, dtype=torch.long)
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

        optimizer = optim.Adam(self.ewe_model.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=1e-5)
        for _ in range(self.epochs):
            self.ewe_model.train()
            for _, (x, y) in enumerate(train_loader):
                snnl_loss = self.ewe_model.snnl_loss(x, y, np.zeros([self.batch_size]), self.factors, self.temperatures)
                optimizer.zero_grad()
                snnl_loss.backward()
                optimizer.step()

        if self.distribution == "in":
            # TODO 默认为out，暂不实现
            raise NotImplementedError("in尚未实现")
        else:
            w_pos = [-1, -1]

        # SNNL过程
        w_num_batch = target_data.shape[0] // self.batch_size * 2
        step_list = np.zeros([w_num_batch])
        for batch in range(w_num_batch):
            current_trigger = trigger[batch * half_batch_size: (batch + 1) * half_batch_size]
            for _ in range(self.maxiter):
                while self.validate_watermark(self.ewe_model, current_trigger, self.target) > self.threshold and step_list[batch] < 50:
                    step_list[batch] += 1
                    grad = self.ewe_model.ce_gradient(np.concatenate([current_trigger, current_trigger], 0), self.target)[0]
                    current_trigger = np.clip(current_trigger - self.w_lr * np.sign(grad[:half_batch_size]), 0, 1)
                
                batch_data = np.concatenate([current_trigger, target_data[batch * half_batch_size: (batch + 1) * half_batch_size]], 0)
                grad = self.ewe_model.snnl_gradient(batch_data, self.factors, w_label)[0]
                current_trigger = np.clip(current_trigger + self.w_lr * np.sign(grad[:half_batch_size]), 0, 1)

            for _ in range(5):
                grad = self.ce_gradient(self.ewe_model, np.concatenate([current_trigger, current_trigger], 0), self.target)[0]
                current_trigger = np.clip(current_trigger - self.w_lr * np.sign(grad[:half_batch_size]), 0, 1)
            trigger[batch * half_batch_size: (batch + 1) * half_batch_size] = current_trigger
        
        trigger_label = np.zeros([self.batch_size, self.num_classes]) # batch_size * num_class
        trigger_label[:, self.target] = 1
        num_batch = self.x_train.shape[0] // self.batch_size
        index = np.arange(self.y_train.shape[0])
        for _ in range(round(self.w_epochs * num_batch / w_num_batch)):
            if self.shuffle:
                np.random.shuffle(index)
                x_train = x_train[index]
                y_train = y_train[index]
            j = 0
            normal = 0
            for batch in range(w_num_batch):
                if self.ratio >= 1:
                    for i in range(int(self.ratio)):
                        if j >= num_batch:
                            j = 0
                        snnl_loss = self.ewe_model.snnl_loss( x_train[j * self.batch_size: (j + 1) * self.batch_size], y_train[j * self.batch_size: (j + 1) * self.batch_size], np.zeros([self.batch_size]), self.factors, self.temperatures)
                        optimizer.zero_grad()
                        snnl_loss.backward()
                        optimizer.step()
                        j += 1
                        normal += 1

                if self.ratio > 0 and self.ratio % 1 != 0 and self.ratio * batch >= j:
                    if j >= num_batch:
                        j = 0
                    snnl_loss = self.ewe_model.snnl_loss(x_train[j * self.batch_size: (j + 1) * self.batch_size], y_train[j * self.batch_size: (j + 1) * self.batch_size], np.zeros([self.batch_size]), self.factors, self.temperatures)
                    optimizer.zero_grad()
                    snnl_loss.backward()
                    optimizer.step()
                    j += 1
                    normal += 1

                batch_data = np.concatenate([trigger[batch * half_batch_size: (batch + 1) * half_batch_size], target_data[batch * half_batch_size: (batch + 1) * half_batch_size]], 0) 
                self.temperatures.required_grad_(True)
                snnl_loss = self.ewe_model.snnl_loss(batch_data, trigger_label, w_label, self.factors, self.temperatures)
                grad = torch.autograd.grad(outputs=snnl_loss, inputs=self.temperatures, grad_outputs=torch.ones_like(snnl_loss), create_graph=True)
                self.temperatures -= self.t_lr * grad[0]

        victim_error_list = []
        num_test = self.x_test.shape[0] // self.batch_size
        for batch in range(num_test):
            victim_error_list.append(self.ewe_model.error_rate(self.x_test[batch * self.batch_size: (batch + 1) * self.batch_size], self.y_test[batch * self.batch_size: (batch + 1) * self.batch_size],))
        victim_error = np.average(victim_error_list)

        victim_watermark_acc_list = []
        for batch in range(w_num_batch):
            victim_watermark_acc_list.append(self.validate_watermark(self.ewe_model, trigger[batch * half_batch_size: (batch + 1) * half_batch_size], self.target))
        victim_watermark_acc = np.average(victim_watermark_acc_list)

        if self.verbose:
            print(f"Victim Model || validation accuracy: {1 - victim_error}, "
                f"watermark success: {victim_watermark_acc}")


    def validate_watermark(self, model, trigger_set, label):
        labels = torch.zeros([self.batch_size, self.num_class], device=self.device)
        # 设置目标标签
        labels[:, label] = 1
        
        # 如果触发数据集的大小小于 batch_size，则重复触发数据以填充 batch_size
        if trigger_set.shape[0] < self.batch_size:
            trigger_data = np.concatenate([trigger_set, trigger_set], 0)[:self.batch_size]
        else:
            trigger_data = trigger_set
        
        trigger_data = torch.tensor(trigger_data, device=self.device).float()
        model.eval()
        # 计算误差率（假设模型输出为 logits，使用交叉熵损失）
        with torch.no_grad():
            outputs = model(trigger_data)
            preds = outputs.argmax(dim=1)
            correct_predictions = (preds == label).float().mean().item()
        
        return correct_predictions
