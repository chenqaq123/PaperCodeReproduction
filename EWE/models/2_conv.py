import torch
import torch.nn as nn
import torch.nn.functional as F

class Plain_2_conv(nn.Module):
    def __init__(self, num_classes, batch_size):
        super(Plain_2_conv, self).__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)  # Adjust the dimensions as per your input size
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, inter_results=False):
        """
        inter_results: 是否输出中间层结果 bool
        """
        inter_res = []
        
        x = self.pool(F.relu(self.conv1(x)))
        if inter_results:
            inter_res.append(x.clone())
        
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        if inter_results:
            inter_res.append(x.clone())
        
        x = self.dropout(x)
        x = x.view(self.batch_size, -1)  # Flatten the tensor
        if inter_results:
            inter_res.append(x.clone())
        
        x = F.relu(self.fc1(x))
        if inter_results:
            inter_res.append(x.clone())
        
        x = self.dropout(x)
        x = self.fc2(x)
        if inter_results:
            inter_res.append(x.clone())
        
        if inter_results:
            return x, inter_res
        else:
            return x

class EWE_2_conv(nn.Module):
    def __init__(self, num_classes, batch_size):
        self.num_classes = num_classes
        self.batch_size = batch_size
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64*6*6, 128)

    def forward(self, x, inter_results=False):
        """
        inter_results: 是否输出中间层结果 bool
        """
        inter_res = []
        
        x = self.pool(F.relu(self.conv1(x)))
        if inter_results:
            inter_res.append(x.clone())
        
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        if inter_results:
            inter_res.append(x.clone())
        
        x = self.dropout(x)
        x = x.view(self.batch_size, -1)  # Flatten the tensor
        if inter_results:
            inter_res.append(x.clone())
        
        # TODO 这里与源代码relu位置不一样，是否会有问题
        x = F.relu(self.fc1(x))
        if inter_results:
            inter_res.append(x.clone())
        
        x = self.dropout(x)
        x = self.fc2(x)
        if inter_results:
            inter_res.append(x.clone())
        
        if inter_results:
            return x, inter_res
        else:
            return x
