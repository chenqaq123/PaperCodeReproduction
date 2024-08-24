import torch.nn as nn
import torch.nn.functional as F
import torch
from EWE.utils.utils import snnl_single

class ConvModel(nn.Module):
    def __init__(self, num_classes, batch_size, in_channels):
        super(ConvModel, self).__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)  # Adjust the dimensions as per your input size
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, output_inter_results=False):
        """
        output_inter_results: 是否输出中间层结果 bool
        """
        inter_res = []
        
        x = self.conv1(x)
        if output_inter_results:
            inter_res.append(x.clone())

        x = self.pool(F.relu(x))
        x = self.dropout(x)
        x = self.conv2(x)
        if output_inter_results:
            inter_res.append(x.clone())

        x = x = self.pool(F.relu(x))
        x = self.dropout(x)
        x = x.view(self.batch_size, -1)  # Flatten the tensor
        
        x = self.fc1(x)
        x = self.dropout(x)
        if output_inter_results:
            inter_res.append(x.clone())

        x = F.relu(x)
        x = self.fc2(x)
        
        if output_inter_results:
            return [inter_res[0], inter_res[1], inter_res[2], x]
        else:
            return x
        
    def snnl(self, outputs, temperatures, w_label):
        x0 = outputs[0]
        x1 = outputs[1]
        x2 = outputs[2]
        inv_temp_0 = 100. / temperatures[0]
        inv_temp_1 = 100. / temperatures[1]
        inv_temp_2 = 100. / temperatures[2]
        loss0 = snnl_single(x0, w_label, inv_temp_0)
        loss1 = snnl_single(x1, w_label, inv_temp_1)
        loss2 = snnl_single(x2, w_label, inv_temp_2)
        res = [loss0, loss1, loss2]
        return res
    
    def ce_loss(self, x, y):
        loss = F.cross_entropy(self.forward(x), y)
        return loss

    def snnl_loss(self, x, factors, temperatures, w_label):
        outputs = self.forward(x, inter_results=True)
        snnl_val = self.snnl(outputs, temperatures, w_label)
        soft_nearest_neighbor = factors[0] * snnl_val[0] + factors[1] * snnl_val[1] + factors[2] * snnl_val[2]
        mean_w = torch.mean(w_label)
        soft_nearest_neighbor = (mean_w > 0).float() * soft_nearest_neighbor
        return soft_nearest_neighbor

    def ce_gradient(self, x, y):
        x.required_grad_(True)
        outputs = self.forward(x)
        ce_loss = self.ce(outputs, y)
        gradient = torch.autograd.grad(outputs=ce_loss, inputs=x, grad_outputs=torch.ones_like(ce_loss), create_graph=True)[0]
        return gradient

    def snnl_gradient(self, x, factors, w_label):
        x.require_grad_(True)
        soft_nearest_neighbor = self.snnl_loss(x, factors, w_label)
        gradient = torch.autograd.grad(outputs=soft_nearest_neighbor, inputs=x, grad_outputs=torch.ones_like(soft_nearest_neighbor), create_graph=True)
        return gradient

    def error_rate(self, x, y):
        mistakes = torch.argmax(y, dim=1) != torch.argmax(self.forward(x), dim=1)
        error_rate = torch.mean(mistakes.float())
        return error_rate
    
    