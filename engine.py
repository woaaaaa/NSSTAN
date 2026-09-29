"""NSSTAN-only optimizer wrapper; no baseline dispatcher."""
import torch
import torch.optim as optim
import util
from model import NSSTAN

class trainer:

    def __init__(self, scaler, task, model_name, in_dim, layer_num, seq_length, num_nodes, batch_size, nhid, dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit):
        self.model_name = model_name
        self.task = task
        self.in_dim = in_dim
        self.fadn_freq_topk = 1
        self.fadn_rfft = True
        self.model = NSSTAN(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, seq_length=seq_length, nhid=nhid, fadn_freq_topk=self.fadn_freq_topk, fadn_rfft=self.fadn_rfft)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5
        self.w_a = 0.99
        self.w_b = 0.01
        print('A+B loss')

    def train(self, input, real_val, ind, input_cluster, epoch, save_interval=10):
        torch.autograd.set_detect_anomaly(True)
        if torch.isnan(input).any():
            print('Input data contains NaN values!')
        self.model.train()
        self.optimizer.zero_grad()
        if self.in_dim == 1:
            if self.task == 'A':
                input = input[:, :1, :, :]
            elif self.task == 'B':
                input = input[:, -1:, :, :]
        output = self.model(input)
        if len(output.shape) >= 4:
            output = output.transpose(1, 3)
        predict = self.scaler.inverse_transform(output)
        real = torch.squeeze(real_val)
        predict = torch.squeeze(predict)
        torch.isnan(real).any()
        torch.isinf(predict).any()
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return (loss.item(), mape, rmse)

    def eval(self, input, real_val, ind, input_cluster, epoch, save_interval=10):
        self.model.eval()
        if self.in_dim == 1:
            if self.task == 'A':
                input = input[:, :1, :, :]
            elif self.task == 'B':
                input = input[:, -1:, :, :]
        output = self.model(input)
        if len(output.shape) >= 4:
            output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        real = torch.squeeze(real_val)
        predict = torch.squeeze(predict)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return (loss.item(), mape, rmse)
