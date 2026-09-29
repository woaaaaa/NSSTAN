"""NSSTAN model. See THIRD_PARTY_NOTICES.md for component attribution.

Only the active forecasting implementation and its dependencies are included.
Constructor order and compatibility members are retained to preserve parameter
initialization and checkpoint keys. No comparison models are distributed here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class nconv1(nn.Module):

    def __init__(self):
        super(nconv1, self).__init__()

    def forward(self, x, A, dims):
        if dims == 2:
            x = torch.einsum('ncvl,vw->ncwl', (x, A))
        elif dims == 3:
            x = torch.einsum('ncvl,nvw->ncwl', (x, A))
        else:
            raise NotImplementedError('Dimension not supported: ' + str(dims))
        return x.contiguous()

class linear(nn.Module):

    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):

    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv1 = nconv1()
        self.c_in = c_in
        c_in = (order * support_len + 1) * self.c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv1(x, a.to(x.device), a.dim())
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv1(x1, a.to(x1.device), a.dim())
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class FrequencyProjector(nn.Module):

    def __init__(self, seq_len, pred_len, enc_in):
        super(FrequencyProjector, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.model_freq = nn.Sequential(nn.Linear(self.seq_len, 64), nn.ReLU())
        self.model_all = nn.Sequential(nn.Linear(64 + seq_len, 128), nn.ReLU(), nn.Linear(128, pred_len))

    def forward(self, main_freq, x):
        main_freq_out = self.model_freq(main_freq)
        inp = torch.concat([main_freq_out, x], dim=-1)
        return self.model_all(inp)

def frequency_decomposition_and_normalization(input, top_k, use_rfft=True):
    """
    Decompose input data into cyclic components and normalize it using frequency domain analysis.

    Parameters:
    - input: Tensor of shape [B, C, N, T], where B is batch size, C is channels, N is nodes, T is time steps.
    - top_k: Integer, number of top frequency components to retain.
    - use_rfft: Boolean, whether to use real FFT (rfft) or full FFT (fft).

    Returns:
    - normalized_input: Tensor with cyclic components removed.
    - cyclic_component: Tensor containing the extracted cyclic components.
    """
    if use_rfft:
        freq_domain = torch.fft.rfft(input, dim=3)
    else:
        freq_domain = torch.fft.fft(input, dim=3)
    magnitudes = freq_domain.abs()
    (_, top_indices) = torch.topk(magnitudes, top_k, dim=3)
    mask = torch.zeros_like(freq_domain)
    mask.scatter_(3, top_indices, 1)
    freq_domain_filtered = freq_domain * mask
    if use_rfft:
        cyclic_component = torch.fft.irfft(freq_domain_filtered, dim=3).real.float()
    else:
        cyclic_component = torch.fft.ifft(freq_domain_filtered, dim=3).real.float()
    normalized_input = input - cyclic_component
    return (normalized_input, cyclic_component)

class FADN(nn.Module):

    def __init__(self, seq_len, pred_len, enc_in, freq_topk=20, rfft=True, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.epsilon = 1e-08
        self.freq_topk = freq_topk
        print('freq_topk : ', self.freq_topk)
        self.rfft = rfft
        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.enc_in))

    def _build_model(self):
        self.model_freq = FrequencyProjector(seq_len=self.seq_len, pred_len=self.pred_len, enc_in=self.enc_in)

    def normalize(self, input):
        (norm_input, x_filtered) = frequency_decomposition_and_normalization(input, self.freq_topk, self.rfft)
        '残差部分用MLP处理'
        return (norm_input, x_filtered)

    def forward(self, batch_x, mode='n'):
        if mode == 'n':
            return self.normalize(batch_x)
        elif mode == 'd':
            return self.denormalize(batch_x)

class NAR(nn.Module):

    def __init__(self, d_series, d_core):
        super(NAR, self).__init__()
        self.gen1 = nn.Linear(d_series, d_core)
        self.gen2 = nn.Linear(d_series + d_core, d_series)

    def forward(self, input, *args, **kwargs):
        (batch_size, channels, num_node) = input.shape
        combined_mean = F.gelu(self.gen1(input))
        if self.training:
            ratio = F.softmax(combined_mean, dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, channels)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
            combined_mean = torch.gather(combined_mean, 1, indices)
            combined_mean = combined_mean.repeat(1, channels, 1)
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1)
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = F.gelu(self.gen2(combined_mean_cat))
        output = combined_mean_cat
        return (output, None)

class NSSTANBackbone(nn.Module):

    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=1, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2, d_core=256):
        super(NSSTANBackbone, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.start_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        self.start_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        self.filter_convs1 = nn.ModuleList()
        self.gate_convs1 = nn.ModuleList()
        self.residual_convs1 = nn.ModuleList()
        self.skip_convs1 = nn.ModuleList()
        self.bn1 = nn.ModuleList()
        self.filter_convs2 = nn.ModuleList()
        self.gate_convs2 = nn.ModuleList()
        self.residual_convs2 = nn.ModuleList()
        self.skip_convs2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.skip_convs_gcn = nn.ModuleList()
        self.bn_gcn = nn.ModuleList()
        self.nar_modules1 = nn.ModuleList([NAR(d_series=dilation_channels, d_core=d_core) for _ in range(blocks * layers)])
        self.nar_modules2 = nn.ModuleList([NAR(d_series=dilation_channels, d_core=d_core) for _ in range(blocks * layers)])
        self.nar_modules_aux = nn.ModuleList([NAR(d_series=dilation_channels, d_core=d_core) for _ in range(blocks * layers)])
        self.supports = supports
        self.supports_len = 0 if supports is None else len(supports)
        if gcn_bool and addaptadj:
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True)
                self.supports_len += 1
            else:
                (m, p, n) = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1
        receptive_field = 1
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs1.append(nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs1.append(nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.residual_convs1.append(nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1, 1)))
                self.skip_convs1.append(nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1)))
                self.bn1.append(nn.BatchNorm2d(residual_channels))
                self.filter_convs2.append(nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs2.append(nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.residual_convs2.append(nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1, 1)))
                self.skip_convs2.append(nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1)))
                self.bn2.append(nn.BatchNorm2d(residual_channels))
                if gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))
                    self.skip_convs_gcn.append(nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, 1)))
                    self.bn_gcn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1), bias=True)
        self.receptive_field = receptive_field

    def forward(self, input1, input2):
        (batch_size, _, num_nodes, time_steps) = input1.size()
        in_len = time_steps
        if in_len < self.receptive_field:
            input1 = nn.functional.pad(input1, (self.receptive_field - in_len, 0, 0, 0))
            input2 = nn.functional.pad(input2, (self.receptive_field - in_len, 0, 0, 0))
        feature1 = self.start_conv1(input1)
        feature2 = self.start_conv2(input2)
        skip = 0
        for i in range(self.blocks * self.layers):
            residual1 = feature1
            filter1 = torch.tanh(self.filter_convs1[i](residual1))
            gate1 = torch.sigmoid(self.gate_convs1[i](residual1))
            x1 = filter1 * gate1
            (b, c, n, t) = x1.shape
            x1 = x1.permute(0, 3, 2, 1).contiguous().view(b * t, n, c)
            (x1, _) = self.nar_modules1[i](x1)
            x1 = x1.view(b, t, n, c).permute(0, 3, 2, 1)
            s1 = self.skip_convs1[i](x1)
            if skip is not 0:
                skip = skip[:, :, :, -s1.size(3):]
            skip = skip + s1
            feature1 = self.residual_convs1[i](x1) + residual1[:, :, :, -x1.size(3):]
            feature1 = self.bn1[i](feature1)
            residual2 = feature2
            filter2 = torch.tanh(self.filter_convs2[i](residual2))
            gate2 = torch.sigmoid(self.gate_convs2[i](residual2))
            x2 = filter2 * gate2
            (b, c, n, t) = x2.shape
            x2 = x2.permute(0, 3, 2, 1).contiguous().view(b * t, n, c)
            (x2, _) = self.nar_modules2[i](x2)
            x2 = x2.view(b, t, n, c).permute(0, 3, 2, 1)
            s2 = self.skip_convs2[i](x2)
            skip = skip[:, :, :, -s2.size(3):]
            skip = skip + s2
            feature2 = self.residual_convs2[i](x2) + residual2[:, :, :, -x2.size(3):]
            feature2 = self.bn2[i](feature2)
        c = feature1 + feature2
        if self.gcn_bool and self.supports is not None:
            if self.addaptadj:
                adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
                new_supports = self.supports + [adp]
            else:
                new_supports = self.supports
            for i in range(self.blocks * self.layers):
                c = self.gconv[i](c, new_supports)
                s_gcn = self.skip_convs_gcn[i](c)
                skip = skip[:, :, :, -s_gcn.size(3):]
                skip = skip + s_gcn
                c = self.bn_gcn[i](c)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = x.squeeze(-1).transpose(1, 2)
        return x

class NSSTAN(nn.Module):

    def __init__(self, device, num_nodes, dropout, supports, gcn_bool, addaptadj, aptinit, in_dim, seq_length, nhid, fadn_freq_topk, fadn_rfft=True):
        super(NSSTAN, self).__init__()
        self.device = device
        self.supports = supports
        self.fadn_module = FADN(seq_len=seq_length, pred_len=seq_length, enc_in=in_dim, freq_topk=fadn_freq_topk, rfft=fadn_rfft).to(device)
        self.backbone = NSSTANBackbone(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16).to(device)

    def forward(self, input_data):
        """NSSTAN forecasting path."""
        (norm_input, x_filtered) = self.fadn_module.normalize(input_data)
        processed = self.backbone(norm_input, x_filtered)
        return processed

