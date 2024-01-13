import mindspore
from mindspore import nn, ops
from scipy.signal import get_window
import numpy as np
from mindspore.nn.wrap import TrainOneStepCell
from mindspore.ops import composite as C
from mindspore.ops import functional as F

def init_kernels(win_len, win_inc, fft_len, win_type="hamming", invers=False):
    window = get_window(win_type, win_len, fftbins=True)  # win_len
    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T  # 514,400
    if invers:
        kernel = np.linalg.pinv(kernel).T
    kernel = kernel * window
    kernel = kernel[:, None, :]
    return mindspore.Tensor(kernel.astype(np.float32), mindspore.float32), mindspore.Tensor(window[None, :, None].astype(np.float32), mindspore.float32)

class STFT(nn.Cell):
    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming'):
        super().__init__()
        self.fft_len = fft_len
        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
        self.kernel = mindspore.Parameter(kernel, requires_grad=False)
        self.padding_len = win_len-win_inc
        self.stride = win_inc
        self.pad = nn.Pad(paddings=((0, 0), (0, 0), (self.padding_len, self.padding_len)), mode="CONSTANT")
    
    def construct(self, x):
        x = self.pad(x)
        x = mindspore.ops.conv1d(x, self.kernel, stride=self.stride)
        return mindspore.ops.stop_gradient(x)

class ISTFT(nn.Cell):
    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming'):
        super().__init__()
        self.fft_len = fft_len
        kernel, window = init_kernels(win_len, win_inc, self.fft_len, win_type, invers=True)
        self.kernel = mindspore.Parameter(mindspore.ops.expand_dims(kernel, 1), requires_grad=False)
        self.window = mindspore.Parameter(window, requires_grad=False)
        self.convT = nn.Conv1dTranspose(
            in_channels=514, out_channels=1, kernel_size=512, stride=win_inc, pad_mode='valid'
        )
        self.convT.weight = self.kernel
        self.convT_weight = nn.Conv1dTranspose(
            in_channels=win_len, out_channels=1, kernel_size=win_len, stride=win_inc, pad_mode='valid'
        )
        self.enframe = mindspore.Parameter(mindspore.Tensor(np.eye(win_len)[:, None, None, :], mindspore.float32), requires_grad=False)
        self.convT_weight.weight = self.enframe
        self.padding_len = win_len-win_inc
    
    def construct(self, x):
        T = x.shape[-1]
        x = self.convT(x)
        window = mindspore.numpy.tile(self.window, (1, 1, T))
        coff = self.convT_weight(window)
        x = x / (coff + 1e-8)
        x = x[:, 0, self.padding_len:-self.padding_len]
        return x

class NsNet(nn.Cell):
    def __init__(self, nfft, hop_len):
        super().__init__()
        self.nfft = nfft
        # self.stft = STFT(win_len=nfft, win_inc=hop_len, fft_len=nfft)
        # self.istft = ISTFT(win_len=nfft, win_inc=hop_len, fft_len=nfft)
        self.encoder = nn.SequentialCell(
            nn.Dense(257, 400, weight_init="normal", bias_init="zeros"),
            nn.PReLU(),
            nn.LayerNorm((400,))
        )
        self.rnn = nn.GRU(400, 400, 2, has_bias=True, batch_first=True, bidirectional=False)
        self.decoder = nn.SequentialCell(
            nn.Dense(400, 600, weight_init="normal", bias_init="zeros"),
            nn.LayerNorm((600,)),
            nn.PReLU(),
            nn.Dense(600, 600, weight_init="normal", bias_init="zeros"),
            nn.LayerNorm((600,)),
            nn.PReLU(),
            nn.Dense(600, 257, weight_init="normal", bias_init="zeros"),
            nn.Sigmoid(),
        )
        self.mse = nn.MSELoss()
    
    def construct(self, noisy_batch, clean_batch, noisy_cplx_batch, clean_cplx_batch):
        # x = mindspore.ops.expand_dims(noisy_batch, 1)
        # cplx = self.stft(x) 
        # B, F*2, T
        real, imag = noisy_cplx_batch[:, :self.nfft//2+1, :], noisy_cplx_batch[:, -(self.nfft//2+1):, :]
        magnitude = ((real ** 2 + imag ** 2 + 1e-8) ** 0.5).log().permute(0, 2, 1)
        
        ff_result = self.encoder(magnitude)
        rnn_result, _ = self.rnn(ff_result)
        mask = self.decoder(rnn_result).permute(0, 2, 1)
        real_result = real * mask
        imag_result = imag * mask
        est_cplx = mindspore.ops.cat([real_result, imag_result], 1)
        
        # est_waveform = self.istft(est_cplx)
        
        loss = self.mse(est_cplx, clean_cplx_batch)
        return loss

    def infer(self, noisy_cplx_batch):
        real, imag = noisy_cplx_batch[:, :self.nfft//2+1, :], noisy_cplx_batch[:, -(self.nfft//2+1):, :]
        magnitude = ((real ** 2 + imag ** 2 + 1e-8) ** 0.5).log().permute(0, 2, 1)
        
        ff_result = self.encoder(magnitude)
        rnn_result, _ = self.rnn(ff_result)
        mask = self.decoder(rnn_result).permute(0, 2, 1)
        real_result = real * mask
        imag_result = imag * mask
        est_cplx = mindspore.ops.cat([real_result, imag_result], 1)
        return est_cplx

class OneStep(TrainOneStepCell):
    def __init__(self, model, optimizer):
        super(OneStep, self).__init__(model, optimizer)
        self.network.set_train()
        self.network.set_grad()
    
    def construct(self, *inputs):
        batch_size = inputs[0].shape[0]
        loss = self.network(*inputs)
        grads = self.grad_no_sens(self.network, self.weights)(*inputs)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss, batch_size
    
class EvalOneStep(nn.Cell):
    def __init__(self, model):
        super(EvalOneStep, self).__init__()
        self.network = model
    
    def construct(self, *inputs):
        # batch_size = inputs[0].shape[0]
        loss = self.network(*inputs)
        return loss