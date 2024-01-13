from nsnet import NsNet, OneStep, EvalOneStep
from mindspore import load_checkpoint, load_param_into_net, nn, save_checkpoint, Tensor, ops
from mindspore.train import Model
from mindspore.nn import Adam
import mindspore
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, SummaryCollector
from mindspore.train.callback._callback import Callback
from mindspore.communication.management import get_group_size, get_rank
import time
import logging
from mindspore import context
import librosa as lib
import numpy as np
import soundfile as sf

def stft(x):
    x = lib.stft(x, n_fft=512, hop_length=128, win_length=512, window='hann')
    x = np.concatenate([x.real, x.imag], axis=0)
    return x

cast = ops.Cast()
model = NsNet(nfft=512, hop_len=60)

param = {}
ms_param = load_checkpoint(r'/home/ma-user/work/ms_tutorial_speech/speech_enhancement/exp/CKP_2-800_4.ckpt')
for key in ms_param.keys():
    if key.find("moment") != -1:
        continue
    param[key.replace("network.", "")] = ms_param[key]
print(load_param_into_net(model, param))
model.set_train(False)
model.set_grad(False)


noisy, sr = lib.load(r"/home/ma-user/work/ms_tutorial_speech/data/1H_-5to20/noisy/book_00101_chp_0007_reader_01727_4_aCjArL_otEY_snr9_tl-30_fileid_242.wav", sr=16000)
noisy_cplx = stft(noisy)
print(noisy_cplx.shape)
est = model.infer(cast(Tensor(noisy_cplx[None, :]), mindspore.float32)).asnumpy()
est_real, est_imag = est[0, :257], est[0, 257:]
est_wave = lib.istft(est_real+1j*est_imag, hop_length=128, win_length=512, window='hann')
print(est_wave.shape)
sf.write("./est.wav", est_wave.flatten(), 16000)