import os
import numpy as np
import librosa as lib
from pathlib import Path
import mindspore.dataset.engine as de

def get_all_wavs(root):
    # 先获取目录下的所有.wav文件
    files = []
    for p in Path(root).iterdir():
        if str(p).endswith(".wav"):
            files.append(str(p))
        for s in p.rglob('*.wav'):
            files.append(str(s))
    return list(set(files))

def get_pair_data(root):
    # 筛选出其中的Noisy数据（数据名包含了fileid信息，可以以此获取到Clean数据）
    datas = []
    for idx, wav_path in enumerate(get_all_wavs(root)):
        id_ = os.path.basename(wav_path).split("_")[-1].split(".")[0]
        if str(os.path.basename(wav_path)).startswith("clean_fileid_"):
            continue
        noisy_path = wav_path
        clean_path = os.path.join(os.path.dirname(wav_path), "../clean", "clean_fileid_%s.wav" % id_)
        datas.append((noisy_path, clean_path))
    return datas

class SEDataset:
    def __init__(self, root, audio_len, batch_size):
        self.data = get_pair_data(root)
        self.audio_len = audio_len
        self.batch_size = batch_size
        self.data = np.array_split(np.array(self.data), len(self.data)//batch_size)
    
    def __getitem__(self, index):
        batches = self.data[index]
        return batches

    def __len__(self):
        return len(self.data)
        
def get_dataset(train_root, test_root, batch_size, num_worker, audio_len):
    train_dataset = de.GeneratorDataset(
        source=SEDataset(root=train_root, audio_len=audio_len, batch_size=batch_size),
        column_names=["batches"], # 这里要与Dataset中getitem的输出的名字保持一致
        num_parallel_workers=num_worker,
        shuffle=True
    )
    if test_root is not None:
        test_dataset = de.GeneratorDataset(
            source=SEDataset(root=test_root, audio_len=audio_len, batch_size=batch_size),
            column_names=["batches"],
            num_parallel_workers=num_worker,
            shuffle=False
        )
    
    def padding(x, audio_len):
        if len(x) > audio_len:
            return x[:audio_len]
        elif len(x) < audio_len:
            return np.pad(x, audio_len-len(x))
        else:
            return x
        
    def stft(x):
        x = lib.stft(x, n_fft=512, hop_length=128, win_length=512, window='hann')
        x = np.concatenate([x.real, x.imag], axis=1)
        return x
        
    def sampler(batches):
        # Sampler函数，输入的就是get_item的输出，后续读我们写的HuBERT代码可以看到，多级多卡时数据的划分也是在这里完成的（根据rank_id去分）
        noisy_batch = []
        clean_batch = []
        noisy_cplx_batch = []
        clean_cplx_batch = []
        for noisy_path, clean_path in batches:
            noisy = lib.load(noisy_path, sr=16000)[0]
            clean = lib.load(clean_path, sr=16000)[0]
            noisy = padding(noisy, audio_len)
            clean = padding(clean, audio_len)
            noisy_batch.append(noisy)
            clean_batch.append(clean)
            
        noisy_batch = np.stack(noisy_batch, axis=0)
        clean_batch = np.stack(clean_batch, axis=0)
        # 写到循环外面会稍微快一些，但是你们试了就知道了，华为的CPU很慢，所以在做训练的时候这里非常不推荐on-the-fly地去处理数据（这里stft都会使模型训练非常慢。。。。），推荐提前把特征提取好存下来，后面通过路径去读
        noisy_cplx_batch = stft(noisy_batch) 
        clean_cplx_batch = stft(clean_batch)
        return noisy_batch, clean_batch, noisy_cplx_batch, clean_cplx_batch

    train_dataset = train_dataset.map(
        operations=sampler, 
        input_columns=[
            'batches'
        ], # sampler 函数的输入变量名
        output_columns=[
            'noisy_batch', 'clean_batch', 'noisy_cplx_batch', 'clean_cplx_batch'
        ], # sampler 函数的输出变量名，这些名字必须对应上，不然在建图的时候会报错
        num_parallel_workers=num_worker
    ).project([
        'noisy_batch', 'clean_batch', 'noisy_cplx_batch', 'clean_cplx_batch'
    ])
    if test_root is not None:
        test_dataset = test_dataset.map(
            operations=sampler, 
            input_columns=[
                'batches'
            ],
            output_columns=[
                'noisy_batch', 'clean_batch', 'noisy_cplx_batch', 'clean_cplx_batch'
            ],
            num_parallel_workers=num_worker
        ).project([
            'noisy_batch', 'clean_batch', 'noisy_cplx_batch', 'clean_cplx_batch'
        ])
    else:
        test_dataset = None
    
    return train_dataset, test_dataset

