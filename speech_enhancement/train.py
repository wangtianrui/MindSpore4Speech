from dataset import *
from nsnet import NsNet, OneStep, EvalOneStep
from mindspore.train import Model
from mindspore.nn import Adam
import mindspore
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, SummaryCollector
from mindspore.train.callback._callback import Callback
from mindspore.communication.management import get_group_size, get_rank
import time
import logging
from mindspore import context


    
class Logger(Callback):
    def __init__(self, dataset_size, eval_dataset, log_interval, logger, model):
        super().__init__()
        self.dataset_size = dataset_size
        self.eval_dataset = eval_dataset
        self.model = model
        try:
            self.rank = get_rank()
        except (ValueError, RuntimeError):
            self.rank = 0
        self.log_interval = log_interval
        self.logger = logger
        self.step = 0
        self.losses = []
    
    def on_train_epoch_begin(self, run_context):
        self.model.set_train()
    
    def on_train_step_begin(self, run_context):
        self.step_time = time.time()
    
    def on_train_step_end(self, run_context):
        step_seconds = time.time() - self.step_time
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs[0].asnumpy()
        self.losses.append(loss)
        batchsize = cb_params.net_outputs[1]
        if self.step % 10 == 0:
            self.logger.info(
                "[Train]Epoch:%d; Step:%d; Time:%.2f; Loss:%s; BS:%d" % (
                    int(self.step/self.dataset_size), 
                    self.step,
                    step_seconds,
                    np.mean(self.losses),
                    batchsize
                )
            )
            self.losses.clear()
        self.step += 1

    
if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=1)
    
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("SE")
    logger.setLevel(logging.DEBUG)
    model = NsNet(nfft=512, hop_len=60)
    logger.info("model inited")
    optimizer = Adam(params=model.trainable_params(), learning_rate=0.00005, weight_decay=0.000001)
    one_step = OneStep(model, optimizer)
    train_dataset, test_dataset = get_dataset(
        train_root=r"/home/ma-user/work/ms_tutorial_speech/data/1H_-5to20",
        test_root=None,
        batch_size=80,
        num_worker=1,
        audio_len=10*16000
    )
    callback_list = [
        Logger(len(train_dataset), test_dataset, log_interval=10, logger=logger, model=EvalOneStep(model)), 
        ModelCheckpoint(directory="./exp", config=CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=5)),
    ]
    model = mindspore.Model(one_step)
    logger.info("training start")
    model.fit(800, train_dataset, callbacks=callback_list)
    