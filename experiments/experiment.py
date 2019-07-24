import tensorflow as tf
from ..model import message_passing as msg_ps
from ..model import deep_sets
import trainer
import fetcher

class Experiment():
    def __init__(
        self, model_class=msg_ps.MessagePassing, trainer_class=trainer.Trainer, 
        dataset_fetcher=fetcher.DatasetFetcher,
    ):
        self.model_class = model_class
        self.trainer_class = trainer_class
        self.fetcher = dataset_fetcher('/home/ubuntu/data/ModelNet40_cloud.h5', batch_size=64)

    def run_experiment(self):
        self.trainer_class(self.model_class, self.fetcher).train()

Experiment().run_experiment()