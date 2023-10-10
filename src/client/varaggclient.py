from .baseclient import  BaseClient
from torch.utils.data import Dataset, RandomSampler, DataLoader
from src.config import 
class VaraggClient(BaseClient):
    def __init__(self, cfg:VaraggClientConfig, **kwargs):
        super().__init__(**kwargs)


        self.train_loader_list = self._create_shuffled_loaders(self.cfg)

    def _create_shuffled_loaders(self, num_loaders: int, dataset:Dataset ):

    def train(self):

        return super().train()
    