from my_custom_ai.utils.misc_utils import Config
from my_custom_ai.utils.train_utils import FunctionContainer
from dl_proj.models.generation_models import MyGenerator, MyGeneratorLSH

import torch.nn as nn
import torch


class MyConfig(Config):
    def __init__(self,
            n_samples: int = 1000,
            val_samples: int = 1000,
            test_samples: int = 1000,
            vocab_size: int = 128,
            max_len: int = 256,

            embedding_dim: int = 32,
            heads=8,
            d_k=64,
            d_v=64,
            expansion_factor=4,
            dropout=0.1,
            attn_dropout=0.1,
            depth=1,
            causal_mask=True,
            self_attn_mask=True,

            attn_type='lsh',
            n_rounds: int = 1,
            bucket_size=32,

            lr: float = 1e-4,

            save_on_wb=False,

            task=None,
            # for patched classification
            patch_on_channels=False,
            dataset=None,
            labels=None,
            patch_size=1,
            **kwargs):

        self.n_samples = n_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.n_rounds = n_rounds
        self.bucket_size = bucket_size
        self.heads = heads
        self.d_k = d_k
        self.d_v = d_v
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.lr = lr
        self.causal_mask = causal_mask
        self.self_attn_mask = self_attn_mask
        self.attn_type = attn_type
        self.depth = depth
        self.patch_on_channels = patch_on_channels
        self.dataset = dataset
        self.labels = labels
        self.patch_size = patch_size
        self.task = task

        logger_init = {
            'entity': "imangw-florence-university",  # Your name
            'project': 'DL Project',                 # Your project name
            'name': kwargs['experiment_name'],
            'configs': {
                'length': max_len,
                'vocab_size': vocab_size,
                'attn_type': attn_type,
                'bucket_size': bucket_size,
                'n_rounds': n_rounds,

            } if attn_type == 'lsh' else {
                'length': max_len,
                'vocab_size': vocab_size,
                'attn_type': attn_type,
            }
        }


        if save_on_wb:
            super().__init__(logger_init=logger_init, **kwargs)
        else:
            super().__init__(**kwargs)

class SyntFunctionContainer(FunctionContainer):

    def __init__(self, configs):
        self.configs = configs
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        super().__init__()

    def batch_extractor(self, batch, *args, **kwargs):
        b, y = batch
        y = y.reshape(y.shape[0] * y.shape[1])
        return b.to(self.configs.device), y.to(self.configs.device)


    def loss_function(self, model_output, y, *args, **kwargs):
        model_output = model_output.reshape(model_output.size(0) * model_output.size(1), model_output.size(-1))
        loss = self.criterion(model_output, y)
        return loss


    def validation_performance(self, model, loader, *args, **kwargs):
        total = 0
        corrects = 0
        scores = {}
        for idx, batch in enumerate(loader):
            batch, y = self.batch_extractor(batch)
            output = model(batch)
            output = output.reshape(output.size(0) * output.size(1), output.size(-1))

            predictions = torch.argmax(output, dim=-1)
            b_corrects = torch.sum(predictions == y)

            total += y.size(0) // 2
            corrects += b_corrects.item()

        score = corrects / total
        scores['score'] = score
        return scores


    def test_performance(self, model, loader, pbar, *args, **kwargs):
        pass

def get_model(configs):
    if configs.attn_type == 'full':
        model = MyGenerator(configs=configs).to(configs.device)
    elif configs.attn_type == 'lsh':
        model = MyGeneratorLSH(configs=configs).to(configs.device)
    else:
        raise ValueError('Unknown attention type')
    return model

