import math
from my_custom_ai.utils.train_utils import FunctionContainer
import torch.nn as nn
import torch
from dl_proj.models.generation_models import CondGenerator, CondGeneratorLSH


class GenerationContainer(FunctionContainer):
    def __init__(self, configs, pixel_recurrency):
        super().__init__()
        self.configs = configs

        weights = (1 / pixel_recurrency**0.85).to(configs.device)
        weights = weights / torch.min(weights)
        self.weights = torch.log(1 + weights**2)

        self.criterion = nn.CrossEntropyLoss(weight=weights)

    def batch_extractor(self, batch, *args, **kwargs):
        b, target, y = batch
        b_size, n_pixel = b.shape
        target = target.reshape(b_size * n_pixel)
        return {'x': b.to(self.configs.device), 'label': y.to(self.configs.device)}, target.to(self.configs.device)

    def loss_function(self, model_output, y, *args, **kwargs):
        model_output = model_output.reshape(model_output.size(0) * model_output.size(1), model_output.size(-1))

        # CE LOSS
        # loss = self.criterion(model_output, y)

        # FOCAL LOSS
        gamma = 2.5
        log_probs = torch.log_softmax(model_output, dim=-1)
        probs = log_probs.exp()

        y_true_prob = probs[torch.arange(probs.size(0)), y]
        y_true_log_prob = log_probs[torch.arange(probs.size(0)), y]
        loss = - ((1 - y_true_prob) ** gamma) * y_true_log_prob * self.weights[y]
        loss = loss.mean()
        return loss

    def validation_performance(self, model, loader, *args, **kwargs):
        total = 0
        bpd = 0.

        criterion = nn.NLLLoss(reduction='sum')
        log_softmax = nn.LogSoftmax(dim=-1)
        scores = {}

        for idx, batch in enumerate(loader):
            batch, y = self.batch_extractor(batch)
            output = model(**batch)
            output = output.reshape(output.size(0) * output.size(1), output.size(-1))

            total += y.size(0)
            bpd += criterion(log_softmax(output), y).item()

        score = bpd / (math.log(2) * total)
        scores['score'] = score
        return scores

    def test_performance(self, model, loader, pbar, *args, **kwargs):
        pass

def get_gen_model(configs):
    if configs.attn_type == 'full':
        model = CondGenerator(configs=configs).to(configs.device)
    elif configs.attn_type == 'lsh':
        model = CondGeneratorLSH(configs=configs).to(configs.device)
    else:
        raise ValueError('Unknown attention type')
    return model