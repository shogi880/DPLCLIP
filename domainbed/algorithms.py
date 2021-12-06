# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from re import X
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import copy
import numpy as np

from domainbed import networks
from domainbed.lib.misc import random_pairs_of_minibatches

# xxx-CLIP
import clip
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import wandb
import math
import time
from torch import Tensor

ALGORITHMS = [
    'ERM',
    'CLIP',
    'DomainCLIP',
    'DPCLIP',
    'DPICLIP',
    'APCLIP',
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
    

class CLIP(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CLIP, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.hparams = hparams
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        assert hparams['backbone'] == "ViT-B/16"
        
        self.clip_model = clip.load(hparams['backbone'])[0].float()

        for param in self.clip_model.parameters():
            param.requires_grad = False
        print('Set self.clip_model.parameters.reguires_grad = False!')
        
        # embedding dim for image and text encoder.
        self.EMBEDDING_DIM = 512
        
        classnames = [name.replace('_', ' ') for name in hparams['class_names']]
        self.prompt = torch.cat([clip.tokenize(ppt) for ppt in classnames]).to(self.device)
        
        # text_features = self.clip_model.encode_text(self.prompt)
        # self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # self.logit_scale_exp = self.clip_model.logit_scale.exp()
        
        
    def update(self, minibatches, unlabeled=None):
        return {'loss': 0}
    
    def predict(self, x):
        # x = self.clip_model.encode_image(x)
        # x = x / x.norm(dim=-1, keepdim=True)
        # return self.logit_scale_exp * x @ self.text_features.t()
        logits_per_image, _ = self.clip_model(x, self.prompt)
        return logits_per_image.softmax(dim=-1)
    
    
class DomainCLIP(CLIP):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DomainCLIP, self).__init__(input_shape, num_classes, num_domains, hparams)
        # assert hparams['prompt'] == 'domain_name'
        domain_name = hparams["domain_name"][hparams["test_envs"][0]].replace('_', ' ')
        class_names = [name.replace('_', ' ') for name in hparams['class_names']]
        print(f'Target-Domain-Prompt: a {domain_name} of a {class_names[0]}')
        self.prompt = torch.cat([clip.tokenize(f'a {domain_name} of a {ppt}') for ppt in class_names]).to(self.device)


class DPCLIP(CLIP):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DPCLIP, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.time = 0   
        #  initial prompt.
        prompt_prefix = ' '.join(['X'] * hparams['num_domain_tokens'])
        
        classnames = [name.replace('_', ' ') for name in hparams['class_names']]
        prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]
        # prompts:  ['X X X X X X X X dog.', 'X X X X X X X X elephant.' ...]
        
        #  to get default token_prefix and token_suffix.
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        # tokenized_prompts[0] = tensor([49406,   343,   343,   343,   343,   343,   343,   343,   343,  1929, 269, 49407, 0, 0, ...])
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(self.tokenized_prompts).type(self.clip_model.dtype)
        
        self.register_buffer('token_prefix', embedding[:, :1, :])  # SOS
        #  torch.Size([7, 1, 512])
        #  [-0.0001,  0.0002, -0.0046,  ...,  0.0010,  0.0025,  0.0049]
        
        self.register_buffer('token_suffix', embedding[:, hparams['num_domain_tokens'] + 1:, :])  # CLS, EOS
        # torch.Size([7, 68, self.EMBEDDING_DIM]), 68 := 77 - num_domain_tokens_tokens - 2.
        # [ 0.0013,  0.0046, -0.0115,  ...,  0.0112,  0.0147,  0.0040],...,.
        
        self.network = networks.MLP(self.EMBEDDING_DIM, self.EMBEDDING_DIM * hparams['num_domain_tokens'], hparams).to(device=self.device, dtype=self.clip_model.dtype)
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)
        
        self.network.apply(init_weights)
        
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"]
        )
            
    def update(self, minibatches, unlabeled=None):
        # minibatches = [[domain_1], [domain_2], [domain_3]]
        all_x = [data[0].cuda().float() for data in minibatches]
        all_y = torch.cat([data[1].cuda().long() for data in minibatches])

        #  encode image for each domain.
        image_features = [self.clip_model.encode_image(x) for x in all_x]
        
        #  extract domain_feature for each domain. [32, self.EMBEDDING_DIM] -> [32, self.EMBEDDING_DIM * num_domain_tokens] -> [self.EMBEDDING_DIM * num_domain_tokens].
        domain_features = [self.network(feature) for feature in image_features]
        image_features = torch.cat(image_features)
        #  reshape [self.batch_size, self.EMBEDDING_DIM.]:  -> [1, self.EMBEDDING_DIM.]
        mean_domain_features = [feature.mean(dim=0, keepdim=True) for feature in domain_features]
        
        #  reshape [1, self.EMBEDDING_DIM.]:  -> [7, self.EMBEDDING_DIM.]
        _mean_domain_features = [feature.repeat_interleave(len(self.hparams['class_names']), dim=0) for feature in mean_domain_features]
        
        #  generate text_feature from domain_feature. text_features.size = [3, 7, 512]
        # text_features = [self._get_text_features(feature) for feature in _mean_domain_features]
        text_features = torch.cat([self._get_text_features(feature) for feature in _mean_domain_features])
            
        intra_loss = 0
        inter_loss = 0
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits_per_image = self.clip_model.logit_scale.exp() * image_features @ text_features.t()
        loss = F.cross_entropy(logits_per_image, all_y)
            
        intra_loss, inter_loss = self.domain_loss(domain_features, mean_domain_features)
        
        total_loss = self.calcurate_loss(loss, intra_loss, inter_loss)
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return {"total_loss": total_loss.item(), "intra_loss": intra_loss.item()}

    def calcurate_loss(self, *loss):
        total_loss = self.hparams['gamma'] * loss[0] + (1 - self.hparams['gamma']) * (self.hparams['beta'] * loss[1] + (1 - self.hparams['beta']) * loss[2])
        return total_loss

    def domain_loss(self, domain_features, mean_domain_features):
        intra_loss = 0
        inter_loss = 0
        for i in range(len(domain_features)):
            # simple cosine similarity loss for inner_domain_feature.
            intra_loss += 1 - F.cosine_similarity(domain_features[i], mean_domain_features[i].repeat_interleave(len(domain_features[i]), dim=0)).mean()
            for j in range(i + 1, len(domain_features)):
                # simple cosine similarity loss for outer_domain_feature.
                inter_loss += F.cosine_similarity(mean_domain_features[i], mean_domain_features[j])
                
        # to balance between the loss and the loss.
        if len(domain_features) > 1:
            inter_loss /= (len(domain_features) * (len(domain_features) - 1) / 2)
            intra_loss /= len(domain_features)
            
        return intra_loss, inter_loss
 
    def _get_text_features(self, domain_feature, coop=False):
        #  reshape domain_feature: [7, 16 * self.EMBEDDING_DIM] -> [7, 16, self.EMBEDDING_DIM]
        if coop:
            domain_feature = domain_feature.unsqueeze(0).expand(len(self.hparams['class_names']), -1, -1)
        domain_feature = domain_feature.reshape(-1, self.hparams['num_domain_tokens'], self.EMBEDDING_DIM)
        #  reshape domain_feature: [7, 16, self.EMBEDDING_DIM] -> [7, 77, self.EMBEDDING_DIM]
        domain_feature = torch.cat([self.token_prefix, domain_feature, self.token_suffix], dim=1)
        #  refer CoOp: CoOP github. https://github.com/KaiyangZhou/CoOp/blob/b0a058869cef00a4e4ea5256d40fd7681119c099/trainers/coop.py#L46
        x = domain_feature + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        #  mapping domain_features to text_features.
        text_features = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ self.clip_model.text_projection      
        return text_features

    def predict(self, x):
        image_feature = self.clip_model.encode_image(x)
        
        domain_feature = self.network(image_feature)
        mean_domain_feature = torch.mean(domain_feature, dim=0, keepdim=True).repeat_interleave(len(self.hparams['class_names']), dim=0)
        text_feature = self._get_text_features(mean_domain_feature)
        
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        return self.clip_model.logit_scale.exp() * image_feature @ text_feature.t()


class DPICLIP(DPCLIP):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DPICLIP, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.hparams['beta'] = 1
        dim_domain_feature = self.EMBEDDING_DIM * hparams['num_domain_tokens']
        self.classification_network = networks.Classifier(
                                        dim_domain_feature,
                                        num_domains,
                                        self.hparams
        )
        self.optimizer = torch.optim.SGD(params=[
            {"params": self.network.parameters(), "lr": self.hparams["lr"]},
            {"params": self.classification_network.parameters(), "lr": self.hparams["lr"] * 0.5},]
        )
        
    def domain_loss(self, domain_features, mean_domain_features):
        num_batch = len(domain_features[0])
        all_domain_feature = torch.cat(domain_features)
        
        prediction = self.classification_network(all_domain_feature)
        y = torch.tensor([0] * num_batch + [1] * num_batch + [2] * num_batch, device=self.device).long()
        
        return F.cross_entropy(prediction, y), 0
    
    def calcurate_loss(self, *loss):
        total_loss = self.hparams['gamma'] * loss[0] + (1 - self.hparams['gamma']) * loss[1]
        return total_loss


class APCLIP(DPCLIP):
    """
    Simply Amortize Prompt for CLIP.
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(APCLIP, self).__init__(input_shape, num_classes, num_domains, hparams)
        
    def domain_loss(self, domain_features, mean_domain_features):
        return torch.tensor(0), torch.tensor(0)
    
    def calcurate_loss(self, *loss):
        # only classification loss.
        return loss[0]


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams)

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

    def forward(self, x):
        return self.predict(x)
