from packaging import version
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models.wav2vec import Wav2VecModel


class Wav2Vec(nn.Module):

    def __init__(self):
        super(Wav2Vec, self).__init__()
        ckpt = torch.load('./wav2vec/wav2vec_large.pt')
        self.model = Wav2VecModel.build_model(ckpt['args'], task=None)
        self.model.load_state_dict(ckpt['models'])
        self.model = self.model.feature_extractor.conv_layers
        self.model.eval()

        self.slice0 = self.model[0].eval()
        self.slice1 = self.model[1].eval()
        self.slice2 = self.model[2].eval()
        self.slice3 = self.model[3].eval()
        self.slice4 = self.model[4].eval()
        self.slice5 = self.model[5].eval()
        self.slice6 = self.model[6].eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, inputs):
        feat0 = self.slice0(inputs.unsqueeze(1))
        # print(feat0.size())
        feat1 = self.slice1(feat0)
        # print(feat1.size())
        feat2 = self.slice2(feat1)
        # print(feat2.size())
        feat3 = self.slice3(feat2)
        # print(feat3.size())
        feat4 = self.slice4(feat3)
        # print(feat4.size())
        feat5 = self.slice5(feat4)
        # print(feat5.size())
        feat6 = self.slice6(feat5)
        # print(feat6.size())

        return [feat0, feat1, feat2, feat4, feat6]


class ContrastLoss(nn.Module):

    def __init__(self, ablation=False):
        super(ContrastLoss, self).__init__()
        self.wav2vec = Wav2Vec()
        self.l1_loss = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation

    def forward(self, anchor, positive, negative):
        a_feat, p_feat, n_feat = self.wav2vec(anchor), \
                                 self.wav2vec(positive),\
                                 self.wav2vec(negative)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_feat)):
            # print(a_feat[i].size())
            d_ap = self.l1_loss(a_feat[i], p_feat[i].detach())
            if not self.ab:
                d_an = self.l1_loss(a_feat[i], n_feat[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive

        return loss