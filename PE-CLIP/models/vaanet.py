import paddle
import paddle.nn as nn
from models.visual_stream import VisualStream
from paddle.vision.models import resnet18

class VAANet(VisualStream):
    def __init__(self,
                 snippet_duration=16,
                 sample_size=112,
                 n_classes=8,
                 seq_len=10,
                 pretrained_resnet101_path='',
                 audio_embed_size=256,
                 audio_n_segments=16,):
        super(VAANet, self).__init__(
            snippet_duration=snippet_duration,
            sample_size=sample_size,
            n_classes=n_classes,
            seq_len=seq_len,
            pretrained_resnet101_path=pretrained_resnet101_path
        )

        self.audio_n_segments = audio_n_segments
        self.audio_embed_size = audio_embed_size

        a_resnet = resnet18(pretrained=True)
        a_conv1 = nn.Conv2D(1, 64, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0), bias_attr=False)
        a_avgpool = nn.AvgPool2D(kernel_size=[8, 2])
        a_modules = [a_conv1] + list(a_resnet.children())[1:-2] + [a_avgpool]
        self.a_resnet = nn.Sequential(*a_modules)
        self.a_fc = nn.Sequential(
            nn.Linear(a_resnet.fc.weight.shape[0], self.audio_embed_size),
            nn.BatchNorm1D(self.audio_embed_size),
            nn.Tanh()
        )

        self.aa_net = {
            'conv': nn.Sequential(
                nn.Conv1D(self.audio_embed_size, 1, 1, bias_attr=False),
                nn.BatchNorm1D(1),
                nn.Tanh(),
            ),
            'fc': nn.Linear(self.audio_n_segments, self.audio_n_segments, bias_attr=True),
            'relu': nn.ReLU(),
        }

        self.av_fc = nn.Linear(self.audio_embed_size + self.hp['k'], self.n_classes)

    def forward(self, visual, audio):
        visual = visual.transpose([1, 0, 2, 3, 4, 5])

        visual = visual/self.NORM_VALUE
        visual = visual-self.MEAN

        # Visual branch
        seq_len, batch, nc, snippet_duration, sample_size, _ = visual.shape
        visual = visual.reshape([seq_len * batch, nc, snippet_duration, sample_size, sample_size])
        with paddle.no_grad():
            F = self.resnet(visual)
            F = paddle.squeeze(F, axis=2)
            F = paddle.flatten(F, start_axis=2)
        F = self.conv0(F)  # [B x 512 x 16]

        Hs = self.sa_net[0](F)
        Hs = paddle.squeeze(Hs, axis=1)
        Hs = self.sa_net[1](Hs)
        As = self.sa_net[2](Hs)
        As = As*self.hp['m']
        alpha = As.reshape([seq_len, batch, self.hp['m']])

        fS = paddle.multiply(F, paddle.unsqueeze(As, axis=1).tile([1, self.hp['k'], 1]))

        G = fS.transpose([0, 2, 1])
        Hc = self.cwa_net[0](G)
        Hc = paddle.squeeze(Hc, axis=1)
        Hc = self.cwa_net[1](Hc)
        Ac = self.cwa_net[2](Hc)
        Ac = Ac*self.hp['k']
        beta = Ac.reshape([seq_len, batch, self.hp['k']])

        fSC = paddle.multiply(fS, paddle.unsqueeze(Ac, axis=2).tile([1, 1, self.hp['m']]))
        fSC = paddle.mean(fSC, axis=2)
        fSC = fSC.reshape([seq_len, batch, self.hp['k']])
        fSC = fSC.transpose([1, 2, 0])

        Ht = self.ta_net[0](fSC)
        Ht = paddle.squeeze(Ht, axis=1)
        Ht = self.ta_net[1](Ht)
        At = self.ta_net[2](Ht)
        gamma = At.reshape([batch, seq_len])

        fSCT = paddle.multiply(fSC, paddle.unsqueeze(At, axis=1).tile([1, self.hp['k'], 1]))
        fSCT = paddle.mean(fSCT, axis=2)  # [bs x 512]

        # Audio branch
        bs = audio.shape[0]
        audio = audio.transpose([1, 0, 2])
        audio = audio.chunk(self.audio_n_segments, axis=0)
        audio = paddle.stack(audio, axis=0)
        audio = audio.transpose([0, 2, 1, 3])  # [16 x bs x 256 x 32]
        audio = paddle.flatten(audio, start_axis=0, stop_axis=1)  # [B x 256 x 32]
        audio = paddle.unsqueeze(audio, axis=1)
        audio = self.a_resnet(audio)
        audio = paddle.flatten(audio, start_axis=1)
        audio = self.a_fc(audio)
        audio = audio.reshape([self.audio_n_segments, bs, self.audio_embed_size])
        audio = audio.transpose([1, 2, 0])

        Ha = self.aa_net['conv'](audio)
        Ha = paddle.squeeze(Ha, axis=1)
        Ha = self.aa_net['fc'](Ha)
        Aa = self.aa_net['relu'](Ha)

        fA = paddle.multiply(audio, paddle.unsqueeze(Aa, axis=1).tile([1, self.audio_embed_size, 1]))
        fA = paddle.mean(fA, axis=2)  # [bs x 256]

        # Fusion
        fSCTA = paddle.concat([fSCT, fA], axis=1)
        output = self.av_fc(fSCTA)

        return output, alpha, beta, gamma
