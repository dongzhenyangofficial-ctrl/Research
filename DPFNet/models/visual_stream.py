import paddle
import paddle.nn as nn
from models.resnet import pretrained_resnet101


class VisualStream(nn.Layer):
    def __init__(self,
                 snippet_duration,
                 sample_size,
                 n_classes,
                 seq_len,
                 pretrained_resnet101_path):
        super(VisualStream, self).__init__()
        self.snippet_duration = snippet_duration
        self.sample_size = sample_size
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.ft_begin_index = 5
        self.pretrained_resnet101_path = pretrained_resnet101_path

        self._init_norm_val()
        self._init_hyperparameters()
        self._init_encoder()
        self._init_attention_subnets()
        self._init_params()

    def _init_norm_val(self):
        self.NORM_VALUE = 1.0
        self.MEAN = 100.0 / 255.0

    def _init_encoder(self):
        resnet, _ = pretrained_resnet101(snippet_duration=self.snippet_duration,
                                         sample_size=self.sample_size,
                                         n_classes=self.n_classes,
                                         ft_begin_index=self.ft_begin_index,
                                         pretrained_resnet101_path=self.pretrained_resnet101_path)

        children = list(resnet.children())
        self.resnet = nn.Sequential(*children[:-2])  # delete the last fc and the avgpool layer
        for param in self.resnet.parameters():
            param.trainable = False
        # print(len([_ for _ in filter(lambda p: p.trainable, self.resnet.parameters())]), len([_ for _ in filter(lambda p: not p.trainable, self.resnet.parameters())]), len([_ for _ in self.resnet.parameters()]))
    def _init_hyperparameters(self):
        self.hp = {
            'nc': 2048,
            'k': 512,
            'm': 16,
            'hw': 4
        }

    def _init_attention_subnets(self):
        self.conv0 = nn.Sequential(
            *[nn.Conv1D(self.hp['nc'], self.hp['k'], 1, bias_attr=True),
              nn.BatchNorm1D(self.hp['k']),
              nn.ReLU()])

        self.sa_net = nn.LayerList([
            nn.Sequential(
                nn.Conv1D(self.hp['k'], 1, 1, bias_attr=False),
                nn.BatchNorm1D(1),
                nn.Tanh(),
            ),
            nn.Linear(self.hp['m'], self.hp['m'], bias_attr=False),
            nn.Softmax(axis=1)
        ])

        self.ta_net = nn.LayerList([
            nn.Sequential(
                nn.Conv1D(self.hp['k'], 1, 1, bias_attr=False),
                nn.BatchNorm1D(1),
                nn.Tanh(),
            ),
            nn.Linear(self.seq_len, self.seq_len, bias_attr=True),
            nn.ReLU()
        ])

        self.cwa_net = nn.LayerList([
            nn.Sequential(
                nn.Conv1D(self.hp['m'], 1, 1, bias_attr=False),
                nn.BatchNorm1D(1),
                nn.Tanh(),
            ),
            nn.Linear(self.hp['k'], self.hp['k'], bias_attr=False),
            nn.Softmax(axis=1)
        ])

        self.fc = nn.Linear(self.hp['k'], self.n_classes)

    def _init_params(self):
        for subnet in [self.conv0, self.sa_net, self.ta_net, self.cwa_net, self.fc]:
            if subnet is None:
                continue
            for m in subnet.sublayers():
                self._init_module(m)
        for m in self.ta_net[1].sublayers():
            nn.initializer.Constant(1.0)(m.bias)

    def _init_module(self, m):
        if isinstance(m, nn.BatchNorm1D):
            nn.initializer.Constant(1.0)(m.weight)
            nn.initializer.Constant(0.0)(m.bias)
        elif isinstance(m, nn.Conv1D):
            nn.initializer.KaimingNormal()(m.weight)

    def forward(self, input):
        input = input.transpose(0, 1).contiguous()  # input.shape=[seq_len, batch, 3, 16, 112, 112]
        input.div_(self.NORM_VALUE).sub_(self.MEAN)

        seq_len, batch, nc, snippet_duration, sample_size, _ = input.size()
        input = input.view(seq_len * batch, nc, snippet_duration, sample_size, sample_size)
        with paddle.no_grad():
            output = self.resnet(input)
            output = paddle.squeeze(output, axis=2)
            output = paddle.flatten(output, start_axis=2)
        F = self.conv0(output)  # [B x 512 x 16]

        Hs = self.sa_net[0](F)
        Hs = paddle.squeeze(Hs, axis=1)
        Hs = self.sa_net[1](Hs)
        As = self.sa_net[2](Hs)
        As = paddle.mul(As, self.hp['m'])
        alpha = As.view(seq_len, batch, self.hp['m'])

        fS = paddle.mul(F, paddle.unsqueeze(As, axis=1).repeat(1, self.hp['k'], 1))

        G = fS.transpose(1, 2).contiguous()
        Hc = self.cwa_net[0](G)
        Hc = paddle.squeeze(Hc, axis=1)
        Hc = self.cwa_net[1](Hc)
        Ac = self.cwa_net[2](Hc)
        Ac = paddle.mul(Ac, self.hp['k'])
        beta = Ac.view(seq_len, batch, self.hp['k'])

        fSC = paddle.mul(fS, paddle.unsqueeze(Ac, axis=2).repeat(1, 1, self.hp['m']))
        fSC = paddle.mean(fSC, axis=2)
        fSC = fSC.view(seq_len, batch, self.hp['k']).contiguous()
        fSC = fSC.permute(1, 2, 0).contiguous()

        Ht = self.ta_net[0](fSC)
        Ht = paddle.squeeze(Ht, axis=1)
        Ht = self.ta_net[1](Ht)
        At = self.ta_net[2](Ht)
        gamma = At.view(batch, seq_len)

        fSCT = paddle.mul(fSC, paddle.unsqueeze(At, axis=1).repeat(1, self.hp['k'], 1))
        fSCT = paddle.mean(fSCT, axis=2)

        output = self.fc(fSCT)
        return output, alpha, beta, gamma
