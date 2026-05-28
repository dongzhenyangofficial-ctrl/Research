import paddle.nn as nn
import paddle.nn.functional as f
import paddle
import numpy as np

class MultiHybridLoss(nn.Layer):
    def __init__(self, alpha=0.5, lambda_0=0):
        super(MultiHybridLoss, self).__init__()
        self.alpha = alpha
        self.lambda_0 = lambda_0
    def forward(self, features, visual, target):
        X, Y = features
        cseloss = (CosineEmbeddingLoss()(X[:, 1:, :], visual) + CosineEmbeddingLoss()(Y[:, 1:, :], visual)) / 2
        # x, y = X[:, 0], Y[:, 0]
        # T = paddle.to_tensor(np.load('Text_vector.npy'))
        # x /= x.norm(axis=-1, keepdim=True)
        # y /= y.norm(axis=-1, keepdim=True)
        # T /= T.norm(axis=-1, keepdim=True)
        # output = 100.0 * (x @ T.t() + y @ T.t()) / 2
        # pcceloss = PCCEVE8(lambda_0=self.lambda_0)(output, target)
        return cseloss
        # return self.alpha * cseloss + (1-self.alpha) * pcceloss

class HybridLoss(nn.Layer):
    def __init__(self, alpha=0.5, lambda_0=0):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.lambda_0 = lambda_0
    def forward(self, output, F, T, A, target):
        pcceloss1 = nn.CrossEntropyLoss()(output, target)
        pcceloss2 = nn.CrossEntropyLoss()(F, target)
        # mse1 = paddle.nn.MSELoss(reduction='mean')(F, A)
        # mse2 = paddle.nn.MSELoss(reduction='mean')(T, A)
        return pcceloss1 * 0.7 + pcceloss2 * 0.3


class CosineEmbeddingLoss(nn.Layer):
    def __init__(self):
        super(CosineEmbeddingLoss, self).__init__()
    def forward(self, x, y):
        cs = f.cosine_similarity(x, y, axis=-1)
        return 1-paddle.mean(cs)

class PCCEVE8(nn.Layer):
    """
    0 Anger
    1 Anticipation
    2 Disgust
    3 Fear
    4 Joy
    5 Sadness
    6 Surprise
    7 Trust
    Positive: Anticipation, Joy, Surprise, Trust
    Negative: Anger, Disgust, Fear, Sadness
    """

    def __init__(self, lambda_0=0):
        super(PCCEVE8, self).__init__()
        self.POSITIVE = {1, 4, 6, 7}
        self.NEGATIVE = {0, 2, 3, 5}

        self.lambda_0 = lambda_0

        self.f0 = nn.CrossEntropyLoss()

    def forward(self, y_pred, y):
        batch_size = y_pred.shape[0]
        weight = [1] * batch_size

        out = self.f0(y_pred, y)
        _, y_pred_label = f.softmax(y_pred, axis=1).topk(k=1, axis=1)
        y_pred_label = y_pred_label.squeeze(axis=1)
        y_numpy = y.numpy()
        y_pred_label_numpy = y_pred_label.numpy()
        for i, y_numpy_i, y_pred_label_numpy_i in zip(range(batch_size), y_numpy, y_pred_label_numpy):
            if (y_numpy_i in self.POSITIVE and y_pred_label_numpy_i in self.NEGATIVE) or (
                    y_numpy_i in self.NEGATIVE and y_pred_label_numpy_i in self.POSITIVE):
                weight[i] += self.lambda_0
        weight_tensor = paddle.to_tensor(np.array(weight), dtype='float32')
        out = paddle.multiply(out, weight_tensor)
        out = paddle.mean(out)
        return out


def get_loss(opt):
    if opt.loss_func == 'ce':
        return nn.CrossEntropyLoss()
    elif opt.loss_func == 'pcce_ve8':
        return PCCEVE8(lambda_0=opt.lambda_0)
    elif opt.loss_func == 'cse':
        return CosineEmbeddingLoss()
    elif opt.loss_func == 'hybrid':
        return HybridLoss(lambda_0=opt.lambda_0)
    elif opt.loss_func == 'mutihybrid':
        return MultiHybridLoss(lambda_0=opt.lambda_0)
    else:
        raise Exception
