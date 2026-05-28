import numpy as np
import paddle  
  
class MaskingGenerator(object):
    def __init__(self, token_num=16, ratio=0.5):
        self.token_num = token_num
        self.ratio = ratio
    def __call__(self):
        arr = paddle.zeros([self.token_num, 1], dtype='int32')  
        num_elements_to_change = int(self.ratio * self.token_num)  
        indices = np.random.choice(np.arange(arr.size), size=num_elements_to_change, replace=False)  
        arr[indices] = 1  
        return arr