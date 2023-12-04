import numpy as np

class Adam:
    def __init__(self, params, alpha=0.001,betas=(0.9,0.999),eps=1e-8):
        self.tensors = []
        self.num_params = 0
        for tensor in params:
            self.tensors.append(tensor)
            self.num_params += tensor.value.size
        self.moment1 = np.zeros(self.num_params)
        self.moment2 = np.zeros(self.num_params)
        self.alpha = alpha
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.t = 0


    def step(self):
        self.t += 1
        g = np.zeros(self.num_params)
        ind = 0
        for tensor in self.tensors:
            for x in np.nditer(tensor.grad):
                g[ind] = x
                ind += 1
        self.moment1 = (self.beta1 * self.moment1) + (1. - self.beta1) * g
        self.moment2 = (self.beta2 * self.moment2) + (1. - self.beta2) * (g ** 2)
        at = self.alpha * np.sqrt(1. - (self.beta2**self.t))/ (1. - (self.beta1**self.t))

        theta = np.zeros(self.num_params)
        ind = 0
        for tensor in self.tensors:
            for v in np.nditer(tensor.value):
                theta[ind] = v
                ind += 1

        old_theta = theta
        theta = theta - at*self.moment1 / (np.sqrt(self.moment2) + self.eps)
        # print("adam step change: {}".format(old_theta - theta))

        ind = 0
        for tensor in self.tensors:
            for i in np.ndindex(tensor.shape):
                tensor.value[i] = theta[ind]
                ind += 1
