#!/usr/bin/python3
import numpy as np
from minl import Adam, Tensor

def simple_loss(output, target):
    one_hot = np.zeros(output.shape)
    one_hot.flat[target] = 1.
    print("one_hot: {}".format(one_hot))
    target_v = Tensor(one_hot)

    total = output.sum()
    print("total: {}".format(total.value))
    total = total.broadcast(output.shape)
    print("total: {}".format(total.value))
    prob = output * total.inv()
    print("prob: {}".format(prob.value))

    diff = target_v - prob
    print("diff: {}".format(diff.value))
    loss = (diff * diff).sum()
    print("loss: {}".format(loss.value))

    return loss


def main():
    a = Tensor(np.array([0.,1.,3.,3.]))
    b = Tensor(np.array([2.,2.,4.,4.]))
    c = Tensor(np.array([-0.5,-1.5,-2.5,-3.5]))
    x = a * b
    x = x + c

    print("x.value = {}".format(x.value))

    loss = simple_loss(x, 1)
    print("loss.value = {}".format(loss.value))
    loss.backward()
    print(a.grad)
    print(b.grad)
    print(c.grad)

if __name__ == "__main__":
    main()
