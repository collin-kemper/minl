import math
from enum import Enum

class Op(Enum):
    DECL = 0
    NEG  = 1
    RELU = 2
    ADD  = 3
    MUL  = 4
    POW  = 5

class Value:
    def __init__(self, value, op=Op.DECL, l=None, r=None):
        self.value = float(value)
        self.op = op
        self.l = l
        self.r = r
        self.grad = 0.
        self.grad_updates = 0
        self.uses = 0

    def __neg__(self):
        return Value(-self.value, Op.NEG, self)

    def relu(self):
        return Value(max(0., self.value), Op.RELU, self)

    def __add__(self, r):
        return Value(self.value + r.value, Op.ADD, self, r)

    def __sub__(self, r):
        neg_r = Value(-r.value, Op.NEG, r)
        return Value(self.value + neg_r.value, Op.ADD, self, neg_r)

    def __mul__(self, r):
        return Value(self.value * r.value, Op.MUL, self, r)

    def __pow__(self, r):
        return Value(self.value ** r.value, Op.POW, self, r)

    def grad_update(self):
        self.grad_updates += 1
        if self.grad_updates != self.uses:
            return
        self.uses = 0
        self.grad_updates = 0

        if self.op == Op.DECL:
            return
        elif self.op == Op.NEG:
            self.l.grad += -self.grad
            self.l.grad_update()
        elif self.op == Op.RELU:
            self.l.grad += self.grad if self.value >= 0. else 0.
            self.l.grad_update()
        elif self.op == Op.ADD:
            self.l.grad += self.grad
            self.r.grad += self.grad
            self.l.grad_update()
            self.r.grad_update()
        elif self.op == Op.MUL:
            self.l.grad += self.grad * self.r.value
            self.r.grad += self.grad * self.l.value
            self.l.grad_update()
            self.r.grad_update()
        elif self.op == Op.POW:
            self.l.grad += self.grad * self.r.value * (self.l.value ** (self.r.value - 1.))
            self.r.grad += self.grad * math.log(self.r.value) * (self.l.value ** (self.r.value))
            self.l.grad_update()
            self.r.grad_update()

    def setup_graph(self):
        self.uses += 1
        if self.l:
            self.l.setup_graph()
        if self.r:
            self.r.setup_graph()

    def backward(self):
        self.setup_graph()
        self.grad = 1.
        self.grad_update()
