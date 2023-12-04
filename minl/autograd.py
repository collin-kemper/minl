import numpy as np
from enum import Enum

class Op(Enum):
    DECL  = 0
    NEG   = 1
    RELU  = 2
    INV   = 3
    ADD   = 4
    MUL   = 5
    POW   = 6
    SUM   = 7
    BROAD = 8

def get_broadcast_dims(shape1, shape2):
    shape1 = list(shape1)
    shape2 = list(shape2)
    assert(len(shape1) <= len(shape2))
    while len(shape1) != len(shape2):
        shape1.insert(0, 1)
    dims = []
    for i in range(len(shape1)):
        if shape1[i] == 1 and shape2[i] != 1:
            dims.append(i)
    if len(dims) == 0:
        return None
    return tuple(dims)

# class Value:
#     def __init__(self, value, op=Op.DECL, l=None, r=None):
#         self.value = float(value)
#         self.op = op
#         self.l = l
#         self.r = r
#         self.grad = 0.
#         self.grad_updates = 0
#         self.uses = 0

#     def __neg__(self):
#         return Value(-self.value, Op.NEG, self)

#     def relu(self):
#         return Value(max(0., self.value), Op.RELU, self)

#     def __add__(self, r):
#         return Value(self.value + r.value, Op.ADD, self, r)

#     def __sub__(self, r):
#         neg_r = Value(-r.value, Op.NEG, r)
#         return Value(self.value + neg_r.value, Op.ADD, self, neg_r)

#     def __mul__(self, r):
#         return Value(self.value * r.value, Op.MUL, self, r)

#     def __pow__(self, r):
#         return Value(self.value ** r.value, Op.POW, self, r)

#     def grad_update(self):
#         self.grad_updates += 1
#         if self.grad_updates != self.uses:
#             return
#         self.uses = 0
#         self.grad_updates = 0

#         if self.op == Op.DECL:
#             return
#         elif self.op == Op.NEG:
#             self.l.grad += -self.grad
#             self.l.grad_update()
#         elif self.op == Op.RELU:
#             self.l.grad += self.grad if self.value >= 0. else 0.
#             self.l.grad_update()
#         elif self.op == Op.ADD:
#             self.l.grad += self.grad
#             self.r.grad += self.grad
#             self.l.grad_update()
#             self.r.grad_update()
#         elif self.op == Op.MUL:
#             self.l.grad += self.grad * self.r.value
#             self.r.grad += self.grad * self.l.value
#             self.l.grad_update()
#             self.r.grad_update()
#         elif self.op == Op.POW:
#             self.l.grad += self.grad * self.r.value * (self.l.value ** (self.r.value - 1.))
#             self.r.grad += self.grad * math.log(self.r.value) * (self.l.value ** (self.r.value))
#             self.l.grad_update()
#             self.r.grad_update()

#     def setup_graph(self):
#         self.uses += 1
#         if self.l:
#             self.l.setup_graph()
#         if self.r:
#             self.r.setup_graph()

#     def backward(self):
#         self.setup_graph()
#         self.grad = 1.
#         self.grad_update()

tensor_id = 0
debug_depth = 0

class Tensor:
    def __init__(self, value, op=Op.DECL, l=None, r=None):
        self.value = value
        self.shape = value.shape
        self.op = op
        self.l = l
        self.r = r
        self.grad = np.zeros(self.shape)
        self.grad_updates = 0
        self.uses = 0
        global tensor_id
        self.id = tensor_id
        tensor_id += 1

        if op == Op.BROAD:
            self.broadcast_dims = get_broadcast_dims(self.l.shape, self.shape)

    # element-wise operations
    def __neg__(self):
        return Tensor(-self.value, Op.NEG, self)

    def relu(self):
        return Tensor(np.maximum(self.value, 0.), Op.RELU, self)

    def inv(self):
        return Tensor(1./self.value, Op.INV, self)

    def __add__(self, r):
        return Tensor(self.value + r.value, Op.ADD, self, r)

    def __sub__(self, r):
        neg_r = Tensor(-r.value, Op.NEG, r)
        return Tensor(self.value + neg_r.value, Op.ADD, self, neg_r)

    def __mul__(self, r):
        value = self.value * r.value
        return Tensor(value, Op.MUL, self, r)

    def __pow__(self, r):
        return Tensor(self.value ** r.value, Op.POW, self, r)

    def sum(self, axis=None):
        if axis == None:
            val = np.array([np.sum(self.value)], ndmin=len(self.shape))
            return Tensor(val, Op.SUM, self)
        else:
            return Tensor(np.sum(self.value, axis=axis, keepdims=True), Op.SUM, self)

    def broadcast(self, shape):
        return Tensor(np.broadcast_to(self.value, shape), Op.BROAD, self)

    def grad_update(self):
        self.grad_updates += 1
        # print("grad_update: {} {}".format(self.id, self.op))
        # print("\tgrad_updates {} uses {}".format(self.grad_updates, self.uses))
        if self.grad_updates != self.uses:
            return
        self.uses = 0
        self.grad_updates = 0
        # print("\t{}".format(self.grad))

        if self.op == Op.DECL:
            return
        elif self.op == Op.NEG:
            self.l.grad += -self.grad
            self.l.grad_update()
        elif self.op == Op.RELU:
            self.l.grad += self.grad * np.heaviside(self.value, 0.5)
            self.l.grad_update()
        elif self.op == Op.INV:
            self.l.grad += -self.grad * self.l.value ** -2
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
            self.r.grad += self.grad * np.log(self.r.value) * (self.l.value ** (self.r.value))
            self.l.grad_update()
            self.r.grad_update()
        elif self.op == Op.SUM:
            self.l.grad += self.grad
            self.l.grad_update()
        elif self.op == Op.BROAD:
            self.l.grad += self.grad.sum(axis=self.broadcast_dims)
            self.l.grad_update()

    def setup_graph(self, visited):
        global debug_depth
        if self.uses == 0:
            self.grad[:] = 0.
        # print("{}id {} {} used".format(debug_depth*"|  ", self.id, self.op))
        self.uses += 1
        if self in visited:
            return
        visited.add(self)
        debug_depth += 1
        if self.l:
            # print("{}uses l".format(debug_depth*"|  ", self.id, self.op))
            self.l.setup_graph(visited)
        if self.r:
            # print("{}uses r".format(debug_depth*"|  ", self.id, self.op))
            self.r.setup_graph(visited)
        debug_depth -= 1

    def backward(self):
        self.setup_graph(set())
        self.grad = np.full(self.value.shape, 1.)
        self.grad_update()
