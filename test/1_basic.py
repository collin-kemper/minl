#!/usr/bin/python3

from minl import Value

a = Value(1)
b = Value(2)
c = Value(-3)
d = a + b
e = d * c
f = e ** b
g = f - a
h = g.relu()
i = -h

print("a.value: {}".format(a.value))
print("b.value: {}".format(b.value))
print("c.value: {}".format(c.value))
print("d.value: {}".format(d.value))
print("e.value: {}".format(e.value))
print("f.value: {}".format(f.value))
print("g.value: {}".format(g.value))
print("h.value: {}".format(h.value))
print("i.value: {}".format(i.value))

print()

i.backward()

print()

print("a.grad: {}".format(a.grad))
print("b.grad: {}".format(b.grad))
print("c.grad: {}".format(c.grad))
print("d.grad: {}".format(d.grad))
print("e.grad: {}".format(e.grad))
print("f.grad: {}".format(f.grad))
print("g.grad: {}".format(g.grad))
print("h.grad: {}".format(h.grad))
print("i.grad: {}".format(i.grad))
