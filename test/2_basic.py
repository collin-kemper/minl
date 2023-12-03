#!/usr/bin/python3

from minl import Value

a = Value(1)
b = Value(0.5)
c = a * b
d = c + a
e = a + d

print("a.value: {}".format(a.value))
print("b.value: {}".format(b.value))
print("c.value: {}".format(c.value))
print("d.value: {}".format(d.value))
print("e.value: {}".format(e.value))

print()

e.backward()

print()

print("a.grad: {}".format(a.grad))
print("b.grad: {}".format(b.grad))
print("c.grad: {}".format(c.grad))
print("d.grad: {}".format(d.grad))
print("e.grad: {}".format(e.grad))
