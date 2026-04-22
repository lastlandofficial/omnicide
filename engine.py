import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0 # Starts at 0. Represents how much this value affects the final output
        # Internal variables to build the graph
        self._backward = lambda: None 
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        # Allow adding a Value and a regular number
        other = other if isinstance(other, Value) else Value(other)
        
        # Forward pass
        out = Value(self.data + other.data, (self, other), '+')

        # Backward pass: The derivative of addition is 1. 
        # So we just pass the gradient backwards equally to both children.
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        
        # Forward pass
        out = Value(self.data * other.data, (self, other), '*')

        # Backward pass: The Power Rule / Product Rule.
        # The derivative of x*y with respect to x is y.
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def relu(self):
        # Forward pass: max(0, x)
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        # Backward pass: derivative is 1 if x > 0, else 0
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # Topological order all of the children in the graph
        # This ensures we calculate gradients from the output backwards to the inputs
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Go one variable at a time and apply the chain rule
        self.grad = 1.0 # The derivative of the output with respect to itself is 1
        for node in reversed(topo):
            node._backward()