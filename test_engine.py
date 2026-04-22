from engine import Value

# 1. Define our inputs as Value objects
x = Value(-2.0, label='x')
y = Value(3.0, label='y')

# 2. Build the computational graph (Forward Pass)
# Equation: out = (x * y) + relu(x)
xy = x * y; xy.label = 'x*y'
relu_x = x.relu(); relu_x.label = 'relu(x)'
out = xy + relu_x; out.label = 'out'

print(f"Forward pass output: {out.data}") # Expected: (-2 * 3) + 0 = -6

# 3. Calculate gradients (Backward Pass)
out.backward()

# 4. Check the gradients
print("\nGradients after backward pass:")
print(f"Derivative with respect to x (dx): {x.grad}") 
print(f"Derivative with respect to y (dy): {y.grad}")