import math
def sigmoid(x):
    y=1/(1+math.exp(-x))
    return y
def activate(inputs,weights):
    h=0
    for x,w in zip(inputs,weights):
        h+=x*w
    
    # sigmoid
    return sigmoid(h)

inputs=[.5,.3,.2]
weights=[.4,.7,.2]
output=activate(inputs,weights)
print(output)