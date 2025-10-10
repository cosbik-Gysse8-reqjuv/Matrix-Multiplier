import sympy as sp

betaT = sp.Matrix([[1, -0.2104, -0.8075]])
a = sp.Matrix([
     [0.0323], 
     [-0.0941], 
     [-0.0295]
])
n = sp.Matrix([
     [0.0054], 
     [0.0063], 
     [0.0052]
])
I = sp.eye(3)
gcon = -0.4168
B = I - a * betaT
C = n + a * gcon
print("Matrix B:")
sp.pprint(B)
print("Matrix C:")
sp.pprint(C)