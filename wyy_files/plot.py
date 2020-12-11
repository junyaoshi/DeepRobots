
import numpy as np
import matplotlib.pyplot as plt
t = np.arange(0,2*np.pi,0.01)
lent = len(t)
x = np.zeros(lent)
y = np.zeros(lent)
theta = np.zeros(lent)

ua = 0.2
uf = 2

zcy = 0.3
zcydot = 0
# zcy = ua*np.sin(2*np.pi*uf*t)
# zcydot = ua*(2*np.pi*uf)*np.cos(2*np.pi*uf*t)


rc = 1

ak = 0.85**2/(rc**2-0.15**2)
zcx = (1-ak)/(1+ak)*np.sqrt(rc**2 - zcy**2)

zc = complex(zcx, zcy)
print(zc,zcy,zcydot)
a = -2*ak/(1+ak)*np.sqrt(rc**2 - zcy**2)
b = 0

zeta = np.zeros(lent, dtype=complex)
z = np.zeros(lent, dtype=complex)
ththeta = 0 # for simplification
xx = 0
yy = 0
for k in range(lent):
    zeta[k] = rc*np.exp(complex(0,t[k]))
    z[k] = complex(((zeta[k] + zc + (a**2+b**2)/(zeta[k] + zc) - (a*b)**2/3/(zeta[k]+zc)**3 )*np.exp(complex(0, ththeta))+ xx) ,yy)

print(z)

plt.plot(z.real, z.imag,'k', linewidth=0.5)
plt.axis('equal')

plt.show()



