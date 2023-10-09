import numpy as np
import matplotlib.pyplot as plt

def myfunc(x):
   return np.cos(x) * np.exp(x)

x = np.linspace(-2*np.pi, 2*np.pi, 100)
y = myfunc(x)
plt.plot(x, y, linewidth=2)
plt.grid();plt.ylabel('cos(x) * exp(x)--->');plt.xlabel('x--->');plt.title("cos(x)*exp(x)")
plt.savefig('myfunc.png')
plt.show()

np.random.seed(0)

mu,sig = 5.0,2.0
normal_v = np.random.normal(mu, sig, 100000)
uniform_v = np.random.uniform(0, 10, 100000)
print(np.mean(normal_v), np.std(normal_v))
print(np.mean(uniform_v), np.std(uniform_v))

_, x, _ = plt.hist(uniform_v, 100, density=True)
plt.plot(x,np.ones_like(x)*0.1,linewidth = 2)
plt.show()

_, x, _ = plt.hist(normal_v, 100, density=True)
plt.plot(x,1/(sig * np.sqrt(2 * np.pi))*np.exp( - (x - mu)**2 / (2 * sig**2) ),linewidth = 2)
plt.show()
