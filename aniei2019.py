from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn import linear_model,neighbors
from matplotlib import pyplot
from pandas import DataFrame
import numpy as np
import math
import time

N = 100000
theta = 0
alpha = 0.1
beta = 0.001
s = 64

def act(x):
	if x > theta:
		return 1
	else:
		return 0

def hebb(w1,w2,X,y):
	for i in range(len(y)):
		if y[i] == 1:
			w2[-int(X[i][1])+12][int(X[i][0])+12] += alpha
			
		else:
			w1[-int(X[i][1])+12][int(X[i][0])+12] += alpha

	return w1,w2


def oja(w1,w2,X,y):
        for i in range(len(y)):
                if y[i] == 1:
                        w2[-int(X[i][1])+12][int(X[i][0])+12] += alpha - beta*w2[-int(X[i][1])+12][int(X[i][0])+12]

                else:
                        w1[-int(X[i][1])+12][int(X[i][0])+12] += alpha - beta*w1[-int(X[i][1])+12][int(X[i][0])+12]

        return w1,w2

def cost(w,X,y):
	costo = 0
	for i in range(len(y)):
		if y[i] != act(np.dot(w,X[i])):
			costo += 1
	return costo

print("Algoritmos hebbianos")
print(" ___________________")
print("|                   |")
print("|Br.Fernando Aguilar|")
print("|                   |")
print("|___________________|")

print 

print "1-Blobs Classification Problem"
print "2-Moons Classification Problem"
print "3-Circles Classification Problem"

flag1 = input("Introduzca la opcion>>")

# generate 2d classification dataset

if flag1 == 1:
	Xg, yg = make_blobs(n_samples=10*N, centers=2, n_features=2)

elif flag1 == 2:
	Xg, yg = make_moons(n_samples=10*N, noise=0.1)

elif flag1 == 3:
	Xg, yg = make_circles(n_samples=10*N, noise=0.05)


for idx in range(len(Xg)):
	Xg[idx] = s*Xg[idx]

X = Xg[0:9*N]
y = yg[0:9*N]

Xt = Xg[9*N:10*N]
yt = yg[9*N:10*N]

print(len(y))
print(len(yt))

if flag1 == 1:
        # scatter plot, dots colored by class value
        df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
        colors = {0:'red', 1:'blue', 2:'green'}
        fig, ax = pyplot.subplots()
        grouped = df.groupby('label')
        for key, group in grouped:
                group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
        pyplot.show()
elif flag1 == 2:
        df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
        colors = {0:'red', 1:'blue'}
        fig, ax = pyplot.subplots()
        grouped = df.groupby('label')
        for key, group in grouped:
                group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
        pyplot.show()
elif flag1 == 3:
	df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
	colors = {0:'red', 1:'blue'}
	fig, ax = pyplot.subplots()
	grouped = df.groupby('label')
	for key, group in grouped:
    		group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
	pyplot.show()



#print X
#print y



# Entrenamiento

W1 = np.zeros((s*24,s*24))
W2 = np.zeros((s*24,s*24))

W1,W2 = hebb(W1,W2,X,y)



#print W1

# Evaluacion

c = 0

#if len(y) != N:
#	print "ooops"


for i in range(len(y)):
        a = W1[-int(X[i][1])+12][int(X[i][0])+12]
        b = W2[-int(X[i][1])+12][int(X[i][0])+12]
        if a > b:
                if y[i] == 1:
                        c += 1
        else:
                if y[i] == 0:
                        c += 1

print("Metodo hebbiano")
print("costo=",c)
print("Efectividad =",1-float(c)/(10*N))

c = 0

for i in range(len(yt)):
	a = W1[-int(Xt[i][1])+12][int(Xt[i][0])+12]
	b = W2[-int(Xt[i][1])+12][int(Xt[i][0])+12]
	if a > b:
		if yt[i] == 1:
			c += 1
	else:
		if yt[i] == 0:
			c += 1

print("Testing")
print("costo=",c)
print("Efectividad =",1-float(c)/N)

# SEPARACION LINEAL

N = len(y)

PHI = np.ones([N,3])  # 4

for i in range(N):
        PHI[i] = [1,X[i][0],X[i][1]]#,X[i][2]]


transPHI = np.transpose(PHI)

m = np.matmul(transPHI,PHI)

W = np.matmul(np.linalg.inv(m),transPHI) #*np.transpose(y)

W = np.matmul(W,y)


# EVALUACION

# Training set

c = 0
est = 0
fp = 0
fn = 0


for i in range(len(y)):
        x = np.array([X[i][0],X[i][1]])#,X[i][2]])
        phi = [1,x[0],x[1]]#,x[2]]
        est = np.dot(W,phi)

        if est > 0.5:
                est = 1
        else:
                est = 0

        if est == y[i]:
                pass
        else:
                c += 1
                if est == 1:
                        fp += 1
                else:
                        fn += 1


print("Separador lineal")
print("Costo=",c)
print("Efectividad =",1-float(c)/(10*N))


# Testing set

c = 0
est = 0
fp = 0
fn = 0


for i in range(len(yt)):
        x = np.array([Xt[i][0],Xt[i][1]])#,Xt[i][2]])
        phi = [1,x[0],x[1]]#,x[2]]
        est = np.dot(W,phi)

        if est > 0.5:
                est = 1
        else:
                est = 0

        if est == yt[i]:
                pass
        else:
		c += 1
                if est == 1:
                        fp += 1
                else:
                        fn += 1


print("Testing")
print("Costo=",c)
print("Efectividad =",1-float(c)/N)




# REGRESION LOGISTICA

logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=5000,
                                           multi_class='multinomial')
print('LogisticRegression score: %f'
      % logistic.fit(X, y).score(X, y))

print('LogisticRegression score testing: %f'
      % logistic.fit(X, y).score(Xt, yt))


# KNN

knn = neighbors.KNeighborsClassifier()

print('kNN score: %f' % knn.fit(X, y).score(X, y))
print('kNN score testing: %f' % knn.fit(X, y).score(Xt, yt))

"""
for i in range(100):
	W = hebb(W,X,y)
	print w
	c = cost(W,X,y)
	print("costo=",c)
"""
