#-*- coding:utf-8 -*-
import random
import numpy as np
from math import sqrt

class LassoRegression:
	"""Lasso regression and sequentially update wj (Error function: (y-xw)^2-lambdaw)"""
	def __init__(self, dim, size, lamb = 1):
		self.dim = dim # dimention of w and y
		self.size = size # sample size
		self.x = np.zeros(dim*size).reshape(dim,size)
		self.y = np.matrix(np.zeros(size))
		self.w = np.matrix(np.zeros(dim))
		self.lamb = lamb

	def readx(self, file):
		self.x = (np.loadtxt(file, delimiter="\t")).reshape(self.dim, self.size)

	def ready(self, file, val = 1):
		self.y = np.matrix(np.loadtxt(file, delimiter="\t")).T/val

	def updatewj(self, j):
		a, c = 0.0, 0.0
		for i in xrange(size):
			a += 2.0*self.x.item(j, i)**2
			wx = 0
			for h in xrange(dim):
				if h == j: continue
				wx += self.w.item(h)*self.x.item(h, i)
			c += 2.0*self.x.item(j, i)*(self.y.item(h)-wx)
		if c < -self.lamb:
			self.w.itemset(j, (c+self.lamb)/a)
		elif c > self.lamb:
			self.w.itemset(j, (c-self.lamb)/a)
		else:
			self.w.itemset(j, 0)

	def update(self):
		for j in xrange(self.dim):
			self.updatewj(j)

	def recursiveUpdate(self, count):
		for i in xrange(count):
			self.update()
			if i%(count/10) == 0: print(self.w)

	def error(self, x, y):
		return sqrt((y-np.dot(x, self.w))**2)

	def getValue(self, x):
		return np.dot(x, self.w)

	def diff(self, x, y):
		print(y-np.dot(x, self.w))

if __name__ == '__main__':
	dim, size = 3, 100
	reg = LassoRegression(dim, size)
	truew = np.matrix(np.linspace(0, 1, dim)).T
	print(truew)
	for i in xrange(size):
		for j in xrange(dim):
			reg.x.itemset((j, i), np.random.normal(0, 1, 1))
	for i in xrange(size):
		for j in xrange(dim):
			reg.y.itemset(i, reg.y.item(i)+reg.x.item(j, i)*truew.item(j))
	reg.recursiveUpdate(1000)

