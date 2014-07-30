#-*- coding:utf-8 -*-
import numpy as np
import struct
import sys
import random

class Regression:
	"""Linear Regression (solve w=((xTx)^(-1)(xTy))"""
	def __init__(self, length):
		self.length = length
		self.xTx = np.zeros(length*length).reshape(length,length)
		self.xTy = np.matrix(np.zeros(length)).T

	def updatexTx(self, temp):
		self.xTx = self.xTx+np.matrix(temp).T*temp

	def updatexTy(self, x, y):
		self.xTy = self.xTy+np.matrix(x).T*y

	def updatexTy(self, x, y):
		self.xTy = self.xTy+np.matrix(x).T*y

	def getW(self):
		return np.linalg.solve((self.xTx),(self.xTy))

	def update(self, x, y):
		self.updatexTx(x)
		self.updatexTy(x, y)

	def xTynorm(self):
		return np.linalg.norm(self.xTy)

	def printMat(self):
		print self.xTx
		print self.xTy

class LassoRegression(Regression):
	"""Linear Regression (solve w=((xTx)^(-1)(xTy-lambdaE)) """
	"""This is an incorrect solution because |w| is not continuous."""
	def __init__(self, tlambda, length):
		Regression.__init__(self, length)
		self.tlambda = tlambda

	def updatexTylasso(self, x, y):
		self.xTy = self.xTy+np.matrix(x).T*y

	def update(self, x, y):
		self.updatexTx(x)
		self.updatexTylasso(x, y)

	def getW(self):
		return np.linalg.solve(self.xTx, self.xTy-self.tlambda*np.matrix(np.ones(self.length)).T)

class RidgeRegression(Regression):
	"""Linear Regression (solve w=((xTx+lambdaE)^(-1)(xTy)) """
	def __init__(self, tlambda, length):
		Regression.__init__(self, length)
		self.tlambda = tlambda

	def updatexTyridge(self, x, y):
		self.xTy = self.xTy+np.matrix(x).T*y

	def update(self, x, y):
		self.updatexTx(x)
		self.updatexTyridge(x, y)

	def getW(self):
		return np.linalg.solve(self.xTx+self.tlambda*np.matrix(np.identity(self.length)), self.xTy)



if __name__=="__main__":
	kmer = 3
	reg = RidgeRegression(1.0, 4**kmer)
	kmers = np.array([0]*(4**kmer))
	items = range(0, 200)
	for i in range(200):
		for j in range(len(kmers)):
			kmers[j] = random.sample(items, 1)[0]
		reg.update(kmers/200.0, random.sample(items, 1))
	print(reg.getW())


