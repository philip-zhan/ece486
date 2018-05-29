"""
file	  matlib.py
author	Ernesto P. Adorio, PhD.
		  UPDEPP (UP Clarkfield)
		  ernesto.adorio@gmail.com
revisions 2009.01.16 added matdots,matrandom, isiterable
"""
def tabular(hformat, headers, bformat, M):
	# added nov.29, 2008.
	# prints the table data.
	nrows = len(M)
	ncols = len(M[0])
	# print headers.
	for j, heading in enumerate(headers):
		print hformat[j] % heading,
	print
	# print the body.
	for i, row in enumerate(M):
		for j, col in enumerate(M[i]):
			print bformat[j] % M[i][j],
		print
	print
def vecadd(X,Y):
	n = len(X)
	if n != len(Y):
	   return None
	return [x + y for x,y in zip(X,Y)]
def vecsub(X, Y):
	n = len(X)
	if n != len(Y):
	   return None
	return [x - y for x,y in zip(X,Y)]
def eye(m, n= None):
	if n is None:
		n = m
	B= [[0]* n for i in range(m)]
	for i in range(m):
		B[i][i] = 1.0
	return B
matiden = eye
def matzero(m, n = None):
	"""
	Returns an m by n zero matrix.
	"""
	if n is None:
		n = m
	return [[0]* n for i in range(m)]
def matdiag(D):
	"""
	Returns a diagonal matrix D.
	"""
	n = len(D)
	A = [[0] * n for i in range(n)]
	for i in range(n):
		A[i][i] = D[i]
	return A
def matcol(X, j):
	# Returns the jth column of matrix X.
	nrows = len(X)
	return [X[i][j] for i in range(nrows)]
def trace(A):
	"""
	Returns the trace of a matrix.
	"""
	return sum([A[i][i] for i in range(len(A))])
def matadd(A, B):
	"""
	Returns C = A + B.
	"""
	try:
		m = len(A)
		if m != len(B):
			return None
		n = len(A[0])
		if n != len(B[0]):
			return None
		C = matzero(m, n)
		for i in range(m):
			for j in range(n):
				C[i][j] = A[i][j] + B[i][j]
		return C
	except:
		return None
def matsub(A, B):
	"""
	returns C = A - B.
	"""
	try:
		m = len(A)
		if m != len(B):
			return None
		n = len(A[0])
		if n != len(B[0]):
			return None
		C = matzero(m, n)
		for i in range(m):
			for j in range(n):
				C[i][j] = A[i][j] - B[i][j]
		return C
	except:
		return None
def matcopy(A):
	B = []
	for a in A:
	   B.append(a[:])
	return B
def matkmul(A, k):
	"""
	Multiplies each element of A by k.
	"""
	B = matcopy(A)
	for i in range(len(A)):
		for j in range(len(A[0])):
			B[i][j] *= k
	return B
def transpose(A):
	"""
	Returns the transpose of A.
	"""
	m,n = matdim(A)
	At = [[0] * m for j in range(n)]
	for i in range(m):
		for j in range(n):
			At[j][i] = A[i][j]
	return At
matt = transpose
mattrans = transpose
def matdim(A):
	# Returns the number of rows and columns of A.
	if hasattr(A, "__len__"):
	   m = len(A)
	   if hasattr(A[0], "__len__"):
		  n = len(A[0])
	   else:
		  n = 0
	else:
	   m = 0
	   n = 0
	return (m, n)
def matprod(A, B):
	"""
	Computes the product of two matrices.
	2009.01.16 Revised for matrix or vector B.
	"""
	m, n = matdim(A)
	p, q = matdim(B)
	if n!= p:
	   return None
	try:
	   if iter(B[0]):
		  q = len(B[0])
	except:
	   q = 1
	C = matzero(m, q)
	for i in range(m):
		for j in range(q):
			if q == 1:
			   t = sum([A[i][k] * B[j] for k in range(p)])
			else:
			   t = sum([A[i][k] * B[k][j] for k in range(p)])
			C[i][j] = t
	return C
matmul = matprod
def matvec(A, y):
	"""
	Returns the product of matrix A with vector y.
	"""
	m = len(A)
	n = len(A[0])
	out = [0] * m
	for i in range(m):
		for j in range(n):
			out[i] += A[i][j] * y[j]
	return out
def mattvec(A, y):
	"""
	Returns the vector A^t y.
	"""
	At = transpose(A)
	return matvec(At, y)
def dot(X, Y):
	return sum(x* y for (x,y) in zip(X,Y))
def matdots(X):
	# Added Jan 16, 2009.
	# Returns the matrix of dot products of the column vectors
	# This is the same as X^t X.
	(nrow, ncol) = matdim(X)
	M = [[0.0] * ncol for i in range(ncol)]
	for i in range(ncol):
		for j in range(i+1):
			dot = sum([X[p][i]* X[p][j] for p in range(ncol)])
			M[i][j] = dot
			if i != j:
			   M[j][i] = M[i][j]
	return M
def mattmat(A, B):
	"""
	Returns the product transpose(A) B
	"""
	# print "Inside mattmat:"
	AtB = matprod(transpose(A), A)
	# print "AtA"
	matprint(AtB)
	# print "mattmat: returning now:"
	return AtB
def matrandom(nrow, ncol = None):
	# Added Jan. 16, 2009
	if ncol is None:
	   ncol = nrow
	R = []
	for i in range(nrow):
		R.append([random.random() for j in range(ncol)])
	return R
def matunitize(X, inplace = False):
	# Added jan. 16, 2009
	# Transforms each vector in X to have unit length.
	if not inplace:
	   V = [x[:] for x in X]
	else:
	   V = X
	nrow = len(X)
	ncol = len(X[0])
	for j in range(ncol):
		recipnorm = sum([X[j][j]**2 for j in range(ncol)])
		for i in range(nrow):
			V[i][j] *= recipnorm
	return V
def matprint(A,format= "%8.4f"):
	#prints the matrix A using format
	if hasattr(A, "__len__"):
	  for i,row in enumerate(A):
		try:
		  if iter(row):
			 for c in row:
			   print format % c,
			 print
		except:
		   print row
	else:
		print "Not a matrix!"
	print # prints a blank line after matrix
def mataugprint(A,Y, format= "%8.4f"):
	#prints the augmented matrix A|Y using format
	for i,row in enumerate(A):
		for c in row:
		   print format % c,
		print "|", format % Y[i]
def gjinv(AA,inplace = False):
	"""
	Determines the inverse of a square matrix BB by Gauss-Jordan reduction.
	"""
	n = len(AA)
	B = eye(n)
	if not inplace:
		A = [row[:] for row in AA]
	else:
		A = AA
	for i in range(n):
		#Divide the ith row by A[i][i]
		m = 1.0 / A[i][i]
		for j in range(i, n):
			A[i][j] *= m  # # this is the same as dividing by A[i][i]
		for j in range(n):
			B[i][j] *= m
		#lower triangular elements.
		for k in range(i+1, n):
			m = A[k][i]
			for j in range(i+1, n):
				A[k][j] -= m * A[i][j]
			for j in range(n):
				B[k][j] -= m * B[i][j]
		#upper triangular elements.
		for k in range(0, i):
			m = A[k][i]
			for j in range(i+1, n):
				A[k][j] -= m * A[i][j]
			for j in range(n):
				B[k][j] -= m * B[i][j]
	return B
matinverse = gjinv
def Test():
	X = [1,1,1]
	print dot(X, X)
	AA = [[1,2,3],
		  [4,5,8],
		  [9,7,6]]
	BB = eye(3)
	print "inputs:"
	print AA
	print BB
	print "product"
	print matprod(AA, AA)
	print "inverse of AA:"
	BB = gjinv(AA)
	print BB
	print matprod(AA ,BB)
if __name__ == "__main__":
	Test()