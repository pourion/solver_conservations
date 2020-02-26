import numpy as np
import matplotlib.pyplot as plt
#import pdb

class WENO:
	def __init__(self, u, dxy, m, n, BC):
		self.u = u
		self.dxy = dxy
		self.m = m
		self.n = n
		self.BC = BC

	def U_LRx(self):  # on faces, there are n+1 faces in x-direction
		U_R = np.zeros((self.m, self.n+1))	   # face values 
		U_L = np.zeros((self.m, self.n+1))     # face values
		u_padd = np.zeros((self.m, self.n+6))  # cell-centered values

		for i in range(self.m):
			for j in range(self.n):
				u_padd[i, j+3] = self.u[i, j]

			if self.BC[0]==0:				# 0: linear extrapolation, 1: periodic BC in X direction
				u_padd[i, 2]=2*u_padd[i, 3]-u_padd[i, 4]
				u_padd[i, 1]=2*u_padd[i, 2]-u_padd[i, 3]
				u_padd[i, 0]=2*u_padd[i, 1]-u_padd[i, 2]
				u_padd[i, self.n+3]=2*u_padd[i, self.n+2]-u_padd[i, self.n+1]
				u_padd[i, self.n+4]=2*u_padd[i, self.n+3]-u_padd[i, self.n+2]
				u_padd[i, self.n+5]=2*u_padd[i, self.n+4]-u_padd[i, self.n+3]
			elif self.BC[0]==1:
				u_padd[i, 2]=self.u[i, self.n-1]
				u_padd[i, 1]=self.u[i, self.n-2]
				u_padd[i, 0]=self.u[i, self.n-3]
				u_padd[i, self.n+3]=self.u[i, 0]
				u_padd[i, self.n+4]=self.u[i, 1]
				u_padd[i, self.n+5]=self.u[i, 2]

		for i in range(self.m):
			for j in range(3,self.n+4):
				beta0 = (13./12.)*(u_padd[i, j] - 2*u_padd[i, j+1] + u_padd[i, j+2])**2 + 0.25*(3*u_padd[i,j] - 4*u_padd[i, j+1] + u_padd[i, j+2])**2
				beta1 = (13./12.)*(u_padd[i, j-1] - 2*u_padd[i,j] + u_padd[i,j+1])**2 + 0.25*(u_padd[i,j-1] - u_padd[i,j+1])**2
				beta2 = (13./12.)*(u_padd[i, j-2] - 2*u_padd[i,j-1] + u_padd[i,j])**2 + 0.25*(u_padd[i,j-2] - 4*u_padd[i,j-1] + 3*u_padd[i,j])**2

				d0 = 3./10.
				d1 = 3./5.
				d2 = 1./10.
				alpha0 = d0/(beta0 + 1e-6)**2
				alpha1 = d1/(beta1 + 1e-6)**2
				alpha2 = d2/(beta2 + 1e-6)**2
				w0 = alpha0/(alpha0 + alpha1 + alpha2)
				w1 = alpha1/(alpha0 + alpha1 + alpha2)
				w2 = 1 - w0 - w1
				U_L[i,j-3] = (w0/6.0)*( -u_padd[i,j+1] + 5*u_padd[i,j] +  2*u_padd[i,j-1]) + (w1/6.0)*(-u_padd[i,j-2]+5*u_padd[i,j-1]+2*u_padd[i,j]) + (w2/6.0)*(2*u_padd[i,j-3] - 7*u_padd[i,j-2]+11*u_padd[i,j-1])

				d0 = 1./10.
				d1 = 3./5.
				d2 = 3./10.
				alpha0 = d0/(beta0 + 1e-6)**2
				alpha1 = d1/(beta1 + 1e-6)**2
				alpha2 = d2/(beta2 + 1e-6)**2
				w0 = alpha0/(alpha0 + alpha1 + alpha2)
				w1 = alpha1/(alpha0 + alpha1 + alpha2)
				w2 = 1 - w0 - w1
				U_R[i,j-3] = (w0/6.0)*(2*u_padd[i,j+2] - 7*u_padd[i,j+1] + 11*u_padd[i,j]) + (w1/6.0)*(-u_padd[i,j+1]+5*u_padd[i,j]+2*u_padd[i,j-1]) + (w2/6.0)*(-u_padd[i,j-2] + 5*u_padd[i,j-1] + 2*u_padd[i,j])

		return U_L, U_R	


	

	def U_LRy(self):  # there are m+1 faces in y-direction
		U_R = np.zeros((self.m+1, self.n))
		U_L = np.zeros((self.m+1, self.n))
		u_padd = np.zeros((self.m+6, self.n))

		for j in range(self.n):
			for i in range(self.m):
				u_padd[i+3, j] = self.u[i, j]

			if self.BC[1]==0:				# 0: linear extrapolation, 1: periodic BC in X direction
				u_padd[2, j]=2*u_padd[3, j]-u_padd[4, j]
				u_padd[1, j]=2*u_padd[2, j]-u_padd[3, j]
				u_padd[0, j]=2*u_padd[1, j]-u_padd[2, j]
				u_padd[self.m+3, j]=2*u_padd[self.m+2, j]-u_padd[self.m+1, j]
				u_padd[self.m+4, j]=2*u_padd[self.m+3, j]-u_padd[self.m+2, j]
				u_padd[self.m+5, j]=2*u_padd[self.m+4, j]-u_padd[self.m+3, j]
			elif self.BC[1]==1:
				u_padd[2, j]=self.u[self.m-1, j]
				u_padd[1, j]=self.u[self.m-2, j]
				u_padd[0, j]=self.u[self.m-3, j]
				u_padd[self.m+3, j]=self.u[0, j]
				u_padd[self.m+4, j]=self.u[1, j]
				u_padd[self.m+5, j]=self.u[2, j]

		for j in range(self.n):
			for i in range(3,self.m+4):
				
				beta0 = (13./12.)*(u_padd[i  , j] - 2*u_padd[i+1, j] + u_padd[i+2, j])**2 + 0.25*(3*u_padd[i,j] - 4*u_padd[i+1, j] + u_padd[i+2, j])**2
				beta1 = (13./12.)*(u_padd[i-1, j] - 2*u_padd[i  , j] + u_padd[i+1,j])**2 + 0.25*(u_padd[i-1,j] - u_padd[i+1,j])**2
				beta2 = (13./12.)*(u_padd[i-2, j] - 2*u_padd[i-1, j] + u_padd[i,j])**2 + 0.25*(u_padd[i-2,j] - 4*u_padd[i-1,j] + 3*u_padd[i,j])**2
				
				d0 = 3./10.
				d1 = 3./5.
				d2 = 1./10.
				alpha0 = d0/(beta0 + 1e-6)**2
				alpha1 = d1/(beta1 + 1e-6)**2
				alpha2 = d2/(beta2 + 1e-6)**2
				w0 = alpha0/(alpha0 + alpha1 + alpha2)
				w1 = alpha1/(alpha0 + alpha1 + alpha2)
				w2 = 1 - w0 - w1
				U_L[i-3,j] = (w0/6.0)*( -u_padd[i+1,j] + 5*u_padd[i,j] +  2*u_padd[i-1,j]) + (w1/6.0)*(-u_padd[i-2,j]+5*u_padd[i-1,j]+2*u_padd[i,j]) + (w2/6.0)*(2*u_padd[i-3,j] - 7*u_padd[i-2,j]+11*u_padd[i-1,j])

				d0 = 1./10.
				d1 = 3./5.
				d2 = 3./10.
				alpha0 = d0/(beta0 + 1e-6)**2
				alpha1 = d1/(beta1 + 1e-6)**2
				alpha2 = d2/(beta2 + 1e-6)**2
				w0 = alpha0/(alpha0 + alpha1 + alpha2)
				w1 = alpha1/(alpha0 + alpha1 + alpha2)
				w2 = 1 - w0 - w1
				U_R[i-3,j] = (w0/6.0)*(2*u_padd[i+2,j] - 7*u_padd[i+1,j] + 11*u_padd[i,j]) + (w1/6.0)*(-u_padd[i+1,j]+5*u_padd[i,j]+2*u_padd[i-1,j]) + (w2/6.0)*(-u_padd[i-2,j] + 5*u_padd[i-1,j] + 2*u_padd[i,j])

		return U_L, U_R



	def update_u(self, u):
		self.u = u

	def Dx_m(self):
		Dx_m = np.zeros((self.m, self.n))
		u_padd = np.zeros((self.m, self.n+5))

		for i in range(self.m):
			for j in range(self.n):
				u_padd[i, j+3] = self.u[i, j]

			if self.BC[0]==0:				# 0: linear extrapolation, 1: periodic BC in X direction
				u_padd[i, 2]=2*u_padd[i, 3]-u_padd[i, 4]
				u_padd[i, 1]=2*u_padd[i, 2]-u_padd[i, 3]
				u_padd[i, 0]=2*u_padd[i, 1]-u_padd[i, 2]
				u_padd[i, self.n+3]=2*u_padd[i, self.n+2]-u_padd[i, self.n+1]
				u_padd[i, self.n+4]=2*u_padd[i, self.n+3]-u_padd[i, self.n+2]
			elif self.BC[0]==1:
				u_padd[i, 2]=self.u[i, self.n-1]
				u_padd[i, 1]=self.u[i, self.n-2]
				u_padd[i, 0]=self.u[i, self.n-3]
				u_padd[i, self.n+3]=self.u[i, 0]
				u_padd[i, self.n+4]=self.u[i, 1]

		for i in range(self.m):
			for j in range(3,self.n+3):
				# WENO formula:
				d1=(u_padd[i, j-2]-u_padd[i, j-3])/self.dxy[0]
				d2=(u_padd[i, j-1]-u_padd[i, j-2])/self.dxy[0]
				d3=(u_padd[i, j  ]-u_padd[i, j-1])/self.dxy[0]
				d4=(u_padd[i, j+1]-u_padd[i, j  ])/self.dxy[0]
				d5=(u_padd[i, j+2]-u_padd[i, j+1])/self.dxy[0]

				S1=(13./12)*(d1-2*d2+d3)**2 + .25*(d1-4*d2+3*d3)**2
				S2=(13./12)*(d2-2*d3+d4)**2 + .25*(d2-d4)**2
				S3=(13./12)*(d3-2*d4+d5)**2 + .25*(3*d3-4*d4+d5)**2
	
				Dsquare=[d1**2, d2**2, d3**2, d4**2, d5**2]
				epsilon=1e-6*np.max(Dsquare) + 1e-99
	
				alpha1=.1/((S1+epsilon)**2)
				alpha2=.6/((S2+epsilon)**2)
				alpha3=.3/((S3+epsilon)**2)
	
				Sum=alpha1+alpha2+alpha3
				omega1=alpha1/Sum
				omega2=alpha2/Sum
				omega3=1-omega1-omega2
	
				ux1=  d1/3. - 7.*d2/6. + 11.0*d3/6.
				ux2= -d2/6. + 5.*d3/6. + d4/3.
				ux3=  d3/3. + 5.*d4/6. - d5/6.
	
				Dx_m[i, j-3]=omega1*ux1 + omega2*ux2 + omega3*ux3
		return Dx_m


	def Dx_p(self):
		Dx_p = np.zeros((self.m, self.n))
		u_padd = np.zeros((self.m, self.n+5))

		for i in range(self.m):
			for j in range(self.n):
				u_padd[i, j+2] = self.u[i, j]

			if self.BC[0]==0:				# 0: linear extrapolation, 1: periodic BC in X direction
				u_padd[i, 1]=2*u_padd[i, 2]-u_padd[i, 3]
				u_padd[i, 0]=2*u_padd[i, 1]-u_padd[i, 2]
				u_padd[i, self.n+2]=2*u_padd[i, self.n+1]-u_padd[i, self.n]
				u_padd[i, self.n+3]=2*u_padd[i, self.n+2]-u_padd[i, self.n+1]
				u_padd[i, self.n+4]=2*u_padd[i, self.n+3]-u_padd[i, self.n+2]
			elif self.BC[0]==1:
				u_padd[i, 1]=self.u[i, self.n-1]
				u_padd[i, 0]=self.u[i, self.n-2]
				u_padd[i, self.n+2]=self.u[i, 2]
				u_padd[i, self.n+3]=self.u[i, 3]
				u_padd[i, self.n+4]=self.u[i, 4]

		for i in range(self.m):
			for j in range(2,self.n+2):
				# WENO formula:
				d1=(u_padd[i, j-1] - u_padd[i, j-2])/self.dxy[0]
				d2=(u_padd[i, j  ] - u_padd[i, j-1])/self.dxy[0]
				d3=(u_padd[i, j+1] - u_padd[i, j  ])/self.dxy[0]
				d4=(u_padd[i, j+2] - u_padd[i, j+1])/self.dxy[0]
				d5=(u_padd[i, j+3] - u_padd[i, j+2])/self.dxy[0]

				S1=(13./12)*(d1-2*d2+d3)**2 + .25*(d1-4*d2+3*d3)**2
				S2=(13./12)*(d2-2*d3+d4)**2 + .25*(d2-d4)**2
				S3=(13./12)*(d3-2*d4+d5)**2 + .25*(3*d3-4*d4+d5)**2
	
				Dsquare=[d1**2, d2**2, d3**2, d4**2, d5**2]
				epsilon=1e-6*np.max(Dsquare) + 1e-99
	
				alpha1=.1/((S1+epsilon)**2)
				alpha2=.6/((S2+epsilon)**2)
				alpha3=.3/((S3+epsilon)**2)
	
				Sum=alpha1+alpha2+alpha3
				omega1=alpha1/Sum
				omega2=alpha2/Sum
				omega3=1-omega1-omega2
	
				ux1=  d1/3. - 7.*d2/6. + 11.0*d3/6.
				ux2= -d2/6. + 5.*d3/6. + d4/3.
				ux3=  d3/3. + 5.*d4/6. - d5/6.
	
				Dx_p[i, j-2]=omega1*ux1 + omega2*ux2 + omega3*ux3
		return Dx_p


	def Dy_m(self):
		Dy_m = np.zeros((self.m, self.n))
		u_padd = np.zeros((self.m+5, self.n))

		for j in range(self.n):
			for i in range(self.m):
				u_padd[i+3, j] = self.u[i, j]

			if self.BC[1]==0:				# 0: linear extrapolation, 1: periodic BC in X direction
				u_padd[2, j]=2*u_padd[3, j]-u_padd[4, j]
				u_padd[1, j]=2*u_padd[2, j]-u_padd[3, j]
				u_padd[0, j]=2*u_padd[1, j]-u_padd[2, j]
				u_padd[self.m+3, j]=2*u_padd[self.m+2, j]-u_padd[self.m+1, j]
				u_padd[self.m+4, j]=2*u_padd[self.m+3, j]-u_padd[self.m+2, j]
			elif self.BC[1]==1:
				u_padd[2, j]=self.u[self.m-1, j]
				u_padd[1, j]=self.u[self.m-2, j]
				u_padd[0, j]=self.u[self.m-3, j]
				u_padd[self.m+3, j]=self.u[0, j]
				u_padd[self.m+4, j]=self.u[1, j]

		for j in range(self.n):
			for i in range(3,self.m+3):
				# WENO formula:
				d1=(u_padd[i-2, j]-u_padd[i-3, j])/self.dxy[1]
				d2=(u_padd[i-1, j]-u_padd[i-2, j])/self.dxy[1]
				d3=(u_padd[i, j  ]-u_padd[i-1, j])/self.dxy[1]
				d4=(u_padd[i+1, j]-u_padd[i, j  ])/self.dxy[1]
				d5=(u_padd[i+2, j]-u_padd[i+1, j])/self.dxy[1]

				S1=(13./12)*(d1-2*d2+d3)**2 + .25*(d1-4*d2+3*d3)**2
				S2=(13./12)*(d2-2*d3+d4)**2 + .25*(d2-d4)**2
				S3=(13./12)*(d3-2*d4+d5)**2 + .25*(3*d3-4*d4+d5)**2
	
				Dsquare=[d1**2, d2**2, d3**2, d4**2, d5**2]
				epsilon=1e-6*np.max(Dsquare) + 1e-99
	
				alpha1=.1/((S1+epsilon)**2)
				alpha2=.6/((S2+epsilon)**2)
				alpha3=.3/((S3+epsilon)**2)
	
				Sum=alpha1+alpha2+alpha3
				omega1=alpha1/Sum
				omega2=alpha2/Sum
				omega3=1-omega1-omega2
	
				uy1=  d1/3. - 7.*d2/6. + 11.0*d3/6.
				uy2= -d2/6. + 5.*d3/6. + d4/3.
				uy3=  d3/3. + 5.*d4/6. - d5/6.
	
				Dy_m[i-3, j]=omega1*uy1 + omega2*uy2 + omega3*uy3
		return Dy_m


	def Dy_p(self):
		Dy_p = np.zeros((self.m, self.n))
		u_padd = np.zeros((self.m+5, self.n))

		for j in range(self.n):
			for i in range(self.m):
				u_padd[i+2, j] = self.u[i, j]

			if self.BC[1]==0:				# 0: linear extrapolation, 1: periodic BC in X direction
				u_padd[1, j]=2*u_padd[2, j]-u_padd[3, j]
				u_padd[0, j]=2*u_padd[1, j]-u_padd[2, j]
				u_padd[self.m+2, j]=2*u_padd[self.m+1, j]-u_padd[self.m  , j]
				u_padd[self.m+3, j]=2*u_padd[self.m+2, j]-u_padd[self.m+1, j]
				u_padd[self.m+4, j]=2*u_padd[self.m+3, j]-u_padd[self.m+2, j]
			elif self.BC[1]==1:
				u_padd[1, j]=self.u[self.m-1, j]
				u_padd[0, j]=self.u[self.m-2, j]
				u_padd[self.m+2, j]=self.u[2, j]
				u_padd[self.m+3, j]=self.u[3, j]
				u_padd[self.m+4, j]=self.u[4, j]

		for j in range(self.n):
			for i in range(2,self.m+2):
				# WENO formula:
				d1=(u_padd[i-1, j] - u_padd[i-2, j])/self.dxy[1]
				d2=(u_padd[i,   j] - u_padd[i-1, j])/self.dxy[1]
				d3=(u_padd[i+1, j] - u_padd[i  , j])/self.dxy[1]
				d4=(u_padd[i+2, j] - u_padd[i+1, j])/self.dxy[1]
				d5=(u_padd[i+3, j] - u_padd[i+2, j])/self.dxy[1]

				S1=(13./12)*(d1-2*d2+d3)**2 + .25*(d1-4*d2+3*d3)**2
				S2=(13./12)*(d2-2*d3+d4)**2 + .25*(d2-d4)**2
				S3=(13./12)*(d3-2*d4+d5)**2 + .25*(3*d3-4*d4+d5)**2
	
				Dsquare=[d1**2, d2**2, d3**2, d4**2, d5**2]
				epsilon=1e-6*np.max(Dsquare) + 1e-99
	
				alpha1=.1/((S1+epsilon)**2)
				alpha2=.6/((S2+epsilon)**2)
				alpha3=.3/((S3+epsilon)**2)
	
				Sum=alpha1+alpha2+alpha3
				omega1=alpha1/Sum
				omega2=alpha2/Sum
				omega3=1-omega1-omega2
	
				uy1=  d1/3. - 7.*d2/6. + 11.0*d3/6.
				uy2= -d2/6. + 5.*d3/6. + d4/3.
				uy3=  d3/3. + 5.*d4/6. - d5/6.
	
				Dy_p[i-2, j]=omega1*uy1 + omega2*uy2 + omega3*uy3
		return Dy_p





def main():
	n_rows = 50
	n_cols = 100
	Lx = 2
	Ly = 2
	x = np.linspace(-1,1, n_cols)
	y = np.linspace(-1,1, n_rows)
	X, Y = np.meshgrid(x,y)
	dx = x[1] - x[0]
	dy = y[1] - y[0]
	dxy = [dx, dy]
	BC = [0, 0]			# 0: ..., 1: periodic, first entry is x direction, 2nd is Y-dir.
	
	U =  np.exp(-X**2/0.5 - Y**2/0.5) 
	weno = WENO(U, dxy, n_rows, n_cols, BC)
	dxp = weno.Dx_p()

	plt.figure(figsize=(7,7))
	plt.pcolor(X, Y, U, cmap='jet')
	plt.show()

	plt.figure(figsize=(7,7))
	plt.pcolor(X,Y,dxp, cmap='jet')
	plt.show()
  
if __name__== "__main__":
	main()

