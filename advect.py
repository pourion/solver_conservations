import numpy as np
import matplotlib.pyplot as plt
import pdb

from WENO import WENO

class advect:
	def __init__(self, u, dxy, m, n, BC, tf, box, vel_x, vel_y, X, Y):
		self.u = u
		self.dxy = dxy
		self.m = m
		self.n = n
		self.BC = BC
		self.tf = tf
		self.vel_y = vel_y
		self.vel_x = vel_x
		self.xmax = box[0][1]
		self.xmin = box[0][0]
		self.ymax = box[1][1]
		self.ymin = box[1][0]
		self.X = X
		self.Y = Y
		self.Uf = np.zeros((self.m, self.n))
		for i in range(self.m):
				for j in range(self.n):
					self.Uf[i,j] = self.u[i,j]

	def apply_BC(self):
		if self.BC[0]==0:   # Dirichlet
			for i in range(self.m):
				self.Uf[i, 0]=0
				self.Uf[i,-1] = 0
		if self.BC[1]==0:
			for j in range(self.n):
				self.Uf[ 0,j] = 0
				self.Uf[-1,j] = 0
		
		if self.BC[0]==1:  # Periodic
			for i in range(self.m):
				if self.vel_x[i,0]>=0:
					self.Uf[i, 0]=self.Uf[i,-1]
				else:
					self.Uf[i, -1]=self.Uf[i,0]	
		if self.BC[1]==1:		
			for j in range(self.n):
				if self.vel_y[0,j]>0:
					self.Uf[ 0,j] = self.Uf[-1,j]
				else:
					self.Uf[-1,j] = self.Uf[0,j]

	#############################
	##  ADVECTION EQUATION SOLVER
	#############################
	def advection_WENO_2D(self):
		plt.ion()
		fig = plt.figure(figsize=(7,7))
		im = plt.imshow(self.Uf, extent=[self.xmin,self.xmax,self.ymin,self.ymax], cmap='jet')
		plt.xlabel(r'$x$', fontsize=20)
		plt.ylabel(r'$y$', fontsize=20)
		plt.tight_layout()
		plt.draw()
		t = 0
		dt = 0.5*self.dxy[0]
		while t < self.tf:
			print "time ", t
			if (t+dt>self.tf):
				dt = self.tf - t
			weno = WENO(self.Uf, self.dxy, self.m, self.n, self.BC)
			dxp = weno.Dx_p()
			dxm = weno.Dx_m()
			dym = weno.Dy_m()
			dyp = weno.Dy_p()

			for i in range(self.m):
				for j in range(self.n):
					if self.vel_x[i,j]>=0:
						self.Uf[i,j] -= dt*dxm[i,j]
					elif self.vel_x[i,j]<0:
						self.Uf[i,j] -= dt*dxp[i,j]

					if self.vel_y[i,j]>=0:
						self.Uf[i,j] -= dt*dym[i,j]
					elif self.vel_y[i,j]<0:
						self.Uf[i,j] -= dt*dyp[i,j]
			self.apply_BC()
			t += dt
			im.set_data(self.Uf)
			fig.canvas.draw()
			fig.canvas.flush_events()
		return self.Uf

	
	##############################
	###  CONSERVATION LAWS BELOW!
	##############################
	def conservation_WENO_RIEMANN_TVDRK3_solver_2D(self, riemann):
		plt.ion()
		fig = plt.figure(figsize=(7,7))
		im = plt.imshow(self.Uf, extent=[self.xmin,self.xmax,self.ymin,self.ymax], cmap='jet')
		plt.xlabel(r'$x$', fontsize=20)
		plt.ylabel(r'$y$', fontsize=20)
		plt.tight_layout()
		plt.draw()

		cfl = 0.47
		t = 0
		dt = cfl*self.dxy[0] # np.min((self.dxy[0]/np.max(abs(self.vel_x)), self.dxy[1]/np.max(abs(self.vel_y))))
		counter = 0
		while t < self.tf:
			print "iter ", counter, " time: ", str(t), " conserved mass: ", np.sum(self.Uf)*self.dxy[0]*self.dxy[1]
			if (t+dt>self.tf):
				dt = self.tf - t
			counter += 1
			# CFL condition
			vmax = np.max((np.max(self.vel_x), np.max(self.vel_y)))
			dt = cfl*np.min(self.dxy)/vmax

			# 1. do x-dimension for dt, row by row
			# WENO reconstruction on faces along x dimension
			weno = WENO(self.Uf, self.dxy, self.m, self.n, self.BC)
			ULface_x, URface_x = weno.U_LRx()  # face values of solution
			ULface_y, URface_y = weno.U_LRy()

			weno = WENO(self.vel_x, self.dxy, self.m, self.n, self.BC)
			Vx_Lface_x, Vx_Rface_x = weno.U_LRx()	# face values of Vx
			weno = WENO(self.vel_y, self.dxy, self.m, self.n, self.BC)
			Vy_Lface_x, Vy_Rface_x = weno.U_LRx()   # face values of Vy
			# vx_xface = 0.5*(Vx_Lface_x + Vx_Rface_x) # avg vels on x-faces
			# vy_xface = 0.5*(Vy_Lface_x + Vy_Rface_x)


			if riemann=='MUSTA':
				Fface_x = self.MUSTA(ULface_x, URface_x, Vx_Lface_x, Vx_Rface_x, dt, self.dxy[0])
			elif riemann=='HLL':
				Fface_x = self.HLL(ULface_x, URface_x, Vx_Lface_x, Vx_Rface_x)			
			# weno = WENO(self.vel_x, self.dxy, self.m, self.n, self.BC)
			# Vx_Lface_y, Vx_Rface_y = weno.U_LRy()	# face values of Vx
			weno = WENO(self.vel_y, self.dxy, self.m, self.n, self.BC)
			Vy_Lface_y, Vy_Rface_y = weno.U_LRy()   # face values of Vy
			# vx_yface = 0.5*(Vx_Lface_y + Vx_Rface_y)  # avg vels on y faces
			# vy_yface = 0.5*(Vy_Lface_y + Vy_Rface_y)
			if riemann=='MUSTA':
				Fface_y = self.MUSTA(ULface_y, URface_y, Vy_Lface_y, Vy_Rface_y, dt, self.dxy[1])
			elif riemann=='HLL':
				Fface_y = self.HLL(ULface_y, URface_y, Vy_Lface_y, Vy_Rface_y)

			# TVD_RK3 ...
			Q1 = self.Uf + dt*((Fface_x[:,:-1] - Fface_x[:,1:])/self.dxy[0] + (Fface_y[:-1,:] - Fface_y[1:,:])/self.dxy[1])
			weno = WENO(Q1, self.dxy, self.m, self.n, self.BC)
			ULx, URx = weno.U_LRx()  # ULx[i,j]=UL_{i-1/2,j}, URx[i,j]=UR_{i-1/2,j} 
			if riemann=='MUSTA':
				Fstar_x = self.MUSTA(ULx, URx, Vx_Lface_x, Vx_Rface_x, dt, self.dxy[0])
			elif riemann=='HLL':
				Fstar_x = self.HLL(ULx, URx, Vx_Lface_x, Vx_Rface_x)
			ULy, URy = weno.U_LRy()
			if riemann=='MUSTA':
				Fstar_y = self.MUSTA(ULy, URy, Vy_Lface_y, Vy_Rface_y, dt, self.dxy[1])
			elif riemann=='HLL':
				Fstar_y = self.HLL(ULy, URy, Vy_Lface_y, Vy_Rface_y)
			
			Q2 = 0.75*self.Uf +0.25*Q1 + 0.25*dt*((Fstar_x[:,:-1] - Fstar_x[:,1:])/self.dxy[0] + (Fstar_y[:-1,:] - Fstar_y[1:,:])/self.dxy[1])
			
			weno = WENO(Q2, self.dxy, self.m, self.n, self.BC)
			ULx, URx = weno.U_LRx()  # ULx[i,j]=UL_{i-1/2,j}, URx[i,j]=UR_{i-1/2,j} 
			if riemann=='MUSTA':
				Fstar_x = self.MUSTA(ULx, URx, Vx_Lface_x, Vx_Rface_x, dt, self.dxy[0])
			elif riemann=='HLL':
				Fstar_x = self.HLL(ULx, URx, Vx_Lface_x, Vx_Rface_x)
			ULy, URy = weno.U_LRy()
			if riemann=='MUSTA':
				Fstar_y = self.MUSTA(ULy, URy, Vy_Lface_y, Vy_Rface_y, dt, self.dxy[1])
			elif riemann=='HLL':
				Fstar_y = self.HLL(ULy, URy, Vy_Lface_y, Vy_Rface_y)
			self.Uf = (1./3.)*self.Uf + (2./3.)*Q2 + (2./3.)*dt*((Fstar_x[:,:-1] - Fstar_x[:,1:])/self.dxy[0] + (Fstar_y[:-1,:] - Fstar_y[1:,:])/self.dxy[1])
	
			#self.apply_BC()
			t += dt

			im.set_data(self.Uf)
			fig.canvas.draw()
			fig.canvas.flush_events()
		return self.Uf


	def flux_faces(self, Q, v):
		return v*Q


	def MUSTA(self, QL, QR, VL, VR, dt, dx):
		QL_l = QL  # QL = Q^L{i-1/2}
		QR_l = QR  # QR = Q^R[i-1/2]
		for l in range(4):
			FL_l = self.flux_faces(QL_l, VL) 
			FR_l = self.flux_faces(QR_l, VR)
			# flux evaluation
			QM_l = 0.5*(QL_l + QR_l) - 0.5*(dt/dx)*(FR_l - FL_l)
			FM_l = self.flux_faces(QM_l, 0.5*(VL + VR))
			Fstar_l = 0.25*(FL_l + 2*FM_l + FR_l - (dx/dt)*(QR_l - QL_l)) # Fpstar = F_{i+1/2}
			# open Riemann fan
			QL_l = QL_l - (dt/dx)*(Fstar_l - FL_l)
			QR_l = QR_l - (dt/dx)*(FR_l - Fstar_l)
		return Fstar_l


	def HLL(self, QLs, QRs, vx, vy):
		print 'under construction'
		return 0

	def HLL_cell(self, QL, QR, VL, VR):
		SL = np.min(QL, QR) - np.max(VL, VR)
		SR = np.max(QL, QR) + np.max(VL, VR)
		#Qstar = (SR*UR -SL*UL - (FR - FL))/(SR - SL)
		# define fluxes on faces
		FL = QL*VL
		FR = QR*VR
		if SL>=0:
			return FL
		elif SR>=0:
			return FR
		elif (SL<0 and SR>0):
			return (SR*FL -SL*FR + SL*SR*(QR - QL))/(SR- SL)


	# 2. conservation law with Strang splitting, Godunov fluxes + Euler in time
	def conservation_Burgers_solver_2D(self):
		plt.ion()
		fig = plt.figure(figsize=(7,7))
		im = plt.imshow(self.Uf, extent=[self.xmin,self.xmax,self.ymin,self.ymax], cmap='jet')
		plt.xlabel(r'$x$', fontsize=20)
		plt.ylabel(r'$y$', fontsize=20)
		plt.tight_layout()
		plt.draw()

		cfl = 0.4
		t = 0
		vmax = np.max((self.vel_x**2 + self.vel_y**2)**0.5)
		dt = cfl*np.min(self.dxy)/vmax

		DFx = np.zeros((self.m, self.n))   # Fx_{i+1/2}^n - Fx_{i-1/2}^n
		DFy = np.zeros((self.m, self.n))   # Fy_{i+1/2}^n - Fy_{i-1/2}^n
		while t < self.tf:
			print "time ", t
			if (t+dt>self.tf):
				dt = self.tf - t
			# CFL condition
			vmax = np.max((np.max(self.vel_x), np.max(self.vel_y)))
			dt = cfl*np.min(self.dxy)/vmax
			# 1. do x-dimension for dt, row by row
			for i in range(self.m):
				DFx = self.godunov_1D(self.Uf[i,:]*self.vel_x[i,:]) 
			self.Uf -= (dt/self.dxy[0])*DFx
			self.apply_BC()
			# 2. do y-dimension for dt, row by row
			for j in range(self.n):
				DFy = self.godunov_1D(self.Uf[:,j]*self.vel_y[:,j]) 
			self.Uf -= (dt/self.dxy[1])*DFy
			self.apply_BC()
			t += dt

			im.set_data(self.Uf)
			fig.canvas.draw()
			fig.canvas.flush_events()
			#pdb.set_trace()
		return self.Uf

	def godunov_1D(self, U0):
		length = len(U0)
		DU = np.zeros(length) 
		# Du [i] ---> F_ {i + 1/2} -F_ {i-1/2}    
		# First and last cell -> double the status (free border CC)
		#DU [ 0 ] = - self.fgodunov_1D(U0[ 0 ], U0[ 0 ]) 
		DU [ 0 ] = - self.GodunovFlux_Burgers(U0[ 0 ], U0[ 0 ]) 
		for i in  range( 0 , length - 1 ):
			fIntercell = self.GodunovFlux_Burgers(U0[i], U0[i + 1 ])
			DU[i    ] += fIntercell
			DU[i + 1] -= fIntercell

		DU[length - 1] += self.GodunovFlux_Burgers(U0[length - 1], U0[length - 1])
		return DU
		
	# Riemann solvers for Burgers equation
	def GodunovFlux_Burgers(self, v,w):
		if (v == w) or ((v**2 - w**2)/(v-w) == 0):
			F = 0.5*v**2
		elif ((v**2 - w**2)/(v-w)>0):
			F = 0.5*v**2
		elif (v**2 - w**2)/(v-w)<0:
			F = 0.5*w**2
		return F

	def fgodunov_1D_Burgers(self, ul , ur ):
		if ul >= ur: # Shock
			s = 0.5*(ul + ur)
			if s >= 0 :
				z = ul**2/2
				return z
			else:
				z = ur**2/2
				return z    
		else : # Rare Wave
			if ur < 0:
				z = ur**2/2
				return z
			elif ul<0  and ur>0 :
				z = 0
				return z
			elif ul>0 :
				z = ul**2/2
				return z






def main():
	# time final
	tf = 20.0

	# domain
	n_rows = 100
	n_cols = 100
	xmin = -1
	xmax = 1
	ymin = -1
	ymax = 1
	box = [[xmin, xmax], [ymin, ymax]]
	x = np.linspace(xmin,xmax, n_cols)
	y = np.linspace(ymin,ymax, n_rows)
	X, Y = np.meshgrid(x,y)
	dx = x[1] - x[0]
	dy = y[1] - y[0]
	dxy = [dx, dy]
	BC = [1, 1]			

	# initial condition
	U =  np.exp(-X**2/0.65 - Y**2/0.65) 
	# U = np.zeros((n_rows, n_cols)) 
	# for i in range(n_rows):
	# 	for j in range(n_cols):
	# 		if abs(X[i,j]) + abs(Y[i,j])<0.2:
	# 			U[i,j] = 1.0
	# 		else:
	# 			U[i,j] = 1e-2

	# velocities
	# vel_x = np.ones((n_rows, n_cols))  		
	# vel_y = np.zeros((n_rows, n_cols))
	vel_x = np.sin(np.pi*X)*np.cos(np.pi*Y)
	vel_y = np.sin(np.pi*X)*np.sin(np.pi*Y)



	adv = advect(U, dxy, n_rows, n_cols, BC, tf, box, vel_x, vel_y, X,Y)
	#Uf = adv.advection_WENO_2D()
	#Uf = adv.conservation_Burgers_solver_2D()
	Uf = adv.conservation_WENO_RIEMANN_TVDRK3_solver_2D('MUSTA')

	plt.figure(figsize=(7,7))
	plt.pcolor(X, Y, Uf - U, cmap='jet')
	plt.xlabel('X', fontsize=25)
	plt.ylabel('Y', fontsize=25)
	plt.title(r'$U(t_f) - U(t=0)$', fontsize=25)
	plt.show()
	pdb.set_trace()

# if __name__== "__main__":
# 	main()


