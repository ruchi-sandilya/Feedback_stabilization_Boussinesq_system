from dolfin import *
from bmark import *
import numpy as np
import scipy.sparse as sps
import scipy.io as sio
import scipy.sparse.linalg as la

# Position of some boundary zones
x1,x2,y1,y2 = 0.4,0.6,0.1,0.4

#-----------------------------------------------------------------------------
def g(s):
    if s < -1.0:
        return 0.0
    elif s > 1.0:
        return 1.0
    else:
        return 0.5 + s*(0.9375 - s*s*(0.625 - 0.1875*s*s))
#-----------------------------------------------------------------------------
# Smooth ramp from 0 to 1; middle of ramp is at T, width dt
# Put large negative value to disable ramp
def f(t):
    T, dt = -20.0, 2.0
    t1 = (t - T)/dt
    return g(t1)

#-----------------------------------------------------------------------------
class UnitNormal(UserExpression):
   def eval(self, value, x):
       if near(x[0],0.0):
           value[0] = -1.0
           value[1] =  0.0
       elif near(x[0],1.0):
           value[0] =  1.0
           value[1] =  0.0
       elif near(x[1],0.0):
           value[0] =  0.0
           value[1] = -1.0
       elif near(x[1],1.0):
           value[0] =  0.0
           value[1] =  1.0
       else:
           value[0] =  0.0
           value[1] =  0.0
       return value
   def value_shape(self):
      return (2,)
#-----------------------------------------------------------------------------
class HeatSource(UserExpression):
   def eval(self, value, x):
      value[0] = 7.0*sin(2.0*pi*x[0])*cos(2.0*pi*x[1])
      return value
#-----------------------------------------------------------------------------
class SourcePerturbation(UserExpression):
   def eval(self, value, x):
      value[0] = sin(2.0*pi*x[0])*cos(2.0*pi*x[1])
      return value
#------------------------------------------------------------------------------
class HeatFlux(UserExpression):
   def __init__(self,amp, **kwargs):
      super().__init__(**kwargs)
      self.amp = amp
   def eval(self, value, x):
      ff = ((x[0]-x1)*(x[0]-x2))**2
      if ff < DOLFIN_EPS:
         value[0] = 0.0
      elif x[0]>x1 and x[0]<x2:
         value[0] = 0.4 * exp(-0.00001/ff)
      else:
         value[0] = 0.0
      #value[0] *= self.amp
      return value
#-----------------------------------------------------------------------------
# Velocity at inflow boundary
class velocity(UserExpression):
   def __init__(self, amp,  y3, y4, **kwargs):
      super().__init__(**kwargs)      
      self.amp = amp
      self.y3  = y3
      self.y4  = y4
   def eval(self, value, x):
      ff = ((x[1]-self.y3)*(x[1]-self.y4))**2
      if ff < DOLFIN_EPS:
         value[0] = 0.0
         value[1] = 0.0
      elif x[1]>self.y3 and x[1]<self.y4:
         value[0] = exp(-0.0001/ff)
         value[1] = 0.0
      else:
         value[0] = 0.0
         value[1] = 0.0
      value[0] *= self.amp
      value[1] = 0.0
      return value
   def value_shape(self):
      return(2,)
#-----------------------------------------------------------------------------
# temperature at inflow boundary
class temperature(UserExpression):
   def __init__(self, amp, y3, y4, **kwargs):
      super().__init__(**kwargs)
      self.amp = amp
      self.y3  = y3
      self.y4  = y4
   def eval(self, value, x):
      ff = ((x[1]-self.y3)*(x[1]-self.y4))**2
      if ff < DOLFIN_EPS:
         value[0] = 0.0
      elif x[1]>self.y3 and x[1]<self.y4:
         value[0] = 0.2 * exp(-0.00001/ff)
      else:
         value[0] = 0.0
      value[0] *= self.amp
      return value
#-----------------------------------------------------------------------------
# all variables at inflow boundary
# we only need x velocity and temperature at inflow boundary
# which is the control boundary
class allvar(UserExpression):
   def __init__(self, vamp, tamp, y3, y4, **kwargs):
      super().__init__(**kwargs)
      self.vamp = vamp
      self.tamp = tamp
      self.y3   = y3
      self.y4   = y4
   def eval(self, value, x):
      ff = ((x[1]-self.y3)*(x[1]-self.y4))**2
      if ff < DOLFIN_EPS:
         value[0] = 0.0
         value[2] = 0.0
      elif x[1]>self.y3 and x[1]<self.y4:
         value[0] = exp(-0.0001/ff)
         value[2] = 0.2 * exp(-0.00001/ff)
      else:
         value[0] = 0.0
         value[2] = 0.0
      value[0] *= self.vamp  # x velocity
      value[1] = 0.0         # y velocity
      value[2] *= self.tamp  # temperature
      value[3] = 0.0         # pressure
      return value
   def value_shape(self):
      return (4,)


#-----------------------------------------------------------------------------
# Returns symmetric part of strain tensor
def epsilon(u):
   return 0.5*(nabla_grad(u) + nabla_grad(u).T)
#-----------------------------------------------------------------------------
def nonlinear_form(Re,Gr,Pr,hf,ds,up,vp):
   nu    = 1/Re
   k     = 1/(Re*Pr)
   G     = Gr/Re**2
   hs    = HeatSource(degree=2)
   # Trial function
   u  = as_vector((up[0],up[1]))   # velocity
   T  = up[2]                      # temperature
   p  = up[3]                      # pressure
   tau= 2*nu*epsilon(u)
   # Test function
   v  = as_vector((vp[0],vp[1]))   # velocity
   S  = vp[2]                      # temperature
   q  = vp[3]                      # pressure
   Fns   =   inner(grad(u)*u, v)*dx      \
          + inner(tau,epsilon(v))*dx     \
          - div(v)*p*dx                  \
          - G*T*v[1]*dx
   Ftemp =   inner(grad(T),u)*S*dx       \
           + k*inner(grad(T),grad(S))*dx \
           - hs*S*dx                     \
           - hf*S*ds(3)
   Fdiv  =  - q*div(u)*dx
   return Fns + Ftemp + Fdiv

#-----------------------------------------------------------------------------
# contribution of heat flux term is not included here
def linear_form(Re,Gr,Pr,us,Ts,u,T,p,v,S,q):
   nu    = 1/Re
   k     = 1/(Re*Pr)
   G     = Gr/Re**2
   tau   = 2*nu*epsilon(u)
   Aform = - inner( grad(u)*us, v )*dx \
           - inner( grad(us)*u, v )*dx \
           + div(v)*p*dx \
           - inner( tau, epsilon(v) )*dx \
           + G*T*v[1]*dx \
           + div(u)*q*dx \
           - inner(grad(Ts),u)*S*dx \
           - inner(grad(T),us)*S*dx \
           - k*inner(grad(T),grad(S))*dx
   return Aform
#-----------------------------------------------------------------------------
class NSProblem():
   def __init__(self, udeg, Re, Gr, Pr, y3, y4):
      mesh = Mesh("square.xml")
      sub_domains = create_subdomains(mesh,x1,x2,y1,y2,y3,y4)
      self.udeg = udeg
      self.tdeg = udeg
      self.pdeg = udeg - 1
      self.Re = Re
      self.Gr = Gr
      self.Pr = Pr
      self.ds = Measure("ds",domain=mesh,subdomain_data=sub_domains)
      self.y3 = y3
      self.y4 = y4

      self.V = VectorFunctionSpace(mesh, "CG", self.udeg)  # velocity
      self.W = FunctionSpace(mesh, "CG", self.tdeg)        # temperature
      self.Q = FunctionSpace(mesh, "CG", self.pdeg)        # pressure
      Ve = VectorElement("CG", mesh.ufl_cell(), self.udeg)
      We = FiniteElement("CG", mesh.ufl_cell(), self.tdeg)
      Qe = FiniteElement("CG", mesh.ufl_cell(), self.pdeg)
      self.X = FunctionSpace(mesh, MixedElement([Ve,We,Qe]))

      print ('Number of degrees of freedom = ', self.X.dim())

      # Velocity bc
      noslipbc1 = DirichletBC(self.X.sub(0), (0,0), sub_domains, 0)
      noslipbc2 = DirichletBC(self.X.sub(0), (0,0), sub_domains, 3)
      # velocity control boundary
      self.gs   = velocity(0.0,self.y3,self.y4,degree=self.udeg)
      vconbc    = DirichletBC(self.X.sub(0), self.gs, sub_domains, 2)
      # Temperature bc
      tbc1    = DirichletBC(self.X.sub(1), 0.0, sub_domains, 0)
      self.ts = temperature(0.0,self.y3,self.y4,degree=self.tdeg)
      tbc2    = DirichletBC(self.X.sub(1), self.ts, sub_domains, 2)

      # All dirichlet bc
      self.bc  = [noslipbc1, noslipbc2, vconbc, tbc1, tbc2]
      # Control dirichlet bc, inlet
      self.bc2 = [vconbc, tbc2]
      # Homogeneous dirichlet bc
      self.bc1 = [noslipbc1, noslipbc2, tbc1]
      self.vbc = vconbc
      self.tbc = tbc2
      self.hf  = HeatFlux(0.0,degree=self.tdeg)

   # solves steady state equations
   def steady_state(self, Relist):
      up = Function(self.X)
      vp = TestFunction(self.X)
      Re = Constant(self.Re)
      Gr = Constant(self.Gr)
      Pr = Constant(self.Pr)
      F = nonlinear_form(Re,Gr,Pr,self.hf,self.ds,up,vp)

      dup = TrialFunction(self.X)
      dF  = derivative(F, up, dup)
      problem = NonlinearVariationalProblem(F, up, self.bc, dF)
      solver  = NonlinearVariationalSolver(problem)
      #solver.parameters['newton_solver']['linear_solver'] = 'gmres'
      #solver.parameters['newton_solver']['absolute_tolerance'] = 1.0e-2
      #solver.parameters['newton_solver']['relative_tolerance'] = 1.0e-1
      #info(solver.parameters, True)

      for R in Relist:
         print ("-----------------------------------------------------")
         print ("Reynolds number = ", R) 
         print ("-----------------------------------------------------")
         Re.assign(R)
         solver.solve()
         u = as_vector((up[0],up[1]))
         d = assemble(div(u)*div(u)*dx)
         print ("Divergence L2 norm = ", np.sqrt(d))

      # Save FE solution
      print ("Saving FE solution into steady.xml")
      File("steady.xml") << up.vector()
      # Save vtk format
      u,T,p = up.split()
      u.rename("v","velocity"); T.rename("T","temperature"); p.rename("p","pressure");
      print ("Saving vtk files steady_u.pvd, steady_p.pvd, steady_T.pvd")
      File("steady_u.pvd") << u
      File("steady_p.pvd") << p
      File("steady_T.pvd") << T
   
   # Returns dof indices which are free, used for nonlinear solve
   # freeinds = free indices of velocity, temperature, pressure
   # pinds    = free indices of pressure
   def get_indices(self):
      # Get all dofs other than dirichlet. This is different from freeinds
      # given by get_indices_lagrange
      bcinds = []
      for b in self.bc:
         bcdict = b.get_boundary_values()
         bcinds.extend(bcdict.keys())

      # total number of dofs
      N = self.X.dim()

      # indices of free nodes
      freeinds = np.setdiff1d(range(N),bcinds,assume_unique=True).astype(np.int32)

      # pressure indices
      pinds = self.X.sub(2).dofmap().dofs()

      return freeinds, pinds

   # Returns dof indices which are free, using for linear system only
   # freeinds = free indices of velocity, temperature, pressure
   # pinds    = free indices of pressure
   def get_indices_lagrange(self):
      # Collect all dirichlet boundary dof indices
      bcinds = []
      for b in self.bc:
         bcdict = b.get_boundary_values()
         bcinds.extend(bcdict.keys())

      # Homogeneous dirichlet dofs
      bcinds1 = []
      for b1 in self.bc1:
         bcdict1 = b1.get_boundary_values()
         bcinds1.extend(bcdict1.keys())

      # Control dirichlet dofs
      bcinds2 = []
      for b2 in self.bc2:
         bcdict2 = b2.get_boundary_values()
         bcinds2.extend(bcdict2.keys())

      # bcinds1 must not contain dofs on control region ends since they
      # should be free dofs, with corresponding Lagrange multiplier.
      # Hence do bcinds1 = bcinds1 - bcinds2
      bcinds1 = np.setdiff1d(bcinds1,bcinds2,assume_unique=True).astype(np.int32)

      # total number of dofs
      N = self.X.dim()

      # indices of free nodes: everything other than homogenerous dofs
      freeinds = np.setdiff1d(range(N),bcinds1,assume_unique=True).astype(np.int32)

      # pressure indices
      pinds = self.X.sub(2).dofmap().dofs()

      return freeinds, pinds, bcinds2


###################################################################

# Generate linear state representation
   # Also compute k eigenvalues/vectors
   def linear_system(self):
      parameters['linear_algebra_backend'] = 'Eigen'
       
      # Load Stationary solution from file
      ups = Function(self.X)
      File("steady.xml") >> ups.vector()
      us, Ts = as_vector((ups[0],ups[1])), ups[2]

      u,T,p = TrialFunctions(self.X)
      v,S,q = TestFunctions(self.X)

      # Mass matrix
      Ma = assemble(inner(u,v)*dx + T*S*dx)
      Ma = as_backend_type(Ma).sparray()

      Re = Constant(self.Re)
      Gr = Constant(self.Gr)
      Pr = Constant(self.Pr)

      Aform = linear_form(Re,Gr,Pr,us,Ts,u,T,p,v,S,q)
      Aa = assemble(Aform)

      # Convert to sparse format
      Aa = as_backend_type(Aa).sparray()

      # Boundary matrices
      Avs1 = assemble(inner(u,v)*self.ds(2))
      Avs1 = as_backend_type(Avs1).sparray()

      Ats2 = assemble(T*S*self.ds(2))
      Ats2 = as_backend_type(Ats2).sparray()

      As = assemble(inner(u,v)*self.ds(2)+T*S*self.ds(2))
      As = as_backend_type(As).sparray()

      freeinds,pinds,bcinds2 = self.get_indices_lagrange()

      print ("Writing free indices into freeinds.txt")
      f = open('freeinds.txt','w')
      for item in freeinds:
          f.write("%d\n" % item)
      f.close()

      print ("Writing pressure indices into pinds.txt")
      f = open('pinds.txt','w')
      for item in pinds:
          f.write("%d\n" % item)
      f.close()

      print ("Writing control boundary indices into bcinds2.txt")
      f = open('bcinds2.txt','w')
      for item in bcinds2:
          f.write("%d\n" % item)
      f.close()

      # indices of velocity control
      vinds = list(self.vbc.get_boundary_values().keys())
      # indices of temperature control
      tinds = list(self.tbc.get_boundary_values().keys())

      print ('size of pinds =', len(pinds))
      print ('size of vinds =', len(vinds))
      print ('size of tinds =', len(tinds))
      
      # mass matrix
      Ma   = Ma  [freeinds,:][:,freeinds]
      Aa   = Aa  [freeinds,:][:,freeinds]
      As   = As  [freeinds,:][:,bcinds2]
      Avs1 = Avs1 [freeinds,:][:,vinds]
      Ats2 = Ats2 [freeinds,:][:,tinds]

      print ("Size of Ma   = ",Ma.shape)
      print ("Size of Aa   = ",Aa.shape)
      print ("Size of As   = ",As.shape)
      print ("Size of Avs1 = ",Avs1.shape)
      print ("Size of Ats2 = ",Ats2.shape)

      # velocity control operator
      alpha = velocity(1.0,self.y3,self.y4,degree=self.udeg)
      Bv = assemble(inner(alpha,v)*self.ds(2)).get_local()
      Bv = Bv[vinds]
      print ("Size of Bv =",Bv.shape[0])
      # temperature control operator
      beta = temperature(1.0,self.y3,self.y4,degree=self.tdeg)
      Bt = assemble(beta*S*self.ds(2)).get_local()
      Bt = Bt[tinds]
      print ("Size of Bt =",Bt.shape[0])

      # heat flux control operator
      gamma = HeatFlux(1.0,degree=self.tdeg)
      Bh = assemble(gamma*S*self.ds(3)).get_local()
      Bh = Bh[freeinds]
      print ("Size of Bh =",Bh.shape[0])

      # Save matrices in matlab format
      print ("Saving linear system into linear.mat")
      sio.savemat('linear.mat', 
              mdict={'Ma':Ma,'Aa':Aa,'Avs1':Avs1,'Ats2':Ats2,'Bv':Bv,'Bt':Bt,'Bh':Bh}, 
              oned_as='column')

   ##### Compute most unstable eigenvectors ##########
   # Compute eigenvectors of linearized NS around steady state solution
   def compute_eigenvectors(self, k):
      parameters['linear_algebra_backend'] = 'Eigen'
       
      # Load Stationary solution from file
      ups = Function(self.X)
      File("steady.xml") >> ups.vector()
      us= as_vector((ups[0],ups[1]));
      Ts= ups[2]

      u,T,p = TrialFunctions(self.X)
      v,S,q = TestFunctions(self.X)

      # Mass matrix
      Ma = assemble(inner(u,v)*dx + T*S*dx)

      Ma = as_backend_type(Ma).sparray()
      print ("Size of Ma =",Ma.shape)

      Re = Constant(self.Re)
      Gr = Constant(self.Gr)
      Pr = Constant(self.Pr)
      Aform = linear_form(Re,Gr,Pr,us,Ts,u,T,p,v,S,q)
      Aa = assemble(Aform)

      # Convert to sparse format
      Aa = as_backend_type(Aa).sparray()
      print ("Size of Aa =",Aa.shape)

      # indices of free nodes
      freeinds,pinds = self.get_indices()

      # mass matrix
      M = Ma[freeinds,:][:,freeinds]
      print ("Size of M =",M.shape)

      # stiffness matrix
      A = Aa[freeinds,:][:,freeinds]
      print ("Size of A =",A.shape)

      # Now compute eigenvalues/vectors
      # Compute eigenvalues/vectors of (A,M)
      print ("Computing eigenvalues/vectors of (A,M) ...")
      sigma = -0.5
      print ("Using shift =", sigma)
      print ("Eigenvalue solver may not converge if shift is not good")
      vals, vecs = la.eigs(A, k=k, M=M, sigma=sigma, which='LR')
      for val in vals:
          print (np.real(val), np.imag(val))
      print ("********* ARE WE GETTING THE UNSTABLE EIGENVALUES *********")
      
      # TODO: eigenvectors are complex
      ua = Function(self.X)

      # Save real part of eigenvector. << outputs only real part
      ua.vector()[freeinds] = np.array(vecs[:,0].real)
      File("evec1.xml") << ua.vector()
      u,T,p = ua.split()
      File("evec1_u.pvd") << u
      File("evec1_T.pvd") << T
      File("evec1_p.pvd") << p

      # Save imaginary part of eigenvector. << outputs only real part
      ua.vector()[freeinds] = np.array(vecs[:,0].imag)
      File("evec2.xml") << ua.vector()
      u,T,p = ua.split()
      File("evec2_u.pvd") << u
      File("evec2_T.pvd") << T
      File("evec2_p.pvd") << p

      # Compute eigenvalues/vectors of (A^T,M^T)
      # First transpose A; M is symmetric
      print ("Computing eigenvalues/vectors of (A^T,M^T) ...")
      A.transpose()
      vals, vecs = la.eigs(A, k=k, M=M, sigma=sigma, which='LR')
      for val in vals:
          print (np.real(val), np.imag(val))
      
      for e in range(0,k,2):
          filename = "evec"+str(e+1)+"a"
          print ("Writing into file ", filename)
          # Save real part of eigenvector. << outputs only real part
          ua.vector()[freeinds] = np.array(vecs[:,e].real)
          File(filename+".xml") << ua.vector()
          u,T,p = ua.split()
          File(filename+"_u.pvd") << u
          File(filename+"_T.pvd") << T
          File(filename+"_p.pvd") << p

          filename = "evec"+str(e+2)+"a"
          print ("Writing into file ", filename)
          # Save imaginary part of eigenvector. << outputs only real part
          ua.vector()[freeinds] = np.array(vecs[:,e].imag)
          File(filename+".xml") << ua.vector()
          u,T,p = ua.split()
          File(filename+"_u.pvd") << u
          File(filename+"_T.pvd") << T
          File(filename+"_p.pvd") << p






###################################################################
   # Runs nonlinear model
   def run(self,with_control=True):
      up = Function(self.X)
      vp = TestFunction(self.X)
      v,S,q = TestFunctions(self.X)
      #hs = HeatSource(degree=2)
      sp = SourcePerturbation(degree=2)

      Re = Constant(self.Re)
      Gr = Constant(self.Gr)
      Pr = Constant(self.Pr)
      F = nonlinear_form(Re,Gr,Pr,self.hf,self.ds,up,vp)


      mesh = Mesh("square.xml")
      # hmin, area are arrays, one value per triangle
      hmin = np.array([cell.h() for cell in cells(mesh)])
      area = np.array([cell.volume() for cell in cells(mesh)])
      cfl = 1.0
      X0 = FunctionSpace(mesh, "DG", 0)
      cf = TestFunction(X0)

      fcont = open('control.dat','w')
      if with_control:
         # compute indices of velocity and temperature
         freeinds,pinds,bcinds2 = self.get_indices_lagrange()
         vTinds = np.setdiff1d(freeinds,pinds,assume_unique=True).astype(np.int32)
         gain = sio.loadmat('gain0.mat')

      fhist = open('history.dat','w')

      ups = Function(self.X)
      File("steady.xml") >> ups.vector()
      us,Ts,ps = ups.split()
      KEs = assemble(0.5*inner(us,us)*dx)
      print ('Kinetic energy of steady state =', KEs)

      # Set initial condition
      up1 = Function(self.X)
      up1.assign(ups)

      
      uppert = Function(self.X)

      fu = File("u.pvd")
      ft = File("T.pvd")

      # Compute KE
      u,T,p = up1.split()
      KE  = assemble(0.5*inner(u,u)*dx)
      # Compute perturbation energy
      uppert.vector()[:] = up1.vector() - ups.vector()
      u,T,p = uppert.split()
      #fu << (u,0.0)
      #ft << (T,0.0)
      dKE = assemble(0.5*inner(u,u)*dx)
      dHE = assemble(0.5*inner(T,T)*dx)
      print ('Kinetic energy =', KEs, KE, dKE, dHE)
      fhist.write(str(0)+" "+str(KEs)+" "+str(KE)+" "+str(dKE)+" "+str(dHE)+"\n")
       
      # Smooth bump function gt(t) to be multiplied with source perturbation; middle of the bump is at t0
      gt = Expression('Alpha*exp(-Beta*(t-t0)*(t-t0))', degree = 2, Alpha = 1.0, Beta = 50.0, t0 = 2.0, t = 0.0)

      # velocity to compute initial dt
      vel = as_vector((up1[0],up1[1]))
      Fvel = sqrt(dot(vel,vel))*cf*dx(mesh)
      vel_avg = assemble(Fvel).get_local()/area
      dt0 = hmin / (vel_avg + 1.0e-13)
      dt00 = cfl*dt0.min()
      print("dt00 = ", dt00)
      
  
      final_time = 30
      time, iter = 0, 0

      if with_control:
        dy = up1.vector().get_local() - ups.vector().get_local()
        a = -f(time)*np.dot(gain['Kt'], dy[vTinds])
        self.gs.amp = a[0]
        self.ts.amp = a[1]
        #self.hf.amp = a[2]
        fcont.write(str(time)+" "+str(a[0])+" "+str(a[1])+"\n")

      # First time step, we do backward euler
      gt.t = dt00
      idt0 = Constant(0)
      B1 = (idt0)*inner(up[0] - up1[0], vp[0])*dx     \
         + (idt0)*inner(up[1] - up1[1], vp[1])*dx     \
         + (idt0)*inner(up[2] - up1[2], vp[2])*dx + F - gt*sp*S*dx   # Add perturbation in source function
      idt0.assign(1/dt00)

      dup = TrialFunction(self.X)
      dB1 = derivative(B1, up, dup)
      problem1 = NonlinearVariationalProblem(B1, up, self.bc, dB1)
      solver1  = NonlinearVariationalSolver(problem1)

      up.assign(up1)
      solver1.solve()
      iter += 1
      time += dt00
      print ('Iter = {:5d}, t = {:f}'.format(iter, time))

      # Compute KE
      u,T,p = up.split()
      KE  = assemble(0.5*inner(u,u)*dx)
      # Compute perturbation energy
      uppert.vector()[:] = up.vector() - ups.vector()
      u,T,p = uppert.split()
      #fu << (u,time)
      #ft << (T,time)
      dKE = assemble(0.5*inner(u,u)*dx)
      dHE = assemble(0.5*inner(T,T)*dx)
      print ('Kinetic energy =', KEs, KE, dKE, dHE)
      fhist.write(str(time)+" "+str(KEs)+" "+str(KE)+" "+str(dKE)+" "+str(dHE)+"\n")
      print ('--------------------------------------------------------------')

      # From now on use BDF2
      up2 = Function(self.X)
      idt = Constant(0)
      rdt = Constant(0)
      B2 = (idt)*inner((2*rdt+1)/(rdt+1)*up[0] - (rdt+1)*up1[0] + (rdt**2)/(rdt+1)*up2[0], vp[0])*dx     \
         + (idt)*inner((2*rdt+1)/(rdt+1)*up[1] - (rdt+1)*up1[1] + (rdt**2)/(rdt+1)*up2[1], vp[1])*dx     \
         + (idt)*inner((2*rdt+1)/(rdt+1)*up[2] - (rdt+1)*up1[2] + (rdt**2)/(rdt+1)*up2[2], vp[2])*dx + F - gt*sp*S*dx # Add perturbation in source function
      dB2 = derivative(B2, up, dup)
      problem2 = NonlinearVariationalProblem(B2, up, self.bc, dB2) 
      solver2  = NonlinearVariationalSolver(problem2)

      while time < final_time:
         up2.assign(up1)
         up1.assign(up)
         vel = as_vector((up1[0],up1[1]))
         Fvel = sqrt(dot(vel,vel))*cf*dx(mesh)
         vel_avg = assemble(Fvel).get_local()/area
         dt0 = hmin / (vel_avg + 1.0e-13)
         dt = cfl*dt0.min()
         #print("dt = ", dt)
         idt.assign(1/dt)
         rdt.assign(dt/dt00)
         # initial guess by extrapolation
         up.vector()[:] =  up2.vector() + (dt+dt00)/(dt00)*(up1.vector() - up2.vector())
         dt00 = dt
         if with_control:
            dy = up.vector().get_local() - ups.vector().get_local()
            a = -f(time)*np.dot(gain['Kt'], dy[vTinds])
            self.gs.amp = a[0]
            self.ts.amp = a[1]
            #self.hf.amp = a[2]
            fcont.write(str(time)+" "+str(a[0])+" "+str(a[1])+"\n")
            fcont.flush()
         gt.t = time + dt
         solver2.solve()
         iter += 1
         time += dt
         print ('Iter = {:5d}, t = {:f}, dt = {:f}'.format(iter, time, dt))
         # Compute KE
         u,T,p = up.split()
         KE  = assemble(0.5*inner(u,u)*dx)
         # Compute perturbation energy
         uppert.vector()[:] = up.vector() - ups.vector()
         u,T,p = uppert.split()
         #if iter%10 == 0:
            #fu << (u,time)
            #ft << (T,time)
         dKE = assemble(0.5*inner(u,u)*dx)
         dHE = assemble(0.5*inner(T,T)*dx)
         print ('Kinetic energy =', KEs, KE, dKE, dHE)
         fhist.write(str(time)+" "+str(KEs)+" "+str(KE)+" "+str(dKE)+" "+str(dHE)+"\n")
         fhist.flush()
         print ('--------------------------------------------------------------')

      fhist.close()
