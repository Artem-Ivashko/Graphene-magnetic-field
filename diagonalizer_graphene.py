from matplotlib import pyplot as plt
from scipy.sparse import linalg as lag
from math import sqrt, sin, cos, fabs
import numpy as np
from numpy import pi, exp, sign

import kwant



from tinyarray import array as ta
sigma0 = ta([[1, 0], [0, 1]])
sigma1 = ta([[0, 1], [1, 0]])
sigma2 = ta([[0, -1j], [1j, 0]])

# Simple Namespace
class SimpleNamespace(object):
    """A simple container for parameters."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)



def LandauEnergyTh(LLNumber, Parameters, rho1 = 0., rho2 = 0.):#mu_nonmin = 0., muprime_nonmin = 0. ):
    tempVar = rho1**2 * Parameters.lBinv2**2 + \
    2. * fabs(LLNumber) * Parameters.lBinv2 * ( Parameters.FermiVelocity + 2.*fabs(LLNumber)*rho2*Parameters.lBinv2 )**2
    
    if LLNumber == 0:
        return -3.*Parameters.tNNN + rho1 * Parameters.lBinv2
    else:
        return -3.*Parameters.tNNN + 2.* fabs(LLNumber) * rho1 * Parameters.lBinv2 + sign(LLNumber)*sqrt(tempVar)
    
    
    
def onsite_1D( site,p ):
    # for magnetic field in z-direction, Landau gauge
    #I have checked that the parameter lBinv2 entering here is (aNN/lB)^2, where lB = sqrt(hbar c/eB) is the real magnetic length
    x,=site.pos
    pyLong = p.py - float(x-p.x_shift)*float(p.lBinv2)
    return  p.tNN * ( cos(pyLong) * sigma1 - sin(pyLong) * sigma2 )



def hopNN_1D( s1,s2,p ):
    x,=s2.pos
    pyLong = p.py - float(x-p.x_shift)*float(p.lBinv2)
    return p.tNN * ( cos(pyLong/2.) * sigma1 + sin(pyLong/2.) * sigma2 ) + p.tNNN * 2. * cos(3.*pyLong/2.) * sigma0



def hopNNN_1D( s1,s2,p ):
    return p.tNNN * sigma0



def FinalizedSystem_1D( SitesCount_X, p ):
    # lattices
    lat = kwant.lattice.general( ((p.LatticeSpacing,),) )
    # builder
    sys = kwant.Builder()
    # first, define all sites and their onsite energies
    for nx in range(SitesCount_X):
        sys[lat(nx,)]=onsite_1D
    # hoppings
    for nx in range(SitesCount_X-1):
        sys[lat(nx+1,),lat(nx,)]=hopNN_1D
    for nx in range(SitesCount_X-2):
        sys[lat(nx+2,),lat(nx,)]=hopNNN_1D        
    # finalize the system
    return sys.finalized()        
     
     

def diagonalize_1D( FinalizedSystem, Parameters):
    ham_sparse_coo = FinalizedSystem.hamiltonian_submatrix( args=([Parameters]), sparse=True )
    ham_sparse = ham_sparse_coo.tocsc()
    
    EigenValues, EigenVectors = lag.eigsh( ham_sparse, k=Parameters.EigenvectorsCount, which='LM', \
                                          sigma = Parameters.FermiEnergy, tol = Parameters.EnergyPrecision )
    EigenVectors = np.transpose(EigenVectors)
    #Sorting the wavefunctions by eigenvalues, so that the states with the lowest energies come first
    idx = EigenValues.argsort()
    EigenValues = EigenValues[idx]
    EigenVectors = EigenVectors[idx]
    
    SitesCount_X = len(FinalizedSystem.sites)
    EigenVectors = np.reshape(EigenVectors, (Parameters.EigenvectorsCount, SitesCount_X, \
                                             Parameters.WavefunctionComponents) )
    
    return EigenValues, EigenVectors


    
def pSweep_1D( FinalizedSystem, Parameters, pyMin, pyMax, pyCount):
    pSweep = np.linspace(pyMin, pyMax, pyCount)
    EigenValuesSweep = []
    EigenVectorsSweep = []
    for p in pSweep:
        Parameters.py = p
        EigenValues, EigenVectors = diagonalize_1D( FinalizedSystem, Parameters )
        EigenValuesSweep.append(EigenValues)
        EigenVectorsSweep.append(EigenVectors)
        
    return EigenValuesSweep, EigenVectorsSweep



def spectrum_plot_1D( EigenValues, pMin, pMax, pCount ):
    pSweep = np.linspace(pMin, pMax, pCount)
    plt.plot(pSweep, EigenValues,"k.",markersize=3)
    plt.xlim(pMin,pMax)
    plt.ylabel('Energy [t]')
    plt.show()

    
    
def realspace_position_1D( FinalizedSystem, LatticeSpacing ):
    SitesCount_X = len(FinalizedSystem.sites)
    Tolerance = 10**(-3)
    return [[j for j in range(SitesCount_X) if fabs(FinalizedSystem.sites[j].pos[0] - i*LatticeSpacing) < Tolerance][0] for i in range(SitesCount_X)]



#The procedure plots the number density as a function of coordinate (horizontal axis)
#The sweep is over different energies (vertical axis, however the coordinate on that axis is not the energy itself, but rather an index of the energy)
def density_plot_1D( FinalizedSystem, Parameters, EigenVectors ):
    position_1D = realspace_position_1D(FinalizedSystem, Parameters.LatticeSpacing)
    SitesCount_X = len(FinalizedSystem.sites)

    density = [[np.vdot(EigenVectors[i][position_1D[j]],EigenVectors[i][position_1D[j]]) for j in range(SitesCount_X)] \
               for i in range(Parameters.EigenvectorsCount)]
    density = np.real(density)

    plt.pcolor( np.linspace(0,SitesCount_X,SitesCount_X), np.linspace(0,1,Parameters.EigenvectorsCount), density, \
               vmin = 0, vmax= 3./SitesCount_X)
    plt.colorbar()
    plt.show()