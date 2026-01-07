import numpy as np
import numba as nb
from numba import types, typed
from typing import Union, List

from tracking_analysis.config.configvar import NUM_PULSES

def indexToSpace(index, size_nm, px_nm):
    space = np.zeros(2)
    space[0] = index[1]*px_nm
    space[1] = index[0]*px_nm
    return np.array(space)

def likelihood(NUM_PULSES, PSF, n, λb, pos_nm, step_nm, size_nm):
    
    """
    Computes the full likelihood for a given MINFLUX experiment 
    
    Input
    ----------
    NUM_PULSES : int, number of excitation beams
    PSF : (NUM_PULSES, size, size) array, experimental or simulated PSF
    n :  (1, NUM_PULSES) array , photon collection 
    λb : float, bkgd level
    pos _nm : (NUM_PULSES, 2) array, centers of the EBP positions
    step_nm : step of the grid in nm
    size_nm : size of the grid in nm
    Returns
    -------
    Like : (size, size) array, Likelihood function in each position
    
    """
    
    # size of the (x,y) grid
    size = int(size_nm/step_nm)
    
    # different arrays
    mle = np.zeros((size, size, NUM_PULSES))
    p_array = np.zeros((size, size, NUM_PULSES))
    λ_array = np.zeros((size, size, NUM_PULSES))
    λb_array = np.ones((size, size)) * λb

    
    # λs in each (x,y)
    for i in np.arange(NUM_PULSES):
        λ_array[:, :, i] = PSF[i, :, :]        
    
    norm_array = (NUM_PULSES*λb + np.sum(λ_array, axis=2))
        
    # probabilities in each (x,y)
    for i in np.arange(NUM_PULSES):
        p_array[:, :, i] = (λ_array[:, :, i] + λb_array)/norm_array
    
    # Likelihood
    for i in np.arange(NUM_PULSES):
        mle[:, :, i] = n[i] * np.log(p_array[:, :, i])
        
    Like = np.sum(mle, axis = 2)
        
    return Like

def pos_minflux(n, normed_psfs, sbr, bckg_contrib, step_nm):
    
    """    
    MINFLUX position estimator (using MLE) for a single localization
    
    Inputs
    ----------
    n : acquired photon collection (NUM_PULSES)
    PSF : array with EBP (NUM_PULSES x size x size)
    SBR : estimated (exp) Signal to Bkgd Ratio
    bckg_contrib: relative contribution to background counts per each pulse (NUM_PULSES)
    Returns
    -------
    pos_estimator : position estimator (MLE)
    
    Parameters 
    ----------
    step_nm : grid step in nm
        
    """

    # FOV size
    size = np.shape(normed_psfs)[1]
    
    # probabilitiy vector 
    p = np.zeros((NUM_PULSES, size, size))

    for pulse_idx in np.arange(NUM_PULSES):
        p[pulse_idx,:,:] = (sbr/(sbr + 1)) * normed_psfs[pulse_idx,:,:] + (1/(sbr + 1)) * bckg_contrib[pulse_idx]

    # likelihood function
    L = np.zeros((NUM_PULSES,size, size))
    for pulse_idx in np.arange(NUM_PULSES):
        L[pulse_idx, :, :] = n[pulse_idx] * np.log(p[pulse_idx, : , :])
        
    Ltot = np.sum(L, axis = 0)

    # maximum likelihood estimator for the position    
    indrec = np.unravel_index(np.argmax(Ltot, axis=None), Ltot.shape)
    pos_estimator = indexToSpace(indrec, size, step_nm)
    
    return pos_estimator

def loc_trace_minflux(ph_perloc_perpulse, bckg_ph_perloc_perpulse, sbr_perloc, psfs, step_nm):
    """
    This function computes the whole localization trace using MINFLUX localization algorithm (MLE)
    
    Inputs
    ----------
    ph_perloc_perpulse : array of number of photons for each localization and pulse (number of pulses x number of locs)
    bckg_ph_perloc_perpulse: number of background photons per localization and for each pulse (number of locs)
    sbr_perloc: SBR for each localization (number of locs)
    psfs : array with PSFs (number of pulses x size x size)
    
    Returns
    -------
    locs : array of all estimated localizations
    
    Parameters 
    ----------
    step_nm : grid step in nm
    """
    # compute normalized psfs
    psf_norm = np.sum(psfs, axis = 0)
    normed_psfs = psfs / psf_norm
    n_loc = len(ph_perloc_perpulse[0,:])
    tot_bckg_ph_inloc = np.sum(bckg_ph_perloc_perpulse)
    bckg_contrib = np.empty(NUM_PULSES, dtype=float)
    bckg_contrib = bckg_ph_perloc_perpulse / tot_bckg_ph_inloc
    locs = np.empty((n_loc, 2), dtype=float)
    for loc_idx in range(n_loc):
        locs[loc_idx, :] = pos_minflux(ph_perloc_perpulse[:, loc_idx], normed_psfs, sbr_perloc[loc_idx], bckg_contrib, step_nm)
    return locs

def crb_minflux(NUM_PULSES, PSF, SBR, px_nm, size_nm, N, method='1'):
    
    """
    
    Cramer-Rao Bound for a given MINFLUX experiment 
    
    Input
    ----------
    NUM_PULSES : int, number of excitation beams
    PSF : (NUM_PULSES, size, size) array, experimental or simulated PSF 
    SBR : float, signal to background ratio
    px_nm : pixel of the grid in nm
    size_nm : size of the grid in nm
    N : total number of photons
    method: parameter for the chosen method
    
    There are three methods to calculate it. They should be equivalent but
    provide different outputs.
    
    Method 1: calculates the σ_CRB using the most analytical result 
    (S26, 10.1126/science.aak9913)
    
    Output 1: σ_CRB (size, size) array, mean of CRB eigenval in each position
    
    Method 2: calculates the Σ_CRB from the Fisher information matrix in 
    emitter position space (Fr), from there it calculates Σ_CRB and σ_CRB
    (S11-13, 10.1126/science.aak9913)
    
    Output 2: Fr, Σ_CRB, σ_CRB
    
    Method 3: calculates the Fisher information matrix in reduced probability
    space and calculates J jacobian transformation matrices. From there it
    calculates Fr, Σ_CRB, σ_CRB. Fp, Σ_CRB_p and σ_CRB_p are additional outputs
    (S8-10, 10.1126/science.aak9913)
    
    Output 3: Fr, Σ_CRB, σ_CRB, Fp, Σ_CRB_p, σ_CRB_p
    """
    
    # size of the σ_CRB matrix in px and dimension d=2
    size = int(size_nm/px_nm)
    d = 2
    
    # size of the (x,y) grid
    dx = px_nm
    dy = px_nm
    
    if method=='1':
                
        # define different arrays needed to compute CR
        
        p, λ, dpdx, dpdy, A, B, C, D = (np.zeros((NUM_PULSES, size, size)) for i in range(8))
        
        # normalization of PSF to Ns = N*(SBR/(SBR+1))

        for i in range(NUM_PULSES):
    
            λ[i, :, :] = N*(SBR/(SBR+1)) * (PSF[i, :, :]/np.sum(PSF, axis=0))
            
        # λb using the approximation in Balzarotti et al, (S29)
            
        λb = np.sum(λ[:, int(size/2), int(size/2)])/(NUM_PULSES*SBR)
        
        # probabilities in each (x,y)
        
        for i in np.arange(NUM_PULSES):
            
            # probability arrays
    
            p[i, :, :] = (λ[i, :, :] + λb)/(NUM_PULSES*λb + np.sum(λ, axis=0))
            
            # plot of p
            
            locx = (size/4) * np.sqrt(2)/2
            locy = (size/4) * np.sqrt(2)/2
            
            #plt.figure(str(i))
            #plt.plot(np.arange(-size/2, size/2), p[i, int(size/2 - locx), :], label='p x axis')
            #plt.plot(np.arange(-size/2, size/2), p[i, ::-1, int(size/2 - locy)], label='p y axis')
                                    
            # gradient of ps in each (x,y)
            
            dpdy[i, :, :], dpdx[i, :, :] = np.gradient(p[i, :, :], -dy, dx)
           
            # terms needed to compute CR bound in aeach (x,y)
            
            A[i, :, :] = (1/p[i, :, :]) * dpdx[i, :, :]**2
            B[i, :, :] = (1/p[i, :, :]) * dpdy[i, :, :]**2
            C[i, :, :] = (1/p[i, :, :]) *(dpdx[i, :, :] * dpdy[i, :, :])
            D[i, :, :] = (1/p[i, :, :]) * (dpdx[i, :, :]**2 + dpdy[i, :, :]**2)
    
        # sigma Cramer-Rao numerator and denominator    
        E = np.sum(D, axis=0) 
        F = (np.sum(A, axis=0) * np.sum(B, axis=0)) - np.sum(C, axis=0)**2
        
        σ_CRB = np.sqrt(1/(d*N))*np.sqrt(E/F)
            
        return σ_CRB
    
    if method=='2':
    
        # initialize different arrays needed to compute σ_CRB, Σ_CRB and Fr
        
        σ_CRB = np.zeros((size, size))
        p, λ, dpdx, dpdy = (np.zeros((NUM_PULSES, size, size)) for i in range(4))
        Fr, Σ_CRB = (np.zeros((d, d, size, size)) for i in range(2))
        
        Fr_aux = np.zeros((NUM_PULSES, d, d, size, size))
        
        # normalization of PSF to Ns = N*(SBR/(SBR+1))

        for i in range(NUM_PULSES):
            
            λ[i, :, :] = N*(SBR/(SBR+1)) * (PSF[i, :, :]/np.sum(PSF, axis=0))
            
        # λb using the approximation in Balzarotti et al, (S29)
          
        λb = np.sum(λ[:, int(size/2), int(size/2)])/(NUM_PULSES*SBR)
            
        for i in range(NUM_PULSES):
            
            # probability arrays
        
            p[i, :, :] = (λ[i, :, :] + λb)/(NUM_PULSES*λb + np.sum(λ, axis=0))

            # partial derivatives in x and y direction
    
            dpdy[i, :, :], dpdx[i, :, :] = np.gradient(p[i, :, :], -dy, dx)
            
        # compute relevant information for every (i, j) position
        # TODO: vectorize this part of the code
            
        for i in range(size):
            for j in range(size):
                
                for k in range(NUM_PULSES):
            
                    A = np.array([[dpdx[k, i, j]**2, 
                                   dpdx[k, i, j]*dpdy[k, i, j]],
                                  [dpdx[k, i, j]*dpdy[k, i, j], 
                                   dpdy[k, i, j]**2]])
        
                    Fr_aux[k, :, :, i, j] = (1/p[k, i, j]) * A
                    
                Fr[:, :, i, j] = N * np.sum(Fr_aux[:, :, :, i, j], axis=0)
                                    
                Σ_CRB[:, :, i, j] = np.linalg.inv(Fr[:, :, i, j])
                σ_CRB[i, j] = np.sqrt((1/d) * np.trace(Σ_CRB[:, :, i, j]))
                
        
        return σ_CRB, Σ_CRB, Fr
            
         
    if method=='3':
    
        # initalize σ_CRB and E(logL)
        
        I_f = np.zeros((size, size))
        σ_CRB = np.zeros((size, size))
        σ_CRB2 = np.zeros((size, size))
        σ_CRB_p = np.zeros((size, size))
        
        logL = np.zeros((size, size))
    
        # initialize different arrays needed to compute σ_CRB, Σ_CRB, Fr, etc
    
        p, λ, dpdx, dpdy, logL_aux = (np.zeros((NUM_PULSES, size, size)) for i in range(5))
        Fr, Σ_CRB = (np.zeros((d, d, size, size)) for i in range(2))
        Fp, Σ_CRB_p = (np.zeros((NUM_PULSES-1, NUM_PULSES-1, size, size)) for i in range(2))
    
        J = np.zeros((NUM_PULSES-1, d, size, size))
        
        diag_aux = np.zeros(NUM_PULSES-1)
        
        # normalization of PSF to Ns = N*(SBR/(SBR+1))
                      
        for i in range(NUM_PULSES):
            
            λ[i, :, :] = N*(SBR/(SBR+1)) * (PSF[i, :, :]/np.sum(PSF, axis=0))
                    
        # λb using the approximation in Balzarotti et al, (S29)
              
        λb = np.sum(λ[:, int(size/2), int(size/2)])/(NUM_PULSES*SBR)
            
        for i in range(NUM_PULSES):
            
            # probability arrays
            
            p[i, :, :] = (λ[i, :, :] + λb)/(NUM_PULSES*λb + np.sum(λ, axis=0))
            
            logL_aux[i, :, :] = N * p[i, :, :] * np.log(p[i, :, :])
            
            # partial derivatives in x and y direction

            dpdy[i, :, :], dpdx[i, :, :] = np.gradient(p[i, :, :], -dy, dx)
            
#            locx = (size/4) * np.sqrt(2)/2
#            locy = (size/4) * np.sqrt(2)/2
            
            locx = (size/4)
            locy = (size/4)
#            
#            plt.figure(str(i))
#            plt.imshow(p[i, :, :])
            
#            plt.figure()
#            plt.plot(np.arange(-size/2, size/2), p[i, int(size/2), :], label='p x axis')
#            plt.plot(np.arange(-size/2, size/2), p[i, ::-1, int(size/2 - locy)], label='p y axis')
#            plt.legend()   
            
            
        for i in range(size):
            for j in range(size):
                    
                for k in range(NUM_PULSES):
                    
                    if k < NUM_PULSES-1:
                        
                        J[k, :, i, j] = np.array([dpdx[k, i, j], dpdy[k, i, j]])
                        
#                    if k < NUM_PULSES-2:
                        
                        diag_aux[k] = 1/p[k, i, j]
                        
                    else:
                        
                        pass
                    
                logL[i, j] = np.sum(logL_aux[:, i, j], axis=0)
                        
                p_aux = np.diag(diag_aux)
                
                Fp[:, :, i, j] = N * (p_aux + np.ones((NUM_PULSES-1, NUM_PULSES-1))*(1/p[NUM_PULSES-1, i, j]))
                Fr[:, :, i, j] = J[:, :, i, j].T.dot(Fp[:, :, i, j]).dot(J[:, :, i, j])
                                
                Σ_CRB[:, :, i, j] = np.linalg.inv(Fr[:, :, i, j])
                σ_CRB[i, j] = np.sqrt((1/d) * np.trace(Σ_CRB[:, :, i, j]))
                
                Σ_CRB_p[:, :, i, j] = np.linalg.inv(Fp[:, :, i, j])
                σ_CRB_p[i, j] = np.sqrt((1/(NUM_PULSES-1)) * np.trace(Σ_CRB_p[:, :, i, j]))
                
                I_f[i, j] = np.sqrt((1/d) * np.trace(Fr[:, :, i, j]))
                σ_CRB2[i, j] = 1/I_f[i, j] 

                
        print(Fr[:, :, int(size/2), int(size/2 - locx)])
        print(Fr[:, :, int(size/2 + locy), int(size/2)])
        print(Fr[:, :, int(size/2), int(size/2 + locy)])
        print(Fr[:, :, int(size/2 - locy), int(size/2)])
        
                
        
#        print(p[:, int(size/2 - locy), int(size/2 - locx)])
#        print(p[:, int(size/2 + locy), int(size/2 - locx)])
#        print(p[:, int(size/2 - locy), int(size/2 + locx)])
#        print(p[:, int(size/2 + locy), int(size/2 + locx)])
        
        print(p[:, int(size/2), int(size/2)])
        
        print(p[:, int(size/2), int(size/2 - locx)])
        print(p[:, int(size/2 + locy), int(size/2)])
        print(p[:, int(size/2), int(size/2 + locy)])
        print(p[:, int(size/2 - locy), int(size/2)])
        
        print(σ_CRB[int(size/2), int(size/2 - locx)])
        print(σ_CRB[int(size/2 + locy), int(size/2)])
        print(σ_CRB[int(size/2), int(size/2 + locy)])
        print(σ_CRB[int(size/2 - locy), int(size/2)])


        return σ_CRB, Σ_CRB, Fr, σ_CRB_p, Σ_CRB_p, Fp, logL, I_f, σ_CRB2
    
    else:
        
        raise ValueError('Invalid method number, please choose 1, 2 or 3 \
                         according to the desired calculation')