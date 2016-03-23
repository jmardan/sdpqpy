# -*- coding: utf-8 -*-
"""
This module provides exact diagonalization code to benchmark the results
obtained from the SDP numerics.

@author: Christian Gogolin, Peter Wittek
"""
from __future__ import print_function, division
import cmath
import csv
import math
import os
import pickle
import time
from abc import ABCMeta, abstractmethod

import sympy
from sympy import adjoint, conjugate, S, Symbol, Pow, Number, expand, I, Matrix, Lambda
from sympy.physics.quantum import Dagger
from ncpol2sdpa import RdmHierarchy, get_neighbors, get_next_neighbors, \
                       generate_variables, bosonic_constraints, flatten, \
                       fermionic_constraints, SdpRelaxation

import multiprocessing
import numpy as np
import scipy
import itertools as it
from scipy.sparse.linalg import eigsh
import scipy.sparse as sps


class EDFermiHubbardModel():
    __metaclass__ = ABCMeta
        
    def __init__(self, lattice_length, lattice_width, outputDir,
                 periodic=0, window_length=0):
        self._lattice_length = lattice_length
        self._lattice_width = lattice_width
        self._periodic = periodic
        self._outputDir = outputDir
        self.__hamiltonian = None
        if window_length == 0:
            self.window_length = lattice_length
        else:
            self.window_length = window_length
        self.mu, self.t, self.h, self.U = 0, 0, 0, 0
        self.n = None
        self.energy = None
        self.groundstate = None
        self.dimh = None
        self._level = -1
        self._periodic = periodic
        self.nmax = None
        self.localNmax = None
        self.n = None
        self.nmin = None
        self.annihiliationOperators = None
        self.monomialvector = None
        self.hdictfull = None

    def setParameters(self, **kwargs):
        if "mu" in kwargs:
            self.mu = kwargs.get("mu")
            self.invalidateHamiltonian()
        if "t" in kwargs:
            self.t = kwargs.get("t")
            self.invalidateHamiltonian()
        if "h" in kwargs:
            self.h = kwargs.get("h")
            self.invalidateHamiltonian()
        if "U" in kwargs:
            self.U = kwargs.get("U")
            self.invalidateHamiltonian()

    def invalidateHamiltonian(self):
        """Deletes the cached Hamiltonian so that it is regenerted from the set
        parameters during the next call to getHamiltonian().
        """
        self.invalidateSolution()
        self.__hamiltonian = None

    def invalidateHilbertSpace(self):
        """Deletes the cached Hilbert specifide so that it is regenerted from the set
        parameters during the next call to getHilbertSpace().
        """
        self.__hilberspace = None
        self.invalidateHamiltonian()

    def invalidateSolution(self):
        """Invalidates the stored solution. This function is mainly for
        intermal use and is caled for example when the paramters of the system 
        are changed.
        """
        self.energy = None
        self.groundstate = None
        
    def setConstraints(self, **kwargs):
        """Sets the specifide constaints. If a constraint was not previously
        set or the value of it was changed it invalidates the Hamiltonian and
        the Hilbert space.
        """
        if "localNmax" in kwargs and self.localNmax != kwargs.get("localNmax"):
            raise Exception("Not implemented!")
        if "n" in kwargs and self.n != kwargs.get("n"):
            self.invalidateHilbertSpace()
            self.n = int(kwargs.get("n"))
            
        if "nmin" in kwargs and self.nmin != kwargs.get("nmin"):
            raise Exception("Not implemented!")
        if "nmax" in kwargs and self.nmax != kwargs.get("nmax"):
            raise Exception("Not implemented!")
        
    def getHamiltonian(self):
        """Cached getter mehtod that calls createHamiltonian() if necessary.
        """
        if self.__hamiltonian is None:
            self.__hamiltonian = self.createHamiltonian()
        return self.__hamiltonian

    def getHilbertSpace(self):
        """Cached getter mehtod that calls createHilbertSpace() if necessary.
        """
        return self.createHilbertSpace()
    
    def createHilbertSpace(self):
        L = self.getLength()
        if self.n is None:
            self.dimh = int(pow(2, 2*L))
            return it.product([0, 1], repeat = 2*L)
        else:
            self.dimh = int(scipy.special.binom(2*L, self.n))
            #taken from http://stackoverflow.com/questions/6284396/permutations-with-unique-values
            def unique_permutations(elements):
                if len(elements) == 1:
                    yield (elements[0],)
                else:
                    unique_elements = set(elements)
                    for first_element in unique_elements:
                        remaining_elements = list(elements)
                        remaining_elements.remove(first_element)
                        for sub_permutation in unique_permutations(remaining_elements):
                            yield (first_element,) + sub_permutation
            
            return unique_permutations(list(it.chain(it.repeat(0, 2*L - int(self.n)), it.repeat(1, int(self.n)))))
            
    def createHamiltonian(self):
        time0 = time.time()
        if self._lattice_width != 1:
            raise Exception("Higher dimension not implemented!")
        
        def hop(j,k,vec):
            if vec[j] == 0 or vec[k] == 1:
                return None
            else:
                newvec = list(vec)
                newvec[j] = 0
                newvec[k] = 1
                return tuple(newvec)
            
        L = self.getLength()
        
        # if self.n is None:
        #     #iteratros can only be used once in python, so we rather define a generating function...
        #     def h(): return it.product([0, 1], repeat = 2*L)
        #     dimh = int(pow(2*L, 2))
        # else:
        #     #taken from http://stackoverflow.com/questions/6284396/permutations-with-unique-values
        #     def unique_permutations(elements):
        #         if len(elements) == 1:
        #             yield (elements[0],)
        #         else:
        #             unique_elements = set(elements)
        #             for first_element in unique_elements:
        #                 remaining_elements = list(elements)
        #                 remaining_elements.remove(first_element)
        #                 for sub_permutation in unique_permutations(remaining_elements):
        #                     yield (first_element,) + sub_permutation
            
        #     #iteratros can only be used once in python, so we rather define a generating function...
        #     def h(): return unique_permutations(list(it.chain(it.repeat(0, 2*L - int(self.n)), it.repeat(1, int(self.n)))))
        #     dimh = int(scipy.special.binom(2*L, self.n))

        #reverse lookup table for the elements of the hilbet space to efficiently generate the off diagonal part of the Hamiltonian
        hdict = {}
        for row, vec in enumerate(self.getHilbertSpace()):
            hdict[vec] = row

        H = sps.dok_matrix((self.dimh, self.dimh), dtype=np.float32)

        #diagonal part
        for row, vec in enumerate(self.getHilbertSpace()):
            vecu = vec[:L]
            vecd = vec[L:]
            H[row,row] = \
            self.U * np.dot(vecu, vecd) \
            - self.mu * sum(vec) \
            - self.h/2 * (sum(vecu) - sum(vecd))

            #off-diagonal part
            if self._periodic or self._periodic==1:  
                last = L-1
            elif not self._periodic or self._periodic==0:
                last = L-2
            else:
                raise Exception("Not implemented!")
            
            for j1 in range(last):
                k1 = j1+1 % L
                j2 = j1+L
                k2 = k1 + L
                #print('hop('+str(j1)+','+str(k1)+','+str(vec)+')='+str(hop(j1,k1,vec)))
                newvec = hop(j1,k1,vec)
                if newvec is not None:
                    col = hdict[newvec]
                    H[row,col] = H[col,row] = -self.t
                newvec = hop(j2,k2,vec)
                if newvec is not None:
                    col = hdict[newvec]
                    H[row,col] = H[col,row] = -self.t
            
        #print('H='+str(H.toarray()))
        print("Hilbertspace and Hamiltonian generated in", time.time()-time0, "seconds")
        #print(str(H.toarray()))
        return H

    def getSize(self):
        """Returns the total size (volume) of the system.
        """
        return self._lattice_length * self._lattice_width

    def getLength(self):
        """Returns the length of the system.
        """
        return self._lattice_length

    
    def solve(self):
        H = self.getHamiltonian()

        try:
            with open(self._outputDir + "/" +"edEnergy" +
                      self.getSuffix() + ".pickle", 'rb') as handle:
                return pickle.load(handle)
            with open(self._outputDir + "/" +"edGreoundState" +
                      self.getSuffix() + ".pickle", 'rb') as handle:
                return pickle.load(handle)
        except:
            if H.shape == (1,1):
                self.energy, self.groundstate = H[0,0], [1]
            else:
                try:
                    time0 = time.time()
                    self.energy, self.groundstate = eigsh(H, 1, which="SA")
                    print("Hamiltonian ground state found in", time.time()-time0, "seconds")

                    if self._outputDir is not None:
                        if not os.path.isdir(self._outputDir):
                            os.mkdir(self._outputDir)
                            with open(self._outputDir + "/" +"edEnergy" +
                                      self.getShortSuffix() + ".pickle", 'wb') as handle:
                                pickle.dump(self.energy, handle)
                                with open(self._outputDir + "/" +"edGreoundState" +
                                          self.getShortSuffix() + ".pickle", 'wb') as handle:
                                    pickle.dump(self.groundstate, handle)
                    
                except:
                    print("paramters="+self.getSuffix())
                    print("ERROR finding the ground state of H="+str(H))
                    self.energy, self.groundstate = -1, [1/self.dimh] * self.dimh
                    raise
        
    def getEnergy(self):
        if self.energy is None:
            self.solve()
        return self.energy
        
    def getMagnetization(self):
        if self.groundstate is None:
            self.solve()
        Mdiag = [0.5*(sum(vec[:self.getLength()]) - sum(vec[self.getLength():])) for vec in self.getHilbertSpace()]
        #print('Mdiag='+str(Mdiag)+" self.groundstate="+str(self.groundstate))
        return np.dot(Mdiag, [c * np.conj(c) for c in self.groundstate])


    def expectationValue(self, operator):
        """Returns the expectation value of the given operator.
        """
        return 

    def getXMat(self, variables, monomials):
        L = self.getLength()
        
        old = list(variables)
        old.append(Dagger)
        
        print("generating uplifted ground state")
        upliftedgroundstate = np.zeros(int(pow(2, 2*L)))

        #print("hilbert space"+str(list(self.getHilbertSpace())))
        #print("hdict full"+str(self.getHdictFull()))
        #exit()
        
        for row, vec in enumerate(self.getHilbertSpace()):
            hdict = self.getHdictFull()
            col = hdict[tuple(vec)]
            upliftedgroundstate[col] = self.groundstate[row]
            
        #print(str(upliftedgroundstate))
        new = self.getAnnihiliationOperators()
        new.append(Dagger)
        
        print("generating output")
        time0 = time.time()
        output = [[ np.vdot(np.dot(m1,upliftedgroundstate), np.dot(m2,upliftedgroundstate)) for m1 in self.getMonomialVector(old, new, monomials)] for m2 in self.getMonomialVector(old, new, monomials)]
        print("done in ", time.time()-time0, "seconds")
        return np.array(output, dtype=float)

        # mon = self.getMonomialVector(old, new, monomials)
        # def f(i, j):
        #     return np.vdot(np.dot(mon[i],upliftedgroundstate), np.dot(mon[j],upliftedgroundstate))
        # nparray = np.fromfunction(f, (len(mon), len(mon)), dtype=int)
        # print("done in ", time.time()-time0, "seconds")
        # return nparray
    
        #return np.fromiter(            ([(np.vdot(np.dot(m1,upliftedgroundstate), np.dot(m2,upliftedgroundstate)) for m1 in self.getMonomialVector(old, new, monomials)), float, count=len(new)] for m2 in self.getMonomialVector(old, new, monomials))            , float, count=len(new))
        
    def getSuffix(self):
        suffix = "_lat=" + str(self._lattice_length) + "x" + \
                 str(self._lattice_width) + "_periodic=" + str(self._periodic)\
                 + "_mu=" + str(self.mu) + "_t=" + str(self.t) + "_h=" + \
                 str(self.h) + "_U="+str(self.U)
        if self.n is not None:
            suffix += "_n="+str(self.n)
        if self.nmax is not None:
            suffix += "_nmax="+str(self.nmax)
        if self.nmin is not None:
            suffix += "_nmin="+str(self.nmin)
        if self.localNmax is not None:
            suffix += "_localnmax="+str(self.localNmax)
        if self._periodic:
            suffix += "_periodic=" + str(self._periodic)
        if self.window_length != self._lattice_length:
            suffix += "_window=" + str(self.window_length)
        suffix += "_level="+str(self._level)
        return suffix


    def getAnnihiliationOperators(self):
        L = self.getLength()
        if self.annihiliationOperators is None:
            print("generating annihiliation operators")
            self.annihiliationOperators = [self.createAnnihiliationOperator(j) for j in range(0, 2*L)]
        return self.annihiliationOperators
    
    def createAnnihiliationOperator(self, j):
        L = self.getLength()
        #a = sps.dok_matrix((int(pow(2, 2*L)), int(pow(2, 2*L))), dtype=np.float32)
        a = sympy.zeros(int(pow(2, 2*L)), int(pow(2, 2*L)))
        for row, vec in enumerate(it.product([0, 1], repeat = 2*L)):
            if vec[j] == 1:
                newvec = list(vec)
                newvec[j] = 0
                col = self.getHdictFull()[tuple(newvec)]
                a[col,row] = pow(-1, sum(vec[0:j]))
        #return Matrix(a.toarray())
        return a

    def getHdictFull(self):
        if self.hdictfull is None:
            self.hdictfull = self.createHdictFull()
        return self.hdictfull
    
    def createHdictFull(self):
        L = self.getLength()
        #reverse lookup table for the elements of the hilbet space to efficiently generate the annihilation operators
        print("generating lookup table")
        hdict = {}
        for row, vec in enumerate(it.product([0, 1], repeat = 2*L)):
            hdict[vec] = row
        return hdict

    def getMonomialVector(self, old, new, monomials):
        if self.monomialvector is None:
            self.createMonomialVector(old, new, monomials)
        return self.monomialvector

    def createMonomialVector(self, old, new, monomials):
        L = self.getLength()
        print("generating monomial vector of length "+str(len(flatten(monomials))))
        time0 = time.time()
        #self.monomialvector = [ np.array(sympy.lambdify(old, monomial, modules="numpy")(*new)) for monomial in flatten(monomials)]

        pool = multiprocessing.Pool(processes=cpus)
        def monomialmatrix(monomial):
            return np.array(sympy.lambdify(old, monomial, modules="numpy")(*new))
        self.monomialvector = pool.map(monomialmatrix, flatten(monomials))
        
        print("done in ", time.time()-time0, "seconds")

        #print("self.monomialvector")
        #print(self.monomialvector)
        #print("self.monomialvector[1]")
        #print(self.monomialvector[1])
        #exit()
