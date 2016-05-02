# -*- coding: utf-8 -*-
"""
This module provides exact diagonalization code to benchmark the results
obtained from the SDP numerics.

@author: Christian Gogolin, Peter Wittek
"""
from __future__ import print_function, division
import sys
import cmath
import csv
import math
import os
import pickle
import time
from abc import ABCMeta, abstractmethod

import sympy
from sympy import adjoint, conjugate, S, Symbol, Pow, Number, expand, I, Matrix, Lambda, Mul, Add, Integer, Float
from sympy.core.numbers import NegativeOne
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.operator import Operator
from ncpol2sdpa import RdmHierarchy, get_neighbors, get_next_neighbors, \
                       generate_variables, bosonic_constraints, flatten, \
                       fermionic_constraints, SdpRelaxation

import multiprocessing, logging
# mpl = multiprocessing.log_to_stderr()
# mpl.setLevel(logging.INFO)

from functools import partial
import numpy as np
import scipy
import itertools as it
import functools as ft
from scipy.sparse.linalg import eigsh
import scipy.sparse as sps


def write_array(filename, array):
    file_ = open(filename, 'w')
    writer = csv.writer(file_)
    writer.writerow(array)
    file_.close()
    

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
            self.window_length = lattice_length * lattice_width
        else:
            if self._lattice_width != 1:
                raise Exception("Windowed models in more than 1D not implemented!")
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
        V = self.getSize()
        if self.n is None:
            self.dimh = int(pow(2, 2*V))
            return it.product([0, 1], repeat = 2*V)
        else:
            self.dimh = int(scipy.special.binom(2*V, self.n))
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
            
            return unique_permutations(list(it.chain(it.repeat(0, 2*V - int(self.n)), it.repeat(1, int(self.n)))))
            
    def createHamiltonian(self):
        if self._periodic == -1:
            raise Exception("Not implemented!")
        
        time0 = time.time()        
            
        V = self.getSize()        

        #reverse lookup table for the elements of the hilbet space to efficiently
        #generate the off diagonal part of the Hamiltonian
        hdict = {}
        for row, vec in enumerate(self.getHilbertSpace()):
            hdict[vec] = row

        H = sps.dok_matrix((self.dimh, self.dimh), dtype=np.float32)

        for row, vec in enumerate(self.getHilbertSpace()):
            vecu = vec[:V]
            vecd = vec[V:]

            #diagonal part
            H[row,row] = \
            self.U * np.dot(vecu, vecd) \
            - self.mu * sum(vec) \
            - self.h/2 * (sum(vecu) - sum(vecd))

            #off-diagonal part
            for j1 in range(V):
                for k1 in get_neighbors(j1, self.getLength(), width=self.getWidth(), periodic=self._periodic):
                    j2 = j1+V
                    k2 = k1+V
                    sign, newvec = self.hop(j1,k1,vec)
#                    print("hop",j1,"->",k1,vec,newvec)
                    if newvec is not None:
                        col = hdict[newvec]
                        H[row,col] += -self.t*sign
                        H[col,row] += -self.t*sign
                    sign, newvec = self.hop(j2,k2,vec)
#                    print("hop",j2,"->",k2,vec,newvec)
                    if newvec is not None:
                        col = hdict[newvec]
                        H[row,col] += -self.t*sign
                        H[col,row] += -self.t*sign
                        
        print("Hilbert space and Hamiltonian for system of dimension "+str(len(H))+" generated in", time.time()-time0, "seconds")
        return H

    def getSize(self):
        """Returns the total size (volume) of the system.
        """
        return self._lattice_length * self._lattice_width

    def getLength(self):
        """Returns the length of the system.
        """
        return self._lattice_length

    def getWidth(self):
        """Returns the length of the system.
        """
        return self._lattice_width
    
    def solve(self):
        H = self.getHamiltonian()

        try:
            with open(self._outputDir + "/" +"edEnergy" +
                      self.getSuffix() + ".pickle", 'rb') as handle:
                self.energy = pickle.load(handle)
            with open(self._outputDir + "/" +"edGreoundState" +
                      self.getSuffix() + ".pickle", 'rb') as handle:
                self.groundstate = pickle.load(handle)
            print("Hamiltonian and ground state energy succesfully unpickled")
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            if H.shape == (1,1):
                self.energy, self.groundstate = H[0,0], [1]
            else:
                try:
                    time0 = time.time()
                    self.energy, self.groundstate = eigsh(H, 1, which="SA", maxiter=1000, tol=1E-5)
                    print("Ground state of Hamiltonian of dimension "+str(len(H))+" found in", time.time()-time0, "seconds")

                    if self._outputDir is not None:
                        if not os.path.isdir(self._outputDir):
                            os.mkdir(self._outputDir)
                        with open(self._outputDir + "/" +"edEnergy" +
                                  self.getSuffix() + ".pickle", 'wb') as handle:
                            pickle.dump(self.energy, handle)
                        with open(self._outputDir + "/" +"edGreoundState" +
                                  self.getSuffix() + ".pickle", 'wb') as handle:
                            pickle.dump(self.groundstate, handle)
                except (KeyboardInterrupt, SystemExit):
                    raise 
                except:
                    print("paramters="+self.getSuffix())
                    print("ERROR finding the ground state of H="+str(H)+" in Hilbert space "+str(self.getHilbertSpace()))
                    self.energy, self.groundstate = -1, [1/self.dimh] * self.dimh
                    raise

    def getPrimal(self):
        return self.getEnergy()
                
    def getEnergy(self):
        if self.energy is None:
            self.solve()
        return self.energy
        
    def getMagnetization(self):
        if self.groundstate is None:
            self.solve()
        Mdiag = [0.5*(sum(vec[:self.getSize()]) - sum(vec[self.getSize():])) for vec in self.getHilbertSpace()]
        #print('Mdiag='+str(Mdiag)+" self.groundstate="+str(self.groundstate))
        return np.dot(Mdiag, [c * np.conj(c) for c in self.groundstate])

    def expectationValue(self, operator):
        """Returns the expectation value of the given operator.
        """
        if self.groundstate is None:
            self.solve()
        return np.vdot(self.groundstate, np.dot(operator,self.groundstate))

    def getXMat(self, variables, monomials):
        V = self.getSize()
        
        print("generating uplifted ground state")
        upliftedgroundstate = np.zeros(int(pow(2, 2*V)))
        
        for row, vec in enumerate(self.getHilbertSpace()):
            hdict = self.getHdictFull()
            col = hdict[tuple(vec)]
            upliftedgroundstate[col] = self.groundstate[row]
        
        print("generating xmat entries")
        time0 = time.time()
        # the following is roughly equivalent to
        # output = [[ np.vdot(np.dot(m1,upliftedgroundstate), np.dot(m2,upliftedgroundstate)) for m1 in self.getMonomialVector(old, new, monomials)] for m2 in self.getMonomialVector(old, new, monomials)]
        monomialvec = self.getMonomialVector(variables, monomials)
        pool = multiprocessing.Pool()
        m = pool.map_async(partial(npdotinverted, upliftedgroundstate), monomialvec).get(0xFFFF) #this makes keyboard interrup work, see: http://stackoverflow.com/questions/1408356/keyboard-interrupts-with-pythons-multiprocessing-pool
        pool.close()
        pool.join()
        
        output = np.empty([len(m), len(m)])
        pool2 = multiprocessing.Pool()
        lock = multiprocessing.Lock()
        m2 = pool2.imap(npstardot, it.product(m, repeat=2))
        for i, out in enumerate(m2, 1):
            lock.acquire()
            row = (i-1) % len(m)
            col = (i-1 - row)/len(m)
            if row >= col:
                output[row, col] = output[col, row] = out
            sys.stdout.write("\r\x1b[Kprocessed "+str(i)+" xmat entries of "+str(len(m)*len(m))+" in "+str(time.time()-time0)+" seconds ")
            sys.stdout.flush()
            lock.release()
        pool2.close()
        pool2.join()
        print("done")
        return np.array(output, dtype=float)
        
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
        # if self._periodic:
        #     suffix += "_periodic=" + str(self._periodic)
        if self.window_length != self._lattice_length * self._lattice_width:
            suffix += "_window=" + str(self.window_length)
        if self.mu != 0:
            suffix += "_mu=" + str(self.mu)
        suffix += "_level="+str(self._level)
        return suffix

    def getAnnihiliationOperators(self):
        V = self.getSize()
        if self.annihiliationOperators is None:
            print("generating annihiliation operators")
            self.annihiliationOperators = [self.createAnnihiliationOperator(j) for j in range(0, 2*V)]
        return self.annihiliationOperators
    
    def createAnnihiliationOperator(self, j):
        V = self.getSize()
        a = np.zeros((int(pow(2, 2*V)), int(pow(2, 2*V))))
        for row, vec in enumerate(it.product([0, 1], repeat = 2*V)):
            if vec[j] == 1:
                newvec = list(vec)
                newvec[j] = 0
                col = self.getHdictFull()[tuple(newvec)]
                a[col,row] = pow(-1, sum(vec[0:j]))
        return a

    def getHdictFull(self):
        if self.hdictfull is None:
            self.hdictfull = self.createHdictFull()
        return self.hdictfull
    
    def createHdictFull(self):
        V = self.getSize()
        #reverse lookup table for the elements of the hilbet space to efficiently generate the annihilation operators
        print("generating lookup table")
        hdict = {}
        for row, vec in enumerate(it.product([0, 1], repeat = 2*V)):
            hdict[vec] = row
        return hdict

    def getMonomialVector(self, variables, monomials):
        if self.monomialvector is None:
            try:
                with open(self._outputDir + "/" +"edMonomialVector" +
                          self.getSuffix() + ".pickle", 'rb') as handle:
                    self.monomialvector = pickle.load(handle)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:           
                self.createMonomialVector(variables, monomials)
        return self.monomialvector

    def createMonomialVector(self, variables, monomials):
        
        flatmonomials = flatten(monomials)
        monomialsLength = len(flatmonomials)
        print(" generating monomial vector")
        time0 = time.time()
        
        self.monomialvector = []
        pool = multiprocessing.Pool()
        monomials = pool.imap(partial(expressionToMatrix, variables=list(variables), matrices=self.getAnnihiliationOperators()), flatmonomials)
        for i, monom in enumerate(monomials, 1):
            self.monomialvector.append(monom)
            sys.stdout.write("\r\x1b[K processed "+str(i)+" of "+str(monomialsLength)+" monomials in "+str(time.time()-time0)+" seconds ")
            sys.stdout.flush()
        pool.close()
        pool.join()

        if self._outputDir is not None:
            if not os.path.isdir(self._outputDir):
                os.mkdir(self._outputDir)
                with open(self._outputDir + "/" +"edMonomialVector" +
                          self.getSuffix() + ".pickle", 'wb') as handle:
                    pickle.dump(self.monomialvector, handle)
        print("done")

    def getPhysicalQuantities(self):
        """Returns a dictionaly of names and corresponding functions of all
        physical quantities that are to be written in writeData(). This should
        be overwritten in subclasses.
        """
        return {"/edPrimal": [self.getPrimal()],
                "/edMagnetization": [self.getMagnetization()]}

    def writeData(self, which=None):
        """Writes the values of all physical quantities returned by
        getPhysicalQuantities() to the respective files.
        """
        if which==None:
            which = self.getPhysicalQuantities().items()
        for key, data in iter(which):
            write_array(self._outputDir + key + self.getSuffix() + ".csv",
                        data)

    def hop(self, j, k, vec):
        if j==k:
            raise Exception('j=k not implemented')
        if vec[j] == 0 or vec[k] == 1:
            return None, None
        else:
            newvec = list(vec)
            newvec[j] = 0
            newvec[k] = 1
            sign = 1
            if (j<k and sum(vec[j+1:k]) % 2 == 1):
                sign = -1
            elif (j>k and sum(vec[k+1:j]) % 2 == 1):
                sign = -1
                
            return sign, tuple(newvec)

            
    def projectorOntoHilbertSpace(self):
        V = self.getSize()
        P = sps.dok_matrix((int(pow(2, 2*V)), self.dimh), dtype=np.float32)
        #print("P shape="+str(P.get_shape()))
        
        dict = self.getHdictFull()
        for row, vec in enumerate(self.getHilbertSpace()):
            col = dict[vec]
            #print("vec="+str(vec)+" col="+str(col)+" row="+str(row))
            P[col,row] = 1
        

        # for col, (vec, row) in enumerate(self.getHdictFull().iteritems()):
        #     print("vec="+str(vec)+" col="+str(col)+" row="+str(row))
        #     if sum(vec) == self.n:
        #         P[row,col] = 1
        return P    
        
            
        
def expressionToMatrix(expr, variables=None, matrices=None):
    """Converts sympy expression expr formulated in terms of the variables variables to a numpy array, by replacing every occurance of a variable in variables with the corresponding numpy matrix/array in matrices.
    """

    def evalmonomial(expr, dictionary):
        if expr.func == Operator:
            return dictionary[expr]
        elif expr.func == Integer or expr.func == NegativeOne:
            return int(expr)
        elif expr.func == Float:
            return float(expr)
        elif expr.func == Dagger:
            return evalmonomial(expr.args[0], dictionary).conj().T
        elif expr.func == Pow:
            return np.linalg.matrix_power(evalmonomial(expr.args[0],dictionary),int(expr.args[1]))
        elif expr.func == Mul:
            return ft.reduce(np.dot, (evalmonomial(arg, dictionary) for arg in expr.args))
        elif expr.func == Add:
            return ft.reduce(np.add, (evalmonomial(arg, dictionary) for arg in expr.args))
        else:
            raise Exception("unknown sympy func: "+str(expr.func))
    
    dictionary = dict(zip(variables, matrices))
    try:
        matrix = evalmonomial(expr, dictionary)
        return matrix
    except:
        print("\nproblem while processing expr="+str(expr)+"wich consists of:")
        def printtree(expr, level):
            print("level",level,":", expr, "of type ", expr.func)
            for arg in expr.args:
                printtree(arg, level+1)
        printtree(expr, 0)
        raise

def npdotinverted(b, a):
    return np.dot(a,b)

def npstardot(ab):
    return np.dot(ab[0],ab[1])
