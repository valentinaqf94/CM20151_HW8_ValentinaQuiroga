
# -*- coding: utf-8 -*-
#Ahora buscamos limitar las activaciones de las neuronas, si es menor a 0.5 
#los valores se envían a cero y si son mayores, se aproximan a uno. Se genera la funcion logística para
#introducir no linealidad en el modelo, general una combinacion lineal de dichos inputs y limitar las respuestas
#de interes segun el valor

from __future__ import division
import os
import numpy as np


def flatten_matrix(mat):
    mat = mat.flatten(1)
    mat = mat.reshape(len(mat), 1, order='F')
    return mat

def logistic(x): #funcion logistica, no lineal para limitar las activaciones entre 0 y 1
    return 1.0/(1 + np.exp(-x))

def logistic_grad(x):
    return x*(1-x)

def sparseLinearNNCost(theta, visibleSize, hiddenSize, outputSize, _lambda, sparsityParam, beta, data, labels):
     #Aquí se desenvuelve el vector en las matrices y vectores de los pesos
   
    W1 = theta[0:hiddenSize*visibleSize]
    W1 = W1.reshape(hiddenSize,visibleSize,order='F')
    W2 = theta[hiddenSize*visibleSize:(hiddenSize*visibleSize)+(hiddenSize*outputSize)]
    W2 = W2.reshape(outputSize,hiddenSize,order='F')
    b1 = theta[(hiddenSize*visibleSize)+(hiddenSize*outputSize):(hiddenSize*visibleSize)+(hiddenSize*outputSize)+hiddenSize]
    b2 = theta[(hiddenSize*visibleSize)+(hiddenSize*outputSize)+hiddenSize:]

    #Funciones lineales de los datos
    #realizamos una regresion lineal sobre los datos
    m = data.shape[1]
    rho = sparsityParam
    Z2 = np.dot(W1,data) + b1
    A2 = logistic(Z2)
    Z3 = np.dot(W2, A2) + b2
    A3 = Z3

    #Calculo de costo por minimos cuadrados,corresponde a la diferencia de mínimos cuadrados 
    #entre la salida de la neurona y lo valores predichos, se promedian.
    J = (1/m) * (0.5 * np.sum((A3 - labels.T)**2))
    #Weight decay, se le suman los costos de los pesos
    J += (_lambda/2) * (np.sum(W1**2) + np.sum(W2**2))
    #Peso de disspersion de datos
    Rho = np.sum(A2, axis = 1)/m
    Rho = Rho.reshape(len(Rho), 1, order = 'F')
     #Calculo de la divergencia, costo de dispersión de los datos, KL evalúa como una distribución de datos (La red),
    #genera otra distribución (los valores de salida)
    KL = (rho * np.log(rho/Rho)) + ((1 - rho) * np.log((1 - rho) / (1 - Rho)))
    KL = np.sum(KL)
    #Costo total
    cost = J + beta*KL

    #Calculo del gradiente de la función de costo respecto a los pesos para poder entrenar las redes con 
    #el cálculo del gradiente pues siempre apunta en la dirección mínima de la función. 
    #Con esto queremos buscar valores de gradiente que se aproximen a cero.
    D3 = -(labels.T - A3)
    D2 = np.dot(W2.T, D3)
    P = beta*((-rho/Rho)+((1-rho)/(1-Rho)))
    D2 = (D2 + P)*logistic_grad(A2)

     #Delta rule,intenta minimizar el error en la salida de los datos dado por el gradiente 
    DELTA_1 = np.dot(D2, data.T)
    DELTA_2 = np.dot(D3, A2.T)

     #Calculo del gradiente
    W1_grad = (DELTA_1 / m) + _lambda * W1
    W2_grad = (DELTA_2 / m) + _lambda * W2
    b1_grad = (np.sum(D2, axis = 1))/m
    b2_grad = (np.sum(D3, axis = 1))/m

    grad = np.vstack([flatten_matrix(W1_grad), flatten_matrix(W2_grad), flatten_matrix(b1_grad), flatten_matrix(b2_grad)])

    return [cost, grad]


def predict(data, theta, visibleSize, hiddenSize, outputSize):
    W1 = theta[0:hiddenSize*visibleSize]
    W1 = W1.reshape(hiddenSize,visibleSize,order='F')
    W2 = theta[hiddenSize*visibleSize:(hiddenSize*visibleSize)+(hiddenSize*outputSize)]
    W2 = W2.reshape(outputSize,hiddenSize,order='F')
    b1 = theta[(hiddenSize*visibleSize)+(hiddenSize*outputSize):(hiddenSize*visibleSize)+(hiddenSize*outputSize)+hiddenSize]
    b2 = theta[(hiddenSize*visibleSize)+(hiddenSize*outputSize)+hiddenSize:]

    Z2 = np.dot(W1,data) + b1
    A2 = logistic(Z2)
    Z3 = np.dot(W2, A2) + b2
    return flatten_matrix(Z3)


