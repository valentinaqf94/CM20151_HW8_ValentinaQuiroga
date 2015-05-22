
# -*- coding: utf-8 -*-
#acá se realiza la parte de red neuronal en la cual se realiza un preprocesamiento de los datos. Se calcula el
#vector de entrada promedio de los 137 datos, luego a la matriz se le resta dicho promedio y se calcula la matriz
#de covarianza, esto con el fin de calcular la SVD y obtener las matrices que describen "rotaciones" y "deformaciones"
#sobre la matriz inicial, siguiendo el modelo de whitening.

from __future__ import division
import os
import cost
import minimize
import numpy as np



def flatten_matrix(mat): #Colapsa las listas a 1D
    mat = mat.flatten(1)
    mat = mat.reshape(len(mat), 1, order='F')
    return mat


def initializeParameters(outputSize, hiddenSize, visibleSize):  #Inicializa los pesos aleatorios de la red neuronal
    r = np.sqrt(6) / np.sqrt(hiddenSize+visibleSize+1)
    W1 = np.random.rand(hiddenSize, visibleSize) * 2 * r - r
    W2 = np.random.rand(outputSize, hiddenSize) * 2 * r - r
    b1 = np.zeros((hiddenSize, 1))
    b2 = np.zeros((outputSize, 1))
    return np.vstack([flatten_matrix(W1), flatten_matrix(W2), b1, b2])

def preProcess(inputs, epsilon): #Funcion que calcula la Descomposicion singular de Valores o SVD para
    #calcular la matriz de Covarianza y de Valores propios para poder eliminar informacion redundante para el modelo.
    #preprocesamiento de los datos
    m = inputs.shape[1]
    meanInput = np.mean(inputs, 1)
    meanInput = flatten_matrix(meanInput)
    inputs = inputs-meanInput
    sigma = np.dot(inputs, inputs.T)/m
    U,s,V = np.linalg.svd(sigma)
    S = np.zeros(V.shape)
    for i in range(0, V.shape[0]):
        S[i,i] = s[i]
    ZCAWhite = np.dot(np.dot(U, np.diag(1 / np.sqrt(np.diag(S) + epsilon))), U.T) 
    inputs = np.dot(ZCAWhite, inputs)
    return [inputs, meanInput, ZCAWhite]

def process_data(inputs, values): #Funcion que ejecuta la red neuronal como tal
    _beta = 2 #penalidad de la dispersión de datos, limite de dispersion del modelo
    _lambda = 1e-4 #limita la variación de los pesos o weight decay
    _epsilon = 0.1 #evita tener valores propios en la matriz iguales a cero
    _sparsityParam = 0.6 #la activación promedio deseada en cada neurona, entre 0 y 1
    num_iter = 5000 #número máximo de iteraciones

    inputSize = inputs.shape[0] #cantidad de variables de entrada, 6 en este caso
    m = inputs.shape[1]#cantidad de casos de entrenamiento
    hiddenSize = 180 #cantidad de neuronas ocultas, ocultas porque no se sabe bien que hacen
    outputSize = 1 #las dimensiones de salida, en este caso, 1, porque es un problema de regresión

    theta = initializeParameters(outputSize, hiddenSize, inputSize) #inicializa los pesos y los sesgos de la red
    #y retorna un vector de dimension hidden*input + hidden*output + hidden + output
    inputs, meanInput, ZCAWhite = preProcess(inputs, _epsilon)# inicialización de los parámetros
    #retorna números aleatorios como una primera aproximacion
    costF = lambda p: cost.sparseLinearNNCost(p, inputSize, hiddenSize, outputSize, _lambda, _sparsityParam, _beta, inputs, values) #define la función de costo, la cual recibe por parámetro al vector de parámetros theta

    optTheta,costV,i = minimize.minimize(costF,theta,maxnumlinesearch=num_iter)
    pred = cost.predict(inputs, optTheta, inputSize, hiddenSize, outputSize)

    diff = np.linalg.norm(pred-values)/np.linalg.norm(pred+values) #peso de los parametros

    print "RMSE: %g" % (diff)
    

    np.savez('parameters.npz', optTheta = optTheta, meanInput = meanInput, ZCAWhite = ZCAWhite)

  
def predict(inputs):
    visibleSize = inputs.shape[0]
    hiddenSize = 180
    outputSize = 1
    parameters = np.load('parameters.npz')
    meanInput = parameters['meanInput']
    ZCAWhite = parameters['ZCAWhite']
    optTheta = parameters['optTheta']
    inputs = inputs - meanInput
    inputs = np.dot(ZCAWhite, inputs)
    values = cost.predict(inputs, optTheta, visibleSize, hiddenSize, outputSize)
    return values


