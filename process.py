
# -*- coding: utf-8 -*-
#Estas funciones toman los archivs de entrada y constuyen una matriz que relaciona las variables con los casos
#es decir 6 x 137. Posterior a eso, se crea un vector de numero de casos lineal.
import os
import sys
import glob
import linear
import cPickle
import numpy as np

def predict():
   f = glob.glob('*.d')#.d Es el diccionario que toma un clima y le asigna un número Se genera en
#cuanto procesas los datos

   if 'states.d' not in f:
       raise "No data has been processed, cannot express any prediction."

   with open('states.d', 'rb') as fp:
        states = cPickle.load(fp)

   with open('test_set.csv', 'rb') as fp:
        lines = fp.readlines()

   lines = lines[1:]  #retorna lista
   data = [line[0:line.find('\n')].split(',') for line in lines]
   dates = map(lambda x: x[0], data)
   data = [f[1:] for f in data]
   inputs = np.zeros((len(data[0]), len(data)))

   for i,input in enumerate(data): #retorna lista
       l = map(lambda x: float(x), input[0:-1])
       l.append(states[input[-1]])
       inp = np.array([l])
       inputs[:, i] = inp

   values = linear.predict(inputs) #guarda los datos de prediccion en el formato esperado
   
   with open('output.csv', 'wb') as fp:
        fp.write('"fecha", "conteo_ordenes"\n')
        for i in range(0, len(dates)):
            line = '"'+dates[i]+'"'+','+str(int(np.round(values[i][0])))+'\n'
            fp.write(line)


def process():
   states = {}
   count = 0

   with open('training_set.csv') as fp:
        lines = fp.readlines()

   lines = lines[1:]
   data = [line[0:line.find('\n')].split(',')[1:] for line in lines]
   inputs = np.zeros((len(data[0])-1, len(data)))
   values = np.zeros((len(data), 1))

   for i,input in enumerate(data): 
       l = [float(input[0])] 
       l += map(lambda x: float(x), input[2:-1]) 
       try: 
         l.append(states[input[-1]]) 
       except KeyError: 
         states[input[-1]] = count 
         l.append(count) 
         count += 1 
       inp = np.array([l]) 
       inputs[:, i] = inp 
       values[i] = float(input[1])

   linear.process_data(inputs, values)

  
   with open('states.d', 'wb') as fp:
        cPickle.dump(states, fp)
   


if __name__ == '__main__': #guarda los datos de prediccion en el formato esperado
   if sys.argv[1] == 'process':
      process()
   elif sys.argv[1] == 'predict':
      predict()


