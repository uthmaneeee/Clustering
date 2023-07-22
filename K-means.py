import matplotlib
import matplotlib.pyplot as plt 
from sklearn import datasets
import numpy as np
from numpy.linalg import norm

colors =['r','b','g','c','m','o']
n_colors = 6
def my_kmeans(X,K,Visualisation=False,Seuil=0.001,Max_iterations = 100):
    
    N,p = np.shape(X)
    iteration = 0        
    Dist=np.zeros((K,N))
    J=np.zeros(Max_iterations+1)
    J[0] = 10000000
    
    # Initialisation des clusters
    # par tirage de K exemples, pour tomber dans les données     
    Index_init = np.random.choice(N, K,replace = False)
    C = np.zeros((p,K))
    for k in range(K):
        C[:,k] = X[Index_init[k]].T
        

    while iteration < Max_iterations:
        iteration +=1
        #################################################################
        # E step : estimation des données manquantes 
        #          affectation des données aux clusters les plus proches

        for i in range(K):
            # Dist[i,:]=np.linalg.norm(X-C[:,i])
            Dist[i,:] = np.square(norm(X - C[:,i],axis=1))
            
        y = np.argmin(Dist,axis=0)

        J[iteration] += np.sum(np.min(Dist[y,:],axis=0))/N # Critière variance intra totale

        if Visualisation:
            fig = plt.figure(iteration+10, figsize=(8, 6))
            for k in range(K):
                plt.plot(X[y==k, 0], X[y==k, 1], colors[k%n_colors]+'o')
            plt.plot(C[0, :], C[1, :],'kx')
            plt.show()
        
        ################################################################
        # M Step : calcul des meilleurs centres   
        #nouveau clusters
        for k in range(K):
            C[:,k] = np.mean(X[y==k,:],axis=0)
            
        if np.abs(J[iteration]-J[iteration-1])/J[iteration-1] < Seuil:
            break 
        
        
            

        


        # test de convergence

    if Visualisation:
        fig = plt.figure(figsize=(8, 6))
        plt.plot(J[1:iteration]/N, 'o-')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.show()
            
    return C, y, J[1:iteration]/N


if __name__ == '__main__':

#########################################################
#''' K means '''
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    K= 3


    fig = plt.figure(2, figsize=(8, 6))
    plt.clf()
    plt.scatter(X[0:50, 0], X[0:50, 1], cmap=plt.cm.Set1,edgecolor='k',label=iris.target_names[0])
    plt.scatter(X[50:100, 0], X[50:100, 1], cmap=plt.cm.Set1,edgecolor='k',label=iris.target_names[1])
    plt.scatter(X[100:150, 0], X[100:150, 1], cmap=plt.cm.Set1,edgecolor='k',label=iris.target_names[2])
    
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.legend(scatterpoints=1)


    Cluster, y, Critere = my_kmeans(iris.data,K)#,Visualisation = True)
    
    print("centre 1 : " + str(Cluster[:,0]))
    
    print("centre 2 : " + str(Cluster[:,1]))

    print("centre 3 : " + str(Cluster[:,2]))

    
    
    fig = plt.figure(3, figsize=(8, 6))
    for k in range(K):
        plt.plot(X[y==k, 0], X[y==k, 1], colors[k%n_colors]+'o')
    plt.plot(Cluster[0, :], Cluster[1, :],'kx')
    plt.title('K moyennes ('+str(K)+')')
    plt.show()
    
    fig = plt.figure(figsize=(8, 6))
    plt.plot(Critere, 'o-')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Evolution du critère')
    plt.show()
        
