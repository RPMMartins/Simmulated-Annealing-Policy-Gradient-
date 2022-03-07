import numpy as np
import pandas as pd
import random

from scipy.spatial import distance_matrix

#Number of nodes in the planar graph
dim_matrix=100

#default features
features={'lags': 15, 'rolling_mean': True,
        'volatility': True,  'bias': True}


#function that finds the size of the features used
def size_features(features):
    
    size=features['lags']

    if features['rolling_mean']:
        size +=2

    if features['volatility']:
        size +=1


    if features['bias']:
        size +=1

    return size


#function that generates random weights on the features of the state space
def random_theta(size_features):

    return np.random.uniform(-1,1,size_features)

#generate random graph
def generate_plane_graph(size_graph=20):

    #generate the size_graph array of random nodes onto a 2d plane 
    df =pd.DataFrame(np.random.uniform(-1,1,size=(size_graph,2)))

    #compute the size_graph x size_graph distance matrix
    #from the random arrays of m 2-dimensional nodes
    df = pd.DataFrame(distance_matrix(df.values, df.values), index=df.index, columns=df.index)

    return df

#generate random path given the number of nodes in the graph
def generate_path(size_graph=20):

    path=np.arange(0, size_graph)
    np.random.shuffle(path)
    
    return path

#function that given a path in a planar graph
#gives out its cost
def cost_function(path,graph):
    total=0
    for i in range(1,len(path)):
        total +=graph[path[i-1]][path[i]]
    total +=graph[path[0]][path[len(path)-1]]
    
    return total


#Sigma function that summarises the state into the selected features
def Sigma(costs,features):
        
        #get the first-order difference of the costs
        diff_costs=np.diff(costs)

        #set the Sigma vector
        Sigma=np.zeros(size_features(features))
        lags=features['lags']

        if lags !=0:
            
                Sigma[0:lags]=diff_costs[-lags:]

        if features['rolling_mean']:
            Sigma[lags]= diff_costs[-5:].mean()
            Sigma[lags+1]= diff_costs[-20:].mean()
        else: 
            Sigma[lags]=0
            Sigma[lags+1]= 0

        if features['volatility']:
            Sigma[lags+2]= diff_costs[-20:].std()
        else: 
            Sigma[lags+2]= 0
        
        Sigma[lags+3]=1

        return Sigma

#function that gets the action and the new temperature
#for the simulated annealing
def action(sigma,theta,temp=1):

    a=np.random.normal(np.inner(sigma,theta), 1)
    temp+= a
    temp=min(1,max(temp,0.01))
    return temp,a

from itertools import accumulate

def weighted(rewards,gamma=1):
    
    reversed_rewards = rewards[::-1] #list reversal
    acc = list(accumulate(reversed_rewards, lambda x,y: x*gamma + y))
    return np.array(acc[::-1]) 

#function that given a path and its cost value it returns a new path, with new cost value and the difference of the cost values
#the new path is created by switching two random nodes
def new_random_path(path,cost,graph):
    new_path=path.copy()
    
    #pick two random node locations
    x =random.sample(range(len(path)), 2)
    #switch the two random nodes
    new_path[x] = new_path[[x[1],x[0]]]
    
    #pick the nodes that were switched
    
    x1,x0=path[x]
    
    #select the neighboring nodes of the previously selected two nodes
    if x[0] ==0:
        x0_m,x0_p=path[[len(path)-1,1]]
    
    elif x[0] == len(path)-1:
        x0_m,x0_p=path[[x[0]-1,0]]

    else:    
        x0_m,x0_p=path[[x[0]-1,x[0]+1]]
    
    if x[1] == len(path)-1:
        x1_m,x1_p=path[[x[1]-1,0]]

    elif x[1] ==0:
        x1_m,x1_p=path[[len(path)-1,1]]

    else:    
        x1_m,x1_p=path[[x[1]-1,x[1]+1]]

    #compute the distances of the selected nodes with their neighbouring nodes
    dist_original=graph[x0][x0_m]+graph[x0][x0_p]+graph[x1][x1_m]+graph[x1][x1_p]
    new_dist=graph[x1][x0_m]+graph[x1][x0_p]+graph[x0][x1_m]+graph[x0][x1_p]
    
    diff_cost=new_dist-dist_original
    new_cost=cost +diff_cost
    
    return new_path, cost_function(new_path,graph), cost_function(new_path,graph)-cost

#function that applies the annealing process 
from scipy.stats import bernoulli

def anneling(temp,path, cost,best_path,best_cost,graph,n=10):
    
    #repeat the annealing process n times
    for i in range(n):
        #print(i)
        #print(cost)    
        new_path,new_cost,diff_cost=new_random_path(path,cost,graph)
        
        #if the new path is better than the best path, replace the current path and the best path
        if new_cost <best_cost:
            best_cost=new_cost
            best_path=new_path  
            cost=new_cost
            path=new_path 
    
        #replace the path, if the new path as a lower cost function
        elif diff_cost <0:
            cost=new_cost
            path=new_path
    
        #replace the path, with probability of exp(-diff_cost/temp), by the simulated annealing processure
        else:
            if bernoulli.rvs(size=1,p=np.exp(-diff_cost/temp))==1:
                cost=new_cost
                path=new_path
              #  print('worse choosen')
    
    return  path, cost, best_path,best_cost


def episode(features,theta,
            size_graph=dim_matrix,size_ep=500,
            alpha=0.000001,gamma=0.90):

     ###### Set lists with the history of the episode  ###### 

    #list of actions taken in throughout the episode
    actions=np.zeros(size_ep-1)
    #list of the cost of each path throughout the episode
    costs=np.zeros(size_ep+50)
    #list of rewards throughout the episode (diference list of the list of costs)
    rewards=np.zeros(size_ep-1)
    #list of expected values throughout the episode
    expected_values=np.zeros(size_ep-1)
    #list of sigmas throughout the episode
    sigmas=[]

    ##### initialise the episode ######

    #Generate a random plane graph
    graph=generate_plane_graph(size_graph)
    #generate an initial first path
    current_path=best_path=initial_path=generate_path(size_graph)
    #get the cost of the initial path in the graph
    current_cost=best_cost=initial_cost=cost_function(current_path,graph)

    #fill the first 50 elemts of the cost list with the same value
    costs[0:50]=[current_cost]*50

    #####  run the episode  #####
    temp=1
    for k in range(size_ep-1):

        sigmas.append(Sigma(costs,features))
        temp,a =action(sigmas[k],theta,temp)

        #update the list of actions
        actions[k]=a
        (current_path,   current_cost,
        best_path,       best_cost)  = anneling(temp,current_path,
                                                current_cost,best_path,
                                                best_cost,graph,n=1)
        costs[50+k]=current_cost

    
    #get all the rewards throughout the episode (the difference of the cost list)
    rewards=-np.diff(costs[-size_ep:])
    rewards[size_ep-2]+=initial_cost-best_cost
    
    #get the expect returns list (the weighted sum of the rewards)
    expected_returns=weighted(rewards,gamma)
    
    for k in range(size_ep-1):
        SIGMA=sigmas[k]
        a=actions[k]
        v=expected_returns[k]
        log_regression=(a-np.inner(SIGMA,theta))*SIGMA

        theta+=alpha*v*log_regression

    #update the theta
    print(theta)      

    print(f'''  Initial cost: {initial_cost}
    Best Cost: {best_cost}
    Reduction Percentange: {round(100*(initial_cost-best_cost)/initial_cost,3) }%''')

    #return the new improved theta
    return theta,initial_cost-best_cost


#### run one single episode



def REINFORCE(features, number_ep=100):
    #initialize a random theta
    theta=random_theta(size_features(features))

    #list of improviments of every episode
    diffs=np.zeros(number_ep)

    print(theta)

    for k in range(number_ep):
        theta,diff=episode(features,theta)

        diffs[k]=diff
    
    print(theta)
    

def render(page):

    if page == "menu":
        Input=input("""
          ########  MAIN MENU ########

1) Train a new Model.
2) Apply Model to a Graph.          (Work in Progress)
3) Generate a new Random Graph.     (Work in Progress)
4) Settings.                        (Work in Progress)
5) About Program.                   (Work in Progress)

0) Close Program: """)



    return Input



#REINFORCE(features)

#####################################################
##################  Main Function  ##################
#####################################################

def main():
    ### print welcome text
    print("""
#####################################################
                SIMMULATED ANNEALING 
            USING REINFORCEMENT LEARNING 
                 (WORK IN PROGRESS)
#####################################################""")
    ### render main menu
    Input='1'

    while Input != '0':

        Input=render("menu")

        if Input == "1":

            REINFORCE(features, number_ep=100)
        elif Input !='0':
            print("""
#####################################################
                    Work In Progress
#####################################################
""")
        
    print("""#####################################################
                    PROGRAM OVER
#####################################################""")

###################################################################
#######################   Run the Program   #######################
###################################################################

#Run the Script
if __name__ == '__main__':            
    main()