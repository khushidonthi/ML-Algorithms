#KNN ALGORITHM-supervised machine learning algorithm,requires training set and test set

import random as r
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from scipy import stats
from stats import mode
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_dist(x1,y1,x2,y2):
    return ((x2-x1)**2 + (y2-y1)**2)**0.5

def newcoords(cluster):
    sumx,sumy=0,0
    for i in cluster:
        sumx+=i[0]
        sumy+=i[1]
    newx = sumx/len(cluster)
    newy = sumy/len(cluster)
    return [newx,newy]

def check(existingx,existingy,newx,newy):
    if existingx==newx and existingy==newy:
        return 1
    else:
        return 0

sum_n,sum_f,li=0,0,[]
newc1,newc2=[0,0],[0,0]
healthy,fever,y1,y2,y=0,0,[],[],[]

i=0
while(i<98):
    normaltemp = round((r.randrange(97,99) + r.random()),2)
    fevertemp = round(r.randrange(99, 104) + r.random(), 2)
    sum_n+=normaltemp
    sum_f+= fevertemp
    li.append([i,normaltemp])
    i+=1
    li.append([i, fevertemp])
    i+=1
    y1.append(normaltemp)
    y2.append(fevertemp)
    y.append(normaltemp)
    y.append(fevertemp)
    healthy += normaltemp
    fever += fevertemp

avg_temp=sum(y) / len(y)
print("AVERAGE TEMPERATURE:",avg_temp,'\n\n')
print("y:",y,'\n\n')
print("y1:",y1,)
print(len(y1),'\n\n')
print("y2:",y2)
print(len(y2),'\n\n')
print("----->",li,'\n\n')
x=[]
for i in range(100):
    x.append(i)

cluster1, cluster2 = [], []
for a in range(len(li)):
    if li[a][1] < avg_temp:
        cluster1.append(li[a])
    else:
        cluster2.append(li[a])


mean_healthy=healthy/49     #Mean healthy teamperature
v=np.var(y1)                #Variance
s=np.std(y1)                #Standard deviation
med=np.median(y1)           #Median
mod = stats.mode(y1)        #Mode

#for j in range(50):     # assuming 20% of the population would have a fever

mean_fever=fever/49         #Mean fever temperature
v1=np.var(y2)               #Variance
s1=np.std(y2)               #Standard deviation
med1=np.median(y2)          #Median
mod1 = stats.mode(y2)       #Mode

# cntr = r.sample(li,2)
# existingc1,existingc2 = cntr[0],cntr[1]

# while True:
#     cluster1,cluster2=[],[]
#     for a in li:
#         x1, y3 = a[0], a[1]
#         d1 = get_dist(x1, y3, existingc1[0], existingc1[1])
#         d2 = get_dist(x1, y3, existingc2[0], existingc2[1])
#         if d1 < d2:
#             cluster1.append(a)
#         else:
#             cluster2.append(a)
#     newc1 = newcoords(cluster1)
#     newc2 = newcoords(cluster2)
#
#     if check(existingc1[0], existingc1[1],newc1[0],newc1[1]) and check(existingc2[0], existingc2[1],newc2[0],newc2[1]):
#         break
#     existingc1 = newc1
#     existingc2 = newc2
# print('\n\nfinal centroid coordinates are:\n', newc1, newc2, sep='\t\t',end='\n\n')


xc1,yc1,xc2,yc2=[],[],[],[]
for d in cluster1:
    xc1.append(d[0])
    yc1.append(d[1])
for b in cluster2:
    xc2.append(b[0])
    yc2.append(b[1])

y_cluster = []

for l in range(len(cluster1)):
    if cluster1[l][1] in y1:
        y_cluster.append("NORMAL TEMPERATURE")
    else:
        y_cluster.append("FEVER TEMPERATURE")

for m in range(len(cluster2)):
    if cluster2[m][1] in y2:
        y_cluster.append("FEVER TEMPERATURE")
    else:
        y_cluster.append("NORMAL TEMPERATURE")

print(cluster1,len(cluster1))
print(cluster2,len(cluster2))
#two types of data scaling methods:normalization and standardization
def kneighbors(cntr,n_neighbors):
    d2 = []
    dict={}
    for a in li:
        x1, y3 = a[0], a[1]
        d1 = get_dist(x1, y3, cntr[0], cntr[1])
        d2.append(d1)
        dict[x1,y3]=d1
    d2.sort()
    import operator
    sorted_x = sorted(dict.items(), key=operator.itemgetter(1))
    print("++++",sorted_x)

    count1 = 0
    count2 = 0
    for k in range(n_neighbors):
        print("====",d2[k])
        print("####",sorted_x[k][0])
        plt.scatter(sorted_x[k][0][0],sorted_x[k][0][1], marker='x')

        if sorted_x[k][0][1] in yc1:
            count1+=1
        else:      #sorted_x[k][0][1] in y2:
            count2+=1
    print(count1,count2)
    if count1>count2:
        output="NORMAL TEMPERATURE"
        print("THE PATIENT HAS NORMAL TEMPERATURE")
    else:
        output="FEVER TEMPERATURE"
        print("THE PATIENT HAS FEVER TEMPERATURE")
    y.append(cntr[1])
    y_cluster.append(output)

kneighbors([20,102],10)

#FOR KNN ALGORITHM:
x_dataframe=pd.DataFrame(y) #contains attributes
y_dataframe=pd.DataFrame(y_cluster) #contains labels
print(x_dataframe,y_dataframe)

x_train,x_test,y_train,y_test=train_test_split(x_dataframe,y_dataframe,test_size=0.2) #80% of data-training set and 20% of data-test set(Validation Set Approach)

# print("y$$$$",y,len(y))
# print("y_cluster$$$$",y_cluster,len(y_cluster))
#print("Classification:",y_cluster)

# scaling features for uniform evaluation (data normalizaion method)
scaler=StandardScaler()
scaler.fit(x_train)#normalized data-used to estimate minimum and maximum observable values
x_train=scaler.transform(x_train)#applying scaling to trained data
x_test=scaler.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=10)  #ideal number of neighbors=(100)**0.5
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
# print("x_test",x_test,len(x_test))    #20 entries
# print("x_train",x_train,len(x_train))   #80 entries
print("y_pred",y_pred,len(y_pred))     #predicting 20 entries(test set) class as list
print("y_test",y_test,len(y_test))     #predicting 20 entries(test set) class as a data frame

y_testlist = y_test.values.tolist()
print(y_testlist)
cntt=0
# for ii in y_testlist:
#     for jj in y_pred:
#         if jj==ii[0]:
#             cntt+=1
# print('LOOK HERE',cntt)

# for jj in range(len(y_pred)):
#     for ii in range(len(y_testlist)):
#         for kk in ii:
#             if kk==jj:
#                 cntt+=1
# print('LOOK HERE',cntt)


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print("SCORE(compares y_test and y_pred)",classifier.score(x_test,y_test))

# assigning data
mydata = [["Healthy", "0.5",mean_healthy,med,mod,v,s],
          ["Fever", "0.5",mean_fever,med1,mod1,v1,s1]]

# creating header
head = ["X", "P(X)","Mean","Median","Mode","Variance","Standard Deviation"]

# displaying table
print("PROBABILITY DISTRIBUTION/PMF:")
print(tabulate(mydata, headers=head, tablefmt="grid"))

plt.ylim(96.5,104.5)
plt.scatter(xc1,yc1,c='g',marker='.')
plt.scatter(xc2,yc2,c='r',marker='.')
plt.scatter(newc1[0],newc1[1],c='cyan',marker='D')
plt.scatter(newc2[0],newc2[1],c='cyan',marker='D')
plt.scatter(20,102,marker='+')
plt.xlabel("Number of patients")
plt.ylabel("Temperatures")
plt.show()