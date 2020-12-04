import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import copy

'''First step: read the data from the frame or file, Divide the data into the cluster k and initialize 
random centroids to each cluster from the given data. 
Assign color to the centroids colmap.
'''

df = pd.DataFrame({
    'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'y': [80, 39, 40, 60, 70, 20, 10, 17, 22, 30, 40, 50, 60, 88, 22, 56, 45, 39, 11],

})

np.random.seed(30)
k =4 
#centroids[i] = [x, y]

centroids = {
    i+1: [np.random.randint(0, 20), np.random.randint(0, 100)]
    for i in range(k)
}

colmap = {1: 'r', 2:'g', 3: 'b', 4: 'y'}
review = {1: 'Average', 2: 'Good', 3: 'Bad', 4: 'Excellent'}


#Assignment state 

''' Calculate the distance from each points to the cluter's centroid and take the mean value. 
Reapeat the step until the mean value becomes stable.'''

def assignment(df, centroids):
    
    for i in centroids.keys():
        #sqrt((x1-x2)^2-(y1-y2)^2)
        df['distance_from_{}'.format(i)] = (
        np.sqrt(
            (df['x']  - centroids[i][0]) ** 2 
            + (df['y'] - centroids[i][1]) ** 2
            )
        )

        
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    df['Perform'] = df['closest'].map(lambda x: review[x])
    
    return df

df = assignment(df, centroids)


#update_stage

old_centroids = copy.deepcopy(centroids)

def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
        
    return k 

centroids = update(centroids)
        

## Repeat Assignment stage 

df = assignment(df, centroids)


while True:
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df, centroids)
    
    print("This is your final cluster!")
    
    if closest_centroids.equals(df['closest']):
        break

print(df)      

#plot the final result.

fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')

for i in centroids.keys():
    a = plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 20)
plt.ylim(0, 100)
plt.show()