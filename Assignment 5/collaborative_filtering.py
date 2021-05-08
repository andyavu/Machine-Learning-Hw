#-------------------------------------------------------------------------
# AUTHOR:           Andy Vu
# FILENAME:         collaborative_filtering.py
# SPECIFICATION:    Read file to make user-based recommendations
# FOR:              CS 4210 - Assignment #5
# TIME SPENT:       05/04/2021-05/08/2021
#-------------------------------------------------------------------------

# importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('trip_advisor_data.csv', sep=',', header=0) #reading the data by using the Pandas library ()

# iterate over the other 99 users to calculate their similarity with the active user (user 100) according to their category ratings (user-item approach)
# do this to calculate the similarity:
# vec1 = np.array([[1,1,0,1,1]])
# vec2 = np.array([[0,1,0,1,1]])
# cosine_similarity(vec1, vec2)
# do not forget to discard the first column (User ID) when calculating the similarities
columns = list(df.columns)
columns.pop(0)
columns.pop(columns.index("galleries"))
columns.pop(columns.index("restaurants"))
cos_sim_list = []
for i in range(99):
    vec1 = np.array([df[columns].loc[i, :]])
    vec2 = np.array([df[columns].loc[99, :]])
    cos_sim = cosine_similarity(vec1, vec2)
    cos_sim_list.append((cos_sim, df.loc[i, :]))

# find the top 10 similar users to the active user according to the similarity calculated before
cos_sim_list.sort(key=lambda x: -x[0])
top_users = cos_sim_list[:10]

sim_sum = 0
sum_galleries = 0
sum_restaurants = 0
# Compute a prediction from a weighted combination of selected neighbors’ for both categories evaluated (galleries and restaurants)
for sim, user in top_users:
    # sim = sim[0][0]
    # sim_sum += sim
    sim_sum += sim[0][0]
    temp = [float(x) for x in user[1:]]
    sum_galleries += (sim[0][0] * (float(user["galleries"]) - sum(temp)/len(temp)))
    sum_restaurants += (sim[0][0] * (float(user["restaurants"]) - sum(temp)/len(temp)))

user_avg = sum(df[columns].loc[99, :])/len(df[columns].loc[99, :])
gallery = round(user_avg + (sum_galleries / sim_sum))
restaurant = round(user_avg + (sum_restaurants / sim_sum))

print("Gallery:", gallery)
print("Restaurant:", restaurant)
