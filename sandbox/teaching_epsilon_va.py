import numpy as np

#start with some rewards 
epsilon = 0.01
dim = 3
R_star = 10*np.random.rand(dim) - 5
print("R_star", R_star)

R_star_min = np.min(R_star)
R_star_max = np.max(R_star)
R_star_min_idx = list(R_star).index(R_star_min)
R_star_max_idx = list(R_star).index(R_star_max)
print(R_star_min_idx)
print(R_star_max_idx)
R_star_normalized = (R_star - R_star_min)/(R_star_max - R_star_min)
print("R_star normalized", R_star_normalized)
#create tests 
u_L = (R_star - R_star_min)/(R_star_max - R_star_min)  - epsilon / (R_star_max - R_star_min)
u_R = (R_star - R_star_min)/(R_star_max - R_star_min) + epsilon / (R_star_max - R_star_min)
print("L", u_L * R_star_max + (1-u_L)*R_star_min)
print("U", u_R * R_star_max + (1-u_R)*R_star_min)

max_norm = 0
for i in range(10000):
    #sample R
    R = 10*np.random.rand(len(R_star)) - 5
    R_max = np.max(R)
    R_min = np.min(R)
    if (u_L * R[R_star_max_idx] + (1-u_L)*R[R_star_min_idx] < R).all() and (R < u_R * R[R_star_max_idx] + (1-u_R)*R[R_star_min_idx]).all():
        
        normalized_R = (R - R_min)/(R_max-R_min)
        if np.linalg.norm(normalized_R - R_star_normalized, np.inf) > epsilon:
            print(R)
            print(normalized_R)
            input()
        else:
            print(normalized_R)
            if np.linalg.norm(normalized_R - R_star_normalized, np.inf) > max_norm:
                max_norm = np.linalg.norm(normalized_R - R_star_normalized, np.inf)
print(max_norm)
            # input()
        