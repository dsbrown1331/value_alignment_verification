import numpy as np

#this is the correct one I think..
seed = 123
np.random.seed(seed)

#start with some rewards 
epsilon = 0.01
dim = 3
R_star = 10*np.random.rand(dim) - 5
print("R_star", R_star)
R_star_min = np.min(R_star)
R_star_max = np.max(R_star)

R_star_normalized = (R_star - R_star_min)/(R_star_max - R_star_min)
R_star_min_idx = list(R_star).index(R_star_min)
R_star_max_idx = list(R_star).index(R_star_max)
print(R_star_min_idx)
print(R_star_max_idx)



print("R_star normalized", R_star_normalized)
#create tests 
u_L = R_star_normalized  - epsilon 
u_R = R_star_normalized + epsilon 
# print("L", u_L * R_star_max + (1-u_L)*R_star_min)
# print("U", u_R * R_star_max + (1-u_R)*R_star_min)

max_norm = 0
for i in range(100000):
    #print(i)
    #sample R
    R = 10*np.random.rand(len(R_star)) - 5
    for s in range(dim):
        if s not in [R_star_max_idx, R_star_min_idx]:
            #print(u_L[s] * R[R_star_max_idx] + (1-u_L[s])*R[R_star_min_idx] >= R[s])
            #print(R[s] >= u_R[s] * R[R_star_max_idx] + (1-u_R[s])*R[R_star_min_idx])
            # input()
            
            R_max = np.max(R)
            R_min = np.min(R)
            
            normalized_R = (R - R_min)/(R_max-R_min)

            if u_L[s] * normalized_R[R_star_max_idx] + (1-u_L[s])*normalized_R[R_star_min_idx] >= normalized_R[s] or normalized_R[s] >= u_R[s] * normalized_R[R_star_max_idx] + (1-u_R[s])*normalized_R[R_star_min_idx]:
                break
            else:
                #if we make it this far we are certified
                
                if np.linalg.norm(normalized_R - R_star_normalized, np.inf) > epsilon:
                    print(R)
                    print(normalized_R)
                    input("found counter example!")
                else:
                    #print(R)
                    print(normalized_R)
                    if np.linalg.norm(normalized_R - R_star_normalized, np.inf) > max_norm:
                        max_norm = np.linalg.norm(normalized_R - R_star_normalized, np.inf)
                    #input()
print(max_norm)
            # input()
        