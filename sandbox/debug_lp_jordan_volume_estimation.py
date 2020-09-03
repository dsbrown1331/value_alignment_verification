import numpy as np
from pathlib import Path
from src.linear_programming import is_redundant_constraint, remove_redundant_constraints
from scipy.spatial import distance


def mc_volume(H, num_samples, rand_seed=None):
    #estimate volume of intersection of halfspaces via MC sampling

    if seed:
        np.random.seed(seed)
    num_constraints, num_vars = H.shape
    
    r = 1 - 2*np.random.rand(num_vars, num_samples)
    #print(r)
    bool_check = np.dot(H, r)>0
    check_cols = np.sum(bool_check.astype(int), axis=0)
    intersection = np.sum(check_cols == num_constraints)
    good_samples = r[:,check_cols == num_constraints]

    #return volume estimate and samples inside intersection
    return intersection / num_samples, good_samples

def mc_volume_normal(H, num_samples, mean_r, stdev, rand_seed=None):
    #sample from a gaussian centered at mean_r with standard deviation stdev and see how many in intersection of halfspaces
    if seed:
        np.random.seed(seed)
    num_constraints, num_vars = H.shape
    
    r = mean_r.reshape(len(mean_r),1) + stdev * np.random.randn(num_vars, num_samples)
    #print(r)
    bool_check = np.dot(H, r)>0
    check_cols = np.sum(bool_check.astype(int), axis=0)
    intersection = np.sum(check_cols == num_constraints)
    good_samples = r[:,check_cols == num_constraints]
    
    #return proportion of samples inside intersection along with the actual samples in intersection
    return intersection / num_samples, good_samples





reward = np.load(Path("/home/dsbrown/Code/batch-active-preference-based-learning/preferences/reward.npy"))
print("reward", reward)
print(reward.shape)
normals = np.load(Path("/home/dsbrown/Code/batch-active-preference-based-learning/preferences/psi.npy"))
print("normals", normals.shape)
preferences = np.load(Path("/home/dsbrown/Code/batch-active-preference-based-learning/preferences/s.npy"))
print("preferences", preferences.shape)

#brute force estimation of halfspace constraint volume
num_mc_samples = 200000 #number of MC samples to estimate volume (if two sets of halfspaces have same volume they are probably the same)
seed = 1234  #make sure we sample exactly the same MC samples when estimating volume
stdev = 0.001 #how far to explore around the true reward when doing MC sampling
precision = 0.00001  #for removing duplicate halfspaces

def halfspace_debugging_report(H):
    bool_check = np.dot(H, reward)>0
    print("percent halfplanes containing true reward", np.sum(bool_check)/ normals.shape[0])
    prop, normal_hits = mc_volume_normal(H, num_mc_samples, reward, stdev, rand_seed=seed)
    vol, hits = mc_volume(H, num_mc_samples, rand_seed=seed)
    print("Percent samples in polytope = {}\%".format(prop*100))
    print("Volume estimate", vol)
    print("first 10 valid samples from normal dist")
    print(normal_hits[:,:10].transpose())

print()
print("="*10)
print("Using all halfspace normals")
correct_normals = np.array([preferences[i]*normals[i] for i in range(normals.shape[0])])
halfspace_debugging_report(correct_normals)

print("Removing halfspaces that are identical or zero")
preprocessed_normals = []
for n in correct_normals:
    if np.linalg.norm(n) < precision:
        continue #zero for all intents and purposes so non-binding
    already_in_list = False
    #search through preprocessed_normals for close match
    for pn in preprocessed_normals:
        if distance.cosine(n, pn) < precision:
            already_in_list = True
            break
    if not already_in_list:
        #add to list
        preprocessed_normals.append(n)
preprocessed_normals = np.array(preprocessed_normals)
print("size of preprocessed normals", preprocessed_normals.shape)
halfspace_debugging_report(preprocessed_normals)

print()
print("="*10)
print("Removing halfspaces that are redundant via LP")

non_redundant = remove_redundant_constraints(preprocessed_normals)
non_redundant = np.array(non_redundant)
print("size minimal test needed for alignment verification")
print(non_redundant.shape)
halfspace_debugging_report(non_redundant)