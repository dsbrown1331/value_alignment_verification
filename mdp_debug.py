import mdp

#features is a 2-d array of tuples
num_rows = 2
num_cols = 3
features =[[(1, 0), (0, 1), (1, 0)],
           [(1, 0), (1, 0), (1, 0)]]
weights = [-1,-4]
initials = [(r,c) for r in range(num_rows) for c in range(num_cols)] #states indexed by row and then column
#print(initials)
terminals = [(0,0)]
gamma = 0.9

world = mdp.LinearFeatureGridWorld(features, weights, initials, terminals, gamma)

print("rewards")
world.print_rewards()

print("values")
V = mdp.value_iteration(world)
print(V)
print(world.to_grid(V))
world.print_2darray(world.to_grid(V))
world.print_map(V)

Q = mdp.compute_q_values(world, V)
print("Q-values")
print(Q)

print("optimal policy")
opt_policy = mdp.find_optimal_policy(world, Q=Q)
print(opt_policy)
print(world.print_map(world.to_arrows(opt_policy)))