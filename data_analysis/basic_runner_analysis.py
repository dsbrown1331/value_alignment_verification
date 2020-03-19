import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

num_rows_list = [4,8,16]
num_cols_list = [4,8,16]
num_features_list = [2,3,4,5,6,7,8]
verifier_list = ["ranking-halfspace", "state-optimal-action-ranker","state-value-critical-0.5","state-value-critical-0.1"]
name_map = {"ranking-halfspace":"AEC-w", "state-optimal-action-ranker":"AEC-AP","state-value-critical-0.5":"CS-0.5","state-value-critical-0.1":"CS-0.1"}
exp_data_dir = "./experiment_data/"

color_lines = ['b^-', 'gs--', 'ro-.', 'kx:']


##first let's plot the number of states along the x-axis and accuracy and test size on y-axis for the different methods

#go through and calculate the means for each of these 
for num_features in num_features_list:

    all_accuracies = {}
    all_test_sizes = {}
    for v in verifier_list:
        all_accuracies[v] = []
        all_test_sizes[v] = []
    for num_rows in num_rows_list:
        num_cols = num_rows #keep it square grid for  now
        for i, verifier_name in enumerate(verifier_list):
            filename = "{}_states{}x{}_features{}.txt".format(verifier_name, num_rows, num_cols, num_features)
            print("reading from", filename)
            my_data = genfromtxt(exp_data_dir + filename, delimiter=',')
            #columns are num_correct, num_tested, size_test_set
            num_correct = my_data[:,0]
            num_tested = my_data[:,1]
            test_sizes = my_data[:,2]
            ave_accuracy = np.mean(num_correct / num_tested)
            all_accuracies[verifier_name].append(ave_accuracy)
            if verifier_name == "ranking-halfspace":
                all_test_sizes[verifier_name].append(1)
            else:
                all_test_sizes[verifier_name].append(np.mean(test_sizes))

    print(all_accuracies)
    print(all_test_sizes)

    #make plot of accuracies
    plt.rc('font', family='serif')
    fig = plt.figure()
    plt.title("Num Features = {}".format(num_features), fontsize=19)
    for i,v in enumerate(verifier_list):
        plt.plot(num_rows_list, all_accuracies[v], color_lines[i], linewidth=3, label=name_map[v])
    plt.xticks(num_rows_list,fontsize=15) 
    plt.yticks(fontsize=15) 
    plt.xlabel('Grid world width', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.legend(loc='best', fontsize=15)
    plt.tight_layout()
    plt.savefig('./data_analysis/figs/basic_features{}_accuracy.png'.format(num_features))


    #make plot of test sizes
    plt.figure()
    plt.title("Num Features = {}".format(num_features), fontsize=19)
    ax = plt.subplot(1,1,1)
    for i,v in enumerate(verifier_list):
        ax.semilogy(num_rows_list, all_test_sizes[v], color_lines[i], linewidth=3, label=name_map[v])
    #ax.tick_params(bottom=False, top=False, left=True, right=True)
    #ax.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False)
    plt.xticks(num_rows_list,fontsize=15) 
    plt.yticks(fontsize=15) 
    plt.xlabel('Grid world width', fontsize=18)
    plt.ylabel('Test questions', fontsize=18)
    plt.legend(loc='best', fontsize=15)
    plt.tight_layout()
    plt.savefig('./data_analysis/figs/basic_features{}_queries.png'.format(num_features))

#plt.show()


verifier_list = ["ranking-halfspace","state-optimal-action-ranker",  "state-value-critical-0.5","state-value-critical-0.1"]

num_features_list = [2,3,4,5,6,7,8]

##now let's look at number of features along the xaxis for a could different grid sizes
#go through and calculate the means for each of these 
for num_rows in [4,8,16]:

    all_accuracies = {}
    all_test_sizes = {}
    for v in verifier_list:
        all_accuracies[v] = []
        all_test_sizes[v] = []
    for num_features in num_features_list:
        num_cols = num_rows #keep it square grid for  now
        for i, verifier_name in enumerate(verifier_list):
            filename = "{}_states{}x{}_features{}.txt".format(verifier_name, num_rows, num_cols, num_features)
            print("reading from", filename)
            my_data = genfromtxt(exp_data_dir + filename, delimiter=',')
            #columns are num_correct, num_tested, size_test_set
            num_correct = my_data[:,0]
            num_tested = my_data[:,1]
            test_sizes = my_data[:,2]
            ave_accuracy = np.mean(num_correct / num_tested)
            all_accuracies[verifier_name].append(ave_accuracy)
            all_test_sizes[verifier_name].append(np.mean(test_sizes))

    print(all_accuracies)
    print(all_test_sizes)

    #make plot of accuracies
    plt.rc('font', family='serif')
    fig = plt.figure()
    plt.title("Grid World Width = {}".format(num_rows), fontsize=19)
    for i,v in enumerate(verifier_list):
        plt.plot(num_features_list, all_accuracies[v], color_lines[i], linewidth=3, label=name_map[v])
    plt.xticks(num_features_list,fontsize=15) 
    plt.yticks(fontsize=15) 
    plt.xlabel('Number features', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.legend(loc='best', fontsize=15)
    plt.tight_layout()
    plt.savefig('./data_analysis/figs/basic_size{}_accuracy.png'.format(num_rows))


    #make plot of test sizes
    plt.figure()
    plt.title("Grid World Width = {}".format(num_rows), fontsize=19)
    ax = plt.subplot(1,1,1)
    for i,v in enumerate(verifier_list):
        ax.semilogy(num_features_list, all_test_sizes[v], color_lines[i], linewidth=3, label=name_map[v])
    #ax.tick_params(bottom=False, top=False, left=True, right=True)
    #ax.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False)
    plt.xticks(num_features_list,fontsize=15) 
    plt.yticks(fontsize=15) 
    plt.xlabel('Number features', fontsize=18)
    plt.ylabel('Test questions', fontsize=18)
    plt.legend(loc='best', fontsize=15)
    plt.tight_layout()
    plt.savefig('./data_analysis/figs/basic_size{}_queries.png'.format(num_rows))

#plt.show()