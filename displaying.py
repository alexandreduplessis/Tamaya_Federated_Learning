import numpy as np
import os
import matplotlib.pyplot as plt
# for args
import argparse
# for interpolation by 1/t, 1/t^2, 1/t^3 ...
from scipy.interpolate import interp1d

def accuracy_display(directorypath):
    merger_names = {}
    merger_accs = {}
    # Load the dictionaries in all files of directorypath with name beginning by "accuracies_"
    for filename in os.listdir(directorypath):
        if filename.startswith("accuracies_"):
            # get the string between "accuracies_" and the next "_"
            merger_name = filename.split("_")[1]
            nb = (filename.split("_")[2]).split(".")[0]
            # if it is FedSoftMax500 pass
            if merger_name == "FedSoftMax500":
                continue
            # if merger_name == "FedMaxk" or merger_name == "FedMix":
            # load the dictionary
            new_dict = np.load(os.path.join(directorypath, filename), allow_pickle=True).item()
            if merger_name not in merger_names:
                merger_names[merger_name] = 1
                merger_accs[merger_name] = [np.array([np.mean(new_dict[round]) for round in range(len(new_dict))])]
            else:
                merger_names[merger_name] += 1
                merger_accs[merger_name].append([np.mean(new_dict[round]) for round in range(len(new_dict))])
    # Divide the values of merger_accs by the corresponding values of merger_names
    merger_accs_means = {}
    merger_accs_stds = {}
    for merger_name in merger_names:
        merger_accs_means[merger_name] = np.mean(merger_accs[merger_name], axis=0)
        # but take std of the accs minus the ones of corresponding fedavg
        try:
            copy_of_fedavg = np.array(merger_accs["FedAvg"])[:len(merger_accs[merger_name])]
            merger_accs_stds[merger_name] = np.std((merger_accs[merger_name]- copy_of_fedavg), axis=0)
        except:
            merger_accs_stds[merger_name] = np.std(merger_accs[merger_name], axis=0)
        
    if "FedAvg" not in merger_names:
        merger_names["FedAvg"] = 0
    # plot merger_accs[merger_name] for each merger_name
    for merger_name in merger_names:
        if merger_names[merger_name] == 0:
            continue
        plt.plot(merger_accs_means[merger_name], label=merger_name)
        # plot 95% confidence interval (don't forget square root of number of curves)
        plt.fill_between(range(len(merger_accs_means[merger_name])), merger_accs_means[merger_name] - 1.96 * merger_accs_stds[merger_name] / np.sqrt(merger_names[merger_name]), merger_accs_means[merger_name] + 1.96 * merger_accs_stds[merger_name] / np.sqrt(merger_names[merger_name]), alpha=0.1)
    plt.legend()
    plt.title("Number of curves: " + str(merger_names["FedAvg"]))
    plt.show()
    # save as png in the folder "bash"
    plt.savefig("./bash/accuracies.png")
    plt.close()

def accuracy_display_2(directorypath):
    merger_names = {}
    merger_accs = {}
    # Load the dictionaries in all files of directorypath with name beginning by "accuracies_"
    for filename in os.listdir(directorypath):
        if filename.startswith("accuracies_"):
            # get the string between "accuracies_" and the next "_"
            merger_name = filename.split("_")[1]
            # if it is FedSoftMax500 pass
            if merger_name == "FedSoftMax500":
                continue
            # if merger_name == "FedMaxk" or merger_name == "FedMix":
            # load the dictionary
            new_dict = np.load(os.path.join(directorypath, filename), allow_pickle=True).item()
            if merger_name not in merger_names:
                merger_names[merger_name] = 1
                merger_accs[merger_name] = np.array([np.mean(new_dict[round]) for round in range(len(new_dict))])
            else:
                merger_names[merger_name] += 1
                merger_accs[merger_name] += [np.mean(new_dict[round]) for round in range(len(new_dict))]
    # Divide the values of merger_accs by the corresponding values of merger_names
    for merger_name in merger_names:
        merger_accs[merger_name] /= merger_names[merger_name]
    
    # plot merger_accs[merger_name] for each merger_name
    for merger_name in merger_names:
        plt.plot(merger_accs[merger_name], label=merger_name)
        print(merger_name, merger_names[merger_name])
    plt.legend()
    plt.title("Number of curves: " + str(merger_names["FedAvg"]))
    plt.show()
    # save as png in the folder "bash"
    plt.savefig("./bash/accuracies.png")
    plt.close()

def loss_display(directorypath):
    merger_names = {}
    merger_losses = {}
    # Load the dictionaries in all files of directorypath with name beginning by "accuracies_"
    for filename in os.listdir(directorypath):
        if filename.startswith("loss_dict_"):
            # get the string between "accuracies_" and the next "_"
            merger_name = filename.split("_")[2]
            # load the dictionary
            new_dict = np.load(os.path.join(directorypath, filename), allow_pickle=True).item()
            if merger_name not in merger_names:
                merger_names[merger_name] = 1
                merger_losses[merger_name] = np.array([np.mean(new_dict[round]) for round in range(len(new_dict))])
            else:
                merger_names[merger_name] += 1
                merger_losses[merger_name] += [np.mean(new_dict[round]) for round in range(len(new_dict))]
    # Divide the values of mrger_accs by the corresponding values of merger_names
    for merger_name in merger_names:
        merger_losses[merger_name] /= merger_names[merger_name]
    
    # plot merger_accs[merger_name] for each merger_name
    for merger_name in merger_names:
        plt.plot(merger_losses[merger_name], label=merger_name)
    plt.legend()
    plt.show()
    # save as png in the folder "bash"
    plt.savefig("./bash/losses.png")
    plt.close()

def loss_train_display(directorypath):
    merger_names = {}
    merger_losses = {}
    merger_autocorrelation = {}
    # Load the dictionaries in all files of directorypath with name beginning by "accuracies_"
    for filename in os.listdir(directorypath):
        if filename.startswith("train_accuracies_"):
            # get the string between "accuracies_" and the next "_"
            merger_name = filename.split("_")[2]
            # load the dictionary
            new_dict = np.load(os.path.join(directorypath, filename), allow_pickle=True).item()
            if merger_name not in merger_names:
                merger_names[merger_name] = 1
                merger_losses[merger_name] = np.array([np.mean(new_dict[round]) for round in range(len(new_dict))])
                # compute the lag one difference
                diffs = np.diff(merger_losses[merger_name])
                # compute correlation as std of diffs divided by sthe absolute value of the mean of diffs
                merger_autocorrelation[merger_name] = np.std(diffs)/np.abs(np.mean(diffs))
            else:
                merger_names[merger_name] += 1
                # take the minimum between the two values
                merger_losses[merger_name] += [np.mean(new_dict[round]) for round in range(len(new_dict))]
                # compute autocorrelation
                diffs = np.diff(merger_losses[merger_name])
                merger_autocorrelation[merger_name] += np.std(diffs)/np.abs(np.mean(diffs))
                # plt.plot([np.mean(new_dict[round]) for round in range(len(new_dict))], label=merger_name)
                # plt.legend()
                # plt.title(merger_name)
                # plt.show()
    # Divide the values of mrger_accs by the corresponding values of merger_names
    for merger_name in merger_names:
        merger_losses[merger_name] /= merger_names[merger_name]
        merger_autocorrelation[merger_name] /= merger_names[merger_name]
        print(merger_name, merger_autocorrelation[merger_name])
    
    # plot merger_accs[merger_name] for each merger_name
    for merger_name in merger_names:
        if merger_name != "FedSoftMin":
            plt.plot(merger_losses[merger_name], label=merger_name)
    plt.legend()
    # plt.title("Number of curves: " + str(merger_names["FedAvg"]))
    plt.show()
    # save as png in the folder "bash"
    plt.savefig("./bash/train_losses.png")
    plt.close()

def loss_train_matrix_display(directorypath):
    merger_names = {}
    merger_losses = {}
    # Load the dictionaries in all files of directorypath with name beginning by "accuracies_"
    for filename in os.listdir(directorypath):
        if filename.startswith("train_accuracies_"):
            # get the string between "accuracies_" and the next "_"
            merger_name = filename.split("_")[3]
            # load the dictionary
            new_dict = np.load(os.path.join(directorypath, filename), allow_pickle=True).item()
            if merger_name not in merger_names:
                merger_names[merger_name] = 1
                merger_losses[merger_name] = [new_dict[round] for round in range(len(new_dict))]
            else:
                a = 1
    
    # plot merger_accs[merger_name] for each merger_name
    for merger_name in merger_names:
        # plot the matrix of alphas
        # print(merger_name)
        plt.imshow(merger_losses[merger_name], label=merger_name)
        plt.legend()
        plt.show()

def loss_matrix_display(directorypath):
    merger_names = {}
    merger_losses = {}
    # Load the dictionaries in all files of directorypath with name beginning by "accuracies_"
    for filename in os.listdir(directorypath):
        if filename.startswith("loss_dict_"):
            # get the string between "accuracies_" and the next "_"
            merger_name = filename.split("_")[2]
            # load the dictionary
            new_dict = np.load(os.path.join(directorypath, filename), allow_pickle=True).item()
            if merger_name not in merger_names:
                merger_names[merger_name] = 1
                merger_losses[merger_name] = [new_dict[round] for round in range(len(new_dict))]
            else:
                a = 1
    
    # plot merger_accs[merger_name] for each merger_name
    for merger_name in merger_names:
        # plot the matrix of alphas
        # print(merger_name)
        plt.imshow(merger_losses[merger_name], label=merger_name)
        plt.legend()
        plt.show()

def print_info(directorypath):
    # load the dictionary in the file of name "info"
    info = np.load(os.path.join(directorypath, "info.npy"), allow_pickle=True).item()
    # for key in info:
    #     print(key, ":", info[key])
    return info["nb_clients"]

def alpha_display(directorypath, pi):
    merger_names = {}
    merger_alphas = {}
    # Load the dictionaries in all files of directorypath with name beginning by "accuracies_"
    for filename in os.listdir(directorypath):
        if filename.startswith("alpha_dict_"):
            # get the string between "accuracies_" and the next "_"
            merger_name = filename.split("_")[2]
            # load the dictionary
            new_dict = np.load(os.path.join(directorypath, filename), allow_pickle=True).item()
            if merger_name not in merger_names:
                merger_names[merger_name] = 1
                merger_alphas[merger_name] = [new_dict[round] for round in range(len(new_dict))]
            else:
                a = 1
    
    # plot merger_accs[merger_name] for each merger_name
    for merger_name in merger_names:
        # plot the matrix of alphas
        print(merger_name)
        matrix = np.array(merger_alphas[merger_name])
        # remove first line
        # matrix = matrix[2:]
        # # remove half of the lines (keep only the first half)
        # matrix = matrix[:len(matrix)//2]
        nb_values = 10
        # for each line of matrix put the nb_values highest values to 1 and the others to 0
        # also take the sum of the alpha values for the values set to 1
        sum_list = []
        for i in range(len(matrix)):
            # get the indices of the nb_values highest values
            indices = np.argpartition(matrix[i], -nb_values)[-nb_values:]
            sum_list.append(np.sum(matrix[i][indices]))
            # set the nb_values highest values to 1
            matrix[i][indices] = 1
            # set the others to 0
            matrix[i][np.argwhere(matrix[i] != 1)] = 0

        # plt.imshow(matrix, label=merger_name)
        # plt.legend()
        # plt.show()
        # plt.plot(sum_list, label=merger_name)
        # plt.legend()
        # plt.show()
        # take the absolute value of the alphas minus pi for each value of alpha
        norm_matrix = np.abs(np.array(merger_alphas[merger_name]) - pi)
        # average over the clients
        norm_matrix_avg = np.mean(norm_matrix, axis=1)
        # take inverse
        norm_matrix_avg =  norm_matrix_avg

        plt.plot(norm_matrix_avg, label=merger_name)
    plt.legend()
    plt.show()
    # save as png
    plt.savefig("./bash/alpha_display.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # all directories in outputs/ have datetime formats"%Y-%m-%d_%H-%M-%S"
    # look for the directory with the latest datetime format
    latest_string = "0000-00-00_00-00-00"
    for filename in os.listdir("outputs"):
        if filename[0:2] == "20" and filename > latest_string:
            latest_string = filename
    parser.add_argument("--dir", type=str, default=os.path.join("outputs", latest_string), help="Directory path to the results")
    parser.add_argument("--accs", type=int, default=0, help="Display the accuracies")
    parser.add_argument("--losses", type=int, default=0, help="Display the losses")
    parser.add_argument("--info", type=int, default=0, help="Display the info")
    parser.add_argument("--alphas", type=int, default=0, help="Display the alphas")
    parser.add_argument("--loss_matrix", type=int, default=0, help="Display the loss matrix")
    parser.add_argument("--loss_train_matrix", type=int, default=0, help="Display the loss matrix")
    parser.add_argument("--loss_train", type=int, default=0, help="Display the loss matrix")
    args = parser.parse_args()
    pi = 1 / print_info(args.dir)
    if args.accs:
        accuracy_display(args.dir)
    if args.losses:
        loss_display(args.dir)
    if args.info:
        print_info(args.dir)
    if args.alphas:
        alpha_display(args.dir, pi)
    if args.loss_matrix:
        loss_matrix_display(args.dir)
    if args.loss_train_matrix:
        loss_train_matrix_display(args.dir)
    if args.loss_train:
        loss_train_display(args.dir)