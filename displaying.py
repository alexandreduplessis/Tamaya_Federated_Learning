import numpy as np
import os
import matplotlib.pyplot as plt
# for args
import argparse

def accuracy_display(directorypath):
    merger_names = {}
    merger_accs = {}
    # Load the dictionaries in all files of directorypath with name beginning by "accuracies_"
    for filename in os.listdir(directorypath):
        if filename.startswith("accuracies_"):
            # get the string between "accuracies_" and the next "_"
            merger_name = filename.split("_")[1]
            # load the dictionary
            new_dict = np.load(os.path.join(directorypath, filename), allow_pickle=True).item()
            if merger_name not in merger_names:
                merger_names[merger_name] = 1
                print(np.mean(new_dict[0]), np.mean(new_dict[1]))
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
    plt.legend()
    plt.show()

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

def print_info(directorypath):
    # load the dictionary in the file of name "info"
    info = np.load(os.path.join(directorypath, "info.npy"), allow_pickle=True).item()
    for key in info:
        print(key, ":", info[key])

def alpha_display(directorypath):
    merger_names = {}
    merger_alphas = {}
    # Load the dictionaries in all files of directorypath with name beginning by "accuracies_"
    for filename in os.listdir(directorypath):
        if filename.startswith("alphas_"):
            # get the string between "accuracies_" and the next "_"
            merger_name = filename.split("_")[1]
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
        plt.imshow(merger_alphas[merger_name], label=merger_name)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # all directories in outputs/ have datetime formats"%Y-%m-%d_%H-%M-%S"
    # look for the directory with the latest datetime format
    latest_string = "0000-00-00_00-00-00"
    for filename in os.listdir("outputs"):
        if filename > latest_string:
            latest_string = filename
    parser.add_argument("--dir", type=str, default=os.path.join("outputs", latest_string), help="Directory path to the results")
    parser.add_argument("--accs", type=int, default=0, help="Display the accuracies")
    parser.add_argument("--losses", type=int, default=0, help="Display the losses")
    parser.add_argument("--info", type=int, default=0, help="Display the info")
    parser.add_argument("--alphas", type=int, default=0, help="Display the alphas")
    args = parser.parse_args()
    if args.accs:
        accuracy_display(args.dir)
    if args.losses:
        loss_display(args.dir)
    if args.info:
        print_info(args.dir)
    if args.alphas:
        alpha_display(args.dir)