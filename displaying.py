import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def plot(filename):
    with open(filename) as f:
        # for each line
        count = 0
        min_count = 0
        max_count = 0
        for line in f:
            # if line starts with "[FedAvg:gain]"
            if line.startswith("[FedAvg:gain]"):
                # then convert the rest of the line into list of floats (numbers are separated by ",")
                count += 1
                if count == 1:
                    avg_gains = np.array([float(x) for x in line.split("[FedAvg:gain]")[1].split(",")])
                else:
                    avg_gains += np.array([float(x) for x in line.split("[FedAvg:gain]")[1].split(",")])
            elif line.startswith("[FedSoftmin:gain]"):
                # then convert the rest of the line into list of floats (numbers are separated by ",")
                min_count += 1
                if min_count == 1:
                    min_gains = np.array([float(x) for x in line.split("[FedSoftmin:gain]")[1].split(",")])
                else:
                    min_gains += np.array([float(x) for x in line.split("[FedSoftmin:gain]")[1].split(",")])
            elif line.startswith("[FedSoftmax:gain]"):
                # then convert the rest of the line into list of floats (numbers are separated by ",")
                max_count += 1
                if max_count == 1:
                    max_gains = np.array([float(x) for x in line.split("[FedSoftmax:gain]")[1].split(",")])
                else:
                    max_gains += np.array([float(x) for x in line.split("[FedSoftmax:gain]")[1].split(",")])
    # plot the list of floats
    plt.plot(avg_gains/count, label="FedAvg")
    try:
    	plt.plot(min_gains/min_count, label="FedSoftMin")
    except:
    	print("no min")
    try:
    	plt.plot(max_gains/max_count, label="FedSoftMax")
    except:
    	print("no max")
    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Gain")
    plt.title(f"Gain averaged over {count} curves for {filename.split('_')[0].split('/')[1]}")
    plt.savefig("outputs/" + filename.split('_')[0].split('/')[1] + "_gain.png")
    plt.show()

def plot_real_accuracy(filename):
    with open(filename) as f:
    	for line in f:
    	    if line.startswith("[FedAvg:pi]"):
    	    	pi = np.array([float(x) for x in line.split("[FedAvg:pi]")[1].split(",")])
    pi = pi/np.sum(pi)
    with open(filename) as f:
        # for each line
        count = 0
        small_count = 0
        min_count = 0
        small_min_count = 0
        max_count = 0
        small_max_count = 0
        local_count = 0
        small_local_count = 0
        for line in f:
            # if line starts with "[FedAvg:gain]"
            if line.startswith("[FedAvg:global:") and not line[15] == 'g':
                # then convert the rest of the line into list of floats (numbers are separated by ",")
                i0 = 15
                client_string = ""
                while line[i0] != ']':
                    client_string += line[i0]
                    i0 += 1
                client = int(client_string)
                small_count += 1
                if client == 0:
                    count += 1
                if small_count == 1:
                    avg_gains = pi[client]*np.array([float(x) for x in line.split(f"[FedAvg:global:{client}]")[1].split(",")])
                else:
                    avg_gains += pi[client]*np.array([float(x) for x in line.split(f"[FedAvg:global:{client}]")[1].split(",")])
            elif line.startswith("[FedSoftmax:global:") and not line[19] == 'g':
                # then convert the rest of the line into list of floats (numbers are separated by ",")
                i0 = 19
                client_string = ""
                while line[i0] != ']':
                    client_string += line[i0]
                    i0 += 1
                client = int(client_string)
                small_max_count += 1
                if client == 0:
                    max_count += 1
                if small_max_count == 1:
                    max_gains = pi[client]*np.array([float(x) for x in line.split(f"[FedSoftmax:global:{client}]")[1].split(",")])
                else:
                    max_gains += pi[client]*np.array([float(x) for x in line.split(f"[FedSoftmax:global:{client}]")[1].split(",")])
            elif line.startswith("[FedSoftmin:global:") and not line[19] == 'g':
                # then convert the rest of the line into list of floats (numbers are separated by ",")
                i0 = 19
                client_string = ""
                while line[i0] != ']':
                    client_string += line[i0]
                    i0 += 1
                client = int(client_string)
                small_min_count += 1
                if client == 0:
                    min_count += 1
                if small_min_count == 1:
                    min_gains = pi[client]*np.array([float(x) for x in line.split(f"[FedSoftmin:global:{client}]")[1].split(",")])
                else:
                    min_gains += pi[client]*np.array([float(x) for x in line.split(f"[FedSoftmin:global:{client}]")[1].split(",")])
            elif line.startswith("locally_local_"):
                # then convert the rest of the line into list of floats (numbers are separated by ",")
                i0 = 14
                client_string = ""
                while line[i0] != ' ':
                    client_string += line[i0]
                    i0 += 1
                client = int(client_string)
                small_local_count += 1
                if client == 0:
                    local_count += 1
                if small_local_count == 1:
                    locally = pi[client]*np.array([float(x) for x in line.split(f"locally_local_{client}")[1].split(",")])
                else:
                    locally += pi[client]*np.array([float(x) for x in line.split(f"locally_local_{client}")[1].split(",")])
    # plot the list of floats
    plt.plot(avg_gains/count, label="FedAvg")
    try:
    	plt.plot(min_gains/min_count, label="FedSoftMin")
    except:
    	print("no min")
    try:
    	plt.plot(max_gains/max_count, label="FedSoftMax")
    except:
    	print("no max")
    try:
    	plt.plot(locally/local_count, label="Local")
    except:
    	print("no local")
    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy averaged over {count} curves for {filename.split('_')[0].split('/')[1]}")
    plt.savefig("outputs/" + filename.split('_')[0].split('/')[1] + "_gain.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None, help="The file to plot")
    parser.add_argument("--accuracy", type=str, default=None, help="Accuracy instead of gain")
    args = parser.parse_args()
    if args.accuracy is None:
        if args.file is None:
            for filename in os.listdir("outputs/"):
                if filename.endswith(".inf"):
                    try:
                        plot("outputs/" + filename)
                    except:
                        print("Failed to plot " + filename)
        else:
            plot(args.file)
    else:
        if args.file is None:
            for filename in os.listdir("outputs/"):
                if filename.endswith(".inf"):
                    try:
                        plot_real_accuracy("outputs/" + filename)
                    except:
                        print("Failed to plot " + filename)
        else:
            plot_real_accuracy(args.file)
