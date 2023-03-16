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
    plt.plot(min_gains/min_count, label="FedSoftMin")
    plt.plot(max_gains/max_count, label="FedSoftMax")
    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Gain")
    plt.title(f"Gain averaged over {count} curves for {filename.split('_')[0].split('/')[1]}")
    plt.savefig("outputs/" + filename.split('_')[0].split('/')[1] + "_gain.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None, help="The file to plot")
    args = parser.parse_args()
    if args.file is None:
        for filename in os.listdir("outputs/"):
            if filename.endswith(".inf"):
                try:
                    plot("outputs/" + filename)
                except:
                    print("Failed to plot " + filename)
    else:
        plot(args.file)