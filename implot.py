import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merger", type=str, default="FedAvg", help="name of merger")
    parser.add_argument("--exp", type=str, default="e3", help="name of experiment")
    parser.add_argument("--max", type=int, default=70, help="nax")
    args = parser.parse_args()

    merger_name = args.merger
    exp_name = args.exp

    for file in glob.glob(f"./outputs/{exp_name}/{merger_name}/*"):
        npz = np.load(file, allow_pickle=True)
        if npz["global"][-1] > args.max:
            print("file:", file)
            print(npz["global"])
            pi = (npz["pi"]/sum(npz["pi"]))
            alphas = npz["alphas"]
            malpha = np.mean(alphas, axis=0)
            print(malpha.shape)

            res = np.concatenate([alphas, np.zeros((1, pi.shape[0])), malpha.reshape(1, -1), np.zeros((1, pi.shape[0])), pi.reshape(1, -1)], axis=0)

            plt.imshow(alphas)
            plt.show()