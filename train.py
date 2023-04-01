import argparse
import copy
import random
import logging
import numpy as np
import time
import os
import torch

# metrics
from src.accuracy import get_accuracy

# models
from src.tf_cifar import TFCifar
from src.convnet import ConvNet

# datasets
from src.datasets import MNIST, FMNIST, CIFAR10

# dataset splitting algorithms
from src.datasplitting import split_dataset_iid, split_dataset_noniid

# strategies
from src.client_output import ClientOutput
from src.classic_mergers import Merger_FedAvg
from src.soft_mergers import Merger_FedSoft, Merger_FedSuperSoft
from src.topk_mergers import Merger_FedTopK
from src.hybrid_mergers import Merger_Hybrid
from src.control_mergers import Merger_FedWCostAvg, Merger_FedDiff, Merger_FedControl1, Merger_FedControl2

# other stuff
from src.utils import reset_parameters, fmttime


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
    format='| %(levelname)s | %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="name of device")
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "FMNIST", "CIFAR10"], help="name of dataset")
    parser.add_argument("--clients", type=int, default=100, help="number of clients")
    parser.add_argument("--batch", type=int, default=64, help="batch size")
    parser.add_argument("--rounds", type=int, default=100, help="nb of rounds")
    parser.add_argument("--epochs", type=int, default=1, help="nb of local epochs")
    parser.add_argument("--balanced", type=str, default='iid', help="iid for balanced clients")
    parser.add_argument("--shardsize", type=int, default=30, help="shardsize argument for unbalanced datasets")
    parser.add_argument("--lrmethod", type=str, default="decay", choices=["const", "decay"], help="learning rate method")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate argument")
    parser.add_argument("--experiment", type=str, help="names of experiment to run")
    parser.add_argument("--output", type=str, help="output file")
    parser.add_argument("--max", type=float, default=0.81, help="maximum accuracy")
    args = parser.parse_args()

    logging.info(f"Device = {args.device}")
    device = torch.device(args.device)

    logging.info(f"Dataset = {args.dataset}")
    t0 = time.perf_counter()
    if args.dataset == "MNIST":
        data_train = MNIST(True, device)
        data_test = MNIST(False, device)
    elif args.dataset == "FMNIST":
        data_train = FMNIST(True, device)
        data_test = FMNIST(False, device)
    elif args.dataset == "CIFAR10":
        data_train = CIFAR10(True, device)
        data_test = CIFAR10(False, device)
    else:
        print(f"Unknown dataset.")
        exit(1)

    model_name = "ConvNet" if args.dataset in ["MNIST", "FMNIST"] else "TFCifar"
    logging.info(f"Model = {model_name}")
    t0 = time.perf_counter()
    if model_name == "ConvNet":
        model = ConvNet()
    elif model_name == "TFCifar":
        model = TFCifar()
    else:
        print(f"Unknown model.")
        exit(1)
    model.to(device=device)

    logging.info(f"Number of clients = {args.clients}")
    nb_clients = args.clients

    logging.info(f"Number of rounds = {args.rounds}")
    rounds = args.rounds
    logging.info(f"Number of local epochs = {args.epochs}")
    epochs = args.epochs
    logging.info(f"Batch size = {args.batch}")
    batchsize = args.batch

    if args.lrmethod == "const":
        learningrates = [args.lr for r in range(rounds)]
        logging.info(f"Learning rate: {args.lr}")
    elif args.lrmethod == "decay":
        learningrates = [args.lr * (0.99**r) for r in range(rounds)]
        logging.info(f"Learning rate: {args.lr} * (0.99**r)")

    balanced = (args.balanced == 'iid')
    if balanced:
        logging.info(f"Datasplitting: IID")
    if not(balanced):
        logging.info(f"Datasplitting: non-IID (shardsize={args.shardsize})")
        shardsize = args.shardsize

    logging.info(f"Experiment: {args.experiment}")
    if args.experiment == "extra2":
        mergers = [("FedAvg", Merger_FedAvg()),
                   ("FedSoftMax", Merger_FedSoft(+5.0)),
                   ("FedTopK", Merger_FedTopK(0.1))]*30

    elif args.experiment == "extra1":
        mergers = [("FedAvg", Merger_FedAvg()),
                   ("FedSoftMax", Merger_FedSoft(+5.0)),
                   ("FedSoftMin", Merger_FedSoft(-5.0))]*500
    elif args.experiment == "exp1":
        mergers = [("FedAvg", Merger_FedAvg()),
                   ("FedSoftmax", Merger_FedSoft(+5.0)),
                   ("FedSoftmin", Merger_FedSoft(-5.0))]*100
    elif args.experiment == "exp2":
        mergers = [("FedAvg", Merger_FedAvg()),
                   ("FedAvgSmax", Merger_Hybrid([Merger_FedAvg(), Merger_FedSoft(+5.0)],
                                                [0 if (r < 20) else 1 for r in range(rounds)])),
                   ("FedAvgSmin", Merger_Hybrid([Merger_FedAvg(), Merger_FedSoft(-5.0)],
                                                [0 if (r < 20) else 1 for r in range(rounds)])),
                   ("FedSmaxAvg", Merger_Hybrid([Merger_FedAvg(), Merger_FedSoft(+5.0)],
                                                [1 if (r < 20) else 0 for r in range(rounds)])),
                   ("FedSminAvg", Merger_Hybrid([Merger_FedAvg(), Merger_FedSoft(-5.0)],
                                                [1 if (r < 20) else 0 for r in range(rounds)]))]*100
    elif args.experiment == "exp3":
        mergers = [("FedMaxk", Merger_FedTopK(+0.1)),
                   ("FedMink", Merger_FedTopK(-0.1)),
                   ("FedMaxK", Merger_FedTopK(+0.2)),
                   ("FedMinK", Merger_FedTopK(-0.2))]*100
    elif args.experiment == "exp4":
        mergers = [("FedAvg", Merger_FedAvg()),
                   ("FedWCostAvg", Merger_FedWCostAvg(0.5))]*100
    elif args.experiment == "exp5":
        mergers = [("FedAvg", Merger_FedAvg()),
                   ("FedSoftMax", Merger_FedSoft(+15.0)),
                   ("FedSoftMin", Merger_FedSoft(-15.0))]*100
    elif args.experiment == "exp6":
        mergers = [("FedAvg", Merger_FedAvg()),
                   ("FedWCostAvg", Merger_FedWCostAvg(0.5)),
                   ("FedDiff", Merger_FedDiff(0.5))]*200
    elif args.experiment == "exp7":
        mergers = [("FedAvg", Merger_FedAvg()),
                   ("FedSmaxAvgSmin", Merger_Hybrid([Merger_FedSoft(+5.0), Merger_FedAvg(), Merger_FedSoft(-5.0)],
                                                [0 if (r < 15) else (1 if (r < 30) else 2) for r in range(rounds)])),
                   ("FedSminAvgSmax", Merger_Hybrid([Merger_FedSoft(-5.0), Merger_FedAvg(), Merger_FedSoft(+5.0)],
                                                [0 if (r < 15) else (1 if (r < 30) else 2) for r in range(rounds)]))]*100
    elif args.experiment == "exp8":
        mergers = [("FedWCostAvg", Merger_FedWCostAvg(0.5)),
                   ("FedDiff", Merger_FedDiff(0.5))]*200
    elif args.experiment == "exp10":
        mergers = [("FedControl1", Merger_FedControl1(1/3.0, 1/3.0)),
                   ("FedWCostAvg", Merger_FedWCostAvg(0.5)),
                   ("FedControl2", Merger_FedControl2(1/3.0, 1/3.0))]*200
    else:
        logging.error(f"Experiment '{args.experiment}' is not defined!")
        exit(1)

    logging.info(f"Output file: {args.output}")
    output_file = args.output
    if output_file is None: logging.warning("Output file not defined!")

    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    testloader = torch.utils.data.DataLoader(data_test, batch_size=1000, shuffle=False, num_workers=0)

    steps = len(mergers) * rounds
    step = 0
    t0 = time.perf_counter()
    for merger_name, merger in mergers:
        print(f"[{merger_name}] Begin...")
        merger.reset()

        reset_parameters(model)
        W0 = copy.deepcopy(model.state_dict())

        print("Apprentissage fédéré sur chaque client...")
        model.load_state_dict(W0)
        W = copy.deepcopy(model.state_dict())


        if not(balanced): datasets = split_dataset_noniid(data_train, nb_clients, shard_size=shardsize, ratio_test=0.15)
        else: datasets = split_dataset_iid(data_train, nb_clients, ratio_test=0.15)

        accuracies = {'global-global': [get_accuracy(model, testloader)]}
        alphas = []
        for client_id in range(nb_clients):
            accuracies[f'local_{client_id}-local_{client_id}'] = []
            accuracies[f'local_{client_id}-global'] = []
            accuracies[f'global-local_{client_id}'] = []

        for round in range(rounds):
            t1 = time.perf_counter()
            outputs = []
            for client_id in range(nb_clients):
                output = ClientOutput(client_id)
                output.size = len(datasets[client_id]['train'])
                output.losses = []
                output.round = round

                model.load_state_dict(W)
                model.train()
                optim = torch.optim.SGD(model.parameters(), lr=learningrates[round])

                dataloader = torch.utils.data.DataLoader(datasets[client_id]['train'], batch_size=batchsize, shuffle=True, num_workers=0)
                for epoch in range(epochs):
                    output.losses.append(0.0)
                    for (x, y) in dataloader:
                        optim.zero_grad()
                        y_out = model(x)
                        loss = loss_fn(y_out, y)
                        with torch.no_grad():
                            output.losses[-1] += loss.detach().item()
                        loss.backward()
                        optim.step()
                    output.losses[-1] /= output.size

                output.weight = copy.deepcopy(model.state_dict())
                outputs.append(output)

                testclient = torch.utils.data.DataLoader(datasets[client_id]['test'], batch_size=1000, shuffle=False, num_workers=0)

                # accuracies[f'local_{client_id}-local_{client_id}'].append(get_accuracy(model, testclient))
                # accuracies[f'local_{client_id}-global'].append(get_accuracy(model, testloader))
            W, alpha = merger(outputs)
            alphas.append(alpha)
            model.load_state_dict(W)

            step += 1
            elapsed_time = time.perf_counter() - t1
            remaining_time = (time.perf_counter() - t0) * (steps-step)/step
            print(f"[{merger_name}:{round+1}/{rounds}]: {elapsed_time:.2f}sec/round, remaining time: {fmttime(int(remaining_time))}")
            accuracies['global-global'].append(get_accuracy(model, testloader))
            for client_id in range(nb_clients):
                testclient = torch.utils.data.DataLoader(datasets[client_id]['test'], batch_size=1000, shuffle=False, num_workers=0)
                # accuracies[f'global-local_{client_id}'].append(get_accuracy(model, testclient))

            if accuracies['global-global'][-1] >= args.max:
                print("Interruption")
                break

        if not os.path.exists("./outputs/"):
            os.mkdir("./outputs/")

        if not os.path.exists(f"./outputs/{output_file}"):
            os.mkdir(f"./outputs/{output_file}")
        sum_train_sets = sum([len(datasets[client_id]['train']) for client_id in range(nb_clients)])
        # print(f"Final accuracy of global model for local: {np.sum([len(datasets[client_id]['train'])/sum_train_sets*accuracies[f'global-local_{client_id}'][-1] for client_id in range(nb_clients)])}")
        # print(f"Final accuracy of global model for global: {accuracies['global-global'][-1]}")

        if not os.path.exists(f"./outputs/{output_file}/{merger_name}/"):
            os.mkdir(f"./outputs/{output_file}/{merger_name}/")

        run_id = random.randrange(0, 1000)
        data = {'alphas': np.array(alphas),
                'global': np.array(accuracies['global-global']),
                'pi': np.array([len(datasets[client_id]['train']) for client_id in range(nb_clients)])}

        if False:
            for client_id in range(nb_clients):
                data[f'local_{client_id}-local_{client_id}'] = np.array(accuracies[f'local_{client_id}-local_{client_id}'])
                data[f'local_{client_id}-global'] = np.array(accuracies[f'local_{client_id}-global'])
                data[f'global-local_{client_id}'] = np.array(accuracies[f'global-local_{client_id}'])

        np.savez(f"./outputs/{output_file}/{merger_name}/run_{run_id}.npz", **data)

        if False:
            with open(f"./outputs/{output_file}_accs.inf", 'a') as file:
                file.write(f"[{merger_name}:global:global] {', '.join(map(str, accuracies['global-global']))}\n")
                for client_id in range(nb_clients):
                    file.write(f"[{merger_name}:{client_id}:{client_id}] {', '.join(map(str, accuracies[f'local_{client_id}-local_{client_id}']))}\n")
                    file.write(f"[{merger_name}:{client_id}:global] {', '.join(map(str, accuracies[f'local_{client_id}-global']))}\n")
                    file.write(f"[{merger_name}:global:{client_id}] {', '.join(map(str, accuracies[f'global-local_{client_id}']))}\n")
                file.write(f"[{merger_name}:pi] {', '.join(map(str, [len(datasets[client_id]['train']) for client_id in range(nb_clients)]))}\n")


