import argparse
import copy
import logging
import numpy as np
import time
import os
import torch
# import StepLR
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm

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
from src.classic_mergers_partial import Merger_FedPar
from src.classic_mergers_partial_conv import Merger_FedParConv
from src.soft_mergers import Merger_FedSoft, Merger_FedSuperSoft
from src.topk_mergers import Merger_FedTopK
from src.hybrid_mergers import Merger_Hybrid
from src.control_mergers import Merger_FedWCostAvg, Merger_FedDiff, Merger_FedControl1, Merger_FedControl2

# other stuff
from src.utils import reset_parameters, fmttime


if __name__ == '__main__':
    size_list = [100]*500
    numbers_list = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] for _ in range(500)]

    logging.basicConfig(level=logging.INFO,
    format='| %(levelname)s | %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="name of device")
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "FMNIST", "CIFAR10"], help="name of dataset")
    parser.add_argument("--clients", type=int, default=100, help="number of clients")
    parser.add_argument("--batch", type=int, default=64, help="batch size")
    parser.add_argument("--rounds", type=int, default=100, help="nb of rounds")
    parser.add_argument("--nbexp", type=int, default=1, help="nb of differente methods tested")
    parser.add_argument("--epochs", type=int, default=1, help="nb of local epochs")
    parser.add_argument("--balanced", type=str, default='iid', help="iid for balanced clients")
    parser.add_argument("--shardsize", type=int, default=30, help="shardsize argument for unbalanced datasets")
    parser.add_argument("--lrmethod", type=str, default="decay", choices=["const", "decay"], help="learning rate method")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate argument")
    parser.add_argument("--experiment", type=str, help="names of experiment to run")
    parser.add_argument("--output", type=str, help="output file")
    parser.add_argument("--max", type=float, default=0.81, help="maximum accuracy")
    parser.add_argument("--local_true", type=bool, default=False, help="whether or not to do local training")
    args = parser.parse_args()

    local_true = args.local_true

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
    nb_exp = args.nbexp

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
    if args.experiment == "mainexp":
        mergers = [("FedAvg", Merger_FedAvg()),
                   ("FedPar", Merger_FedPar()),
                   ("FedParConv", Merger_FedParConv()),
                   ("FedSoftmax", Merger_FedSoft(+5.0))]*1
    elif args.experiment == "partial":
        mergers = [("FedPar", Merger_FedPar())]*1
    elif args.experiment == "exp1":
        mergers = [("FedAvg", Merger_FedAvg()),
                   ("FedSoftmax", Merger_FedSoft(+20.0)),
                   ("FedSoftmin", Merger_FedSoft(-20.0))]*10
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

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    testloader = torch.utils.data.DataLoader(data_test, batch_size=1000, shuffle=False, num_workers=0)
    steps = len(mergers) * rounds
    step = 0
    t0 = time.perf_counter()
    counter_merge = 0
    accuracies = {}
    for merger_name, merger in mergers:
        print(f"[{merger_name}] Begin...")
        merger.reset()
        if counter_merge % nb_exp == 0:
            reset_parameters(model)
            W0 = copy.deepcopy(model.state_dict())

        print("Apprentissage fédéré sur chaque client...")
        model.load_state_dict(W0)
        W = copy.deepcopy(model.state_dict())

        if counter_merge % nb_exp == 0:
            if not(balanced): datasets = split_dataset_noniid(data_train, nb_clients, size_list, numbers_list, ratio_test=0.15)
            else: datasets = split_dataset_iid(data_train, nb_clients, ratio_test=0.15)
        
        # union_of_testsets = torch.utils.data.ConcatDataset([datasets[client_id]['test'] for client_id in range(nb_clients)])
        # testloader = torch.utils.data.DataLoader(union_of_testsets, batch_size=1, shuffle=False, num_workers=0)
        testloader = torch.utils.data.DataLoader(data_test, batch_size=1000, shuffle=False, num_workers=0)

        accuracies['global-global'] = []
        for client_id in range(nb_clients):
            accuracies[f'local_{client_id}-local_{client_id}'] = []
            accuracies[f'local_{client_id}-global'] = []
            accuracies[f'global-local_{client_id}'] = []
            if counter_merge % nb_exp == 0:
                accuracies[f'locally_local_{client_id}'] = []

        if counter_merge % nb_exp == 0 and local_true:
            print("Apprentissage classique...")
            for client_id in tqdm(range(nb_clients)):
                model.load_state_dict(W)
                model.train()
                optim = torch.optim.SGD(model.parameters(), lr=args.lr)
                scheduler = StepLR(optim, step_size=epochs, gamma=0.99)
                mini_dataloader = torch.utils.data.DataLoader(datasets[client_id]['train'], batch_size=batchsize, shuffle=True, num_workers=0)
                mini_dataloader_test = torch.utils.data.DataLoader(datasets[client_id]['test'], batch_size=1, shuffle=False, num_workers=0)
                for epoch in range(epochs*rounds):
                    for (x, y) in mini_dataloader:
                        optim.zero_grad()
                        y_out = model(x)
                        loss = loss_fn(y_out, y)
                        loss.backward()
                        optim.step()
                        scheduler.step()
                    if epoch % epochs == epochs-1:
                        accuracies[f'locally_local_{client_id}'].append(get_accuracy(model, mini_dataloader_test))
            # print mean accuracy over clients
            print(f"Mean accuracy over clients: {np.mean([accuracies[f'locally_local_{client_id}'][-1] for client_id in range(nb_clients)])}")
        counter_merge += 1
        if merger_name != "FedPar" and merger_name != "FedParConv":
            for round in tqdm(range(rounds)):
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

                    accuracies[f'local_{client_id}-local_{client_id}'].append(get_accuracy(model, testclient))
                    accuracies[f'local_{client_id}-global'].append(get_accuracy(model, testloader))
                W = merger(outputs)
                model.load_state_dict(W)

                step += 1
                elapsed_time = time.perf_counter() - t1
                remaining_time = (time.perf_counter() - t0) * (steps-step)/step
                # print(f"[{merger_name}:{round+1}/{rounds}]: {elapsed_time:.2f}sec/round, remaining time: {fmttime(int(remaining_time))}")
                accuracies['global-global'].append(get_accuracy(model, testloader))
                # print(f"Global accuracy: {accuracies['global-global'][-1]}")
                for client_id in range(nb_clients):
                    testclient = torch.utils.data.DataLoader(datasets[client_id]['test'], batch_size=1000, shuffle=False, num_workers=0)
                    accuracies[f'global-local_{client_id}'].append(get_accuracy(model, testclient))


                if accuracies['global-global'][-1] >= args.max: break

            if not os.path.exists("./outputs/"):
                os.mkdir("./outputs/")
            sum_train_sets = sum([len(datasets[client_id]['train']) for client_id in range(nb_clients)])
            print(f"Final accuracy of global model for local: {np.sum([len(datasets[client_id]['train'])/sum_train_sets*accuracies[f'global-local_{client_id}'][-1] for client_id in range(nb_clients)])}")
            print(f"Final accuracy of global model for global: {accuracies['global-global'][-1]}")
            if local_true:
                accs_gain = [np.sum([(accuracies[f'global-local_{client_id}'][i] - accuracies[f'locally_local_{client_id}'][i])*len(datasets[client_id]['train'])/sum_train_sets for client_id in range(nb_clients)]) for i in range(len(accuracies[f'global-local_{0}']))]
                ponderated_local = [np.sum([(accuracies[f'locally_local_{client_id}'][i])*len(datasets[client_id]['train'])/sum_train_sets for client_id in range(nb_clients)]) for i in range(len(accuracies[f'global-local_{0}']))]
                print(f"Locally local: {ponderated_local[-1]}")
                print(f"Gain: {accs_gain[-1]}")


            with open(f"./outputs/{output_file}_accs.inf", 'a') as file:
                file.write(f"[{merger_name}:global:global] {', '.join(map(str, accuracies['global-global']))}\n")
                for client_id in range(nb_clients):
                    if local_true:
                        file.write(f"locally_local_{client_id} {', '.join(map(str, accuracies[f'locally_local_{client_id}']))}\n")
                    # file.write(f"[{merger_name}:{client_id}:{client_id}] {', '.join(map(str, accuracies[f'local_{client_id}-local_{client_id}']))}\n")
                    # file.write(f"[{merger_name}:{client_id}:global] {', '.join(map(str, accuracies[f'local_{client_id}-global']))}\n")
                    file.write(f"[{merger_name}:global:{client_id}] {', '.join(map(str, accuracies[f'global-local_{client_id}']))}\n")
                file.write(f"[{merger_name}:pi] {', '.join(map(str, [len(datasets[client_id]['train']) for client_id in range(nb_clients)]))}\n")
                if local_true:
                    file.write(f"[{merger_name}:gain] {', '.join(map(str, accs_gain))}\n")
        elif merger_name == "FedPar" or merger_name == "FedParConv":
            W_list = [copy.deepcopy(model.state_dict()) for _ in range(nb_clients)]
            for round in tqdm(range(rounds)):
                t1 = time.perf_counter()
                outputs = []
                for client_id in range(nb_clients):
                    output = ClientOutput(client_id)
                    output.size = len(datasets[client_id]['train'])
                    output.losses = []
                    output.round = round

                    model.load_state_dict(W_list[client_id])
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

                    accuracies[f'local_{client_id}-local_{client_id}'].append(get_accuracy(model, testclient))
                    accuracies[f'local_{client_id}-global'].append(get_accuracy(model, testloader))
                W_list = [merger(outputs, i) for i in range(nb_clients)]
                model.load_state_dict(W)

                step += 1
                elapsed_time = time.perf_counter() - t1
                remaining_time = (time.perf_counter() - t0) * (steps-step)/step
                for client_id in range(nb_clients):
                    model.load_state_dict(W_list[client_id])
                    testclient = torch.utils.data.DataLoader(datasets[client_id]['test'], batch_size=1000, shuffle=False, num_workers=0)
                    accuracies[f'local_{client_id}-local_{client_id}'].append(get_accuracy(model, testclient))


            if not os.path.exists("./outputs/"):
                os.mkdir("./outputs/")
            sum_train_sets = sum([len(datasets[client_id]['train']) for client_id in range(nb_clients)])
            print(f"Final accuracy for partial update: {np.sum([len(datasets[client_id]['train'])/sum_train_sets*accuracies[f'local_{client_id}-local_{client_id}'][-1] for client_id in range(nb_clients)])}")


            with open(f"./outputs/{output_file}_accs.inf", 'a') as file:
                for client_id in range(nb_clients):
                    file.write(f"[{merger_name}:{client_id}:{client_id}] {', '.join(map(str, accuracies[f'local_{client_id}-local_{client_id}']))}\n")
                file.write(f"[{merger_name}:pi] {', '.join(map(str, [len(datasets[client_id]['train']) for client_id in range(nb_clients)]))}\n")
