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
    # each clients gets 2 classes
    # numbers_list = [[1/2, 1/2, 0., 0., 0., 0., 0., 0., 0., 0.] for _ in range(nb_clients//5)] + [[0., 0., 1/2, 1/2, 0., 0., 0., 0., 0., 0.] for _ in range(nb_clients//5)] + [[0., 0., 0., 0., 1/2, 1/2, 0., 0., 0., 0.] for _ in range(nb_clients//5)] + [[0., 0., 0., 0., 0., 0., 1/2, 1/2, 0., 0.] for _ in range(nb_clients//5)] + [[0., 0., 0., 0., 0., 0., 0., 0., 1/2, 1/2] for _ in range(nb_clients//5)]

    logging.basicConfig(level=logging.INFO,
    format='| %(levelname)s | %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="name of device")
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "FMNIST", "CIFAR10"], help="name of dataset")
    parser.add_argument("--clients", type=int, default=100, help="number of clients")
    parser.add_argument("--batch", type=int, default=64, help="batch size")
    parser.add_argument("--rounds", type=int, default=100, help="nb of rounds")
    parser.add_argument("--nbexp", type=int, default=1, help="nb of differente methods tested")
    parser.add_argument("--sizeclient", type=int, default=200, help="nb of samples per client")
    parser.add_argument("--epochs", type=int, default=1, help="nb of local epochs")
    parser.add_argument("--balanced", type=str, default='iid', help="iid for balanced clients")
    parser.add_argument("--shardsize", type=int, default=30, help="shardsize argument for unbalanced datasets")
    parser.add_argument("--lrmethod", type=str, default="decay", choices=["const", "decay"], help="learning rate method")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate argument")
    parser.add_argument("--experiment", type=str, help="names of experiment to run")
    parser.add_argument("--output", type=str, help="output file")
    parser.add_argument("--max", type=float, default=0.81, help="maximum accuracy")
    parser.add_argument("--topk", type=float, default=0.1, help="maximum accuracy")
    parser.add_argument("--softmin", type=float, default=0.1, help="alpha parameter for hybrid")
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
    #assert len(numbers_list) == args.clients
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

    size_client = args.sizeclient
    size_list = [size_client]*nb_clients
    numbers_list = [[1/3, 1/3, 1/3, 0., 0., 0., 0., 0., 0., 0.] for _ in range(nb_clients//5)] + [[0., 0., 1/3, 1/3, 1/3, 0., 0., 0., 0., 0.] for _ in range(nb_clients//5)] + [[0., 0., 0., 1/3, 1/3, 1/3, 0., 0., 0., 0.] for _ in range(nb_clients//5)] + [[0., 0., 0., 0., 0., 1/3, 1/3, 1/3, 0., 0.] for _ in range(nb_clients//5)] + [[0., 0., 0., 0., 0., 0., 0., 1/3, 1/3, 1/3] for _ in range(nb_clients//5)]

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
        mergers = [("FedPar", Merger_FedPar()),
                   ("FedParConv", Merger_FedParConv())]*1
    elif args.experiment == "gaia":
        mergers = [("FedAvg", Merger_FedAvg()), ("FedSoftmin", Merger_FedTopK(+0.2)), ("FedSoftmax", Merger_Hybrid([Merger_FedAvg(), Merger_FedTopK(+0.2)],
                                                [0 if (r < 15) else 1 for r in range(rounds)]))]*10
    elif args.experiment == "avg":
        mergers = [("FedAvg", Merger_FedAvg())]*1
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
        mergers = [("FedAvg", Merger_FedAvg()),
                   ("FedSoftmax", Merger_FedSoft(+3.0)),
                   ("FedSoftmin", Merger_FedSoft(-3.0)),
                   ("FedMink1", Merger_FedTopK(-0.05)),
                   ("FedMink2", Merger_FedTopK(-0.1)),
                   ("FedMink3", Merger_FedTopK(-0.2)),
                   ("FedMink4", Merger_FedTopK(-0.4))]*10
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
    info = {}
    
    if not os.path.exists("./outputs/"):
        os.mkdir("./outputs/")
    # create a folder for the experiment with the current date and time
    from datetime import datetime
    datetime_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists("./outputs/" + datetime_string):
        os.mkdir("./outputs/" + datetime_string)


    # put all the parameters in the info dictionary
    info["nb_clients"] = args.clients
    info["nb_exp"] = args.nbexp
    info["rounds"] = args.rounds
    info["epochs"] = args.epochs
    info["local_true"] = args.local_true
    info["balanced"] = args.balanced
    info["shardsize"] = args.shardsize
    info["lr"] = args.lr
    info["experiment"] = args.experiment
    info["sizeclient"] = args.sizeclient
    info["batch"] = args.batch
    info["dataset"] = args.dataset
    info["device"] = args.device

    # save the info dictionary in a npy file
    np.save("./outputs/" + datetime_string + "/info.npy", info)

    for merger_name, merger in mergers:
        accuracies_dict = {}
        loss_dict = {}
        alpha_dict = {}
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
            testloader = torch.utils.data.DataLoader(data_test, batch_size=1000, shuffle=False, num_workers=0)
            # define test and train dataloaders for each client
            train_dataloaders = [torch.utils.data.DataLoader(datasets[client_id]['train'], batch_size=batchsize, shuffle=True, num_workers=0) for client_id in range(nb_clients)]
            test_dataloaders = [torch.utils.data.DataLoader(datasets[client_id]['test'], batch_size=1, shuffle=False, num_workers=0) for client_id in range(nb_clients)]
        


        if counter_merge % nb_exp == 0 and local_true:
            print("Apprentissage localolocal...")
            loc_accuracies = {}
            loc_loss_dict = {}
            for client_id in tqdm(range(nb_clients)):
                accuracies_list = []
                loss_list = []
                model.load_state_dict(W)
                model.train()
                optim = torch.optim.SGD(model.parameters(), lr=args.lr)
                scheduler = StepLR(optim, step_size=epochs, gamma=0.99)
                for epoch in range(epochs*rounds):
                    for (x, y) in train_dataloaders[client_id]:
                        optim.zero_grad()
                        y_out = model(x)
                        loss = loss_fn(y_out, y)
                        loss.backward()
                        optim.step()
                        scheduler.step()
                    if epoch % epochs == epochs-1:
                        loss_list.append(loss.item())
                        accuracies_list.append(get_accuracy(model,test_dataloaders[client_id]))
                loc_accuracies[client_id] = accuracies_list
                loc_loss_dict[client_id] = loss_list
            print(f"Mean accuracy over clients: {np.mean([loc_accuracies[client_id][-1] for client_id in range(nb_clients)])}")
            # save in a npy file the accuracies and the losses
            np.save("./outputs/" + datetime_string + "/loc_accuracies_" + "local" + str(counter_merge // nb_exp) + ".npy", loc_accuracies)
            np.save("./outputs/" + datetime_string + "/loc_loss_dict_" + "local" + str(counter_merge // nb_exp) + ".npy", loc_loss_dict)

        counter_merge += 1
        
        for round in tqdm(range(rounds)):
            accuracies_list = []
            loss_list = []
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
                
                for epoch in range(epochs):
                    output.losses.append(0.0)
                    for (x, y) in train_dataloaders[client_id]:
                        optim.zero_grad()
                        y_out = model(x)
                        loss = loss_fn(y_out, y)
                        with torch.no_grad():
                            output.losses[-1] += loss.detach().item()
                        loss.backward()
                        optim.step()
                    output.losses[-1] /= output.size
                # update loss_list
                loss_list.append(output.losses[-1])
                accuracies_list.append(get_accuracy(model,test_dataloaders[client_id]))

                output.weight = copy.deepcopy(model.state_dict())
                outputs.append(output)

            # update accuracies_dict and loss_dict
            accuracies_dict[round] = accuracies_list
            loss_dict[round] = loss_list

            W, alpha = merger(outputs, accuracies_dict[round])
            model.load_state_dict(W)
            alpha_dict[round] = alpha
            
            step += 1
            elapsed_time = time.perf_counter() - t1
            remaining_time = (time.perf_counter() - t0) * (steps-step)/step
            

            if accuracies_dict[round][-1] > args.max:
                break

        # save in a npy file the accuracies and the losses
        np.save("./outputs/" + datetime_string + "/accuracies_" + merger_name + str(counter_merge // nb_exp) + ".npy", accuracies_dict)
        np.save("./outputs/" + datetime_string + "/loss_dict_" + merger_name + str(counter_merge // nb_exp) + ".npy", loss_dict)