import argparse
import copy
import logging
import numpy as np
import time
import os
import torch
# use tensorboard
from torch.utils.tensorboard import SummaryWriter
# import StepLR
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm

import matplotlib.pyplot as plt

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
from src.classic_mergers import Merger_FedAvg, Merger_Layer
from src.classic_mergers_partial import Merger_FedPar
from src.classic_mergers_partial_conv import Merger_FedParConv
from src.soft_mergers import Merger_FedSoft, Merger_FedSuperSoft, Merger_FednegHess, Merger_FedHess, Merger_FedSoftTop
from src.topk_mergers import Merger_FedTopK, Merger_FedTopKnegHess, Merger_FedTopKHess
from src.hybrid_mergers import Merger_Hybrid
from src.control_mergers import Merger_FedWCostAvg, Merger_FedDiff, Merger_FedControl1, Merger_FedControl2

# other stuff
from src.utils import reset_parameters, fmttime


if __name__ == '__main__':
    # each clients gets 2 classes
    # numbers_list = [[1/2, 1/2, 0., 0., 0., 0., 0., 0., 0., 0.] for _ in range(nb_clients//5)] + [[0., 0., 1/2, 1/2, 0., 0., 0., 0., 0., 0.] for _ in range(nb_clients//5)] + [[0., 0., 0., 0., 1/2, 1/2, 0., 0., 0., 0.] for _ in range(nb_clients//5)] + [[0., 0., 0., 0., 0., 0., 1/2, 1/2, 0., 0.] for _ in range(nb_clients//5)] + [[0., 0., 0., 0., 0., 0., 0., 0., 1/2, 1/2] for _ in range(nb_clients//5)]

    logging.basicConfig(level=logging.INFO,
    format='| %(levelname)s | %(message)s')

    # initialize tensorboard
    # writer = SummaryWriter()

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="name of device")
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "FMNIST", "CIFAR10"], help="name of dataset")
    parser.add_argument("--clients", type=int, default=50, help="number of clients")
    parser.add_argument("--batch", type=int, default=64, help="batch size")
    parser.add_argument("--rounds", type=int, default=50, help="nb of rounds")
    parser.add_argument("--nbexp", type=int, default=1, help="nb of differente methods tested")
    parser.add_argument("--sizeclient", type=int, default=200, help="nb of samples per client")
    parser.add_argument("--epochs", type=int, default=1, help="nb of local epochs")
    parser.add_argument("--balanced", type=str, default='iid', help="iid for balanced clients")
    parser.add_argument("--shardsize", type=int, default=30, help="shardsize argument for unbalanced datasets")
    parser.add_argument("--lrmethod", type=str, default="decay", choices=["const", "decay"], help="learning rate method")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate argument")
    parser.add_argument("--experiment", type=str, help="names of experiment to run")
    parser.add_argument("--output", type=str, help="output file")
    parser.add_argument("--max", type=float, default=1.1, help="maximum accuracy")
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
    # numbers_list = [[1/3, 1/3, 1/3, 0., 0., 0., 0., 0., 0., 0.] for _ in range(nb_clients//5)] + [[0., 0., 1/3, 1/3, 1/3, 0., 0., 0., 0., 0.] for _ in range(nb_clients//5)] + [[0., 0., 0., 1/3, 1/3, 1/3, 0., 0., 0., 0.] for _ in range(nb_clients//5)] + [[0., 0., 0., 0., 0., 1/3, 1/3, 1/3, 0., 0.] for _ in range(nb_clients//5)] + [[0., 0., 0., 0., 0., 0., 0., 1/3, 1/3, 1/3] for _ in range(nb_clients//5)]

    numbers_list = [[1/2, 1/2, 0., 0., 0., 0., 0., 0., 0., 0.] for _ in range(nb_clients//5)] + [[0., 0., 1/2, 1/2, 0., 0., 0., 0., 0., 0.] for _ in range(nb_clients//5)] + [[0., 0., 0., 0., 1/2, 1/2, 0., 0., 0., 0.] for _ in range(nb_clients//5)] + [[0., 0., 0., 0., 0., 0., 1/2, 1/2, 0., 0.] for _ in range(nb_clients//5)] + [[0., 0., 0., 0., 0., 0., 0., 0., 1/2, 1/2] for _ in range(nb_clients//5)]

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
        if args.dataset == "MNIST":
            learningrates = [1e-4 * (1.**r) for r in range(rounds)]
            epochs = 1
            print("WARNING: epochs set to 1 for MNIST")
        elif args.dataset == "FMNIST":
            learningrates = [1e-3 * (0.99**r) for r in range(rounds)]
        elif args.dataset == "CIFAR10":
            learningrates = [1e-2 * (0.9**r) for r in range(rounds)]
            epochs = 2
            print("WARNING: epochs set to 2 for CIFAR10")
        logging.info(f"Learning rate: {learningrates[0]} (decay)")

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
    elif args.experiment == "avgonly":
        mergers = [("FedAvg", Merger_FedAvg()),
                   ("FedLayer", Merger_Layer()),
                   ]*1
        nb_exp = 2
    elif args.experiment == "soft":
        mergers = [("FedAvg", Merger_FedAvg()),
                   ("FedSoftMax5", Merger_FedSoft(+5.0)),
                   ("FedSoftMax20", Merger_FedSoft(+20.0)),
                   ("FedSoftMax50", Merger_FedSoft(+50.0)),
                   ("FedSoftMax100", Merger_FedSoft(+100.0)),
                   ("FedSoftMax200", Merger_FedSoft(+200.0)),
                   ("FedTop1c", Merger_FedTopK(1/nb_clients)),
                        ("FedTop3c", Merger_FedTopK(3/nb_clients)),
                                ("FedTop05", Merger_FedTopK(0.5)),
                                    ("FedTop08", Merger_FedTopK(0.8)),

                   ]*100
        nb_exp = 10
    elif args.experiment == "top":
        mergers = [("FedAvg", Merger_FedAvg()),
                   ("FedSoftMax5", Merger_FedSoft(+5.0)),
                   ("FedTop1c", Merger_FedTopK(1/nb_clients)),
                        ("FedTop3c", Merger_FedTopK(3/nb_clients)),
                            ("FedTop03", Merger_FedTopK(0.3)),
                                ("FedTop05", Merger_FedTopK(0.5)),
                                    ("FedTop08", Merger_FedTopK(0.8)),

                   ]*100
        nb_exp = 7
    elif args.experiment == "expnewpanel":
        mergers = [("FedAvg", Merger_FedAvg()),
                   ("FedSoftMax50", Merger_FedSoft(+50.0)),
                   ("FedMaxk", Merger_FedTopK(1/nb_clients)),
                   ]*100
        nb_exp = 3
    elif args.experiment == "expnewpanel2":
        mergers = [("FedAvg", Merger_FedAvg()),
                   ("FedMaxk", Merger_FedTopK(1/nb_clients)),
                   ("FedTopHess", Merger_FedTopKHess(1/nb_clients)),
                   ("FedSoftTop", Merger_FedSoftTop(+7.0, proportional=False)),
                   ("FedProp", Merger_FedSoft(+20., proportional=True)),
                   ("FedSoftMax7", Merger_FedSoft(+7.0)),
                   
                   ]*100
        nb_exp = 6
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

    test_accs = {}
    test_accs_history = {}
    test_accs_conf_int = {}
    count = {}
    alpha_dict_gen = {}

    for merger_name, merger in mergers:
        # create a new summary writer for each merger
        writer = SummaryWriter("./runs/" + datetime_string + "/" + merger_name)
        accuracies_dict = {}
        train_accuracies_dict = {}
        loss_dict = {}
        alpha_dict = {}
        alpha_gen_list = []
        mean_accuracies_dict = []
        print(f"[{merger_name}] {counter_merge} / {nb_exp}")
        merger.reset()
        if counter_merge % nb_exp == 0:
            reset_parameters(model)
            W0 = copy.deepcopy(model.state_dict())
        
        model.load_state_dict(W0)
        W = copy.deepcopy(model.state_dict())

        # print("Apprentissage fédéré sur chaque client...")

        if counter_merge % nb_exp == 0:
            if not(balanced): datasets = split_dataset_noniid(data_train, nb_clients, size_list, numbers_list, ratio_test=0.15)
            else: datasets = split_dataset_iid(data_train, nb_clients, ratio_test=0.15)
            testloader = torch.utils.data.DataLoader(data_test, batch_size=1000, shuffle=False, num_workers=0)
            # define test and train dataloaders for each client
            train_dataloaders = [torch.utils.data.DataLoader(datasets[client_id]['train'], batch_size=batchsize, shuffle=True, num_workers=0) for client_id in range(nb_clients)]
            test_dataloaders = [torch.utils.data.DataLoader(datasets[client_id]['test'], batch_size=1, shuffle=False, num_workers=0) for client_id in range(nb_clients)]
            # define the train and test dataloaders of the reunion of the datasets of all clients
            union_train_dataset = torch.utils.data.ConcatDataset(datasets[client_id]['train'] for client_id in range(nb_clients))
            union_train_dataloader = torch.utils.data.DataLoader(union_train_dataset, batch_size=batchsize, shuffle=True, num_workers=0)
            union_test_dataset = torch.utils.data.ConcatDataset(datasets[client_id]['test'] for client_id in range(nb_clients))
            union_test_dataloader = torch.utils.data.DataLoader(union_test_dataset, batch_size=1, shuffle=False, num_workers=0)

        

        counter_merge += 1
        accuracies_dict = {}
        for round in tqdm(range(rounds)):
            accuracies_list = []
            train_accuracies_list = []
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
                # optim = torch.optim.SGD(model.parameters(), lr=learningrates[round],
                #       momentum=0.9, weight_decay=5e-4)
                # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=200)    
                
                for epoch in range(epochs):
                    output.losses.append(0.0)
                    for (x, y) in train_dataloaders[client_id]:
                        model.train()
                        optim.zero_grad()
                        y_out = model(x)
                        loss = loss_fn(y_out, y)
                        with torch.no_grad():
                            output.losses[-1] += loss.detach().item()
                        loss.backward()
                        optim.step()
                        # scheduler.step()
                    output.losses[-1] /= output.size
                # update loss_list
                loss_list.append(output.losses[-1])
                accuracies_list.append(get_accuracy(model,test_dataloaders[client_id]))
                train_accuracies_list.append(get_accuracy(model,train_dataloaders[client_id]))

                output.weight = copy.deepcopy(model.state_dict())
                outputs.append(output)

            # update accuracies_dict and loss_dict
            accuracies_dict[round] = accuracies_list
            train_accuracies_dict[round] = train_accuracies_list
            loss_dict[round] = loss_list
            mean_accuracies_dict.append(np.mean(accuracies_list))
            writer.add_scalar(f'Accuracy/test_{merger_name}', np.mean(accuracies_list), round)

            

            W, alpha = merger(outputs, accuracies_dict[round])
            model.load_state_dict(W)
            alpha_dict[round] = alpha

            alpha_gen_list.append(np.linalg.norm(alpha-np.array([1/nb_clients for _ in range(nb_clients)])))
            
            step += 1
            elapsed_time = time.perf_counter() - t1
            remaining_time = (time.perf_counter() - t0) * (steps-step)/step

            writer.close()
            

            if accuracies_dict[round][-1] > args.max:
                break
        # plot alpha_dict with a curve for each client
        for client_id in range(nb_clients):
            plt.plot([alpha_dict[round][client_id] for round in range(len(alpha_dict))])
        plt.legend()
        plt.savefig(f"./outputs/{datetime_string}/alpha_{merger_name}.png")
        plt.close()
        # plot the mean
        plt.plot([np.mean([alpha_dict[round][client_id] for client_id in range(nb_clients)]) for round in range(len(alpha_dict))])
        plt.savefig(f"./outputs/{datetime_string}/alpha_mean_{merger_name}.png")
        plt.close()
        merger_name_copy = merger_name
        # if merger_name in test_accs
        mean_accuracies_dict = np.array(mean_accuracies_dict)
        if merger_name in test_accs.keys():
            test_accs[merger_name] += mean_accuracies_dict
            alpha_dict_gen[merger_name] = alpha_gen_list
            count[merger_name] += 1
            test_accs_history[merger_name].append(list(mean_accuracies_dict))
            # compute the 95% confidence interval

            mean = np.mean(np.array(test_accs_history[merger_name]), axis=0)
            std = np.std(np.array(test_accs_history[merger_name]), axis=0)
            conf_int = 1.96 * std / np.sqrt(count[merger_name])
            # add to dictionary
            test_accs_conf_int[merger_name] = conf_int
        else:
            test_accs[merger_name] = mean_accuracies_dict
            alpha_dict_gen[merger_name] = alpha_gen_list
            test_accs_history[merger_name] = [list(mean_accuracies_dict)]
            count[merger_name] = 1
        # plot all the curves in test_accs with matplotlib
        for merger_name in test_accs:
            plt.plot(test_accs[merger_name]/count[merger_name], label=merger_name)
            if merger_name in test_accs_conf_int:
                plt.fill_between(np.arange(len(test_accs[merger_name]/count[merger_name])), test_accs[merger_name]/count[merger_name] - test_accs_conf_int[merger_name], test_accs[merger_name]/count[merger_name] + test_accs_conf_int[merger_name], alpha=0.1)

        plt.legend()
        # title with the number of curves of the last merger_name
        plt.title(f"Test accuracies for {merger_name_copy} with {count[merger_name_copy]} curves")
        plt.savefig("./outputs/plots/" + datetime_string + "test_accs.png")
        plt.close()
        # plot all the curves in alpha_dict_gen with matplotlib
        for merger_name in alpha_dict_gen:
            plt.plot(alpha_dict_gen[merger_name], label=merger_name)
        plt.legend()
        plt.savefig("./outputs/plots/" + datetime_string + "alpha_gen.png")
        plt.close()
        merger_name = merger_name_copy
        # save in a npy file the accuracies and the losses
        print(f"Saving {counter_merge // nb_exp} {str(counter_merge // nb_exp)} at time {datetime_string}")
        np.save("./outputs/" + datetime_string + "/accuracies_" + merger_name + "_" + str(counter_merge // nb_exp) + ".npy", accuracies_dict)
        np.save("./outputs/" + datetime_string + "/train_accuracies_" + merger_name + "_" + str(counter_merge // nb_exp) + ".npy", train_accuracies_dict)
        np.save("./outputs/" + datetime_string + "/loss_dict_" + merger_name + "_" + str(counter_merge // nb_exp) + ".npy", loss_dict)
        np.save("./outputs/" + datetime_string + "/alpha_dict_" + merger_name + "_" + str(counter_merge // nb_exp) + ".npy", alpha_dict)

