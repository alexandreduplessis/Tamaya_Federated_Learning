from extra.utils import *
from extra.plots import *







if __name__ == '__main__':
    ### load curves
    filenames = ["final/exp1_accs.inf", "final/exp2_accs.inf", "final/exp3_accs.inf", "final/exp5_accs.inf", "outputs/exp7iid_accs.inf", "outputs/exp7niid_accs.inf", "outputs/exp10iid_accs.inf"]
    extras = ["exp1_", "exp2_", "exp3_", "exp5_", "exp7iid_", "exp7niid_", "exp10iid_"]
    exp1, exp2, exp3, exp5, exp7iid, exp7niid, exp10iid = filenames
    files = {filename: read_file(filename, extra)[1] for filename, extra in zip(filenames, extras)}

    ### rename curves

    for data in files:
        print(data, ':')
        summary(files[data])

    input("-"*133)

    datas = {}
    datas["SET1"] = transform(files[exp1], {"FedAvg": "exp1_FedAvg", "FedSoftWorse": "exp1_FedSoftmax", "FedSoftBetter": "exp1_FedSoftmin"})
    datas["SET2"] = transform(files[exp2], {"FedAvg": "exp2_FedAvg", "FedSoftWorseAvg": "exp2_FedSmaxAvg", "FedSoftBetterAvg": "exp2_FedSminAvg"})
    datas["SET3"] = transform(files[exp5], {"FedAvg": "exp5_FedAvg", "FedSoftWorse": "exp5_FedSoftMax", "FedSoftBetter": "exp5_FedSoftMin"})
    datas["SET4"] = transform(files[exp3], {"FedWorse(k=10%)": "exp3_FedMaxk", "FedWorse(k=20%)": "exp3_FedMaxK", "FedBetter(k=10%)": "exp3_FedMink", "FedBetter(k=20%)": "exp3_FEdMinK"})
    datas["SET5"] = transform({**files[exp1], **files[exp3]}, {"FedSoftWorse": "exp1_FedSoftmax", "FedSoftBetter": "exp1_FedSoftmin", "FedWorse(k=10%)": "exp3_FedMaxk", "FedBetter(k=10%)": "exp3_FedMink"})
    datas["SET6"] = transform({**files[exp1], **files[exp3]}, {"FedSoftWorse": "exp1_FedSoftmax", "FedSoftBetter": "exp1_FedSoftmin", "FedWorse(k=20%)": "exp3_FedMaxK", "FedBetter(k=20%)": "exp3_FEdMinK"})
    datas["SET7"] = transform(files[exp7niid], {"FedAvg": "exp7niid_FedAvg", "FedSBetterAvgSWorse": "exp7niid_FedSminAvgSmax", "FedSWorseAvgSBetter": "exp7niid_FedSmaxAvgSmin"})
    datas["SET8"] = transform(files[exp7iid], {"FedAvg": "exp7iid_FedAvg", "FedSBetterAvgSWorse": "exp7iid_FedSminAvgSmax", "FedSWorseAvgSBetter": "exp7iid_FedSmaxAvgSmin"})
    datas["SET9"] = transform(files[exp2], {"FedAvg": "exp2_FedAvg", "FedSoftWorseAvg": "exp2_FedSmaxAvg", "FedSoftBetterAvg": "exp2_FedSminAvg", "FedAvgSoftWorse": "exp2_FedAvgSmax", "FedAvgSoftBetter": "exp2_FedAvgSmin"})
    # datas["exp2"] = transform(files[exp2], {"FedAvg": "FedAvg", "FedSmaxAvg": "FedSmaxAvg", "FedSminAvg": "FedSminAvg"})
    # datas["exp3"] = transform({**files[exp3], **files[exp1]}, {"FedMax": "FedMaxK", "FedMin": "FEdMinK", "FedSoftmax": "FedSoftmax", "FedSoftmin": "FedSoftmin"})
    # datas["exp5"] = transform(files[exp5], {"FedAvg": "FedAvg", "FedSoftmax": "FedSoftMax", "FedSoftmin": "FedSoftMin"})
    datas["SET10"] = transform({**files[exp10iid], **files[exp7iid]}, {"FedAvg": "exp7iid_FedAvg", "FedWCostAvg": "exp10iid_FedWCostAvg", "FedControl1": "exp10iid_FedControl1", "FedControl2": "exp10iid_FedControl2"})

    for data in datas:
        print(data, ':')
        summary(datas[data])

    input("-"*133)
    ### color curves
    names = set([key for data in datas for key in datas[data].keys()])
    fixed_colors = make_colors({"FedAvg": (0.8,0,0),
                                "FedWorse(k=20%)": (0.1,0.6,1),
                                "FedBetter(k=20%)": (0.7,0.6,1),
                                "FedSoftWorse": (0.0,0.9,1),
                                "FedSoftBetter": (0.6,0.9,1),
                                "FedSoftWorseAvg": (0.0,0.6,1),
                                "FedSoftBetterAvg": (0.6,0.6,1),
                                "FedWorse(k=10%)": (0.1,1,1),
                                "FedBetter(k=10%)": (0.7,1,1),
                                "FedSBetterAvgSWorse": (0.1,1,1),
                                "FedSWorseAvgSBetter": (0.7,1,1),
                                "FedAvgSoftWorse": (0.9,1,1),
                                "FedAvgSoftBetter": (0.5,1,1)}, names)

    ### plots

    Rp("SET10", datas["SET10"])
    basic_plot(datas["SET10"], fixed_colors)
    exit(0)
    Rp("SET1", datas["SET1"], filename="set1.txt")
    last_accuracy("SET1", datas["SET1"])
    doublezoom_plot(datas["SET1"], fixed_colors, option={"x0": 5, "x1": 12, "y0": 56, "y1": 70, "x2": 130, "x3": 150, "y2": 78, "y3": 79}, filename="set1.png")

    Rp("SET2", datas["SET2"], filename="set2.txt")
    last_accuracy("SET2", datas["SET2"])
    doublezoom_plot(datas["SET2"], fixed_colors, option={"x0": 2, "x1": 12, "y0": 40, "y1": 70, "x2": 130, "x3": 150, "y2": 78, "y3": 80}, filename="set2.png")
    #
    Rp("SET3", datas["SET3"], filename="set3.txt")
    last_accuracy("SET3", datas["SET3"])
    doublezoom_plot(datas["SET3"], fixed_colors, option={"x0": 2, "x1": 12, "y0": 56, "y1": 83, "x2": 80, "x3": 100, "y2": 88.5, "y3": 90}, filename="set3.png")

    Rp("SET4", datas["SET4"], filename="set4.txt")
    last_accuracy("SET4", datas["SET4"])
    doublezoom_plot(datas["SET4"], fixed_colors, option={"x0": 1, "x1": 10, "y0": 20, "y1": 65, "x2": 130, "x3": 150, "y2": 74, "y3": 79}, filename="set4.png")

    Rp("SET5", datas["SET5"], filename="set5.txt")
    basic_plot(datas["SET5"], fixed_colors, filename="set5.png")
    Rp("SET6", datas["SET6"], filename="set6.txt")
    basic_plot(datas["SET6"], fixed_colors, filename="set6.png")

    Rp("SET8", datas["SET8"], filename="set8.txt")
    last_accuracy("SET8", datas["SET8"])
    doublezoom_plot(datas["SET8"], fixed_colors, option={"y2": 75, "y3": 78}, filename="set8.png")
    
    Rp("SET9", datas["SET9"], filename="set9.txt")
    doublezoom_plot(datas["SET9"], fixed_colors, option={"y2": 75, "y3": 78}, filename="set9.png")


    input("-"*133)
    exit(0)
    plot_complete(datas["exp10iid"], fixed_colors=fixed_colors)
    exit(0)
    R60("exp1", datas["exp1"], filename="r60_exp1.txt")
    plot_complete(datas["exp1"], fixed_colors=fixed_colors, option="zoom1", filename="curve_exp1.png")

    R60("exp2", datas["exp2"], filename="r60_exp2.txt")
    plot_complete(datas["exp2"], fixed_colors=fixed_colors, option="zoom2", filename="curve_exp2.png")

    R60("exp3", datas["exp3"], filename="r60_exp3.txt")
    plot_complete(datas["exp3"], fixed_colors=fixed_colors, filename="curve_exp3.png")

    R60("exp5", datas["exp5"], filename="r60_exp5.txt")
    plot_complete(datas["exp5"], fixed_colors=fixed_colors, option="zoom3", filename="curve_exp5.png")
    exit(0)
    # fix colors

    # Expérience 1

    print("Courbe 1:")
    summary(files[exp1][1])
    print("R60=", get_R(files[exp1][1], r=60))
    print("R70=", get_R(files[exp1][1], r=70))
    plot_complete(files[exp1][1], fixed_colors=fixed_colors, filename="curve1_zoom.png")
    plot_complete(files[exp1][1])

    reg_title = re.compile("(?P<name>[a-zA-Z0-9]+)(\((?P<num>[-+]?(?:\d*\.\d+|\d+))\))?")

    exit(0)
    names = {}
    names_l = []
    for title in datas:
        name = reg_title.fullmatch(title).group('name')
        if name not in names:
            names[name] = None
            names_l.append(name)
    c0 = random.uniform(0, 1)
    colors = {}
    for title in datas:
        if True:
            match = reg_title.fullmatch(title)
            matchname = match.group('name')
            matchvalue = match.group('num')
            if matchvalue is None: matchvalue = 1.0
            else: matchvalue = (1.0 + float(matchvalue))/2.0
            if names[matchname] is None: names[matchname] = (c0 + names_l.index(matchname)/len(names)) % 1
            colors[title] = hsv_to_rgb(names[matchname], 1.0, matchvalue)
        else:
            print(f"Random color picked on unknown title {title}.")
            colors[title] = np.random.uniform(0, 1, size=3)

    for title in sorted(datas.keys()):
        epochs = list(range(len(datas[title][0])))
        print(f"[{title}]: {len(datas[title])} curves with {len(epochs)-1} epochs")
        if not(title in colors): colors[title] = np.random.uniform(0, 1, size=3)
        if subcurves:
            for courbe in datas[title]: plt.plot(epochs, courbe, color=colors[title], linestyle=':', alpha=0.5)

        if confidence is None:
            mean = mean_confidence_interval(datas[title], None)
        else:
            mean, low, high = mean_confidence_interval(datas[title], confidence)

        plt.plot(epochs, mean, color=colors[title], label=title)
        plt.text(epochs[-1], mean[-1], title, color=colors[title])

        if confidence is not None:
            plt.fill_between(epochs, low, high, color=colors[title], alpha=0.3)

    plt.legend()
    plt.show()
