import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats
import re

def hsv_to_rgb(h, s, v):
        if s == 0.0: return (v, v, v)
        i = int(h*6.) # XXX assume int() truncates!
        f = (h*6.)-i
        p,q,t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f)); i%=6
        if i == 0: return (v, t, p)
        if i == 1: return (q, v, p)
        if i == 2: return (p, v, t)
        if i == 3: return (p, q, v)
        if i == 4: return (t, p, v)
        if i == 5: return (v, p, q)

def mean_confidence_interval(datas, confidence=0.99):
    mean, low, high = [], [], []
    for t in range(len(datas[0])):
        data = [datas[i][t] for i in range(len(datas))]
        a = 1.0 * np.array(data)
        n = len(a)
        if confidence is None:
            mean.append(np.mean(a))
        else:
            m, se = np.mean(a), scipy.stats.sem(a)
            h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
            mean.append(m)
            low.append(m-h)
            high.append(m+h)
    if confidence is None: return np.array(mean)
    return np.array(mean), np.array(low), np.array(high)

def read_file(filepath):
    text = []
    datas = {}
    with open(filepath, 'r') as file:
        for line in file.readlines():
            line = line.replace('\n', '')
            if line[0] == '#': text.append(line[2:])
            elif line[0] == '[':
                title, data = line[1:].split(']')
                if not(title in datas): datas[title] = []
                datas[title].append(100*np.array([float(x) for x in data.split(', ')]))

    return '\n'.join(text), datas

if __name__ == '__main__':
    filename = sys.argv[1] if len(sys.argv) > 1 else "0"
    confidence = float(sys.argv[2]) if (len(sys.argv) > 2 and sys.argv[2] != '0') else None
    subcurves = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
    text, datas = read_file(f"{filename}.inf") if ("/" in filename) else read_file(f"outputs/{filename}.inf")

    reg_title = re.compile("(?P<name>[a-zA-Z0-9]+)(\((?P<num>[-+]?(?:\d*\.\d+|\d+))\))?")

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
