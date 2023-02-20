import numpy as np
import random
import scipy.stats

### load file
def read_file(filepath, extra=''):
    text = []
    datas = {}
    with open(filepath, 'r') as file:
        for line in file.readlines():
            line = line.replace('\n', '')
            if line[0] == '#': text.append(line[2:])
            elif line[0] == '[':
                title, data = line[1:].split(']')
                if not((extra+title) in datas): datas[extra+title] = []
                datas[extra+title].append(100*np.array([float(x) for x in data.split(', ')]))

    return '\n'.join(text), datas

def transform(datas, transformation):
    return {name: datas[transformation[name]] for name in transformation}

def summary(datas):
    for name in datas:
        print(f"- {name}: {len(datas[name])} curves with {datas[name][0].shape[0]-1} rounds each.")


### HSV colors
def hsv_to_rgb(h, s, v):
        if s == 0.0: return (v, v, v)
        i = int(h*6.)
        f = (h*6.)-i
        p,q,t = v*(1.-s), v*(1.-s*f), v*(1.-s*(1.-f)); i%=6
        if i == 0: return (v, t, p)
        if i == 1: return (q, v, p)
        if i == 2: return (p, v, t)
        if i == 3: return (p, q, v)
        if i == 4: return (t, p, v)
        if i == 5: return (v, p, q)

def new_color(distributed, n):
    if not distributed: return random.uniform(0, 1)
    return (distributed[-1] + 1.0/n)%1

def make_colors(fixed_colors, names):
    return {name: fixed_colors[name] if name in fixed_colors else (random.uniform(0,1), 1, 1) for name in names}


### confidence interval
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
    
    
### compute Rp
def get_R(datas, r=60):
    res = {}
    for name in datas:
        r_mins = []
        for line in datas[name]:
            r_min = min(filter(lambda i: line[i] >= r, range(len(line))))
            r_mins.append(r_min)
        r_mins = np.array(r_mins)
        res[name] = (r_mins.mean(), r_mins.std())
    return res

def Rp(name, data, p=60, alpha=95, filename=None):
    talpha = {95: 1.96}[alpha]
    if filename is None:
        print(f"R{p} of {name} at {alpha}%:")
        r60 = get_R(data, r=p)
        for c in r60:
            print(f"- {c}: {r60[c][0]} ± {talpha * r60[c][1] / np.sqrt(len(data[c]))}")
    else:
        with open(filename, 'w') as file:
            file.write(f"R{p} of {name} at {alpha}%:\n")
            r60 = get_R(data, r=p)
            for c in r60:
                file.write(f"- {c}: {r60[c][0]} ± {talpha * r60[c][1] / np.sqrt(len(data[c]))}\n")


### compute last accuracy
def last_accuracy(name, data):
    print(f"Last accuracy of {name}:")
    for c in data:
        print(f"> {c}: {np.array(data[c])[:,-1].mean()}%")