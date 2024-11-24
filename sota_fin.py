import random
import csv
with open('s&p.csv', 'r') as file:
    data = [float(row[0]) for row in csv.reader(file) if row[0].replace('.', '', 1).isdigit()]

funcs = ['and', 'or', 'not', 'xor', 'nor']
dir_thresh = (0.01, 0.1)
min_depth = 0
max_depth = 8
pop_size = 100
gens = 50
t_size = 2
cross_prob = 0.8
chance_next_tree = 0.5
mut_prob = 0.01
init_cash = 10000

def make_tree(depth=0):
    if depth < min_depth:
        f = random.choice(funcs)
        if f == 'not':
            return (f, make_tree(depth + 1))
        return (f, make_tree(depth + 1), make_tree(depth + 1))
    if depth >= max_depth or random.random() < chance_next_tree:
        return round(random.uniform(*dir_thresh), 2)
    f = random.choice(funcs)
    if f == 'not':
        return (f, make_tree(depth + 1))
    return (f, make_tree(depth + 1), make_tree(depth + 1))

def make_pop():
    return [make_tree() for _ in range(pop_size)]

def dissect(data, thresh):
    hi, lo = data[0], data[0]
    ch = []
    for p in data:
        if p >= hi * (1 + thresh):
            ch.append("up")
            hi = p
            lo = p
        elif p <= lo * (1 - thresh):
            ch.append("down")
            hi = p
            lo = p
        else:
            ch.append(None)
    return ch

def eval_tree(tree, changes):
    if isinstance(tree, float):
        return changes.get(tree, [])
    op = tree[0]
    if op == 'not':
        child = eval_tree(tree[1], changes)
        r = []
        for val in child:
            if val is None:
                r.append(None)
            else:
                r.append(not val)
        return r
    l = eval_tree(tree[1], changes)
    r = eval_tree(tree[2], changes) if len(tree) > 2 else []
    out = []
    for a, b in zip(l, r):
        if a is None or b is None:
            out.append(None)
        elif op == 'and':
            out.append(a and b)
        elif op == 'or':
            out.append(a or b)
        elif op == 'xor':
            out.append(a != b)
        elif op == 'nor':
            out.append(not (a or b))
    return out

def fitness(tree, data):
    tmap = {}
    for _ in range(10):
        t = round(random.uniform(*dir_thresh), 2)
        tmap[t] = dissect(data, t)
    sig = eval_tree(tree, tmap)
    c, bal = init_cash, 0
    for s, p in zip(sig, data):
        if s == "up" and c >= p:
            c = c - p
            bal = bal + 1
        elif s == "down" and bal > 0:
            c = c + p
            bal = bal - 1
    return c + bal * data[-1]

def evaluate_strategy(tree, data):
    tmap = {}
    for _ in range(10):
        t = round(random.uniform(*dir_thresh), 2)
        tmap[t] = dissect(data, t)
    sig = eval_tree(tree, tmap)
    c, bal = init_cash, 0
    for s, p in zip(sig, data):
        if s == "up" and c >= p:
            c = c - p
            bal = bal + 1
        elif s == "down" and bal > 0:
            c = c + p
            bal = bal - 1
    return c + bal * data[-1]

def tour_select(pop, scores):
    pick = random.sample(pop, t_size)
    #pop.remove(pick[0])
    #pop.remove(pick[1])
    best = pick[0]
    best_score = scores[best]
    for t in pick[1:]:
        if scores[t] > best_score:
            best = t
            best_score = scores[t]
    return best

def cross(p1, p2):
    if random.random() > cross_prob:
        return p1, p2
    if isinstance(p1, float) or isinstance(p2, float):
        return p1, p2
    if len(p1) < 2 or len(p2) < 2:
        return p1, p2
    c1 = list(p1)
    c2 = list(p2)
    c1[1], c2[1] = c2[1], c1[1]
    return tuple(c1), tuple(c2)

def mutate(tree):
    if random.random() > mut_prob:
        return tree
    if isinstance(tree, float):
        return round(random.uniform(*dir_thresh), 2)
    tree = list(tree)
    idx = random.randint(1, len(tree) - 1)
    tree[idx] = make_tree()
    return tuple(tree)

def gp(data):
    train = data[:len(data)//2]
    test = data[len(data)//2:]
    pop = make_pop()
    for g in range(gens):
        scores = {}
        for t in pop:
            scores[t] = fitness(t, train)
        new_pop = []
        while len(new_pop) < pop_size:
            p1 = tour_select(pop, scores)
            p2 = tour_select(pop, scores)
            p1s=evaluate_strategy(p1,test)
            p2s=evaluate_strategy(p2,test)
            #print(len(pop))
            if p1s>p2s:
              new_pop.append(p1)
            if p1s<p2s:
              new_pop.append(p2)
            if p1s==p2s:
              if random.random()>0.5:
                new_pop.append(p1)
              else:
                new_pop.append(p2)

            c1, c2 = cross(p1, p2)
            c1s=evaluate_strategy(c1,test)
            c2s=evaluate_strategy(c2,test)
            if c1s>c2s:
              new_pop.append(mutate(c1))
            if c1s<c2s:
              new_pop.append(mutate(c2))
            if c1s==c2s:
              if random.random()>0.5:
                new_pop.append(mutate(c1))
              else:
                new_pop.append(mutate(c2))
            #print(len(new_pop))

            #new_pop.append(mutate(c1))
            #new_pop.append(mutate(c2))
        pop = new_pop
        best_tree = max(pop, key=lambda t: fitness(t, train))
        final_value = evaluate_strategy(best_tree, test)
        percent_return = ((final_value - init_cash) / init_cash) * 100
        if percent_return !=-1:
          print(f"gen "+str(g)+", final value "+str(final_value)+", "+str(percent_return)+"% return")
    return best_tree

prices = data
best_tree = gp(prices)
print("best tree:", best_tree)