import random
import csv
import copy

def generate_tree(depth):
    '''recursively generate trees'''
  global mgd

  if depth>mgd:
    mgd=depth

  if depth==0:
    gentree=[]

  if depth>mgd:
    mgd=depth
  if depth < min_depth or (random.random()<chance_next_tree and depth<max_depth):
    f=random.choice(funcs)
    if f=="not":
      gentree=[f, generate_tree(depth+1)]
    else:
      gentree=[f, generate_tree(depth+1), generate_tree(depth+1)]

  else:
      gentree=round(random.uniform(*dir_thresh), 2)
      return gentree

  return(gentree)

def subtree_metadata(tree, current_position=0, current_level=0):
    """
    Generates tree-specific metadata regarding all of its subtrees
    Output - list of lists:
    -[0] position reference.
    -[1] subtree depth.
    -[2] level.
    """
    if not isinstance(tree, list):
        return [[current_position, 0, current_level]]

    results = []
    max_subtree_depth = 0
    child_position = current_position + 1

    for child in tree[1:]:
        child_results = subtree_metadata(child, child_position, current_level + 1)
        results.extend(child_results)
        child_position += len(child_results)

        max_subtree_depth = max(max_subtree_depth, child_results[0][1] + 1)


    results.insert(0, [current_position, max_subtree_depth, current_level])
    return results

"""Access functions below"""

def get_subtree(tree, path):

    #Retrieve a subtree from a nested list structure using a path.

    subtree = tree
    for step in path:
        subtree = subtree[step]
    return subtree


def set_subtree(tree, path, new_subtree):

    #Set (replace) a subtree in a nested list structure using a path.

    if not path:
        return new_subtree

    tree_copy = copy.deepcopy(tree)
    subtree = tree_copy
    for step in path[:-1]:
        subtree = subtree[step]


    subtree[path[-1]] = new_subtree
    return tree_copy


def count_nodes(tree):
    #Count the number of nodes in the tree (used in subtree_byindex).

    if not isinstance(tree, list):
        return 1
    return 1 + sum(count_nodes(child) for child in tree[1:])

def subtree_byindex(tree, target_index, current_index=0):
    
    ''' Retrieve a subtree from a nested list structure using a unique index (see subtree_metadata position reference).
        Easy way to reference to a particular subtree. Used for gene crossover
        tree-> metadata index -> path -> reference '''
    if current_index == target_index:
        return tree, ()

    if isinstance(tree, list):
        next_index = current_index + 1
        for i, child in enumerate(tree[1:], start=1):
            subtree, path = subtree_byindex(child, target_index, next_index)
            if subtree is not None:
                return subtree, (i,) + path
            next_index += count_nodes(child)

    return None, None


"""Access functions above"""


def crossover(tree1,tree2):

  mt1=subtree_metadata(tree1)
  mt2=subtree_metadata(tree2)

  ttypes=["SSS"]

  #SSS - Same size subtree; switch only subtrees that have the same size.
  # analiza metotde de crossover
  cross_type=random.choice(ttypes)

  if cross_type=="SSS":
    tree1_picklist=[]
    tree2_picklist=[]

    mm_depth=min(mt1[0][1],mt2[0][1],max_depth-1)
    try:
      sss_depth=random.randint(0,mm_depth)
    except:
      sss_depth=0

    for elem in mt1:
      if elem[1]==sss_depth:
        tree1_picklist.append(elem[0])

    for elem in mt2:
      if elem[1]==sss_depth:
        tree2_picklist.append(elem[0])


    tree1_pick=random.choice(tree1_picklist)
    tree2_pick=random.choice(tree2_picklist)

    st=get_subtree(tree1,subtree_byindex(tree1,tree1_pick)[1])
    st2=get_subtree(tree2,subtree_byindex(tree2,tree2_pick)[1])

    child1=set_subtree(tree1,subtree_byindex(tree1,tree1_pick)[1],st2)
    child2=set_subtree(tree2,subtree_byindex(tree2,tree2_pick)[1],st)

    return child1,child2

def mutate(tree):

  #lc = leaf change
  #os = operation swap
  # - incrucisare cu un arbore aleator
  types=["lc"]
  mutation_type=random.choice(types)
  meta=subtree_metadata(tree)

  # os not finished yet
  if mutation_type=="os":
    nodes=[elem for elem in meta if elem[1]!=0 ]
    node=random.choice(nodes)
    print(node)
    print(subtree_byindex(tree,node[0]))
    print(subtree_byindex(tree,node[0])[0])
    print(get_subtree(tree,subtree_byindex(tree,node[0])[1]))

  if mutation_type=="lc":
    leaves=[elem for elem in meta if elem[1]==0 ]
    leaf=random.choice(leaves)
    mutated_tree=set_subtree(tree,subtree_byindex(tree,leaf[0])[1],round(random.uniform(*dir_thresh), 2))

  return mutated_tree

def tourname(left,right):
  lr=fitness(left,data,init_cash)[0]
  rr=fitness(right,data,init_cash)[0]
  if lr>rr:
    return left
  if lr<rr:
    return right
  if lr==rr:
    return random.choice([left,right])

def dissect(data, thresh):
    point=data[0]
    ch = []
    hi,lo=data[0],data[0]

    last=None
    for p in data:
      if last==None:

        if p>=lo:
          if p>=lo*(1+thresh):
            last=True
            ch.append(last)
            point=p
            continue

          if p<lo*(1+thresh):
           hi=p

        if p<=hi:
          if p<=hi*(1-thresh):
            last=False
            ch.append(last)
            point=p
            continue

          if p>hi*(1+thresh):
            lo=p

      if last==True:

        if p<=point*(1-thresh):
          last=False
          point=p

        if p>=point:
          point=p

      if last==False:
        if p>=point*(1+thresh):
          last=True
          point=p

        if p<=point:
          point=p

      ch.append(last)
    return ch

def lookup(thrh,dic):
  for elem in dic:
    if elem[0]==thrh:
      return elem[1]

def run_tree(tree,data):
  if type(tree)==int or type(tree)==float:
    return lookup(tree,data)

  op=tree[0]

  if op=="not":
    mid=tree[1]

    if type(mid)==list:
      mid=run_tree(mid,data)

    mid=lookup(mid,data)

  else:
    left=tree[1]
    right=tree[2]

    if type(left)==list:
      left=run_tree(left,data)

    if type(right)==list:
      right=run_tree(right,data)
    if left != True and left != False:
      left=lookup(left,data)
    if right != True and right != False:
      right=lookup(right,data)


  if op=="not":
    return not mid
  if op=="or":
    return left or right
  if op=="and":
    return left and right
  if op=="nor":
    return not (left or right)
  if op=="xor":
    return left != right

def fitness(tree,data,init_cash,cooldown=0):

  maxx=0
  l1=[]
  dc_lst=[]
  st_mt=subtree_metadata(tree)
  cd_buy=0
  cd_sell=0
  cash=init_cash


  for i in range(len(st_mt)):
    if st_mt[i][1]==0:
      l1.append(subtree_byindex(tree,st_mt[i][0])[0])

  l1=set(l1)

  for elem in l1:
    dc_lst.append([elem,dissect(data,elem)])
  # data is trimmed to only have points where there is no "none" value
  for elem in dc_lst:
    cn=elem[1].count(None)
    if cn>maxx:
      maxx=cn

  for elem in dc_lst:
    elem[1]=elem[1][maxx:]

  data=data[maxx:]
  stock=0
  entries=0
  exits=0

  for i in range(len(data)):
    p=data[i]
    dict_=[[dc_lst[j][0],dc_lst[j][1][i]] for j in range(len(dc_lst))]
    response=(run_tree(tree,dict_))

    if response==True:
      if cash>p and cd_buy==0:
        stock=stock+1
        cash=cash-p
        entries=entries+1
        cd_buy=cooldown
      if cd_buy>0:
        cd_buy=cd_buy-1

    if response==False:
      if stock>0 and cd_sell==0:
        stock=stock-1
        cash=cash+p
        exits=exits+1
        cd_sell=cooldown
      if cd_sell>0:
        cd_sell=cd_sell-1

  if stock>0:
    cash=cash+stock*p

  return cash/init_cash*100,entries,exits


def GA_loop(curr_gen,data):

  kq=0
  results=[]

  for z in range(gens):
    next_gen=[]
    results=[]

    for elem in curr_gen:
      results.append([elem,fitness(elem,data,init_cash)[0]])


    results.sort(reverse=True,key=lambda x: x[1])

    if len(results)>pop_size:
      print("Warning: Results list has too many elements!")
    if len(curr_gen)>pop_size:
      print("Warning: Current generation list has too many elements!")

    print("Gen "+str(z+1)+": "+str(round(results[0][1],2))+" // "+str(round(results[1][1],2))+" // "+str(round(results[2][1],2))+" // "+str(round(results[3][1],2))+" // "+str(round(results[4][1],2)))

    if results[0][0]==results[1][0]==results[2][0]==results[3][0]==results[4][0]:
      print("Identical top 5 solutions; computation halted at "+str(z+1)+"th computation.")
      print("Max. achieved return: "+str(results[0][1])+"%")

      if z+1>=gens/4:
        print("Best solution: ")
        print(results[0][0])
      print("\n")

      if z+1>gens/4:
        kq=1
        break
      else:
        print("Computation resumed; algorithm must go thru a minimum of a quarter of the intended generations.")
        print("\n")

    if round(results[0][1],5)==round(results[1][1],5)==round(results[2][1],5)==round(results[3][1],5)==round(results[4][1],5):
      print("Identical performance top 5 solutions; computation halted at "+str(z+1)+"th computation.")
      print("Max. achieved return: "+str(round(results[0][1],2))+"%")

      if z+1>=gens/4:
        print("Best solution: ")
        print(results[0][0])
      print("\n")

      if z+1>=gens/4:
        kq=1
        break
      else:
        print("Computation resumed; algorithm must go thru a minimum of a quarter of the intended generations.")
        print("\n")

    #elitism // duplicates are skipped
    #elitism_ratio*pop_size

    i=0
    while len(next_gen)<max(int(elitism_ratio*pop_size),1):

      if results[i][0] not in next_gen:
        next_gen.append(results[i][0])
      i=i+1

    #mutation, crossover, reproduction
    while len(next_gen)<= pop_size:

        rand=round(random.uniform(0,1), 3)

        if rand<=mut_prob:
          t1=results[random.randint(1,len(results))-1][0]
          t2=results[random.randint(1,len(results))-1][0]
          w=tourname(t1,t2)
          next_gen.append(mutate(w))


        if rand<=(mut_prob+repr_prob) and rand>mut_prob:
          t1=results[random.randint(1,len(results))-1][0]
          t2=results[random.randint(1,len(results))-1][0]
          w=tourname(t1,t2)
          next_gen.append(w)


        if rand>(mut_prob+repr_prob):
          i11=random.randint(1,len(results))-1
          p11=results[i11-1]


          i12=random.randint(1,len(results))-1
          p12=results[i12]


          i21=random.randint(1,len(results))-1
          p21=results[i21]


          i22=random.randint(1,len(results))-1
          p22=results[i22]


          if p11[1]>p12[1]:
            p1=p11[0]
          if p11[1]<p12[1]:
            p1=p11[0]
          if p11[1]==p12[1]:
            p1=random.choice([p11[0],p12[0]])

          if p21[1]>p22[1]:
            p2=p21[0]
          if p21[1]<p22[1]:
            p2=p21[0]
          if p21[1]==p22[1]:
            p2=random.choice([p21[0],p22[0]])

          ch1,ch2=crossover(p1,p2)
          next_gen.append(ch1)
          next_gen.append(ch2)
          next_gen.append(p1)
          next_gen.append(p2)

    while len(next_gen)>50:
      next_gen.pop(-1)

    curr_gen=next_gen

  if kq==1:
    return results

  else:
    results=[]
    for elem in curr_gen:
      results.append([elem,fitness(elem,data,init_cash)[0]])
    results.sort(reverse=True,key=lambda x: x[1])

    return results


#arguments
funcs = ['and', 'or', 'not', 'xor', 'nor']
dir_thresh = (0.01, 0.1)
min_depth = 2
max_depth = 8
pop_size = 100
gens = 100
t_size = 2
chance_next_tree = 0.65
cross_prob = 0.97
mut_prob = 0.01
repr_prob=0.01
elitism_ratio=0.01
init_cash = 100000
mgd=0

#initialise population
init_pop=[generate_tree(0) for _ in range(pop_size)]
print("Population initialised\n")

#read data
with open('s&p.csv', 'r') as file:
  data = [float(row[0]) for row in csv.reader(file) if row[0].replace('.', '', 1).isdigit()]


#run the GA
final=GA_loop(init_pop,data)

