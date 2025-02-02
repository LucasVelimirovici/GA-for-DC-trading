import random
import csv
import copy
from collections import Counter
import statistics

funcs = ['and', 'or']
comps=['<','>']
indicators=['TMV/NA', 'OSV/NA', 'AVG_OSV/3', 'AVG_OSV/5', 'AVG_OSV/10', 'RDC/NA', 'AVG_RDC/3',
'AVG_RDC/5', 'AVG_RDC/10', 'TDC/NA', 'AVG_TDC/3', 'AVG_TDC/5', 'AVG_TDC/10',
'NDC/10', 'NDC/20', 'NDC/30', 'NDC/40', 'NDC/50', 'CDC/10', 'CDC/20', 'CDC/30',
'CDC/40', 'CDC/50', 'AT/10', 'AT/20', 'AT/30', 'AT/40', 'AT/50']
dir_thresh = (0.00, 1)
min_depth = 1
max_depth = 6
pop_size = 500
gens = 50
cross_prob=0.95
mut_prob=1-cross_prob
t_size = 2
chance_next_tree = 0.65
elitism_ratio=0.01
mgd=0

def generate_tree(depth):
  global mgd

  if depth>mgd:
    mgd=depth

  if depth==0:
    gentree=[]

  if depth>mgd:
    mgd=depth
  if depth < min_depth or (random.random()<chance_next_tree and depth<max_depth):
    f=random.choice(funcs)
    gentree=[f, generate_tree(depth+1), generate_tree(depth+1)]

  else:
      gentree=[random.choice(comps),random.choice(indicators),round(random.uniform(*dir_thresh), 2)]
      return gentree

  return(gentree)

def subtree_metadata(tree, current_position=0, current_level=0):
    """
    Output - list of lists:
    -[0] position reference.
    -[1] subtree depth.
    -[2] level.
    """
    if tree[0] not in funcs:
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

def count_nodes(tree):
    #Count the number of nodes in the tree (used in subtree_byindex).

    if tree[0] not in funcs:
        return 1
    return 1 + sum(count_nodes(child) for child in tree[1:])

def subtree_byindex(tree, target_index, current_index=0):
    # Retrieve a subtree from a nested list structure using a unique index (see subtree_metadata position reference).
    # Easy way to reference to a particular subtree. Used for gene crossover
    # tree-> metadata index -> path -> reference
    if current_index == target_index:
        return tree, ()

    if tree[0] in funcs:
        next_index = current_index + 1
        for i, child in enumerate(tree[1:], start=1):
            subtree, path = subtree_byindex(child, target_index, next_index)
            if subtree is not None:
                return subtree, (i,) + path
            next_index += count_nodes(child)

    return None, None

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
  types=["lc","os"]
  mutation_type=random.choice(types)
  meta=subtree_metadata(tree)
  # os not finished yet
  if mutation_type=="os":

    nodes=[elem for elem in meta if elem[1]!=0 and elem[1]<=max_depth]
    node=random.choice(nodes)

    st=subtree_byindex(tree,node[0])[0]
    adrs=subtree_byindex(tree,node[0])[1]
    if st[0]=="and":
      st[0]="or"
    else:
      st[0]="and"

    mutated_tree=set_subtree(tree,adrs,st)


  if mutation_type=="lc":
    nodes=[elem for elem in meta if elem[1]==0 ]
    node=random.choice(nodes)
    node=subtree_byindex(tree,node[0])
    leaf=node[0]
    adr=node[1]

    guess=random.random()
    if guess>2/3:
      leaf=[random.choice(comps),random.choice(indicators),round(random.uniform(*dir_thresh), 2)]

    if guess>1/3 and guess<=2/3:
      notto=random.choice([0,1,2])
      newleaf=[random.choice(comps),random.choice(indicators),round(random.uniform(*dir_thresh), 2)]
      newleaf[notto]=leaf[notto]

      leaf=newleaf

    if guess<=1/3:
      to=random.choice([0,1,2])
      if to==0:
        leaf[0]=random.choice(comps)
      if to==1:
        leaf[1]=random.choice(indicators)
      if to==2:
        leaf[2]=round(random.uniform(*dir_thresh), 2)

    mutated_tree=set_subtree(tree,adr,leaf)

  return mutated_tree

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

def dissect2(data, thresh):
  point=data[0]
  ch=[]
  hi,lo=data[0],data[0]
  last=None
  count=0
  j=0
  for p in data:
    if last==None:

      if p>=lo:
          if p>=lo*(1+thresh):
            last="UpC"
            ch.append(last)
            point=p
            for i in range(j-count,j):
              ch[i]="Up"
            count=0
            continue

          if p<lo*(1+thresh):
           hi=p
           ch.append(None)

      if p<=hi:
          if p<=hi*(1-thresh):
            last="DownC"
            ch.append(last)
            point=p
            for i in range(j-count,j):
              ch[i]="Down"
            count=0
            continue

          if p>hi*(1+thresh):
            lo=p
            ch.append(None)

    if last=="UpC" or last=="OsU":
      act=0
      if p<=point*(1-thresh):
        last="DownC"
        ch.append("DownC")
        for i in range(j-count+1,j+1):
            ch[i]="Down"
        count=0
        point=p
        act=1
        j=j+1
        continue

      if p>=point:
        point=p
        count=0

      if act==0:
        ch.append("OsU")
        last="OsU"

    if last=="DownC" or last=="OsD":
      act=0
      if p>=point*(1+thresh):
        last="UpC"
        ch.append("UpC")
        for i in range(j-count+1,j+1):
            ch[i]="Up"
        act=1
        count=0
        point=p
        j=j+1
        continue

      if p<=point:
        point=p
        count=0

      if act==0:
        ch.append("OsD")
        last="OsD"

    count=count+1
    j=j+1

  return ch

def NDC(data,pricedata,thresh,period):
  serie=[]

  for i in range(len(data)):
    try:
      rang=data[i-period+1:i+1]
      serie.append(Counter(rang)["UpC"]+Counter(rang)["DownC"])
    except:
      serie.append(None)
  return serie

def NOS(data,pricedata,thresh,period):
  serie=[]

  for i in range(len(data)):
    try:
      ct=0
      rang=data[i-period+1:i+1]
      for k in range(1,len(rang)):
        if rang[k]=="Down" and (rang[k-1]=="UpC" or rang[k-1]=="OsU"):
          ct=ct+1
        if rang[k]=="uP" and (rang[k-1]=="DownC" or rang[k-1]=="OsD"):
          ct=ct+1
      serie.append(ct)
    except:
      serie.append(None)
  return serie

def OSV(data,pricedata,thresh,period):
  serie=[]
  serie.append(None)
  for i in range(1,len(data)):
    found=0
    for j in reversed(range(0,i)):
      if data[j]=="DownC" or data[j]=="UpC":
        found=1
        break
    if found==1:
      serie.append((pricedata[i]-pricedata[j])/(pricedata[j]*thresh))
    else:
      serie.append(None)
  return serie

def AVG_OSV(data,pricedata,thresh,period):
  serie=[]
  values=OSV(data,pricedata,thresh,period)
  for i in range(len(data)):
    try:
      rang_data=values[i-period+1:i+1]
      mean=0
      ct=0
      for elem in rang_data:
        if elem != None:
          mean=mean+elem
          ct=ct+1
      serie.append(mean/ct)
    except:
      serie.append(None)
  return serie

def TMV(data,pricedata,thresh,period):
  serie=[]
  time=[]
  ttype=[]
  t=None
  for i in range(len(data)-1):
    if data[i] in ["Up","DownC","OsU"]:
      t="Up"
      j=i
      while data[j] not in ["Down","UpC","OsD"]:
        j=j+1
        if j==len(data)-1:
          break
    if data[i] in ["Down","UpC","OsD"]:
      t="Down"
      j=i
      while data[j] not in ["Up","DownC","OsU"]:
        j=j+1
        if j==len(data)-1:
          break

    serie.append(pricedata[j]-pricedata[i])
    time.append(abs(i-j))
    ttype.append(t)
  ttype.append(None)
  serie.append(None)
  time.append(None)
  return serie,time,ttype

def RDC(data,pricedata,thresh,period):
  serie=[]
  TMV_V=TMV(data,pricedata,thresh,period)[0]
  TMV_T=TMV(data,pricedata,thresh,period)[1]
  for i in range(len(TMV_V)):
    try:
      serie.append(TMV_V[i]*thresh/TMV_T[i])
    except:
      serie.append(None)

  return serie

def AVG_RDC(data,pricedata,thresh,period):
  serie=[]
  values=RDC(data,pricedata,thresh,period)
  for i in range(len(data)):
    try:
      rang_data=values[i-period+1:i+1]
      mean=0
      ct=0
      for elem in rang_data:
        if elem != None:
          mean=mean+elem
          ct=ct+1
      serie.append(mean/ct)
    except:
      serie.append(None)
  return serie

def TDC(data,pricedata,thresh,period):
  time=TMV(data,pricedata,thresh,period)[1]
  #print(serie)
  return time

def AVG_TDC(data,pricedata,thresh,period):
  serie=[]
  values=TDC(data,pricedata,thresh,period)
  for i in range(len(data)):
    try:
      rang_data=values[i-period+1:i+1]
      mean=0
      ct=0
      for elem in rang_data:
        if elem != None:
          mean=mean+elem
          ct=ct+1
      serie.append(mean/ct)
    except:
      serie.append(None)
  return serie

def AT(data,pricedata,thresh,period):
  serie=[]
  ttype=TMV(data,pricedata,thresh,period)[2]
  time=TMV(data,pricedata,thresh,period)[1]
  #print(time)
  for i in range(len(data)):
    uptime=0
    downtime=0
    try:
      rang_data=ttype[i-period+1:i+1]
      #print(rang_data)
      for j in range(len(rang_data)):
        if rang_data[j]=="Up":
          uptime=uptime+1
        if rang_data[j]=="Down":
          downtime=downtime+1
      serie.append(uptime-downtime)
    except Exception as e:
      #print(e)
      serie.append(None)
  return serie


def CDC(data,pricedata,thresh,period):
  serie=[]
  values=TMV(data,pricedata,thresh,period)[0]
  for i in range(len(data)):
    try:
      rang_data=values[i-period+1:i+1]
      mean=0
      for elem in rang_data:
        if elem is not None:
          mean=mean+abs(elem)
      serie.append(mean)
    except Exception as e:
      #print(e)
      serie.append(None)
  return serie

def normalise(lst):
    numeric_values = [x for x in lst if x is not None]
    min_val = min(numeric_values)
    max_val = max(numeric_values)
    return [(x - min_val) / (max_val - min_val) if x is not None else None for x in lst]

def run_tree(tree,indices):

  if tree[0]=="<" or tree[0]==">":
    for elem in indices:
      if elem[0]==tree[1]:
        break

    if elem[1]==None:
      return None

    if tree[0]=="<":
      if elem[1]<tree[2]:
        return True
      else:
        return False
    if tree[0]==">":
      if elem[1]>tree[2]:
        return True
      else:
        return False

  else:
    if tree[0]=="and":
      return run_tree(tree[1],indices) and run_tree(tree[2],indices)
    else:
      return run_tree(tree[1],indices) or run_tree(tree[2],indices)


def fitness(tree,data,indices,time,rtr):
  index=0
  results=[]
  for i in range(len(data)):
    ind_set=[]
    for elem in indices:
      ind_set.append([elem[0],elem[1][i]])

    response=run_tree(tree,ind_set)

    if response==True and index==0:
      index=1
      buy_time=i
      buy_price=data[i]

    if index==1:
      if i-buy_time>=time or data[i]>=(1+rtr)*buy_price:
        print("exit "+str(buy_price)+" "+str(data[i]))
        index=0
        results.append(round((0.99975*data[i]-1.00025*buy_price)/(1.00025*buy_price)*100,4))

  return results,(sum(results)/len(results)-Risk_free)/(variance(results)**(1/2))

def variance(rez):
    mean_val = sum(rez) / len(rez)
    return sum((x - mean_val) ** 2 for x in rez) / len(rez)


def run_tree(tree,indices):

  if tree[0]=="<" or tree[0]==">":
    for elem in indices:
      if elem[0]==tree[1]:
        break

    if elem[1]==None:
      return None

    if tree[0]=="<":
      if elem[1]<tree[2]:
        return True
      else:
        return False
    if tree[0]==">":
      if elem[1]>tree[2]:
        return True
      else:
        return False

  else:
    if tree[0]=="and":
      return run_tree(tree[1],indices) and run_tree(tree[2],indices)
    else:
      return run_tree(tree[1],indices) or run_tree(tree[2],indices)


def fitness(tree,data,indices,time,rtr):
  index=0
  results=[]
  for i in range(len(data)):
    ind_set=[]
    for elem in indices:
      ind_set.append([elem[0],elem[1][i]])

    response=run_tree(tree,ind_set)

    if response==True and index==0:
      index=1
      buy_time=i
      buy_price=data[i]

    if index==1:
      if i-buy_time>=time or data[i]>=(1+rtr)*buy_price:
        index=0
        results.append(round((0.99975*data[i]-1.00025*buy_price)/(1.00025*buy_price)*100,4))
  try:
    Sharpe=(sum(results)/len(results)-Risk_free)/(variance(results)**(1/2))
  except:
    Sharpe=-100
  return Sharpe, results

def variance(rez):
    mean_val = sum(rez) / len(rez)
    return sum((x - mean_val) ** 2 for x in rez) / len(rez)

def read_data():
  with open('s&p.csv', 'r') as file:
    data = [float(row[0]) for row in csv.reader(file) if row[0].replace('.', '', 1).isdigit()]
  return data

def GA_loop(thresh,rtr,time):
  print("Run with thresh="+str(thresh)+", return target="+str(rtr)+", time interval="+str(time)+".")
  data=read_data()[int(0.65*len(read_data())):]
  testdata=read_data()[int(0.65*len(read_data())):]
  indic_list=[]
  c1=0
  c2=0

  for mix in indicators:
    func_name, arg = mix.split('/')
    arg = int(arg) if arg != 'NA' else None
    if func_name!="TMV":
      indic_list.append([mix,normalise(globals()[func_name](dissect2(data,thresh),data,thresh,arg))])
    else:
      indic_list.append([mix,normalise(globals()[func_name](dissect2(data,thresh),data,thresh,arg)[0])])

  prev_pop=[]
  for tree in range(pop_size):
    prev_pop.append(generate_tree(0))


  for i in range(gens):

    print("Generation "+str(i+1)+":")
    res=[]
    next_gen=[]


    for j in range(len(prev_pop)):
      res.append([fitness(prev_pop[j],data, indic_list,rtr,time)[0],prev_pop[j]])

    res.sort(reverse=True,key=lambda x:x[0])

    if res[0][0]==res[1][0]==res[2][0]==res[3][0]==res[4][0]:
      c1=1
    if i>=gens/2:
      c0=1
    
    print("Shrape ratio top 5: "+str(round(res[0][0],2))+" // "+str(round(res[1][0],2))+" // "+str(round(res[2][0],2))+" // "+str(round(res[3][0],2))+" // "+str(round(res[4][0],2)))
    print("\n")

    if c1==1 and c2==1:
      print("Break conditions met. Algorithm haulted.")

    #Elitism
    next_gen.append(res[0][1])

    while len(next_gen)<=len(res):
      rand=round(random.uniform(0,1), 3)

      if rand<=0.05:

        t1=res[random.randint(0,len(res))-1]
        t2=res[random.randint(0,len(res))-1]

        if t1[0]<t2[0]:
          next_gen.append(mutate(t2[1]))

        if t1[0]>t2[0]:
          next_gen.append(mutate(t1[1]))

        if t1[0]==t2[0]:
          next_gen.append(mutate(random.choice([t1[1],t2[1]])))

      else:

        p11=res[random.randint(0,len(res))-1]
        p12=res[random.randint(0,len(res))-1]
        p21=res[random.randint(0,len(res))-1]
        p22=res[random.randint(0,len(res))-1]

        if p11[0]>p12[0]:
          p1=p11[1]
        if p11[0]<p12[0]:
          p1=p11[1]
        if p11[0]==p12[0]:
          p1=random.choice([p11[1],p12[1]])

        if p21[0]>p22[0]:
          p2=p21[1]
        if p21[0]<p22[0]:
          p2=p21[1]
        if p21[0]==p22[0]:
          p2=random.choice([p21[1],p22[1]])

        ch1,ch2=crossover(p1,p2)

        next_gen.append(p1)
        next_gen.append(p2)
        next_gen.append(c1)
        next_gen.append(c2)
    
    while len(next_gen)>pop_size:
      next_gen.pop(-1)
    
    if res==next_gen:
      print("smth not ok")
    
    prev_gen=next_gen
    #print(prev_gen)

  if c1==1 and c2==1:
    return res[0:5], testdata

  else:
    res=[]
    for j in range(len(prev_pop)):
      res.append([fitness(prev_pop[j],data, indic_list,rtr,time)[0],prev_pop[j]])
    res.sort(reverse=True,key=lambda x:x[0])

    return [res[0:5],testdata]
    

Risk_free=4.2
DC_thresh=[0.001,0.002,0.005,0.01,0.02]
ndays=[1,5,15]
r_target=[1,5,10,20]
sols=[]
for thresh in DC_thresh:
  for days in ndays:
    for rt in r_target:
      sols.append([[thresh,days,rt],GA_loop(thresh,rt,days)])
      print("\n")
