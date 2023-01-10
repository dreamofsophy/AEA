from g_fun import *
from load_data import *
from Algorithms import *
import time

def SCAFFOLD(train_settings, flag):
    Dtr, num_nodes, h, iters, gname, Label, n, eval_param = train_settings
    local_datasets, local_label = Local_datasets(num_nodes, flag, Dtr, Label)
    #
    N1 = Dtr.shape[0]
    P = Dtr.shape[1]
    ind = np.random.permutation(P)
    Dr = Dtr[:, ind]
    wt = np.zeros(((N1 + 1)*n, 1))
    z = int(P/num_nodes)
    m = z
    Train_e = []
    Test_e = []
    Time_tr = []
    risk_list =[]
    cum_list = []
    sharpe_list = []
    #
    ci_list = []
    for k in range(num_nodes):
        ci_list.append(np.zeros(((N1 + 1)*n, 1)))
    c = np.zeros(((N1 + 1)*n, 1))
    #
    for s in range(iters):
        gt = gname(wt, Dtr, Label, n)
        local_x = []
        local_time = []
        for k in range(num_nodes):
            t0 = time.time()   
            wk = copy.deepcopy(wt)
            Pk = local_datasets[k]
            label_k = local_label[k]
            gik_sum  = 0
            for t in range(m):
                ind2 = np.random.permutation(Pk.shape[1])
                i = ind2[0]
                vi = Pk[:, i].reshape(-1, 1)
                y = [label_k[i]]
                gik = gname(wk, vi, y, n)
                gik_sum += gik
                #git = gname(wt, vi, y, n)
                #wk = wk - h*(gik - git + gt)
                wk = wk - h*(gik - ci_list[k] + c)
            ci_list[k] = gik_sum/m
            local_x.append(wk - wt)
            local_time.append(time.time() - t0)
        c = sum(ci_list)/num_nodes
        local_t = max(local_time)
        wt = wt + sum(local_x)/num_nodes 
        etr, ete, Port_1 = evaluation_rounds(wt, Dtr, Label, eval_param)
        Train_e.append(etr)
        Test_e.append(ete)
        Time_tr.append(local_t)
        risk, cum_return, sharpe_ratio = Port_1
        risk_list.append(risk)
        cum_list.append(cum_return)
        sharpe_list.append(sharpe_ratio)
    Loss_metric = [Train_e, Test_e, Time_tr]
    Portfolio_metric = [risk_list, cum_list, sharpe_list]
    return Loss_metric, Portfolio_metric

def FedProx(train_settings, flag):
    Dtr, num_nodes, h, iters, gname, Label, n, eval_param = train_settings
    local_datasets, local_label = Local_datasets(num_nodes, flag, Dtr, Label)
    #
    N1 = Dtr.shape[0]
    P = Dtr.shape[1]
    ind = np.random.permutation(P)
    Dr = Dtr[:, ind]
    wt = np.zeros(((N1 + 1)*n, 1))
    z = int(P/num_nodes)
    m = z
    Train_e = []
    Test_e = []
    Time_tr = []
    risk_list =[]
    cum_list = []
    sharpe_list = []
    #Proximal : mu
    mu = 0.01
    for s in range(iters):
        local_x = []
        local_time = []
        for k in range(num_nodes):
            t0 = time.time()   
            wk = copy.deepcopy(wt)
            Pk = local_datasets[k]
            label_k = local_label[k]
            for t in range(m):
                ind2 = np.random.permutation(Pk.shape[1])
                i = ind2[0]
                vi = Pk[:, i].reshape(-1, 1)
                y = [label_k[i]]
                gi = gname(wk, vi, y, n)
                #Prox
                gik = gi + mu*(wk - wt)	
                wk = wk - h*gik
            local_x.append(wk - wt)
            local_time.append(time.time() - t0)
        local_t = max(local_time)
        wt = wt + sum(local_x)/num_nodes 
        etr, ete, Port_1 = evaluation_rounds(wt, Dtr, Label, eval_param)
        Train_e.append(etr)
        Test_e.append(ete)
        Time_tr.append(local_t)
        risk, cum_return, sharpe_ratio = Port_1
        risk_list.append(risk)
        cum_list.append(cum_return)
        sharpe_list.append(sharpe_ratio)
    Loss_metric = [Train_e, Test_e, Time_tr]
    Portfolio_metric = [risk_list, cum_list, sharpe_list]
    return Loss_metric, Portfolio_metric



