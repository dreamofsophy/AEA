from g_fun import *
from load_data import *
import time

def model_feature(model_list):
    X = model_list[0].reshape(-1, 1)
    K = len(model_list)
    for i in range(1, K):
        X =  np.c_[X, model_list[i]]
    X_mean = np.mean(X, axis=1).reshape(-1, 1)
    Xh = X - X_mean
    matrix = Xh.dot(Xh.T)/K
    V, U = np.linalg.eigh(matrix)
    #r = int(U.shape[1]*0.5)
    r = 1
    U = U[:, :r]
    p = np.sum(U.T.dot(Xh), axis=1).reshape(-1,1)
    w = U.dot(p/K) + X_mean
    return w
#################################
def TD_model_feature(w, D, reduce_rate):
    TD_w = D @ w
    L = int(w.size * reduce_rate)
    TD_w[-L:] = 0
    return TD_w
#################################

def my_TD_naive_FSVRG(train_settings, flag, reduce_rate):
    Dtr, num_nodes, h, iters, gname, Label, n, eval_param = train_settings
    local_datasets, local_label = Local_datasets(num_nodes, flag, Dtr, Label)
    #
    N1 = Dtr.shape[0]
    P = Dtr.shape[1]
    ind = np.random.permutation(P)
    Dr = Dtr[:, ind]
    wt = np.zeros(((N1 + 1)*n, 1))
    D = DCT_1D((N1 + 1)*n)
    #
    z = int(P/num_nodes)
    m = z
    Train_e = []
    Test_e = []
    Time_tr = []
    risk_list =[]
    cum_list = []
    sharpe_list = []
    for s in range(iters):
        gt = gname(wt, Dtr, Label, n)
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
                gik = gname(wk, vi, y, n)
                git = gname(wt, vi, y, n)
                wk = wk - h*(gik - git + gt)
            w_diff_0 = wk - wt
            w_diff_1 = TD_model_feature(w_diff_0, D, reduce_rate)
            local_x.append(w_diff_1)
            local_time.append(time.time() - t0)
        local_t = max(local_time)
        w_c = sum(local_x)/num_nodes
        wt = wt + D.T @ w_c
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

def my_naive_FSVRG(train_settings, flag):
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
    for s in range(iters):
        gt = gname(wt, Dtr, Label, n)
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
                gik = gname(wk, vi, y, n)
                git = gname(wt, vi, y, n)
                wk = wk - h*(gik - git + gt)
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


def my_FedAvg(train_settings, flag):
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
                gik = gname(wk, vi, y, n)
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


def my_encode_naive_FSVRG(train_settings, flag):
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
    for s in range(iters):
        gt = gname(wt, Dtr, Label, n)
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
                gik = gname(wk, vi, y, n)
                git = gname(wt, vi, y, n)
                wk = wk - h*(gik - git + gt)
            local_x.append(wk - wt)
            local_time.append(time.time() - t0)
        local_t = max(local_time)
        wt = wt + model_feature(local_x)
        etr, ete = evaluation_rounds(wt, Dtr, Label, eval_param)
        Train_e.append(etr)
        Test_e.append(ete)
        Time_tr.append(local_t)
    return Train_e, Test_e, Time_tr

def evaluation_rounds(ws_h, Dtr, Label_tr, eval_param):    
    n_stocks, Dte, Label_te, price_te, return_te = eval_param
    y_hat_tr = Prediction(ws_h, Dtr, n_stocks)
    y_hat_te = Prediction(ws_h, Dte, n_stocks)
    RMSE_train = Root_Mean_Squared_Error(y_hat_tr, Label_tr)
    RMSE_test = Root_Mean_Squared_Error(y_hat_te, Label_te)
    risk, cum_return, sharpe_ratio = test_portfolio(y_hat_te, price_te, return_te)
    Portfolio = [risk, cum_return, sharpe_ratio]
    return RMSE_train, RMSE_test, Portfolio

def evaluation(ws_h, n_stocks, Dtr, Dte, Label_tr, Label_te):    
    y_hat_tr = Prediction(ws_h, Dtr, n_stocks)
    y_hat_te = Prediction(ws_h, Dte, n_stocks)
    RMSE_train = Root_Mean_Squared_Error(y_hat_tr, Label_tr)
    RMSE_test = Root_Mean_Squared_Error(y_hat_te, Label_te)
    print("Root Mean Squared Error for Train Dataset: %.4f"%RMSE_train)
    print("Root Mean Squared Error for Test Dataset: %.4f"%RMSE_test)
    #Plot_figure(y_hat_te, y_te)

def Prediction(w, D, n):
    y_hat = []
    N1, P = D.shape
    N1 = N1 + 1
    Xh = np.r_[D, np.ones((1, P))]
    W = w.reshape(N1, n, order='F')
    y_hat = []
    y_pre =  np.zeros((n, P))
    for i in range(n):
        wi = W[:, i]
        for p in range(P):
            xp = Xh[:, p]
            tp = expit(xp.dot(wi))
            t0 = sum(expit(xp.dot(W)))
            y_pre[i][p] = tp/t0    
    for i in range(P):
        y_hat.append(y_pre[:, i])
    return y_hat

def Root_Mean_Squared_Error(y_hat, y):
    num_samples = len(y)
    squared_error = []
    for p in range(num_samples):
        squared_error.append(LA.norm(y_hat[p] - y[p])**2)
    rmse = sqrt(sum(squared_error)/num_samples)
    return rmse
