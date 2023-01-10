from hog import *
from Algorithms import *
from FL_baseline import *

def recording_results(Loss_metric, Port_metric, results_tem):
    results_train, results_test, results_time, results_risk,results_cum,results_sharpe = results_tem
    train_tem, test_tem, time_tem = Loss_metric
    results_train.append(train_tem)
    results_test.append(test_tem)
    results_time.append(time_tem)
    risk_list, cum_list, sharpe_list = Port_metric
    results_risk.append(risk_list)
    results_cum.append(cum_list)
    results_sharpe.append(sharpe_list)
    results_tem = [results_train, results_test, results_time, results_risk,results_cum,results_sharpe]
    return results_tem

def Convergence_reduce_rate():
    results_list = []
    X_list = [0.2, 0.4, 0.6, 0.8]
    folder = 'ex14-' + str(flag)
    X_name = 'Number of Rounds'
    plot_flag = 'line'
    show_param = [iters, Train, show, adjust, plot_flag, fn_name, X_list, folder, X_name]
    if Train:
        results_train = []
        results_test = []
        results_time = []
        results_risk = []
        results_cum = []
        results_sharpe = []
        results_tem = [results_train, results_test, results_time, results_risk,results_cum,results_sharpe]
        Dtr, Dte = Dtr_r, Dte_r
        Label_tr = Label_tr_r
        Label_te = Label_te_r
        eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
        train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
        if alg_flag[0] == 1 :
            Loss_metric, Port_metric = my_FedAvg(train_settings, flag)
            results_tem = recording_results(Loss_metric, Port_metric, results_tem)
        if alg_flag[1] == 1 :
            Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
            results_tem = recording_results(Loss_metric, Port_metric, results_tem)
        if alg_flag[2] == 1 :
            Loss_metric, Port_metric = FedProx(train_settings, flag)
            results_tem = recording_results(Loss_metric, Port_metric, results_tem)
        if alg_flag[3] == 1 :
            Loss_metric, Port_metric = SCAFFOLD(train_settings, flag)
            results_tem = recording_results(Loss_metric, Port_metric, results_tem)
        if alg_flag[4] == 1 :
            Dtr, Dte = Dtr_h, Dte_h
            eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
            results_tem = recording_results(Loss_metric, Port_metric, results_tem)
        Dtr, Dte = Dtr_hw, Dte_hw
        eval_param = [n_stocks, Dte, Label_te,  price_te, return_te]
        train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
        if alg_flag[5] == 1:
            Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
            results_tem = recording_results(Loss_metric, Port_metric, results_tem)
        if alg_flag[6] == 1:
            for reduce_rate in X_list:
                Loss_metric, Port_metric = my_TD_naive_FSVRG(train_settings, flag, reduce_rate)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
        results_train, results_test, results_time, results_risk,results_cum,results_sharpe = results_tem
        results_loss = [results_train, results_test, results_time]
        results_port = [results_risk, results_cum, results_sharpe]
        results_list = [results_loss, results_port]

def Different_reduce_rate():
    results_list = []
    X_list = [0.2, 0.4, 0.6, 0.8]
    folder = 'ex13-' + str(flag)
    X_name = 'Feature Reducing Rate'
    show_param = [iters, Train, show, adjust, plot_flag, fn_name, X_list, folder, X_name]
    if Train:
        #
        results_train = []
        results_test = []
        results_time = []
        results_risk = []
        results_cum = []
        results_sharpe = []
        results_tem = [results_train, results_test, results_time, results_risk,results_cum,results_sharpe]
        for reduce_rate in X_list:
            Dtr, Dte = Dtr_r, Dte_r
            Label_tr = Label_tr_r
            Label_te = Label_te_r
            eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            if alg_flag[0] == 1 :
                Loss_metric, Port_metric = my_FedAvg(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[1] == 1 :
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[2] == 1 :
                Dtr, Dte = Dtr_h, Dte_h
                eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
                train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            Dtr, Dte = Dtr_hw, Dte_hw
            eval_param = [n_stocks, Dte, Label_te,  price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            if alg_flag[3] == 1:
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[4] == 1:
                Loss_metric, Port_metric = my_TD_naive_FSVRG(train_settings, flag, reduce_rate)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
        results_train, results_test, results_time, results_risk,results_cum,results_sharpe = results_tem
        results_loss = [results_train, results_test, results_time]
        results_port = [results_risk, results_cum, results_sharpe]
        results_list = [results_loss, results_port]

def Number_Individual_machines():
    results_list = []
    X_list = [20, 40, 60, 80]
    folder = 'ex12-' + str(flag)
    X_name = 'Number of AEAs'
    show_param = [iters, Train, show, adjust, plot_flag, fn_name, X_list, folder, X_name]
    if Train:
        #
        results_train = []
        results_test = []
        results_time = []
        results_risk = []
        results_cum = []
        results_sharpe = []
        results_tem = [results_train, results_test, results_time, results_risk,results_cum,results_sharpe]
        for num_nodes in X_list:
            #
            Dtr, Dte = Dtr_r, Dte_r
            Label_tr = Label_tr_r
            Label_te = Label_te_r
            eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            if alg_flag[0] == 1 :
                Loss_metric, Port_metric = my_FedAvg(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[1] == 1 :
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[2] == 1 :
                Dtr, Dte = Dtr_h, Dte_h
                eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
                train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            Dtr, Dte = Dtr_hw, Dte_hw
            eval_param = [n_stocks, Dte, Label_te,  price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            if alg_flag[3] == 1:
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[4] == 1:
                Loss_metric, Port_metric = my_TD_naive_FSVRG(train_settings, flag, reduce_rate)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
        results_train, results_test, results_time, results_risk,results_cum,results_sharpe = results_tem
        results_loss = [results_train, results_test, results_time]
        results_port = [results_risk, results_cum, results_sharpe]
        results_list = [results_loss, results_port]

def HoG_feature_size():
    results_list = []
    X_list = [3,5,7,9]
    folder = 'ex11-' + str(flag)
    X_name = 'HoG Block Feature Size'
    show_param = [iters, Train, show, adjust, plot_flag, fn_name, X_list, folder, X_name]
    if Train:
        #
        results_train = []
        results_test = []
        results_time = []
        results_risk = []
        results_cum = []
        results_sharpe = []
        results_tem = [results_train, results_test, results_time, results_risk,results_cum,results_sharpe]
        for B in X_list:
            hog_params = [d, B, n_stocks]
            Dtr_h, Dte_h, Dtr_hw, Dte_hw = feature_design(hog_params, sample_pairs, W_list, delta, n_stocks)
            Dtr, Dte = Dtr_r, Dte_r
            Label_tr = Label_tr_r
            Label_te = Label_te_r
            eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            if alg_flag[0] == 1 :
                Loss_metric, Port_metric = my_FedAvg(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[1] == 1 :
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[2] == 1 :
                Dtr, Dte = Dtr_h, Dte_h
                eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
                train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            Dtr, Dte = Dtr_hw, Dte_hw
            eval_param = [n_stocks, Dte, Label_te,  price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            if alg_flag[3] == 1:
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[4] == 1:
                Loss_metric, Port_metric = my_TD_naive_FSVRG(train_settings, flag, reduce_rate)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
        results_train, results_test, results_time, results_risk,results_cum,results_sharpe = results_tem
        results_loss = [results_train, results_test, results_time]
        results_port = [results_risk, results_cum, results_sharpe]
        results_list = [results_loss, results_port]

def HoG_block_size():
    results_list = []
    X_list = [3,4,5]
    folder = 'ex10-' + str(flag)
    X_name = 'HoG Block Size'
    show_param = [iters, Train, show, adjust, plot_flag, fn_name, X_list, folder, X_name]
    if Train:
        results_train = []
        results_test = []
        results_time = []
        results_risk = []
        results_cum = []
        results_sharpe = []
        results_tem = [results_train, results_test, results_time, results_risk,results_cum,results_sharpe]
        for d in X_list:
            hog_params = [d, B, n_stocks]
            Dtr_h, Dte_h, Dtr_hw, Dte_hw = feature_design(hog_params, sample_pairs, W_list, delta, n_stocks)
            Dtr, Dte = Dtr_r, Dte_r
            Label_tr = Label_tr_r
            Label_te = Label_te_r
            eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            if alg_flag[0] == 1 :
                Loss_metric, Port_metric = my_FedAvg(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[1] == 1 :
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[2] == 1 :
                Dtr, Dte = Dtr_h, Dte_h
                eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
                train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            Dtr, Dte = Dtr_hw, Dte_hw
            eval_param = [n_stocks, Dte, Label_te,  price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            if alg_flag[3] == 1:
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[4] == 1:
                Loss_metric, Port_metric = my_TD_naive_FSVRG(train_settings, flag, reduce_rate)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
        results_train, results_test, results_time, results_risk,results_cum,results_sharpe = results_tem
        results_loss = [results_train, results_test, results_time]
        results_port = [results_risk, results_cum, results_sharpe]
        results_list = [results_loss, results_port]

def Lookback_length():
    results_list = []
    X_list = [10, 20]
    folder = 'ex9-' + str(flag)
    X_name = 'Training Feature Size'
    show_param = [iters, Train, show, adjust, plot_flag, fn_name, X_list, folder, X_name]
    if Train:
        results_train = []
        results_test = []
        results_time = []
        results_risk = []
        results_cum = []
        results_sharpe = []
        results_tem = [results_train, results_test, results_time, results_risk,results_cum,results_sharpe]
        for lookback in X_list:
            feature_params = [lookback, gap, horizon, block, max_weight, gamma]
            Dtr_r, Label_tr_r, return_tr = data_structure_r(train_r, feature_params)
            Dte_r, Label_te_r, return_te = data_structure_r(test_r, feature_params)
            sample_pairs = [Dtr_r, Dte_r]
            price_te = data_structure_p(test_p, feature_params)
            Dtr_h, Dte_h, Dtr_hw, Dte_hw = feature_design(hog_params, sample_pairs, W_list, delta, n_stocks)
            Dtr, Dte = Dtr_r, Dte_r
            Label_tr = Label_tr_r
            Label_te = Label_te_r
            eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            if alg_flag[0] == 1 :
                Loss_metric, Port_metric = my_FedAvg(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[1] == 1 :
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[2] == 1 :
                Dtr, Dte = Dtr_h, Dte_h
                eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
                train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            Dtr, Dte = Dtr_hw, Dte_hw
            eval_param = [n_stocks, Dte, Label_te,  price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            if alg_flag[3] == 1:
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[4] == 1:
                Loss_metric, Port_metric = my_TD_naive_FSVRG(train_settings, flag, reduce_rate)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
        results_train, results_test, results_time, results_risk,results_cum,results_sharpe = results_tem
        results_loss = [results_train, results_test, results_time]
        results_port = [results_risk, results_cum, results_sharpe]
        results_list = [results_loss, results_port]

def Horizon_length():
    results_list = []
    X_list = [10, 20, 30, 40]
    folder = 'ex8-' + str(flag)
    X_name = 'Different Predicted Horizon'
    show_param = [iters, Train, show, adjust, plot_flag, fn_name, X_list, folder, X_name]
    if Train:
        results_train = []
        results_test = []
        results_time = []
        results_risk = []
        results_cum = []
        results_sharpe = []
        results_tem = [results_train, results_test, results_time, results_risk,results_cum,results_sharpe]
        for horizon in X_list:
            feature_params = [lookback, gap, horizon, block, max_weight, gamma]
            Dtr_r, Label_tr_r, return_tr = data_structure_r(train_r, feature_params)
            Dte_r, Label_te_r, return_te = data_structure_r(test_r, feature_params)
            sample_pairs = [Dtr_r, Dte_r]
            price_te = data_structure_p(test_p, feature_params)
            Dtr_h, Dte_h, Dtr_hw, Dte_hw = feature_design(hog_params, sample_pairs, W_list, delta, n_stocks)
            Dtr, Dte = Dtr_r, Dte_r
            Label_tr = Label_tr_r
            Label_te = Label_te_r
            eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            if alg_flag[0] == 1 :
                Loss_metric, Port_metric = my_FedAvg(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[1] == 1 :
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[2] == 1 :
                Dtr, Dte = Dtr_h, Dte_h
                eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
                train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            Dtr, Dte = Dtr_hw, Dte_hw
            eval_param = [n_stocks, Dte, Label_te,  price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            if alg_flag[3] == 1:
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[4] == 1:
                Loss_metric, Port_metric = my_TD_naive_FSVRG(train_settings, flag, reduce_rate)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
        results_train, results_test, results_time, results_risk,results_cum,results_sharpe = results_tem
        results_loss = [results_train, results_test, results_time]
        results_port = [results_risk, results_cum, results_sharpe]
        results_list = [results_loss, results_port]

def Wavelet_type():
    results_list = []
    X_list = ['Haar', 'Daubechies', 'Lemarie', 'Symlets']
    folder = 'ex7-' + str(flag)
    X_name = 'Different Filter Type'
    show_param = [iters, Train, show, adjust, plot_flag, fn_name, X_list, folder, X_name]
    if Train:
        results_train = []
        results_test = []
        results_time = []
        results_risk = []
        results_cum = []
        results_sharpe = []
        results_tem = [results_train,results_test,results_time,results_risk,results_cum,results_sharpe]
        for f in X_list:
            Wm = io.loadmat(path + 'test/filters/' + f + '/Wm.mat', struct_as_record=False)['Wm']
            Wn = io.loadmat(path + 'test/filters/' + f + '/Wn.mat', struct_as_record=False)['Wn']
            Wmi = io.loadmat(path + 'test/filters/' + f + '/Wmi.mat', struct_as_record=False)['Wmi']
            Wni = io.loadmat(path + 'test/filters/' + f + '/Wni.mat', struct_as_record=False)['Wni']
            W_list = [Wm, Wn, Wmi, Wni]
            Dtr_h, Dte_h, Dtr_hw, Dte_hw = feature_design(hog_params, sample_pairs, W_list, delta, n_stocks)
            Dtr, Dte = Dtr_r, Dte_r
            Label_tr = Label_tr_r
            Label_te = Label_te_r
            eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            if alg_flag[0] == 1 :
                Loss_metric, Port_metric = my_FedAvg(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[1] == 1 :
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[2] == 1 :
                Dtr, Dte = Dtr_h, Dte_h
                eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
                train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            Dtr, Dte = Dtr_hw, Dte_hw
            eval_param = [n_stocks, Dte, Label_te,  price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            if alg_flag[3] == 1:
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[4] == 1:
                Loss_metric, Port_metric = my_TD_naive_FSVRG(train_settings, flag, reduce_rate)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
        results_train, results_test, results_time, results_risk,results_cum,results_sharpe = results_tem
        results_loss = [results_train, results_test, results_time]
        results_port = [results_risk, results_cum, results_sharpe]
        results_list = [results_loss, results_port]

def Wavelet_filter_length():
    results_list = []
    X_list = [2, 4, 6, 8]
    folder = 'ex6-' + str(flag)
    X_name = 'Daubechies Filter Length'
    show_param = [iters, Train, show, adjust, plot_flag, fn_name, X_list, folder, X_name]
    if Train:
        #
        results_train = []
        results_test = []
        results_time = []
        results_risk = []
        results_cum = []
        results_sharpe = []
        results_tem = [results_train, results_test, results_time, results_risk,results_cum,results_sharpe]
        for N in X_list:
            Wm = io.loadmat(path + 'test/daub/' + str(N) + '/Wm.mat', struct_as_record=False)['Wm']
            Wn = io.loadmat(path + 'test/daub/' + str(N) + '/Wn.mat', struct_as_record=False)['Wn']
            Wmi = io.loadmat(path + 'test/daub/' + str(N) + '/Wmi.mat', struct_as_record=False)['Wmi']
            Wni = io.loadmat(path + 'test/daub/' + str(N) + '/Wni.mat', struct_as_record=False)['Wni']
            W_list = [Wm, Wn, Wmi, Wni]
            Dtr_h, Dte_h, Dtr_hw, Dte_hw = feature_design(hog_params, sample_pairs, W_list, delta, n_stocks)
            Dtr, Dte = Dtr_r, Dte_r
            Label_tr = Label_tr_r
            Label_te = Label_te_r
            eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            if alg_flag[0] == 1 :
                Loss_metric, Port_metric = my_FedAvg(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[1] == 1 :
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[2] == 1 :
                Dtr, Dte = Dtr_h, Dte_h
                eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
                train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            Dtr, Dte = Dtr_hw, Dte_hw
            eval_param = [n_stocks, Dte, Label_te,  price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            if alg_flag[3] == 1:
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[4] == 1:
                Loss_metric, Port_metric = my_TD_naive_FSVRG(train_settings, flag, reduce_rate)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
        results_train, results_test, results_time, results_risk,results_cum,results_sharpe = results_tem
        results_loss = [results_train, results_test, results_time]
        results_port = [results_risk, results_cum, results_sharpe]
        results_list = [results_loss, results_port]

def Different_delta():
    results_list = []
    X_list = [0.0001, 0.001, 0.01, 0.1]
    folder = 'ex5-' + str(flag)
    X_name = 'Standard Deviation of Estimated Noise'
    show_param = [iters, Train, show, adjust, plot_flag, fn_name, X_list, folder, X_name]
    if Train:
        results_train = []
        results_test = []
        results_time = []
        results_risk = []
        results_cum = []
        results_sharpe = []
        results_tem = [results_train,results_test,results_time,results_risk,results_cum,results_sharpe]
        for delta in X_list:
            Dtr_h, Dte_h, Dtr_hw, Dte_hw = feature_design(hog_params, sample_pairs, W_list, delta, n_stocks)
            Dtr, Dte = Dtr_r, Dte_r
            Label_tr = Label_tr_r
            Label_te = Label_te_r
            eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            if alg_flag[0] == 1 :
                Loss_metric, Port_metric = my_FedAvg(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[1] == 1 :
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[2] == 1 :
                Dtr, Dte = Dtr_h, Dte_h
                eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
                train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            Dtr, Dte = Dtr_hw, Dte_hw
            eval_param = [n_stocks, Dte, Label_te,  price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            if alg_flag[3] == 1:
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[4] == 1:
                Loss_metric, Port_metric = my_TD_naive_FSVRG(train_settings, flag, reduce_rate)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
        results_train, results_test, results_time, results_risk,results_cum,results_sharpe = results_tem
        results_loss = [results_train, results_test, results_time]
        results_port = [results_risk, results_cum, results_sharpe]
        results_list = [results_loss, results_port]

def Wavelet_layer():
    results_list = []
    X_list = [1, 2, 3, 4]
    folder = 'ex4-' + str(flag)
    X_name = 'Decomposition Layer'
    show_param = [iters, Train, show, adjust, plot_flag, fn_name, X_list, folder, X_name]
    if Train:
        results_train = []
        results_test = []
        results_time = []
        results_risk = []
        results_cum = []
        results_sharpe = []
        results_tem = [results_train, results_test, results_time, results_risk,results_cum,results_sharpe]
        for layer in X_list:
            Wm = io.loadmat(path + 'test/layer/' + str(layer) + '/Wm.mat', struct_as_record=False)['Wm']
            Wn = io.loadmat(path + 'test/layer/' + str(layer) + '/Wn.mat', struct_as_record=False)['Wn']
            Wmi = io.loadmat(path + 'test/layer/'+str(layer) + '/Wmi.mat', struct_as_record=False)['Wmi']
            Wni = io.loadmat(path + 'test/layer/'+str(layer) + '/Wni.mat', struct_as_record=False)['Wni']
            W_list = [Wm, Wn, Wmi, Wni]
            delta = 0.1
            Dtr_h, Dte_h, Dtr_hw, Dte_hw = feature_design(hog_params, sample_pairs, W_list, delta, n_stocks)
            Dtr, Dte = Dtr_r, Dte_r
            Label_tr = Label_tr_r
            Label_te = Label_te_r
            eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            if alg_flag[0] == 1 :
                Loss_metric, Port_metric = my_FedAvg(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[1] == 1 :
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[2] == 1 :
                Dtr, Dte = Dtr_h, Dte_h
                eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
                train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            Dtr, Dte = Dtr_hw, Dte_hw
            eval_param = [n_stocks, Dte, Label_te,  price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            if alg_flag[3] == 1:
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[4] == 1:
                Loss_metric, Port_metric = my_TD_naive_FSVRG(train_settings, flag, reduce_rate)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
        results_train, results_test, results_time, results_risk,results_cum,results_sharpe = results_tem
        results_loss = [results_train, results_test, results_time]
        results_port = [results_risk, results_cum, results_sharpe]
        results_list = [results_loss, results_port]

def Learning_rate():
    results_list = []
    h_list = [0.1, 0.01, 0.001]
    folder = 'ex3-' + str(flag)
    X_name = 'Learning Rate'
    show_param = [iters, Train, show, adjust, plot_flag, fn_name, h_list, folder, X_name]
    if Train:
        results_train = []
        results_test = []
        results_time = []
        results_risk = []
        results_cum = []
        results_sharpe = []
        results_tem = [results_train, results_test, results_time, results_risk,results_cum,results_sharpe]
        for h in h_list:
            Dtr, Dte = Dtr_r, Dte_r
            Label_tr = Label_tr_r
            Label_te = Label_te_r
            eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            if alg_flag[0] == 1 :
                Loss_metric, Port_metric = my_FedAvg(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[1] == 1 :
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[2] == 1 :
                Dtr, Dte = Dtr_h, Dte_h
                eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
                train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            Dtr, Dte = Dtr_hw, Dte_hw
            eval_param = [n_stocks, Dte, Label_te,  price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            if alg_flag[3] == 1:
                Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
            if alg_flag[4] == 1:
                Loss_metric, Port_metric = my_TD_naive_FSVRG(train_settings, flag, reduce_rate)
                results_tem = recording_results(Loss_metric, Port_metric, results_tem)
        results_train, results_test, results_time, results_risk,results_cum,results_sharpe = results_tem
        results_loss = [results_train, results_test, results_time]
        results_port = [results_risk, results_cum, results_sharpe]
        results_list = [results_loss, results_port]

def Convergence():
    results_list = []
    h_list = [0.1, 0.01, 0.001]
    folder = 'ex2-' + str(flag)
    X_name = 'Number of Rounds'
    plot_flag = 'line'
    show_param = [iters, Train, show, adjust, plot_flag, fn_name, h_list, folder, X_name]
    if Train:
        results_train = []
        results_test = []
        results_time = []
        results_risk = []
        results_cum = []
        results_sharpe = []
        results_tem = [results_train, results_test, results_time, results_risk,results_cum,results_sharpe]
        Dtr, Dte = Dtr_r, Dte_r
        Label_tr = Label_tr_r
        Label_te = Label_te_r
        eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
        train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
        if alg_flag[0] == 1 :
            Loss_metric, Port_metric = my_FedAvg(train_settings, flag)
            results_tem = recording_results(Loss_metric, Port_metric, results_tem)
        if alg_flag[1] == 1 :
            Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
            results_tem = recording_results(Loss_metric, Port_metric, results_tem)
        if alg_flag[2] == 1 :
            Dtr, Dte = Dtr_h, Dte_h
            eval_param = [n_stocks, Dte, Label_te, price_te, return_te]
            train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
            Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
            results_tem = recording_results(Loss_metric, Port_metric, results_tem)
        Dtr, Dte = Dtr_hw, Dte_hw
        eval_param = [n_stocks, Dte, Label_te,  price_te, return_te]
        train_settings = [Dtr, num_nodes, h, iters, g_function, Label_tr, n_stocks, eval_param]
        if alg_flag[3] == 1:
            Loss_metric, Port_metric = my_naive_FSVRG(train_settings, flag)
            results_tem = recording_results(Loss_metric, Port_metric, results_tem)
        if alg_flag[4] == 1:
            Loss_metric, Port_metric = my_TD_naive_FSVRG(train_settings, flag, reduce_rate)
            results_tem = recording_results(Loss_metric, Port_metric, results_tem)
        results_train, results_test, results_time, results_risk,results_cum,results_sharpe = results_tem
        results_loss = [results_train, results_test, results_time]
        results_port = [results_risk, results_cum, results_sharpe]
        results_list = [results_loss, results_port]
