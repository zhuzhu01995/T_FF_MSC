% test for Multiview Subspace algorithm
clear
close all

rng('default') % For reproducibility


addpath(genpath('utils/'))
addpath(genpath('algs/'));

data_path = 'data/';

num_runs = 20;


% Algorithm Settings

% Proposed
% flag_LogDet_MSC = 1;
flag_T_FF_MSC = 1;

Data_list = {
    'yale.mat',
    'yaleB.mat',
    'ORL.mat',
    'COIL20MV.mat',
};
Data_name = {
    'Yale',
    'Extended YaleB',
    'ORL',
    'COIL-20',
};
Data_views = {
    3,
    3,
    3,
    3,
};

test_list = 1:1;

for t = test_list
    clear X
    %% Loading data
    fprintf('Testing %s...\n', Data_name{t}) 
    load(fullfile(data_path, Data_list{t}));
    
    for k=1:Data_views{t}
        eval(sprintf('X{%d} = double(X%d);', k, k));
    end

    cls_num = length(unique(gt));
    K = length(X);

    %% Records
    alg_name = {}; 
    alg_cpu = {};
    alg_C = {};     % clustering results
    alg_S = {};     % affinity matrices
    alg_out = {};

    alg_NMI = {};
    alg_ACC = {};
    alg_AR = {};
    alg_fscore = {};   
    alg_precision = {};
    alg_recall = {};

    alg_cnt = 1;
    
    %% Algs Running
    if flag_T_FF_MSC
        Y = X;
        for iv=1:K
            [Y{iv}]=NormalizeData(X{iv});
        end
       
        opts = [];
        opts.Frac_alpha = 5000;
        opts.maxIter = 200;   
        opts.epsilon = 1e-4; 
        opts.flag_debug = 0;
        opts.mu = 1e-5;
        opts.rho = 1e-5;
        opts.eta = 2;
        opts.max_mu = 1e10; 
        opts.max_rho = 1e10;    

        best_params_list = {       
            [0.221],   
            [0.001],
            [0.219], 
            [0.001],
        };
        param = best_params_list{t};
        opts.lambda = param; 


        for kk = 1:num_runs
            time_start = tic;
            [C_T_FF_MSC, S_T_FF_MSC, Out_T_FF_MSC] = alg_T_FF_MSC(Y, cls_num, gt, opts);
            alg_name{alg_cnt} = 'T_FF_MSC';

            alg_cpu{alg_cnt}(kk) = toc(time_start);
            alg_C{alg_cnt}{kk} = C_T_FF_MSC;
            alg_S{alg_cnt}{kk} = S_T_FF_MSC;
            alg_Out{alg_cnt}{kk} = Out_T_FF_MSC;
            alg_NMI{alg_cnt}(kk) = Out_T_FF_MSC.NMI;
            alg_AR{alg_cnt}(kk) = Out_T_FF_MSC.AR;
            alg_ACC{alg_cnt}(kk) = Out_T_FF_MSC.ACC;
            alg_recall{alg_cnt}(kk) = Out_T_FF_MSC.recall;
            alg_precision{alg_cnt}(kk) = Out_T_FF_MSC.precision;
            alg_fscore{alg_cnt}(kk) = Out_T_FF_MSC.fscore;  

        end
        alg_cnt = alg_cnt + 1;
    end

    %% Results report
    flag_report = 1;
    if flag_report
        fprintf('%-6s  %-12s  %8s    %8s    %8s    %8s    %8s   %8s   %8s\n',...\
            'Stats', 'Algs', 'CPU', 'NMI', 'AR', 'ACC', 'Recall', 'Pre', 'F-Score');
        for j = 1:alg_cnt-1
            fprintf('%-6s  %-12s   %8.6f   %8.3f   %8.3f   %8.3f   %8.3f   %8.3f   %8.3f\n',...\
                'Mean', alg_name{j},mean(alg_cpu{j}),mean(alg_NMI{j}),mean(alg_AR{j}),...\
                mean(alg_ACC{j}),mean(alg_recall{j}),mean(alg_precision{j}),mean(alg_fscore{j}));
        end
        for j = 1:alg_cnt-1
            fprintf('%-6s   %-12s   %8.6f   %8.3f   %8.3f   %8.3f   %8.3f   %8.3f   %8.3f\n',...\
                'Std', alg_name{j},std(alg_cpu{j}),std(alg_NMI{j}),std(alg_AR{j}),...\
                std(alg_ACC{j}),std(alg_recall{j}),std(alg_precision{j}),std(alg_fscore{j}));
        end
    end
    
end



