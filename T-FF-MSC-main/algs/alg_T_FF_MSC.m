function [C, S, Out] = alg_T_FF_MSC(X, cls_num, gt, opts)

%% Note: Multiview Subspace Clustering with Tensor Log-determinant Model
% Input:
%   X:          data features
%   cls_num:    number of clusters
%   gt:         ground truth clusters
%   opts:       optional parameters
%               - maxIter: max iteration
%               - mu:  penalty parameter
%               - rho: penalty parameter
%               - epsilon: stopping tolerance
% Outout:
%   C:          clusetering results
%   S:          affinity matrix
%   Out:        other output information, e.g. metrics, history

%% Parameter settings
N = size(X{1}, 2);
K = length(X); % number of views

% Default
maxIter = 200;   
epsilon = 1e-7;
lambda = 0.2;
mu = 1e-5;
rho = 1e-5;
eta = 2;
max_mu = 1e10; 
max_rho = 1e10;  
flag_debug = 0;

if ~exist('opts', 'var')
    opts = [];
end  
if  isfield(opts, 'maxIter');       maxIter = opts.maxIter;         end
if  isfield(opts, 'epsilon');       epsilon = opts.epsilon;         end
if  isfield(opts, 'lambda');        lambda = opts.lambda;           end
if  isfield(opts, 'mu');            mu = opts.mu;                   end
if  isfield(opts, 'rho');           rho = opts.rho;                 end
if  isfield(opts, 'eta');           eta = opts.eta;                 end
if  isfield(opts, 'max_mu');        max_mu = opts.max_mu;           end
if  isfield(opts, 'max_rho');       max_rho = opts.max_rho;         end
if  isfield(opts, 'flag_debug');    flag_debug = opts.flag_debug;   end
if  isfield(opts, 'Frac_alpha');    Frac_alpha = opts.Frac_alpha;   end
if  isfield(opts, 'alpha');         alpha = opts.alpha;             end



%% Initialize...
for k=1:K
    Z{k} = zeros(N,N); 
    W{k} = zeros(N,N);
    G{k} = zeros(N,N);
    E{k} = zeros(size(X{k},1),N); 
    Y{k} = zeros(size(X{k},1),N); 
end

%% Initialize the history structure
 history = struct();
 history.norm_Z_G = [];  % restore ||Z - G||_∞
 history.norm_Z = [];
 history.objval = [];    
 % If other fields (e.g., iteration time) are needed, add them directly here
 % history.iter_time = [];

iter = 0;
Isconverg = 0;

%% Iterating
while(Isconverg == 0)
    if flag_debug
        fprintf('----processing iter %d--------\n', iter+1);
    end

   
    %------------------- Update Z^k -------------------------------    
    for k=1:K
        tmp = (X{k}-E{k})'*Y{k} + mu*(X{k}-E{k})'*(X{k}-E{k}) - W{k} + rho*G{k};
        Z{k}=(mu*(X{k}-E{k})'*(X{k}-E{k})+rho*eye(N,N))\tmp;
    end
    
   
    %------------------- Updat E^k  -------------------
    % Prepare inputs for the proximal gradient method
    C_cell = cell(1, K);
    D_cell = cell(1, K);
    for k = 1:K
        C_cell{k} = Z{k} - eye(N, N);
        D_cell{k} = X{k} - X{k}*Z{k} + Y{k}/mu;
    end
    
    % Set parameters for the proximal gradient solver
    max_iter_inner = 50;  % Inner loop max iterations
    tol_inner = 1e-5;     % Inner loop tolerance
    
    % Create initial E_stacked from current E
    E_stacked = [];
    for k = 1:K
        E_stacked = [E_stacked; E{k}];
    end
    
    % Solve using proximal gradient method
    E_stacked = solve_E_problem(C_cell, D_cell, lambda, mu, ...
                              max_iter_inner, tol_inner, E_stacked);
    
    % Split stacked E back to view-specific matrices
    start_idx = 1;
    for k = 1:K
        d_k = size(X{k}, 1);
        E{k} = E_stacked(start_idx:start_idx+d_k-1, :);
        start_idx = start_idx + d_k;
    end
    %------------------- End of Updated E^k Update -------------------
  
  
    %------------------- Update G---------------------------------
    Z_tensor = cat(3, Z{:,:});
    W_tensor = cat(3, W{:,:});

    [G_tensor, objV] = Frac_Shrink(Z_tensor + W_tensor/rho, 6/rho, 3, Frac_alpha);  

    % % 子问题改成对称
    % Z_tensor = cat(3, Z{:,:});
    % W_tensor = cat(3, W{:,:});
    % plus1=Z_tensor + W_tensor/rho;
    % plus2=permute(Z_tensor + W_tensor/rho,[2,1,3]);
    % plus=(plus1+plus2)/2;
    % % plus=plus1+plus2;
    % [G_tensor, objV] = Frac_Shrink(plus, 6/rho, 3, Frac_alpha); 


    %-------------------Update auxiliary variable---------------
    W_tensor = W_tensor  + rho*(Z_tensor - G_tensor);
    for k=1:K
        Y{k} = Y{k} + mu*((X{k}-E{k})-(X{k}-E{k})*Z{k});
        G{k} = G_tensor(:,:,k);
        W{k} = W_tensor(:,:,k);
    end   
    
    % Record the iteration information
    history.objval(iter+1) = objV;

    % Coverge condition
    Isconverg = 1;
    

    %% Check for convergence
    residual_cell = cell(1, K);
    for k = 1:K
        residual_cell{k} = (X{k}-E{k}) - (X{k}-E{k})*Z{k};
    end
    
    history.norm_Z(iter+1) = max(cellfun(@(x) norm(x, inf), residual_cell));
    
    if (history.norm_Z(iter+1) > epsilon)   
        if flag_debug
            fprintf('norm_Z   %7.10f  \n', history.norm_Z(iter+1));
        end
        Isconverg = 0;                 
    end
    

    history.norm_Z_G(iter+1) = max(cellfun(@(z,g) norm(z-g, inf), Z, G));
    if (history.norm_Z_G(iter+1)>epsilon)   
        if flag_debug
            fprintf('norm_Z_G   %7.10f  \n', history.norm_Z_G(iter+1));
        end
        Isconverg = 0;                 
    end 
    
   
    %% 
    if (iter>maxIter)
        Isconverg  = 1;
    end
    
    % Update penalty params
    mu = min(mu*eta, max_mu);
    rho = min(rho*eta, max_rho);
    
    iter = iter + 1;
end

%% Clustering
S = 0;
for k=1:K
    S = S + abs(Z{k})+abs(Z{k}');
end

% S = 0;
% for k=1:K
%     S = S + abs(Z{k});
% end

C = SpectralClustering(S,cls_num);

[~, nmi, ~] = compute_nmi(gt,C);
ACC = Accuracy(C,double(gt));
[f,p,r] = compute_f(gt,C);
[AR,~,~,~]=RandIndex(gt,C);

%% Record
Out.NMI = nmi;
Out.AR = AR;
Out.ACC = ACC;
Out.recall = r;
Out.precision = p;
Out.fscore = f;

Out.history = history;

