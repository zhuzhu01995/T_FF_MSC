%% ======== Local Function: solve_E_problem ========
function E_stacked = solve_E_problem(C_cell, D_cell, lambda, mu, max_iter, epsilon, E_init)
% FISTA-accelerated proximal gradient method for solving the multi-view E-subproblem
% Inputs:
%   C_cell: cell array {C^{(1)}, C^{(2)}, ..., C^{(V)}} with each C{v} of size N × N
%   D_cell: cell array {D^{(1)}, D^{(2)}, ..., D^{(V)}} with each D{v} of size d_v × N
%   lambda: regularization parameter λ
%   mu: penalty parameter μ (used in ADMM)
%   max_iter: maximum number of iterations
%   epsilon: convergence tolerance
%   E_init: initial stacked matrix (optional)
%
% Output:
%   E_stacked: stacked matrix [E^{(1)}; E^{(2)}; ...; E^{(V)}]

% Retrieve the number of views and dimensional information
V = length(C_cell);  % Number of views
dims = cellfun(@(d) size(d, 1), D_cell);  % Dimension of each view
N = size(D_cell{1}, 2);  % Number of samples
total_dim = sum(dims);  % Total dimension 

% Precompute the starting row index for each view
row_start = [1, cumsum(dims(1:end-1)) + 1];
row_end = cumsum(dims);

%%  Compute the Lipschitz constant
L_val = 0;
for v = 1:V
    CCT = C_cell{v} * C_cell{v}';
    norm_CCT = norm(CCT, 2);  
    
    if norm_CCT > L_val
        L_val = norm_CCT;
    end
end
L_val = mu * L_val;  % Final Lipschitz constant
eta = 1 / L_val;     % stepsize

%% Initialize variables
if nargin >= 7 && ~isempty(E_init)
    E_k = E_init;  
else
    E_k = zeros(total_dim, N);  
end

W_k = E_k;                 % Auxiliary variable (for FISTA acceleration)
t_k = 1;                   % FISTA momentum parameter

%% Main iteration loop
for iter = 1:max_iter
    % Store the solution from the previous iteration 
    E_prev = E_k;
    
    % Compute gradient (view-wise parallel)
    G_cell = cell(1, V);  % Store gradient for each view
    
    for v = 1:V
        % Extract the part of W_k corresponding to the current view
        W_v = W_k(row_start(v):row_end(v), :);
        
        % Compute gradient: G(v) = μ * (W_v * C{v} + D{v}) * C{v}'
        term = W_v * C_cell{v} + D_cell{v};
        G_cell{v} = mu * (term * C_cell{v}');  
    end
    
    % Stack gradients from all views
    G_stacked = vertcat(G_cell{:});
    
    % Gradient step: U = W_k - η * G_stacked ---
    U = W_k - eta * G_stacked;
    
    % Proximal operator: column-wise shrinkage
    E_next = zeros(total_dim, N);
    for j = 1:N
        u_j = U(:, j);          % Get the j-th column
        norm_u = norm(u_j, 2);  % Compute ℓ2 norm 
        
         threshold = eta * lambda;
        
        if norm_u > threshold
            E_next(:, j) = (1 - threshold / norm_u) * u_j;
        else
            E_next(:, j) = 0;
        end
    end
    
    % FISTA acceleration 
    t_next = (1 + sqrt(1 + 4 * t_k^2)) / 2;
    momentum = (t_k - 1) / t_next;
    
    % Update auxiliary variable: W_{k+1} = E_{k+1} + momentum * (E_{k+1} - E_k)
    W_next = E_next + momentum * (E_next - E_k);
    
    % Update iteration variables
    E_k = E_next;
    W_k = W_next;
    t_k = t_next;
    
    % Convergence check
    diff_norm = norm(E_k - E_prev, 'fro');
    if diff_norm < epsilon
        % if iter > 1  
        %     fprintf('Inner E solver converged at iter %d, ||ΔE||_F = %.4e\n', iter, diff_norm);
        % end
        break;
    end
end

% Return the final solution
E_stacked = E_k;
end