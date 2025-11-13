function s_sigma = Frac_update_sigma(sigma, lambda, a)
   
    lambda_mu = lambda;

    % Compute threshold
    a_sq = a * a;
    if lambda_mu <= 1/a_sq
        t_star = (lambda_mu * a)/2;
    else
        t_star = sqrt(lambda_mu) - 1/(2*a);
    end

    % Apply threshold rules
    if abs(sigma) <= t_star
        s_sigma = 0;
    else
        % Call the glu function to compute non-zero solutions
        s_sigma = glu(lambda_mu, a, sigma);
    end
end

