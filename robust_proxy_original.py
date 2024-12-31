import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.optimize import approx_fprime, minimize
from itertools import combinations
import itertools

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm, pinv


###############################################################################
###################### Robust Proximal Causal Inference #######################
###############################################################################

class Robust_Proxy:
    '''
    Proxy Maximum Moment Restriction (PMMR) algorithm
    '''
    def __init__(self, data, gamma = 1, max_iter = 15, tol = 1e-2):
        '''
        Args:
            Y (numpy.ndarray): n x 1 vector of observed outcomes; 
                               n is the number of observations.  
            W (numpy.ndarray): n x 1 vector of outcome-inducing proxies;
            Z (numpy.ndarray): n x m matrix of candidate treatment-inducing proxies;
                               m is the number of candidate proxies.
            A (numpy.ndarray): n x 1 vector of observed treatment;
            X (numpy.ndarray): n x p matrix of observed covariates;
                               p is the number of observed covariates.
            gamma (float): the number of valid treatment-inducing proxies; default is 1.
            max_iter (int): maximum number of iterations of the ACE algorithm; default is 100.
            tol (float): tolerance level for termination of ACE algorithm; default is 1e-6.
        '''
        self.data = data
        self.Y = data['Y']
        self.W = data['W']
        self.Z = data['Z']
        self.A = data['A'].astype(np.float64)
        self.X = data['X']
        self.n, self.p = self.X.shape
        _, self.q = self.Z.shape
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol

        # initialize the variables
        self.matrix = None

    def weight_matrix(self, standardize = True, degree = 2):
        '''
        Compute the weight matrix for the parameter of
        the outcome-inducing bridge function.
        '''

        if standardize:
            scaler = StandardScaler()
            Z_standardized = scaler.fit_transform(self.Z.copy())
        
        else:
            Z_standardized = self.Z.copy()
        
        power_range = list(range(1, degree + 1))
        column_combinations = list(combinations(range(self.q), self.q - self.gamma + 1))
        results = []

        for comb in column_combinations:
            selected_column = Z_standardized[:, comb]

            for powers in itertools.product(power_range, repeat = self.q - self.gamma + 1):
                powered_columns = np.prod(selected_column ** np.array(powers), axis=1)
                powered_columns = self.intersection(f_values = powered_columns, df = 5)['values']
                results.append(powered_columns) 

        A_intersection = self.intersection(f_values = (self.A).copy().reshape(-1))['values']
        results.append(A_intersection)
        results_copy = results.copy()

        for result_column in results_copy:
            product_with_X = self.X*result_column.reshape(-1, 1)
            results.append(product_with_X)

        results_array = np.column_stack(results)

        self.matrix = np.hstack((np.ones((self.n, 1)), results_array, self.X))

        return self.matrix

    def outcome_fit(self, method = 'L-BFGS-B', alpha = 0, inference = 'nominal'):
        '''
        Compute the parameter of interest using the outcome-inducing bridge function.
        '''
        if self.matrix is None:
            self.matrix = self.weight_matrix()

        size = 2 + self.W.shape[1] + self.X.shape[1]
        normalized_W = StandardScaler().fit_transform(self.W.copy())
        normalized_X = StandardScaler().fit_transform(self.X.copy())
        or_covariates = np.hstack((np.ones((self.n, 1)), normalized_W, self.A, normalized_X))
        or_weights = self.matrix.copy()        

        def outcome_moments(beta):
            return or_weights*(self.Y - or_covariates@(beta.reshape(-1,1)))
        
        def outcome_moments_mean(beta):
            return np.mean(outcome_moments(beta), axis = 0)

        def outcome_func(beta):
            return norm(outcome_moments_mean(beta).reshape(-1,1))**2 + alpha*(norm(beta)**2)

        temp_sol_outcome = minimize(outcome_func, x0 = np.zeros(size), method = method)
        temp_beta = temp_sol_outcome.x
        temp_Omega = outcome_moments(temp_beta).T@outcome_moments(temp_beta)/self.n
        temp_Omega_inv = pinv(temp_Omega)

        def outcome_func2(beta):
            temp_moments = outcome_moments_mean(beta).reshape(-1,1)
            return temp_moments.T@temp_Omega_inv@temp_moments
        
        sol_outcome = minimize(outcome_func2, x0 = temp_beta, method = method)
        A_index = 1 + self.W.shape[1]
        
        beta_sol = sol_outcome.x
        Omega = outcome_moments(beta_sol).T@outcome_moments(beta_sol)/self.n
        Omega_inv = pinv(Omega)

        G = - or_weights.T@or_covariates/self.n  #compute_jacobian(outcome_moments_mean, beta_sol)
            
        cov = pinv(G.T@Omega_inv@G)
        se = np.sqrt(cov[A_index, A_index]/self.n)

        if inference == 'nominal':
            return {'est': beta_sol[A_index], 'se': se, 'quantile':1.96}

        else :
            B = 200

            t_boots = np.zeros(B)
            for b in range(B):
                indices = np.random.choice(self.n, size = self.n, replace = True)

                or_covariates_resampled = or_covariates[indices]
                or_weights_resampled = self.matrix.copy()[indices]
                outcome_moments_mean_expanded = np.repeat(outcome_moments_mean(temp_beta)[np.newaxis, :], self.n, axis=0)

                def outcome_moments_resampled(beta):
                    return or_weights_resampled*(self.Y[indices] - or_covariates_resampled@beta.reshape(-1,1)) - outcome_moments_mean_expanded
                
                def outcome_func_resampled(beta):
                    temp_mean_resampled = np.mean(outcome_moments_resampled(beta), axis = 0).reshape(-1,1)
                    return norm(temp_mean_resampled)**2
                
                sol_outcome_resampled = minimize(outcome_func_resampled, x0 = np.zeros(size), method = method)
                temp_beta_boot = sol_outcome_resampled.x
                temp_Omega_boot = outcome_moments_resampled(temp_beta_boot).T@outcome_moments_resampled(temp_beta_boot)/self.n
                temp_Omega_inv_boot = pinv(temp_Omega_boot)
                outcome_moments_mean_expanded2 = np.repeat(outcome_moments_mean(beta_sol)[np.newaxis, :], self.n, axis=0)

                def outcome_moments_resampled2(beta):
                    return or_weights_resampled*(self.Y[indices] - or_covariates_resampled@beta.reshape(-1,1)) - outcome_moments(beta_sol)
                
                def outcome_moments_resampled2_mean(beta):
                    return np.mean(outcome_moments_resampled2(beta), axis = 0).reshape(-1,1)

                def outcome_func_resampled2(beta):
                    return outcome_moments_resampled2_mean(beta).T@temp_Omega_inv_boot@outcome_moments_resampled2_mean(beta)
                
                sol_outcome_resampled = minimize(outcome_func_resampled2, x0 = temp_beta_boot, method = method)
                beta_boot = sol_outcome_resampled.x
                Omega_boot = outcome_moments_resampled2(beta_boot).T@outcome_moments_resampled2(beta_boot)/self.n
                Omega_inv_boot = pinv(Omega_boot)

                G_boot = -or_weights_resampled.T@or_covariates_resampled/self.n #compute_jacobian(outcome_moments_resampled2_mean, beta_boot)
                cov_boot = pinv(G_boot.T@Omega_inv_boot@G_boot)
                se_boot = np.sqrt(cov_boot[A_index, A_index]/self.n)
                t_boots[b] = np.abs((beta_boot[A_index] - beta_sol[A_index])/se_boot)
            
            t_quantile = np.quantile(t_boots, 0.95)
            return {'est': beta_sol[A_index], 'se': se, 'quantile': t_quantile}


    def treatment_fit(self, method = 'L-BFGS-B', alpha = 0, inference = 'nominal'):
        '''
        Compute the parameter of interest using the treatment-inducing bridge function.
        '''
        
        size = 2 + self.Z.shape[1] + self.X.shape[1]
        normalized_Z = StandardScaler().fit_transform(self.Z.copy())
        normalized_X = StandardScaler().fit_transform(self.X.copy())
        tr_covariates = np.hstack((np.ones((self.n, 1)), normalized_Z, self.A, normalized_X))
        tr_weights = np.hstack((np.ones((self.n, 1)), self.W, self.Z, self.A, self.X))

        def q_vector(gamma):
            return (2*self.A - 1)*(1 + np.exp((2*self.A - 1)*(tr_covariates@gamma.reshape(-1,1))))

        subtract_matrix = np.zeros_like(tr_weights) 
        A_column_index = tr_weights.shape[1] - self.X.shape[1] - 1 
        subtract_matrix[:, A_column_index] = 1

        def tr_moments(gamma, tau):
            return np.hstack((tr_weights * q_vector(gamma) - subtract_matrix, self.Y*q_vector(gamma) - tau))
        
        def tr_moments_mean(x):
            gamma, tau = x[:-1], x[-1]
            return np.mean(tr_moments(gamma, tau), axis = 0)
        
        def tr_func(x):
            gamma, tau = x[:-1], x[-1]
            return norm(tr_moments_mean(x))**2 + alpha*(norm(gamma)**2)
        
        temp_sol_treatment = minimize(tr_func, x0 = np.zeros(size + 1), method = method)
        temp_gamma, temp_tau = temp_sol_treatment.x[:-1], temp_sol_treatment.x[-1]

        temp_Omega = tr_moments(temp_gamma, temp_tau).T@tr_moments(temp_gamma, temp_tau)/self.n
        temp_Omega1_inv = pinv(temp_Omega)

        def tr_func2(x):
            temp_moments = tr_moments_mean(x).reshape(-1,1)
            return temp_moments.T@temp_Omega1_inv@temp_moments
        
        sol_treatment = minimize(tr_func2, x0 = temp_sol_treatment.x, method = method)
        gamma_sol, tau_sol = sol_treatment.x[:-1], sol_treatment.x[-1]
        Omega = tr_moments(gamma_sol, tau_sol).T@tr_moments(gamma_sol, tau_sol)/self.n
        Omega_inv = pinv(Omega)

        G = compute_jacobian(tr_moments_mean, np.hstack((gamma_sol, tau_sol)))
        cov = pinv(G.T@Omega_inv@G)
        se = np.sqrt(cov[-1, -1]/self.n)

        if inference == 'nominal':
            return {'est': sol_treatment.x[-1], 'se': se, 'quantile':1.96}
        
        else :
            B = 200
            t_boots = np.zeros(B)
            for b in range(B):
                indices = np.random.choice(self.n, size = self.n, replace = True)

                tr_covariates_resampled = tr_covariates[indices]
                tr_weights_resampled = tr_weights[indices]
                tr_moments_mean_expanded = np.repeat(tr_moments_mean(np.hstack((temp_gamma, temp_tau)))[np.newaxis, :], self.n, axis=0)

                def q_vector_resampled(gamma):
                    return (2*self.A[indices] - 1)*(1 + np.exp((2*self.A[indices] - 1)*(tr_covariates_resampled@gamma.reshape(-1,1))))

                def tr_moments_resampled(x):
                    gamma, tau = x[:-1], x[-1]
                    return np.hstack((tr_weights_resampled * q_vector_resampled(gamma) - subtract_matrix, self.Y[indices]*q_vector_resampled(gamma) - tau)) \
                    - tr_moments_mean_expanded
                
                def tr_func_resampled(x):
                    gamma, tau = x[:-1], x[-1]
                    return norm(np.mean(tr_moments_resampled(x), axis = 0))**2 + alpha*(norm(gamma)**2) 
                
                temp_sol_treatment_resampled = minimize(tr_func_resampled, x0 = np.zeros(size + 1), method = method)
                temp_sol_resampled = temp_sol_treatment_resampled.x

                temp_Omega_resampled = tr_moments_resampled(temp_sol_resampled).T@tr_moments_resampled(temp_sol_resampled)/self.n
                temp_Omega1_inv_resampled = pinv(temp_Omega_resampled)
                tr_moments_mean_expanded2 = np.repeat(tr_moments_mean(np.hstack((gamma_sol, tau_sol)))[np.newaxis, :], self.n, axis=0)

                def tr_moments_resampled2(x):
                    gamma, tau = x[:-1], x[-1]
                    return np.hstack((tr_weights_resampled * q_vector_resampled(gamma) - subtract_matrix, self.Y[indices]*q_vector_resampled(gamma) - tau)) \
                    - tr_moments_mean_expanded2
                
                def tr_moments_resampled2_mean(x):
                    return np.mean(tr_moments_resampled2(x), axis = 0).reshape(-1,1)

                def tr_func2_resampled(x):
                    return tr_moments_resampled2_mean(x).T@temp_Omega1_inv_resampled@tr_moments_resampled2_mean(x)
                
                sol_treatment_resampled = minimize(tr_func2_resampled, x0 = temp_sol_resampled, method = method)
                sol_boot = sol_treatment_resampled.x
                Omega_boot = tr_moments_resampled2(sol_boot).T@tr_moments_resampled2(sol_boot)/self.n
                Omega_inv_boot = pinv(Omega_boot)

                G_boot = compute_jacobian(tr_moments_resampled2_mean, sol_boot)
                cov_boot = pinv(G_boot.T@Omega_inv_boot@G_boot)
                se_boot = np.sqrt(cov_boot[-1, -1]/self.n)
                t_boots[b] = np.abs((sol_boot[-1] - sol_treatment.x[-1])/se_boot)

            t_quantile = np.quantile(t_boots, 0.95)
            return {'est': sol_treatment.x[-1], 'se': se, 'quantile': t_quantile}

    
    def dr_fit(self, method = 'L-BFGS-B', inference = 'nominal'):
        '''
        Compute the parameter of interest using the double robust estimator.
        '''
        if self.matrix is None:
            self.matrix = self.weight_matrix()

        normalized_W = StandardScaler().fit_transform(self.W.copy())
        normalized_Z = StandardScaler().fit_transform(self.Z.copy())
        normalized_X = StandardScaler().fit_transform(self.X.copy())

        tr_covariates = np.hstack((np.ones((self.n, 1)), normalized_Z, self.A, normalized_X))
        or_covariates = np.hstack((np.ones((self.n, 1)), normalized_W, self.A, normalized_X))
        res_covariates = np.hstack((np.ones((self.n, 1)), normalized_Z, normalized_X))

        or_weights = self.matrix.copy()
        tr_weights = np.hstack((np.ones((self.n, 1)), self.W, self.Z, self.A, self.X))
        res_weights = np.hstack((np.ones((self.n, 1)), self.Z, self.A, self.X))

        subtract_matrix = np.zeros_like(tr_weights) 
        A_column_index = tr_weights.shape[1] - self.X.shape[1] - 1 # index where the moment equation for A is located
        subtract_matrix[:, A_column_index] = 1

        A_index = 1 + self.W.shape[1] # index where A is located in beta

        def q_vector(gamma):
            return (2*self.A - 1)*(1 + np.exp((2*self.A - 1)*(tr_covariates@gamma.reshape(-1,1))))

        size_or = 2 + self.W.shape[1] + self.X.shape[1]
        size_tr = 2 + self.Z.shape[1] + self.X.shape[1]
        size_res = res_covariates.shape[1]

        def dr_moments(x):
            beta, gamma, delta, tau = x[:size_or], x[size_or:size_or + size_tr], x[size_or + size_tr:size_or + size_tr + size_res], x[-1]
            return np.hstack((self.matrix*(self.Y - or_covariates@beta.reshape(-1,1)), \
                              tr_weights * q_vector(gamma) - subtract_matrix, \
                              res_weights*(self.Y - or_covariates@beta.reshape(-1,1) - res_covariates@delta.reshape(-1,1)), \
                              beta[A_index] + q_vector(gamma)*(self.Y - or_covariates@beta.reshape(-1,1) - res_covariates@delta.reshape(-1,1)) - tau))
        

        def dr_moments_mean(x):
            return np.mean(dr_moments(x), axis = 0)

        def dr_func(x):
            return norm(dr_moments_mean(x))**2

        temp_sol_dr = minimize(dr_func, x0 = np.zeros(size_or + size_tr + size_res + 1), method = method)
        temp_x = temp_sol_dr.x
        temp_Omega = dr_moments(temp_x).T@dr_moments(temp_x)/self.n
        temp_Omega_inv = pinv(temp_Omega)

        def dr_func2(x):
            temp_moments = dr_moments_mean(x).reshape(-1,1)
            return temp_moments.T@temp_Omega_inv@temp_moments

        sol_dr = minimize(dr_func2, x0 = temp_x, method = method)
        x_sol = sol_dr.x
        G = compute_jacobian(dr_moments_mean, x_sol)
        Omega = dr_moments(x_sol).T@dr_moments(x_sol)/self.n
        Omega_inv = pinv(Omega)
        cov = pinv(G.T@Omega_inv@G)
        se = np.sqrt(cov[-1, -1]/self.n)

        if inference == 'nominal':
            return {'est': sol_dr.x[-1], 'se': se, 'quantile':1.96}
        
        else :
            B = 200
            t_boots = np.zeros(B)
            for b in range(B):
                indices = np.random.choice(self.n, size = self.n, replace = True)

                tr_covariates_resampled = tr_covariates[indices]
                or_covariates_resampled = or_covariates[indices]
                res_covariates_resampled = res_covariates[indices]

                or_weights_resampled = self.matrix.copy()[indices]
                tr_weights_resampled = tr_weights[indices]
                res_weights_resampled = res_weights[indices]

                A_index_resampled = 1 + self.W.shape[1]

                def q_vector_resampled(gamma):
                    return (2*self.A[indices] - 1)*(1 + np.exp((2*self.A[indices] - 1)*(tr_covariates_resampled@gamma.reshape(-1,1))))

                dr_moments_resampled_mean_expanded = np.repeat(dr_moments_mean(sol_dr.x)[np.newaxis, :], self.n, axis=0)

                def dr_moments_resampled(x):
                    beta, gamma, delta, tau = x[:size_or], x[size_or:size_or + size_tr], x[size_or + size_tr:size_or + size_tr + size_res], x[-1]
                    return np.hstack((or_weights_resampled*(self.Y[indices] - or_covariates_resampled@beta.reshape(-1,1)), \
                              tr_weights_resampled * q_vector_resampled(gamma) - subtract_matrix, \
                              res_weights_resampled*(self.Y[indices] - or_covariates_resampled@beta.reshape(-1,1) - res_covariates_resampled@delta.reshape(-1,1)), \
                              beta[A_index] + q_vector_resampled(gamma)*(self.Y[indices] - or_covariates_resampled@beta.reshape(-1,1) - res_covariates_resampled@delta.reshape(-1,1)) - tau)) \
                                - dr_moments_resampled_mean_expanded

                def dr_func_resampled(x):
                    return norm(np.mean(dr_moments_resampled(x), axis = 0))**2

                temp_sol_dr_resampled = minimize(dr_func_resampled, x0 = np.zeros(size_or + size_tr + size_res + 1), method = method)
                temp_x_resampled = temp_sol_dr_resampled.x
                temp_Omega_resampled = dr_moments_resampled(temp_x_resampled).T@dr_moments_resampled(temp_x_resampled)/self.n
                temp_Omega_inv_resampled = pinv(temp_Omega_resampled)
                dr_moments_resampled_mean_expanded2 = np.repeat(dr_moments_mean(x_sol)[np.newaxis, :], self.n, axis=0)

                def dr_moments_resampled2(x):
                    beta, gamma, delta, tau = x[:size_or], x[size_or:size_or + size_tr], x[size_or + size_tr:size_or + size_tr + size_res], x[-1]
                    return np.hstack((or_weights_resampled*(self.Y[indices] - or_covariates_resampled@beta.reshape(-1,1)), \
                              tr_weights_resampled * q_vector_resampled(gamma) - subtract_matrix, \
                              res_weights_resampled*(self.Y[indices] - or_covariates_resampled@beta.reshape(-1,1) - res_covariates_resampled@delta.reshape(-1,1)), \
                              beta[A_index] + q_vector_resampled(gamma)*(self.Y[indices] - or_covariates_resampled@beta.reshape(-1,1) - res_covariates_resampled@delta.reshape(-1,1)) - tau)) \
                                - dr_moments_resampled_mean_expanded2
                
                def dr_moments_resampled2_mean(x):
                    return np.mean(dr_moments_resampled2(x), axis = 0).reshape(-1,1)
                
                def dr_func2_resampled(x):
                    return dr_moments_resampled2_mean(x).T@temp_Omega_inv_resampled@dr_moments_resampled2_mean(x)

                sol_dr_resampled = minimize(dr_func2_resampled, x0 = temp_x_resampled, method = method)
                x_sol_resampled = sol_dr_resampled.x
                Omega_resampled = dr_moments_resampled2(x_sol_resampled).T@dr_moments_resampled2(x_sol_resampled)/self.n
                Omega_inv_resampled = pinv(Omega_resampled)

                G_resampled = compute_jacobian(dr_moments_resampled2_mean, x_sol_resampled)
                cov_resampled = pinv(G_resampled.T@Omega_inv_resampled@G_resampled)
                se_resampled = np.sqrt(cov_resampled[-1, -1]/self.n)
                t_boots[b] = np.abs((x_sol_resampled[-1] - x_sol[-1])/se_resampled)

            t_quantile = np.quantile(t_boots, 0.95)
            return {'est': x_sol[-1], 'se': se, 'quantile': t_quantile}

                
    
    def intersection(self, f_values = None, df = 5, alpha = 1):
        ''' 
        Projection of the given function f onto the intersection of subspaces using neural networks

        Args:
            f_values (numpy.ndarray): n x 1 vector of values of the function to be projected.
        '''

        if f_values is None:
            raise Exception("function values should be provided")

        f_values_copy = f_values.copy()
        column_names = [f"var{i}" for i in range(self.X.shape[1])]
        temp_data = pd.DataFrame(self.X, columns=column_names)
        spline_formula = " * ".join([f"bs({col}, df=df, degree=3)" for col in column_names])
        design_matrix = dmatrix(spline_formula, temp_data)
        f_x = Ridge(alpha=alpha).fit(design_matrix, f_values_copy) #LinearRegression().fit(design_matrix, f_values_copy)
        f_x_values = f_x.predict(design_matrix)

        column_combinations = list(combinations(range(self.q), self.q - self.gamma))
        iter = 0
        f_update_values = f_values.copy()

        while True:
            f_old_values = f_update_values.copy()
            
            for cols in column_combinations:
                combined_matrix = np.hstack((self.Z[:, cols], self.X))

                column_names = [f"var{i}" for i in range(combined_matrix.shape[1])]
                temp_data = pd.DataFrame(combined_matrix, columns=column_names)
                spline_formula = " * ".join([f"bs({col}, df=df, degree=3)" for col in column_names])
                design_matrix = dmatrix(spline_formula, temp_data)
                f_cols = Ridge(alpha = alpha).fit(design_matrix, f_update_values) #LinearRegression().fit(design_matrix, f_update_values)
                f_cols_values = f_cols.predict(design_matrix)

                f_update_values -= f_cols_values
            
            iter += 1
            norm_ratio = norm(f_update_values - f_old_values) / norm(f_old_values)

            if norm_ratio < self.tol or iter >= self.max_iter:
                f_final_values = f_update_values + f_x_values 
                break

        return {'values': f_final_values, 'iter': iter}

###############################################################################
######################### Parametric Proxy Learning ###########################
###############################################################################

class Proxy:
    def __init__(self, data):
        '''
        Args:
            Y (numpy.ndarray): n x 1 vector of observed outcomes; 
                               n is the number of observations.  
            W (numpy.ndarray): n x k vector of outcome-inducing proxies;
                               k is the number of outcome-inducing proxies.
            Z (numpy.ndarray): n x m matrix of treatment-inducing proxies;
                               m is the number of treatment-inducing proxies.
            A (numpy.ndarray): n x 1 vector of observed treatment;
            X (numpy.ndarray): n x p matrix of observed covariates;
                               p is the number of observed covariates.
        '''
        self.data = data
        self.Y = data['Y']
        self.W = data['W']
        self.Z = data['Z']
        self.A = data['A'].astype(np.float64)
        self.X = data['X']
        self.n = self.Y.shape[0]

        self.or_para = None
        self.tr_para = None

    def outcome_fit(self, method = 'L-BFGS-B', alpha = 0):
        '''
        Compute the parameter of interest using the outcome-inducing bridge function.
        '''

        size = 2 + self.W.shape[1] + self.X.shape[1]

        outcome_matrix = np.hstack((np.ones((self.n, 1)), self.Z, self.A, self.X))
        outcome_covariates = np.hstack((np.ones((self.n, 1)), self.W, self.A, self.X))

        def outcome_moments(beta):
            return outcome_matrix*(self.Y - outcome_covariates@beta.reshape(-1,1))

        def outcome_moments_mean(beta):
            return np.mean(outcome_moments(beta), axis = 0)

        func = lambda beta: np.linalg.norm(outcome_moments_mean(beta))**2 + alpha*(np.linalg.norm(beta)**2)

        sol_outcome = minimize(func, x0 = np.zeros(size), method = method)
        A_index = 1 + self.W.shape[1]
        
        G = compute_jacobian(outcome_moments_mean, sol_outcome.x)
        Omega = outcome_moments(sol_outcome.x).T@outcome_moments(sol_outcome.x)/self.n
        G_inv = pinv(G.T@G)
        cov = G_inv@G.T@Omega@G@G_inv
        se = np.sqrt(cov[A_index, A_index]/self.n)

        return {'est': sol_outcome.x[A_index], 'se': se}

    def treatment_fit(self, method = 'L-BFGS-B', alpha = 0):
        '''
        Compute the parameter of interest using the treatment-inducing bridge function.
        '''
        
        size = 2 + self.Z.shape[1] + self.X.shape[1]

        func = lambda beta: self.gmm_treatment_objective(beta) + alpha*(np.linalg.norm(beta)**2)

        sol_treatment = minimize(func, x0 = np.zeros(size), method = method)
        self.tr_para = sol_treatment.x

        covariates = np.hstack((np.ones((self.n, 1)), self.Z, self.A, self.X))
        

        return np.mean(self.Y*((2*self.A - 1)*(1 + np.exp((2*self.A - 1)*(covariates@sol_treatment.x.reshape(-1,1))))))
    
    def dr_fit(self):
        '''
        Compute the parameter of interest using the double robust estimator.
        '''
        if self.or_para is None:
            self.outcome_fit()
        if self.tr_para is None:
            self.treatment_fit()
        
        tr_covariates = np.hstack((np.ones((self.n, 1)), self.Z, self.A, self.X))
        q_vector = (2*self.A - 1)*(1 + np.exp((2*self.A - 1)*(tr_covariates@self.tr_para.reshape(-1,1))))

        or_covariates = np.hstack((np.ones((self.n, 1)), self.W, self.A, self.X))
        h_vector = or_covariates@self.or_para.reshape(-1,1)
        beta = self.or_para[1 + self.W.shape[1]]

        new_matrix = [self.Z]

        for i, j in combinations(range(self.Z.shape[1]), 2):
            interaction = (self.Z[:, i] * self.Z[:, j]).reshape(-1, 1)
            new_matrix.append(interaction)

        # Concatenate all components into a single matrix
        new_matrix = np.hstack(new_matrix)  
        new_matrix = np.hstack((new_matrix, self.X)) 
        reg_z = LinearRegression().fit(new_matrix, self.Y - h_vector)
        r_vector = reg_z.predict(new_matrix)

        return np.mean(beta + q_vector*(self.Y - h_vector - r_vector))

    def gmm_outcome_objective(self, beta: np.ndarray) -> float:
        outcome_matrix = np.hstack((np.ones((self.n, 1)), self.Z, self.A, self.X))
        covariates = np.hstack((np.ones((self.n, 1)), self.W, self.A, self.X))
        moments = outcome_matrix*(self.Y - covariates@beta.reshape(-1,1))
        
        moments_mean = np.mean(moments, axis = 0)
        
        return float(moments_mean.T@moments_mean)

    def gmm_treatment_objective(self, beta: np.ndarray) -> float:
        covariates = np.hstack((np.ones((self.n, 1)), self.Z, self.A, self.X))
        treatment_matrix = np.hstack((np.ones((self.n, 1)), self.W, self.A, self.X))
        q_vector = (2*self.A - 1)*(1 + np.exp((2*self.A - 1)*(covariates@beta.reshape(-1,1))))

        subtract_matrix = np.zeros_like(treatment_matrix)  # Same shape as treatment_matrix
        A_column_index = treatment_matrix.shape[1] - self.X.shape[1] - 1  # Index where self.A is located
        subtract_matrix[:, A_column_index] = 1

        # Compute moments
        moments = treatment_matrix * q_vector - subtract_matrix
        
        moments_mean = np.mean(moments, axis = 0)
        return float(moments_mean.T@moments_mean)
    

###############################################################################
########################### Helper Functions ##################################
###############################################################################

def compute_jacobian(func, x, epsilon=1e-5):
    """
    Compute the Jacobian of a function using scipy's approx_fprime.

    Args:
    func: A callable that takes a vector input and returns a vector output.
    x: The point at which to evaluate the Jacobian.
    epsilon: Step size for finite differences.

    Returns:
    Jacobian matrix of shape (len(func(x)), len(x)).
    """
    n = len(x)
    m = len(func(x))  # Number of outputs
    jacobian = np.zeros((m, n))
    for i in range(m):
        single_func = lambda xi: func(xi)[i]
        jacobian[i, :] = approx_fprime(x, single_func, epsilon)
            
    return jacobian
