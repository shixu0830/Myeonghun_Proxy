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
            gamma (float): the number of valid treatment-inducing proxies; default is 1.
            max_iter (int): maximum number of iterations of the ACE algorithm; default is 100.
            tol (float): tolerance level for termination of ACE algorithm; default is 1e-6.
        '''
        self.data = data
        self.Y = data['Y']
        self.W = data['W']
        self.Z = data['Z']
        self.A = data['A'].astype(np.float64)
        self.n, self.p = 0, self.q = self.Z.shape
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

        

        results_array = np.column_stack(results)

        self.matrix = np.hstack((np.ones((self.n, 1)), results_array))

        return self.matrix

    def outcome_fit(self, method = 'L-BFGS-B', alpha = 0, inference = 'nominal'):
        '''
        Compute the parameter of interest using the outcome-inducing bridge function.
        '''
        if self.matrix is None:
            self.matrix = self.weight_matrix()

        size = 2 + self.W.shape[1]
        normalized_W = StandardScaler().fit_transform(self.W.copy())
        or_covariates = np.hstack((np.ones((self.n, 1)), normalized_W, self.A))
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
        '''
        self.data = data
        self.Y = data['Y']
        self.W = data['W']
        self.Z = data['Z']
        self.A = data['A'].astype(np.float64)
        self.n = self.Y.shape[0]

        self.or_para = None
        self.tr_para = None

    def outcome_fit(self, method = 'L-BFGS-B', alpha = 0):
        '''
        Compute the parameter of interest using the outcome-inducing bridge function.
        '''

        size = 2 + self.W.shape[1]

        outcome_matrix = np.hstack((np.ones((self.n, 1)), self.Z, self.A))
        outcome_covariates = np.hstack((np.ones((self.n, 1)), self.W, self.A))

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



    def gmm_outcome_objective(self, beta: np.ndarray) -> float:
        outcome_matrix = np.hstack((np.ones((self.n, 1)), self.Z, self.A))
        covariates = np.hstack((np.ones((self.n, 1)), self.W, self.A))
        moments = outcome_matrix*(self.Y - covariates@beta.reshape(-1,1))
        
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
