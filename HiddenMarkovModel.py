# This is an implementation of Hidden Markov Models
# And it's three usages with their algorithms
# HMM is mosted used in Natual Language Processing
# Reference: https://zhuanlan.zhihu.com/p/85454896
import numpy as  np 

'''
    Main Achievable goal by modeling HMM
    Probability Calculation: Given parameters of a model & observe sequence
                             Calculate the probability of such observe sequence
    Parameter learning:      Given observe sequence
                             Calculate the paramter of the model
    Decoding:                Given parameters of a model & observe sequence
                             Calculate the best possible latent state sequence 
'''

class HMM(object):
    def __init__(self, N, M, A, B, pi):
        '''
            N:      Total number of possible latent states
            M:      Total number of possible observable stats
            A:      Latent state transition matrix              aij: state i -> state j
            B:      Observe matrix
            pi:     Initial latent state distribution
        '''
        self.N = N
        self.M = M
        self.A = A
        self.B = B
        self.pi = pi
        
    def multinomial_distribution_sample(self, pval):
        sampled_0_1 = np.random.rand()
        for i, p in enumerate(pval):
            if sampled_0_1 < p:
                return i
            sampled_0_1 -= p

    def generate_one_sample_path(self, T: int):
        '''
            T:      length of the path
        '''
        z_0 = self.multinomial_distribution_sample(self.pi)
        x_0 = self.multinomial_distribution_sample(self.B[z_0])
        sample_latent_path = [z_0]
        sample_observe_path = [x_0]
        for _ in range(T):
            z_next = self.multinomial_distribution_sample(self.A[sample_latent_path[-1]])
            x_next = self.multinomial_distribution_sample(self.B[z_next])
            sample_latent_path.append(z_next)
            sample_observe_path.append(x_next)
        print('Sampled Latent State Path: ', sample_latent_path)
        print('Sampled Observe State Path:', sample_observe_path)
    
    def observe_sequence_probability_cal(self, X, use_forward = True):
        '''
            There are two approaches: Forward Calculation & Backward Calculation
        '''
        if use_forward:
            self.forward_pass(X)
        else:
            self.backward_pass(X)

    def forward_pass(self, X, A = None, B = None, pi = None, use_default = False, mute_output = False):
        # Alpha_path stores all the calcultion process, and is kept for future use
        if not use_default:
            A, B, pi = self.A, self.B, self.pi
        alpha_path = [(pi * B[:, X[0]]).tolist()]
        for x in X[1:]:
            alpha_next = [0.0] * self.N
            for i in range(self.N):
                for j in range(self.N):
                    alpha_next[i] += B[i, x] * alpha_path[-1][j] * A[j, i]
            alpha_path.append(alpha_next)
        # Sum the probabilites up
        target_result = []
        for t_row in alpha_path:
            target_result.append(sum(t_row))
        if not mute_output:
            print('Probability of the Given Observed Path by forward_cal is: ', target_result[-1])
        return target_result[-1], alpha_path

    def backward_pass(self, X, A = None, B = None, pi = None, use_default = False, mute_output = False):
        # Beta_path store all the calculation proces, and is kept for future use
        if not use_default:
            A, B, pi = self.A, self.B, self.pi
        beta_path = [[1] * self.N]
        for x in X[:0:-1]:
            beta_next = [0.0] * self.N
            for i in range(self.N):
                for j in range(self.N):
                    beta_next[i] += B[j, x] * beta_path[-1][j] * A[i, j]
            beta_path.append(beta_next)
        # For the last row
        target_result = 0
        for i in range(self.N):
            target_result += B[i, x] * beta_path[-1][i] * pi[i]
        if not mute_output:
            print('Probability of the Given Observed Path by backward_cal is:', target_result)
        return target_result, beta_path[::-1]

    def parameter_learning(self, X, max_iter = 50):
        '''
            Parameter learning given a determined observed sequence
            Done by EM: Expectation Maximization on pi(t) & a(t) & b(t)
        '''
        # Parameter Initialization
        em_pi = np.ones((self.N), dtype = float).tolist()
        em_A  = np.ones((self.N, self.N), dtype = float).tolist()
        em_B  = np.ones((self.N, self.M), dtype = float).tolist()
        em_pi = [x / sum(em_pi) for x in em_pi]
        for row_pnt in range(self.N):
            em_A[row_pnt] = [x / sum(em_A[row_pnt]) for x in em_A[row_pnt]]
            em_B[row_pnt] = [x / sum(em_B[row_pnt]) for x in em_B[row_pnt]]
        T = len(X)
        for _ in range(max_iter):
            _, alpha = self.forward_pass(X, np.asarray(em_A), np.asarray(em_B), np.asarray(em_pi), True, True)
            _, beta = self.backward_pass(X, np.asarray(em_A), np.asarray(em_B), np.asarray(em_pi), True, True)
            gamma = np.multiply(np.asarray(alpha), np.asarray(beta)).tolist()
            new_A = [[0.0] * self.N for _ in range(self.N)]
            new_B = [[0.0] * self.M for _ in range(self.N)]
            new_pi = [x / sum(gamma[-1]) for x in gamma[0]]
            # Update a
            for i in range(self.N):
                for j in range(self.N):
                    denominator, divisor = 0.0, 0.0
                    for t in range(T - 1):
                        denominator += alpha[t][i] * beta[t + 1][j] * em_A[i][j] * em_B[j][X[t + 1]]
                        divisor += alpha[t][i] * beta[t][i]
                    new_A[i][j] = denominator / divisor
            # Update b
            for i in range(self.N):
                for k in range(self.M):
                    denominator, divisor = 0.0, 0.0
                    for t in range(T):
                        denominator += int(X[t] == k) * alpha[t][i] * beta[t][i]
                        divisor += alpha[t][i] * beta[t][i]
                    new_B[i][k] = denominator / divisor
            em_A, em_B, em_pi = new_A, new_B, new_pi
        print('Parameter fit complete')
        print('A:', em_A, '\nB:', em_B, '\npi:', em_pi)

    def viterbi_dp(self, X):
        '''
            Given the parameter & observation sequence
            Calculate the most probable latent state sequence
            It's a Dynamic Programming algorithm, named "Viterbi"
        '''
        T = len(X)
        dp = [[0.0] * self.N for _ in range(T)]
        # Initialization
        for i in range(self.N):
            dp[0][i] = [self.pi[i] * self.B[i][X[0]], str(i)]
        # Forward calculation
        for t in range(1, T):
            for j in range(self.N):
                candidate = [dp[t - 1][k][0] * self.A[k][j] * self.B[j][X[t]] for k in range(self.N)]
                next_state = candidate.index(max(candidate))
                dp[t][j] = [max(candidate), dp[t - 1][next_state][1] + str(next_state)]
        # Find the optimal path
        last_row_transpose = [[dp[-1][j][i] for j in range(self.N)] for i in range(2)]
        optimal_terminal_state = last_row_transpose[0].index(max(last_row_transpose[0]))
        print('Optimal Path:', last_row_transpose[1][optimal_terminal_state])

if __name__ == "__main__":
    pi = np.array([.25, .25, .25, .25])
    A = np.array([
        [0,  1,  0, 0],
        [.4, 0, .6, 0],
        [0, .4, 0, .6],
        [0, 0, .5, .5]])
    B = np.array([
        [.5, .5],
        [.3, .7],
        [.6, .4],
        [.8, .2]])
    HMM_instance = HMM(4, 2, A, B, pi)
    HMM_instance.generate_one_sample_path(20)
    HMM_instance.observe_sequence_probability_cal([0,0,1,1,0], use_forward = True)
    HMM_instance.observe_sequence_probability_cal([0,0,1,1,0], use_forward = False)
    HMM_instance.parameter_learning([0,1,0,1,0,1,0,1,0,1,0,1,0,1])
    HMM_instance.viterbi_dp([0,1,0,1,0,1,0,1])
