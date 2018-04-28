import numpy as np
from joblib import Parallel, delayed
import multiprocessing


class ADMM:
    def __init__(self):
        self.num_cores = multiprocessing.cpu_count()
        self.d = 20
        self.n = 100
        self.A = np.random.randn(self.n, self.d)
        self.b = np.random.randn(self.n, 1)
        self.X_t = np.random.randn(self.d, 1)
        # Modified X_t for threading
        self.X_t = np.tile(self.X_t.T, (self.n,1))
        # print("X_t.shape:", self.X_t.shape
        # z_t = np.random.randn(d, 1)
        # X_t = np.zeros((d,1))
        self.z_t = np.zeros((self.d,1))
        self.rho = 1
        # nu_t = np.random.randn(d, 1)
        self.nu_t = np.zeros((self.d,1))
        self.nu_t = np.tile(self.nu_t.T, (self.n,1))
        # print("nu_t.shape:", nu_t.shape
        self.num_iterations = 100
        self.lamb = 0.1


    def coordinate_descent(self, z, x, nu, rho, lamb):
        e_grad = x + nu*1.0/rho
        # Regularization term gradient
        # This will have a subgradient, with values as -lambda/rho, lambda/rho OR 0

        # print("prev",z)
        z_t = np.zeros_like(z)

        filter_less = -(1.0*lamb/rho)*(z<0)
        # print("less",filter_less)
        filter_greater = (1.0*lamb/rho)*(z>0)
        # print("gt",filter_greater)

        z_t = e_grad - filter_less - filter_greater
        # print(z_t)
        return(z_t)


    def x_update(self, i):

        # ai = self.A[i].reshape(-1,1).T
        # # print("ai.shape",ai.shape)
        # bi = self.b[i].reshape(-1,1)
        # bi = np.asscalar(bi)
        # nui = nu_t[i].reshape(-1,1)
        # # print(ai.shape)
        # # print("nui.shape",nui.shape)
        # term1 = 1/(ai.dot(ai.T) + self.rho)
        # # print(term1.shape)
        # term2 = ai.T*bi + self.rho*z_t - nui
        # # print(term2.shape)
        # term1 = np.asscalar(term1)
        # x = term1*term2
        # x = x.T
        # # print("x",x.shape)
        # self.X_t[i] = x
        # def processInput(i):
        print(i * i)


    def main(self):
        # print(A.shape,b.shape,X_t.shape,z_t.shape,rho,nu_t.shape)
        # Initializations
        X = np.mean(self.X_t, axis=0).reshape(-1,1)
        val = 0.5*np.linalg.norm(self.A.dot(X) - self.b, ord='fro')**2 + self.lamb*np.linalg.norm(X, ord=1)
        print("Initial value:",val)
        for iter in range(self.num_iterations):

            # STEP 1: Calculate X_t
            # This has a closed form solution
            # Thread this later

            # for i in range(len(self.A)):
            #
            #     ai = self.A[i].reshape(-1,1).T
            #     # print("ai.shape",ai.shape)
            #     bi = self.b[i].reshape(-1,1)
            #     bi = np.asscalar(bi)
            #     nui = self.nu_t[i].reshape(-1,1)
            #     # print(ai.shape)
            #     # print("nui.shape",nui.shape)
            #     term1 = 1/(ai.dot(ai.T) + self.rho)
            #     # print(term1.shape)
            #     term2 = ai.T*bi + self.rho*self.z_t - nui
            #     # print(term2.shape)
            #     term1 = np.asscalar(term1)
            #     x = term1*term2
            #     x = x.T
            #     # print("x",x.shape)
            #     self.X_t[i] = x

            Parallel(n_jobs=self.num_cores)(delayed(self.x_update)(i) for i in range(len(self.A)))
            # Parallel(n_jobs=self.num_cores)(delayed(self.x_update)(i) for i in range(10))

            # print("X_t.shape",X_t.shape)
            X = np.mean(self.X_t, axis=0).reshape(-1,1)
            # print("X.shape",X.shape)
            nu = np.mean(self.nu_t, axis=0).reshape(-1,1)
            # print("nu.shape",nu.shape)
            # STEP 2: Calculate z_t
            # Taking the prox, we get the lasso problem again, so, using coordinate_descent
            self.z_t = self.coordinate_descent(self.z_t, X, nu, self.rho, self.lamb)
            # print("z_t.shape",z_t.shape)

            # STEP 3: Update nu_t
            # print("nu_t.shape",nu_t.shape)
            for i in range(self.nu_t.shape[0]):
                # print("X_t[i].reshape(-1,1).shape",X_t[i].reshape(-1,1).shape)
                # print("z_t.shape",z_t.shape)
                self.nu_t[i] = (self.nu_t[i].reshape(-1,1) + self.rho*(self.X_t[i].reshape(-1,1) - self.z_t)).T
            # print("nu_t.shape",nu_t.shape)

        #     nu_t = nu_t + rho*(X_t - z_t)
            val = 0.5*np.linalg.norm(self.A.dot(X) - self.b, ord='fro')**2 + self.lamb*np.linalg.norm(X, ord=1)
            print(val)
        #
        val = 0.5*np.linalg.norm(self.A.dot(X) - self.b, ord='fro')**2 + self.lamb*np.linalg.norm(X, ord=1)
        print(val)

obj = ADMM()
obj.main()
