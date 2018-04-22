import numpy as np

def coordinate_descent(z, x, nu, rho, lamb):
    e_grad = x + nu*1.0/rho
    rho = 1
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



d = 20
n = 100

A = np.random.randn(n, d)
b = np.random.randn(n, 1)

X_t = np.random.randn(d, 1)
# z_t = np.random.randn(d, 1)
# X_t = np.zeros((d,1))
z_t = np.zeros((d,1))

rho = 1

# nu_t = np.random.randn(d, 1)
nu_t = np.zeros((d,1))

num_iterations = 10

print(A.shape,b.shape,X_t.shape,z_t.shape,rho,nu_t.shape)
# Initializations
lamb = 0.1
val = 0.5*np.linalg.norm(A.dot(X_t) - b, ord='fro')**2 + lamb*np.linalg.norm(X_t, ord=1)
print(val)
for iter in range(num_iterations):

    # STEP 1: Calculate X_t
    # This has a closed form solution
    term1 = np.linalg.inv(A.T.dot(A) + rho)
    term2 = A.T.dot(b) + rho*z_t -  nu_t
    X_t = term1.dot(term2)
    # print(term1.shape, term2.shape, X_t.shape)

    # STEP 2: Calculate z_t
    # Taking the prox, we get the lasso problem again, so, using coordinate_descent
    lamb = 0.1
    z_t = coordinate_descent(z_t, X_t, nu_t, rho, lamb)

    # STEP 3: Update nu_t
    nu_t = nu_t + rho*(X_t - z_t)
    val = 0.5*np.linalg.norm(A.dot(X_t) - b, ord='fro')**2 + lamb*np.linalg.norm(X_t, ord=1)
    print(val)

val = 0.5*np.linalg.norm(A.dot(X_t) - b, ord='fro')**2 + lamb*np.linalg.norm(X_t, ord=1)
print(val)
