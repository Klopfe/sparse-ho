import itertools
import numpy as np
import sklearn

from libsvmdata import fetch_libsvm
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sparse_ho import ImplicitForward
from sparse_ho import grad_search, hyperopt_wrapper
from sparse_ho.models import SimplexSVR
from sparse_ho.criterion import HeldOutMSE, CrossVal
from sparse_ho.optimizers import LineSearch, GradientDescent
from sparse_ho.utils import Monitor
from sparse_ho.grid_search import grid_search
from sparse_ho.tests.cvxpylayer import ssvr_cvxpy

print(__doc__)

dataset = 'simu'
# dataset = 'simu'

if dataset != 'simu':
    X, y = fetch_libsvm(dataset)
    X = X.tocsr()
    X = X[:, 0:250]
else:
    n_samples = 200
    n_features = 40
    X, y, b0 = make_regression(shuffle=False, random_state=10, n_samples=n_samples,
                               n_features=n_features, n_informative=n_features//2, 
                               n_targets=1, coef=True)
    b0[b0<0]=0
    b0 = b0 / np.sum(b0)
    y = np.dot(X,b0)
    np.random.seed(0)
    y += np.random.randn(y.shape[0]) * 0.5
# custom = LinearRegression(fit_intercept=False)
# custom.fit(X, y)

# test = LinearSVR(epsilon=0.5, tol=0.0001, C=0.1,
#                  fit_intercept=False, max_iter=10000)
# test.fit(X, y)
# np.linalg.norm(X @ test.coef_ - y) / X.shape[0]
# import ipdb; ipdb.set_trace()
n_samples = len(y)
idx_train = np.arange(0, n_samples)
idx_val = np.arange(0, n_samples)

tol = 1e-6

# algorithms = ['grad_search', 'grid_search10', 'grid_search']

algorithms = ['grad_search', 'grid_search10', 'grid_search']

max_evals = 25
print("Starting path computation...")
for algorithm in algorithms:
    # estimator = LinearSVR(
    #     fit_intercept=False, max_iter=50_000, tol=tol)

    print('%s started' % algorithm)

    model = SimplexSVR()
    criterion = HeldOutMSE(idx_train, idx_val)
    C0 = 2e-5
    epsilon0 = 1
    monitor = Monitor()
    # cross_val_criterion = CrossVal(criterion, cv=kf)
    algo = ImplicitForward()
    # optimizer = LineSearch(n_outer=10, tol=tol, verbose=True)
    if algorithm.startswith('grad_search'):
        if algorithm == 'grad_search':
            optimizer = GradientDescent(
                n_outer=max_evals, tol=tol, verbose=True, p_grad0=1.3)
        else:
            optimizer = LineSearch(n_outer=50, verbose=True, tol=tol)
        grad_search(
            algo, criterion, model, optimizer, X, y, np.array([C0, epsilon0]),
            monitor)

    elif algorithm.startswith('grid_search'):
        if algorithm == 'grid_search10':
            n_alphas = 5
        else:
            n_alphas = 30
        Cs = np.geomspace(2e-7, 2e4, n_alphas)
        epsilons = np.geomspace(2e-7, 2e1, n_alphas)

        grid_alphas = [i for i in itertools.product(Cs, epsilons)]

        grid_search(
            criterion, model, X, y, None, None, monitor,
            alphas=grid_alphas)
    else:
        hyperopt_wrapper(
            algo, criterion, model, X, y, log_alpha_min,
            log_alpha_max, monitor, max_evals=max_evals,
            method=algorithm, size_space=2)
    objs = np.array(monitor.objs)
    alphas = np.log(np.array(monitor.alphas))
    import ipdb; ipdb.set_trace()
    np.save("results/%s_log_alphas_%s_ssvr" % (dataset, algorithm), alphas)
    np.save("results/%s_objs_%s_ssvr" % (dataset, algorithm), objs)
    print('%s finished' % algorithm)



