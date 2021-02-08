import numpy as np
from scipy.sparse import csc_matrix
from sklearn import svm
from celer.datasets import make_correlated_data
from sparse_ho.tests.cvxpylayer import ssvr_cvxpy
import cvxpy as cp
from sparse_ho.models import SimplexSVR as SSVR
from sparse_ho.algo.forward import get_beta_jac_iterdiff
from sparse_ho.algo.implicit_forward import get_beta_jac_fast_iterdiff
from sparse_ho.criterion import HeldOutMSE
from sparse_ho import Forward
from sparse_ho import Implicit
from sparse_ho import ImplicitForward
from sparse_ho.ho import grad_search
from sparse_ho.utils import Monitor

from sparse_ho.optimizers import LineSearch

n_samples = 500
n_features = 10


X, y, beta = make_correlated_data(
    n_samples, n_features, corr=0, snr=10, random_state=42)
X_s = csc_matrix(X)
beta[beta < 0] = 0
beta /= np.sum(beta)
y = X @ beta
idx_train = np.arange(0, 250)
idx_val = np.arange(250, 500)

tol = 1e-16

C = 0.00000000001
log_C = np.log(C)
epsilon = 0.1
log_epsilon = np.log(0.1)
max_iter = 50000

model = SSVR(max_iter=max_iter, estimator=None)
estimator = svm.LinearSVR(
    epsilon=np.exp(log_epsilon), tol=1e-16, C=np.exp(log_C),
    fit_intercept=False, max_iter=1000)
# model_custom = SVR(max_iter=max_iter, estimator=estimator)


def get_v(mask, dense):
    return 2 * (X[np.ix_(idx_val, mask)].T @ (
        X[np.ix_(idx_val, mask)] @ dense - y[idx_val])) / len(idx_val)


def test_beta_jac():
    supp1, dense1, jac1 = get_beta_jac_iterdiff(
        X[idx_train, :], y[idx_train], np.array([log_C, log_epsilon]),
        tol=tol, model=model, compute_jac=True, max_iter=max_iter)

    sol = primal_eps_SVR_constrained(X[idx_train, :],
                                     y[idx_train], cost=C,
                                     eps=np.exp(log_epsilon))
    supp2, dense2, jac2 = get_beta_jac_fast_iterdiff(
        X[idx_train, :], y[idx_train], np.array([log_C, log_epsilon]),
        tol=tol, model=model, tol_jac=1e-8, max_iter=max_iter,
        niter_jac=10000, use_stop_crit=True)
    assert np.allclose(dense1, sol[sol != 0])
    assert np.all(supp1 == supp2)
    assert np.allclose(dense1, dense2)


def test_val_grad():
    #######################################################################
    # Not all methods computes the full Jacobian, but all
    # compute the gradients
    # check that the gradient returned by all methods are the same
    criterion = HeldOutMSE(idx_train, idx_val)
    algo = Forward()
    val_fwd, grad_fwd = criterion.get_val_grad(
        model, X, y, np.array([log_C, log_epsilon]), algo.get_beta_jac_v,
        tol=tol, max_iter=max_iter)

    criterion = HeldOutMSE(idx_train, idx_val)
    algo = ImplicitForward(tol_jac=1e-16, n_iter_jac=10000)
    val_imp_fwd, grad_imp_fwd = criterion.get_val_grad(
        model, X, y, np.array([log_C, log_epsilon]), algo.get_beta_jac_v,
        tol=tol, max_iter=max_iter)

    criterion = HeldOutMSE(idx_train, idx_val)
    algo = ImplicitForward(tol_jac=1e-16, n_iter_jac=10000)
    val_imp_fwd_custom, grad_imp_fwd_custom = criterion.get_val_grad(
        model, X, y, np.array([log_C, log_epsilon]), algo.get_beta_jac_v,
        tol=tol, max_iter=max_iter)
    criterion = HeldOutMSE(idx_train, idx_val)
    algo = Implicit()
    val_imp, grad_imp = criterion.get_val_grad(
        model, X, y, np.array([log_C, log_epsilon]),
        algo.get_beta_jac_v, tol=tol)
    
    val_cvxpy, grad_cvxpy = ssvr_cvxpy(X, y, np.array([C, epsilon]), idx_train, idx_val)
    grad_cvxpy *= np.array([C, epsilon])
    import ipdb; ipdb.set_trace()
    assert np.allclose(val_fwd, val_imp_fwd)
    assert np.allclose(grad_fwd, grad_imp_fwd)
    np.testing.assert_allclose(val_imp_fwd, val_cvxpy)
    assert np.allclose(val_imp_fwd, val_imp_fwd_custom)
    # for the implcit the conjugate grad does not converge
    # hence the rtol=1e-2
    np.testing.assert_allclose(grad_imp_fwd, grad_cvxpy)
    assert np.allclose(grad_imp_fwd, grad_imp_fwd_custom)


def test_grad_search():

    n_outer = 3
    criterion = HeldOutMSE(idx_train, idx_val)
    monitor1 = Monitor()
    algo = Forward()
    optimizer = LineSearch(n_outer=n_outer, tol=1e-16)
    grad_search(
        algo, criterion, model, optimizer, X, y,
        np.array([C, epsilon]), monitor1)

    criterion = HeldOutMSE(idx_train, idx_val)
    monitor2 = Monitor()
    algo = Implicit()
    optimizer = LineSearch(n_outer=n_outer, tol=1e-16)
    grad_search(
        algo, criterion, model, optimizer, X, y, np.array(
            [C, epsilon]), monitor2)

    criterion = HeldOutMSE(idx_train, idx_val)
    monitor3 = Monitor()
    algo = ImplicitForward(tol_jac=1e-10, n_iter_jac=10000,
    use_stop_crit=True)
    optimizer = LineSearch(n_outer=n_outer, tol=1e-16)
    grad_search(
        algo, criterion, model, optimizer, X, y,
        np.array([C, epsilon]), monitor3)
    [np.linalg.norm(grad) for grad in monitor1.grads]
    [np.exp(alpha) for alpha in monitor1.alphas]

    assert np.allclose(
        np.array(monitor1.alphas), np.array(monitor3.alphas))
    assert np.allclose(
        np.array(monitor1.grads), np.array(monitor3.grads), rtol=1e-6)
    assert np.allclose(
        np.array(monitor1.objs), np.array(monitor3.objs), rtol=1e-6)
    assert not np.allclose(
        np.array(monitor1.times), np.array(monitor3.times))
    np.testing.assert_allclose(
        np.array(monitor1.alphas), np.array(monitor2.alphas),
        atol=1e-2)
    np.testing.assert_allclose(
        np.array(monitor1.grads), np.array(monitor2.grads), atol=1e-2)
    np.testing.assert_allclose(
        np.array(monitor1.objs), np.array(monitor2.objs), atol=1e-2)
    assert not np.allclose(
        np.array(monitor1.times), np.array(monitor2.times))


def primal_eps_SVR_constrained(X, y, cost=1, eps=0.5):
    n_samples_train, n_features = X.shape

    # set up variables and parameters
    beta_cp = cp.Variable(n_features)
    xi_cp = cp.Variable(n_samples_train)
    xi_star_cp = cp.Variable(n_samples_train)

    # set up objective
    loss = cp.sum_squares(beta_cp) / 2
    reg = cost * cp.sum(xi_cp + xi_star_cp)
    objective = loss + reg
    # define constraints
    constraints = [y - X @ beta_cp <= eps + xi_cp,
                   X @ beta_cp - y <= eps + xi_star_cp,
                   xi_cp >= 0.0, xi_star_cp >= 0.0,
                   cp.sum(beta_cp) == 1, beta_cp >= 0.0]
    # define problem
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(max_iter=100000, eps_abs=1e-12, eps_rel=1e-12, verbose=True)
    return beta_cp.value


if __name__ == '__main__':
    # test_beta_jac()
    test_val_grad()
    # test_grad_search()
