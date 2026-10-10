"""Training-only class-weight sensitivity for the same multinomial transition model."""
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize

def class_weights(y,power=1.):
    classes,counts=np.unique(y,return_counts=True)
    raw=(len(y)/(len(classes)*counts))**power
    normalizer=float(np.sum(raw*counts)/len(y));raw/=normalizer
    return {int(c):float(w) for c,w in zip(classes,raw)}

def weighted_objective(theta,X,y,k,weights):
    b=theta.reshape(k-1,X.shape[1]);logits=np.c_[X@b.T,np.zeros(len(X))];logs=logits-logsumexp(logits,axis=1)[:,None]
    residual=np.exp(logs)[:,:k-1]
    for j in range(k-1):residual[:,j]-=(y==j)
    return float(-np.dot(weights,logs[np.arange(len(y)),y])),((weights[:,None]*residual).T@X).ravel()

def fit_weighted(frame,base,power):
    features=base['features'];means=base['means'];scales=base['scales']
    X=np.c_[np.ones(len(frame)),np.column_stack([(frame[f]-means[f])/scales[f] for f in features])]
    classes=np.array(base['classes']);y=np.searchsorted(classes,frame.next_state);k=len(classes)
    cw=class_weights(frame.next_state.to_numpy(),power);weights=np.array([cw[int(v)] for v in frame.next_state])
    initial=np.zeros((k-1,X.shape[1]));freq=np.bincount(y,weights=weights,minlength=k)/weights.sum();initial[:,0]=np.log(freq[:-1]/freq[-1])
    opt=minimize(weighted_objective,initial.ravel(),args=(X,y,k,weights),jac=True,method='L-BFGS-B',options={'maxiter':5000,'maxfun':20000,'ftol':1e-14,'gtol':1e-7,'maxls':50})
    loss,grad=weighted_objective(opt.x,X,y,k,weights);b=opt.x.reshape(k-1,X.shape[1]);logits=np.c_[X@b.T,np.zeros(len(X))];p=np.exp(logits-logsumexp(logits,axis=1)[:,None])
    H=np.block([[X.T@((weights*p[:,a]*((a==b)-p[:,b]))[:,None]*X) for b in range(k-1)] for a in range(k-1)]);eigen=np.linalg.eigvalsh(H);gnorm=float(np.abs(grad).max())
    success=bool(opt.success and np.isfinite(opt.x).all() and gnorm<=1e-3 and eigen.min()>0)
    diagnostics={'success':success,'optimizer_success':bool(opt.success),'message':str(opt.message),'iterations':int(opt.nit),'weighted_negative_log_likelihood':loss,'gradient_inf_norm':gnorm,'information_min_eigenvalue':float(eigen.min()),'information_condition':float(np.linalg.cond(H)),'weight_power':power,'class_weights':cw,'n_train':len(frame)}
    model=dict(base,coefficients_standardized=opt.x.reshape(k-1,X.shape[1]).tolist(),weight_power=power,training_class_weights=cw,variant='class_weight_sensitivity')
    return model,diagnostics
