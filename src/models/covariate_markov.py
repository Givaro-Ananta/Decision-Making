"""Unpenalized, reference-category multinomial Markov transition likelihood."""
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize

def objective(theta,X,y,k):
    coefficients=theta.reshape(k-1,X.shape[1])
    logits=np.c_[X@coefficients.T,np.zeros(len(X))]
    logs=logits-logsumexp(logits,axis=1)[:,None]
    probabilities=np.exp(logs)
    residual=probabilities[:,:k-1].copy()
    for j in range(k-1): residual[:,j]-=(y==j)
    return -logs[np.arange(len(X)),y].sum(),(residual.T@X).ravel()

def fit_transition(frame,features,means,scales):
    X=np.c_[np.ones(len(frame)),np.column_stack([(frame[f].to_numpy()-means[f])/scales[f] for f in features])]
    classes=np.sort(frame.next_state.unique()).astype(int); y=np.searchsorted(classes,frame.next_state)
    if len(classes)==1: return {'type':'deterministic','next_state':int(classes[0])},{'success':True,'iterations':0}
    k=len(classes); initial=np.zeros((k-1,X.shape[1])); frequency=np.bincount(y,minlength=k)/len(y); initial[:,0]=np.log(frequency[:-1]/frequency[-1])
    result=minimize(objective,initial.ravel(),args=(X,y,k),jac=True,method='L-BFGS-B',options={'maxiter':5000,'maxfun':20000,'ftol':1e-14,'gtol':1e-7,'maxls':50})
    nll,gradient=objective(result.x,X,y,k)
    logits=np.c_[X@result.x.reshape(k-1,X.shape[1]).T,np.zeros(len(X))]
    probabilities=np.exp(logits-logsumexp(logits,axis=1)[:,None])
    hessian=np.block([[X.T@((probabilities[:,a]*((a==b)-probabilities[:,b]))[:,None]*X) for b in range(k-1)] for a in range(k-1)])
    eigen=np.linalg.eigvalsh(hessian); grad=float(np.max(np.abs(gradient)))
    success=bool(result.success and np.isfinite(result.x).all() and grad<=1e-3 and eigen.min()>0)
    diag={'success':success,'optimizer_success':bool(result.success),'message':str(result.message),'iterations':int(result.nit),'negative_log_likelihood':float(nll),'gradient_inf_norm':grad,'information_min_eigenvalue':float(eigen.min()),'information_condition':float(np.linalg.cond(hessian)),'max_abs_standardized_coefficient':float(np.abs(result.x).max())}
    model={'type':'multinomial_logistic','classes':classes.tolist(),'reference_state':int(classes[-1]),'features':features,'means':means,'scales':scales,'coefficients_standardized':result.x.reshape(k-1,X.shape[1]).tolist(),'penalty':None}
    return model,diag

def predict_log_proba(model,frame):
    output=np.full((len(frame),4),-np.inf)
    if model['type']=='deterministic': output[:,model['next_state']]=0; return output
    X=np.c_[np.ones(len(frame)),np.column_stack([(frame[f].to_numpy()-model['means'][f])/model['scales'][f] for f in model['features']])]
    logits=np.c_[X@np.asarray(model['coefficients_standardized']).T,np.zeros(len(frame))]
    output[:,model['classes']]=logits-logsumexp(logits,axis=1)[:,None]
    return output
