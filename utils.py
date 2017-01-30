import chainer.functions as F
import numpy as np
from chainer import cuda

def broadcast_second_axis(v,n):
    #v.shape(batchsize,b) --> (batchsize,n,b)
    batchsize = v.data.shape[0]
    b = v.data.shape[1]
    vt = F.transpose(v,axes=(1,0))
    vt_broadcasted = F.broadcast_to(vt,(n,b,batchsize))
    return F.transpose(vt_broadcasted,axes=(2,0,1))

def broadcast_third_axis(v,n):
    #v.shape(batchsize,m) --> (batchsize,m,n)
    #v.shape(batchsize,m,1) --> (batchsize,m,n)
    if v.data.ndim == 2:
        batchsize = v.data.shape[0]
        v_broadcasted = F.broadcast_to(v,[n]+list(v.data.shape))
        return F.transpose(v_broadcasted,axes=(1,2,0))
    elif v.data.ndim == 3:
        assert v.data.shape[2] == 1
        batchsize = v.data.shape[0]
        m = v.data.shape[1]
        vt = F.transpose(v,axes=(2,0,1))
        vt_b = F.broadcast_to(vt,(n,batchsize,m))
        return F.transpose(vt_b,axes=(1,2,0))

def filterbank_matrix(mu,sig2,A,use_gpu=False):
    if use_gpu:
        xp = cuda.cupy
    else:
        xp = np

    #mu ,(batchsize,N,1)
    #sig2,(batchsize,N,1)
    #A: scalar
    
    batchsize, N = mu.data.shape[:2]
    #print "batchsize :{}, N :{}, A:{}".format(batchsize,N,A)
    eps = xp.array([1e-4]).astype(np.float32)
    eps = xp.broadcast_to(eps,(batchsize,N,A))

    mu_br = broadcast_third_axis(mu,A)
    sig2_br = broadcast_third_axis(sig2,A)
    a = xp.broadcast_to(xp.arange(A).astype(np.float32),(batchsize,N,A))
    exp_arg = -(mu_br-a)**2/(2*sig2_br)
    filt = F.exp(F.clip(exp_arg,-50.0,10.0))
    #filt = F.exp(exp_arg)

    #tmax = np.max(cuda.to_cpu(exp_arg.data))
    #tmin = np.min(cuda.to_cpu(exp_arg.data))
    #print "tmax is {}".format(tmax)
    #print "tmin is {}".format(tmin)
    #tmax = np.max(cuda.to_cpu(clips.data))
    #print "clip is {}".format(tmax)

    normalize_factor = broadcast_third_axis(F.sum(filt,axis=2),A)
    #tmax = np.max(cuda.to_cpu(normalize_factor.data))
    #if tmax > 0:
    #print "normalize is {}".format(tmax)

    #return filt/F.maximum(normalize_factor,eps)
    return filt/normalize_factor