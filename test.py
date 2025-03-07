import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from torch import cholesky_solve as cho_solve
from torch.linalg import solve_triangular, cholesky
import os
from scipy.optimize import fsolve,minimize
from torch import sigmoid as expit
from scipy.special import expit as npexpit
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
import pickle


plt.rcParams.update({'text.usetex':True,'font.serif': ['cm'],'font.size':16})
plt.rcParams['figure.dpi'] = 1000
plt.rcParams['savefig.dpi'] = 1000
plt.rc('text', usetex=True)
plt.rc('font',**{'serif':['cm']})
plt.style.use('seaborn-v0_8-paper')
import time as time
figdir='plots'
resultsdir='RESULTS'
datadir='datasets'
#%%
class MyBatcher:
    def __init__(self,data,K,n_paths,strat=None):
        self.data=data
        self.length = len(data)
        shape=tuple([n_paths]+[1 for i in data.shape])
        self.datasource = data[None,...].repeat(shape)
        self.K=K
        self.bs=int(self.length/K) + 1*(self.length%K!=0)
        self.index=0
        self.n_paths=n_paths
        self.strat=None
        self.sample= self.NoSampler
    
    def set_strat(self,strat):
        self.index=0
        if strat=='RR':
            print('RR selected')
            self.strat='RR'
            self.sample=self.RRsampler
        elif strat=='SMS':
            print('SMS selected')
            self.strat='SMS'
            self.sample=self.SMSsampler
        elif strat=='SO':
            print('SO selected')
            self.strat='SO'
            self.datasource=self.data[torch.argsort(torch.rand(size=(self.n_paths,self.length)), dim=-1)]
            self.sample=self.SOsampler
        else:
            print('RM selected')
            self.strat='RM'
            self.sample=self.RMsampler
            
    def RRsampler(self):
        if self.index==0:
            self.datasource=self.data[torch.argsort(torch.rand(size=(self.n_paths,self.length)), dim=-1)]
        data=self.datasource[:,self.index*self.bs:(self.index+1)*self.bs]
        self.index=(self.index+1)%self.K 
        return data
    
    def SOsampler(self):
        data=self.datasource[:,self.index*self.bs:(self.index+1)*self.bs]
        self.index=(self.index+1)%self.K 
        return data
    
    def SMSsampler(self):
        if self.index==0:
            d=self.data[torch.argsort(torch.rand(size=(self.n_paths,self.length)), dim=-1)]
            self.datasource=torch.cat((d,torch.flip(d,dims=(1,))),dim=1)
            idx=self.bs if self.length%self.bs==0 else self.length%self.bs
            data=self.datasource[:,:idx]
            self.datasource=self.datasource[:,idx:]
        else:
            data=self.datasource[:,(self.index-1)*self.bs:self.index*self.bs]
        self.index=(self.index+1)%(2*self.K)
        return data

    def RMsampler(self):
        if self.index==0:
            self.datasource=self.data[torch.argsort(torch.rand(size=(self.n_paths,self.length)), dim=-1)]
        k_=np.random.randint(low=0,high=self.length)
        self.index=(self.index+1)%self.K
        inds=np.arange(k_,k_+self.bs)%self.length
        data=self.datasource[:,inds]
        return data
    
    def NoSampler(self):
        raise Exception('Sampling strategy not been defined!')

class Loss:
    def __init__(self,MAP,myb):
        self.MAP=MAP
        self.mybatcher=myb
    
    def set_strat(self,strat):
        if strat=='RR':
            self.mybatcher.set_strat('RR')
        elif strat=='SMS':
            self.mybatcher.set_strat('SMS')
        else:
            self.mybatcher.set_strat('RM')

    def NLogLoss(self,q):
        return
    
    def grad(self,q,data):
        return
    
    def stochgrad(self,q):
        data=self.mybatcher.sample()
        scaler=data.shape[1]/self.mybatcher.bs
        return self.grad(q,data)*scaler
    
    def calc_MAP(self,epochs):
        return
    
    def smoothness(self):
        return
        
class Optimizer:
    def __init__(self, loss, method='SGD',strat='RM',lr_decay_coef=0,lr_decay_power=1,it_start_decay=0,lr_max=np.inf):
        self.loss = loss
        L=loss.smoothness()
        kappa=L/loss.lam
        self.gamma=1. #-np.sqrt(L)*np.log((np.sqrt(kappa)-1) / (np.sqrt(kappa)+1))
        self.alpha=lambda h: torch.exp(-self.gamma*h)
        self.loss.set_strat(strat)
        self.strat=strat
        self.lr_decay_coef=lr_decay_coef
        self.lr_decay_power=lr_decay_power
        self.it_start_decay=it_start_decay
        self.lr_max=lr_max
        self.method=method.lower()
        # self.preprocessor = lambda q,v,h: (q,v)
        # self.postprocessor = lambda q,v,h: (q,v)
        if self.method=='sgd':
            self.stepper=self.SGD
        elif self.method=='nesterov':
            self.stepper=self.Nesterov
        elif self.method=='euler':
            self.stepper=self.Euler
            # self.preprocessor =self.nesterov_pre
            # self.postprocessor = self.nesterov_post
        elif self.method=='heavyball':
            self.stepper=self.HeavyBall
        # elif method.lower()=='ubu':
        #     self.beta= lambda h: (1./self.alpha(h)-1.)/self.gamma
        #     self.stepper=self.UBU
        #     self.preprocessor =self.ubu_pre
        #     self.postprocessor = self.ubu_post
        else:
            raise ValueError('method arg to Optimizer class not recognised: sgd, nesterov, heavyball and ubu are only available methods.')
            
    def run(self,q0,h0,Niters,dss=False):
        q=q0.clone().detach()
        v=torch.zeros_like(q)
        epochs=Niters//self.loss.mybatcher.K
        epochs+=1*epochs%2 #need number of epochs to be even for SMS
        Niters=epochs*self.loss.mybatcher.K
        samples=torch.zeros((Niters,*q.shape))
        if dss:
            hfunc = lambda n: 1. / (1./h0 + self.lr_decay_coef*max(0, n-self.it_start_decay)**self.lr_decay_power)
        else:
            hfunc = lambda n: h0
        for n in range(0,Niters):
            if n%(2*self.loss.mybatcher.K)==0: 
                h=min(hfunc(n),self.lr_max) #update stepsize every 2 epochs
            q,v=self.stepper(q,v,h)
            samples[n]=q
        return samples


    # def UBU(self,x,v,h,g):
    #     '''
    #     h=self.lr
    #     alpha=exp(-gamma*h)
    #     beta=(1/alpha-1)/gamma
    #     Preprocess: x0,v0=x0+(1-sqrt(alpha))/gamma*v0,v0*sqrt(alpha)
    #     Postprocess: xK,vK=xK-(1-sqrt(alpha))/gamma*vK/sqrt(alpha),vK/sqrt(alpha)
    #     '''
    #     v-=h*g
    #     v*=self.alpha(h)
    #     x+=self.beta(h)*v
    #     return x,v
    
    # def ubu_pre(self,x,v,h):
    #     x+=(1-torch.sqrt(self.alpha(h)))/self.gamma*v
    #     v*=torch.sqrt(self.alpha(h))
    #     return x,v
    # def ubu_post(self,x,v,h): 
    #     v/=torch.sqrt(self.alpha(h))
    #     x-=(1.-torch.sqrt(self.alpha(h)))/self.gamma*v
    #     return x,v
    
    def HeavyBall(self,x,v,h): 
        '''
        Symmetric
        '''
        x+=h*v/2
        g=self.loss.stochgrad(x)
        v*=self.alpha(h)
        v-=h*g
        x+=h*v/2
        return x,v
    
    def Euler(self,x,v,h): 
        g=self.loss.stochgrad(x)
        x+=h*v
        v*=(1.-self.gamma*h)
        v-=h*g
        return x,v

    def Nesterov(self,x,v,h): 
        '''
        Preprocess: x0,v0=x0+alpha*v0,v0*alpha
        Postprocess: xK,vK=xK-vK,vK/alpha
        '''
        v*=self.alpha(h)
        x+=h*v
        g=self.loss.stochgrad(x)
        v-=h*g
        x-=h**2*g
        return x,v
    
    # def nesterov_pre(self,x,v,h): 
    #     v*=self.alpha(h)
    #     x+=h*v
    #     return x,v
    # def nesterov_post(self,x,v,h): 
    #     x-=h*v
    #     v/=self.alpha(h)
    #     return x,v
    
    def SGD(self,x,v,h):
        g=self.loss.stochgrad(x)
        x-=h*g
        return x,v

class LogReg(Loss):
    def __init__(self,data,K,n_paths):
        self.x,self.y=data
        self.n=int(self.x.shape[0])
        #Add dummy for bias
        self.xnew=torch.cat((self.x,torch.ones((self.n,1))),dim=1)
        I=torch.eye(self.xnew.shape[1]).to(self.xnew.dtype)
        self.lam=0
        L=self.smoothness()
        self.lam=L/np.sqrt(self.n)
        self.C=I/self.lam
        self.Cinv=I*self.lam

        MAP=torch.tensor(self.calc_MAP())
        data_comb=torch.cat((self.xnew,self.y[...,None]),dim=-1)
        mybatcher=MyBatcher(data=data_comb,K=K,n_paths=n_paths)
        super().__init__(MAP,mybatcher)
    
    def smoothness(self):
        covariance = self.xnew.T@self.xnew/self.n
        return 0.25*np.max(np.linalg.eigvalsh(covariance)) + self.lam
    
    def U(self,q):
        arg=self.xnew@q
        ans=-torch.sum(self.y[None,...,None]*arg)
        ans+=torch.sum(torch.logaddexp(torch.zeros_like(arg),arg))
        term=q*torch.matmul(self.Cinv[None,...],q)
        return .5*torch.sum(term)+ans
    
    def calc_MAP(self):
        x=np.random.randn(*self.xnew.shape[1:])*.2
        lr=1/self.smoothness()
        kappa = (1/lr)/self.lam
        momentum = (np.sqrt(kappa)-1) / (np.sqrt(kappa)+1)
        x_nest = x.copy()
        history=[x]
        nepochs=600
        for i in range(nepochs):
            x_nest_old = x_nest.copy()
            g = self.fullgradient(x)
            x_nest = x - lr*g
            x = x_nest + momentum*(x_nest-x_nest_old)
            history+=[x]
        history=np.array(history)
        err=np.linalg.norm(history[:-1]-history[-1],axis=-1)
        plt.semilogy(np.arange(len(err)),err)
        plt.xlabel('Iterations')
        plt.ylabel('$\|x-x_*\|$')
        plt.title('Correctly found minimum with fullgrad Nesterov')
        return x
    
    def fullgradient(self,q): ##np version
        term=q*self.lam
        arg=self.xnew.numpy()@q
        temp=(self.y.numpy()-npexpit(arg))
        return term-self.xnew.numpy().T@temp/self.n
    
    def grad(self,q,data):
       x,y=data[...,:-1],data[...,-1] #x has shape (n_paths,n,n_features)
       term=torch.matmul(self.Cinv[None,...],q) #q has shape (n_paths,n_features,1)
       arg=torch.matmul(x,q) #has shape (n_paths,n,1)
       temp=y[...,None]-expit(arg) #has shape (n_paths,n,1)
       bs=x.shape[1] #self.n divide term/self.mybatcher.K for true splitting scheme
       return term-torch.matmul(x.permute(0,2,1),temp)/bs

class LinReg(Loss):
    def __init__(self,data,K,n_paths,strat='RM'):
        self.x,self.y=data
        self.n=int(self.x.shape[0])
        #Add dummy for bias
        self.xnew=torch.cat((self.x,torch.ones((self.n,1))),dim=1)
        # self.xnew=self.x
        MAP=self.calc_MAP()
        data_comb=torch.cat((self.xnew,self.y[...,None]),dim=-1)
        mybatcher=MyBatcher(data=data_comb,K=K,n_paths=n_paths,strat=strat)
        super().__init__(MAP,mybatcher)

    def calc_MAP(self):
        return torch.linalg.lstsq(self.xnew, self.y).solution
    
    def grad(self,q,data):
       x,y=data[...,:-1],data[...,-1] #x has shape (n_paths,n,n_features)
       term=torch.matmul(x,q) #q has shape (n_paths,n_features,1)
       temp=term-y[...,None] #has shape (n_paths,n,1)
       term=torch.matmul(x.transpose(1,2),temp) #q has shape (n_paths,n_features,1)
       bs=x.shape[1]
       return term/bs

def getbias(opt,h,Niters,polyak_average=False):
    opt.lr_decay_coef=0
    opt.lr_max=np.inf
    shape=tuple([opt.loss.mybatcher.n_paths]+[1 for i in opt.loss.MAP.shape])
    q0=opt.loss.MAP[None,...].repeat(shape).unsqueeze(-1)
    s=[]
    plt.figure()
    K=opt.loss.mybatcher.K
    for h_,n in zip(h,Niters):
        temp=opt.run(q0, h_, Niters=n)
        if polyak_average:
            s+=[temp[-1]]
        else:
            s+=[temp[2*K-1::2*K].mean(dim=0)]
        err=torch.linalg.norm(temp-q0[None,...],dim=(-1,-2)).mean(dim=-1)
        plt.semilogy(err)
    plt.title(f'{opt.method}-{opt.strat}')
    return s

def getprogress(opt,h, Niters):
    shape=tuple([opt.loss.mybatcher.n_paths]+[1 for i in opt.loss.MAP.shape])
    q0=0.*opt.loss.MAP[None,...].repeat(shape).unsqueeze(-1)
    s=opt.run(q0, h, Niters=Niters,dss=True)
    return s

def plotBias(opt_dict,base=10):
    ##Plotting
    plt.figure(figsize=(3,2))
    K=opt_dict['K']
    etarange=opt_dict['etarange']
    xopt=opt_dict['xopt']
    lines={'SMS':'--','RR':'-','RM':':','SO':'-.'}
    methods={'nesterov':'r','heavyball':'g','sgd':'b','euler':'c'}
    methodslist=list(opt_dict.keys())[3:]
    for method in methodslist:
        loc=opt_dict[method]
        color=methods[method]
        plt.loglog([],[],color=color,base=base,ls='-',label=method)
        for strat in loc.keys():
            ls=lines[strat]
            x=torch.stack(loc[strat]).squeeze(1)
            err=torch.linalg.norm(x.squeeze(-1)-xopt[None,None],dim=(-1)).mean(dim=-1)
            plt.loglog(etarange,err,color=color,base=base,ls=ls)

    for strat in loc.keys():   
        ls=lines[strat]
        plt.loglog([],[],color='k',base=base,ls=ls,label=strat)

    plt.title(f'Bias: {exp} {expname}, $R={K}$')
    plt.xlabel('$h$')
    plt.ylabel('$\|x-x_*\|$')
    plt.legend()
    plt.savefig(os.path.join(figdir,f'Bias{exp}{expname}K{K}.pdf'),format='pdf',bbox_inches='tight')

def plotDSS(opt_dict):
    ##Plotting
    plt.figure(figsize=(3,2))
    K=opt_dict['K']
    xopt=opt_dict['xopt']
    for color,method in zip(['r','g','b'],['nesterov','heavyball','sgd']):
        loc=opt_dict[method]
        plt.semilogy([],[],color=color,base=2,ls='-',label=method)
        for ls,strat in zip(['-','--',':'],loc.keys()):
            x=loc[strat]
            err=torch.linalg.norm(x.squeeze(-1)-xopt[None,None],dim=(-1)).mean(dim=-1)
            plt.semilogy(np.arange(len(err))/K,err,color=color,base=2,ls=ls)

    for ls,strat in zip(['-','--',':'],loc.keys()):   
        plt.semilogy([],[],color='k',base=2,ls=ls,label=strat)
    
    plt.title(f'DSS: {exp} {expname}, $R={K}$')
    plt.xlabel('Epochs')
    plt.ylabel('$\|x-x_*\|$')
    plt.legend()
    plt.savefig(os.path.join(figdir,f'DSS{exp}{expname}K{K}.pdf'),format='pdf',bbox_inches='tight')
    
def getExp(expname,K,n_paths=100,exp='LogReg'):
    
    if expname=='Chess':
        data = pd.read_table(datadir+'/chess.txt', sep=",", header=None)
        y = np.array(data.iloc[:,-1]=='won',dtype=np.float64)
        X = data.iloc[:,:-1]
        x = np.zeros_like(X,dtype=np.float64)
        for i in range(x.shape[-1]): 
            x[:,i] = pd.factorize(X.iloc[:,i],sort=True)[0]
        x,y=torch.tensor(x),torch.tensor(y)
    elif expname=='StatLog':
        data = pd.read_table(datadir+'/satTrn.txt', header=None, sep=' ')
        X = np.array(data.iloc[:,:-1])
        x = StandardScaler().fit_transform(X)
        y = np.array(data.iloc[:,-1])
        y=np.where(y==2,1,0)
        x,y=torch.tensor(x),torch.tensor(y)
    elif expname=='CTG':
        ctg = pd.read_table(datadir+'/CTG.txt',header=0)
        X = np.array(ctg.iloc[:,:21])
        x = StandardScaler().fit_transform(X)
        y = np.array(ctg.iloc[:,-1])
        y=np.where(y>2,1,0)
        x,y=torch.tensor(x),torch.tensor(y)
    elif expname=='SimData':
        try:
            with open(datadir+"/SimData.pkl", 'rb') as f:
                d=pickle.load(f)
                x=torch.tensor(d['x'])
                y=torch.tensor(d['y'])
        except:
            print('Generating simulated data for log reg experiment.')
            np.random.seed(2024)
            d=25
            p=d+1
            N=2**10
            scaler=np.hstack((5*np.ones(shape=(1,5)),np.ones(shape=(1,5)),.2*np.ones(shape=(1,d-10))))
            params=np.random.normal(size=(p,))
            x=np.random.normal(size=(N,d),scale=scaler) #input data
            xnew=np.hstack((np.ones(shape=(N,1)),x))
            p_i=expit(torch.tensor(xnew@params))
            y=np.random.binomial(1, p_i).flatten() # output data
            with open("SimData.pkl", 'wb') as f:
                pickle.dump({'x':torch.tensor(x),'y':torch.tensor(y),'params':params},f)
    elif expname=='SimpleData':
        try:
            with open(datadir+"/SimpleData.pkl", 'rb') as f:
                d=pickle.load(f)
                x=d['x']
                y=d['y']
        except:
            print('Generating simulated data for lin reg experiment.')
            torch.manual_seed(2024)
            # True parameters
            w_true = 2.0
            b_true = 0.1
            
            # Generate noisy dataset
            x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float64)[...,None]
            y = (w_true * x + b_true).flatten() + 0.2*torch.randn(size=(len(x),))

            with open(datadir+"/SimpleData.pkl", 'wb') as f:
                pickle.dump({'x':x,'y':y,'params':[b_true,w_true]},f)
    
    else:
        raise ValueError('expname not valid: choose one of StatLog,Chess,CTG,SimData.')
    
    N=len(x)
    if exp=='LogReg':
        loss=LogReg([x,y],K,n_paths=n_paths)
    elif exp=='LinReg':
        loss=LinReg([x,y],K,n_paths=n_paths)
    else:
        raise ValueError('Exp type not recognised.')
    return loss

#%%
exp='LogReg'
expname='SimData'
K=8
loss=getExp(expname,K,exp=exp,n_paths=100)
etarange = 2.**torch.arange(-13,-3)
K=loss.mybatcher.K
Niters=torch.tensor(np.maximum(10+.2/(etarange.numpy()),60)).to(torch.int32)
Niters+=1*Niters%2
Niters*=K
# Niters=50*K*torch.ones_like(etarange).to(torch.int32)
strats=['RR','SMS','RM']
methods=['heavyball','sgd','euler']

#Get Bias
opt_dict={}
opt_dict['K']=K
opt_dict['etarange']=etarange
opt_dict['xopt']=loss.MAP
for method in methods:
    methoddict={}
    for strat in strats:
        opt=Optimizer(loss,method=method,lr_decay_coef=0,strat=strat)
        bias=getbias(opt,etarange,Niters,polyak_average=True) 
        methoddict[strat]=bias
    opt_dict[method]=methoddict
    
with open(resultsdir+f"/Bias{exp}{expname}_K{K}.pkl", 'wb') as f:
    pickle.dump(opt_dict,f)
with open(resultsdir+f"/Bias{exp}{expname}_K{K}.pkl", 'rb') as f:
    opt_dict=pickle.load(f)

plotBias(opt_dict)

#Get Progress with decreasing stepsize
opt_dict={}
opt_dict['K']=K
l2=loss.smoothness()
lr0 = torch.tensor(1 / l2)
opt_dict['lr0']=lr0
opt_dict['xopt']=loss.MAP
opt_dict['smoothness']=l2
Niters=400*K
for method in methods:
    methoddict={}
    for strat in strats:
        opt=Optimizer(loss,method=method,lr_decay_coef=l2/K/3,strat=strat,it_start_decay=20*K)
        prog=getprogress(opt,lr0,Niters) 
        methoddict[strat]=prog
    opt_dict[method]=methoddict
with open(resultsdir+f"/DSS{exp}{expname}_K{K}.pkl", 'wb') as f:
    pickle.dump(opt_dict,f)
with open(resultsdir+f"/DSS{exp}{expname}_K{K}.pkl", 'rb') as f:
    opt_dict=pickle.load(f)
    
plotDSS(opt_dict)
