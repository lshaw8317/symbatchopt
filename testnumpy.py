import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from scipy.special import expit as npexpit
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
        # shape=tuple([n_paths]+[1 for i in data.shape])
        # self.datasource = data[None,...].repeat(shape)

        self.datasource = data[None,...].repeat(repeats=n_paths,axis=0)
        self.K=min(K,self.length)
        print(f'Set K to {self.K}')
        self.bs=int(self.length/K) + 1*(self.length%K!=0)
        self.index=0
        self.n_paths=n_paths
        self.strat=None
        self.sample= self.NoSampler
    
    def redraw(self):
        d=self.data[np.argsort(np.random.rand(*(self.n_paths,self.length)), axis=-1)]
        if self.strat=='SMS':
            self.datasource=np.concatenate((d,np.flip(d,axis=(1,))),axis=1)
        else:
            self.datasource=d

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
            self.redraw()
            self.sample=self.SOsampler
        else:
            print('RM selected')
            self.strat='RM'
            self.sample=self.RMsampler
            
    def RRsampler(self):
        if self.index==0:
            self.redraw()
        data=self.datasource[:,self.index*self.bs:(self.index+1)*self.bs]
        self.index=(self.index+1)%self.K 
        return data
    
    def SOsampler(self):
        data=self.datasource[:,self.index*self.bs:(self.index+1)*self.bs]
        self.index=(self.index+1)%self.K 
        return data
    
    def SMSsampler(self):
        if self.index==0:
            self.redraw()
            idx=self.bs if self.length%self.bs==0 else self.length%self.bs
            data=self.datasource[:,:idx]
            self.datasource=self.datasource[:,idx:]
        else:
            data=self.datasource[:,(self.index-1)*self.bs:self.index*self.bs]
        self.index=(self.index+1)%(2*self.K)
        return data

    def RMsampler(self):
        if self.index==0:
            self.redraw()
        k_=np.random.randint(low=0,high=self.length)
        self.index=(self.index+1)%self.K
        inds=np.arange(k_,k_+self.bs)%self.length
        data=self.datasource[:,inds]
        return data
    
    def NoSampler(self):
        raise Exception('Sampling strategy not been defined!')

class Loss:
    def __init__(self,data,K,n_paths,strat='RM',lam=None):
        self.x, self.y = data
        self.n = int(self.x.shape[0])
        # Add dummy for bias
        self.xnew = np.concatenate((self.x, np.ones((self.n, 1))), axis=1)
        # self.xnew=self.x
        if not lam:
            self.lam = 0
            L = self.smoothness()
            self.lam = L / np.sqrt(self.n)
        else:
            self.lam=0
        self.MAP = self.calc_MAP()
        data_comb = np.concatenate((self.xnew, self.y[..., None]), axis=-1)
        self.mybatcher = MyBatcher(data=data_comb, K=K, n_paths=n_paths, strat=strat)

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
        # L=loss.smoothness()
        # kappa=L/loss.lam
        self.gamma=1. #-np.sqrt(L)*np.log((np.sqrt(kappa)-1) / (np.sqrt(kappa)+1))
        self.alpha=lambda h: np.exp(-self.gamma*h)
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
        elif self.method=='aoboa':
            self.stepper=self.AOBOA
        elif self.method=='heavyball':
            self.stepper=self.HeavyBall
        elif method.lower()=='ubu':
        #     self.beta= lambda h: (1./self.alpha(h)-1.)/self.gamma
            self.stepper=self.UBU
        #     self.preprocessor =self.ubu_pre
        #     self.postprocessor = self.ubu_post
        else:
            raise ValueError('method arg to Optimizer class not recognised: sgd, nesterov, heavyball and ubu are only available methods.')
            
    def run(self,q0,h0,Niters,dss=False):
        q=np.float64(q0.copy())
        v=np.zeros_like(q)
        epochs=Niters//self.loss.mybatcher.K
        epochs+=1*epochs%2 #need number of epochs to be even for SMS
        Niters=epochs*self.loss.mybatcher.K
        samples=np.zeros((Niters,*q.shape))
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

    
    def AOBOA(self,x,v,h): 
        '''
        Symmetric: AOBOA
        '''
        x+=h*v/2
        g=self.loss.stochgrad(x)
        v*=self.alpha(h/2)
        v-=h*g
        v*=self.alpha(h/2)
        x+=h*v/2
        return x,v
    
    def HeavyBall(self,x,v,h): 
        '''
        Not Symmetric: ABO
        '''
        
        g=self.loss.stochgrad(x)
        v*=self.alpha(h)
        v-=h*g
        x+=h*v

        return x,v
    
    def UBU(self,x,v,h): 
        '''
        Symmetric: UBU
        '''
        x+=(1-self.alpha(h/2))*v/self.gamma
        v*=self.alpha(h/2)
        g=self.loss.stochgrad(x)
        v-=h*g
        x+=(1-self.alpha(h/2))*v/self.gamma
        v*=self.alpha(h/2)
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
    
    def SGD(self,x,v,h):
        g=self.loss.stochgrad(x)
        x-=h*g
        return x,v

class LogReg(Loss):
    def __init__(self,data,K,n_paths):
        super().__init__(data,K,n_paths)

    
    def smoothness(self):
        covariance = self.xnew.T@self.xnew/self.n
        return 0.25*np.max(np.linalg.eigvalsh(covariance)) + self.lam
    
    def U(self,q):
        arg=self.xnew@q
        ans=-np.sum(self.y[None,...,None]*arg)
        ans+=np.sum(np.logaddexp(np.zeros_like(arg),arg))
        term=q*self.Cinv[None,...]@q
        return .5*np.sum(term)+ans
    
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
        arg=self.xnew@q
        temp=(self.y-npexpit(arg))
        return term-self.xnew.T@temp/self.n
    
    def grad(self,q,data):
       x,y=data[...,:-1],data[...,-1] #x has shape (n_paths,n,n_features)
       term=q*self.lam #q has shape (n_paths,n_features,1)
       arg=np.matmul(x,q) #has shape (n_paths,n,1)
       temp=y[...,None]-npexpit(arg) #has shape (n_paths,n,1)
       bs=x.shape[1] #self.n divide term/self.mybatcher.K for true splitting scheme
       return term-np.matmul(x.transpose(0,2,1),temp)/bs

class LinReg(Loss):
    def __init__(self,data,K,n_paths,strat='RM'):
        super().__init__(data,K,n_paths,strat='RM')

    def calc_MAP(self):
        return np.linalg.lstsq(self.xnew, self.y)[0]
    
    def grad(self,q,data):
       x,y=data[...,:-1],data[...,-1] #x has shape (n_paths,n,n_features)
       term=x@q #q has shape (n_paths,n_features,1)
       temp=term-y[...,None] #has shape (n_paths,n,1)
       term=x.transpose((0,2,1))@temp #q has shape (n_paths,n_features,1)
       bs=x.shape[1]
       return term/bs

def getbias(opt,h,Niters,polyak_average=False):
    opt.lr_decay_coef=0
    opt.lr_max=np.inf
    # shape=tuple([opt.loss.mybatcher.n_paths]+[1 for i in opt.loss.MAP.shape])
    # q0=opt.loss.MAP[None,...].repeat(shape)[...,None]
    q0 = opt.loss.MAP[None,...,None].repeat(repeats=opt.loss.mybatcher.n_paths,axis=0)

    s=[]
    plt.figure()
    K=opt.loss.mybatcher.K
    for h_,n in zip(h,Niters):
        temp=opt.run(q0, h_, Niters=n)
        if not polyak_average:
            s+=[temp[-1]]
        else:
            s+=[temp[2*K-1::2*K].mean(dim=0)]
        err=np.linalg.norm(temp[2*K-1::2*K]-q0[None,...],axis=(-1,-2)).mean(axis=-1)
        plt.semilogy(err)
    plt.title(f'{opt.method}-{opt.strat}')
    plt.savefig(f'ConvergenceCheck_{opt.method}_{opt.strat}.pdf',format='pdf')
    plt.close()
    return s

def getprogress(opt,h, Niters):
    #shape=tuple([opt.loss.mybatcher.n_paths]+[1 for i in opt.loss.MAP.shape])
    #q0=0.*opt.loss.MAP[None,...].repeat(shape).unsqueeze(-1)
    q0 = opt.loss.MAP[None,...,None].repeat(repeats=opt.loss.mybatcher.n_paths,axis=0)
    s=opt.run(q0, h, Niters=Niters,dss=True)
    return s

def plotBias(opt_dict,base=10):
    ##Plotting
    plt.figure(figsize=(3,2))
    K=opt_dict['K']
    etarange=opt_dict['etarange']
    xopt=np.float64(opt_dict['xopt'])
    lines={'SMS':'--','RR':'-','RM':':','SO':'-.'}
    # methods={'nesterov':'r','heavyball':'g','sgd':'b','euler':'c'}
    methodslist=list(opt_dict.keys())[-3:]
    colorlist=['r','g','b','c']
    for i,method in enumerate(methodslist):
        loc=opt_dict[method]
        color=colorlist[i]
        plt.loglog([],[],color=color,base=base,ls='-',label=method)
        for strat in loc.keys():
            ls=lines[strat]
            x=np.array(loc[strat])
            x=np.float64(x)
            err=np.linalg.norm(x.squeeze(-1)-xopt[None,None],axis=(-1)).mean(axis=-1)
            plt.loglog(etarange,err,color=color,base=base,ls=ls)
            a=np.vstack((np.log(etarange),np.ones(len(etarange)))).T
            slope,_=np.linalg.lstsq(a,np.log(err))[0]
            print(f'Order of {method}-{strat} = {round(slope,2)}')

    for strat in loc.keys():   
        ls=lines[strat]
        plt.loglog([],[],color='k',base=base,ls=ls,label=strat)

    plt.title(f'Bias: {exp} {expname}, $R={K}$')
    plt.xlabel('$h$')
    plt.ylabel('$\|x-X_*\|$')
    plt.legend()
    plt.savefig(os.path.join(figdir,f'Bias{exp}{expname}K{K}.pdf'),format='pdf',bbox_inches='tight')

def plotDSS(opt_dict):
    ##Plotting
    plt.figure(figsize=(3,2))
    K=opt_dict['K']
    xopt=opt_dict['xopt']
    lines={'SMS':'--','RR':'-','RM':':','SO':'-.'}
    for color,method in zip(['r','g','b'],['nesterov','heavyball','sgd']):
        loc=opt_dict[method]
        plt.semilogy([],[],color=color,base=2,ls='-',label=method)
        for strat in loc.keys():
            x=loc[strat]
            ls=lines[strat]
            err=np.linalg.norm(x.squeeze(-1)-xopt[None,None],axis=(-1)).mean(axis=-1)
            timerange=np.arange(len(err))/K
            newerr=np.min(err.reshape(-1,K),axis=1)
            plt.semilogy(timerange[::K],newerr,color=color,base=2,ls=ls)
         
    for strat in loc.keys():
        ls = lines[strat]
        plt.semilogy([],[],color='k',base=2,ls=ls,label=strat)
    
    plt.title(f'DSS: {exp} {expname}, $R={K}$')
    plt.xlabel('Epochs')
    plt.ylabel('$\|x-X_*\|$')
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
        #x,y=torch.tensor(x),torch.tensor(y)
    elif expname=='StatLog':
        data = pd.read_table(datadir+'/satTrn.txt', header=None, sep=' ')
        X = np.array(data.iloc[:,:-1])
        x = StandardScaler().fit_transform(X)
        y = np.array(data.iloc[:,-1])
        y=np.where(y==2,1,0)
        #x,y=torch.tensor(x),torch.tensor(y)
    elif expname=='CTG':
        ctg = pd.read_table(datadir+'/CTG.txt',header=0)
        X = np.array(ctg.iloc[:,:21])
        x = StandardScaler().fit_transform(X)
        y = np.array(ctg.iloc[:,-1])
        y=np.where(y>2,1,0)
        #x,y=torch.tensor(x),torch.tensor(y)
    elif expname=='SimData':
        try:
            with open(datadir+"/SimData.pkl", 'rb') as f:
                d=pickle.load(f)
                x=d['x']
                y=d['y']
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
            p_i=npexpit((xnew@params))
            y=np.random.binomial(1, p_i).flatten() # output data
            #x=torch.tensor(x)
            #y=torch.tensor(y)
            with open("SimData.pkl", 'wb') as f:
                pickle.dump({'x':x,'y':y,'params':params},f)
    elif expname=='SimpleData':
        try:
            with open(datadir+"/SimpleData.pkl", 'rb') as f:
                d=pickle.load(f)
                x=d['x']
                y=d['y']
        except:
            print('Generating simulated data for lin reg experiment.')
            np.random.seed(2024)
            # True parameters
            w_true = 2.0
            b_true = 0.1
            
            # Generate noisy dataset
            x = np.array([1, 2, 3, 4, 5], dtype=np.float64)[...,None]
            y = (w_true * x + b_true).flatten() + 0.2*np.random.randn(len(x))
    
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
expname='CTG'
loss=getExp(expname,K=8,exp=exp,n_paths=100)
etarange = 10.**np.arange(-3,0,step=0.5,dtype=np.float64)
K=loss.mybatcher.K
Niters=np.int32(np.maximum(5/(etarange),500))
Niters+=1*Niters%2
Niters*=K
strats=['SMS','RR','RM']
methods=['heavyball','nesterov','sgd']

#Get Bias
opt_dict={}
opt_dict['K']=K
opt_dict['etarange']=etarange
opt_dict['xopt']=loss.MAP
opt_dict['experiment']=exp+expname
for method in methods:
    methoddict={}
    for strat in strats:
        opt=Optimizer(loss,method=method,lr_decay_coef=0,strat=strat)
        bias=getbias(opt,etarange,Niters,polyak_average=False) 
        methoddict[strat]=bias
    opt_dict[method]=methoddict
    
with open(resultsdir+f"/Bias{exp}{expname}_K{K}.pkl", 'wb') as f:
     pickle.dump(opt_dict,f)
with open(resultsdir+f"/Bias{exp}{expname}_K{K}.pkl", 'rb') as f:
    opt_dict=pickle.load(f)
plotBias(opt_dict)
#%%
#Get Progress with decreasing stepsize
exp='LogReg'
expname='StatLog'
loss=getExp(expname,K=8,exp=exp,n_paths=100)
K=loss.mybatcher.K
strats=['SMS','RR','RM']
methods=['heavyball','nesterov','sgd']
factor={'CTG':4, 'SimData':3, 'Chess':7, 'StatLog':6}[expname]
opt_dict={}
K=loss.mybatcher.K
opt_dict['K']= K
opt_dict['experiment']=exp+expname
l2=loss.smoothness()
lr0 = 1 / l2
opt_dict['lr0']=lr0
opt_dict['xopt']=loss.MAP
opt_dict['smoothness']=l2
Niters=1000*K
for method in methods:
    methoddict={}
    for strat in strats:
        opt=Optimizer(loss,method=method,lr_decay_coef=l2/K/factor,strat=strat,it_start_decay=20*K)
        prog=getprogress(opt,lr0,Niters) 
        methoddict[strat]=prog
    opt_dict[method]=methoddict
with open(resultsdir+f"/DSS{exp}{expname}_K{K}.pkl", 'wb') as f:
    pickle.dump(opt_dict,f)
with open(resultsdir+f"/DSS{exp}{expname}_K{K}.pkl", 'rb') as f:
    opt_dict=pickle.load(f)
    
plotDSS(opt_dict)
