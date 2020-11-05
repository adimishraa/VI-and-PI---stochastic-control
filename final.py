from scipy.stats import multivariate_normal
import numpy as np
%matplotlib qt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import normalize
import time
from collections import Counter

#functions to measure speed
def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

"""
==================================
Inverted pendulum animation class
==================================

Adapted from the double pendulum problem animation.
https://matplotlib.org/examples/animation/double_pendulum_animated.html
"""


class EnvAnimate:
    '''
    Initialize Inverted Pendulum Animation Settings
    '''
    def __init__(self): 
        self.t=np.array([1,1,1,1])
        pass
        
    def load_random_test_trajectory(self,):
        # Random trajectory for example
        self.theta = np.linspace(-np.pi, np.pi, self.t.shape[0])
        self.u = np.zeros(self.t.shape[0])

        self.x1 = np.sin(self.theta)
        self.y1 = np.cos(self.theta)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,autoscale_on=False, xlim=(-2,2), ylim=(-2,2))
        self.ax.grid()
        self.ax.axis('equal')
        plt.axis([-2, 2, -2, 2])

        self.line, = self.ax.plot([],[], 'o-', lw=2)
        self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
        self.time_text = self.ax.text(0.05, 0.8, '', transform=self.ax.transAxes)
        pass
    
    '''
    Provide new rollout trajectory (theta and control input) to reanimate
    '''
    def load_trajectory(self, theta, u):
        """
        Once a trajectory is loaded, you can run start() to see the animation
        ----------
        theta : 1D numpy.ndarray
            The angular position of your pendulum (rad) at each time step
        u : 1D numpy.ndarray
            The control input at each time step
            
        Returns
        -------
        None
        """
        self.theta = theta
        self.x1 = np.sin(theta)
        self.y1 = np.cos(theta)
        self.u = u
        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,autoscale_on=False, xlim=(-2,2), ylim=(-2,2))
        self.ax.grid()
        self.ax.axis('equal')
        plt.axis([-2, 2,-2, 2])
        self.line, = self.ax.plot([],[], 'o-', lw=2)
        self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
        self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)
        pass
    
    # region: Animation
    # Feel free to edit (not necessarily)
    def init(self):
        self.line.set_data([], [])
        self.time_text.set_text('')
        return self.line, self.time_text

    def _update(self, i):
        thisx = [0, self.x1[i]]
        thisy = [0, self.y1[i]] 
        self.line.set_data(thisx, thisy)
        self.time_text.set_text(self.time_template % (self.t[i], self.theta[i], self.u[i]))
        return self.line, self.time_text

    def start(self):
        print('Starting Animation')
        print()
        # Set up plot to call animate() function periodically
        
        self.ani = FuncAnimation(self.fig, self._update, frames=range(len(self.x1)), interval=25, blit=True, init_func=self.init, repeat=False)
        plt.show()
    # endregion: Animation
class InvertedPendulum(EnvAnimate):
    def __init__(self):
        EnvAnimate.__init__(self,)
        # Change this to match your discretization
        # Usually, you will need to load parameters including
        # constants: dt, vmax, umax, n1, n2, nu
        # parameters: a, b, σ, k, r, γ
        self.a=0.1
        self.b=0.1
        self.k=0.1
        self.r=0.1
        self.gamma=0.1
        self.tau=10
        self.n1=10
        self.n2=10
        self.nu=10
        self.v_max=0.5
        self.u_max=0.1
        self.dt = 0.05
        self.sigma=np.array([0.01,0.01]).reshape((2,1))
        self.t = np.arange(0.0, 2.0, self.dt)
        self.load_random_test_trajectory()
        pass
        
    # TODO: Implement your own VI and PI algorithms
    # Feel free to edit/delete these functions
    def l_xu(self, x1, x2, u):
        # Stage cost
        sc=1-np.exp(self.k*np.cos(x1)-self.k)+(self.r/2)*u**2
        return sc*self.tau
    
    def f_xu(self, x1, x2, u):
        f=np.array([0,0])
        f[0]=x2
        f[1]=self.a*np.sin(x1)-self.b*x2+u
        # Motion model
        return f
    
    def value_iteration(self,):
        V_x, VI_policy = None, None
        return V_x, VI_policy,
    
    def policy_iteration(self,):
        V_x, PI_policy = None, None
        return V_x, PI_policy,
    
    def generate_trajectory(self, init_state, policy, t):
        theta, u = None, None
        return theta, u
    
def all_states(v_max,n1,n2):
    '''
    discretize the state space
    '''
    x1=np.linspace(-np.pi,np.pi,n1)
    x2=np.linspace(-v_max,v_max,n2)
    statelist=[]
    for x in x1:
        for y in x2:
            statelist.append((x,y))
    return np.array(statelist)

def l_xu(states_all,k,r,policy,control_space):
    '''
    returns stage cost
    '''
    return 1-np.exp(k*np.cos(states_all[:,0])-k)+(r/2)*control_space[policy]**2

def gaussian_f_xu(states,sigma,policy,control_space,a,b,tau):
    '''
    generate transition probabilities using gaussian
    '''
    f_xu=np.array([0,0])
    prob_list=[]
    for i,s in enumerate(states):
        f_xu[0]=s[1]
        f_xu[1]=a*np.sin(s[0])-b*s[1]+control_space[policy[i]]
        mu=s+f_xu*tau
        var=multivariate_normal(mu,sigma)
        rand_var=var.pdf(states)
        rand_var[np.where(rand_var<1e-5)]=0
        rand_var=rand_var/np.sum(rand_var)
        prob_list.append(rand_var)
    return np.array(prob_list)

def policy_improvement_pi(vpi,states,k,r,policy,control_space,a,b,tau,gamma):
    '''
    select the best policy for every state
    '''
    for i,s in enumerate(states):
        res_min=[]
        for j,p in enumerate(control_space):
           # print(act_trans[:,i,j].reshape((1,81)).shape)
            cost=l_xu(s.reshape((1,-1)),k,r,[j],control_space).reshape((-1,1))
            prb=gaussian_f_xu_1(s,sigma,p,a,b,tau,states) #generate transition probabilities
            res_min.append(cost*tau+gamma*np.matmul(prb,vpi))
        amin=np.argmin(np.array(res_min)) #find optimum policy
        #print(amax)
        policy[i]=amin
    return policy      

def policy_evaluation(gamma,prob,l_xu):
    '''
    returns Value for every state
    '''
    return np.matmul(np.linalg.inv(np.eye(prob.shape[0])-gamma*prob),l_xu)

def policy_iteration(gamma,a,b,k,r,tau,sigma,states,control_space):
    '''
    policy iteration algorithm
    '''
    policy=[2]*states.shape[0]
    for i in range(0,10):
        tr_prob=gaussian_f_xu(states,sigma,policy,control_space,a,b,tau)
        cost=l_xu(states,k,r,policy,control_space).reshape((-1,1))
        #print(i)
        t0=tic()
        vpi=policy_evaluation(gamma,tr_prob,cost*tau)
       # print('values')
       # print(vpi)
        toc(t0,'evaluation')
        t0=tic()
        policy_new=policy_improvement_pi(vpi,states,k,r,policy,control_space,a,b,tau,gamma)
        toc(t0,'improvement')
        #print('policy')
        #print(policy)
        policy_old=policy
        policy=policy_new
    return policy,policy_old,vpi

def policy_improvement(tau,a,b,k,r,gamma,sigma,states,control_space,vpi,policy):
    '''
    find the optimum policy for every state
    '''
    for i,s in enumerate(states):
        res_min=[]
        for u in range(control_space.shape[0]):
            prob=gaussian_f_xu_1(s,sigma,control_space[u],a,b,tau,states)
            cost=l_xu(s.reshape((1,-1)),k,r,[u],control_space).reshape((-1,1))
            val=cost*tau+gamma*np.matmul(prob,vpi) 
            res_min.append(val)
        policy[i]=np.argmin(np.array(res_min))
    return policy

def value_update(tau,a,b,k,r,gamma,sigma,states,control_space,vpi,policy):
    '''
    update the value for the next iteration for each state
    '''
    vpi_next=np.copy(vpi)
    tr_prob=gaussian_f_xu(states,sigma,policy,control_space,a,b,tau)
    reward=l_xu(states,k,r,policy,control_space)
    vpi_next=reward.reshape((-1,1))+gamma*np.matmul(tr_prob,vpi)
    return vpi_next

def value_iteration(tau,a,b,k,r,gamma,sigma,states,control_space,vpi,policy):
    '''
    value iteration algorithm
    '''
    vpi=vpi.astype('float')
    for i in range(30):
        t0=tic()
        policy_new=policy_improvement(tau,a,b,k,r,gamma,sigma,states,control_space,vpi,policy)
        vpi_next=value_update(tau,a,b,k,r,gamma,sigma,states,control_space,vpi,policy_new)
        policy=np.copy(policy_new)
        vpi=np.copy(vpi_next)
        toc(t0,'iteration '+str(i))
    return policy,vpi

def get_trajectory(control_space,actions,start_state,states,goal_state,tau,a,b,vmax,sigma):
    f_xu=np.array([0,0])
    f_xu[0]=start_state[1]
    path=[]
    path.append(start_state)
    f_xu[1]=a*np.sin(start_state[0])-b*start_state[1]+\
    control_space[actions[np.argmin(np.array([np.linalg.norm(y-start_state) for y in states]))]]
    #choose the control which is assigned to the closest discretized state
    mu=start_state+f_xu*tau
    for i in range(500):
        #print(i)
        var=multivariate_normal(mu,sigma)
        n_state=var.rvs(size=1)
        #print(n_state)
        new_state=states[np.argmin(np.array([np.linalg.norm(y-n_state) for y in states]))]
        #check if the state is within bounds, wrap the state if not
        if(new_state[0]>np.pi):
            #print(i)
            new_state[0]=new_state[0]-2*np.pi
        elif(new_state[0]<-np.pi):
            #print(i)
            new_state[0]=new_state[0]+2*np.pi
        if(new_state[1]<-vmax):
            new_state[1]=-vmax
        elif(new_state[1]>vmax):
            new_state[1]=vmax
        path.append(new_state) 
        f_xu[0]=new_state[1]
        f_xu[1]=a*np.sin(new_state[0])-b*new_state[1]\
        +control_space[actions[np.argmin(np.array([np.linalg.norm(y-new_state) for y in states]))]]
        mu=new_state+f_xu*tau
    return path

def main():
    #essential parameters
    tau=0.18
    vmax=4
    umax=3
    n1=201
    n2=51
    nu=51
    k=2
    r=0.01
    a=1
    b=0.01
    gamma=0.9
    sig=np.array([0.1,0.1])
    eps=0.1
    sigma=sig*sig.T*tau+eps*np.eye(2)
    
    states=all_states(vmax,n1,n2) #discretize state space
    control_space=np.linspace(-umax,umax,nu) # discretize U
    policy=[2]*states.shape[0] #initialize policy
    # Value iteration
    vpi=np.array([1]*states.shape[0]).reshape((-1,1))
    policy=[0]*states.shape[0]
    actions_vi,vpf_vi=value_iteration(tau,a,b,k,r,gamma,sigma,states,control_space,vpi,policy)
    
    vf=vpf_vi.reshape((n1,n2))
    #Value function plot
    plt.imshow(vf, cmap='hot', interpolation='nearest')
    plt.show()
    #Policy iteration
    t0=tic()
    policy_pi,p_old,vpi_pi=policy_iteration(gamma,a,b,k,r,tau,sigma,states,control_space)
    toc(t0,'policy iteration')
    
    vf=vpf_pi.reshape((n1,n2))
    #Value function plot for PI
    plt.imshow(vf, cmap='hot', interpolation='nearest')
    plt.show()
    
    #generate mp4 for value_iteration
    start_state=np.array([-np.pi,0])
    goal_state=np.array([0,0])
    path,control_taken=get_trajectory(control_space,actions_vi,start_state,states,goal_state,tau,a,b,vmax,sigma)
    
    path_final=np.array(path)
    u=np.array(control_taken)
    theta=path_final[:,0]
    inv_pendulum=InvertedPendulum()
    inv_pendulum.load_trajectory(theta,u)
    inv_pendulum.start()
    

if __name__ == "__main__":
    main()