import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import seaborn as sns

class env:
    def __init__(self, Slice_num:int, Max_Len = 1):
        #초기 설정
        self.Mx_My = Max_Len
        self.Slice_num = Slice_num
        self.c = 1

        #맵 및 속도 x, y 생성
        x = np.linspace(-self.Mx_My/2, self.Mx_My/2, self.Slice_num+1)
        y = np.linspace(-self.Mx_My/2, self.Mx_My/2, self.Slice_num+1)
        self.X, self.Y = np.meshgrid(x, y)
        self.u = np.zeros_like(self.X)
        self.v = np.zeros_like(self.X)

        #수치미분을 위한 x,y,t 간격
        self.k = x[1] - x[0]
        self.h = y[1] - y[0]#99 100 101
        dx = self.Mx_My / self.Slice_num
        dy = self.Mx_My / self.Slice_num
        self.dt = 0.5*(dx+dy)/self.c

    #파원 생성
    def set_source(self, x_start, y_start, R):
        r = np.hypot(self.X-x_start, self.Y-y_start)
        self.u += (r < R) * ((self.Mx_My + np.cos(np.pi * r / R)) / 2)

    #파동 갱신
    def update_wave(self, u_:np.array, v_:np.array):
        du = np.zeros_like(self.u)
        dv = np.zeros_like(v_)
        du_2_dx_2 = (u_[1:,2:] - 2*u_[1:,1:-1] + u_[1:,0:-2]) / (self.k**2)
        du_2_dy_2 = (u_[2:,1:] - 2*u_[1:-1,1:] + u_[0:-2,1:]) / (self.h**2)
        du = np.zeros_like(self.u)
        dv = np.zeros_like(self.v)
        du[1:, 1:] = v_[1:, 1:]
        du_2_dx_2 = np.pad(du_2_dx_2, ((0,0),(0,1)), 'constant', constant_values=0)
        du_2_dy_2 = np.pad(du_2_dy_2, ((0,1),(0,0)), 'constant', constant_values=0)
        dv[1:, 1:] = (self.c**2)*(du_2_dx_2 + du_2_dy_2)
        return du, dv

#룽지-쿠타 방법을 위한 K값 반환 함수
def Get_K(u_, v_, env_:env):
    du ,dv = env_.update_wave(u_, v_)
    ku = du*env_.dt
    kv = dv*env_.dt
    return ku, kv

#룽지-쿠타 방법을 통해 다음 파동을 구현하는 함수
def Runge_Kutta(env_:env):
    k1u, k1v = Get_K(env_.u, env_.v, env_)
    k2u, k2v = Get_K(env_.u+k1u/2,env_.v+k1v/2, env_)
    k3u, k3v = Get_K(env_.u+k2u/2,env_.v+k2v/2, env_)
    k4u, k4v = Get_K(env_.u+k3u,env_.v+k3v, env_)
    env_.u += (k1u + 2*k2u + 2*k3u + k4u)/6
    env_.v += (k1v + 2*k2v + 2*k3v + k4v)/6

#그래프 갱신 함수
def update_plot(t):
    global t_, env1, mode, ax
    t_ += env1.dt
    Runge_Kutta(env1)

    if mode == 0: #3D 그래프
        ax.clear()
        ax.set_xlim(-env1.Mx_My/2,env1.Mx_My/2)
        ax.set_ylim(-env1.Mx_My/2,env1.Mx_My/2)
        ax.set_zlim(-0.5,0.5)
        ax.plot_surface(env1.X,env1.Y,env1.u, cmap='viridis')
    
    elif mode == 1: #xy평면 Heatmap 그래프
        plt.clf()
        ax = sns.heatmap(env1.u, cmap='viridis', cbar=False)
        ax.tick_params(left=False, bottom=False)
    
    elif mode == 2: #xz평면 그래프
        plt.clf()
        Y_u = env1.u[int(env1.Slice_num//2)]
        plt.ylim(-0.5, 0.5)
        plt.plot(range(len(Y_u)), Y_u)

    elif mode == 3:  #yz평면 그래프
        plt.clf()
        Y_u = env1.u[:,int(env1.Slice_num//2)].reshape(-1)
        plt.ylim(-0.5, 0.5)
        plt.plot(range(len(Y_u)), Y_u)

    plt.title(f't={t_}')
    
    print((t/(t_max//env1.dt)*100),end='\r')

#그래프 설정
mode = 1
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

#환경 생성
env1 = env(100, 1)
env1.set_source(0.39, 0.0, 0.11)

#시작 시간 및 최대 시간
t_ = 0
t_max = 6.5

#그래프 생성
ani = FuncAnimation(plt.gcf(), update_plot, frames=int(t_max//env1.dt), interval=1)
ani.save('./wave_xy3.gif', fps=17)