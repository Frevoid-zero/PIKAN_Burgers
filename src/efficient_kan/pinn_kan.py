import torch
import torch.nn as nn
from tqdm import tqdm

from kan import *

from kan import *
torch.set_default_dtype(torch.float32)

from collections import OrderedDict

from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
np.random.seed(0)
loss_arr_KAN = []
loss_arr_PINN = []

class PhysicsInformedNN_KAN:
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu):

        # boundary condition
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # data
        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device) #选择输入的x
        self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device) #选择输入的t
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)

        self.prune_flag=0
        self.layers = layers
        self.nu = nu

        # deep neural network
        # self.dnn = DNN(layers).to(device)
        self.dnn = KAN([2,32,32,32,32,32,16,1],grid_size=15, base_activation=nn.Identity).to(device)
        # self.dnn = KAN([2,32,1],grid_size=10, base_activation=nn.Identity).to(device)

        # optimizer with same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=0.8,
            max_iter=100,
            max_eval=100,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"   # strong wolfe interpolation
        )

        self.iter = 0

    def net_u(self, x, t):
        u = self.dnn(torch.cat([x, t], dim=1)) #把第一维的输出
        return u

    def net_f(self, x, t):
        u = self.net_u(x, t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]                                            #自动微分
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        # convection diffusion equation
        f = u_t + u * u_x - self.nu * u_xx

        return f

    def loss_fun(self):
        self.optimizer.zero_grad()

        u_pred = self.net_u(self.x_u, self.t_u)
        f_pred = self.net_f(self.x_f, self.t_f)
        loss_u = torch.mean((self.u - u_pred) ** 2) #数值的loss
        loss_f = torch. mean(f_pred ** 2) #物理信息的loss

        loss = loss_u + loss_f

        loss.backward()
        self.iter += 1
        if self.iter % 50 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' %
                (self.iter, loss.item(), loss_u.item(), loss_f.item())
            )
            loss_arr_KAN.append([self.iter, loss_u.detach().cpu().numpy(), loss_f.detach().cpu().numpy(), loss.detach().cpu().numpy()])
        return loss

    def train(self):
        # self.dnn.train()
        # self.dnn = self.dnn.prune()
        # # backward & optimizer
        # self.optimizer.step(self.loss_fun)
        # self.dnn.fit(self.dataset, opt="LBFGS", steps=50,loss_fn=self.loss_fun());
        self.dnn.train()

        self.optimizer.step(self.loss_fun)


    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_u(x, t)
        f = self.net_f(x, t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f


class PhysicsInformedNN_KAN_1:
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu):

        # boundary condition
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # data
        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device) #选择输入的x
        self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device) #选择输入的t
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)

        self.prune_flag=0
        self.layers = layers
        self.nu = nu

        # deep neural network
        # self.dnn = DNN(layers).to(device)
        self.dnn = KAN([2,32,32,32,32,32,16,1],grid_size=15, base_activation=nn.Identity).to(device)
        # self.dnn = KAN([2,32,1],grid_size=10, base_activation=nn.Identity).to(device)

        # optimizer with same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=0.8,
            max_iter=2000,
            max_eval=2000,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"   # strong wolfe interpolation
        )

        self.iter = 0

    def net_u(self, x, t):
        u = self.dnn(torch.cat([x, t], dim=1)) #把第一维的输出
        return u

    def net_f(self, x, t):
        u = self.net_u(x, t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]                                            #自动微分
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        # convection diffusion equation
        f = u_t + u * u_x - self.nu * u_xx

        return f

    def loss_fun(self):
        self.optimizer.zero_grad()

        u_pred = self.net_u(self.x_u, self.t_u)
        f_pred = self.net_f(self.x_f, self.t_f)
        loss_u = torch.mean((self.u - u_pred) ** 2) #数值的loss
        loss_f = torch. mean(f_pred ** 2) #物理信息的loss

        loss = loss_u + loss_f

        loss.backward()
        self.iter += 1
        if self.iter % 50 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' %
                (self.iter, loss.item(), loss_u.item(), loss_f.item())
            )
            loss_arr_KAN.append([self.iter, loss_u.detach().cpu().numpy(), loss_f.detach().cpu().numpy(), loss.detach().cpu().numpy()])
        return loss

    def train(self):
        # self.dnn.train()
        # self.dnn = self.dnn.prune()
        # # backward & optimizer
        # self.optimizer.step(self.loss_fun)
        # self.dnn.fit(self.dataset, opt="LBFGS", steps=50,loss_fn=self.loss_fun());
        self.dnn.train()

        self.optimizer.step(self.loss_fun)


    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_u(x, t)
        f = self.net_f(x, t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f



# deep neural network
class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameter
        self.depth = len(layers) - 1

        # layer order dict
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


class PhysicsInformedNN_PINN:
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu):

        # boundary condition
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # data
        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device) #选择输入的x
        self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device) #选择输入的t
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)

        self.layers = layers
        self.nu = nu

        # deep neural network
        self.dnn = DNN(layers).to(device)

        # optimizer with same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=10000,
            max_eval=10000,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"   # strong wolfe interpolation
        )
        self.iter = 0

    def net_u(self, x, t):
        u = self.dnn(torch.cat([x, t], dim=1)) #把第一维的输出
        return u

    def net_f(self, x, t):
        u = self.net_u(x, t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]                                            #自动微分
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        # convection diffusion equation
        f = u_t + u * u_x - self.nu * u_xx

        return f

    def loss_fun(self):
        self.optimizer.zero_grad()

        u_pred = self.net_u(self.x_u, self.t_u)
        f_pred = self.net_f(self.x_f, self.t_f)
        loss_u = torch.mean((self.u - u_pred) ** 2) #数值的loss
        loss_f = torch. mean(f_pred ** 2) #物理信息的loss

        loss = loss_u + loss_f
        loss.backward()

        self.iter += 1
        if self.iter % 50 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' %
                (self.iter, loss.item(), loss_u.item(), loss_f.item())
            )
            loss_arr_PINN.append([self.iter, loss_u.detach().cpu().numpy(), loss_f.detach().cpu().numpy(), loss.detach().cpu().numpy()])
        return loss

    def train(self):
        self.dnn.train()

        # backward & optimizer
        self.optimizer.step(self.loss_fun)

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_u(x, t)
        f = self.net_f(x, t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f


def test_mul():
    kan = KAN([2,16,32,16,8,1],grid_size=10, base_activation=nn.Identity)
    optimizer = torch.optim.LBFGS(kan.parameters(), lr=1)
    with tqdm(range(100)) as pbar:
        for i in pbar:
            loss, reg_loss = None, None

            def closure():
                optimizer.zero_grad()
                x = torch.rand(1024, 2)
                y = kan(x)

                assert y.shape == (1024, 1)
                nonlocal loss, reg_loss
                u = x[:, 0]
                v = x[:, 1]
                loss = nn.functional.mse_loss(y.squeeze(-1), (u + v) / (1 + u * v))
                reg_loss = kan.regularization_loss(1, 0)
                (loss + 1e-5 * reg_loss).backward()
                return loss + reg_loss

            optimizer.step(closure)
            pbar.set_postfix(mse_loss=loss.item(), reg_loss=reg_loss.item())


if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    nu = 0.01 / np.pi
    noise = 0.0

    N_u = 100
    N_f = 10000
    layers = [2, 32, 32, 32, 32, 32, 32, 32, 32, 1]

    data = scipy.io.loadmat('../../data/burgers_shock.mat')

    t = data['t'].flatten()[:, None]  # 100 1
    x = data['x'].flatten()[:, None]  # 256 1
    Exact = np.real(data['usol']).T  # 100 256 第一维表示t 第二维表示x

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]  # X_start, U_start

    # Doman boundary
    lb = X_star.min(0)
    ub = X_star.max(0)

    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))  # 在t=0时刻，x的所有值
    uu1 = Exact[0:1, :].T  # u(x,0)
    xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))  # X = -1 boundary，所有t的值
    uu2 = Exact[:, 0:1]
    xx3 = np.hstack((X[:, -1:], T[:, -1:]))  # X = 1 boundary，所有t的值
    uu3 = Exact[:, -1:]

    X_u_train = np.vstack([xx1, xx2, xx3])  # 初始值 边界值
    X_f_train = lb + (ub - lb) * lhs(2, N_f)  # 采样
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])  # 构造全部标签

    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]  # 训练集数据
    u_train = u_train[idx, :]  # 训练集标签

    # create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
    model_KAN = PhysicsInformedNN_KAN(X_u_train, u_train, X_f_train, 0, lb, ub, nu)

    model_KAN.train()

    u_pred_KAN, f_pred = model_KAN.predict(X_star)

    error_u_KAN = np.linalg.norm(u_star - u_pred_KAN, 2) / np.linalg.norm(u_star, 2)
    print('Error u: %e' % (error_u_KAN))

    U_pred_KAN = griddata(X_star, u_pred_KAN.flatten(), (X, T), method='cubic')
    Error = np.abs(Exact - U_pred_KAN)

    # Visualization ##############################################################
    # 0 u(t, x)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    h = ax.imshow(U_pred_KAN.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    ax.plot(
        X_u_train[:, 1],
        X_u_train[:, 0],
        # 'kx', label='Data (%d points)' % (u_train.shape([0])),
        'kx', label='Data (%d points)' % (u_train.shape[0]),
        markersize=4,
        clip_on=False,
        alpha=1.0
    )

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[25] * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[75] * np.ones((2, 1)), line, 'w-', linewidth=1)

    ax.set_xlabel('$t$', size=20)
    ax.set_ylabel('$x$', size=20)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )
    ax.set_title('$u(t, x)$', fontsize=20)
    ax.tick_params(labelsize=15)

    plt.show()

    # 1 u(t, x) slices

    fig = plt.figure(figsize=(14, 10))
    # ax = fig.add_subplot(111)

    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1.0 / 3.0 - 0.1, bottom=1.0 - 2.0 / 3.0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, Exact[25, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred_KAN[25, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t, x)$')
    ax.set_title('$t = 0.25$', fontsize=15)
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, Exact[50, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred_KAN[50, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t, x)$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('$t = 0.50$', fontsize=15)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, Exact[75, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred_KAN[75, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t, x)$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('$t = 0.75$', fontsize=15)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
    plt.savefig("KAN.png", dpi=300, bbox_inches='tight')  # 保存为高分辨率图像
    plt.show()

    # loss_arr_KAN = np.array(loss_arr_KAN)
    # plt.figure()
    #
    # # 设置图表标题和坐标轴标签
    # plt.title("Burgers Identification KAN")
    # plt.xlabel("Step")
    # plt.ylabel("Loss")
    #
    #
    # # 绘制折线图
    # plt.plot(loss_arr_KAN[:,0], loss_arr_KAN[:,1], color='red',label='Loss_u')
    # plt.plot(loss_arr_KAN[:,0], loss_arr_KAN[:,2], color='green',label='Loss_f')
    # plt.plot(loss_arr_KAN[:,0], loss_arr_KAN[:,3], color='blue',label='Loss_sum')
    # plt.legend()
    #
    #
    # # 显示图形
    # plt.show()

    #
    #
    # #PINN____________________________________________
    # model_PINN = PhysicsInformedNN_PINN(X_u_train, u_train, X_f_train, layers, lb, ub, nu)
    #
    # model_PINN.train()
    #
    # u_pred_PINN, f_pred_PINN = model_PINN.predict(X_star)
    #
    # error_u_PINN = np.linalg.norm(u_star - u_pred_PINN, 2) / np.linalg.norm(u_star, 2)
    # print('Error u: %e' % (error_u_PINN))
    #
    # U_pred_PINN = griddata(X_star, u_pred_PINN.flatten(), (X, T), method='cubic')
    # Error = np.abs(Exact - U_pred_PINN)
    #
    # # Visualization ##############################################################
    # # 0 u(t, x)
    #
    # fig = plt.figure(figsize=(9, 5))
    # ax = fig.add_subplot(111)
    #
    # h = ax.imshow(U_pred_PINN.T, interpolation='nearest', cmap='rainbow',
    #               extent=[t.min(), t.max(), x.min(), x.max()],
    #               origin='lower', aspect='auto')
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.10)
    # cbar = fig.colorbar(h, cax=cax)
    # cbar.ax.tick_params(labelsize=15)
    #
    # ax.plot(
    #     X_u_train[:, 1],
    #     X_u_train[:, 0],
    #     # 'kx', label='Data (%d points)' % (u_train.shape([0])),
    #     'kx', label='Data (%d points)' % (u_train.shape[0]),
    #     markersize=4,
    #     clip_on=False,
    #     alpha=1.0
    # )
    #
    # line = np.linspace(x.min(), x.max(), 2)[:, None]
    # ax.plot(t[25] * np.ones((2, 1)), line, 'w-', linewidth=1)
    # ax.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
    # ax.plot(t[75] * np.ones((2, 1)), line, 'w-', linewidth=1)
    #
    # ax.set_xlabel('$t$', size=20)
    # ax.set_ylabel('$x$', size=20)
    # ax.legend(
    #     loc='upper center',
    #     bbox_to_anchor=(0.9, -0.05),
    #     ncol=5,
    #     frameon=False,
    #     prop={'size': 15}
    # )
    # ax.set_title('$u(t, x)$', fontsize=20)
    # ax.tick_params(labelsize=15)
    #
    # plt.show()
    #
    # # 1 u(t, x) slices
    #
    # fig = plt.figure(figsize=(14, 10))
    # # ax = fig.add_subplot(111)
    #
    # gs1 = gridspec.GridSpec(1, 3)
    # gs1.update(top=1 - 1.0 / 3.0 - 0.1, bottom=1.0 - 2.0 / 3.0, left=0.1, right=0.9, wspace=0.5)
    #
    # ax = plt.subplot(gs1[0, 0])
    # ax.plot(x, Exact[25, :], 'b-', linewidth=2, label='Exact')
    # ax.plot(x, U_pred_PINN[25, :], 'r--', linewidth=2, label='Prediction')
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$u(t, x)$')
    # ax.set_title('$t = 0.25$', fontsize=15)
    # ax.axis('square')
    # ax.set_xlim([-1.1, 1.1])
    # ax.set_ylim([-1.1, 1.1])
    #
    # for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
    #              ax.get_xticklabels() + ax.get_yticklabels()):
    #     item.set_fontsize(15)
    #
    # ax = plt.subplot(gs1[0, 1])
    # ax.plot(x, Exact[50, :], 'b-', linewidth=2, label='Exact')
    # ax.plot(x, U_pred_PINN[50, :], 'r--', linewidth=2, label='Prediction')
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$u(t, x)$')
    # ax.axis('square')
    # ax.set_xlim([-1.1, 1.1])
    # ax.set_ylim([-1.1, 1.1])
    # ax.set_title('$t = 0.50$', fontsize=15)
    # ax.legend(
    #     loc='upper center',
    #     bbox_to_anchor=(0.5, -0.15),
    #     ncol=5,
    #     frameon=False,
    #     prop={'size': 15}
    # )
    #
    # for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
    #              ax.get_xticklabels() + ax.get_yticklabels()):
    #     item.set_fontsize(15)
    #
    # ax = plt.subplot(gs1[0, 2])
    # ax.plot(x, Exact[75, :], 'b-', linewidth=2, label='Exact')
    # ax.plot(x, U_pred_PINN[75, :], 'r--', linewidth=2, label='Prediction')
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$u(t, x)$')
    # ax.axis('square')
    # ax.set_xlim([-1.1, 1.1])
    # ax.set_ylim([-1.1, 1.1])
    # ax.set_title('$t = 0.75$', fontsize=15)
    #
    # for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
    #              ax.get_xticklabels() + ax.get_yticklabels()):
    #     item.set_fontsize(15)
    #
    # plt.show()
    #
    # loss_arr_PINN = np.array(loss_arr_PINN)
    # plt.figure()
    #
    # # 设置图表标题和坐标轴标签
    # plt.title("Burgers Identification PINN")
    # plt.xlabel("Step")
    # plt.ylabel("Loss")
    #
    # # 绘制折线图
    # plt.plot(loss_arr_PINN[:, 0], loss_arr_PINN[:, 1], color='red', label='Loss_u')
    # plt.plot(loss_arr_PINN[:, 0], loss_arr_PINN[:, 2], color='green', label='Loss_f')
    # plt.plot(loss_arr_PINN[:, 0], loss_arr_PINN[:, 3], color='blue', label='Loss_sum')
    # plt.legend()
    #
    # # 显示图形
    # plt.show()
    #
    # plt.figure()
    #
    # # 设置图表标题和坐标轴标签
    # plt.title("Burgers Identification")
    # plt.xlabel("Step")
    # plt.ylabel("Loss")
    #
    # # 绘制折线图
    # plt.plot(loss_arr_KAN[:,0], loss_arr_KAN[:,1], color='red',label='Loss_u_KAN')
    # plt.plot(loss_arr_KAN[:,0], loss_arr_KAN[:,2], color='blue',label='Loss_f_KAN')
    # plt.plot(loss_arr_KAN[:,0], loss_arr_KAN[:,3], color='green',label='Loss_sum_KAN')
    # plt.plot(loss_arr_PINN[:, 0], loss_arr_PINN[:, 1], color='black', label='Loss_u_PINN')
    # plt.plot(loss_arr_PINN[:, 0], loss_arr_PINN[:, 2], color='orange', label='Loss_f_PINN')
    # plt.plot(loss_arr_PINN[:, 0], loss_arr_PINN[:, 3], color='purple', label='Loss_sum_PINN')
    # plt.legend()
    #
    # # 显示图形
    # plt.show()
    #
    # plt.figure()
    #
    # # 设置图表标题和坐标轴标签
    # plt.title("Burgers Identification")
    # plt.xlabel("Step")
    # plt.ylabel("Loss")
    #
    # # 绘制折线图
    # plt.plot(loss_arr_KAN[:, 0], loss_arr_KAN[:, 3], color='green', label='Loss_sum_KAN')
    # plt.plot(loss_arr_PINN[:, 0], loss_arr_PINN[:, 3], color='purple', label='Loss_sum_PINN')
    # plt.legend()
    #
    # # 显示图形
    # plt.show()
    #
    # plt.figure()
    #
    # # 设置图表标题和坐标轴标签
    # plt.title("Burgers Identification")
    # plt.xlabel("Step")
    # plt.ylabel("Loss")
    #
    # # 绘制折线图
    # plt.plot(loss_arr_KAN[:, 0], loss_arr_KAN[:, 1], color='red', label='Loss_u_KAN')
    # plt.plot(loss_arr_KAN[:, 0], loss_arr_KAN[:, 2], color='blue', label='Loss_f_KAN')
    # plt.plot(loss_arr_PINN[:, 0], loss_arr_PINN[:, 1], color='black', label='Loss_u_PINN')
    # plt.plot(loss_arr_PINN[:, 0], loss_arr_PINN[:, 2], color='orange', label='Loss_f_PINN')
    # plt.legend()
    #
    # # 显示图形
    # plt.show()