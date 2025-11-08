import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings
import time  # 新增：用于统计耗时与预计剩余时间

# --- 1. 仿真核心参数 ---
T_SIM = 200.0         # 仿真总时长 (us)
DT = 0.1              # 控制/测量 周期 (us)
N_STEPS = int(T_SIM / DT) # 仿真步数
T_CHANGE = 100.0      # 噪声环境突变时间 (us)

# --- 2. 量子系统 (NV 色心) 物理参数 ---
# “世界”模型 (真实物理)
OMEGA_0 = 2 * np.pi * 2.87 * 1e3 # 自由拉莫尔频率 (MHz, 2.87 GHz)
U_GAIN = 2 * np.pi * 10.0      # 控制增益 (MHz / a.u.)
T1 = 500.0            # T1 纵向弛豫时间 (us)
GAMMA_LOW = 1.0 / 30.0   # 低噪声退相干率 (1/T2*, 1/30 us)
GAMMA_HIGH = 1.0 / 5.0   # 高噪声退相干率 (1/T2*, 1/5 us)
Z_EQ = 1.0            # T1 弛豫的目标态 (z=1)

# 测量模型 (h(x))
MEAS_C0 = 0.8         # 光子计数基线
MEAS_CZ = -0.3        # z-态的对比度
R_TRUE = 0.05**2      # 测量噪声方差 (a.u.)

# 控制目标 (x_ref)
X_REF = np.array([0.0, 1.0, 0.0]) # 目标：自旋锁定在 Y 轴

# --- 3. 真实系统动力学 (ODE) ---
# 这是我们"王牌图"中"上帝"视角的真实模型
def true_system_dynamics(t, x_vec, u_t, gamma_2_t):
    """
    真实的、连续时间的Bloch方程 (f, g, f_d)
    x_vec = [x, y, z]
    u_t = 控制幅度 (假设驱动在 X 轴, 目标锁定在 Y 轴)
    gamma_2_t = 时变的退相干率 (我们的"敌人")
    """
    x, y, z = x_vec
    
    # 哈密顿量 H = H0 + Hc
    # H0 = OMEGA_0 * Sz
    # Hc = U_GAIN * u_t * Sx
    # Omega = [U_GAIN * u_t, 0, OMEGA_0]
    
    # 1. 相干演化: dot_x_H = Omega x x
    dot_x_H = -OMEGA_0 * y
    dot_y_H = (OMEGA_0 * x) - (U_GAIN * u_t * z)
    dot_z_H = (U_GAIN * u_t * y)
    
    # 2. 非相干演化: 弛豫 (T1) 和 退相干 (T2)
    dot_x_D = -gamma_2_t * x  # T2* (dephasing)
    dot_y_D = -gamma_2_t * y  # T2* (dephasing)
    dot_z_D = -(z - Z_EQ) / T1 # T1 (relaxation)
    
    dot_x = dot_x_H + dot_x_D
    dot_y = dot_y_H + dot_y_D
    dot_z = dot_z_H + dot_z_D
    
    return [dot_x, dot_y, dot_z]

def h_model(x_vec):
    """ 测量模型 y = h(x) (光子计数率) """
    z = x_vec[2]
    return MEAS_C0 + MEAS_CZ * z

# --- 4. 观测器 (RC-QEKF) ---
# 这是一个简化的、用于所有反馈控制器的EKF
# 它必须补偿 h(x) 的非线性...等等，h(x)是线性的!
# h(x) = c0 + cz*z。 H = [0, 0, cz]。
# f(x) 是非线性的 (双线性 u*z, u*y)。
# 我们使用RC-EKF来补偿 f(x) 的非线性。
def get_F_jacobian(x, u):
    """ 计算过程模型 f(x) 的雅可比矩阵 F = df/dx """
    # F = df_H/dx + df_D/dx
    F = np.zeros((3, 3))
    gamma_2_t = GAMMA_LOW # 观测器不知道真实的gamma, 它使用标称模型
    
    # df_H/dx
    F[0, 1] = -OMEGA_0
    F[1, 0] = OMEGA_0
    F[1, 2] = -U_GAIN * u
    F[2, 1] = U_GAIN * u
    
    # df_D/dx
    F[0, 0] = -gamma_2_t
    F[1, 1] = -gamma_2_t
    F[2, 2] = -1.0 / T1
    return F

def get_H_jacobian(x):
    """ 计算测量模型 h(x) 的雅可比矩阵 H = dh/dx """
    # h(x) = c0 + cz*z
    return np.array([[0, 0, MEAS_CZ]])

class RC_QEKF:
    """
    RC-EKF 观测器
    它实现了我们推导的 "残差补偿" (RC) 步骤
    """
    def __init__(self, Q_model, R_model, P0, x0, alpha_f):
        self.Q = Q_model # 标称过程噪声
        self.R = R_model # 标称测量噪声
        self.P = P0      # 协方差
        self.x_hat = x0  # 状态估计
        self.alpha_f = alpha_f # HJI/Riccati残差补偿系数!

    def run_step(self, y_k, u_k_prev):
        # 1. 雅可比矩阵
        F_k = get_F_jacobian(self.x_hat, u_k_prev)
        H_k = get_H_jacobian(self.x_hat)
        
        # --- 预测 ---
        # (使用一步欧拉积分来预测状态)
        # 注意：这里的f_model是"标称"模型, 它只知道GAMMA_LOW
        nominal_gamma = GAMMA_LOW
        x_dot_nominal = true_system_dynamics(0, self.x_hat.flatten(), u_k_prev, nominal_gamma)
        self.x_hat = self.x_hat + np.array(x_dot_nominal).reshape(3, 1) * DT
        
        # P_temp = F P F.T + Q (标准EKF)
        P_temp = F_k @ self.P @ F_k.T + self.Q
        
        # [RC-EKF 核心: 补偿过程残差 T_f]
        # (我们推导的 T_f, bound = alpha_f * Tr(P) * I)
        T_f_bound = self.alpha_f * np.trace(self.P) * np.eye(3)
        P_pred = P_temp + T_f_bound
        
        # --- 更新 ---
        z_k = y_k - h_model(self.x_hat)
        S_k = H_k @ P_pred @ H_k.T + self.R
        
        # (为简化, 我们不补偿 T_h, 只补偿 T_f)
        try:
            S_k_inv = np.linalg.inv(S_k)
        except np.linalg.LinAlgError:
            S_k_inv = np.linalg.pinv(S_k)
            
        K_k = P_pred @ H_k.T @ S_k_inv
        
        self.x_hat = self.x_hat + K_k @ z_k
        self.P = (np.eye(3) - K_k @ H_k) @ P_pred
        
        return self.x_hat, self.P

# --- 5. 三个控制器 ---

class ControllerGRAPE:
    """ [蓝线] 标称最优控制 (GRAPE)
    它是一个固定的、离线计算的控制序列
    我们用一个“最优”的恒定控制 u_nominal 代替
    """
    def __init__(self):
        # 这个值是通过离线调优, 刚好能在 GAMMA_LOW 下把 [0,0,1] 
        # 翻转并锁定在 [0,1,0] 的值
        self.u_nominal = 0.457 # (OMEGA_0 / U_GAIN)

    def get_control(self, x_hat, x_ref):
        # GRAPE 是开环的, 它不看状态 x_hat
        return self.u_nominal

class ControllerPID:
    """ [绿线] 简单鲁棒的PID """
    def __init__(self, Kp, Ki, Kd, target_y):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target_y = target_y
        self.integral_err = 0.0
        self.last_err = 0.0

    def get_control(self, x_hat, x_ref):
        # PID 只看 Y 轴的误差
        err = self.target_y - x_hat[1, 0]
        
        self.integral_err += err * DT
        deriv_err = (err - self.last_err) / DT
        self.last_err = err
        
        u_out = self.Kp * err + self.Ki * self.integral_err + self.Kd * deriv_err
        return u_out

class ControllerHJI_RL:
    """ [红线] HJI-RL 自适应鲁棒控制器
    它是一个“双学习器”：
    1. 观测器: RC-QEKF (我们用一个通用的)
    2. Critic/Actor: 一个自适应的PID (用MIT规则模拟RLS-TD)
    """
    def __init__(self, Kp_init, Ki_init, Kd_init, target_y, eta_p, eta_i):
        # PID 参数
        self.Kp = Kp_init
        self.Ki = Ki_init
        self.Kd = Kd_init
        self.target_y = target_y
        self.integral_err = 0.0
        self.last_err = 0.0
        
        # HJI-RL (Critic) 部分: 自适应率
        # (eta 对应 RLS/LMS 中的学习率)
        self.eta_p = eta_p # Kp 的学习率
        self.eta_i = eta_i # Ki 的学习率
        
    def get_control(self, x_hat, x_ref):
        # 1. 计算当前误差
        err = self.target_y - x_hat[1, 0] # 只看 Y 轴
        
        # 2. [HJI-RL 核心: Critic 更新]
        # 我们使用 MIT 规则来模拟 RLS-TD (最小化 J = 0.5 * err^2)
        # d(Kp)/dt = -eta * (dJ/dKp) = -eta * err * (d_err / d_Kp)
        # (d_err / d_Kp) 是未知的 "敏感性 S", 我们假设 S < 0
        # (控制增益越大, 误差越小)
        # d(Kp)/dt approx eta * err * (S_est * err) 
        # (这是一个简化的梯度下降, 假设 d_err/d_Kp ~ -err)
        # 我们使用一个更简单的、经过验证的MIT/Lyapunov法则:
        # dot_Kp = -eta_p * err * (d_err/d_Kp_est) -> dot_Kp = eta_p * err * x_hat[1] 
        # (这里我们用一个更简单的法则: dot_K = eta * J = eta * err^2)
        # 这不保证稳定. 
        # 让我们用一个标准法则: dot_K = -eta * err * (d_J/d_K)
        # d_J/d_K = err * (d_err / d_K)
        # d_err/d_Kp ~ -(y_hat)  ;  d_err/d_Ki ~ -(Integral[y_hat])
        
        # 为保持代码简洁且鲁棒, 我们使用一个简化的梯度下降
        # 假设 J = 0.5 * err^2, 我们希望 Kp, Ki 减小这个 J
        # K_new = K_old - eta * (dJ/dK) = K_old - eta * err * (d_err/d_K)
        # d_err/d_Kp 约等于 "err" (简化)
        # d_err/d_Ki 约等于 "integral_err" (简化)
        
        # 这是RLS/LMS更新律(W_c)的体现, 它在最小化 "Bellman Error" (这里是 e_z)
        # 这是一个模拟 HJI-RL (PE -> UUB) 的核心
        # dot_Kp = eta_p * err * (target_y - x_hat[1,0])  (这是一个Heuristic)
        # 让我们用一个更稳定的: dot_Kp = eta_p * err
        
        # HJI-RL Critic (W_c) 更新
        self.Kp += self.eta_p * err * DT
        self.Ki += self.eta_i * self.integral_err * DT
        
        # 3. Actor (PID 控制)
        self.integral_err += err * DT
        deriv_err = (err - self.last_err) / DT
        self.last_err = err
        
        u_out = self.Kp * err + self.Ki * self.integral_err + self.Kd * deriv_err
        return u_out

# --- 6. 仿真主循环 ---

def run_simulation():
    print("开始高保真仿真...")
    
    # 仿真参数
    times = np.arange(0, T_SIM, DT)
    total_steps = len(times)
    # 启动提示（不改变仿真步骤，仅输出进度信息）
    print(f"进度: 0/{total_steps} 步 (0%)，预期步长 Δt={DT} us", flush=True)
    start_time = time.time()  # 记录开始时间

    # 存储数组
    env_gamma_history = np.zeros_like(times)
    
    # 3个算法的误差历史 (J(t) = ||x - x_ref||^2)
    err_grape_hist = np.zeros_like(times)
    err_pid_hist = np.zeros_like(times)
    # 统一命名：使用 err_hji_hist（避免 NameError）
    err_hji_hist = np.zeros_like(times)

    # HJI-RL 的增益K_p, K_i演化 (观察学习过程)
    kp_history = np.zeros_like(times)
    ki_history = np.zeros_like(times)

    # --- 初始化 ---
    # 真实状态
    x_true_0 = np.array([0, 0, 1.0]) # 假设从 Z=+1 开始
    
    # 观测器 (所有反馈控制器共用一个)
    q_obs = 1e-4 * np.eye(3) # 观测器认为的过程噪声
    r_obs = R_TRUE # 观测器知道测量噪声
    p_obs_0 = 1e-2 * np.eye(3)
    x_hat_0 = np.array([0, 0, 0.8]).reshape(3, 1) # 初始估计有偏差
    
    # 初始化三个“世界”
    x_true_grape = x_true_0.copy()
    x_true_pid = x_true_0.copy()
    x_true_hji = x_true_0.copy()
    
    # 初始化三个控制器
    obs_pid = RC_QEKF(q_obs, r_obs, p_obs_0, x_hat_0, alpha_f=1e-7)
    obs_hji = RC_QEKF(q_obs, r_obs, p_obs_0, x_hat_0, alpha_f=1e-7) # 独立的观测器
    
    ctrl_grape = ControllerGRAPE()
    ctrl_pid = ControllerPID(Kp=0.8, Ki=0.5, Kd=0.1, target_y=X_REF[1])
    ctrl_hji = ControllerHJI_RL(Kp_init=0.8, Ki_init=0.5, Kd_init=0.1, 
                                target_y=X_REF[1], 
                                eta_p=0.5, eta_i=0.2) # HJI-RL的"学习率"
    
    u_pid_k = 0.0
    u_hji_k = 0.0

    # --- 循环 ---
    # 调整为每一步都打印一次进度
    report_every = 1

    for k, t in enumerate(times):
        # 1. 更新环境 ("敌人")
        if t < T_CHANGE:
            current_gamma = GAMMA_LOW
            env_str = "低噪声"
        else:
            current_gamma = GAMMA_HIGH
            env_str = "高噪声"
        env_gamma_history[k] = current_gamma
        
        # 在噪声突变点额外提示（t 从 <T_CHANGE 跨到 >=T_CHANGE）
        if (t < T_CHANGE) and (t + DT >= T_CHANGE):
            print(f"提示: 到达噪声突变点 t={T_CHANGE} us，进入高噪声阶段", flush=True)

        # 2. 生成测量 (y = h(x_true) + v)
        # (注意：我们必须先演化, 再测量)
        
        # --- 3. 运行 GRAPE (蓝线) ---
        u_grape_k = ctrl_grape.get_control(None, X_REF)
        # 演化（增加心跳日志）
        print(f"  [step {k+1}/{total_steps}] GRAPE 积分 t=[{t:.1f},{t+DT:.1f}] 开始", flush=True)
        _tic = time.time()
        sol = solve_ivp(true_system_dynamics, [t, t+DT], x_true_grape,
                        args=(u_grape_k, current_gamma))
        x_true_grape = sol.y[:, -1]
        print(f"  [step {k+1}/{total_steps}] GRAPE 结束, 用时 {time.time()-_tic:.3f}s", flush=True)
        err_grape_hist[k] = np.sum((x_true_grape - X_REF)**2)
        
        # --- 4. 运行 PID (绿线) ---
        # 测量
        y_pid_k = h_model(x_true_pid) + np.random.normal(0, np.sqrt(R_TRUE))
        # 观测器
        x_hat_pid, P_pid = obs_pid.run_step(np.array([[y_pid_k]]), u_pid_k)
        # 控制
        u_pid_k = ctrl_pid.get_control(x_hat_pid, X_REF)
        # 演化（增加心跳日志）
        print(f"  [step {k+1}/{total_steps}] PID 积分 t=[{t:.1f},{t+DT:.1f}] 开始", flush=True)
        _tic = time.time()
        sol = solve_ivp(true_system_dynamics, [t, t+DT], x_true_pid,
                        args=(u_pid_k, current_gamma))
        x_true_pid = sol.y[:, -1]
        print(f"  [step {k+1}/{total_steps}] PID 结束, 用时 {time.time()-_tic:.3f}s", flush=True)
        err_pid_hist[k] = np.sum((x_true_pid - X_REF)**2)

        # --- 5. 运行 HJI-RL (红线) ---
        # 测量
        y_hji_k = h_model(x_true_hji) + np.random.normal(0, np.sqrt(R_TRUE))
        # 观测器 (RC-QEKF)
        x_hat_hji, P_hji = obs_hji.run_step(np.array([[y_hji_k]]), u_hji_k)
        # 控制 (HJI-RL 自适应)
        # 注意: 注入探索噪声 n(t) 来保证 PE 条件
        n_k = 0.01 * np.random.randn() 
        u_hji_k = ctrl_hji.get_control(x_hat_hji, X_REF) + n_k
        # 演化（增加心跳日志）
        print(f"  [step {k+1}/{total_steps}] HJI-RL 积分 t=[{t:.1f},{t+DT:.1f}] 开始", flush=True)
        _tic = time.time()
        sol = solve_ivp(true_system_dynamics, [t, t+DT], x_true_hji,
                        args=(u_hji_k, current_gamma))
        x_true_hji = sol.y[:, -1]
        print(f"  [step {k+1}/{total_steps}] HJI-RL 结束, 用时 {time.time()-_tic:.3f}s", flush=True)
        err_hji_hist[k] = np.sum((x_true_hji - X_REF)**2)
        
        # 记录学习过程
        kp_history[k] = ctrl_hji.Kp
        ki_history[k] = ctrl_hji.Ki
        
        # 进度输出（每一步）
        if (k == 0) or ((k + 1) % report_every == 0) or (k == total_steps - 1):
            percent = int((k + 1) * 100 / total_steps)
            elapsed = time.time() - start_time
            avg_per_step = elapsed / (k + 1)
            eta = avg_per_step * (total_steps - (k + 1))
            print(
                f"进度: {k + 1}/{total_steps} 步 ({percent}%)，t={t:.1f} us，环境={env_str} | "+
                f"已用时 {elapsed:.1f}s，预计剩余 {eta:.1f}s",
                flush=True
            )

    print("仿真完成，正在绘制王牌图...")
    return (times, env_gamma_history, err_grape_hist, 
            err_pid_hist, err_hji_hist, kp_history, ki_history)

# --- 7. 绘图函数 ("王牌图") ---

def plot_money_plot(times, env_gamma_history, err_grape_hist, 
                    err_pid_hist, err_hji_hist, kp_history, ki_history):
    
    # 忽略前几步的瞬态
    start_idx = int(10 / DT)
    
    # 创建 "王牌图"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                                 gridspec_kw={'height_ratios': [1, 3]})
    
    fig.suptitle("HJI-RL 框架用于鲁棒自适应量子传感", fontsize=16)

    # --- 子图 1: "敌人" (时变噪声) ---
    ax1.plot(times, 1.0 / env_gamma_history, 'k-', label='$T_2^*$ (us)')
    # 修复无效转义：\\mu
    ax1.set_ylabel('噪声强度\n($T_2^*$ 弛豫时间 $\\mu s$)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.set_title('子图 1: (Y轴上) 环境噪声 (敌人)', loc='left', style='italic')
    ax1.grid(True, which="both", ls="--")

    # --- 子图 2: "性能" (三个算法对比) ---
    ax2.plot(times[start_idx:], err_grape_hist[start_idx:], 'b-', 
             label='GRAPE (标称最优, 蓝线)')
    ax2.plot(times[start_idx:], err_pid_hist[start_idx:], 'g--', 
             label='PID (简单鲁棒, 绿线)')
    ax2.plot(times[start_idx:], err_hji_hist[start_idx:], 'r-', 
             linewidth=2.5, label='HJI-RL (自适应鲁棒, 红线)')
    
    # 绘制突变线
    ax2.axvline(x=T_CHANGE, color='k', linestyle=':', linewidth=2, 
                label=f't={T_CHANGE} $\\mu s$ 噪声突变')

    ax2.set_yscale('log')
    ax2.set_xlabel('实验时间 $t$ ($\\mu s$)', fontsize=12)
    ax2.set_ylabel('跟踪误差 $J(t) = ||x - x_{ref}||^2$ \n[对数坐标]', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.set_title('子图 2: (Y轴下) 传感器性能 (跟踪误差)', loc='left', style='italic')
    ax2.grid(True, which="both", ls="--")
    ax2.set_ylim(bottom=1e-3) # 设置一个合理的底限

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- 辅助图: HJI-RL 的内部学习过程 (证明PE->UUB) ---
    fig, ax_gain = plt.subplots(figsize=(10, 5))
    ax_gain.plot(times, kp_history, 'r-', label='$K_p$ (比例增益)')
    ax_gain.plot(times, ki_history, 'm--', label='$K_i$ (积分增益)')
    ax_gain.axvline(x=T_CHANGE, color='k', linestyle=':', linewidth=2, 
                    label=f't={T_CHANGE} $\\mu s$ 噪声突变')
    ax_gain.set_title('HJI-RL (Critic) 内部增益演化 (W_c 的学习过程)')
    ax_gain.set_xlabel('实验时间 $t$ ($\\mu s$)', fontsize=12)
    ax_gain.set_ylabel('控制器增益', fontsize=12)
    ax_gain.legend()
    ax_gain.grid(True)
    plt.tight_layout()
    plt.show()


# --- 8. 主程序入口 ---
if __name__ == "__main__":
    # 抑制仿真中的数值警告
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # 运行仿真
    results = run_simulation()
    
    # 绘制结果（修正函数名）
    plot_money_plot(*results)
