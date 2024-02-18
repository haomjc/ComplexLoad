from feon.sa import *
from feon.tools import pair_wise
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import scipy.sparse as sp
import scipy.sparse.linalg as sl
import matlab.engine

# =======================螺纹牙刚度计算==========================
e1 = 2.12e5  # 丝杆：GCr15材料的弹性模量
e2 = 2.12e5  # 滚柱：GCr15材料的弹性模量
e3 = 2.12e5  # 螺母：GCr15材料的弹性模量
v1 = 0.29  # 丝杆：GCr15材料的泊松比模量
v2 = 0.29  # 滚柱：GCr15材料的泊松比模量
v3 = 0.29  # 螺母：GCr15材料的泊松比模量

ats = 1.95
atr = 1.95
atn = 1.95
bts = 0.85
btr = 0.85
btn = 0.85
cts = 0.55
ctr = 0.55
ctn = 0.55

P = 2

ps = P
pr = P
pn = P

# rr = 8 / 2
# rs = 24 / 2
# rn = 40 / 2

rr = 5.5 / 2
rs = 14.5 / 2
rn = 25.5 / 2

ds = rs * 2
dr = rr * 2
dn = rn * 2

ns = 5
nr = 1
nn = 5

Pr = pr * nr
Ps = ps * ns
Pn = pn * nn

a1 = np.arctan(Ps / (np.pi * 2 * rs))  # 丝杠螺旋升角

a2 = np.arctan(Pn / (np.pi * 2 * rn))  # 螺母螺旋升角
a3 = np.arctan(Pr / (np.pi * 2 * rr))  # 滚柱螺旋升角

# 丝杆
beta = np.pi / 4
v = v1
E = e1

FNSRi = 1
FNNRi = 1
p = P

Ra = 0.3
L = 0.5
Aa = L * L

Ss1 = (1 - v ** 2) * 3 * FNSRi * np.cos(beta) * np.cos(a3) / 4 / E * (
        (1 - (2 - bts / ats) ** 2 + 2 * np.log(ats / bts)) * (1 / np.tan(beta)) ** 3 - 4 * (cts / ats) ** 2 * np.tan(
    beta))
Ss2 = (1 + v) * 6 * FNSRi * np.cos(beta) * np.cos(a3) / 5 / E * 1 / np.tan(beta) * np.log(ats / bts)
Ss3 = (1 - v ** 2) * 12 * FNSRi * np.cos(beta) * np.cos(a3) * cts / (np.pi * E * ats ** 2) * (
        cts - bts / 2 * np.tan(beta))
Ss4 = (1 - v ** 2) * 2 * FNSRi * np.cos(a3) * np.cos(beta) / (np.pi * E) * (
        p / ats * np.log((p + ats / 2) / (p - ats / 2)) + 1 / 2 * np.log(4 * p ** 2 / ats ** 2 - 1))
Ss5 = (1 - v) * (np.tan(beta)) ** 2 / 2 * ds / p * FNSRi * np.cos(beta) * np.cos(a3) / E
KST = FNSRi / (Ss1 + Ss2 + Ss3 + Ss4 + Ss5)  # Axial stiffness of the screw thread
#     KST=2.078e5

KST0 = int(KST * 1000)

print('KST0：', KST0)

# 滚柱
Ssr1 = (1 - v ** 2) * 3 * FNSRi * np.cos(beta) * np.cos(a3) / 4 / E * (
        (1 - (2 - btr / atr) ** 2 + 2 * np.log(atr / btr)) * (1 / np.tan(beta)) ** 3 - 4 * (ctr / atr) ** 2 * np.tan(
    beta))
Ssr2 = (1 + v) * 6 * FNSRi * np.cos(beta) * np.cos(a3) / 5 / E * 1 / np.tan(beta) * np.log(atr / btr)
Ssr3 = (1 - v ** 2) * 12 * FNSRi * np.cos(beta) * np.cos(a3) * ctr / (np.pi * E * atr ** 2) * (
        ctr - btr / 2 * np.tan(beta))
Ssr4 = (1 - v ** 2) * 2 * FNSRi * np.cos(beta) * np.cos(a3) / (np.pi * E) * (
        p / atr * np.log((p + atr / 2) / (p - atr / 2)) + 1 / 2 * np.log(4 * p ** 2 / atr ** 2 - 1))
Ssr5 = (1 - v) * (np.tan(beta)) ** 2 / 2 * dr / p * FNSRi * np.cos(beta) * np.cos(a3) / E
KRNT = FNSRi / (Ssr1 + Ssr2 + Ssr3 + Ssr4 + Ssr5)
KRNT0 = int(KRNT * 1000)
print('KRNT0：', KRNT0)
KRST = FNNRi / (Ssr1 + Ssr2 + Ssr3 + Ssr4 + Ssr5)
KRST0 = int(KRST * 1000)
print('KRST0：', KRST0)
# 螺母
Snr1 = (1 - v ** 2) * 3 * FNNRi * np.cos(beta) * np.cos(a3) / 4 / E * (
        (1 - (2 - btr / atr) ** 2 + 2 * np.log(btr / atr)) * (1 / np.tan(beta)) ** 3 - 4 * (ctr / atr) ** 2 * np.tan(
    beta))
Snr2 = (1 + v) * 6 * FNNRi * np.cos(beta) * np.cos(a3) / 5 / E * 1 / np.tan(beta) * np.log(atr / btr)
Snr3 = (1 - v ** 2) * 12 * FNNRi * np.cos(beta) * np.cos(a3) * ctr / (np.pi * E * atr ** 2) * (
        ctr - btr / 2 * np.tan(beta))
Snr4 = (1 - v ** 2) * 2 * FNNRi * np.cos(beta) * np.cos(a3) / (np.pi * E) * (
        p / atr * np.log((p + atr / 2) / (p - atr / 2)) + 1 / 2 * np.log(4 * p ** 2 / atr ** 2 - 1))
Snr5 = (1 - v) * (np.tan(beta)) ** 2 / 2 * dr / p * FNNRi * np.cos(beta) * np.cos(a3) / E
KNT = FNNRi / (Snr1 + Snr2 + Snr3 + Snr4 + Snr5)
KNT0 = int(KNT * 1000)

# KRNT0 = KST0
# KRST0 = KST0
# KNT0 = KST0

# KST0 = 3E6
# KRNT0 = 3E6
# KRST0 = 3E6
# KNT0 = 3E6

print('KNT0：', KNT0)
# ==============================================
# 丝杠节点
P = 2E-3  # 螺距
n = 12  # 螺纹牙数
tt = 6  # 滚柱根数
# 从原点位置开始

# rr = 8E-3 / 2
# rs = 24E-3 / 2
# rn = 40E-3 / 2
# Rn = 60E-3 / 2
rr = 5.5E-3 / 2
rs = 14.5E-3 / 2
rn = 25.5E-3 / 2
Rn = 36.5E-3 / 2

ds = rs * 2
dr = rr * 2
dn = rn * 2
D0 = Rn * 2

hf = 0E-3

E = 2.12E8
miu = 0.29
G = E / (2 * (1 + miu))

# E_rigid = 212E6  # 螺纹牙变形单元的弹性模量
# miu_rigid = 0.29
# G_rigid = E_rigid / (2 * (1 + miu_rigid))

A_nut = np.pi * (D0 ** 2 - (dn - hf) ** 2) / 4 / tt  # 截面积

A_roller = np.pi * (dr - 2 * hf) ** 2 / 4
# print(A_roller)
A_screw = np.pi * (ds - hf) ** 2 / 4

# A_rigid_nut = A_nut
# A_rigid_roller = A_nut
# A_rigid_screw = A_nut

rho_steel = 7850
m_roller = A_roller * rho_steel * P
# print("m_roller:", m_roller)
m_screw = A_screw * rho_steel * P
m_nut = A_nut * rho_steel * P

L_1 = (np.pi * dn + np.pi * D0) / 2 / tt
L_2 = P
L_3 = (D0 - dn) / 2

# J = 1 / n * m_nut * (L_1 ** 2 + L_3 ** 2)
# I_y = 1 / n * m_nut * (L_1 ** 2 + L_2 ** 2)
# I_z = 1 / n * m_nut * (L_2 ** 2 + L_3 ** 2)
J = L_2 * L_3 ** 3 / 12
I_y = L_2 * L_1 ** 3 / 12
I_z = L_2 * L_3 ** 3 / 12

I_nut = [J, I_y, I_z]

# J = m_roller * rr ** 2 / 2
# I_y = m_roller * rr ** 2 / 4 + m_roller * P ** 2 / n
# I_z = m_roller * rr ** 2 / 4 + m_roller * P ** 2 / n
J = P * (2 * rr) ** 3 / 12
I_y = np.pi * (2 * rr) ** 4 / 64
I_z = np.pi * (2 * rr) ** 4 / 64

I_roller = [J, I_y, I_z]

# J = m_screw * rs ** 2 / 2
# I_y = m_screw * rs ** 2 / 4 + m_screw * P ** 2 / n
# I_z = m_screw * rs ** 2 / 4 + m_screw * P ** 2 / n
J = P / tt * (2 * rs) ** 3 / 12
I_y = np.pi * (2 * rs) ** 4 / 64
I_z = np.pi * (2 * rs) ** 4 / 64

I_screw = [J, I_y, I_z]

# I_rigid = [1, 1, 1]
# I_rigid_roller = I_rigid
# I_rigid_screw = I_rigid
# I_rigid_nut = I_rigid

# 这里需要改成实际的螺纹牙变形刚度
# J = P * (2 * rr) ** 3 / 12
# I_y = np.pi * (2 * rr) ** 4 / 64
# I_z = np.pi * (2 * rr) ** 4 / 64
# I_rigid_roller = [J, I_y, I_z]
#
# J = P * (2 * rr) ** 3 / 12
# I_y = np.pi * (2 * rr) ** 4 / 64
# I_z = np.pi * (2 * rr) ** 4 / 64
# I_rigid_screw = [J, I_y, I_z]
#
# J = P * (2 * rr) ** 3 / 12
# I_y = np.pi * (2 * rr) ** 4 / 64
# I_z = np.pi * (2 * rr) ** 4 / 64
# I_rigid_nut = [J, I_y, I_z]

# 丝杠节点 （丝杠接触点节点，假设接触点增量相同，投影到同一点）
eng = matlab.engine.start_matlab()
# eng.sigangniehewucha_python(self.sc_filename, matlab.double([self.sc_mn]), nargout=0)
[xxs, yys, xxn, yyn, terr, terr1] = eng.sigangniehewucha_python_FEMvalid(n, tt, nargout=6)

# matlab中使用毫米制单位，转化为米制单位
xxs = np.array(xxs) * 1E-3
yys = np.array(yys) * 1E-3
xxn = np.array(xxn) * 1E-3
yyn = np.array(yyn) * 1E-3
Terr = np.array(terr) * 1E-3
Terr1 = np.array(terr1) * 1E-3
# print(xxs)
# print(Terr)

# print(Pr)
L_b = (n - 2) / 2 * Pr

e_rxi_max = 0.02 * 1E-3  # 米制单位
E_rxi = []
for i in range(n):
    if i * Pr <= L_b and i * Pr >= 0:
        e_rxi = e_rxi_max * ((i * Pr - L_b) / L_b) ** 2
        # print(1)
    elif i * Pr > L_b and i * Pr < n * Pr - L_b:
        e_rxi = 0
    elif i * Pr >= n * Pr - L_b and i * Pr <= n * Pr:
        e_rxi = e_rxi_max * ((i * Pr - n * Pr + L_b) / L_b) ** 2
    else:
        # print(4)
        pass

    E_rxi.append(e_rxi)

print(E_rxi)

# terr = np.zeros((tt, n))
# terr1 = np.zeros((tt, n))

# terr 是丝杠侧， terr1 是螺母侧
# print(xxs)
# print(yys)
# print(xxn)
# print(yyn)
# print(terr)
# print(terr1)

namespace = globals()

# K_v = 2E4  # 刚度是多少

n_0 = 2 / 3
alpha_sv = a1  # 丝杠侧
alpha_nv = a2  # 螺母侧 helix angle 螺旋角
alpha_rv = a3

# beta_v = np.deg2rad(45)  # flank angle 需要修改
beta_v = beta  # 滚柱牙侧角
# miu_v = 0.1  # coefficient of friction
# phi_v = np.arccos(np.cos(alpha_nv) * np.cos(beta_v) - miu_v * np.sin(beta_v))
# 滚柱与螺母接触点单元集合 (考虑螺纹牙变形)
# if epoch == 0 or epoch == 2:
xi_qs = 10.0  # curvature ratio 长短半轴之比
xi_qn = 10.0

R_qs = rr / np.sin(beta_v)  # equivalent radius of curvature
print(R_qs)
R_qn = rr / np.sin(beta_v)

# R_qs = 1E-2
# R_qn = 1E-2

for k in range(tt):
    terr[k] = (terr[k] - np.min(terr[k]))  # 假设螺母接触点在右侧
    terr1[k] = (terr1[k] - np.min(terr1[k]))

withMod = 0  # 1 for modification ,0 for no modification

if withMod == 1:
    Terr = Terr + E_rxi
    Terr1 = Terr1 + E_rxi
else:
    pass
# print(Terr)

Epoch = 3

F_0x = 3000
F_0y = -0
F_0z = -1000

M_0x = 0
M_0y = 0
M_0z = 0

T_C = 0  # T_C state=1; C_C state=0

ACC = 1E-1

# N_0s = 0
# N_0n = 0
for epoch in range(Epoch):  # 弹簧变刚度的大循环
    innerCycle = -1
    acc = 1E10  # 初始精度
    while acc > ACC * (1.5 ** (epoch + 1) / 1.5 ** Epoch) and innerCycle < 100:
        # while acc > ACC * (1.5 ** (epoch + 1) / 1.5 ** Epoch):

        innerCycle = innerCycle + 1
        # print(acc)
        F_x = F_0x * (1.5 ** (epoch + 1) / 1.5 ** Epoch)
        F_y = F_0y * (1.5 ** (epoch + 1) / 1.5 ** Epoch)
        F_z = F_0z * (1.5 ** (epoch + 1) / 1.5 ** Epoch)

        M_x = M_0x * (1.5 ** (epoch + 1) / 1.5 ** Epoch)
        M_y = M_0y * (1.5 ** (epoch + 1) / 1.5 ** Epoch)
        M_z = M_0z * (1.5 ** (epoch + 1) / 1.5 ** Epoch)

        s = []

        force_gz_sg_0_x = F_x / (n * tt)
        force_gz_lm_0_x = F_x / (n * tt)

        force_gz_sg_0_y = F_y / (n * tt)
        force_gz_lm_0_y = F_y / (n * tt)

        force_gz_sg_0_z = F_z / (n * tt)
        force_gz_lm_0_z = F_z / (n * tt)

        print(epoch, innerCycle)
        if epoch == 0 and innerCycle == 0:
            for j in range(tt):
                namespace['K_s_%d_x' % j] = np.zeros(n)
                namespace['K_s_%d_y' % j] = np.zeros(n)
                namespace['K_s_%d_z' % j] = np.zeros(n)

                namespace['K_n_%d_x' % j] = np.zeros(n)
                namespace['K_n_%d_y' % j] = np.zeros(n)
                namespace['K_n_%d_z' % j] = np.zeros(n)

            terr = np.zeros((tt, n))
            terr1 = np.zeros((tt, n))
            for j in range(tt):
                for i in range(n):
                    namespace['K_s_%d_x' % j][i] = abs(force_gz_sg_0_x) ** (1 / 3) * (
                            16 * E ** 2 * R_qs / (9 * np.cos(alpha_sv) * np.cos(beta_v))) ** (1 / 3) * (
                                                           1 / xi_qs)  # 第一根滚柱

                    namespace['K_s_%d_y' % j][i] = abs(force_gz_sg_0_y) ** (1 / 3) * abs(
                        16 * E ** 2 * R_qs / (9 * (
                                np.sin(alpha_sv) * np.sin(beta_v) * np.sin(2 * np.pi * j / tt) + np.cos(
                            alpha_sv) * np.sin(beta_v) * np.cos(2 * np.pi * j / tt)))) ** (1 / 3) * (
                                                           1 / xi_qs)  # 第一根滚柱

                    namespace['K_s_%d_z' % j][i] = abs(force_gz_sg_0_z) ** (1 / 3) * abs(
                        16 * E ** 2 * R_qs / (9 * (
                                np.sin(alpha_sv) * np.sin(beta_v) * np.cos(2 * np.pi * j / tt) - np.cos(
                            alpha_sv) * np.sin(beta_v) * np.sin(2 * np.pi * j / tt)))) ** (1 / 3) * (
                                                           1 / xi_qs)  # 第一根滚柱

                    namespace['K_n_%d_x' % j][i] = abs(force_gz_lm_0_x) ** (1 / 3) * (
                            16 * E ** 2 * R_qn / (9 * np.cos(alpha_nv) * np.cos(beta_v))) ** (1 / 3) * (
                                                           1 / xi_qn)

                    namespace['K_n_%d_y' % j][i] = abs(force_gz_lm_0_y) ** (1 / 3) * abs(
                        16 * E ** 2 * R_qn / (9 * (
                                np.sin(alpha_nv) * np.sin(beta_v) * np.sin(2 * np.pi * j / tt) + np.cos(
                            alpha_nv) * np.sin(beta_v) * np.cos(2 * np.pi * j / tt)))) ** (1 / 3) * (
                                                           1 / xi_qn)

                    namespace['K_n_%d_z' % j][i] = abs(force_gz_lm_0_z) ** (1 / 3) * abs(
                        16 * E ** 2 * R_qn / (9 * (
                                np.sin(alpha_nv) * np.sin(beta_v) * np.cos(2 * np.pi * j / tt) - np.cos(
                            alpha_nv) * np.sin(beta_v) * np.sin(2 * np.pi * j / tt)))) ** (1 / 3) * (
                                                           1 / xi_qn)

                # print(namespace['K_s_%d' % j])
                # print(namespace['K_n_%d' % j])
        else:
            terr = Terr
            terr1 = Terr1
            # terr = np.zeros((tt, n))
            # terr1 = np.zeros((tt, n))
            # for j in range(tt):
            # namespace['K_s_%d' % j] = []
            # namespace['K_n_%d' % j] = []
            for j in range(tt):
                for i in range(n):
                    # print([nd.disp["Ux"] for nd in namespace['nds_gz_c_%d_s' % j]][i])
                    disp_gz_sg_x = [nd.disp["Ux"] for nd in namespace['nds_gz_c_%d_s' % j]][i] - \
                                   [nd.disp["Ux"] for nd in nds_sg_c][tt * i + j] + terr[j][i]
                    disp_gz_sg_x0 = [nd.disp["Ux"] for nd in namespace['nds_gz_c_%d_s' % j]][i] - \
                                    [nd.disp["Ux"] for nd in nds_sg_c][tt * i + j]
                    disp_gz_sg_y = [nd.disp["Uy"] for nd in namespace['nds_gz_c_%d_s' % j]][i] - \
                                   [nd.disp["Uy"] for nd in nds_sg_c][tt * i + j] + terr[j][i] / np.sin(
                        2 * np.pi * j / tt + 1E-5)  # 加上1E-5是为了避免除0
                    disp_gz_sg_y0 = [nd.disp["Uy"] for nd in namespace['nds_gz_c_%d_s' % j]][i] - \
                                    [nd.disp["Uy"] for nd in nds_sg_c][tt * i + j]
                    disp_gz_sg_z = [nd.disp["Uz"] for nd in namespace['nds_gz_c_%d_s' % j]][i] - \
                                   [nd.disp["Uz"] for nd in nds_sg_c][tt * i + j] + terr[j][i] / np.cos(
                        2 * np.pi * j / tt + 1E-5)  # 真实间隙要换算
                    disp_gz_sg_z0 = [nd.disp["Uz"] for nd in namespace['nds_gz_c_%d_s' % j]][i] - \
                                    [nd.disp["Uz"] for nd in nds_sg_c][tt * i + j]
                    # print(np.array([el.force["N"] for el in namespace['els_sz_%d' % j]]))
                    force_gz_sg_x = np.array([el.force["N"] for el in namespace['els_sz_%d_x' % j]])[i][0][0]
                    force_gz_sg_y = np.array([el.force["N"] for el in namespace['els_sz_%d_y' % j]])[i][0][0]
                    force_gz_sg_z = np.array([el.force["N"] for el in namespace['els_sz_%d_z' % j]])[i][0][0]
                    # 不是disp而是绝对位置
                    # print(force_gz_sg_x,"force_gz_sg_x")
                    # print(disp_gz_sg)
                    force_gz_sg = np.sqrt(force_gz_sg_x ** 2 + force_gz_sg_y ** 2 + force_gz_sg_z ** 2)

                    # force_gz_sg_x = force_gz_sg*np.sin(beta_v)*np.cos(alpha_nv)
                    force_gz_sg_x = force_gz_sg * np.sin(beta_v)
                    force_gz_sg_y = force_gz_sg * (
                            np.sin(alpha_sv) * np.sin(beta_v) * np.sin(2 * np.pi * j / tt) + np.cos(
                        alpha_sv) * np.sin(beta_v) * np.cos(2 * np.pi * j / tt))
                    force_gz_sg_z = force_gz_sg * (
                            np.sin(alpha_sv) * np.sin(beta_v) * np.cos(2 * np.pi * j / tt) - np.cos(
                        alpha_sv) * np.sin(beta_v) * np.sin(2 * np.pi * j / tt))

                    if disp_gz_sg_x < 0 and disp_gz_sg_x0 != 0:
                        # namespace['K_s_%d' % j][i] = 1 / (np.cos(phi_v) ** (n_0)) * K_v * abs(disp_gz_sg) ** (
                        #         n_0 - 1)  # 第一根滚柱
                        if force_gz_sg_x == 0:
                            force_gz_sg_x = 1E-5
                            # force_gz_sg = force_gz_sg_0

                        namespace['K_s_%d_x' % j][i] = abs(force_gz_sg_x) ** (1 / 3) * (
                                16 * E ** 2 * R_qs / (9 * np.cos(alpha_sv) * np.cos(beta_v))) ** (1 / 3) * (
                                                               1 / xi_qs) * abs(disp_gz_sg_x / disp_gz_sg_x0)  # 第一根滚柱

                    else:
                        namespace['K_s_%d_x' % j][i] = 1E-5

                    if disp_gz_sg_y * np.sin(2 * np.pi * j / tt) < 0 and disp_gz_sg_y0 != 0:  # 只是粗略的判断
                        if force_gz_sg_y == 0:
                            force_gz_sg_y = 1E-5

                        namespace['K_s_%d_y' % j][i] = abs(force_gz_sg_y) ** (1 / 3) * abs(
                            16 * E ** 2 * R_qs / (9 * (
                                    np.sin(alpha_sv) * np.sin(beta_v) * np.sin(2 * np.pi * j / tt) + np.cos(
                                alpha_sv) * np.sin(beta_v) * np.cos(2 * np.pi * j / tt)))) ** (1 / 3) * (
                                                               1 / xi_qs) * abs(disp_gz_sg_y / disp_gz_sg_y0)  # 第一根滚柱
                    else:
                        namespace['K_s_%d_y' % j][i] = 1E-5

                    if disp_gz_sg_z * np.cos(2 * np.pi * j / tt) < 0 and disp_gz_sg_z0 != 0:
                        if force_gz_sg_z == 0:
                            force_gz_sg_z = 1E-5

                        namespace['K_s_%d_z' % j][i] = abs(force_gz_sg_z) ** (1 / 3) * abs(
                            16 * E ** 2 * R_qs / (9 * (
                                    np.sin(alpha_sv) * np.sin(beta_v) * np.cos(2 * np.pi * j / tt) - np.cos(
                                alpha_sv) * np.sin(beta_v) * np.sin(2 * np.pi * j / tt)))) ** (1 / 3) * (
                                                               1 / xi_qs) * abs(disp_gz_sg_z / disp_gz_sg_z0)  # 第一根滚柱
                    else:
                        namespace['K_s_%d_z' % j][i] = 1E-5

                    disp_gz_lm_x = [nd.disp["Ux"] for nd in namespace['nds_lm_c_%d' % j]][i] - \
                                   [nd.disp["Ux"] for nd in namespace['nds_gz_c_%d_n' % j]][i] + terr1[j][i]
                    disp_gz_lm_x0 = [nd.disp["Ux"] for nd in namespace['nds_lm_c_%d' % j]][i] - \
                                    [nd.disp["Ux"] for nd in namespace['nds_gz_c_%d_n' % j]][i]
                    disp_gz_lm_y = [nd.disp["Uy"] for nd in namespace['nds_lm_c_%d' % j]][i] - \
                                   [nd.disp["Uy"] for nd in namespace['nds_gz_c_%d_n' % j]][i] + terr1[j][i] / np.sin(
                        2 * np.pi * j / tt + 1E-5)
                    disp_gz_lm_y0 = [nd.disp["Uy"] for nd in namespace['nds_lm_c_%d' % j]][i] - \
                                    [nd.disp["Uy"] for nd in namespace['nds_gz_c_%d_n' % j]][i]
                    disp_gz_lm_z = [nd.disp["Uz"] for nd in namespace['nds_lm_c_%d' % j]][i] - \
                                   [nd.disp["Uz"] for nd in namespace['nds_gz_c_%d_n' % j]][i] + terr1[j][i] / np.cos(
                        2 * np.pi * j / tt + 1E-5)
                    disp_gz_lm_z0 = [nd.disp["Uz"] for nd in namespace['nds_lm_c_%d' % j]][i] - \
                                    [nd.disp["Uz"] for nd in namespace['nds_gz_c_%d_n' % j]][i]

                    force_gz_lm_x = np.array([el.force["N"] for el in namespace['els_lz_%d_x' % j]])[i][0][0]
                    force_gz_lm_y = np.array([el.force["N"] for el in namespace['els_lz_%d_y' % j]])[i][0][0]
                    force_gz_lm_z = np.array([el.force["N"] for el in namespace['els_lz_%d_z' % j]])[i][0][0]

                    force_gz_lm = np.sqrt(force_gz_lm_x ** 2 + force_gz_lm_y ** 2 + force_gz_lm_z ** 2)

                    # force_gz_lm_x = force_gz_lm*np.sin(beta_v)*np.cos(alpha_nv)
                    force_gz_lm_x = force_gz_lm * np.cos(alpha_nv) * np.cos(beta_v)
                    force_gz_lm_y = force_gz_lm * (
                            np.sin(alpha_nv) * np.sin(beta_v) * np.sin(2 * np.pi * j / tt) + np.cos(
                        alpha_nv) * np.sin(beta_v) * np.cos(2 * np.pi * j / tt))
                    force_gz_lm_z = force_gz_lm * (
                            np.sin(alpha_nv) * np.sin(beta_v) * np.cos(2 * np.pi * j / tt) - np.cos(
                        alpha_nv) * np.sin(beta_v) * np.sin(2 * np.pi * j / tt))

                    if disp_gz_lm_x < 0 and disp_gz_lm_x0 != 0:
                        # namespace['K_n_%d' % j][i] = 1 / (np.cos(phi_v) ** (n_0)) * K_v * abs(disp_gz_lm) ** (
                        #         n_0 - 1)  # 第一根滚柱
                        if force_gz_lm_x == 0:
                            force_gz_lm_x = 1E-5
                            # force_gz_lm = force_gz_lm_0

                        namespace['K_n_%d_x' % j][i] = abs(force_gz_lm_x) ** (1 / 3) * (
                                16 * E ** 2 * R_qn / (9 * np.cos(alpha_nv) * np.cos(beta_v))) ** (1 / 3) * (
                                                               1 / xi_qn) * abs(disp_gz_lm_x / disp_gz_lm_x0)  # 第一根滚柱
                    else:
                        namespace['K_n_%d_x' % j][i] = 1E-5

                    if disp_gz_lm_y * np.sin(2 * np.pi * j / tt) < 0 and disp_gz_lm_y0 != 0:
                        if force_gz_lm_y == 0:
                            force_gz_lm_y = 1E-5

                        namespace['K_n_%d_y' % j][i] = abs(force_gz_lm_y) ** (1 / 3) * abs(
                            16 * E ** 2 * R_qn / (9 * (
                                    np.sin(alpha_nv) * np.sin(beta_v) * np.sin(2 * np.pi * j / tt) + np.cos(
                                alpha_nv) * np.sin(beta_v) * np.cos(2 * np.pi * j / tt)))) ** (1 / 3) * (
                                                               1 / xi_qn) * abs(
                            disp_gz_lm_y / disp_gz_lm_y0)  # y方向作用力迭代如何表示
                    else:
                        namespace['K_n_%d_y' % j][i] = 1E-5

                    if disp_gz_lm_z * np.cos(2 * np.pi * j / tt) < 0 and disp_gz_lm_z0 != 0:
                        if force_gz_lm_z == 0:
                            force_gz_lm_z = 1E-5

                        namespace['K_n_%d_z' % j][i] = abs(force_gz_lm_z) ** (1 / 3) * abs(
                            16 * E ** 2 * R_qn / (9 * (
                                    np.sin(alpha_nv) * np.sin(beta_v) * np.cos(2 * np.pi * j / tt) - np.cos(
                                alpha_nv) * np.sin(beta_v) * np.sin(2 * np.pi * j / tt)))) ** (1 / 3) * (
                                                               1 / xi_qn) * abs(
                            disp_gz_lm_z / disp_gz_lm_z0)  # 平均以提高收敛性
                    else:
                        namespace['K_n_%d_z' % j][i] = 1E-5

                # print(namespace['K_s_%d_z' % j])
                # print([nd.disp["Ux"] for nd in namespace['nds_gz_c_%d_n' % j]][i])

        # for j in range(tt):
        #     namespace['K_s_%d_y' % j] = np.zeros(n)
        #     namespace['K_s_%d_z' % j] = np.zeros(n)
        #
        #     namespace['K_n_%d_y' % j] = np.zeros(n)
        #     namespace['K_n_%d_z' % j] = np.zeros(n)

        for j in range(tt):
            namespace['els_sz_%d_x' % j] = []  # 丝杠与滚柱接触单元集合
            namespace['els_sz_%d_y' % j] = []
            namespace['els_sz_%d_z' % j] = []

            namespace['els_lz_%d_x' % j] = []  # 螺母与滚柱接触单元集合
            namespace['els_lz_%d_y' % j] = []
            namespace['els_lz_%d_z' % j] = []

        for i in range(n):
            for j in range(tt):
                namespace['sg_%d_%d' % (i, j)] = Node(i * P + j * 1 / tt * P, 0, 0)
                # print(namespace['sg_%si_%sj' % (i, j)])

        x_sg = []
        y_sg = []
        z_sg = []

        for k in range(tt):
            x_sg.append(P / 4 - (terr[k]))  # 假设丝杠接触点在左侧
            # 将间隙最小值设为0
            # y_sg.append(1 / 2 * (rs + rr) * np.sin(1 / tt * 2 * np.pi * k))
            # z_sg.append(1 / 2 * (rs + rr) * np.cos(1 / tt * 2 * np.pi * k))
            y_sg.append((xxs[k]) * np.sin(1 / tt * 2 * np.pi * k) + (yys[k]) * np.cos(1 / tt * 2 * np.pi * k))
            z_sg.append((xxs[k]) * np.cos(1 / tt * 2 * np.pi * k) - (yys[k]) * np.sin(1 / tt * 2 * np.pi * k))
        # 接触点位置的节点
        for i in range(n):
            for j in range(tt):
                namespace['sg_%d_%d_c' % (i, j)] = Node(i * P + j * 1 / tt * P + x_sg[j][i], 0 + y_sg[j][i],
                                                        0 + z_sg[j][i])

        # 螺母节点

        for i in range(n):
            for j in range(tt):
                namespace['lm_%d_%d' % (i, j)] = Node(i * P + j * 1 / tt * P, rn * np.sin(1 / tt * 2 * np.pi * j),
                                                      rn * np.cos(1 / tt * 2 * np.pi * j))

        x_lm = []
        y_lm = []
        z_lm = []
        for k in range(tt):
            x_lm.append(P / 4 + (terr1[k]))  # 假设螺母接触点在右侧
            # y_lm.append(-(rn - (rs + rr)) * np.sin(1 / tt * 2 * np.pi * j))
            # z_lm.append(-(rn - (rs + rr)) * np.cos(1 / tt * 2 * np.pi * j))
            y_lm.append((xxn[k]) * np.sin(1 / tt * 2 * np.pi * k) + (yyn[k]) * np.cos(1 / tt * 2 * np.pi * k))
            z_lm.append((xxn[k]) * np.cos(1 / tt * 2 * np.pi * k) - (yyn[k]) * np.sin(1 / tt * 2 * np.pi * k))

        for i in range(n):
            for j in range(tt):
                namespace['lm_%d_%d_c' % (i, j)] = Node(i * P + j * 1 / tt * P + x_lm[j][i],
                                                        0 + y_lm[j][i],
                                                        0 + z_lm[j][i])

        # 滚柱节点

        for i in range(n):
            for j in range(tt):
                namespace['gz_%d_%d' % (i, j)] = Node(i * P + j * 1 / tt * P,
                                                      (rs + rr) * np.sin(1 / tt * 2 * np.pi * j),
                                                      (rs + rr) * np.cos(1 / tt * 2 * np.pi * j))
        # 滚柱（丝杠侧边节点）
        x_gz_s = []
        y_gz_s = []
        z_gz_s = []
        for k in range(tt):
            x_gz_s.append(P / 4 + np.zeros(n))
            y_gz_s.append((xxs[k]) * np.sin(1 / tt * 2 * np.pi * k) + (yys[k]) * np.cos(1 / tt * 2 * np.pi * k))
            z_gz_s.append((xxs[k]) * np.cos(1 / tt * 2 * np.pi * k) - (yys[k]) * np.sin(1 / tt * 2 * np.pi * k))

        for i in range(n):
            for j in range(tt):
                namespace['gz_%d_%d_c_s' % (i, j)] = Node(i * P + j * 1 / tt * P + x_gz_s[j][i],
                                                          0 + y_gz_s[j][i],
                                                          0 + z_gz_s[j][i])

        # 滚柱（螺母侧边节点）
        x_gz_n = []
        y_gz_n = []
        z_gz_n = []
        for k in range(tt):
            x_gz_n.append(P / 4 + np.zeros(n))
            y_gz_n.append((xxn[k]) * np.sin(1 / tt * 2 * np.pi * k) + (yyn[k]) * np.cos(1 / tt * 2 * np.pi * k))
            z_gz_n.append((xxn[k]) * np.cos(1 / tt * 2 * np.pi * k) - (yyn[k]) * np.sin(1 / tt * 2 * np.pi * k))

        for i in range(n):
            for j in range(tt):
                namespace['gz_%d_%d_c_n' % (i, j)] = Node(i * P + j * 1 / tt * P + x_gz_n[j][i],
                                                          0 + y_gz_n[j][i],
                                                          0 + z_gz_n[j][i])

        # 丝杠节点集合
        nds_sg = []
        for i in range(n):
            for j in range(tt):
                nds_sg.append(namespace['sg_%d_%d' % (i, j)])

        # 丝杠接触点集合
        nds_sg_c = []
        for i in range(n):
            for j in range(tt):
                nds_sg_c.append(namespace['sg_%d_%d_c' % (i, j)])

        # 螺母节点集合
        for j in range(tt):
            namespace['nds_lm_%d' % j] = []

        for i in range(tt):
            for j in range(n):
                namespace['nds_lm_%d' % i].append(namespace['lm_%d_%d' % (j, i)])

        # 螺母接触点集合
        for j in range(tt):
            namespace['nds_lm_c_%d' % j] = []

        for i in range(tt):
            for j in range(n):
                namespace['nds_lm_c_%d' % i].append(namespace['lm_%d_%d_c' % (j, i)])

        # 滚柱节点集合
        for j in range(tt):
            namespace['nds_gz_%d' % j] = []

        for i in range(tt):
            for j in range(n):
                namespace['nds_gz_%d' % i].append(namespace['gz_%d_%d' % (j, i)])

        # 滚柱接触点集合 （丝杠侧）
        for j in range(tt):
            namespace['nds_gz_c_%d_s' % j] = []

        for i in range(tt):
            for j in range(n):
                namespace['nds_gz_c_%d_s' % i].append(namespace['gz_%d_%d_c_s' % (j, i)])

        # 滚柱接触点集合 （螺母侧）
        for j in range(tt):
            namespace['nds_gz_c_%d_n' % j] = []

        for i in range(tt):
            for j in range(n):
                namespace['nds_gz_c_%d_n' % i].append(namespace['gz_%d_%d_c_n' % (j, i)])

        # 丝杠单元集合
        els_sg = []
        els_sg_c = []

        for j in range(tt):
            namespace['els_gz_%d' % j] = []
            namespace['els_gz_c_%d_s' % j] = []
            namespace['els_gz_c_%d_n' % j] = []

        for j in range(tt):
            namespace['els_lm_%d' % j] = []
            namespace['els_lm_c_%d' % j] = []

        # 丝杠单元集合
        for nd in pair_wise(nds_sg, True):  # 丝杠
            els_sg.append(Beam3D11(nd, E, G, A_screw, I_screw))

        # 丝杠与丝杠接触点单元集合 (考虑螺纹牙变形)
        for i in range(n):
            for j in range(tt):
                els_sg_c.append(
                    Spring3D11_xdof((nds_sg[i + j * n], nds_sg_c[i + j * n]), KST0 * np.cos(alpha_rv) * np.cos(beta_v)))
                els_sg_c.append(
                    Spring3D11_ydof((nds_sg[i + j * n], nds_sg_c[i + j * n]),
                                    KST0 * (np.sin(alpha_rv) * np.sin(beta_v) * np.sin(2 * np.pi * j / tt) + np.cos(
                                        alpha_rv) * np.sin(beta_v) * np.cos(2 * np.pi * j / tt))))
                els_sg_c.append(
                    Spring3D11_zdof((nds_sg[i + j * n], nds_sg_c[i + j * n]),
                                    KST0 * (
                                            np.sin(alpha_rv) * np.sin(beta_v) * np.cos(2 * np.pi * j / tt) - np.cos(
                                        alpha_rv) * np.sin(beta_v) * np.sin(2 * np.pi * j / tt))))

                # 螺母单元集合
        for j in range(tt):
            for nd in pair_wise(namespace['nds_lm_%d' % j], True):  # 螺母1
                namespace['els_lm_%d' % j].append(Link3D11(nd, E, A_nut))
                # namespace['els_lm_%d' % j].append(Beam3D11(nd, E, G, A_nut, I_nut))
        # 螺母与螺母接触点单元集合 (考虑螺纹牙变形)
        for i in range(n):
            for j in range(tt):
                namespace['els_lm_c_%d' % j].append(
                    Spring3D11_xdof((namespace['nds_lm_%d' % j][i], namespace['nds_lm_c_%d' % j][i]),
                                    KNT0 * np.cos(alpha_rv) * np.cos(beta_v)))
                namespace['els_lm_c_%d' % j].append(
                    Spring3D11_ydof((namespace['nds_lm_%d' % j][i], namespace['nds_lm_c_%d' % j][i]),
                                    KNT0 * np.sin(beta_v) * (
                                            np.sin(alpha_rv) * np.sin(beta_v) * np.sin(2 * np.pi * j / tt) + np.cos(
                                        alpha_rv) * np.sin(beta_v) * np.cos(2 * np.pi * j / tt))))
                namespace['els_lm_c_%d' % j].append(
                    Spring3D11_zdof((namespace['nds_lm_%d' % j][i], namespace['nds_lm_c_%d' % j][i]),
                                    KNT0 * (
                                            np.sin(alpha_rv) * np.sin(beta_v) * np.cos(2 * np.pi * j / tt) - np.cos(
                                        alpha_rv) * np.sin(beta_v) * np.sin(2 * np.pi * j / tt))))

        # 滚柱单元集合
        for j in range(tt):
            for nd in pair_wise(namespace['nds_gz_%d' % j], True):  # 滚柱#1
                namespace['els_gz_%d' % j].append(Beam3D11(nd, E, G, A_roller, I_roller))

        # 滚柱与丝杠接触点单元集合 (考虑螺纹牙变形)
        for i in range(n):
            for j in range(tt):
                namespace['els_gz_c_%d_s' % j].append(
                    Spring3D11_xdof((namespace['nds_gz_%d' % j][i], namespace['nds_gz_c_%d_s' % j][i]),
                                    KRST0 * np.cos(alpha_rv) * np.cos(beta_v)))  # 滚柱和丝杠接触
                namespace['els_gz_c_%d_s' % j].append(
                    Spring3D11_ydof((namespace['nds_gz_%d' % j][i], namespace['nds_gz_c_%d_s' % j][i]),
                                    KRST0 * np.sin(beta_v) * (
                                            np.sin(alpha_rv) * np.sin(beta_v) * np.sin(2 * np.pi * j / tt) + np.cos(
                                        alpha_rv) * np.sin(beta_v) * np.cos(2 * np.pi * j / tt))))
                namespace['els_gz_c_%d_s' % j].append(
                    Spring3D11_zdof((namespace['nds_gz_%d' % j][i], namespace['nds_gz_c_%d_s' % j][i]),
                                    KRST0 * (
                                            np.sin(alpha_rv) * np.sin(beta_v) * np.cos(2 * np.pi * j / tt) - np.cos(
                                        alpha_rv) * np.sin(beta_v) * np.sin(2 * np.pi * j / tt))))

                namespace['els_gz_c_%d_n' % j].append(
                    Spring3D11_xdof((namespace['nds_gz_%d' % j][i], namespace['nds_gz_c_%d_n' % j][i]),
                                    KRNT0 * np.cos(alpha_rv) * np.cos(beta_v)))  # 滚柱和螺母接触
                namespace['els_gz_c_%d_n' % j].append(
                    Spring3D11_ydof((namespace['nds_gz_%d' % j][i], namespace['nds_gz_c_%d_n' % j][i]),
                                    KRNT0 * np.sin(beta_v) * (
                                            np.sin(alpha_rv) * np.sin(beta_v) * np.sin(2 * np.pi * j / tt) + np.cos(
                                        alpha_rv) * np.sin(beta_v) * np.cos(2 * np.pi * j / tt))))
                namespace['els_gz_c_%d_n' % j].append(
                    Spring3D11_zdof((namespace['nds_gz_%d' % j][i], namespace['nds_gz_c_%d_n' % j][i]),
                                    KRNT0 * (
                                            np.sin(alpha_rv) * np.sin(beta_v) * np.cos(2 * np.pi * j / tt) - np.cos(
                                        alpha_rv) * np.sin(beta_v) * np.sin(2 * np.pi * j / tt))))

        # 滚柱1与丝杠、螺母的作用
        # for i in range(n):
        #     for j in range(tt):
        #         namespace['els_sz_%d' % j].append(Spring3D11((nds_sg[tt * i], namespace['els_gz_%d' % j][i]), K))
        #
        # for i in range(n):
        #     for j in range(tt):
        #         namespace['els_lz_%d' % j].append(
        #             Spring3D11((namespace['nds_lm_%d' % j][i], namespace['nds_gz_%d' % j][i]), K))

        for i in range(n):  # 丝杠与滚柱   使用1D单元增强收敛性(如何表示偏斜时的受力？)
            els_sz_0_x.append(Spring3D11_xdof((nds_sg_c[tt * i + 0], nds_gz_c_0_s[i]), K_s_0_x[i]))
            els_sz_0_y.append(Spring3D11_ydof((nds_sg_c[tt * i + 0], nds_gz_c_0_s[i]), K_s_0_y[i]))
            els_sz_0_z.append(Spring3D11_zdof((nds_sg_c[tt * i + 0], nds_gz_c_0_s[i]), K_s_0_z[i]))

        for i in range(n):  # 螺母与滚柱
            els_lz_0_x.append(Spring3D11_xdof((nds_lm_c_0[i], nds_gz_c_0_n[i]), K_n_0_x[i]))
            els_lz_0_y.append(Spring3D11_ydof((nds_lm_c_0[i], nds_gz_c_0_n[i]), K_n_0_y[i]))
            els_lz_0_z.append(Spring3D11_zdof((nds_lm_c_0[i], nds_gz_c_0_n[i]), K_n_0_z[i]))

        # 滚柱2与丝杠、螺母的作用
        for i in range(n):
            els_sz_1_x.append(Spring3D11_xdof((nds_sg_c[tt * i + 1], nds_gz_c_1_s[i]), K_s_1_x[i]))
            els_sz_1_y.append(Spring3D11_ydof((nds_sg_c[tt * i + 1], nds_gz_c_1_s[i]), K_s_1_y[i]))
            els_sz_1_z.append(Spring3D11_zdof((nds_sg_c[tt * i + 1], nds_gz_c_1_s[i]), K_s_1_z[i]))

        for i in range(n):
            els_lz_1_x.append(Spring3D11_xdof((nds_lm_c_1[i], nds_gz_c_1_n[i]), K_n_1_x[i]))
            els_lz_1_y.append(Spring3D11_ydof((nds_lm_c_1[i], nds_gz_c_1_n[i]), K_n_1_y[i]))
            els_lz_1_z.append(Spring3D11_zdof((nds_lm_c_1[i], nds_gz_c_1_n[i]), K_n_1_z[i]))

        # 滚柱3与丝杠、螺母的作用

        for i in range(n):
            els_sz_2_x.append(Spring3D11_xdof((nds_sg_c[tt * i + 2], nds_gz_c_2_s[i]), K_s_2_x[i]))
            els_sz_2_y.append(Spring3D11_ydof((nds_sg_c[tt * i + 2], nds_gz_c_2_s[i]), K_s_2_y[i]))
            els_sz_2_z.append(Spring3D11_zdof((nds_sg_c[tt * i + 2], nds_gz_c_2_s[i]), K_s_2_z[i]))

        for i in range(n):
            els_lz_2_x.append(Spring3D11_xdof((nds_lm_c_2[i], nds_gz_c_2_n[i]), K_n_2_x[i]))
            els_lz_2_y.append(Spring3D11_ydof((nds_lm_c_2[i], nds_gz_c_2_n[i]), K_n_2_y[i]))
            els_lz_2_z.append(Spring3D11_zdof((nds_lm_c_2[i], nds_gz_c_2_n[i]), K_n_2_z[i]))

        # 滚柱4与丝杠、螺母的作用

        for i in range(n):
            els_sz_3_x.append(Spring3D11_xdof((nds_sg_c[tt * i + 3], nds_gz_c_3_s[i]), K_s_3_x[i]))
            els_sz_3_y.append(Spring3D11_ydof((nds_sg_c[tt * i + 3], nds_gz_c_3_s[i]), K_s_3_y[i]))
            els_sz_3_z.append(Spring3D11_zdof((nds_sg_c[tt * i + 3], nds_gz_c_3_s[i]), K_s_3_z[i]))

        for i in range(n):
            els_lz_3_x.append(Spring3D11_xdof((nds_lm_c_3[i], nds_gz_c_3_n[i]), K_n_3_x[i]))
            els_lz_3_y.append(Spring3D11_ydof((nds_lm_c_3[i], nds_gz_c_3_n[i]), K_n_3_y[i]))
            els_lz_3_z.append(Spring3D11_zdof((nds_lm_c_3[i], nds_gz_c_3_n[i]), K_n_3_z[i]))

        # 滚柱5与丝杠、螺母的作用

        for i in range(n):
            els_sz_4_x.append(Spring3D11_xdof((nds_sg_c[tt * i + 4], nds_gz_c_4_s[i]), K_s_4_x[i]))
            els_sz_4_y.append(Spring3D11_ydof((nds_sg_c[tt * i + 4], nds_gz_c_4_s[i]), K_s_4_y[i]))
            els_sz_4_z.append(Spring3D11_zdof((nds_sg_c[tt * i + 4], nds_gz_c_4_s[i]), K_s_4_z[i]))

        for i in range(n):
            els_lz_4_x.append(Spring3D11_xdof((nds_lm_c_4[i], nds_gz_c_4_n[i]), K_n_4_x[i]))
            els_lz_4_y.append(Spring3D11_ydof((nds_lm_c_4[i], nds_gz_c_4_n[i]), K_n_4_y[i]))
            els_lz_4_z.append(Spring3D11_zdof((nds_lm_c_4[i], nds_gz_c_4_n[i]), K_n_4_z[i]))

        # 滚柱6与丝杠、螺母的作用

        for i in range(n):
            els_sz_5_x.append(Spring3D11_xdof((nds_sg_c[tt * i + 5], nds_gz_c_5_s[i]), K_s_5_x[i]))
            els_sz_5_y.append(Spring3D11_ydof((nds_sg_c[tt * i + 5], nds_gz_c_5_s[i]), K_s_5_y[i]))
            els_sz_5_z.append(Spring3D11_zdof((nds_sg_c[tt * i + 5], nds_gz_c_5_s[i]), K_s_5_z[i]))

        for i in range(n):
            els_lz_5_x.append(Spring3D11_xdof((nds_lm_c_5[i], nds_gz_c_5_n[i]), K_n_5_x[i]))
            els_lz_5_y.append(Spring3D11_ydof((nds_lm_c_5[i], nds_gz_c_5_n[i]), K_n_5_y[i]))
            els_lz_5_z.append(Spring3D11_zdof((nds_lm_c_5[i], nds_gz_c_5_n[i]), K_n_5_z[i]))

        s = System()

        s.add_nodes(nds_sg, nds_sg_c, nds_lm_0, nds_lm_1, nds_lm_2, nds_lm_3, nds_lm_4, nds_lm_5, nds_lm_c_0,
                    nds_lm_c_1,
                    nds_lm_c_2, nds_lm_c_3, nds_lm_c_4, nds_lm_c_5, nds_gz_0, nds_gz_1, nds_gz_2, nds_gz_3, nds_gz_4,
                    nds_gz_5, nds_gz_c_0_n, nds_gz_c_1_n, nds_gz_c_2_n, nds_gz_c_3_n, nds_gz_c_4_n, nds_gz_c_5_n,
                    nds_gz_c_0_s, nds_gz_c_1_s, nds_gz_c_2_s, nds_gz_c_3_s, nds_gz_c_4_s, nds_gz_c_5_s)
        s.add_elements(els_sg, els_sg_c, els_gz_0, els_gz_1, els_gz_2, els_gz_3, els_gz_4, els_gz_5, els_gz_c_0_s,
                       els_gz_c_1_s, els_gz_c_2_s, els_gz_c_3_s, els_gz_c_4_s, els_gz_c_5_s, els_gz_c_0_n, els_gz_c_1_n,
                       els_gz_c_2_n, els_gz_c_3_n, els_gz_c_4_n, els_gz_c_5_n, els_lm_0, els_lm_1, els_lm_2, els_lm_3,
                       els_lm_4, els_lm_5, els_lm_c_0, els_lm_c_1, els_lm_c_2, els_lm_c_3, els_lm_c_4, els_lm_c_5,
                       els_sz_0_x, els_sz_0_y, els_sz_0_z, els_sz_1_x, els_sz_1_y, els_sz_1_z, els_sz_2_x, els_sz_2_y,
                       els_sz_2_z, els_sz_3_x, els_sz_3_y, els_sz_3_z, els_sz_4_x, els_sz_4_y, els_sz_4_z, els_sz_5_x,
                       els_sz_5_y, els_sz_5_z, els_lz_0_x, els_lz_0_y, els_lz_0_z, els_lz_1_x, els_lz_1_y, els_lz_1_z,
                       els_lz_2_x, els_lz_2_y, els_lz_2_z, els_lz_3_x, els_lz_3_y, els_lz_3_z, els_lz_4_x, els_lz_4_y,
                       els_lz_4_z, els_lz_5_x, els_lz_5_y, els_lz_5_z)

        # 施加作用力
        # for k in range(tt):
        #     s.add_node_force(nds_sg[tt].ID, Fx=3000)
        s.add_node_force(nds_sg[0].ID, Mx=M_x, Fx=F_x, Fy=F_y / 2, Fz=F_z / 2)
        s.add_node_force(nds_sg[-1].ID, Fy=F_y / 2, Fz=F_z / 2)

        # s.add_node_force(nds_sg[0].ID, Fx=F_x, Fy=-416, Fz=F_z / 2)
        # s.add_node_force(nds_sg[-1].ID, Fy=416, Fz=F_z / 2)

        # s.add_node_force(nds_sg[0].ID, Fx=10, Fy=10, Fz=10)
        s.add_node_force(nds_sg[int(n * tt / 2)].ID, My=M_y, Mz=M_z)  # 施加转矩(倾覆力矩)
        # s.add_node_disp(nds_sg[36].ID, Uy=0, Uz=0, Phx=0, Phy=0.5, Phz=0)
        # 螺母边界条件
        for k in range(tt):
            for nd in namespace['nds_lm_%d' % k]:
                s.add_node_disp(nd.ID, Uy=0, Uz=0, Phx=0, Phy=0, Phz=0)

        # 螺母接触点边界条件
        # for k in range(tt):
        #     for nd in namespace['nds_lm_c_%d' % k]:
        #         s.add_node_disp(nd.ID, Phx=0, Phy=0, Phz=0)

        # s.add_fixed_sup(nds_lm_0[0].ID, nds_lm_1[0].ID, nds_lm_2[0].ID, nds_lm_3[0].ID, nds_lm_4[0].ID, nds_lm_5[0].ID)
        for k in range(tt):
            if T_C == 1:
                s.add_node_disp(namespace['nds_lm_%d' % k][0].ID, Ux=0)
            else:
                s.add_node_disp(namespace['nds_lm_%d' % k][-1].ID, Ux=0)
        # 丝杠边界条件
        # for nd in nds_sg:
        #     if nd.ID != nds_sg[-1].ID and nd.ID != nds_sg[0].ID:
        #         s.add_node_disp(nd.ID, Phx=0, Phy=0, Phz=0)
        # for nd in nds_sg:
        #     if nd.ID != nds_sg[-1].ID and nd.ID != nds_sg[0].ID:
        #         s.add_node_disp(nd.ID, Phx=0)

        # s.add_node_disp(nds_sg[-1].ID, Uy=0, Uz=0)
        s.add_node_disp(nds_sg[0].ID, Phx=0)
        s.add_node_disp(nds_sg[-1].ID, Phx=0)

        # 丝杠接触点边界条件
        # for nd in nds_sg_c:
        #     s.add_node_disp(nd.ID, Phx=0, Phy=0, Phz=0)

        # 滚柱边界条件
        for k in range(tt):
            s.add_node_disp(namespace['nds_gz_%d' % k][0].ID, Phx=0, Uy=0, Uz=0)
            s.add_node_disp(namespace['nds_gz_%d' % k][-1].ID, Phx=0, Uy=0, Uz=0)
            #     # 不考虑滚柱与保持架的接触
            # for nd in namespace['nds_gz_%d' % k]:
            #     # pass
            #     if nd.ID != namespace['nds_gz_%d' % k][-1].ID and nd.ID != namespace['nds_gz_%d' % k][0].ID:
            #         s.add_node_disp(nd.ID, Phx=0, Phy=0, Phz=0)  # 影响多根滚柱的载荷分配
        #         if nd.ID != namespace['nds_gz_%d' % k][-1].ID and nd.ID != namespace['nds_gz_%d' % k][0].ID:
        #             s.add_node_disp(nd.ID, Phx=0)  # 影响多根滚柱的载荷分配

        # 滚柱接触点边界条件
        # for k in range(tt):
        #     for nd in namespace['nds_gz_c_%d_n' % k]:
        #         s.add_node_disp(nd.ID, Phx=0, Phy=0, Phz=0)
        #
        #     for nd in namespace['nds_gz_c_%d_s' % k]:
        #         s.add_node_disp(nd.ID, Phx=0, Phy=0, Phz=0)
        # 弹簧（接触点）边界条件

        s.solve()

        # acc = []
        # for j in range(tt):
        #     namespace['N%d_s' % j] = []
        #     namespace['N%d_n' % j] = []
        N_s = []
        N_n = []

        for j in range(tt):
            for i in range(n):
                N_s.append(
                    np.sqrt(np.array([el.force["N"] for el in namespace['els_sz_%d_x' % j]])[i][0][0] ** 2 +
                            np.array([el.force["N"] for el in namespace['els_sz_%d_y' % j]])[i][0][0] ** 2 +
                            np.array([el.force["N"] for el in namespace['els_sz_%d_z' % j]])[i][0][0] ** 2))

                N_n.append(
                    np.sqrt(np.array([el.force["N"] for el in namespace['els_lz_%d_x' % j]])[i][0][0] ** 2 +
                            np.array([el.force["N"] for el in namespace['els_lz_%d_y' % j]])[i][0][0] ** 2 +
                            np.array([el.force["N"] for el in namespace['els_lz_%d_z' % j]])[i][0][0] ** 2))

        if epoch == 0 and innerCycle == 0:
            pass
        else:
            acc = np.abs(np.array(N_s) - N_0s).sum() + np.abs(np.array(N_n) - N_0n).sum()
        print(acc)
        N_0s = N_s
        N_0n = N_n

# N0x = np.array([el.force["N"] for el in els_sz_0_x])
# N1x = np.array([el.force["N"] for el in els_sz_1_x], dtype=object)
# N2x = np.array([el.force["N"] for el in els_sz_2_x])
# N3x = np.array([el.force["N"] for el in els_sz_3_x])
# N4x = np.array([el.force["N"] for el in els_sz_4_x])
# N5x = np.array([el.force["N"] for el in els_sz_5_x])
#
# print(N0x, "N0x")
# print(N1x, "N1x")
# print(N2x, "N2x")
# print(N3x, "N3x")
# print(N4x, "N4x")
# print(N5x, "N5x")
#
# N0y = np.array([el.force["N"] for el in els_sz_0_y])
# N1y = np.array([el.force["N"] for el in els_sz_1_y], dtype=object)
# N2y = np.array([el.force["N"] for el in els_sz_2_y])
# N3y = np.array([el.force["N"] for el in els_sz_3_y])
# N4y = np.array([el.force["N"] for el in els_sz_4_y])
# N5y = np.array([el.force["N"] for el in els_sz_5_y])
#
# print(N0y, "N0_y")
# print(N1y, "N1_y")
# print(N2y, "N2_y")
# print(N3y, "N3_y")
# print(N4y, "N4_y")
# print(N5y, "N5_y")
#
# N0z = np.array([el.force["N"] for el in els_sz_0_z])
# N1z = np.array([el.force["N"] for el in els_sz_1_z], dtype=object)
# N2z = np.array([el.force["N"] for el in els_sz_2_z])
# N3z = np.array([el.force["N"] for el in els_sz_3_z])
# N4z = np.array([el.force["N"] for el in els_sz_4_z])
# N5z = np.array([el.force["N"] for el in els_sz_5_z])
#
# print(N0z, "N0_z")
# print(N1z, "N1_z")
# print(N2z, "N2_z")
# print(N3z, "N3_z")
# print(N4z, "N4_z")
# print(N5z, "N5_z")

for j in range(tt):
    namespace['N%d_s' % j] = []
    namespace['N%d_n' % j] = []

for j in range(tt):
    for i in range(n):
        namespace['N%d_s' % j].append(
            np.sqrt(np.array([el.force["N"] for el in namespace['els_sz_%d_x' % j]])[i][0][0] ** 2 +
                    np.array([el.force["N"] for el in namespace['els_sz_%d_y' % j]])[i][0][0] ** 2 +
                    np.array([el.force["N"] for el in namespace['els_sz_%d_z' % j]])[i][0][0] ** 2))

    # print(namespace['N%d_s' % j], "N_s" + str(j))
# np.savetxt('N_s.csv', np.r_[N0_s, N1_s, N2_s, N3_s, N4_s, N5_s].flatten(), fmt='%s', delimiter=',')
np.savetxt('N_s_0.csv', np.column_stack((N0_s, N1_s, N2_s, N3_s, N4_s, N5_s)), fmt='%s', delimiter=',')

for j in range(tt):
    for i in range(n):
        namespace['N%d_n' % j].append(
            np.sqrt(np.array([el.force["N"] for el in namespace['els_lz_%d_x' % j]])[i][0][0] ** 2 +
                    np.array([el.force["N"] for el in namespace['els_lz_%d_y' % j]])[i][0][0] ** 2 +
                    np.array([el.force["N"] for el in namespace['els_lz_%d_z' % j]])[i][0][0] ** 2))

    # print(namespace['N%d_n' % j], "N_n" + str(j))

np.savetxt('N_n_0.csv', np.column_stack((N0_n, N1_n, N2_n, N3_n, N4_n, N5_n)), fmt='%s', delimiter=',')

# for i in range(n):
#     N1 = np.sqrt(N1x[i][0][0] ** 2 + N1y[i][0][0] ** 2 + N1z[i][0][0] ** 2)
#     print(N1, "N1")
#
# for i in range(n):
#     N2 = np.sqrt(N2x[i][0][0] ** 2 + N2y[i][0][0] ** 2 + N2z[i][0][0] ** 2)
#     print(N2, "N2")
#
# for i in range(n):
#     N3 = np.sqrt(N3x[i][0][0] ** 2 + N3y[i][0][0] ** 2 + N3z[i][0][0] ** 2)
#     print(N3, "N3")
#
# for i in range(n):
#     N4 = np.sqrt(N4x[i][0][0] ** 2 + N4y[i][0][0] ** 2 + N4z[i][0][0] ** 2)
#     print(N4, "N4")
#
# for i in range(n):
#     N5 = np.sqrt(N5x[i][0][0] ** 2 + N5y[i][0][0] ** 2 + N5z[i][0][0] ** 2)
#     print(N5, "N5")

# print(N1, "N1")
# print(N2, "N2")
# print(N3, "N3")
# print(N4, "N4")
# print(N5, "N5")
for j in range(tt):
    namespace['disp_gz%d_y' % j] = []
    namespace['disp_gz%d_z' % j] = []
    namespace['disp_gz%d_Phx' % j] = []

for j in range(tt):
    namespace['disp_gz%d_y' % j] = [nd.disp["Uy"] for nd in namespace['nds_gz_%d' % j]]
    namespace['disp_gz%d_z' % j] = [nd.disp["Uz"] for nd in namespace['nds_gz_%d' % j]]
    # namespace['disp_gz%d_Phx' % j] = [nd.disp["Phx"] for nd in namespace['nds_gz_%d' % j]]

disp_sg_y = [nd.disp["Uy"] for nd in nds_sg]
disp_sg_z = [nd.disp["Uz"] for nd in nds_sg]
# disp_sg_Phx = [nd.disp["Phx"] for nd in nds_sg]

np.savetxt('disp_gz_0.csv', np.column_stack((disp_gz0_y, disp_gz0_z, disp_gz1_y, disp_gz1_z,
                                             disp_gz2_y, disp_gz2_z, disp_gz3_y, disp_gz3_z,
                                             disp_gz4_y, disp_gz4_z, disp_gz5_y, disp_gz5_z,)),
           fmt='%s', delimiter=',')
np.savetxt('disp_sg_0.csv', np.column_stack((disp_sg_y, disp_sg_z)), fmt='%s', delimiter=',')

# print("disp_gz_c_0_s", disp0)
# print("disp_sg_c", disp1)
# print("disp_sg", disp2)
# print("disp_sg_c", disp3)

# print(namespace['K_n_%d' % j])
# print(namespace['K_s_%d' % j])
