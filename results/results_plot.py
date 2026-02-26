import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data_active= pd.read_csv(r'D:\pycharmcode\project\CEMS\optimal_strategy_active.csv',encoding='utf-8')
time= data_active['time']
P_active_all= data_active['P-all']

data_normal=pd.read_csv(r'D:\pycharmcode\project\CEMS\optimal_strategy_normal.csv',encoding='utf-8')
P_normal_all= data_normal['P-all']

data_inactive=pd.read_csv(r'D:\pycharmcode\project\CEMS\optimal_strategy_inactive.csv',encoding='utf-8')
P_inactive_all= data_inactive['P-all']

t= [i for i in range(24)]
P_active=[0 for i in range(24)]
for i in range(24):
    idx= (i+12) % 24
    P_active[idx]=P_active_all[i]
P_normal=[0 for i in range(24)]
for i in range(24):
    idx= (i+12) % 24
    P_normal[idx]=P_normal_all[i]
P_inactive=[0 for i in range(24)]
for i in range(24):
    idx= (i+12) % 24
    P_inactive[idx]=P_inactive_all[i]
P_active = np.array(P_active)
P_normal = np.array(P_normal)
P_inactive = np.array(P_inactive)

# 正值部分

pos_active = np.clip(P_active, 0, None)
pos_normal = np.clip(P_normal, 0, None)
pos_inactive = np.clip(P_inactive, 0, None)

# 负值部分
neg_active = np.clip(P_active, None, 0)
neg_normal = np.clip(P_normal, None, 0)
neg_inactive = np.clip(P_inactive, None, 0)

plt.figure(figsize=(18,16))

reds = [
    'red',          # 纯红
    'darkred',      # 深红
    'firebrick',    # 火砖红
    'crimson',      # 深红紫
    'indianred',    # 印度红
    'lightcoral',   # 浅珊瑚红
    'salmon',       # 鲑鱼红
    'darksalmon',   # 深鲑鱼红
    'lightsalmon',  # 浅鲑鱼红
    'tomato',       # 番茄红
    'orangered',    # 橙红
    'maroon'        # 栗红
]
blues = [
    'blue',         # 纯蓝
    'navy',         # 海军蓝
    'darkblue',     # 深蓝
    'royalblue',    # 皇家蓝
    'mediumblue',   # 中蓝
    'dodgerblue',   # 道奇蓝
    'deepskyblue',  # 深天蓝
    'skyblue',      # 天蓝
    'steelblue'     # 钢蓝
]
greens = [
    'green',        # 纯绿
    'darkgreen',    # 深绿
    'limegreen',    # 亮绿
    'seagreen'      # 海绿
]

label_active=[]
for i in range(12):
    idx=str(i+1)
    label_active.append('积极用户'+idx)

label_normal=[]
for i in range(9):
    idx = str(i+1)
    label_normal.append('普通用户'+idx)

label_inactive=[]
for i in range(4):
    idx = str (i+1)
    label_inactive.append('消极用户'+idx)

pos_active_all=[pos_active]*12
pos_normal_all=[pos_normal]*9
pos_inactive_all=[pos_inactive]*4

neg_active_all=[neg_active]*12
neg_normal_all=[neg_normal]*9
neg_inactive_all=[neg_inactive]*4
# 正向堆叠
plt.bar(t,pos_active_all[0],color=reds[0],label=label_active[0])
for i in range(1,12):
    plt.bar(t, pos_active_all[i],bottom=np.sum(pos_active_all[:i],axis=0),label=label_active[i],color=reds[i])

plt.bar(t,pos_normal_all[0],color=blues[0],label=label_normal[0],bottom=np.sum(pos_active_all,axis=0))
for i in range(1,9):
    plt.bar(t,pos_normal_all[i],bottom=np.sum(pos_active_all,axis=0)+np.sum(pos_normal_all[:i],axis=0),label=label_normal[i],color=blues[i])
# 负向堆叠
plt.bar(t,pos_inactive_all[0],color=greens[0],label=label_inactive[0],bottom=np.sum(pos_active_all,axis=0)+np.sum(pos_normal_all,axis=0))
for i in range(1,4):
    plt.bar(t,pos_inactive_all[i],color=greens[i],label=label_inactive[i],bottom=np.sum(pos_active_all,axis=0)+np.sum(pos_normal_all,axis=0)+np.sum(pos_inactive_all[:i],axis=0))


for i in range(12):
    plt.bar(t,neg_active_all[i],color=reds[i],bottom=np.sum(neg_active_all[:i],axis=0))

for i in range(9):
    plt.bar(t,neg_normal_all[i],color=blues[i],bottom=np.sum(neg_active_all,axis=0)+np.sum(neg_normal_all[:i],axis=0))

for i in range(4):
    plt.bar(t,neg_inactive_all[i],color=greens[i],bottom=np.sum(neg_active_all,axis=0)+np.sum(neg_normal_all,axis=0)+np.sum(neg_inactive_all,axis=0))

plt.xlabel('时间')
plt.ylabel('总功率')
plt.legend()
plt.show()

data_community=pd.read_csv(r'D:\pycharmcode\project\CEMS\optimal_strategy_community.csv',encoding='utf-8')
price_now=data_community['price']
data = pd.read_csv(
    r'D:/pycharmcode/project/CEMS/database/price_data.csv',
    encoding='gbk'
)

prices=data['实时电价']
price_now_data=[0 for i in range(24)]
price_old_data=[0 for i in range(24)]
for i in range(24):
    idx=(i+12)%24
    price_now_data[idx]=price_now[i]
    price_old_data[i]=prices[i]
plt.figure(figsize=(18,16))
plt.xlabel('时间')
plt.ylabel('电价 （度/元）')
plt.plot(t,price_now_data,color='blue',label='修改后的电价',lw=1.5, marker='o')
plt.plot(t,price_old_data,color='green',label='修改前的电价',lw=1.5, marker='s')
plt.legend()
plt.show()

data_CESS=data_community['CESS_remain']

plt.figure(figsize=(18,16))
remain_CESS=[0 for i in range(24)]

for i in range(24):
    idx=(i+12)%24
    remain_CESS[idx]=data_CESS[i]

plt.bar(t,remain_CESS,color='blue')
plt.xlabel('时间')
plt.ylabel('CESS储存量')
plt.show()