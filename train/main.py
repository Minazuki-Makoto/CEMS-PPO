from env.Active_Env import active_env
from env.Normal_Env import normal_env
from env.Inactive_Env import inactive_env
from env.community_Env import community_env
from database.data import T_t,price_t
from agent.PPO import PPO_agent
from agent.PPO_HEMS import PPOHEMS_agent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def smooth_curve(data, window=20):
    data = np.array(data)
    smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
    return smoothed

if __name__ == '__main__':
    SEED = 64
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)


    t = 12
    T_primary = T_t(12) - 4
    trans_MW_t_start = 15
    trans_MW_t_end = 22
    trans_alltime = 2
    trans_WM_P_set = 0.4
    trans_load_ws = 0.5
    trans_DIS_t_start = 19
    trans_DIS_t_end = 22
    trans_DIS_P_set = 0.5
    con_load_light_ws = 2.0
    con_load_light_min = 0.3
    con_load_light_set = 0.5
    con_load_Humid_ws = 1.1
    con_load_Humid_min = 0.2
    con_load_Humid_set = 0.3
    HVAC_p_set = 2.0
    T_best = 24
    HVAC_ws = 0.05
    alpha = 0.05
    beta = 0.85
    error = 1.5
    loss = 0.03
    P_set = 7.5
    energy_eta = 0.95
    t_get = 8
    t_leave = 22
    SOC = 60
    SOC_primary = 40
    anxiety = 0.05
    damage = 0.01
    punish = 0.4
    ESS_P_set = 10
    SOC_max = 30
    SOC_min = 15
    SOC_initial = 21
    energy_convert = 0.95
    PV_P_set = 4
    state_dim = 8
    hidden_dim = 256
    action_dim = 8
    eps = 0.1
    gamma = 0.95
    m=12
    n=9
    p=4
    P_CESS=50
    CESS_SOC=150
    CESS_SOC_max=135
    CESS_SOC_min= 15
    CESS_SOC_primary=75
    w_old=0.01
    out_effiency=0.95
    active_environment=active_env(t,T_t,T_primary,
                                trans_MW_t_start,trans_MW_t_end,trans_alltime,trans_WM_P_set,trans_load_ws,trans_DIS_t_start,trans_DIS_t_end,trans_DIS_P_set,
                                con_load_light_ws,con_load_light_min,con_load_light_set,con_load_Humid_ws,con_load_Humid_min,con_load_Humid_set,
                                HVAC_p_set,T_best,HVAC_ws,alpha,beta,error,loss,
                                P_set,energy_eta,t_get,t_leave,SOC,SOC_primary,anxiety,damage,punish,
                                ESS_P_set,SOC_max,SOC_min,SOC_initial,energy_convert,PV_P_set)

    normal_environment=normal_env(t,T_t,T_primary,
                                trans_MW_t_start,trans_MW_t_end,trans_alltime,trans_WM_P_set,trans_load_ws,trans_DIS_t_start,trans_DIS_t_end,trans_DIS_P_set,
                                con_load_light_ws,con_load_light_min,con_load_light_set,con_load_Humid_ws,con_load_Humid_min,con_load_Humid_set,
                                HVAC_p_set,T_best,HVAC_ws,alpha,beta,error,loss,
                                P_set,energy_eta,t_get,t_leave,SOC,SOC_primary,anxiety,damage,punish,
                                ESS_P_set,SOC_max,SOC_min,SOC_initial,energy_convert)

    inactive_environment=inactive_env(t,T_t,T_primary,
                                      trans_MW_t_start,trans_MW_t_end,trans_alltime,trans_WM_P_set,trans_load_ws,trans_DIS_t_start,trans_DIS_t_end,trans_DIS_P_set,
                                      con_load_light_ws,con_load_light_min,con_load_light_set,con_load_Humid_ws,con_load_Humid_min,con_load_Humid_set,
                                      HVAC_p_set,T_best,HVAC_ws,alpha,beta,error,loss)

    community_environment=community_env(t,T_t,price_t,m,n,p,P_CESS,CESS_SOC_min,CESS_SOC_max,CESS_SOC_primary
                    ,out_effiency,w_old,punish,active_environment,normal_environment,inactive_environment)
    active_state_dim=8
    normal_state_dim=8
    inactive_state_dim=5
    community_state_dim=4
    active_action_dim=8
    normal_action_dim=7
    inactive_action_dim=5
    community_action_dim=2
    history_rewards=[]
    history_home_rewards=[]
    community_action_history_all=[]

    ppo_agent=PPO_agent(community_state_dim,hidden_dim,community_action_dim,gamma)
    ppo_active_agent=PPOHEMS_agent(active_state_dim,hidden_dim,active_action_dim,gamma)
    ppo_normal_agent=PPOHEMS_agent(normal_state_dim,hidden_dim,normal_action_dim,gamma)
    ppo_inactive_agent=PPOHEMS_agent(inactive_state_dim,hidden_dim,inactive_action_dim,gamma)

    '''得到初始化负荷曲线'''
    init_active_rewards_history=[]
    init_normal_rewards_history=[]
    init_inactive_rewards_history=[]

    init_active_P_all=[]
    init_inactive_P_all=[]
    init_normal_P_all=[]
    for alt in range(11000):
        active_environment.reset()
        normal_environment.reset()
        inactive_environment.reset()
        active_state=active_environment.get_state(price_t(active_environment.t))
        inactive_state=inactive_environment.get_state(price_t(inactive_environment.t))
        normal_state=normal_environment.get_state(price_t(normal_environment.t))

        active_P_all_alt=[]
        normal_P_all_alt=[]
        inactive_P_all_alt=[]

        active_rewards=0
        inactive_rewards=0
        normal_rewards=0
        active_state_history,normal_state_history,inactive_state_history=[],[],[]
        active_action_history,normal_action_history,inactive_action_history=[],[],[]
        active_reward_history,normal_reward_history,inactive_reward_history=[],[],[]
        active_dones_history,normal_dones_history,inactive_dones_history=[],[],[]
        active_next_state_history,normal_next_state_history,inactive_next_state_history=[],[],[]
        active_logp_history,normal_logp_history,inactive_logp_history=[],[],[]

        while 1 :

            '''积极用户'''
            active_state_history.append(active_state)
            active_action,log_prob=ppo_active_agent.choose_hems_action(active_state)
            active_next_state,active_reward,done,active_P_all=active_environment.step(active_action,price_t(active_environment.t))
            active_rewards+=active_reward
            active_action_history.append(active_action)
            active_reward_history.append(active_reward)
            active_next_state_history.append(active_next_state)
            active_logp_history.append(log_prob)
            active_dones_history.append(done)

            '''普通用户'''
            normal_state_history.append(normal_state)
            normal_action,normal_log_prob=ppo_normal_agent.choose_hems_action(normal_state)
            normal_next_state,normal_reward,normal_done,normal_P_all=normal_environment.step(normal_action,price_t(normal_environment.t))
            normal_rewards+=normal_reward
            normal_action_history.append(normal_action)
            normal_reward_history.append(normal_reward)
            normal_next_state_history.append(normal_next_state)
            normal_logp_history.append(normal_log_prob)
            normal_dones_history.append(normal_done)

            '''消极用户'''
            inactive_state_history.append(inactive_state)
            inactive_action,inactive_log_prob=ppo_inactive_agent.choose_hems_action(inactive_state)
            inactive_next_state,inactive_reward,inactive_done,inactive_P_all=inactive_environment.step(inactive_action,price_t(inactive_environment.t))
            inactive_rewards+=inactive_reward
            inactive_action_history.append(inactive_action)
            inactive_reward_history.append(inactive_reward)
            inactive_next_state_history.append(inactive_next_state)
            inactive_logp_history.append(inactive_log_prob)
            inactive_dones_history.append(inactive_done)
            inactive_P_all_alt.append(inactive_P_all)
            active_P_all_alt.append(active_P_all)
            normal_P_all_alt.append(normal_P_all)
            if inactive_done:
                break
            inactive_state = inactive_next_state
            active_state=active_next_state
            normal_state=normal_next_state
        ppo_active_agent.hems_update(0.1,active_state_history,active_action_history,active_reward_history,active_next_state_history,active_dones_history,active_logp_history)
        ppo_normal_agent.hems_update(0.1,normal_state_history,normal_action_history,normal_reward_history,normal_next_state_history,normal_dones_history,normal_logp_history)
        ppo_inactive_agent.hems_update(0.1,inactive_state_history,inactive_action_history,inactive_reward_history,inactive_next_state_history,inactive_dones_history,inactive_logp_history)

        init_active_rewards_history.append(active_rewards)
        init_normal_rewards_history.append(normal_rewards)
        init_inactive_rewards_history.append(inactive_rewards)

        init_active_P_all.append(active_P_all_alt)
        init_normal_P_all.append(normal_P_all_alt)
        init_inactive_P_all.append(inactive_P_all_alt)

    best_active_idx=np.argmax(init_active_rewards_history)
    best_normal_idx=np.argmax(init_normal_rewards_history)
    best_inactive_idx=np.argmax(init_inactive_rewards_history)
    init_P_active=init_active_P_all[best_active_idx]
    init_P_normal=init_normal_P_all[best_normal_idx]
    init_P_inactive=init_inactive_P_all[best_inactive_idx]
    print(f'最优积极用户回报为{init_active_rewards_history[best_active_idx]*50},最优普通用户回报为{init_normal_rewards_history[best_normal_idx]*50}，最优消极用户回报为{init_inactive_rewards_history[best_inactive_idx]*50}')
    init_P_all=[ m*init_P_active[i]+n*init_P_normal[i]+p*init_P_inactive[i] for i in range(len(init_P_active))]
    print(f'初始化完成，大迭代开始')
    '''迭代开始'''
    CM_P_all=init_P_all
    cems_reward_history=[]
    hems_reward_history=[]
    community_action_all=[]
    active_action_all=[]
    normal_action_all=[]
    inactive_action_all=[]

    for alt in range(12):
        '''主'''
        cems_reward_eposide=[]
        action_eposide_history=[]
        for eposide in range(20000):
            community_environment.cems_reset()
            community_state=community_environment.get_cems_state(CM_P_all[community_environment.t-12])
            community_rewards=0
            community_state_history,community_next_state_history,community_reward_history,community_action_history,community_dones_history,community_logp_history=[],[],[],[],[],[]
            while 1 :
                community_state_history.append(community_state)
                community_action,community_logp=ppo_agent.choose_cems_action(community_state)
                community_next_state,community_reward,community_done=community_environment.step(community_action,CM_P_all)
                community_rewards+=community_reward
                community_action_history.append(community_action)
                community_next_state_history.append(community_next_state)
                community_reward_history.append(community_reward)
                community_dones_history.append(community_done)
                community_logp_history.append(community_logp)
                if community_done :
                    break
                community_state=community_next_state
            cems_reward_eposide.append(community_rewards)
            action_eposide_history.append(community_action_history)
            ppo_agent.update(0.11,community_state_history,community_action_history,community_reward_history,community_dones_history,community_next_state_history,community_logp_history)

        best_idx_community=np.argmax(cems_reward_eposide)
        best_action=action_eposide_history[best_idx_community]
        community_action_all.append(best_action)
        print(f'第{alt}次迭代中，最好的社区回报为{cems_reward_eposide[best_idx_community]*300}')
        price_h=[price_t(i+12)* best_action[i][0] for i in range(len(best_action))]
        cems_reward_history.append(cems_reward_eposide[best_idx_community])

        alt_active_reward=[]
        alt_normal_reward=[]
        alt_inactive_reward=[]

        alt_active_P_all=[]
        alt_normal_P_all=[]
        alt_inactive_P_all=[]
        '''从'''
        for eps in range(5000):
            active_environment.reset()
            normal_environment.reset()
            inactive_environment.reset()
            active_state = active_environment.get_state(price_h[active_environment.t])
            inactive_state = inactive_environment.get_state(price_h[inactive_environment.t])
            normal_state = normal_environment.get_state(price_h[normal_environment.t])

            active_P_all_alt = []
            normal_P_all_alt = []
            inactive_P_all_alt = []

            active_rewards = 0
            inactive_rewards = 0
            normal_rewards = 0
            active_state_history, normal_state_history, inactive_state_history = [], [], []
            active_action_history, normal_action_history, inactive_action_history = [], [], []
            active_reward_history, normal_reward_history, inactive_reward_history = [], [], []
            active_dones_history, normal_dones_history, inactive_dones_history = [], [], []
            active_next_state_history, normal_next_state_history, inactive_next_state_history = [], [], []
            active_logp_history, normal_logp_history, inactive_logp_history = [], [], []

            while 1:
                '''积极用户'''
                active_state_history.append(active_state)
                active_action, log_prob = ppo_active_agent.choose_hems_action(active_state)
                if  active_environment.t<36:
                    active_next_state, active_reward, done, active_P_all = active_environment.step(active_action, price_h[active_environment.t-12])
                else:
                    active_next_state, active_reward, done, active_P_all = active_environment.step(active_action, price_h[0])
                active_rewards += active_reward
                active_action_history.append(active_action)
                active_reward_history.append(active_reward)
                active_next_state_history.append(active_next_state)
                active_logp_history.append(log_prob)
                active_dones_history.append(done)

                '''普通用户'''
                normal_state_history.append(normal_state)
                normal_action, normal_log_prob = ppo_normal_agent.choose_hems_action(normal_state)
                if normal_environment.t<36:
                    normal_next_state, normal_reward, normal_done, normal_P_all = normal_environment.step(normal_action,price_h[normal_environment.t-12])
                else:
                    normal_next_state, normal_reward, normal_done, normal_P_all = normal_environment.step(normal_action,
                                                                                                          price_h[0])
                normal_rewards += normal_reward
                normal_action_history.append(normal_action)
                normal_reward_history.append(normal_reward)
                normal_next_state_history.append(normal_next_state)
                normal_logp_history.append(normal_log_prob)
                normal_dones_history.append(normal_done)

                '''消极用户'''
                inactive_state_history.append(inactive_state)
                inactive_action, inactive_log_prob = ppo_inactive_agent.choose_hems_action(inactive_state)
                if inactive_environment.t<36:
                    inactive_next_state, inactive_reward, inactive_done, inactive_P_all = inactive_environment.step(inactive_action, price_h[inactive_environment.t-12])
                else:
                    inactive_next_state, inactive_reward, inactive_done, inactive_P_all = inactive_environment.step(
                        inactive_action, price_h[0])
                inactive_rewards += inactive_reward
                inactive_action_history.append(inactive_action)
                inactive_reward_history.append(inactive_reward)
                inactive_next_state_history.append(inactive_next_state)
                inactive_logp_history.append(inactive_log_prob)
                inactive_dones_history.append(inactive_done)
                inactive_P_all_alt.append(inactive_P_all)
                active_P_all_alt.append(active_P_all)
                normal_P_all_alt.append(normal_P_all)
                if inactive_done:
                    break
                inactive_state = inactive_next_state
                active_state = active_next_state
                normal_state = normal_next_state

            if eps % 2000==0:
                print(f'第{alt}次外迭代，第{eps}次内迭代积极用户最优回报为{active_rewards*50},普通用户回报为{normal_rewards*50},消极用户回报为{inactive_rewards*50}')
            ppo_active_agent.hems_update(0.1, active_state_history, active_action_history, active_reward_history,
                                         active_next_state_history, active_dones_history, active_logp_history)
            ppo_normal_agent.hems_update(0.1, normal_state_history, normal_action_history, normal_reward_history,
                                         normal_next_state_history, normal_dones_history, normal_logp_history)
            ppo_inactive_agent.hems_update(0.1, inactive_state_history, inactive_action_history,
                                           inactive_reward_history, inactive_next_state_history, inactive_dones_history,
                                           inactive_logp_history)
            alt_active_reward.append(active_rewards)
            alt_normal_reward.append(normal_rewards)
            alt_inactive_reward.append(inactive_rewards)

            alt_active_P_all.append(active_P_all_alt)
            alt_inactive_P_all.append(inactive_P_all_alt)
            alt_normal_P_all.append(normal_P_all_alt)

        best_idx_active=np.argmax(alt_active_reward)
        best_idx_normal=np.argmax(alt_normal_reward)
        best_idx_inactive=np.argmax(alt_inactive_reward)

        P_active=alt_active_P_all[best_idx_active]
        P_normal=alt_normal_P_all[best_idx_normal]
        P_inactive=alt_inactive_P_all[best_idx_inactive]

        active_action_all.append(P_active)
        normal_action_all.append(P_normal)
        inactive_action_all.append(P_inactive)
        hems_reward=m*alt_active_reward[best_idx_active]+n*alt_normal_reward[best_idx_normal]+p*alt_inactive_reward[best_idx_inactive]
        print(f'第{alt}次外迭代，最优hems回报为{hems_reward*50}')
        CM_P_all=[ P_active[i]*m+P_normal[i]*n+P_inactive[i]*p for i in range(len(P_active))]
        hems_reward_history.append(hems_reward)
        if len(hems_reward_history)>1:
            if abs(hems_reward-hems_reward_history[-2]) < 0.5:
                print('已收敛')
                break
    hems_reward_history=[hems_reward_history[i]*50 for i in range(len(hems_reward_history))]
    cems_reward_history=[cems_reward_history[i]*300 for i in range(len(cems_reward_history))]
    idx=np.argmax(hems_reward_history)
    idx_cems=np.argmax(cems_reward_history)
    active=active_action_all[idx]
    normal=normal_action_all[idx]
    inactive=inactive_action_all[idx]
    cems_action=community_action_all[idx]

    '''保存active用户的数据'''
    time=[ (t+12) % 24 for t in range(24)]
    active_data={
        'time':time,
        'P-all':active
    }
    df = pd.DataFrame(active_data)
    df.to_csv("D:/pycharmcode/project/CEMS/optimal_strategy_active.csv", index=False, encoding="utf-8-sig")

    '''保存normal用户的数据'''
    normal_data={
        'time':time,
        'P-all':normal
    }
    df = pd.DataFrame(normal_data)
    df.to_csv(r'D:\pycharmcode\project\CEMS\optimal_strategy_normal.csv',index=False,encoding='utf-8-sig')

    '''保存inactive用户数据'''
    inactive_data={
        'time':time,
        'P-all':inactive
    }
    df= pd.DataFrame(inactive_data)
    df.to_csv("D:/pycharmcode/project/CEMS/optimal_strategy_inactive.csv", index=False, encoding="utf-8-sig")

    '''保存运营商数据'''
    price = []
    P_CESS = []
    time = []
    CESS_remain = []
    community_environment.cems_reset()
    for i in range(len(cems_action)):
        t = 12 + i
        time.append(t % 24)
        price_now = price_t(t) * best_action[i][0]
        _, P_cess = community_environment.CEES(t, best_action[i][1])
        price.append(price_now)
        P_CESS.append(P_cess)
        CESS_remain.append(community_environment.CESS_remain)
    data = {
        'time': time,
        'price': price,
        'P_CESS': P_CESS,
        'CESS_remain': CESS_remain
    }
    df = pd.DataFrame(data)
    df.to_csv("D:/pycharmcode/project/CEMS/optimal_strategy_community.csv", index=False, encoding="utf-8-sig")

    alter=[i for i in range(len(hems_reward_history))]
    plt.figure(figsize=(18,16))
    plt.grid(True)
    plt.xlim(0,len(alter))
    plt.title('hems随迭代次数的回报')
    plt.plot(alter,hems_reward_history,lw=1.5,color='blue',marker='o')
    plt.show()
    print(f'最小hems回报为{max(hems_reward_history)}')
    alter = [i for i in range(len(cems_reward_history))]
    plt.figure(figsize=(18, 16))
    plt.grid(True)
    plt.xlim(0, len(alter))
    plt.title('cems随迭代次数的回报')
    plt.plot(alter, cems_reward_history, lw=1.5, color='red', marker='s')
    plt.show()