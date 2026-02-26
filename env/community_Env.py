from database.data import T_t,price_t
import numpy as np

class community_env():
    def __init__(self,t,T_t,price_t,m,n,p,P_CESS,CESS_SOC_min,CESS_SOC_max,CESS_SOC_primary
                    ,out_effiency,w_old,punish,active_user,normal_user,inactive_user):
        self.t=t
        self.T_t = T_t
        self.price_t = price_t
        self.m = m              #积极参与的用户个数
        self.n = n              #普通参与的用户个数
        self.p = p              #不积极参与的用户个数
        self.P_CESS = P_CESS
        self.CESS_SOC_min = CESS_SOC_min
        self.CESS_SOC_max = CESS_SOC_max
        self.CESS_SOC_primary = CESS_SOC_primary
        self.CESS_remain=CESS_SOC_primary
        self.out_effiency = out_effiency
        self.w_old = w_old
        self.punish=punish
        self.active_user=active_user
        self.normal_user=normal_user
        self.inactive_user=inactive_user


    def CEES(self,t,action):
        hour=t % 24
        P_out= action* self.P_CESS
        add = P_out * self.out_effiency if P_out >0 else P_out/self.out_effiency
        self.CESS_remain += add
        CESS_damage= -self.w_old * abs (P_out)
        if self. CESS_remain < self. CESS_SOC_min :
            extra_punish=-(self.punish+6)*(self.CESS_SOC_min-self.CESS_remain)
        elif self. CESS_remain > self. CESS_SOC_max :
            extra_punish=-(self.punish+6)*(self.CESS_remain-self.CESS_SOC_max)
        else:
            extra_punish=0
        if hour == 11 and self.CESS_remain != self.CESS_SOC_primary:
            punishment=-(self.punish+6) *abs(self.CESS_remain-self.CESS_SOC_primary)
        else:
            punishment=0
        rewards=CESS_damage+extra_punish+punishment
        return rewards , P_out

    def cems_reset(self):

        self.t = 12
        self.CESS_remain = self.CESS_SOC_primary


    def get_cems_state(self, P_out):
        hour = self.t % 24
        price = price_t(hour)  # 或 self.price_t(hour)

        state = np.array([
            P_out,  # 社区负荷
            hour,  # 时间
            self.CESS_remain,  # CESS SOC
            price  # 原始电价
        ], dtype=np.float32)

        return state

    def step(self, cems_action,CM_P_all):
        price_action=cems_action[0]
        CESS_action=cems_action[1]

        hour = self.t % 24
        base_price = price_t(hour)
        price_now = base_price * price_action  # 服务商发布的新价格
        reward_CESS, P_CESS_out = self.CEES(self.t, CESS_action)

        P_total = CM_P_all[self.t-12] + P_CESS_out  # 社区总功率
        reward_CESS=reward_CESS

        grid_cost = P_total * price_now
        reward_cems =  -grid_cost+reward_CESS

        self.t += 1
        done = True if self.t >= 36 else False  # 12~36 一个周期

        if done == False:
            next_state_cems = self.get_cems_state(CM_P_all[self.t -12])
        else:
            next_state_cems = self.get_cems_state(CM_P_all[0])
        reward_cems=reward_cems
        return next_state_cems, reward_cems/300, done



