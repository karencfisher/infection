# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:03:27 2020

@author: yeshesdawa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# initial state and values
# population
S = 6000 

# recovery_rate
recovery_days = 14.0
gamma = 1 / recovery_days

# initial est. infection (initially also total)
initial_infections = 10  
total_infections = initial_infections 

doubling_time = 6 
n_days = 200 # number of days to project

detection_prob = initial_infections / total_infections
S, I, R = S, initial_infections / detection_prob, 0
intrinsic_growth_rate = 2 ** (1 / doubling_time) - 1


# The SIR model, one time step (from Penn model)
def sir(y, beta, gamma, N):
    S, I, R = y
    Sn = (-beta * S * I) + S
    In = (beta * S * I - gamma * I) + I
    Rn = gamma * I + R
    if Sn < 0:
        Sn = 0
    if In < 0:
        In = 0
    if Rn < 0:
        Rn = 0

    scale = N / (Sn + In + Rn)
    return Sn * scale, In * scale, Rn * scale


# Run the SIR model forward in time (from Penn model)
def sim_sir(S, I, R, beta, gamma, n_days, beta_decay=None):
    N = S + I + R
    s, i, r = [S], [I], [R]
    for day in range(n_days):
        y = S, I, R
        S, I, R = sir(y, beta, gamma, N)
        if beta_decay:
            beta = beta * (1 - beta_decay)
        s.append(S)
        i.append(I)
        r.append(R)

    s, i, r = np.array(s), np.array(i), np.array(r)
    return s, i, r


def project_infect(social_distance):
    relative_contact_rate = social_distance / 100
    beta = (
            intrinsic_growth_rate + gamma
                ) / S * (1-relative_contact_rate) # {rate based on doubling time} / {initial S}

    r_t = beta / gamma * S # r_t is r_0 after distancing
    r_naught = r_t / (1-relative_contact_rate)

    beta_decay = 0.0
    s, i, r = sim_sir(S, I, R, beta, gamma, n_days, beta_decay=beta_decay)
    
    return s, i, r, r_naught
    
    
if __name__ == '__main__':
    
    social_distances = [0, 10, 20, 30, 40, 50]
    for social_distance in social_distances:
        
        s, i, r, r0 = project_infect(social_distance)
        days = np.array(range(0, n_days + 1))
        plt.plot(days, i, label=str(social_distance)+'%')
        
    plt.title('Projected infection by social distancing')
    plt.xlabel('Days')
    plt.ylabel('Total infections')
    plt.legend()
    plt.show()
    
        