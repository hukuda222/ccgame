import random
import pickle
import numpy as np
from get_hand import get_hand

#1が貯め、2が守り、3が攻撃、4が大攻撃
max_count = 20

def judge(hands,history1,history2,charge,turn):
    process(hands,history1,history2,charge,turn)
    for i in range(2):
        if hands[(i+1)%2] == 1 and hands[i] == 3:
            return i
        elif hands[i] == 1 and hands[(i+1)%2] ==4:
            return i
        elif hands[(i+1)%2] == 3 and hands[i] ==4:
            return i
    else:
        return -1

def process(hands,history1,history2,charge,turn):
    for i in range(2):
        history1[turn][i]=hands[i]
        history2[turn][i]=hands[(i+1)%2]
        if hands[i] == 1:
            charge[i]+=1
        elif hands[i] == 3:
            charge[i]-=1
        elif hands[i] == 4:
            charge[i]-=2

def get_hands(charge,history1):
    hands=[0 for i in range(2)]
    if charge[0]>=2:
        hands[0]=get_hand(history1,[1,2,3,4])
    elif charge[0]>=1:
        hands[0]=get_hand(history1,[1,2,3])
    else:
        hands[0]=get_hand(history1,[1,2])

    if charge[1]>=2:
        while hands[1] == 0:
            hands[1] = int(input())
            if hands[1]<1 or hands[1]>4:
                hands[1]=0
    elif charge[1]>=1:
        while hands[1] == 0:
            hands[1] = int(input())
            if hands[1]<1 or hands[1]>3:
                hands[1]=0
    else:
        while hands[1] == 0:
            hands[1] = int(input())
            if hands[1]<1 or hands[1]>2:
                hands[1]=0
    return hands

def get_game():
    history1 = [[0 for j in range(2)]for i in range(20)]
    history2 = [[0 for j in range(2)]for i in range(20)]
    charge = [0 for i in range(2)]
    turn = 0;
    win = -1
    #aが0でbが1、何もなければ-1
    while turn<20 :
        win = judge(get_hands(charge,history1),history1,history2,charge,turn)
        print(history1)
        if win is not -1:
            break;
        turn+=1
    turn+=1

if __name__ == '__main__':
    get_game()
