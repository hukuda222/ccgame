import random
import pickle
import numpy as np
from get_hand import get_hand

#1が貯め、2が守り、3が攻撃、4が大攻撃
max_count = 20

def judge(self):
    #1はチャージ、2はガード、4は攻撃、5は大攻撃、3は祈り
    for i in range(2):
        if self.hands[(i+1)%2] == 1 and self.hands[i] == 4:
            return i
        if self.hands[(i+1)%2] == 3 and self.hands[i] == 4:
            return i
        elif self.hands[i] == 1 and self.hands[(i+1)%2] == 5:
            return i
        elif self.hands[(i+1)%2] == 2 and self.hands[i] == 5:
            return i
        elif self.hands[(i+1)%2] == 3 and self.hands[i] == 5:
            return i
        elif self.hands[(i+1)%2] == 4 and self.hands[i] == 5:
            return i
        elif self.hands[i] == 1 and self.hands[(i+1)%2] == 5:
            return i
        elif self.prays[i] == 2 and self.hands[i] == 3:
            return i
    else:
        return -1

def process(self):
    for i in range(2):
        self.history1[self.turn][i]=self.hands[i]
        self.history2[self.turn][i]=self.hands[(i+1)%2]
        if self.hands[i] == 1:
            self.prays[i]=0
            self.charge[i]+=1
        elif self.hands[i] == 3:
            self.prays[i]=0
            self.charge[i]-=1
        elif self.hands[i] == 4:
            self.prays[i]=0
            self.charge[i]-=2
        elif self.hands[i] == 5:
            self.prays[i]+=1

def get_hands(charge,history1,turn):
    hands=[0 for i in range(2)]
    if charge[0]>=2:
        hands[0]=get_hand(history1,[1,2,3,4,5])
    elif charge[0]>=1:
        hands[0]=get_hand(history1,[1,2,3,4])
    else:
        hands[0]=get_hand(history1,[1,2,3])
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
    '''
    if charge[1]>=2:
        hands[1]=random.randint(1,4)
    elif charge[1]>=1:
        hands[1]=random.randint(1,3)
    else:
        hands[1]=random.randint(1,2)
    '''
    return hands

def get_game():
    history1 = [[0 for j in range(2)]for i in range(20)]
    history2 = [[0 for j in range(2)]for i in range(20)]
    charge = [0 for i in range(2)]
    turn = 0;
    win = -1
    #aが0でbが1、何もなければ-1
    while turn<20 :
        win = judge(get_hands(charge,history1,turn),history1,history2,charge,turn)
        print(history1)
        if win is not -1:
            break;
        turn+=1
    turn+=1
    return win

if __name__ == '__main__':
    get_game()
    '''
    win_count = 0
    for i in range(100000):
        if i%10000==0:
            print(i)
        if get_game()==0:
            win_count+=1
    print(win_count)
    '''
