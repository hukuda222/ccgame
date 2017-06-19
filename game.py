import random
import pickle
import numpy as np
from get_hand import get_hand

class Game:
    def __init__(self):
        self.history1 = [[0 for j in range(2)]for i in range(50)]#プレイヤー1にとっての履歴
        self.history2 = [[0 for j in range(2)]for i in range(50)]#プレイヤー2にとっての履歴
        self.charge = [0 for i in range(2)]#ため時間
        self.turn = 0
        self.win = -1
        self.prays = [0 for i in range(2)]
        self.hands = [0 for i in range(2)]
        self.points = [0 for i in range(2)]

    def judge(self):
        #1はチャージ、2はガード、4は攻撃、5は大攻撃、3は祈り
        for i in range(2):
            if self.prays[i] == 3 and self.hands[i] == 3 and self.hands[(i+1)%2] <= 3:
                self.points[i]+=1
            elif self.hands[(i+1)%2] == 1 and self.hands[i] == 4:
                self.points[i]+=1
            if self.hands[(i+1)%2] == 3 and self.hands[i] == 4:
                self.points[i]+=1
            elif self.hands[i] == 1 and self.hands[(i+1)%2] == 5:
                self.points[i]+=1
            elif self.hands[(i+1)%2] == 2 and self.hands[i] == 5:
                self.points[i]+=1
            elif self.hands[(i+1)%2] == 3 and self.hands[i] == 5:
                self.points[i]+=1
            elif self.hands[(i+1)%2] == 4 and self.hands[i] == 5:
                self.points[i]+=1
            elif self.hands[i] == 1 and self.hands[(i+1)%2] == 5:
                self.points[i]+=1

    def process(self):
        for i in range(2):
            self.history1[self.turn][i]=self.hands[i]
            self.history2[self.turn][i]=self.hands[(i+1)%2]
            if self.hands[i] == 1:
                self.prays[i]=0
                self.charge[i]+=1
            elif self.hands[i] == 4:
                self.prays[i]=0
                self.charge[i]-=1
            elif self.hands[i] == 5:
                self.prays[i]=0
                self.charge[i]-=2
            elif self.hands[i] == 3:
                self.prays[i]+=1

    def get_hand_input(self,i):
        self.hands[i] = 0
        if self.charge[i]>=2:
            while self.hands[i] == 0:
                self.hands[i] = int(input())
                if self.hands[i] <1 or self.hands[i] > 5:
                    self.hands[i]=0
        elif self.charge[i]>=1:
            while self.hands[i] == 0:
                self.hands[i] = int(input())
                if self.hands[i] <1 or self.hands[i] > 4:
                    self.hands[i] = 0
        else:
            while self.hands[i] == 0:
                self.hands[i]= int(input())
                if self.hands[i] < 1 or self.hands[i] > 3:
                    self.hands[i] = 0

    def get_hand_random(self,i):
        if self.charge[i]>=2:
            self.hands[i] = random.randint(1,5)
        elif self.charge[i]>=1:
            self.hands[i] = random.randint(1,4)
        else:
            self.hands[i] = random.randint(1,3)

    def get_hand_speed(self,i):
        koho = [1,4]
        if self.charge[i] >= 2:
            self.hands[i] = 5
        elif self.charge[i] >= 1:
            self.hands[i] = koho[random.randint(1,2)]
        else:
            self.hands[i] = 1

    def get_hand_three(self,i):
        self.hands[i] = 3

    def get_hand_dqn(self,i):
        if self.charge[i]>=2:
            self.hands[i] = get_hand(self.history1,[1,2,3,4,5])
        elif self.charge[i]>=1:
            self.hands[i] = get_hand(self.history1,[1,2,3,4])
        else:
            self.hands[i] = get_hand(self.history1,[1,2,3])
        #return self.hands[i]

    def get_range_max(self,i):
        range_max = 0
        if self.charge[i]>=2:
            range_max = 5
        elif self.charge[i]>=1:
            range_max = 4
        else:
            range_max = 3
        return range_max



if __name__ == '__main__':
    game=Game()
    while game.turn<50:
        game.get_hand_dqn(0)
        game.get_hand_input(1)
        game.process()
        game.win = game.judge()
        print(game.history1)
        if game.turn==49:
            if self.game.win==0:
                self.win_count+=1
            break;
        game.turn+=1
