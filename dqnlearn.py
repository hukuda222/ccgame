import random
import pickle
import chainer
from chainer import Function, Variable, optimizers, utils
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import computational_graph as c
from chainer import serializers

# Network definition
class MLP(chainer.Chain):
    def __init__(self, n_in, n_units, n_out):
        super(MLP, self).__init__(
            l1=L.Linear(n_in, n_units),  # first layer
            l2=L.Linear(n_units, n_units),  # second layer
            l3=L.Linear(n_units, n_units),  # Third layer
            l4=L.Linear(n_units, n_out),  # output layer
        )

    def __call__(self, x, t=None, train=False):
        h = F.leaky_relu(self.l1(x))
        h = F.leaky_relu(self.l2(h))
        h = F.leaky_relu(self.l3(h))
        h = self.l4(h)

        if train:
            return F.mean_squared_error(h,t)
        else:
            return h

    def get(self,x):
        return self.predict(Variable(np.array([x]).astype(np.float32).reshape(1,1))).data[0][0]

class Game:
    def __init__(self):
        self.history1 = [[0 for j in range(2)]for i in range(20)]#プレイヤー1にとっての履歴
        self.history2 = [[0 for j in range(2)]for i in range(20)]#プレイヤー2にとっての履歴
        self.charge = [0 for i in range(2)]#ため時間
        self.turn = 0
        self.win = -1
        self.prays = [0 for i in range(2)]
        self.hands = [0 for i in range(2)]

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
            elif self.hands[i] == 4:
                self.prays[i]=0
                self.charge[i]-=1
            elif self.hands[i] == 5:
                self.prays[i]=0
                self.charge[i]-=2
            elif self.hands[i] == 3:
                self.prays[i]+=1

    def get_hand_random(self,i):
        if self.charge[i]>=2:
            self.hands[i] = random.randint(1,5)
        elif self.charge[i]>=1:
            self.hands[i] = random.randint(1,4)
        else:
            self.hands[i] = random.randint(1,3)
        return self.hands[i]

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
        return self.hands[i]

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

    def get_range_max(self,i):
        range_max = 0
        if self.charge[i]>=2:
            range_max = 5
        elif self.charge[i]>=1:
            range_max = 4
        else:
            range_max = 3
        return range_max

class DQN:
    def __init__(self,e=1,dispPred=False):
        self.model = MLP(40,162,5)
        self.optimizer = optimizers.SGD()
        self.optimizer.setup(self.model)
        self.e=e
        self.gamma=0.9
        self.dispPred=dispPred
        self.last_move=None
        self.history=None
        self.last_pred=None
        self.total_game_count=0
        self.rwin,self.rlose,self.rdraw,self.rmiss=1,-1,-1,-2

    def play(self):
        #aが0でbが1、何もなければ-1
        self.game = Game()
        self.type = random.randint(1,3)
        while self.game.turn<20:
            self.act(0)
            if self.type == 1:
                self.game.get_hand_random(1)
            elif self.type == 2:
                self.game.get_hand_speed(1)
            elif self.type == 3:
                self.game.get_hand_three(1)
            self.game.process()
            self.game.win = self.game.judge()
            self.getGameResult()
            if self.game.win is not -1:
                if self.game.win==0:
                    self.win_count+=1
                break;
            self.game.turn+=1
        serializers.save_npz("model2.npz", self.model)

    def act(self,i):
        self.history=self.game.history1
        x=np.array([self.game.history1],dtype=np.float32).astype(np.float32)
        pred=self.model(x)
        if self.dispPred:
            print(pred.data)
        self.last_pred = pred.data[0,:]
        act=np.argmax(pred.data,axis=1)[0]+1
        if self.e > 0.2:
            self.e -= 1/(2000)
        if random.random() < self.e:
            act = random.randrange(self.game.get_range_max(i))+1
        error_i=0#ルールに違反した回数
        while act < 1 and act > self.game.get_range_max():#ルールに違反した場合
            self.learn(self.history,act, -0.25, self.game)
            x=np.array([self.game.history1],dtype=np.float32).astype(np.float32)
            pred=self.model(x)
            act=np.argmax(pred.data,axis=1)[0]+1
            error_i+=1
            if error_i>10:
                #print("だめ")
                act = random.randrange(self.game.get_range_max(i))+1
        self.last_move=act
        self.game.hands[i]=act

    def getGameResult(self):
        if self.game.win == -1:
            self.learn(self.history,self.last_move, 0, self.game)
        else:
            if self.game.win == 0:
                self.learn(self.history,self.last_move, self.rwin, self.game)
            elif self.game.win == 1:
                self.learn(self.history,self.last_move, self.rlose, self.game)
            else:  #DRAW
                self.learn(self.history,self.last_move, self.rdraw, self.game)
            self.total_game_count+=1
            self.last_move=None
            self.history=None
            self.last_pred=None

    #sは履歴、aはcpuの最後の動き、rは報酬、fsはself.game
    def learn(self,s,a,r,fs):
        if fs.win != -1:
            maxQnew=0
        else:
            x=np.array([self.history],dtype=np.float32).astype(np.float32)
            #maxQnewは次の手で一番報酬が高い手の報酬
            #print(self.model(x).data[0])
            maxQnew=np.max(self.model(x).data[0])
        update=r+self.gamma*maxQnew
        self.last_pred[a-1]=update
        x=np.array([s],dtype=np.float32).astype(np.float32)
        t=np.array([self.last_pred],dtype=np.float32).astype(np.float32)
        self.model.zerograds()
        self.model(x,t,train=True).backward()
        self.optimizer.update()


if __name__ == '__main__':
    dqn = DQN()
    dqn.play()
