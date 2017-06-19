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
        self.history = [[0 for j in range(2)]for i in range(25)]#プレイヤーにとっての履歴
        self.charge = [0 for i in range(2)]#ため時間
        self.turn = 0
        self.win = -1
        self.prays = [0 for i in range(2)]
        self.hands = [0 for i in range(2)]
        self.last_moves = [0 for i in range(2)]
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
            self.history[self.turn][i]=self.hands[i]
            self.last_moves[i]=self.hands[i]
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
        koho2 = [4,5]
        if self.charge[i] >= 2:
            self.hands[i] = koho2[random.randint(0,1)]
        elif self.charge[i] >= 1:
            self.hands[i] = koho[random.randint(0,1)]
        else:
            self.hands[i] = 1

    def get_hand_four(self,i):
        if self.charge[i] >= 1:
            self.hands[i] = 4
        else:
            self.hands[i] = 1

    def get_hand_five(self,i):
        if self.turn == 1:
            self.hands[i] = 2
        elif self.charge[i] >= 2:
            self.hands[i] = 5
        else:
            self.hands[i] = 1

    def get_hand_three(self,i):
        if self.last_moves[i]==3:
            self.hands[i] = 3
        elif self.charge[i]>=2:
            self.hands[i] = random.randint(1,5)
        elif self.charge[i]>=1:
            self.hands[i] = random.randint(1,4)
        else:
            self.hands[i] = random.randint(1,3)
        return self.hands[i]

    def get_hand_all_three(self,i):
        self.hands[i] = 3

    def get_hand_two(self,i):
        if self.charge[(i+1)%2]>=1:
            self.hands[i] = 2
        elif self.charge[i]>=2:
            self.hands[i] = random.randint(1,5)
        elif self.charge[i]>=1:
            self.hands[i] = random.randint(1,4)
        else:
            self.hands[i] = random.randint(1,3)
        return self.hands[i]

    def get_hand_one(self,i):
        self.hands[i] = 1

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
    def __init__(self,e=1):
        self.model = MLP(60,162,5)
        #serializers.load_npz("model2.npz", self.model)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.e=e
        self.gamma=0.8
        self.last_move=None
        self.history=None
        self.last_pred=None
        self.rwin,self.rlose,self.rdraw=1,-1,0

    def play(self):
        self.win_count=0
        #aが0でbが1、何もなければ-1
        for p in range (20000):
            if p % 1000 == 0:
                print(self.win_count,self.e)
                self.win_count=0
            self.game = Game()
            while self.game.turn<30:
                self.act(0)
                if p%6 == 0 and self.e < 0.25:
                    self.game.get_hand_three(1)
                elif p%6 == 1 and self.e < 0.25:
                    self.game.get_hand_four(1)
                elif p%6 == 2 and self.e < 0.25:
                    self.game.get_hand_two(1)
                elif p%6 == 3 and self.e < 0.25:
                    self.game.get_hand_five(1)
                elif p%6 == 4 and self.e < 0.25:
                    self.game.get_hand_four(1)
                elif p%6 == 5 and self.e < 0.25:
                    self.game.get_hand_speed(1)
                else:
                    self.game.get_hand_random(1)
                self.game.process()
                self.game.judge()
                if self.game.turn==29:
                    #どっちの方が勝ってるか
                    self.game.win=np.argmax(np.array(self.game.points))
                    if self.game.win==0:
                        self.win_count+=1
                    break;
                self.getGameResult()
                self.game.turn+=1
        serializers.save_npz("model2.npz", self.model)

    def act(self,i):
        self.history=self.game.history
        x=np.array([self.game.history],dtype=np.float32).astype(np.float32)
        pred=self.model(x)
        self.last_pred = pred.data[0,:]
        #print(self.last_pred,np.argmax(pred.data,axis=1)[0])
        act=np.argmax(pred.data,axis=1)[0]+1
        if self.e > 0.2:
            self.e -= 1/(20000)
        if random.random() < self.e:
            act = random.randint(1,self.game.get_range_max(i))
        error_i=0#ルールに違反した回数
        while act < 1 and act > self.game.get_range_max():#ルールに違反した場合
            self.learn(self.history,act, -0.2, self.game)
            x=np.array([self.game.history],dtype=np.float32).astype(np.float32)
            pred=self.model(x)
            act=np.argmax(pred.data,axis=1)[0]+1
            error_i+=1
            if error_i>10:
                #print("だめ")
                act = random.randint(1,self.game.get_range_max(i))
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
        update = r + (self.gamma*maxQnew)
        self.last_pred[a-1]=update
        x=np.array([s],dtype=np.float32).astype(np.float32)
        t=np.array([self.last_pred],dtype=np.float32).astype(np.float32)
        if abs(update) > 1:
            t=t/abs(update)
        self.model.zerograds()
        self.model(x,t,train=True).backward()
        self.optimizer.update()


if __name__ == '__main__':
    dqn = DQN()
    dqn.play()
