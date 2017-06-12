import random
import pickle
import numpy as np

#1が貯め、2が守り、3が攻撃、4が大攻撃
max_count = 20

def judge(hands,history1,history2,charge,turn):
    process(hands,history1,history2,charge,turn)
    for i in range(2):
        if hands[(i+1)%2] == 1 and hands[i] == 3:
            return i
        elif hands[i] == 1 and hands[(i+1)%2] == 4:
            return i
        elif hands[(i+1)%2] == 2 and hands[i] == 4:
            return i
        elif hands[(i+1)%2] == 3 and hands[i] == 4:
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
    #print(hands)

def get_hands(charge):
    hands=list()
    if charge[0]==0 and charge[1]==0:
        hands.append(1)
        hands.append(1)
    else:
        for i in range(2):
            if charge[i]>=2:
                hands.append(random.randint(1,4))
            elif charge[i]>=1:
                hands.append(random.randint(1,3))
            else:
                hands.append(random.randint(1,2))
        return hands

def get_game():
    history1 = [[0 for j in range(2)]for i in range(20)]
    history2 = [[0 for j in range(2)]for i in range(20)]
    hyoka1=np.array([0 for i in range(20)],dtype=float)
    hyoka2=np.array([0 for i in range(20)],dtype=float)
    charge = [0 for i in range(2)]
    turn = 0;
    win = -1
    #aが0でbが1、何もなければ-1

    while turn<20 :
        win = judge(get_hands(charge),history1,history2,charge,turn)
        if win is not -1:
            break;
        turn+=1
    turn+=1
    loop = turn
    if loop>20:
        loop=20
    for j in range(loop):
        if win == 0:
            hyoka1[j]=1/(turn*(1+turn-j))
            hyoka2[j]=-1/(turn*(1+turn-j))
        else:
            hyoka1[j]=-1/(turn*(1+turn-j))
            hyoka2[j]=1/(turn*(1+turn-j))
    return [history1,history2],[hyoka1,hyoka2],loop

if __name__ == '__main__':
    student=list();
    ans=list();
    for j in range(100000):
        if j%100==0:
            print(j)
        result,hyoka,turn = get_game()
        for r in range(2):
            for i in range(turn):
                po = [result[r][i][0]]
                for r1 in result[r][:i]:
                    for r2 in r1:
                        po.append(r2)
                po.extend([0 for i in range((20-i)*2)])
                #print(po)
                student.append(po)
                ans.append([hyoka[r][i]])
    with open('student.pickle', mode='wb') as f:
        pickle.dump(np.array(student), f)
    with open('ans.pickle', mode='wb') as f2:
        pickle.dump(np.array(ans), f2)
