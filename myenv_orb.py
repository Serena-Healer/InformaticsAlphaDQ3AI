import sys

import gymnasium as gym
import numpy as np
import gymnasium.spaces

debugLogFlag = False

# ひかりのたま検証Ver.

# ドラクエ 3 SFC 版形式の乱数生成
def dq3random(random):
  d = 0
  for i in range(16):
    d += np.floor(random.random() * 16)
  if d < 99:
    d = 99
  if d > 153:
    d = 153
  return d / 256

class MyEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}
    MAP = np.arange(6, dtype=np.float32)
    MAX_STEPS = 1000

    erdrickHealth = 446 # あるすHP
    erdrickHealthLast = 446 # あるすHP
    erdrickMaxHealth = 600
    erdrickMana = 120 # あるすMP
    erdrickManaLast = 120 # あるすMP
    erdrickMaxMana = 200
    kaclang = 0 # アストロン残りターン

    zomaHealth = 1023 # ゾーマHP
    zomaHealthLast = 1023 # ゾーマHP
    zomaMaxHealth = 1023
    darkRobe = 1 # やみのころもフラグ

    steps = 0 # ターン

    # 以下定数
    erdrickStrength = 181 + 120
    erdrickDefence = 244 + 75 + 40
    zomaStrength = 500
    zomaDefence = 350
    zomaAgility = 255

    debugLogFlag = False

    def setLog(self, flag):
      MyEnv.debugLogFlag = flag

    def __init__(self):
        super().__init__()
        # action_space, observation_space, reward_range を設定する
        self.action_space = gym.spaces.Discrete(8)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=2048,
            shape=self.MAP.shape,
            dtype=np.float32
        )
        self.reward_range = [-1., 1.]
        self.reset()

    def reset(self):
        # 諸々の変数を初期化する
        self.done = False
        self.truncated = False
        self.erdrickHealth = MyEnv.erdrickMaxHealth # あるすHP
        self.erdrickMana = MyEnv.erdrickMaxMana # あるすMP
        self.kaclang = 0 # アストロン残りターン
        self.zomaHealth = MyEnv.zomaMaxHealth # ゾーマHP
        self.darkRobe = True # やみのころもフラグ
        self.steps = 0 # ターン
        return self.observe(), {}

    def write(self):
        self.MAP[0] = self.erdrickHealth
        self.MAP[1] = self.erdrickMana
        self.MAP[2] = self.kaclang
        self.MAP[3] = self.zomaHealth
        self.MAP[4] = self.steps
        self.MAP[5] = self.darkRobe
        return self.MAP.copy()

    def step(self, action):
        self.steps += 1
        defendFlag = 0

        self.erdrickHealthLast = self.erdrickHealth
        self.zomaHealthLast = self.zomaHealth

        debugLog = MyEnv.debugLogFlag

        manaLack = False

        if debugLog:
          print("あるすHP", self.erdrickHealth, "あるすMP", self.erdrickMana, "ゾーマHP", self.zomaHealth)

        #あるす行動
        #操作列

        if self.kaclang <= 0:
          if action == 0:
            dmg = np.floor((MyEnv.erdrickStrength - (300 if self.darkRobe else MyEnv.zomaDefence) / 2) * dq3random(np.random))
            if debugLog:
              print ("あるすの こうげき！")
              print ("ゾーマに ", dmg, " の ダメージ！")            
            self.zomaHealth -= dmg
          elif action == 1:
            manaCost = 7
            if self.erdrickMana >= manaCost:
              if debugLog:
                print ("あるすは ベホマを となえた！")
              self.erdrickHealth += 998244353
              self.erdrickMana -= manaCost
            else:
              manaLack = True
          elif action == 2:
            manaCost = 8
            if self.erdrickMana >= manaCost:
              dmg = np.floor(np.random.random() * 20) + 70
              self.zomaHealth -= dmg
              self.erdrickMana -= manaCost
              if debugLog:
                print ("あるすは ライデインを となえた！")
                print ("ゾーマに ", dmg, " の ダメージ！")
            else:
              manaLack = True
          elif action == 3:
            manaCost = 30
            if self.erdrickMana >= manaCost:
              dmg = np.floor(np.random.random() * 40) + 175
              self.zomaHealth -= dmg
              self.erdrickMana -= manaCost
              if debugLog:
                print ("あるすは ギガデインを となえた！")
                print ("ゾーマに ", dmg, " の ダメージ！")
            else:
              manaLack = True
          elif action == 4:
            self.darkRobe = False
            if debugLog:
              print ("あるすは ひかりのたまを つかった！")

          if self.erdrickMaxHealth < self.erdrickHealth:
            self.erdrickHealth = self.erdrickMaxHealth
  
        #ゾーマ行動
        if self.zomaHealth > 0 and self.kaclang <= 0:
          for i in range(2):
            zomaAction = np.random.random()
            if self.darkRobe:
              if (self.steps * 2 + i) % 5 == 0:
                zomaAction = 0.999
              if (self.steps * 2 + i) % 5 == 1:
                zomaAction = 0.749
              if (self.steps * 2 + i) % 5 == 2:
                zomaAction = 0.999
              if (self.steps * 2 + i) % 5 == 3:
                zomaAction = 0.000
              if (self.steps * 2 + i) % 5 == 4:
                zomaAction = 0.624
            dmg = 0
            if zomaAction <= 0.375:
              dmg = np.floor(((550 if self.darkRobe else MyEnv.zomaStrength) - MyEnv.erdrickDefence / 2) * dq3random(np.random))
              if defendFlag:
                dmg = np.floor(dmg / 2)
              if debugLog:
                print ("ゾーマの こうげき！")
                print ("あるすは ", dmg, " の ダメージを うけた！")
            elif zomaAction <= 0.625:
              self.kaclang = 0
              if debugLog:
                print ("ゾーマは いてつくはどうを まきおこした！")
            elif zomaAction <= 0.750:
              dmg = np.floor(np.floor(55 + np.random.random() * 12) * 2 / 3)
              if defendFlag:
                dmg = np.floor(dmg / 2)
              if debugLog:
                print ("ゾーマは マヒャドを となえた！")
                print ("あるすは ", dmg, " の ダメージを うけた！")
            elif zomaAction <= 0.875:
              dmg = np.floor(np.floor(40 + np.random.random() * 20) * 2 / 3)
              if defendFlag:
                dmg = np.floor(dmg / 2)
              if debugLog:
                print ("ゾーマは こおりのいきを はいた！")
                print ("あるすは ", dmg, " の ダメージを うけた！")
            elif zomaAction <= 1.000:
              dmg = np.floor(np.floor(100 + np.random.random() * 40) * 2 / 3)
              if defendFlag:
                dmg = np.floor(dmg / 2)
              if debugLog:
                print ("ゾーマは こごえるふぶきを はいた！")
                print ("あるすは ", dmg, " の ダメージを うけた！")
            self.erdrickHealth -= dmg

        self.kaclang -= 1

        if self.zomaHealth <= 0:
          if debugLog:
            print ("ゾーマを たおした！")

        observation = self.observe()
        reward = self._get_reward(manaLack)
        # self.damage += self._get_damage(self.pos)
        self.done = self._is_done()
        self.truncated = self._is_truncated()
        return observation, reward, self.done, self.truncated, {"orb": not self.darkRobe}

    def render(self, mode='human', close=False):
        # human の場合はコンソールに出力。ansiの場合は StringIO を返す
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write('\n'.join(' '.join(
                self.FIELD_TYPES[elem] for elem in row
                ) for row in self._observe()
            ) + '\n'
        )
        return outfile

    def close(self):
        pass

    def _seed(self, seed=None):
        pass

    def _get_reward(self, manaLackFlag):
        # 報酬を返す
        # 仮設定
        # 倒したorチカラつきた
        if self.zomaHealth <= 0: 
          return max(1.0 - (self.steps * 0.001), 0.5)
        if self.erdrickHealth <= 0: 
          return -(self.zomaHealth / self.zomaMaxHealth)

        # (与ダメージ/最大HP)-(被ダメージ/最大HP)
        reward = 0.01
        reward += (self.zomaHealthLast - self.zomaHealth) / self.zomaMaxHealth * 0.3
        reward += pow(max((self.erdrickHealth - self.erdrickHealthLast) / self.erdrickMaxHealth, 0), 2) * 0.1

        # 残りHP/最大HP
        reward -= (self.zomaHealth / self.zomaMaxHealth) * 0.02

        # あるすHPは対数関数を使って
        reward += np.log(self.erdrickHealth / self.erdrickMaxHealth) * 0.02

        #MPが たりない！ 対策
        if manaLackFlag:
          reward -= 0.1
         
        #終了時以外で過大報酬をもらわないよう絶対値を0.05以下に制限
        if reward > 0.05:
          reward = 0.05
        if reward < -0.05:
          reward = -0.05

        if MyEnv.debugLogFlag:
          print("このターンの報酬:", reward)
        return reward

    def observe(self):
        observation = self.write()
        return observation

    def _is_done(self):
        # 今回は最大で self.MAX_STEPS までとした
        if self.zomaHealth <= 0 or self.erdrickHealth <= 0:
            return True
        else:
            return False

    def _is_truncated(self):
        # 今回は最大で self.MAX_STEPS までとした
        if self.steps > self.MAX_STEPS:
            return True
        else:
            return False

    def _find_pos(self, field_type):
        return np.array(list(zip(*np.where(
        self.MAP == self.FIELD_TYPES.index(field_type)
    ))))

  
