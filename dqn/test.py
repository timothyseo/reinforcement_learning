import sys
import os 
import gym
import pylab
import random
import numpy as np
from collections import deque
import tensorflow as tf
from keras.layers import Dense
from keras.initializers import RandomUniform


# 상태가 입력, 큐함수가 출력인 인공신경망 생성
class DQN(tf.keras.Model):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.fc1 = Dense(24, activation='relu')
        self.fc2 = Dense(24, activation='relu')
        self.fc_out = Dense(action_size,
                            kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.fc_out(x)
        return q


# 카트폴 예제에서의 DQN 에이전트
class DQNAgent:
    def __init__(self, state_size, action_size):
        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # 모델과 타깃 모델 생성
        self.model = DQN(action_size)
        # 학습 코드에서 지정한 디렉토리와 파일 접두사를 동일하게 사용
        save_dir = "./saved_dqn_weights"
        weights_prefix = "dqn_cartpole_weights"
        weights_path = os.path.join(save_dir, weights_prefix)

        # 가중치 파일이 실제로 존재하는지 확인 (더 안전하게)
        # TensorFlow 체크포인트는 여러 파일로 구성되므로, .index 파일 존재 여부 확인
        index_file_path = weights_path + ".index"
        if os.path.exists(index_file_path):
            print(f"Loading weights from: {weights_path}")
            # load_weights에는 파일 접두사만 전달
            self.model.load_weights(weights_path)
        else:
            print(f"Error: Weights file not found at the specified path: {weights_path}")
            print("Please ensure the training script ran successfully and saved the weights.")
            sys.exit() # 가중치 없으면 종료

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        q_value = self.model(state)
        return np.argmax(q_value[0])


if __name__ == "__main__":
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # DQN 에이전트 생성
    agent = DQNAgent(state_size, action_size)

    num_episode = 10
    for e in range(num_episode):
        done = False
        score = 0
        # env 초기화
        state, info = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            env.render()

            # 현재 상태로 행동을 선택
            action = agent.get_action(state)
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, state_size])

            score += reward
            state = next_state

            if done:
                # 에피소드마다 학습 결과 출력
                print("episode: {:3d} | score: {:.3f} ".format(e, score))