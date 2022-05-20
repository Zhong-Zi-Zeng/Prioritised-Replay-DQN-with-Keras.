from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from SumTree import Tree
import tensorflow as tf
import numpy as np


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)


"""建立神經網路"""
def build_dqn(lr,n_actions,input_shape):
    input = Input(shape=(input_shape,))

    h1 = Dense(1024,activation='relu')(input)
    h2 = Dense(512, activation='relu')(h1)
    h3 = Dense(256, activation='relu')(h2)
    h4 = Dense(128, activation='relu')(h3)
    output = Dense(n_actions,activation='linear')(h4)

    model = Model(inputs=[input],outputs=[output])
    model.compile(Adam(learning_rate=lr),loss='mse')
    model.summary()

    return model

"""暫存器設置"""
class ReplayBuffer():
    beta = 0.4
    beta_incremental = 0.001
    abs_err_upper = 1.
    alpha = 0.6

    def __init__(self,max_mem,n_action,input_shape,use_pri):
        self.max_mem = int(2 ** (np.floor(np.log2(max_mem)))) # 保證記憶體大小為2的次方
        self.use_pri = use_pri

        if self.use_pri:
            self.tree = Tree(self.max_mem)

        self.state_memory = np.zeros((self.max_mem,input_shape),dtype=np.float32)
        self.next_state_memory = np.zeros((self.max_mem,input_shape),dtype=np.float32)
        self.action_memory = np.zeros((self.max_mem,n_action),dtype=np.int8)
        self.reward_memory = np.zeros(self.max_mem)
        self.terminal_memory = np.zeros(self.max_mem,dtype=np.float32)
        self.mem_counter = 0

    def store_transition(self,state,action,reward,next_state,done):
        actions = np.zeros(self.action_memory.shape[1])
        actions[action] = 1
        self.action_memory[self.mem_counter] = actions
        self.state_memory[self.mem_counter] = state
        self.reward_memory[self.mem_counter] = reward
        self.next_state_memory[self.mem_counter] = next_state
        self.terminal_memory[self.mem_counter] = 1 - int(done)

        if self.use_pri:
            # 存入新位元組，並賦與最高優先級
            max_p = np.max(self.tree.tree[-self.tree.start:])
            if max_p == 0:
                max_p = self.abs_err_upper
            self.tree.add(self.mem_counter + 1, max_p)

        self.mem_counter += 1
        if(self.mem_counter == self.max_mem):
            self.mem_counter = 0

    """從memory隨機抽取mini_batch的資料"""
    def sample_buffer(self,batch_size):
        if self.use_pri:
            weights = np.zeros((batch_size),dtype=np.float32)
            leaf_idx = np.zeros((batch_size),dtype=np.uint32)
            batch_idx = np.zeros((batch_size),dtype=np.uint32)

            root_node_val = self.tree.get_total()
            linspace = root_node_val / batch_size
            self.beta = np.min([1., self.beta + self.beta_incremental]) # max = 1

            min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / root_node_val
            if min_prob == 0:
                min_prob = 0.00001

            for i in range(batch_size):
                min, max = i * linspace, (i + 1) * linspace
                rand = np.random.uniform(min,max)
                p_idx, leaf_val = self.tree.get_leaf(rand)
                prob = leaf_val / root_node_val
                weights[i] = np.power(prob / min_prob, -self.beta)
                leaf_idx[i] = p_idx
                batch_idx[i] = p_idx - self.tree.start

        else:
            max_mem = np.minimum(self.mem_counter, self.max_mem)
            batch_idx = np.random.choice(max_mem, batch_size)

        batch_state = self.state_memory[batch_idx]
        batch_action = self.action_memory[batch_idx]
        batch_reward = self.reward_memory[batch_idx]
        batch_next_state = self.next_state_memory[batch_idx]
        batch_done = self.terminal_memory[batch_idx]

        if self.use_pri:
            return batch_state, batch_action, batch_reward, batch_next_state, batch_done, weights, leaf_idx
        else:
            return batch_state, batch_action, batch_reward, batch_next_state, batch_done

    def batch_update_tree(self,leaf_idx,abs_error):
        abs_error += 0.01  # 避免概率為0
        clip_error = np.minimum(abs_error, self.abs_err_upper) # 將大於1的p值都設為1
        new_p = np.power(clip_error, self.alpha) # 為了避免過於大的p值頻繁出現，所以讓p值加了一個小於1的次方
        self.tree.batch_update(leaf_idx,new_p)

class Agent():
    def __init__(self
                 ,alpha
                 ,gamma
                 ,n_actions
                 ,epsilon
                 ,batch_size
                 ,epsilon_end
                 ,mem_size
                 ,epsilon_dec
                 ,input_shape
                 ,iteration
                 ,use_pri
                 ,f_name="dqn_model.h5"):

        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.model_file = f_name
        self.input_shape = input_shape
        self.iteration = iteration
        self.use_pri = use_pri
        self.iteration_counter = 0

        """建置資料庫"""
        self.memory = ReplayBuffer(max_mem=mem_size,n_action=n_actions,input_shape=self.input_shape,use_pri=self.use_pri)
        """建置模型"""
        self.q_eval = build_dqn(lr=self.alpha,n_actions=self.n_actions,input_shape=self.input_shape)
        self.q_target_net = build_dqn(lr=self.alpha, n_actions=self.n_actions,input_shape=self.input_shape)
        self.q_target_net.set_weights(self.q_eval.get_weights())

    def remember(self,state,action,reward,next_state,done):
        self.memory.store_transition(state,action,reward,next_state,done)

    def choose_action(self,state):
        state = np.array(state)
        state = state[np.newaxis,:]

        rand = np.random.random()
        if(rand < self.epsilon):
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        return action

    def learn(self):
        if(self.memory.mem_counter < self.batch_size):
            return

        if self.use_pri:
            state, action, reward, next_state, done, weights, leaf_idx = self.memory.sample_buffer(self.batch_size)
        else:
            state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)

        action_values = np.array(self.action_space,dtype=np.int8)
        action_indices = np.dot(action,action_values)

        # 每batch_size次後下降epsilon
        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_end else self.epsilon_end

        # 舊神經網路預測的Q值
        q_eval = self.q_eval.predict(state)

        # 新神經網路預測的Q值
        q_target_pre = self.q_target_net.predict(next_state)

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # 貝爾曼方程
        q_target[batch_index, action_indices] = reward + self.gamma * np.max(q_target_pre, axis=1) * done

        # 更新參數
        if self.use_pri:
            abs_error = abs(q_target[batch_index, action_indices] - q_eval[batch_index, action_indices])
            self.memory.batch_update_tree(leaf_idx, abs_error)

            class_weights = {i : weight for i, weight in zip(range(self.batch_size), weights)}

            _ = self.q_eval.fit(state, q_target, verbose=0, class_weight=class_weights)
        else:
            _ = self.q_eval.fit(state, q_target, verbose=0)

        """到達指定迭代次數後，複製權重給q_target_net"""
        self.iteration_counter += 1
        if(self.iteration_counter == self.iteration):
            self.q_target_net.set_weights(self.q_eval.get_weights())
            self.iteration_counter = 0


    """儲存模型"""
    def save_model(self):
        self.q_eval.save_weights(self.model_file)

    """載入模型"""
    def load_model(self):
        self.q_eval.load_weights(self.model_file)
        self.q_target_net.load_weights(self.model_file)


