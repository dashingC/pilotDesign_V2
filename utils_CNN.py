import math
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Softmax, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant, glorot_normal
from tensorflow.keras.optimizers import Adam
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split  # 用于数据集分割
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU,Reshape,Flatten
from tensorflow.keras.layers import (Conv2D, Input, BatchNormalization, Activation, Lambda, Subtract,Conv2DTranspose, PReLU)




class ConcreteAutoencoderFeatureSelector:

    def __init__(self, K, output_function, num_epochs=300, batch_size=None, learning_rate=0.001, start_temp=10.0,
                 min_temp=0.1, tryout_limit=2):
        self.K = K # K= 48 ,输出函数是插值函数
        self.output_function = output_function
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.tryout_limit = tryout_limit

    def fit(self, X, Y=None, val_X=None, val_Y=None):

        if Y is None:
            Y = X
        assert len(X) == len(Y)
        validation_data = None
        if val_X is not None and val_Y is not None:
            assert len(val_X) == len(val_Y)
            validation_data = (val_X, val_Y)

        if self.batch_size is None:
            self.batch_size = max(len(X) // 256, 16)
        num_epochs = self.num_epochs
        # steps_per_epoch = (len(X) + self.batch_size - 1) // self.batch_size


        for i in range(self.tryout_limit):# 循环的目的是尝试多次训练模型，每次训练时逐步调整训练的超参数

            print(f"\n--- 开始训练尝试 {i + 1}/{self.tryout_limit} ---")  # 添加尝试次数打印
            print(f"   本次尝试最多训练 {num_epochs} 个 epochs")  # 添加 epoch 数打印
            steps_per_epoch = (len(X) + self.batch_size - 1) // self.batch_size

            input_shape_2d = X.shape[1:]  # 获取 (72, 14, 1)
            inputs = Input(shape=input_shape_2d, name="model_input_2d")  # <--- 1. 输入层 (2D)
            flattened_inputs = Flatten(name="flatten_input")(inputs)  # <--- 2. 添加 Flatten 层

            alpha = math.exp(math.log(self.min_temp / self.start_temp) / (num_epochs * steps_per_epoch))

            self.concrete_select = ConcreteSelect(self.K, self.start_temp, self.min_temp, alpha, name='concrete_select') # <--- 3. 初始化 Select
            selected_features = self.concrete_select(flattened_inputs)  # <--- 4. Select 处理展平后输入

            # print("Selected Features Shape:", selected_features.shape)
            outputs = self.output_function(selected_features)  # <--- 5. CNN 插值 (输出 2D)
            # print("outputs Shape:", outputs.shape)
            #self.model = Model(inputs, outputs)
            self.model = Model(inputs=inputs, outputs=outputs, name="FeatureSelectorCNN")  # <--- 6. 构建模型 (Input 2D, Output 2D)

            self.model.compile(Adam(self.learning_rate), loss='mean_squared_error')
            print(self.model.summary())
            stopper_callback = StopperCallback()
            hist = self.model.fit(X, Y, self.batch_size, num_epochs, verbose=1, callbacks=[stopper_callback],
                                  validation_data=validation_data)
            if K.get_value(K.mean(
                    K.max(K.softmax(self.concrete_select.logits, axis=-1)))) >= stopper_callback.mean_max_target:
                break
            num_epochs *= 2

        if self.model is not None and hasattr(self.concrete_select, 'logits'):
            # 获取最终的 logits 张量
            final_logits_tensor = self.concrete_select.logits
            # 计算 logits 每一行的最大值
            max_logits_per_row = K.max(final_logits_tensor, axis=-1)  # axis=-1 通常等同于 axis=1 对于 2D 张量
            # 计算这些最大值的均值
            mean_of_max_logits = K.mean(max_logits_per_row)
            # 获取最终的温度
            final_temp = self.concrete_select.temp

            # 使用 K.get_value() 获取实际的数值
            final_mean_max_logits_value = K.get_value(mean_of_max_logits)
            final_temp_value = K.get_value(final_temp)

            # 打印你需要的最终信息
            print(
                f"\n训练结束: Logits 的均值最大值: {final_mean_max_logits_value:.6f} - 最终温度: {final_temp_value:.6f}")
        else:
            print("\n训练未完成或模型未正确初始化，无法计算最终 Logits 统计信息。")
        # --- >>> 添加结束 <<< ---

        self.probabilities = K.get_value(K.softmax(self.model.get_layer('concrete_select').logits))
        self.indices = K.get_value(K.argmax(self.model.get_layer('concrete_select').logits))
        return self

    def get_indices(self):
        return K.get_value(K.argmax(self.model.get_layer('concrete_select').logits))

    def get_mask(self):
        return K.get_value(K.sum(K.one_hot(K.argmax(self.model.get_layer('concrete_select').logits),
                                           self.model.get_layer('concrete_select').logits.shape[1]), axis=0))

    def transform(self, X):
        return X[self.get_indices()]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices=False):
        return self.get_indices() if indices else self.get_mask()

    def get_params(self):
        return self.model

class ConcreteSelect(Layer):

    def __init__(self, output_dim, start_temp=10.0, min_temp=0.1, alpha=0.99999, **kwargs):
        self.output_dim = output_dim  # 需要选择的特征数量
        self.start_temp = start_temp  # 初始温度
        self.min_temp = K.constant(min_temp)   # 最小温度
        self.alpha = K.constant(alpha)  # 温度衰减率
        super(ConcreteSelect, self).__init__(**kwargs)

    def build(self, input_shape):
        self.temp = self.add_weight(name='temp', shape=[], initializer=Constant(self.start_temp), trainable=False) #温度参数，非训练权重（通过手动更新）
        self.logits = self.add_weight(name='logits', shape=[self.output_dim, input_shape[1]],# 特征选择的概率分布参数（可训练），形状为 [output_dim, input_features]。
                                      initializer=glorot_normal(), trainable=True)
        super(ConcreteSelect, self).build(input_shape)

    def call(self, X, training=None):
        uniform = K.random_uniform(self.logits.shape, K.epsilon(), 1.0)
        gumbel = -K.log(-K.log(uniform)) #  Gumbel噪声
        temp = K.update(self.temp, K.maximum(self.min_temp, self.temp * self.alpha)) # 温度衰减，更新参数
        noisy_logits = (self.logits + gumbel) / temp # 添加噪声并除以温度
        samples = K.softmax(noisy_logits) # 生成平滑选择概率
        discrete_logits = K.one_hot(K.argmax(self.logits), self.logits.shape[1])#one-hot形式
        self.selections = K.in_train_phase(samples, discrete_logits, training)# 训练时使用平滑概率，推理时使用离散选择
        Y = K.dot(X, K.transpose(self.selections))# 选择特征：输入矩阵与选择概率的矩阵乘法
        return Y
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class StopperCallback(EarlyStopping):

    def __init__(self, mean_max_target=0.998):
        self.mean_max_target = mean_max_target
        super(StopperCallback, self).__init__(monitor='', patience=float('inf'), verbose=1, mode='max',
                                              baseline=self.mean_max_target)

    def on_epoch_begin(self, epoch, logs=None):
        print('模型开始训练时的概率最大值的均值:', self.get_monitor_value(logs), '- 初始温度',
              K.get_value(self.model.get_layer('concrete_select').temp))

    def get_monitor_value(self, logs):
        monitor_value = K.get_value(K.mean(K.max(K.softmax(self.model.get_layer('concrete_select').logits), axis=-1)))
        return monitor_value

def interpolate_model(x):
    x = Dense(150)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(320)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(780)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)
    x = Dense(1008)(x)
    return x
def load_channel(num_pilots, SNR):
    # 完美信道估计
    perfect = loadmat("./VehA_perfect_all.mat")["H_p_rearranged"]
    perfect = np.transpose(perfect, [2, 0, 1])
    print("\n输入完美信道大小：", perfect.shape)
    perfect_image = np.zeros((len(perfect), 72, 14, 2))
    perfect_image[:, :, :, 0] = np.real(perfect)
    perfect_image[:, :, :, 1] = np.imag(perfect)
    #1perfect_image = np.concatenate((perfect_image[:, :, :, 0], perfect_image[:, :, :, 1]), axis=0).reshape(2 * len(perfect), 72, 14, 1)
    #1perfect_image = perfect_image.squeeze()
    #1perfect_image = perfect_image.reshape((perfect_image.shape[0], np.dot(perfect_image.shape[1], perfect_image.shape[2])))
    perfect_image = np.concatenate((perfect_image[:, :, :, 0], perfect_image[:, :, :, 1]), axis=0)  # (2*N, 72, 14)
    perfect_image = np.expand_dims(perfect_image, axis=-1)  # <--- 添加通道维 (2*N, 72, 14, 1)
    print("\n分离实部和虚部，展平后的输入完美信道大小：", perfect_image.shape)

    noisy = loadmat("./VehA_noisy_all.mat")["H_p_noisy"]
    noisy = np.transpose(noisy, [2, 0, 1])
    print("\n输入含噪信道大小：", noisy.shape)
    noisy_image = np.zeros((len(noisy), 72, 14, 2))
    noisy_image[:, :, :, 0] = np.real(noisy)
    noisy_image[:, :, :, 1] = np.imag(noisy)
    #1noisy_image = np.concatenate((noisy_image[:, :, :, 0], noisy_image[:, :, :, 1]), axis=0).reshape(2 * len(noisy), 72, 14, 1)
    #1noisy_image = noisy_image.squeeze()
    #1noisy_image = noisy_image.reshape((noisy_image.shape[0], np.dot(noisy_image.shape[1], noisy_image.shape[2])))
    noisy_image = np.concatenate((noisy_image[:, :, :, 0], noisy_image[:, :, :, 1]), axis=0)  # (2*N, 72, 14)
    noisy_image = np.expand_dims(noisy_image, axis=-1)  # <--- 添加通道维 (2*N, 72, 14, 1)
    print("\n分离实部和虚部，展平后的输入含噪信道大小：", noisy_image.shape)

    train_data, test_data, train_label, test_label = train_test_split(noisy_image, perfect_image, test_size=1 / 9,random_state=1)
    train_data, val_data, train_label, val_label = train_test_split(train_data, train_label, test_size=1 / 8,    random_state=1)
    return (train_data, train_label), (val_data, val_label), (test_data, test_label)
def interpolate_model_cnn(selected_features): # 输入 K 个特征 (None, K)
    if isinstance(selected_features, tf.Tensor):
        K_value = selected_features.shape[-1]
        if K_value is None:  # 如果在图模式下仍然是 None，可能需要从类属性获取 K
            # 假设 K 在类实例化时已知，例如 self.K
            # 这里需要一种方法来获取 K，如果上面的方法失败
            # 为了简单起见，暂时硬编码，但更好的方法是传递 K
            print("警告：无法在图模式下自动推断 K，使用硬编码值 48")
            K_value = 48  # 或者从其他地方获取
    else:  # Eager 模式
        K_value = selected_features.shape[-1]

    target_height = 72
    target_width = 14
    target_channels = 1

    # 1. 映射 K 个特征到初始 CNN 状态
    init_h = target_height // 8  # 9
    init_w = target_width // 2  # 7
    init_channels = 128
    x = Dense(init_h * init_w * init_channels)(selected_features)
    x = LeakyReLU(0.2)(x)
    x = Reshape((init_h, init_w, init_channels))(x)  # (None, 9, 7, 128)

    # 2. 上采样放大
    # (18, 14, 64)
    x = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)

    # (36, 14, 32)
    x = Conv2DTranspose(32, kernel_size=(4, 4), strides=(2, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)

    # (72, 14, 16)
    x = Conv2DTranspose(16, kernel_size=(4, 4), strides=(2, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.1)(x)

    # 3. 输出目标形状
    output_image = Conv2D(target_channels, kernel_size=(3, 3), activation='linear', padding='same')(
        x)  # (None, 72, 14, 1)

    # print("CNN Interpolator Output Shape:", output_image.shape) # 运行时打印
    return output_image
