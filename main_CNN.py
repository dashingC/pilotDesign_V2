import numpy as np
from utils_CNN import ConcreteAutoencoderFeatureSelector
from utils_CNN import interpolate_model,interpolate_model_cnn
from utils_CNN import load_channel


def FS_SR():
    print("开始执行")

    num_pilots = 48
    SNR = 12

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_channel(num_pilots, SNR)

    #x_train = np.reshape(x_train, (len(x_train), -1))
    #x_test = np.reshape(x_test, (len(x_test), -1))

    print(f"训练集形状: {x_train.shape}, 标签形状: {y_train.shape}")  # 应为 (N, 72, 14, 1)
    print(f"测试集形状: {x_test.shape}, 标签形状: {y_test.shape}")  # 应为 (N, 72, 14, 1)
    print(f"验证集形状: {x_val.shape}, 标签形状: {y_val.shape}")  # 应为 (N, 72, 14, 1)

    num_epochs = 1

    print("\n=== 正在执行特征选择 ===")
    selector = ConcreteAutoencoderFeatureSelector(K=num_pilots, output_function=interpolate_model_cnn,
                                                  num_epochs=num_epochs, learning_rate=0.001)

    # 训练时使用 2D 输入和 2D 标签
    selector.fit(x_train, y_train, x_val, y_val)  # <--- 传入验证集

    # 保存模型：
    model_save_path = f"trained_model_K{num_pilots}_SNR{SNR}_epoch{num_epochs}.h5"
    # 检查模型是否已训练成功 (selector.model 是否存在)
    if hasattr(selector, 'model') and selector.model is not None:
        print(f"\n=== 正在保存训练好的模型到: {model_save_path} ===")
        try:
            # 使用 Keras 的 save 方法保存整个模型
            selector.model.save(model_save_path)
            print(f"   模型已成功保存！")
        except Exception as e:
            print(f"   模型保存失败: {e}")
    else:
        print("\n错误：模型未训练或训练失败，无法保存。")

    selected_indices_flat = selector.get_support(indices=True)

    if selected_indices_flat is not None:
        print(f"\n选中的特征索引 (展平后): {selected_indices_flat}")
        print(f"选中索引数量: {len(selected_indices_flat)}")

        # --- (可选) 转换为 2D 坐标 ---
        selected_indices_2d = []
        if len(x_train.shape) > 2 and x_train.shape[2] > 0:  # 添加检查确保维度存在且大于0
            num_cols = x_train.shape[2]  # 获取宽度 14
            for index in selected_indices_flat:
                row = index // num_cols
                col = index % num_cols
                selected_indices_2d.append((row, col))
            print(f"对应的 2D 坐标 (行, 列): {selected_indices_2d}")
        # -----------------------------

        # --- 检查重复索引 (修改后，添加重复次数统计) ---
        unique_indices, counts = np.unique(selected_indices_flat, return_counts=True)
        # 找出计数大于1的索引和它们的计数
        duplicates_info = {index: count for index, count in zip(unique_indices, counts) if count > 1}

        if duplicates_info:  # 如果字典不为空，说明有重复
            print(f"\n!!! 选中的索引中存在重复值。")
            print("重复详情:")
            for index, count in duplicates_info.items():
                # 打印每个重复索引及其重复次数
                print(f"   - 索引 {index} 重复了 {count} 次")
        else:
            print("\n选中的索引都是唯一的，没有重复值。")
        # --- 修改结束 ---

    else:
        print("\n未能获取选中的特征索引。")

    print("结束")

if __name__ == '__main__':
    FS_SR()
