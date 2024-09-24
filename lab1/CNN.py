import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

def rotate_image(image, angle):
    """ 旋转图像函数 """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    return rotated

def augment_data(X):
    """ 扩充数据集，进行旋转 """
    augmented_data = []
    for image in X:
        # 原始图像
        augmented_data.append(image)
        # 旋转 45 度
        augmented_data.append(rotate_image(image, 45))
        # 旋转 -45 度
        augmented_data.append(rotate_image(image, -45))
    return np.array(augmented_data)

def Img_to_Mat(fileName):
    f = open(fileName)
    ss = f.readlines()
    l = len(ss)
    f.close()
    returnMat = np.zeros((l,256))
    returnClassVector = np.zeros((l,1))
    for i in range(l):
        s1 = ss[i].split()
        for j in range(256):
            returnMat[i][j] = float(s1[j])
        binary_label = s1[256:266]
        class_label = int(''.join(binary_label), 2)
        returnClassVector[i][0] = class_label
    returnClassVector = returnClassVector.ravel()
    return returnMat, returnClassVector

# 加载原始数据
X, y = Img_to_Mat('semeion.data')
label_mapping = {i: j for j, i in enumerate([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])}
y = np.array([label_mapping[label] for label in y])
# print("Unique classes in original y:", np.unique(y))

# 将数据从 (num_samples, 256) 转换为 (num_samples, 16, 16) 图像格式
X_images = X.reshape(-1, 16, 16)

# 进行数据增强
X_augmented_images = augment_data(X_images)

# 扩展标签
y_augmented = np.repeat(y, 3)  # 因为每个图像旋转了三次
# print("Unique classes in augmented y:", np.unique(y_augmented))

# 准备数据
X_augmented = X_augmented_images.reshape(-1, 16, 16, 1)  # 根据图像尺寸调整
y_categorical = to_categorical(y_augmented, num_classes=10)  # 指定类别数

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_categorical, test_size=0.2, random_state=42)

# 构建CNN模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(16, 16, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.8f}")
