# import numpy as np
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# import matplotlib.pyplot as plt
# from sklearn.metrics import multilabel_confusion_matrix
# from tensorflow.keras.models import load_model
# from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
#
# actions = [
#     'zero',
#     'one',
#     'two',
#     'three',
#     'four',
#     'five'
# ]
#
# # numpy 배열 합치기
# data = np.concatenate([
#     np.load('dataset/seq_zero_1716012717.npy'),
#     np.load('dataset/seq_one_1716012717.npy'),
#     np.load('dataset/seq_two_1716012717.npy'),
#     np.load('dataset/seq_three_1716012717.npy'),
#     np.load('dataset/seq_four_1716012717.npy'),
#     np.load('dataset/seq_five_1716012717.npy')
# ], axis = 0)
#
# print("data shape : "+ str(data.shape))
# # # data의 마지막 값이 라벨이므로 x_data와 labels로 나누기
# # x_data = data[:, :, :-1]
# # labels = data[:, 0, -1]
# # # One-hot encoding 진행
# # y_data = to_categorical(labels, num_classes=len(actions))
# #
# # x_data = x_data.astype(np.float32)
# # y_data = y_data.astype(np.float32)
# #
# # x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2024)
# #
# # model = Sequential([
# #     LSTM(64, activation='relu', input_shape=x_train.shape[1:3]), # input_shape = [30, 99], 30->윈도우의 크기, 99->랜드마크, visibility, 각도
# #     Dense(32, activation='relu'),
# #     Dense(len(actions), activation='softmax')
# # ])
# #
# # model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['acc'])
# # print(str(model.summary()))
# #
# # from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
# #
# # history = model.fit(
# #     x_train,
# #     y_train,
# #     validation_data=(x_val, y_val),
# #     epochs=200,
# #     callbacks=[
# #         ModelCheckpoint('models/model.keras', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
# #         ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
# #     ]
# # )
#
# fig, loss_ax = plt.subplots(figsize=(16, 10))
# acc_ax = loss_ax.twinx()
#
# loss_ax.plot(history.history['loss'], 'y', label='train loss')
# loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
# loss_ax.set_xlabel('epoch')
# loss_ax.set_ylabel('loss')
# loss_ax.legend(loc='upper left')
#
# acc_ax.plot(history.history['acc'], 'b', label='train acc')
# acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
# acc_ax.set_ylabel('accuracy')
# acc_ax.legend(loc='upper left')
#
# plt.show()
# model = load_model('models/model.keras')
#
# y_pred = model.predict(x_val)
#
# multilabel_confusion_matrix(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1))