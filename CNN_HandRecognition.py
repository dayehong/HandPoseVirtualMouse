#코드 작성자: 홍다예 
import tensorflow as tf # 텐서플로우  라이브러리 임포트
import cv2 # opencv 패키지 임포트
import os # os 모듈 임포트
import numpy as np # 행렬 연산에 쓰이는 넘파이 모듈 임포트

from tensorflow.python.framework import graph_util # 텐서 그래프를 조작하는데 도움되는 텐서플로우 모듈 임포트
from tensorflow.python.platform import gfile # 파이썬 파일 입출력 객체에 가까운 api 제공

global Y_train_data # y 학습 데이터 전역 변수
global X_train_data # x 학습 데이터 전역 변수

global Y_test_data # y 테스트 데이터 전역 변수
global X_test_data # x 테스트 데이터 전역 변수

X_test_data = np.array([]) # x 테스트 데이터 배열로 변환
X_train_data = np.array([]) # x 학습 데이터 배열로 변환

Y_test_data = np.array([]) # y 테스트 데이터 배열로 변환
Y_train_data = np.array([]) # y 학습 데이터 배열로 변환

image_size_width = 32 # 이미지 가로 사이즈 정의
image_size_height = 32 # 이미지 세로 사이즈 정의

learning_rate = 0.001 # 학습 속도

# training epoch 수
training_epochs = 100 # 전체 데이터를 이용하여 100바퀴 돌며 학습

classCnt = 5 # 학습 할 데이터의 class 종류 수

# model 만들기 

# X 와 Y 값은 입력으로 받아야 하기 때문에 placeholder 로 선언
X = tf.placeholder(tf.float32, [None, image_size_width, image_size_height, 1], name='data') # X = 입력 값으로 32X32 크기
Y = tf.placeholder(tf.float32, [None, classCnt]) # Y = 결과

# convolution layer 1
conv1 = tf.layers.conv2d(X, 32, [3, 3], padding="same", activation=tf.nn.relu) # kernel 수 32 개, kernel 크기 3 X 3, 선형 활성화 함수:relu
pool1 = tf.layers.max_pooling2d(conv1, [2, 2], strides=2, padding="same") # maxpooling 수행

# convolution layer 2
conv2 = tf.layers.conv2d(pool1, 64, [3, 3],
                         padding="same", activation=tf.nn.relu)# kernel 수 64 개, kernel 크기 3 X 3, 선형 활성화 함수:relu
pool2 = tf.layers.max_pooling2d(conv2, [2, 2], strides=2, padding="same")# maxpooling 수행

flat3 = tf.contrib.layers.flatten(pool2) # fully connected layer를 수행하기 위해 2차원 데이터를 1차원으로 변경

dense3 = tf.layers.dense(flat3, 256, activation=tf.nn.relu) # fully connected 수행, data 수 256, 선형 활성화 함수:relu

logits = tf.layers.dense(dense3, 5, activation=None) # fully connected 수행, 결과는 logit 으로 출력
final_tensor = tf.nn.softmax(logits, name='prob') # logits 을 softmax 연산을 통해 확률 값으로 출력
 

cost =
tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,
                                                          logits=logits))
# 입력된 ground truth 와 계산 결과인 logits 을 비교하여 cost 를 계산

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost) # cost를 낮게 하기 위해 adam optimizer 를 사용


# CNN 에 입력 이미지를 만들기
def preprocessing(original_image): #이미지 전처리 과정; 오리지널 이미지를 입력받아서
    img_y = cv2.resize(original_image, (image_size_width, image_size_height)) # resizing; 이미지 크기 조정
    img_y = img_y / 255.0 # data normalization (값을 0과 1 사이의 값으로 변환)

    return img_y #img_y 리턴


def test_image_store(folder_path, class_num, skip_cnt): # 테스트 데이터를 변수에 저장하는 함수
    global X_test_data #전역변수 테스트 x 데이터
    global Y_test_data  #전역변수 테스트 y 데이터

    filenames = os.listdir(folder_path) # 폴더 내 모든 파일 읽기

    readImgCnt = 0 # 이미지 읽어온 횟수
    for filename in filenames: # filenames에 모든 파일이 들어있는데 한개씩 가져옴
        img_filename = os.path.join(folder_path, filename) # 이미지 파일 이름 해당 위치 경로와 파일 이름을 연결함 'folder_path\filename' 
        ext = os.path.splitext(img_filename)[-1] # 확장자만 따로 분류(리스트로 나타냄)
        if ext == '.BMP' or ext == '.bmp' or ext == '.jpg': # 폴더 내 BMP, bmp, jpg 이면
            print(img_filename) # 파일 이름 읽어서 출력

            if os.path.isfile(img_filename): #파일 이름이 존재하면

                readImgCnt += 1 # 횟수 1 증가
                if (readImgCnt % skip_cnt) != 0: #읽어온 횟수가 끝나지 않았으면
                    continue  # 계속

                imgY = cv2.imread(img_filename, 0)  # 원본 이미지를 로드

                
                buff_y = preprocessing(imgY) # 이미지를 CNN model 에 넣을 수 있는 사이즈로 변경
                
                buff_y = buff_y.reshape(-1, image_size_width, image_size_height, 1, order='c') # shape (1, 32, 32, 1) 로 변경

                if X_test_data.__len__() == 0: # X 테스트 데이터 길이가 0이면
                    X_test_data = buff_y # buff_y를 X 테스트 데이터에 넣음
                    Y_test_data = np.concatenate([[[class_num]]]) # Y테스트 데이터에 해당 배열을 붙여 넣음
                else: # X 테스트 데이터 길이가 0이 아니면
                    X_test_data = np.concatenate([X_test_data, buff_y]) # X 테스트 데이터에 해당 배열을 붙여 넣음
                    Y_test_data = np.concatenate([Y_test_data, [[class_num]]]) # Y테스트 데이터에 해당 배열을 붙여 넣음



def train_image_store(folder_path, class_num, skip_cnt): # 테스트 데이터를 변수에 저장하는 함수
    global X_train_data # X 학습 전역변수
    global Y_train_data # Y 학습 전역변수

    filenames = os.listdir(folder_path) # 폴더 내 모든 파일을 읽어서 filenames에 저장

    readImgCnt = 0 #이미지 파일 읽어온 횟수
    for filename in filenames: #filenames에 저장된 모든 파일들을 하나씩 읽어옴
        img_filename = os.path.join(folder_path, filename) #이미지 파일 이름 해당 위치 경로와 파일 이름을 연결함 'folder_path\filename'
        ext = os.path.splitext(img_filename)[-1]
        if ext == '.BMP' or ext == '.bmp' or ext == '.jpg': # 폴더 내 모든 BMP, bmp, jpg 있으면
            print(img_filename) # 파일 이름 읽어오기

            if os.path.isfile(img_filename): # 경로에 해당 파일이름이 있으면

                readImgCnt += 1 # 이미지를 읽어온 횟수를 1 증가시킴
                if (readImgCnt % skip_cnt) != 0: # 시간이 너무 오래걸려서 skip_cnt를 나눴을 때 0이 아니면 
                    continue # 스킵

                imgY = cv2.imread(img_filename, 0) # 원본 이미지를 로드

                buff_y = preprocessing(imgY) # 이미지를 CNN model 에 넣을 수 있는 사이즈로 변경하여 buff_y에 저장
                buff_y = buff_y.reshape(-1, image_size_width, image_size_height, 1, order='c') # shape (1, 32, 32, 1) 로 변경

                if X_train_data.__len__() == 0: # 학습 데이터의 길이가 0이면
                    X_train_data = buff_y # buff_y를 학습데이터에 저장
                    Y_train_data = np.concatenate([[[class_num]]]) #배열 합치기 
                else:
                    X_train_data = np.concatenate([X_train_data, buff_y]) #배열 합치기
                    Y_train_data = np.concatenate([Y_train_data, [[class_num]]]) #배열 합치기

if __name__ == "__main__": #main함수

    skip_cnt = 11 # 넘긴 횟수

    # training 을 수행 할 폴더를 선택
    train_image_store(r'F:\daye\new_all\type_1', 0, skip_cnt)   # type 0
    train_image_store(r'F:\daye\new_all\type_2', 1, skip_cnt)   # type 1
    train_image_store(r'F:\daye\new_all\type_3', 2, skip_cnt)   # type 2
    train_image_store(r'F:\daye\new_all\type_4', 3, skip_cnt)   # type 3
    train_image_store(r'F:\daye\new_all\type_5', 4, skip_cnt)   # type 4

    # testing 을 수행 할 폴더를 선택
    test_image_store(r'F:\daye\new_all\type_1', 0, skip_cnt)    # type 0
    test_image_store(r'F:\daye\new_all\type_2', 1, skip_cnt)    # type 1
    test_image_store(r'F:\daye\new_all\type_3', 2, skip_cnt)    # type 2
    test_image_store(r'F:\daye\new_all\type_4', 3, skip_cnt)    # type 3
    test_image_store(r'F:\daye\new_all\type_5', 4, skip_cnt)    # type 4

    with tf.Session() as sess: # training 을 하기 위해 session 열기
        sess.run(tf.global_variables_initializer()) # session에 사용할 모든 변수를 초기화

        print('Start learning!') #학습 시작 메세지 출력
        for epoch in range(training_epochs): # training_epochs 만큼 loop 돈다
            
            # 최종 결과를 one-hot encoding 을 통해 동작의 index 를 가져오는 수식을 정의
            y_train_one_hot = sess.run(tf.squeeze(tf.one_hot(Y_train_data, 5), axis=1))

            
            # 입력 데이터는 이전에 저장했던 X_train_data, y_train_one_hot
            _, cost_val = sess.run([optimizer, cost],
                                   feed_dict={X: X_train_data, Y: y_train_one_hot})# optimizer 함수를 호출하여 traing 1 epoch 을 수행

            print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost = ', '{:.4f}'.format(cost_val))# 정보 출력

            if epoch % 3 == 0: # epoch % 3 이 0 일 때 
                y_test_one_hot = sess.run(tf.squeeze(tf.one_hot(Y_test_data, 5), axis=1)) #테스트 수행

                correct_prediction = tf.equal(tf.argmax(final_tensor, 1),
                                              tf.argmax(y_test_one_hot, 1)) # 정답과 같은지 다른지에 대한 결과를 저장하는 수식을 정의
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 정답의 평균을 구하여 최종 정확도를 계산
                acc, cost_val = sess.run([accuracy, cost], feed_dict={
                    X: X_test_data, Y: y_test_one_hot}) # X_test_data, y_test_one_hot 를 입력하여 테스트 결과를 출력

                print('accuracy:', '%04f' % (acc * 100.0), 'Avg. cost = ','{:.4f}'.format(cost_val)) # 최종 결과를 출력

        print('Learning finished!') #메세지 출력

        # C++ 에서 호출하여 사용할 수 있도록 데이터의 graph 저장
        # 학습 데이터는 .pb 파일로 저장
        # C++ opencv 에서 해당 데이터를 로드하여 사용
        # Freeze variables and save pb file
        output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['prob']) #그래프 고정
        with gfile.FastGFile('./hand_recognition_cnn.pb', 'wb') as f: #학습 데이터
            f.write(output_graph_def.SerializeToString()) #파일 쓰기

        print('Save done!') #저장 완료 메세지 출력

