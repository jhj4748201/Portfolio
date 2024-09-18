import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
from PIL import Image
import os
import math
import serial
import time
import json
import openvino as ov
from tkinter import *
from PIL import Image, ImageTk, ImageFont, ImageDraw
import time
from enum import Enum
from queue import Queue
import threading 

# 아두이노 연결 여부를 나타내는 변수 초기화
is_arduino_connected = False

# 아두이노 연결 확인
try:
    py_serial = serial.Serial(    
        # Window
        port='COM3',
        # 보드 레이트 (통신 속도)
        baudrate=9600,
    )
    is_arduino_connected = True
except serial.SerialException:
    print("No connection with Arduino")

# MTCNN과 ResNet 모델 초기화
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# 사전학습할 회원 이미지 폴더 경로
#member_folders = ["Images/Jaehyeok", "Images/Jeong", "Images/Jiwon", "Images/Kihoon"]
member_folders = "Images"
# 모든 회원의 인코딩과 이름 저장 리스트
known_encodings = []
known_names = []

# 사람 정보를 저장할 딕셔너리
#person_data = {}

# 사람 정보 파일 경로
person_data_file = "person_data.json"

# Tkinter 창 생성
root = Tk()
root.title("Moduui Monitor")

# 메시지 큐 생성
message_queue = Queue()
name_queue = Queue()

# 비디오 프레임을 표시할 라벨 생성
video_label = Label(root)
video_label.pack()

# 폰트 설정
font = ("Arial", 20)
name = "Unknown"

# 사용자 이름을 표시할 라벨 생성
name_label = Label(root, text=f"사용자: Unknown", font=font, anchor='w')
name_label.pack(fill='x', padx=10, pady=10)

# 함수 선언
def read_value(key):
    try:
        with open(person_data_file, "r") as file:
            data = json.load(file)
            if key in data:
                value1 = data[key]["value1"]
                value2 = data[key]["value2"]
                #print(f"[{key}] value1: {value1}, value2: {value2}")
                return value1, value2
            else:
                #print(f"'{key}' doesn't exist.")
                return None, None
    except FileNotFoundError:
        print("JSON file doesn't exist.")
        return None, None
    except json.JSONDecodeError:
        print("json.JSONDecodeError")
        return None, None
    
def write_value(key, value1, value2):
    try:
        if not os.path.exists(person_data_file):
            data = {}
        else:
            with open(person_data_file, "r") as file:
                data = json.load(file)

        data[key] = {
            "value1": value1,
            "value2": value2
        }

        with open(person_data_file, "w") as file:
            json.dump(data, file, indent=2)
            print(f"'{key}''s value is saved successfully.")
    except IOError:
        print("JSON file write Fail")

# 파일에서 사람 정보 읽어오기
# if os.path.exists(person_data_file):
#     with open(person_data_file, "r") as file:
#         person_data = json.load(file)

# 각 회원 폴더의 이미지를 로드하여 인코딩
members = os.listdir(member_folders)
for member in members:
    print(member)
    member_path = os.path.join(member_folders, member)
    for image_name in os.listdir(member_path):
        image_path = os.path.join(member_path, image_name)
        image = Image.open(image_path)
        face = mtcnn(image)
        if face is not None:
            encoding = resnet(face.unsqueeze(0)).detach().cpu()
            known_encodings.append(encoding)
            known_names.append(member)          

# 얼굴 각도 추정 모델
core = ov.Core()
device = 'CPU'
model_path_head = "./models/head-pose-estimation-adas-0001/head-pose-estimation-adas-0001.xml"
model_head = core.read_model(model=model_path_head)
compiled_model_head = core.compile_model(model=model_head, device_name=device)

input_layer_head = compiled_model_head.input(0)
output_layer_pitch = compiled_model_head.output(1)
height_head, width_head = list(input_layer_head.shape)[2:4]

# 인식된 사람 변수
previous_name = None
previous_label_name = "NoOne"
current_name = "NoOne"
recognized_time = None
is_changed = True

# 웹캠 비디오 캡처 초기화
video_capture = cv2.VideoCapture(1)

prev_angle = 90
height = 0
angle = 0
name = "Unknown"

# system state
class SystemState(Enum):
    WAIT = 1
    WORKING = 2    
sState = SystemState.WAIT

def update_frame() :
    global previous_name, prev_angle, height, angle, name, is_changed

    #while True:
    # 비디오 프레임 읽기
    ret, frame = video_capture.read()
    if not ret:
        return

    # 현재 프레임에서 얼굴 찾기
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    face = mtcnn(image)

    if face is not None:
        encoding = resnet(face.unsqueeze(0)).detach().cpu()

        # 얼굴 인코딩 비교
        distances = [torch.dist(encoding, known_encoding).item() for known_encoding in known_encodings]
        min_distance = min(distances)
        min_index = distances.index(min_distance)
        
        name = "Unknown"
        is_member = False

        if min_distance < 1.0:  # 임계값 설정 (필요에 따라 조정 가능)
            name = known_names[min_index]
            is_member = True
            if previous_name != name:
                name_queue.put(name)
                previous_name = name
        else:
            is_member = False
            if previous_name != name:
                name_queue.put(name)
                previous_name = name

        # 얼굴 주위에 사각형 그리기 및 이름 표시
        boxes, _ = mtcnn.detect(image)
        # if boxes is not None:
        #     for box in boxes:
        #         cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        #         cv2.putText(frame, name, (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        # 저장 값이 있고 사람 변경 시
        if is_member: #and is_changed:
            is_changed = False
            height_saved, angle_saved = read_value(name)
            #print(f"name = {name}, {height_saved}, {angle_saved}")

            if height_saved == -1 or angle_saved == -1 :
            #if not message_queue.empty():#height_saved == -1 or angle_saved == -1 :              
                #message = message_queue.get() 

                for box in boxes:
                    y_point = (box[1]+box[3])//2
                    y_height = frame.shape[0]
                    height = (y_point/y_height)*180
                    if height < 0:
                        height = 0
                    elif height > 180:
                        height = 180
                    if len(box) == 0:
                        break

                    # 얼굴 좌표가 유효한지 확인
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                        if x1<0:
                            x1=0
                        if y1 <0:
                            y1 = 0
                        if x2 > frame.shape[1]:
                            x2 = frame.shape[1]
                        if y2 > frame.shape[0]:
                            y2 = frame.shape[0]
                        # continue

                    # 얼굴 이미지 추출
                    input_head = frame[y1:y2, x1:x2]
                    if input_head.size == 0:
                        continue

                    input_head = cv2.resize(src=input_head, dsize=(width_head, height_head), interpolation=cv2.INTER_AREA)
                    input_head = input_head.transpose((2, 0, 1))
                    input_head = input_head[np.newaxis, ...]

                    # 각도 추정
                    results_pitch = compiled_model_head([input_head])[output_layer_pitch]
                    angle = np.squeeze(results_pitch)
                    height = int(height)
                    angle = int(angle+90)
                    value = height*1000 + angle
                    print('2_height :', height, 'angle:',angle)
                    angle_str = f"{int(value):06d}\n"  # 정수를 3자리 문자열로 변환하고 나머지 3자리는 "000"으로 채움, 개행 문자 추가
                    print(angle_str)
                      
                    #write_value(name, height, angle)
                    if is_arduino_connected:                        
                        py_serial.write(angle_str.encode())  # 문자열을 바이트로 인코딩하여 전송
                        time.sleep(0.5)

            else :
                height = int(height_saved)
                angle = int(angle_saved)
                value = height*1000 + angle
                angle_str = f"{int(value):06d}\n" 
                if is_arduino_connected:
                    py_serial.write(angle_str.encode())  # 문자열을 바이트로 인코딩하여 전송
                    time.sleep(0.1)
                    py_serial.write(b"MAN\n")
                    time.sleep(0.1)                                              

            #cv2.putText(frame, f"Height: {height:.2f} degrees", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #cv2.putText(frame, f"Angle: {angle:.2f} degrees", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 시리얼 포트로부터 데이터 읽기        
        if is_arduino_connected and py_serial.in_waiting > 0:
            data = py_serial.readline().decode('utf-8').rstrip()
            if len(data) > 0:
                first_char = data[0]
                
                if first_char.isdigit():
                    # 첫 글자가 숫자인 경우 처리
                    values = data.split(',')
                    try:
                        value1 = int(values[0])
                        value2 = int(values[1])
                        #print(f"Received values: {value1}, {value2}")
                        write_value(name, value1, value2)
                        # 필요한 처리 작업 수행
                    except (ValueError, IndexError):
                        print(data)
                else:
                    # 첫 글자가 문자인 경우
                    print(data)
    else:
        # face가 None일 때 이름을 "Unknown"으로 설정
        name = "NoOne"
        is_member = False  
        if previous_name != name:
            name_queue.put(name)
            previous_name = name


    # 프레임을 Tkinter 라벨에 표시
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_image = Image.fromarray(frame_rgb)
    frame_tk = ImageTk.PhotoImage(frame_image)
    video_label.config(image=frame_tk)
    video_label.image = frame_tk

    # 다음 프레임으로 업데이트
    video_label.after(10, update_frame)
# end update_frame() ########################

def update_label():
    global start_time, sState, height, angle, previous_label_name, current_name

    height_saved, angle_saved = read_value(name)
    elapsed_time = time.time() - start_time
    remaining_time = 10 - int(elapsed_time)
    remaining_time2 = 3 - int(elapsed_time)

    if not name_queue.empty():
        #print("message received label")
        current_name = name_queue.get() 
        if previous_label_name != current_name:
            previous_label_name = current_name
            start_time = time.time()
            sState = SystemState.WAIT

    if current_name == "NoOne":
        name_label.config(text=f"")
        start_time = time.time()
        sState = SystemState.WAIT
    elif sState == SystemState.WAIT:
        if current_name == "Undefined":
            print("Name Undefined")
        elif current_name != "Unknown":
            if height_saved == -1 or angle_saved == -1 :
                if remaining_time > 0:  
                    name_label.config(text=f"환영합니다 {current_name}님, {remaining_time}초 후에 수동높이조절 모드입니다.")
                else: 
                    print("message send")
                    write_value(current_name, int(height), int(angle))
                    #message_queue.put("SEND")                            
                    sState = SystemState.WORKING
            else: # 등록된 사용자, 수동조절모드              
                if remaining_time2 > 0:
                    name_label.config(text=f"환영합니다 {current_name}님, 수동높이조절 모드입니다.")
                else:
                    height, angle = read_value(current_name)
                    value = height*1000 + angle
                    print('1_height :', height, 'angle:',angle)
                    angle_str = f"{int(value):06d}\n"  # 정수를 3자리 문자열로 변환하고 나머지 3자리는 "000"으로 채움, 개행 문자 추가
                    print(angle_str)
                    
                    if is_arduino_connected:
                        py_serial.write(angle_str.encode())  # 문자열을 바이트로 인코딩하여 전송
                        time.sleep(0.1)
                        py_serial.write(b"MAN\n")
                        time.sleep(0.1)           
                    name_label.config(text=f"{current_name}님, 수동높이조절 모드입니다.") 
                    sState = SystemState.WORKING                                   
        else:              
            sState = SystemState.WORKING    
    elif sState == SystemState.WORKING:  
        if current_name == "Unknown":  
            name_label.config(text=f"등록된 사용자가 아닙니다.")   
        else :       
            name_label.config(text=f"{current_name}님, 수동높이조절 모드입니다.")                                 
    elif current_name == "Unknown":         
        name_label.config(text=f"등록된 사용자가 아닙니다.")                   
    else:
        name_label.config(text=f"")

    root.after(10, lambda: update_label())

# 시작 시간 기록
start_time = time.time()

if is_arduino_connected:   
    py_serial.write(b"START\n")

# 프레임 업데이트 시작
update_frame()
update_label()

# Tkinter 창 실행
root.mainloop()

# 웹캠 릴리스
video_capture.release()
cv2.destroyAllWindows()
