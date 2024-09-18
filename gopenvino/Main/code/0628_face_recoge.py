import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
from PIL import Image
import os
import serial
import time
import json
import openvino as ov
from tkinter import *
from PIL import Image, ImageTk, ImageFont, ImageDraw
import time
from enum import Enum
from queue import Queue
from model_running import *
from arduino import ArduinoController

# 아두이노 연결 여부를 나타내는 변수 초기화
py_serial= ArduinoController(port='/dev/ttyACM0', baudrate=9600)

is_arduino_connected = py_serial.is_connected()

# 사전학습할 회원 이미지 폴더 경로
member_folders = './Images_face2/Images'
model_face = FaceRecognition(member_folders=member_folders)

model_path_head = "./models/head-pose-estimation-adas-0001/head-pose-estimation-adas-0001.xml"
model_head = AngleEstimation(path = model_path_head)

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
print(model_face.members())
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

# 인식된 사람 변수
previous_name = None
previous_label_name = "NoOne"
current_name = "NoOne"
recognized_time = None
is_changed = True

# 웹캠 비디오 캡처 초기화
video_capture = cv2.VideoCapture(0)
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
    image = frame
    name, boxes, is_member = model_face.recognize(image)

        # face가 None일 때 이름을 "Unknown"으로 설정

    if previous_name != name:
        name_queue.put(name)
        previous_name = name

    # 저장 값이 있고 사람 변경 시
    if is_member: #and is_changed:
        is_changed = False
        print(name)
        height_saved, angle_saved = read_value(name)
        #print(f"name = {name}, {height_saved}, {angle_saved}")

        if not message_queue.empty():#height_saved == -1 or angle_saved == -1 :
            print("messeage received")   
            message = message_queue.get() 

            results = model_head(image, boxes)

            for result in results:
                if len(result) == 0:
                    break
                _, _, y1, y2, angle = result
                y_point = (y2+y1)//2
                y_height = image.shape[0]
                height = (y_point/y_height)*180
                height = int(height)
                angle = int(angle+90)
                value = height*1000 + angle

                print('height :', height, 'angle:',angle)
                angle_str = f"{int(value):06d}"  # 정수를 3자리 문자열로 변환하고 나머지 3자리는 "000"으로 채움, 개행 문자 추가
                print(angle_str)

                write_value(name, height, angle)
                if is_arduino_connected:                        
                    py_serial.send_signal(angle_str)  # 문자열을 바이트로 인코딩하여 전송
                    time.sleep(0.1)
        else :
            height = int(height_saved)
            angle = int(angle_saved)
            value = height*1000 + angle
            angle_str = f"{int(value):06d}" 
            if is_arduino_connected:
                py_serial.send_signal(angle_str)  # 문자열을 바이트로 인코딩하여 전송
                time.sleep(0.1)                                                

        #cv2.putText(frame, f"Height: {height:.2f} degrees", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #cv2.putText(frame, f"Angle: {angle:.2f} degrees", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 시리얼 포트로부터 데이터 읽기        
    if is_arduino_connected:
        data = py_serial.read_response()
        if data is not None:
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
                    print("Invalid input. Expected two integer values separated by a comma.")
            else:
                # 첫 글자가 문자인 경우
                print(data)

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
    remaining_time = 5 - int(elapsed_time)
    remaining_time2 = 3 - int(elapsed_time)

    if not name_queue.empty():
        print("message received label")
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
                    name_label.config(text=f"환영합니다 {current_name}님, {remaining_time}초 후에 자동높이 조절합니다.")
                else: 
                    print("message send")
                    message_queue.put("SEND")                            
                    sState = SystemState.WORKING
            else: # 등록된 사용자, 수정조절모드              
                if remaining_time2 > 0:
                    name_label.config(text=f"환영합니다 {current_name}님, 수동높이조절 모드입니다.")
                else:
                    height, angle = read_value(current_name)
                    value = height*1000 + angle
                    print('height :', height, 'angle:',angle)
                    angle_str = f"{int(value):06d}"  # 정수를 3자리 문자열로 변환하고 나머지 3자리는 "000"으로 채움, 개행 문자 추가
                    print(angle_str)
                    
                    if is_arduino_connected:
                        py_serial.send_signal(angle_str)  # 문자열을 바이트로 인코딩하여 전송
                        time.sleep(0.5)
                        py_serial.send_signal("MAN")
                        time.sleep(0.5)           
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
    py_serial.send_start_signal()

# 프레임 업데이트 시작
update_frame()
update_label()

# Tkinter 창 실행
root.mainloop()

# 웹캠 릴리스
video_capture.release()
cv2.destroyAllWindows()
