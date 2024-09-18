import cv2
import mediapipe as mp
import numpy as np
import time, os
import pygame
from sklearn.metrics import multilabel_confusion_matrix
from tensorflow.keras.models import load_model
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ultralytics import YOLO
# YOLO 블러 처리 모드를 위한 모델 로드
model_yolo = YOLO('./models/yolov8n-seg.pt')
####################################################
######### 부가 함수 정의및 선언부 #########
####################################################
# 파일 경로를 입력받고 해당 파일 경로의 음악을 재생시키는 함수
def play_music(file_path): # 노래 함수
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

# 왼손 또는 오른손을 판별하는 함수, 중앙기준 오른쪽이 오른손 왼쪽이 왼손
def determine_hand_side(hand_landmarks):
    center_x = np.mean([lm.x for lm in hand_landmarks.landmark])
    if center_x < 0.5:
        return 'Left'
    else:
        return 'Right'
# yolo8로 구현한 블러처리 함수
def yolo8_blur(frame):
    # YOLO를 통한 객체 감지
    # Confidence 임계값
    threshold = 0.8
    # 영상에 대한 모델 결과
    results = model_yolo.predict(frame)
    # 사람 객체 영역 초기화
    person_region = []
    # 'person' 클래스 주변에 사각형 그리고 배경 마스크 생성
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            if (result.names[int(cls)] == 'person') and (result.boxes.conf[int(cls)] >= threshold):
                x1, y1, x2, y2 = map(int, box)
                # 이미지 경계를 벗어나지 않도록 좌표를 조정
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                person_frame_Data = result.masks.xy.pop()
                person_frame_Data = person_frame_Data.astype(np.int32)
                person_frame_Data = person_frame_Data.reshape(-1, 1, 2)
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                person_region.append([frame[y1:y2, x1:x2], x1, y1, x2, y2, distance, person_frame_Data])
            else:
                continue
    # 배경 블러 처리
    img = frame
    max_dis = 0
    max_index = -1
    # 가장 크기가 큰 사람 객체만을 포착
    # 배경 마스크 초기화
    background_mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
    blurred_background = cv2.GaussianBlur(frame, (99, 99), 0)
    # 가장 크기가 큰 사람 객체만을 포착
    if person_region != []:  # person_region이 비어 있는지 확인
        for i, person in enumerate(person_region):
            dis = person[5]
            if dis > max_dis:
                max_dis = dis
                max_index = i
        print("max distance value : " + str(max_dis))
        x1, y1, x2, y2 = person_region[max_index][1:5]  # 좌표 가져오기
        person_frame_Data = person_region[max_index][6]  # segmentation 외각 좌표
        # 객체 영역 설정
        cv2.fillPoly(background_mask, [person_frame_Data], (255, 255, 255))
        # 배경 블러 처리
        # 배경과 사람 객체 합치기
        result_frame = cv2.bitwise_or(blurred_background, frame, mask=background_mask)
        # 배경 마스크를 이용하여 배경 블러 처리된 프레임에서 배경만 추출
        foreground = cv2.bitwise_and(frame, frame, mask=background_mask)
        background = cv2.bitwise_and(blurred_background, blurred_background, mask=~background_mask)
        # 배경과 사람 객체 합치기
        result_frame = cv2.add(background, foreground)
        frame = result_frame
        # 클래스 표시
        label = result.names[int(cls)]
        # 박스위 라벨 텍스트 추가
        # cv2.putText(frame, 'person', (person_region[max_index][1], person_region[max_index][2] - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
    else:
        frame = blurred_background
        return frame
####################################################
######### 고정 변수 선언 및 초기화부 #########
####################################################
On = True
static_img = None
save_vol = None
selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation( model_selection=1)
model = load_model('./models/model2.keras')
actions = ['paper','one','two','small','big','gun','OK','Paper2','rock'] # 동작 이름 짓기
seq_length = 30  # 동작 인식 시간
# MediaPipe hands model
mp_hands = mp.solutions.hands  # mediapipe 함수 설정 손 인식 하고 점이랑 선 찍는 거 도와주는 함수
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# 스피커 디바이스와 인터페이스 설정
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# 카메라 화면 출력
cap = cv2.VideoCapture(0)
# 얼굴 인식을 위한 mediapipe 객체 할당
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
# 손 제스처 인식 관련 변수 선언
left_seq = []
right_seq = []
left_action_seq = []
right_action_seq = []
save_left_seq = None
save_right_seq = None
last_action_time = 0
hand_delay = 3
face_filter =False
stop_mode = False
save_frame =None
# 각 기능 결과부 플래그 객체들
hand_blur = False # 블러 처리 모드
hand_blur_yolo = False # 블러 처리 모드(YOLO8)
hand_song = False # 음악 모드
hand_compose = False # 이미지 합성 모드
hand_On_ver1 = False # 영상 On 모드 1
hand_On_ver2 = False # 영상 On 모드 2
hand_Off_ver1 = False # 영상 Off 모드 1
hand_Off_ver2 = False # 영상 Off 모드 2
hand_No_Sound = False # 음소거 모드
hand_down_sound = False # 소리 감소 모드
hand_up_sound = False # 소리 증가 모드
hand_face_filter_mode = False # 영상 필터 합성 모드
hand_stop = False # 영상 필터 합성 모드
# 음소거 이미지 읽어오기
sound_img_vol0 = cv2.imread("./image/sound_img_vol0.jpg")  # 음량 이미지 읽기
sound_img_vol1 = cv2.imread("./image/sound_img_vol1.jpg")
sound_img_vol2 = cv2.imread("./image/sound_img_vol2.jpg")
sound_img_vol3 = cv2.imread("./image/sound_img_vol3.jpg")
# tts경로 읽어오기
tts_path1 = "./voice/대체화면사용.wav"  # 음악 파일의 경로를 지정해주세요
tts_path2 = "./voice/대체화면꺼짐.wav"  # 음악 파일의 경로를 지정해주세요
tts_path3 = "./voice/정지사용.wav"  # 음악 파일의 경로를 지정해주세요
tts_path4 = "./voice/정지꺼짐.wav"  # 음악 파일의 경로를 지정해주세요
tts_path7 = "./voice/필터사용.wav"  # 음악 파일의 경로를 지정해주세요
tts_path8 = "./voice/필터꺼짐.wav"  # 음악 파일의 경로를 지정해주세요
tts_path9 = "./voice/블러사용.wav"  # 음악 파일의 경로를 지정해주세요
tts_path10 = "./voice/블러꺼짐.wav"  # 음악 파일의 경로를 지정해주세요
sound_img = sound_img_vol3  # 초기 음량 이미지 설정
# 반야심경 모음
####################################################
######### 실제 영상 출력부 #########
####################################################
while cap.isOpened():
    current_time = time.time()

    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    detected_left = False
    detected_right = False

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            hand_side = determine_hand_side(res)
            if hand_side == 'Left':
                detected_left = True
                hand_seq = left_seq
                action_seq = left_action_seq
            else:
                detected_right = True
                hand_seq = right_seq
                action_seq = right_action_seq

            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
            angle = np.degrees(angle)

            d = np.concatenate([joint.flatten(), angle])
            hand_seq.append(d)

            # 손 인식 그리기
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(hand_seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(hand_seq[-seq_length:], dtype=np.float32), axis=0)
            y_pred = model.predict(input_data).squeeze()
            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            # 예측된 동작의 신뢰도가 0.9 이상인 경우만 동작 인식
            if conf >= 0.95 and i_pred < len(actions):
                action = actions[i_pred]
                action_seq.append(action)
            # else:
            #     action_seq.append('None')
            if len(action_seq) >= 3:
                if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    if hand_side == 'Left':
                        save_left_seq = action_seq[-1]
                    else:
                        save_right_seq = action_seq[-1]

                cv2.putText(img, f'{action_seq[-1].upper()}',
                            org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    # 손이 감지되지 않은 경우 상태 초기화
    if not detected_left:
        save_left_seq = None
    if not detected_right:
        save_right_seq = None
####################################################
######### 제스처 기능 사용 토글부 #########
####################################################
    if current_time - last_action_time >= hand_delay:
        ### 양손 모드 플래그 토글 부분
        if detected_left and detected_right:
            ###양손모드 코드
            last_action_time = current_time
            # ## 1 - 1 기능 블러 모드(YOLO8) 설정 => zero one
            # if (save_left_seq == 'zero' and save_right_seq == 'one') or (save_left_seq == 'one' and save_right_seq == 'zero'):
            #     hand_blur_yolo = not hand_blur_yolo
            #     print("hand_blur_yolo")
            ## 2 기능 대체 화면 모드 설정 => zero two
            if (save_left_seq == 'two' and save_right_seq == 'two'):
                hand_compose = not hand_compose
                print("hand_compose")
                if(hand_compose==True):
                    play_music(tts_path1)
                else:
                    play_music(tts_path2)

            ## 2 - 1 기능 정지 화면 모드 설정 => zero three
            if (save_left_seq == 'paper' and save_right_seq == 'paper'):
                hand_stop = not hand_stop
                save_frame = img
                print("hand_stop")
                if(hand_stop==True):
                    play_music(tts_path3)
                else:
                    play_music(tts_path4)
            ## 추가 기능 노래 모드 설정 => zero four
            if (save_left_seq == 'rock' and save_right_seq == 'OK') or (save_left_seq == 'OK' and save_right_seq == 'rock'):
                hand_song = not hand_song
                print("hand_song")
            ## 3 기능 얼굴 필터 모드 설정 => zero five
            if (save_left_seq == 'OK' and save_right_seq == 'OK'):
                hand_face_filter_mode = not hand_face_filter_mode
                print("hand_face_filter_mode")
                if(hand_face_filter_mode==True):
                    play_music(tts_path7)
                else:
                    play_music(tts_path8)
            ## 4 - 1기능 화면 On 모드 설정 => one one
            if (save_left_seq == 'rock' and save_right_seq == 'one') or (save_left_seq == 'one' and save_right_seq == 'rock'):
                hand_On_ver2 = True
                hand_Off_ver2 = False
                hand_No_Sound = False
                print("hand_On_ver2")
            ## 4 - 2 기능 화면 Off 모드 설정 => one two
            if save_left_seq == 'rock' and save_right_seq == 'rock':
                hand_Off_ver2 = True
                hand_On_ver2 = False
                hand_No_Sound = True
                print("hand_Off_ver2")
            ## 5 - 2기능 소리 감소 모드 설정 => one four
            # if (save_left_seq == 'small' and save_right_seq == 'one') or (save_left_seq == 'one' and save_right_seq == 'small'):
            #     hand_down_sound = True
            #     print("hand_down_sound")
            ## 5 - 3 기능소리 증가 모드 설정 => one five
            # if (save_left_seq == 'big' and save_right_seq == 'one') or (save_left_seq == 'one' and save_right_seq == 'big'):
            #     hand_up_sound = True
            #     print("hand_up_sound")
        elif detected_left or detected_right:
            ###한손모드 코드
            last_action_time = current_time
            ## 1 기능 블러 모드 설정 => zero zero
            if save_left_seq == 'gun' or save_right_seq == 'gun':
                hand_blur = not hand_blur
                print("hand_blur")
                if(hand_blur==True):
                    play_music(tts_path9)
                else:
                    play_music(tts_path10)
            ## 5 - 1기능 음소거 모드 설정 => one three(양쪽지원)
            if save_left_seq == 'Paper2' or save_right_seq == 'Paper2':
                hand_No_Sound = not hand_No_Sound
                if(hand_No_Sound==True):
                    save_vol = volume.GetMasterVolumeLevelScalar()
                else:
                    volume.SetMasterVolumeLevelScalar(float(save_vol), None)
                print("hand_No_Sound")

    if (save_left_seq == 'small' and save_right_seq == 'one') or (save_left_seq == 'one' and save_right_seq == 'small'):
        hand_down_sound = True
        print("hand_down_sound")
    if (save_left_seq == 'big' and save_right_seq == 'one') or (save_left_seq == 'one' and save_right_seq == 'big'):
        hand_up_sound = True
        print("hand_up_sound")
####################################################
######### 제스처 기능 구현부 #########
####################################################
    #### 기능 4-1 영상 출력 On (창 생성 On)
    if hand_On_ver1:
        ###양손모드 코드
        On = True
    #### 기능 4-1 영상 출력 Off (창 생성 Off)
    if hand_Off_ver1:
        ###양손모드 코드
        On = False
        cv2.destroyAllWindows()
    if(On):
        #### 기능 4-2 영상 출력 Off (검은 화면 출력 On)
        if hand_Off_ver2:
            static_img = np.zeros_like(img)  # 검은색 이미지 생성
            print("test1")
        # 이하 코드는 그대로 유지
        #### 기능 4-2 영상 출력 On (검은 화면 출력 Off)
        if hand_On_ver2:
            static_img = None
        #### 기능 1 : 블러 처리
        if hand_blur:   # 배경 블러 처리
            rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # MediaPipe로 사람 인식
            results = selfie_segmentation.process(rgb_frame)
            # 사람 마스크 생성
            condition = results.segmentation_mask > 0.1
            # 배경 블러 처리
            blurred_frame = cv2.GaussianBlur(img, (55, 55), 0)
            # 마스크를 이용하여 사람 영역은 원본을 유지하고, 배경만 블러 처리
            output_frame = img.copy()
            output_frame[~condition] = blurred_frame[~condition]
            # 윤곽선 찾기
            contours, _ = cv2.findContours(results.segmentation_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            # 가장 큰 윤곽선 찾기
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                # 가장 큰 윤곽선을 따라서 마스크 생성
                mask = np.zeros_like(results.segmentation_mask)
                cv2.drawContours(mask, [max_contour], -1, (255), thickness=cv2.FILLED)
                # 가장 큰 윤곽선에 대해서만 블러 처리 해제
                img = np.where(mask[..., None] != 0, img, blurred_frame)
        # if hand_blur_yolo:
        #     img = yolo8_blur(img)
        #### 2-1번 기능 : 배경 이미지 합성
        if hand_compose:    # 지정한 이미지 배경에 합성 처리
            rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = selfie_segmentation.process(rgb_frame)
            condition = results.segmentation_mask > 0.1
            background_image = cv2.imread('./image/background.jpg')
            background_image_resized = cv2.resize(background_image, (img.shape[1], img.shape[0]))
            composite_frame = img.copy()
            composite_frame[~condition] = background_image_resized[~condition]
            img = composite_frame
        #### 부가 기능 : 노래 출력 가능
        if hand_song:   # 노래 기능
            file_path = "./music/Unavailable.mp3"  # 음악 파일의 경로를 지정해주세요
            play_music(file_path)
        #### 3번 기능 : 얼굴 필터
        if hand_face_filter_mode: ##얼굴 필터
            overlay_image = cv2.imread('./image/image.png', cv2.IMREAD_UNCHANGED)
            prev_face_location = []
            results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.detections:
                prev_face_location = []
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = img.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    overlay_width = min(w, img.shape[1] - x)
                    overlay_height = min(h, img.shape[0] - y)
                    overlay_image_resized = cv2.resize(overlay_image, (overlay_width, overlay_height))
                    alpha_mask = overlay_image_resized[:, :, 3] / 255.0
                    for i in range(overlay_height):
                        for j in range(overlay_width):
                            if 0 <= y + i < img.shape[0] and 0 <= x + j < img.shape[1]:
                                if alpha_mask[i, j] > 0.1:
                                    img[y + i, x + j, :] = (1 - alpha_mask[i, j]) * img[y + i, x + j, :] + \
                                                             alpha_mask[i, j] * overlay_image_resized[i, j, 0:3]
                    prev_face_location = (x, y, w, h)
        #### 5 - 2번 기능 : 음량 감소 기능
        if hand_down_sound:  ##얼굴 필터
            # 한번 호출될때 마다 음량 감소
            current_volume = volume.GetMasterVolumeLevelScalar()  # 현재 시스템 음량 가져오기
            vol = current_volume -  0.1  # 시스템 음량 감소 (0.05는 대략 -5% 감소)
            # 만약 음량값이 0.0보다 작아질 경우 => 0.0으로 고정
            if vol < 0.0:
                vol = 0.0
            # 음소거 이미지 선택
            if 0.65 < vol <= 1.0:
                sound_img = sound_img_vol3
            elif 0.35 < vol <= 0.65:
                sound_img = sound_img_vol2
            elif 0.0 < vol <= 0.35:
                sound_img = sound_img_vol1
            else:
                sound_img = sound_img_vol0
            # 감소된 값으로 음량값 설정
            volume.SetMasterVolumeLevelScalar(float(vol), None)
            # 프레임 크기의 절반으로 음소거 이미지 크기 조정
            sound_img_resized = cv2.resize(sound_img, (img.shape[1] // 2, img.shape[0] // 2))
            # 출력 이미지를 중앙에 배치
            x1 = (img.shape[1] // 2) - (sound_img_resized.shape[1] // 2)
            x2 = (img.shape[1] // 2) + (sound_img_resized.shape[1] // 2)
            y1 = (img.shape[0] // 2) - (sound_img_resized.shape[0] // 2)
            y2 = (img.shape[0] // 2) + (sound_img_resized.shape[0] // 2)
            # 음소거 이미지를 출력 프레임 위에 배치
            output_frame = img.copy()
            # 사이즈가 맞게 잘라냄
            sound_img_resized = sound_img_resized[:y2 - y1, :x2 - x1]
            # 음량 이미지를 그레이스케일로 변환
            sound_img_gray = cv2.cvtColor(sound_img_resized, cv2.COLOR_BGR2GRAY)
            # 음량 이미지를 이진화하여 처리
            _, sound_img_binary = cv2.threshold(sound_img_gray, 127, 255, cv2.THRESH_BINARY)
            # 이진화된 이미지를 3채널로 변환
            sound_img_binary = cv2.cvtColor(sound_img_binary, cv2.COLOR_GRAY2BGR)
            # 음량 이미지의 픽셀값을 조정하여 이미지를 덮어씌움
            mask = sound_img_binary / 255
            output_frame[y1:y2, x1:x2] = output_frame[y1:y2, x1:x2] * (mask) + sound_img_resized * (1 - mask)
            img = output_frame
            # 지속방식이기때문에 False설정
            hand_down_sound = False
            #### 5 - 3번 기능 : 음량 증가 기능
        if hand_up_sound:
            # 한번 호출될때 마다 음량 증가
            current_volume = volume.GetMasterVolumeLevelScalar()  # 현재 시스템 음량 가져오기
            vol = current_volume + 0.1  # 시스템 음량 증가 (0.05는 대략 +5% 감소)
            # 만약 음량값이 0.0보다 작아질 경우 => 0.0으로 고정
            if vol > 1.0:
                vol = 1.0
            # 음소거 이미지 선택
            if 0.65 < vol <= 1.0:
                sound_img = sound_img_vol3
            elif 0.35 < vol <= 0.65:
                sound_img = sound_img_vol2
            elif 0.0 < vol <= 0.35:
                sound_img = sound_img_vol1
            else:
                sound_img = sound_img_vol0
            # 감소된 값으로 음량값 설정
            volume.SetMasterVolumeLevelScalar(float(vol), None)
            # 프레임 크기의 절반으로 음소거 이미지 크기 조정
            sound_img_resized = cv2.resize(sound_img, (img.shape[1] // 2, img.shape[0] // 2))
            # 출력 이미지를 중앙에 배치
            x1 = (img.shape[1] // 2) - (sound_img_resized.shape[1] // 2)
            x2 = (img.shape[1] // 2) + (sound_img_resized.shape[1] // 2)
            y1 = (img.shape[0] // 2) - (sound_img_resized.shape[0] // 2)
            y2 = (img.shape[0] // 2) + (sound_img_resized.shape[0] // 2)
            # 음소거 이미지를 출력 프레임 위에 배치
            output_frame = img.copy()
            # 사이즈가 맞게 잘라냄
            sound_img_resized = sound_img_resized[:y2 - y1, :x2 - x1]
            # 음량 이미지를 그레이스케일로 변환
            sound_img_gray = cv2.cvtColor(sound_img_resized, cv2.COLOR_BGR2GRAY)
            # 음량 이미지를 이진화하여 처리
            _, sound_img_binary = cv2.threshold(sound_img_gray, 127, 255, cv2.THRESH_BINARY)
            # 이진화된 이미지를 3채널로 변환
            sound_img_binary = cv2.cvtColor(sound_img_binary, cv2.COLOR_GRAY2BGR)
            # 음량 이미지의 픽셀값을 조정하여 이미지를 덮어씌움
            mask = sound_img_binary / 255
            output_frame[y1:y2, x1:x2] = output_frame[y1:y2, x1:x2] * (mask) + sound_img_resized * (1 - mask)
            img = output_frame
            # 지속방식이기때문에 False설정
            hand_up_sound = False
            #### 5 - 1번 기능 : 음소거 기능
        if hand_No_Sound:
            # 음소거 이미지 읽어오기
            nosound_img = sound_img_vol0
            # 프레임 크기의 절반으로 음소거 이미지 크기 조정
            sound_img_resized = cv2.resize(nosound_img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
            # 출력 이미지를 중앙에 배치
            x1 = int((img.shape[1] / 2) - (sound_img_resized.shape[1] / 2))
            x2 = int((img.shape[1] / 2) + (sound_img_resized.shape[1] / 2))
            y1 = int((img.shape[0] / 2) - (sound_img_resized.shape[0] / 2))
            y2 = int((img.shape[0] / 2) + (sound_img_resized.shape[0] / 2))
            volume.SetMasterVolumeLevelScalar(float(0.0), None)
            # 음소거 이미지를 출력 프레임 위에 배치
            output_frame = img.copy()
            # 사이즈가 맞게 잘라냄
            sound_img_resized = sound_img_resized[:y2 - y1, :x2 - x1]
            # 음량 이미지를 그레이스케일로 변환
            sound_img_gray = cv2.cvtColor(sound_img_resized, cv2.COLOR_BGR2GRAY)
            # 음량 이미지를 이진화하여 처리
            _, sound_img_binary = cv2.threshold(sound_img_gray, 127, 255, cv2.THRESH_BINARY)
            # 화면 off인지를 확인한다.확인해서 off상태라면 음소거 이미지가 하얀색으로 출력되도록 한다.
            # 이진화된 이미지를 3채널로 변환
            sound_img_binary = cv2.cvtColor(sound_img_binary, cv2.COLOR_GRAY2BGR)
            # 음량 이미지의 픽셀값을 조정하여 이미지를 덮어씌움
            mask = sound_img_binary / 255
            output_frame[y1:y2, x1:x2] = output_frame[y1:y2, x1:x2] * (mask) + sound_img_resized * (1 - mask)
            # static_img가 None이 아닐 경우
            if static_img is not None:
                output_frame = static_img
                sound_img_binary = ~sound_img_binary
                mask = sound_img_binary / 255
                output_frame[y1:y2, x1:x2] = output_frame[y1:y2, x1:x2] * (1 - mask) + sound_img_binary * (mask)
                static_img = output_frame
            # 실제 출력 이미지에 할당
            img = output_frame
            # 지속방식이기때문에 False설정
        #### 2-2번 기능 : 화면 정지
        if hand_stop:  # 화면 멈춤 처리
            if save_frame is not None:
                img = save_frame
    #### On/Off 기능의 검은 화면 출력 조건 삽입부
    if static_img is not None:
        print("staticimage!")
        img = static_img
    #### On/Off 모드 1의 On = True이라면? = > 다른 검은화면x 영상출력
    if(On):
        cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()