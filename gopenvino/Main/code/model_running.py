import openvino as ov
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
from PIL import Image

class FaceRecognition:
    def __init__(self, member_folders, image_size=160, margin=0, min_face_size=20, threshold=1.0):
        self.mtcnn = MTCNN(image_size=image_size, margin=margin, min_face_size=min_face_size)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.known_encodings = []
        self.known_names = []
        self.threshold = threshold
        self.load_members(member_folders)

    def load_members(self, member_folders):
        path_list = []
        self.dir_list = os.listdir(member_folders)
        for i in range(len(self.dir_list)):
            path_list.append(os.path.join(member_folders, self.dir_list[i]))
        for member_folder in path_list:
            member_name = member_folder.split('/')[-1]
            for image_name in os.listdir(member_folder):
                image_path = os.path.join(member_folder, image_name)
                image = Image.open(image_path)
                face = self.mtcnn(image)
                if face is not None:
                    encoding = self.resnet(face.unsqueeze(0)).detach().cpu()
                    self.known_encodings.append(encoding)
                    self.known_names.append(member_name)

    def recognize(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        face = self.mtcnn(image)
        if face is not None:
            encoding = self.resnet(face.unsqueeze(0)).detach().cpu()
            distances = [torch.dist(encoding, known_encoding).item() for known_encoding in self.known_encodings]
            min_distance = min(distances)
            if min_distance < self.threshold:
                min_index = distances.index(min_distance)
                name = self.dir_list[min_index]
                return name, self.mtcnn.detect(image), True
            return "Unknown", self.mtcnn.detect(image), False
        return "NoOne", None, False
    def members(self):
        members = self.dir_list
        return members

class AngleEstimation:
    def __init__(self, path, device='CPU', output='pitch'):
        '''
        load model and set ouput
        default device is CPU
        defalut return is pitch
        '''
        core = ov.Core()
        model_head = core.read_model(model=path)
        self.compiled_model_head = core.compile_model(model=model_head, device_name=device)
        input_layer_head = self.compiled_model_head.input(0)
        if output == 'yaw':
            self.output_layer_pitch = self.compiled_model_head.output(0)
        elif output == 'pitch':
            self.output_layer_pitch = self.compiled_model_head.output(1)
        elif output == 'roll':
            self.output_layer_pitch = self.compiled_model_head.output(2)
        self.height, self.width = list(input_layer_head.shape)[2:4]

    def predict(self, frame, boxes):
        '''
        predict from frame and boxes
        return pitch angle with box point
        '''
        results = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                continue  # Skip invalid boxes

            input_head = frame[y1:y2, x1:x2]
            if input_head.size == 0:
                continue  # Skip empty inputs

            input_head = cv2.resize(input_head, (self.width, self.height), interpolation=cv2.INTER_AREA)
            input_head = input_head.transpose((2, 0, 1))
            input_head = input_head[np.newaxis, ...]

            result = self.compiled_model_head([input_head])[self.output_layer_pitch]
            result = np.squeeze(result)
            results.append([x1,x2,y1,y2,result])
        return results
