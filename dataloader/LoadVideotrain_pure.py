import os
import torch
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import json
from scipy import interpolate
from utils.heartRate import predict_heart_rate

class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, sample):
        video_x, video_y, clip_average_HR, ecg_label, frame_rate, scale= sample['video_x'], sample['video_y'], sample['clip_average_HR'], sample['ecg'], sample['frame_rate'], sample['scale']
        new_video_x = (video_x - 127.5) / 128
        new_video_y = (video_y - 127.5) / 128
        return {'video_x': new_video_x, 'video_y': new_video_y, 'clip_average_HR': clip_average_HR, 'ecg': ecg_label, 'frame_rate': frame_rate, 'scale':scale}


class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        video_x, video_y, clip_average_HR, ecg_label, frame_rate, scale= sample['video_x'], sample['video_y'], sample['clip_average_HR'], sample['ecg'], sample['frame_rate'], sample['scale']

        h, w = video_x.shape[1], video_x.shape[2]
        h1, w1 = video_y.shape[1], video_y.shape[2]
        new_video_x = np.zeros((video_x.shape[0], h, w, 3))
        new_video_y = np.zeros((video_y.shape[0], h1, w1, 3))
        p = random.random()
        if p < 0.5:
            for i in range(video_x.shape[0]):
                # video
                image = video_x[i, :, :, :]
                image = cv2.flip(image, 1)
                new_video_x[i, :, :, :] = image
                image1 = video_y[i, :, :, :]
                image1 = cv2.flip(image1, 1)
                new_video_y[i, :, :, :] = image1
            return {'video_x': new_video_x, 'video_y': new_video_y, 'clip_average_HR': clip_average_HR, 'ecg': ecg_label, 'frame_rate': frame_rate, 'scale':scale}

        else:
            return {'video_x': video_x, 'video_y': video_y, 'clip_average_HR': clip_average_HR, 'ecg': ecg_label, 'frame_rate': frame_rate, 'scale':scale}



class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        video_y = sample['video_y']
        video_x = sample['video_x']

        clip_average_HR = sample['clip_average_HR']
        ecg_label = sample['ecg']
        frame_rate = sample['frame_rate']
        scale= sample['scale']
        video_x = video_x.transpose((3, 0, 1, 2))
        video_x = np.array(video_x)
        video_y = video_y.transpose((3, 0, 1, 2))
        video_y = np.array(video_y)
        clip_average_HR = np.array(clip_average_HR)

        frame_rate = np.array(frame_rate)

        ecg_label = np.array(ecg_label)

        scale = np.array(scale)

        return {'video_x': torch.from_numpy(video_x.astype(np.float)).float(),
                'video_y': torch.from_numpy(video_y.astype(np.float)).float(),
                'clip_average_HR': torch.from_numpy(clip_average_HR.astype(np.float)).float(),
                'ecg': torch.from_numpy(ecg_label.astype(np.float)).float(),
                'frame_rate': torch.from_numpy(frame_rate.astype(np.float)).float(),
                'scale':torch.from_numpy(scale.astype(np.float)).float()}


# train
class PURE_train(Dataset):

    def __init__(self, scale,frames ,test=False, transform=None):

        self.path_json = "path_to_pure"
        self.path_data = "path_to_arbitrary_resolution_data"
        self.length = frames
        self.scale = scale        
        self.videoList = os.listdir(self.path_json)
        self.videoList.remove('pure')
        self.videoList.sort()
        self.idx_scale = 0
        self.idx_scale_vice = 0
        if not test:
            self.videoList = ['06-01', '06-03', '06-04', '06-05', '06-06', '08-01', '08-02', '08-03', '08-04', '08-05', '08-06', '05-01', '05-02', '05-03', '05-04', '05-05', '05-06', '01-01', '01-02', '01-03', '01-04', '01-05', '01-06', '04-01', '04-02', '04-03', '04-04', '04-05', '04-06', '09-01', '09-02', '09-03', '09-04', '09-05', '09-06', '07-01', '07-02', '07-03', '07-04', '07-05', '07-06']
        else:
            self.videoList = ['02-01', '02-02', '02-03', '02-04', '02-05', '02-06', '03-01', '03-02', '03-03', '03-04', '03-05', '03-06', '10-01', '10-02', '10-03', '10-04', '10-05', '10-06']
        self.videoListSegNum = [] 
        self.videoListFrameRate = [30]
        self.BVPList = [] 
        self.HRList = [] 
        self.transform =transform
        for i in range(0,len(self.videoList)):
            tempPath = self.path_json + self.videoList[i] 
            segNum = int(len(os.listdir(tempPath + '/pic/')))//self.length
            with open(tempPath + '/' +self.videoList[i] + ".json", 'r') as f:
                data = json.load(f)
                data = data["/FullPackage"]
                ppg = []
                for item in data:
                    ppg.append(item["Value"]["waveform"])
            while len(ppg) < segNum*2*self.length:
                segNum -= 1

            self.videoListSegNum.append(segNum)   
            ppg = ppg[ :segNum * self.length * 2]  
            x = np.linspace(1, segNum * self.length * 2, segNum * self.length * 2)  
            funcInterpolate = interpolate.interp1d(x, ppg, kind="slinear")

            xNew = np.linspace(1, segNum * self.length * 2, segNum * self.length )
            dataBvp = funcInterpolate(xNew)
            
            self.BVPList.append([])
            self.HRList.append([])

            for j in range(segNum):

                startFrame = self.length * j
                HR = predict_heart_rate(dataBvp[startFrame : startFrame + self.length] ,self.videoListFrameRate[0])

                self.BVPList[i].append(dataBvp[startFrame : startFrame+self.length])
                self.HRList[i].append([HR])
       
        self.sampleCount = sum(self.videoListSegNum)

        self.sampleGetIdList = []
        temp = 0
        for num in self.videoListSegNum:
            temp += num
            self.sampleGetIdList.append(temp)    


    def __len__(self):
        return  self.sampleCount

    def __getitem__(self, idx):
        for i in range(len(self.sampleGetIdList)):
            if(idx < self.sampleGetIdList[i]):
                vId = i
                clipId = idx -  self.sampleGetIdList[i-1]  if i!=0 else  idx 
                break
        picPath =  self.path_data + self.videoList[vId] + '/pic/'
        startFrame = clipId *  self.length   
        video_x = self.get_single_video_x(picPath + "{}/".format(self.scale[self.idx_scale]) , startFrame)

        video_y = self.get_single_video_x(picPath + "{}/".format(self.scale[self.idx_scale_vice]) , startFrame)

        frameRate = self.videoListFrameRate[0]
        ecgLabel = self.BVPList[vId][clipId]
        clipAverageHR =self.HRList[vId][clipId]


        sample = {'video_x': video_x,'video_y': video_y, 'frame_rate': frameRate, 'ecg': ecgLabel, 'clip_average_HR': clipAverageHR, 'scale':self.scale[self.idx_scale]}

        if self.transform:
            sample = self.transform(sample)
        return sample


    def get_single_video_x(self, video_jpgs_path, start_frame):
        image_name ='1' + '.png'
        image_path = os.path.join(video_jpgs_path, image_name)
        image_shape = cv2.imread(image_path).shape
        video_x = np.zeros((self.length, image_shape[0], image_shape[1], 3))

        for i in range(self.length):
            s = start_frame + i
            image_name = str(s) + '.png'

            # face video
            image_path = os.path.join(video_jpgs_path, image_name)

            tmp_image = cv2.imread(image_path)

            if tmp_image is None:

                tmp_image = cv2.imread('./_1.png')
                print("______________________.png")

            video_x[i, :, :, :] = tmp_image

        return video_x

    def set_scale(self, idx_scale,idx_scale_vice):
        self.idx_scale = idx_scale
        self.idx_scale_vice = idx_scale_vice
