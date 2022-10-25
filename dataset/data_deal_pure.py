import os
import cv2
import numpy as np




def image_scale(img,scale,destination,picture_list):
    for i in scale:

        tmp_image = cv2.resize(img, (int(128/i), int(128/i)), interpolation=cv2.INTER_CUBIC)
        save_path = destination + '%.1f'% i
        
        image_path = os.path.join(save_path, picture_list)
        cv2.imwrite(image_path,tmp_image) 

def data_deal():
    scale = np.linspace(1 ,4 ,31 )
    # i = np.around(scale, decimals=1)
    for vdPath_list_1 in os.listdir("path to the dataset"):
        if vdPath_list_1 != 'pure':        
            video_x_3 = "path to the dataset" + vdPath_list_1 + '/pic/'
            destination_3 = "path_to_arbitrary_resolution_data" + vdPath_list_1 + '/pic/'

            print('dealing'+video_x_3)
            if not os.path.exists(destination_3):
                os.makedirs(destination_3)
            for i in scale:            
                save_path = destination_3 + '%.1f'% i
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            if os.path.exists(video_x_3):
                for picture_list in os.listdir(video_x_3):
                    image_path = os.path.join(video_x_3, picture_list)
                    image = cv2.imread(image_path)
                    image_scale (image,scale,destination_3,picture_list)



if __name__ == "__main__":
    data_deal()