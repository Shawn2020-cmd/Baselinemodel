import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from yolo import YOLO

if __name__ == "__main__":

    #------------------------------------------------------------------------------------------------------------------#
    # map_mode is used to specify what this file calculates at runtime
    # map_mode is 0 for the entire map calculation process, including getting the prediction results, getting the true frame, and calculating the VOC_map.
    # map_mode is 1 for getting the prediction result only.
    # map_mode is 2 means only get the real frame.
    # map_mode is 3 for just calculating VOC_map.
    # map_mode is 4 means use COCO toolkit to calculate 0.50:0.95map of current dataset. need to get prediction result, get real frame and install pycocotools.
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = 0
    #--------------------------------------------------------------------------------------#
    # The classes_path here is used to specify the classes for which the VOC_map needs to be measured
    # It is generally sufficient to match the classes_path used for training and prediction
    #--------------------------------------------------------------------------------------#
    classes_path    = 'model_data/voc_classes.txt'
    #--------------------------------------------------------------------------------------#
    # MINOVERLAP is used to specify the mAP0.x you want to get, what is the meaning of mAP0.x please students Baidu.
    # For example, to calculate mAP0.75, you can set MINOVERLAP = 0.75.
    #
    # When a prediction frame overlaps with the true frame by more than MINOVERLAP, the prediction frame is considered as a positive sample, otherwise it is a negative sample. # If a prediction frame overlaps with the true frame by more than MINOVERLAP, the prediction frame is considered as a positive sample.
    # # Therefore the larger the value of MINOVERLAP, the more accurately the predictor frame has to be predicted in order to be considered a positive sample, at which point the lower the calculated mAP value.
    #--------------------------------------------------------------------------------------#
    MINOVERLAP      = 0.5
    #--------------------------------------------------------------------------------------#
    # Due to the limitation of the mAP calculation principle, the network needs to obtain nearly all the prediction frames in order to calculate the mAP.
    # Therefore, the value of confidence should be set as small as possible to obtain all possible prediction frames.
    #
    # This value is generally not adjusted. Because mAP needs to be computed with nearly all prediction frames, the confidence value cannot be changed arbitrarily.
    # # To get the Recall and Precision values for different thresholds, modify the score_threhold below.
    #--------------------------------------------------------------------------------------#
    confidence      = 0.001
    #--------------------------------------------------------------------------------------#
    # The size of the non-extremely suppressed values used in the prediction, with larger values indicating less stringent non-extremely suppressed values.
    # The value is generally not adjusted.
    #--------------------------------------------------------------------------------------#
    nms_iou         = 0.5
    #---------------------------------------------------------------------------------------------------------------#
    # Recall and Precision are not an area concept like AP, so the Recall and Precision values of the network are different when the threshold values are different.
    #
    # By default, the Recall and Precision calculated in this code represent the Recall and Precision values corresponding to when the threshold value is 0.5 (defined here as score_threhold).
    # Because the mAP calculation needs to obtain nearly all the prediction frames, the confidence defined above cannot be changed arbitrarily.
    # Define a score_threhold to represent the threshold value, and then find the Recall and Precision values corresponding to the threshold value when calculating mAP.
    #---------------------------------------------------------------------------------------------------------------#
    score_threhold  = 0.5
    #-------------------------------------------------------#
    # map_vis is used to specify whether to enable visualization of VOC_map calculations or not
    #-------------------------------------------------------#
    map_vis         = False
    # -------------------------------------------------------#
    # Point to the folder where the VOC dataset is located
    # Points to the VOC dataset in the root directory by default
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    # -------------------------------------------------------#
    # Folder for results output, default is map_out
    #-------------------------------------------------------#
    map_out_path    = 'map_out'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO(confidence = confidence, nms_iou = nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            yolo.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")
        
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold = score_threhold, path = map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")
