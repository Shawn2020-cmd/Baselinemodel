#-----------------------------------------------------------------------#
# predict.py puts single image prediction, camera detection, FPS testing and directory traversal detection
#-----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    #----------------------------------------------------------------------------------------------------------#
    # mode is used to specify the mode of the test:
    # 'predict' means single picture prediction, if you want to modify the prediction process, such as saving the picture, intercepting the object, etc., you can first see the detailed comments below
    # 'video' means video detection, you can call the camera or video for detection, see the comments below for details.
    # 'fps' means test fps, the image used is street.jpg inside img, check the comment below for details.
    # 'dir_predict' means traverse the folder to test and save. Default is to traverse the img folder and save the img_out folder, see below for details.
    # 'heatmap' Indicates the heatmap visualization of the predicted results, see the note below for details.
    # 'export_onnx' means export the model to onnx, requires pytorch1.7.1 or above.
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    #-------------------------------------------------------------------------#
    # crop specifies whether to intercept the target after a single image prediction.
    # count specifies whether to count the target
    # crop, count are only valid when mode='predict'.
    #-------------------------------------------------------------------------#
    crop            = False
    count           = False
    #----------------------------------------------------------------------------------------------------------## video_path is used to specify the path of the video, when video_path=0, it means to detect the camera.
    # video_save_path If you want to detect the video, set video_path = "xxx.mp4", it means read out the xxx.mp4 file in the root directory.
    # video_save_path means the path to save the video, when video_save_path="", it means no save.
    # video_save_path Indicates the path to save the video, when video_save_path="", it means not save the video. If you want to save the video, set video_save_path = "yyyy.mp4", it means save it as yyyy.mp4 file in the root directory.
    # video_fps The fps of the video to be saved.
    # video_fps
    # video_path, video_save_path and video_fps are only valid when mode='video'.
    # Save video needs ctrl+c to exit or run to the last frame to complete the full save step.
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    # test_interval Used to specify the number of times the image is detected when measuring fps. Theoretically the larger the test_interval, the more accurate the fps.
    # fps_image_path Used to specify the fps image for testing.
    # fps_image_path
    # test_interval and fps_image_path are only valid for mode='fps'.
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    # dir_origin_path specifies the path to the folder where the image is to be detected.
    # dir_save_path Specifies the path to the folder where the image will be saved after detection.
    # dir_origin_path and dir_save_path are only valid with mode='dir_predict'.
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    #-------------------------------------------------------------------------#
    # dir_origin_path specifies the path to the folder where the image is to be detected.
    # dir_save_path Specifies the path to the folder where the image will be saved after detection.
    # dir_origin_path and dir_save_path
    # dir_origin_path and dir_save_path are only valid with mode='dir_predict'.
    #-------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"
    #-------------------------------------------------------------------------#
    # simplify Using Simplify onnx
    # onnx_save_path specifies the onnx save path
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode == "predict":

        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop, count=count)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("no camera or videos")

        fps = 0.0
        while(True):
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(yolo.detect_image(frame))
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()
        
    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap(image, heatmap_save_path)
                
    elif mode == "export_onnx":
        yolo.convert_to_onnx(simplify, onnx_save_path)
        
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")
