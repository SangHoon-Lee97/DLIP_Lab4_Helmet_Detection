import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

import serial
import numpy as np
from timeit import default_timer as timer
from pygame import mixer

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
@torch.no_grad()


def detect(opt):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    print("Half: ",half) # False

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    t0 = time.time()
    buf_person = []
    buf_helmet = []
    buf_head = [] 
    buf_time   = []
    idx2 = 0
    accum_time = 0
    curr_fps   = 0
    prev_time  = timer()
    fps_my     = "FPS: ??"
    out_Text   = ''
    out_Text2  = ''
    Warning_Check = 0
    Warning_Check2 = 0
    Safe_Check = 0
    SoundFlag1 = 0
    SoundFlag2 = 0
    play_count1 = 0
    play_count2 = 0
    use_arduino = False

    # Arduino Connection
    if(use_arduino):
        arduino = serial.Serial('com5',9600)

    num_save = 0
    # Algorithm Start
    for path, img, im0s, vid_cap in dataset:
        mixer.init()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()

        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        
        t2 = time_synchronized()
        # FPS Calculation.
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1

        buf_time.append(curr_fps)
        print(buf_time[idx2])
        idx2 += 1 

        if accum_time > 1:
            accum_time = accum_time - 1
            fps_my = "FPS: " + str(curr_fps)
            curr_fps = 0

        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s1, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s1, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
            
            
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s1 += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop
            Warning_FLAG = 2 # Default Flag

            # cv2.imshow("image",imc)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for classes_ in det[:, -1].unique(): 
                    n1 = (det[:, -1] == classes_).sum()  # detections per class
                    s1 += f"{n1} {names[int(classes_)]}{'s1' * (n1 > 1)}, "  # add to string

                
                #Variables initialize
                num_person = 0 
                num_helmet = 0 
                num_head   = 0

                # Post Processing.
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    
                    if save_img or opt.save_crop or view_img or webcam:  # Add bbox to image
                        classes_ = int(cls)  # integer class 
                        print(classes_)      # Check Detected Classes

                        label = None if opt.hide_labels else (names[classes_] if opt.hide_conf else f'{names[classes_]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(classes_, True), line_thickness=opt.line_thickness)

                        # Detected Class Counting
                        if(classes_ == 1):
                            num_head   = num_head + 1
                        elif(classes_ == 2):
                            num_helmet = num_helmet + 1

                        if opt.save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[classes_] / f'{p.stem}.jpg', BGR=True)


                num_person = num_helmet + num_head
                # in order to save countings on text file. Store Data in Buffer
                buf_person.append(num_person)
                buf_helmet.append(num_helmet)
                buf_head.append(num_head)
                #####################################################################
                ## 1. helmet_head_person.pt
                ## 씽씽이 경우 
                if(opt.mode == '0'):
                    if (num_person > 1):
                        text_for_warning = "Warning! Only 1 Person Can Ride"
                        if(num_person == num_head):
                            text_for_warning = "Warning! Only 1 Person Can Ride. No Helmet Detected"

                        elif(num_head>0 and num_person == num_helmet+num_head):
                            text_for_warning = "Warning! Only 1 Person Can Ride. More Helmet needed!"

                        elif(num_head == 0 and num_person == num_helmet + num_head):
                            text_for_warning = "Warning! Only 1 Person Can Ride."
                        text_color = (0,0,255)
                        Warning_Check += 1
                        Warning_Check2 += 1

                        if(Warning_Check > 10):
                            Warning_FLAG = 1    
                            Warning_Check = 0
                            Safe_Check = 0
                        if(Warning_Check2 > 200):
                            Warning_FLAG = 1
                            Warning_Check2 =0
                            Safe_Check = 0
                            SoundFlag1 = 0
                            SoundFlag1 = 0
                            play_count1 = 0
                            play_count2 = 0 

                    elif(num_person == 1):
                        if(num_person == num_head):
                            text_for_warning = "Warning! No Helmet Detected!"
                            Warning_Check += 1
                            Warning_Check2 += 1

                            if(Warning_Check > 10):
                                Warning_FLAG = 1    
                                Warning_Check = 0
                                Safe_Check = 0
                            if(Warning_Check2 > 200):
                                Warning_FLAG = 1
                                Warning_Check2 =0
                                Safe_Check = 0
                                SoundFlag1 = 0
                                SoundFlag1 = 0
                                play_count1 = 0
                                play_count2 = 0 
                            text_color = (0,0,255)

                        elif(num_person == num_helmet):
                            text_for_warning = "Good! You are Safe"
                            Safe_Check += 1
                            if(Safe_Check > 10):
                                Warning_FLAG = 0   
                                Safe_Check = 0
                                Warning_Check = 0
                            text_color = (0,255,0)
                
                ## 2. 공사장의 경우 (검문 미션) 1명씩 통과.
                elif(opt.mode == '1'):
                    if (num_person > 1):
                        text_for_warning = "Warning! Enter One By One"
                        text_color = (0,0,255)
                        Warning_Check += 1
                        if(Warning_Check > 10):
                            Warning_FLAG = 1    
                            Warning_Check = 0
                            Safe_Check = 0

                    elif(num_person == 1):
                        if(num_person == num_head):
                            text_for_warning = "Warning! No Helmet Detected!"
                            Warning_Check += 1
                            if(Warning_Check > 10):
                                Warning_FLAG = 1    
                                Warning_Check = 0
                                Safe_Check = 0
                            text_color = (0,0,255)

                        elif(num_person == num_helmet):
                            text_for_warning = "Good! You are Safe"
                            Safe_Check += 1
                            if(Safe_Check > 10):
                                Warning_FLAG = 0   
                                Safe_Check = 0
                                Warning_Check = 0
                            text_color = (0,255,0)

                ## 3. 공사장, 여러명 동시에 검출 경우. 
                elif(opt.mode == '2'):
                    if (num_person > 1):
                        if(num_person == num_helmet):
                            text_for_warning = "Good! This Area is Safe"
                            Safe_Check +=1
                            if(Safe_Check > 10):
                                Warning_FLAG = 0
                                Safe_Check = 0
                                Warning_Check = 0
                            text_color = (0,255,0)
                        else:
                            Danger_workers = num_person -num_helmet
                            text_for_warning = "Warning! "+str(Danger_workers)+" Workers In Danger!"
                            Warning_Check += 1
                            if(Warning_Check > 10):
                                Warning_FLAG = 1    
                                Warning_Check = 0
                                Safe_Check = 0
                            text_color = (0,0,255)

                    elif(num_person == 1):
                        if(num_person == num_head):
                            text_for_warning = "Warning! No Helmet Detected!"
                            Warning_Check += 1
                            if(Warning_Check > 10):
                                Warning_FLAG = 1    
                                Warning_Check = 0
                                Safe_Check = 0
                            text_color = (0,0,255)

                        elif(num_person == num_helmet):
                            text_for_warning = "Good! This Area is Safe"
                            Safe_Check += 1
                            if(Safe_Check > 10):
                                Warning_FLAG = 0   
                                Safe_Check = 0
                                Warning_Check = 0
                            text_color = (0,255,0)

                out_Text  = "Helmet Number: " + str(num_helmet) + " Person Number: "+str(num_person)
                out_Text2 = "Status: " + text_for_warning


                if(Warning_FLAG == 1) : 
                    SoundFlag1 = 1
                    play_count1 +=1
                    if(play_count1>3):
                        alert.stop()
                    else:
                        if(SoundFlag2 == 1):
                            alert.stop()
                            alert = mixer.Sound('Danger.wav')
                            alert.set_volume(0.7)
                            alert.play()
                        elif(SoundFlag2 == 0):
                            alert = mixer.Sound('Danger.wav')
                            alert.set_volume(0.7)
                            alert.play()
                    play_count2 = 0;

                elif(Warning_FLAG == 0): 
                    SoundFlag2 = 1
                    play_count2 +=1
                    if(play_count2>4):
                        alert.stop()
                    else: 
                        if(SoundFlag1 == 1):
                            alert.stop()
                            alert = mixer.Sound('Success.wav')
                            alert.set_volume(0.7)
                            alert.play()
                        elif(SoundFlag1 == 0):
                            alert = mixer.Sound('Success.wav')
                            alert.set_volume(0.7)
                            alert.play()
                    play_count1 = 0;

                if(use_arduino):
                    if(Warning_FLAG == 1):
                        pass_val = str(1)
                        pass_val = pass_val.encode('utf-8')
                        arduino.write(pass_val)
                    if(Warning_FLAG == 0):
                        pass_val = str(0)
                        pass_val = pass_val.encode('utf-8')
                        arduino.write(pass_val)


            ## When No Object is Detected.
            else:
                text_for_warning =""
                text_color = (0,255,0)
                num_person = 0 
                num_helmet = 0 
                num_head   = 0
                out_Text  = "Helmet Number: " + str(num_helmet) + " Person Number: "+str(num_person)
                out_Text2 = "Status: " + text_for_warning
                play_count1= 0
                play_count2 = 0
                SoundFlag1 = 0
                SoundFlag2 = 0

                buf_person.append(num_person)
                buf_helmet.append(num_helmet)
                buf_head.append(num_head)

                if(use_arduino):
                    if(Warning_FLAG == 1):
                        pass_val = str(1)
                        pass_val = pass_val.encode('utf-8')
                        arduino.write(pass_val)
                    if(Warning_FLAG == 0):
                        pass_val = str(0)
                        pass_val = pass_val.encode('utf-8')
                        arduino.write(pass_val)

            num_save += 1
            print("num_save" ,num_save)

            # Print time (inference + NMS)
            print(f'{s1}Done. ({t2 - t1:.3f}s)')
            
            # Stream results
            if view_img: # Webcam Used
                cv2.putText(im0, text= fps_my, org=(15, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1.0, color=(255, 0, 0), thickness=2)
                cv2.putText(im0, text =out_Text, org =(15,140), fontFace =cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1.0, color= (255,0,0), thickness =2)
                cv2.putText(im0, text =out_Text2, org =(15,200), fontFace =cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1.0, color= text_color, thickness =2)
                cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                cv2.imshow("result", im0)
                cv2.waitKey(1)
            else :       # Video Used 
                cv2.putText(im0, text= fps_my, org=(15, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1.0, color=(255, 0, 0), thickness=2)
                cv2.putText(im0, text =out_Text, org =(15,140), fontFace =cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1.0, color= (255,0,0), thickness =2)
                cv2.putText(im0, text =out_Text2, org =(15,200), fontFace =cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1.0, color= text_color, thickness =2)
                cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                cv2.imshow("result", im0)
                cv2.waitKey(1)


    # Arduino Serial Communication
    if(use_arduino):
        pass_val = str(2).encode('utf-8')
        arduino.write(pass_val)
    
    # File save in txt
    f = open("person_head_helmet2.txt",'w')
    for i in range(num_save):
        f.write(str(i)+ ', ' + str(buf_person[i]) + ', ' + str(buf_head[i]) + ', '+ str(buf_helmet[i]) +'\n')
    f.close()


    print(f'Done. ({time.time() - t0:.3f}s)')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='helmet_head_person_s.pt', help='model.pt path(s)') #helmet_head_person_s.pt / #yolov5s_helmet_MSBT
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--mode', type=str, default=0, help='mode selection: --mode 0(Lime/) 1(Construction Detection) 2(Many worker)')
    parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default= 2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    detect(opt=opt)