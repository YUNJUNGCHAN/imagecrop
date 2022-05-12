from re import X
import cv2
import numpy as np

CONFIDENCE = 0.5
THRESHOLD = 0.3
LABELS = ['Person','','','','','','','','','','','','','','','','Dog','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','','Cell Phone','','','','','','','','','','','','','','','','','']
# 사람, 스마트폰 외의 객체는 라벨링을 하지 않았음.

img = cv2.imread('data/dog.jpg', cv2.IMREAD_COLOR)
net = cv2.dnn.readNetFromDarknet('cfg/yolov4.cfg', 'yolov4.weights')
#net = cv2.dnn.readNetFromONNX('/home/yun/yolov5/yolov5s.onnx')


H, W, _ = img.shape

blob = cv2.dnn.blobFromImage(img, scalefactor=1/255., size=(416, 416), swapRB=True)
net.setInput(blob)
output = net.forward()

boxes, confidences, class_ids = [], [], []

for det in output:
    box = det[:4]
    scores = det[5:]
    class_id = np.argmax(scores)
    confidence = scores[class_id]

    if confidence > CONFIDENCE:
        cx, cy, w, h = box * np.array([W, H, W, H])
        x = cx - (w / 2)
        y = cy - (h / 2)

        boxes.append([int(x), int(y), int(w), int(h)])
        confidences.append(float(confidence))
        class_ids.append(class_id)
        
idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

if len(idxs) > 0:
    print('find!')
    for i in idxs.flatten():
        x, y, w, h = boxes[i]
            
        if class_ids[i] == 16:
            cv2.imshow('result2', img[y-10:y+h+10 , x-10:x+w+10])
            cv2.imwrite('result_for_lee.jpg', img[y-10:y+h+10 , x-10:x+w+10])    
            
        print('')
        cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        cv2.putText(img, text='%s %.2f %d' % (LABELS[class_ids[i]], confidences[i], w), org=(x, y - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
            
        
cv2.imshow("result", img)
cv2.waitKey(0)