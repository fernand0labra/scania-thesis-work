import os
import cv2

dir_path = ''

try:     os.mkdir(os.path.join(dir_path, 'jpg'))
except:  pass

for elem in os.listdir(dir_path):
    if not os.path.isfile(os.path.join(dir_path, elem)): continue
    elem_without_end = elem.split('.')[:-1]
    cv2.imwrite(os.path.join(dir_path, 'jpg', os.path.join(*elem_without_end) + '.jpg'), cv2.imread(os.path.join(dir_path, elem)), [int(cv2.IMWRITE_JPEG_QUALITY), 100])