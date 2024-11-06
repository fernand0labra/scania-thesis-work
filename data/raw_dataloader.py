import os
import cv2
import shutil
import rosbag
import subprocess
import numpy as np

###

width=2896; height=1876

data_path = ''
camera = ''; date = ''

file_path = data_path + camera + '/' + date + '/'

for elem in os.listdir(file_path):
    
    rosbag_path = os.path.join(file_path, elem)
    if not os.path.isfile(rosbag_path):  continue

    rosbag_folder = os.path.join(*rosbag_path.split('.')[:-1])
    
    try:     os.mkdir(rosbag_folder)
    except:  continue

    try:     os.mkdir(rosbag_folder + '/raw')
    except:  pass

    os.mkdir(rosbag_folder + '/temp')

    try:
        bag = rosbag.Bag(rosbag_path)
    except:
        subprocess.run(["rosbag", "reindex", rosbag_path]) 
        bag = rosbag.Bag(rosbag_path)

    for idx, (_, msg, _) in enumerate(bag.read_messages(topics=['/driver_nodelet/channel_0/raw_frames'])):

        instance_path = rosbag_folder + '/temp/' + 'out0-0x2c-' + str(width) + 'x' + str(height) + '-16-32.raw'

        descriptor = open(instance_path, 'wb');  descriptor.write(msg.data);  descriptor.close()
        subprocess.run(["sxpfcsi2decode", instance_path], cwd=rosbag_folder + '/temp/')

        img = np.fromfile(open(rosbag_folder + '/temp/' + 'out0-0x2c-' + str(width) + 'x' + str(height) + '-16-32.bin', 'rb'), dtype=np.uint16)
        
        img = img.reshape(height, width)
        img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
        cv2.imwrite(rosbag_folder + '/raw/' + str(idx).zfill(8) + '.png', cv2.resize(img*16, (width, height)))

    bag.close()

    shutil.rmtree(rosbag_folder + '/temp', ignore_errors=True) 