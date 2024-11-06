import os
import av
import cv2
import shutil
import subprocess
import numpy as np

###

width=2896; height=1876

softisp_config_path = ''

data_path = ''
camera = ''; date = ''

file_path = data_path + camera + '/' + date + '/'

for elem in os.listdir(file_path):
    
    rosbag_path = os.path.join(file_path, elem)
    if not os.path.isfile(rosbag_path):  continue

    rosbag_folder = os.path.join(*rosbag_path.split('.')[:-1])
    
    try:     os.mkdir(rosbag_folder + '/temp')
    except:  pass
    
    try:     os.mkdir(rosbag_folder + '/rgb')
    except:  pass

    subprocess.run(["SoftISP", "--config", softisp_config_path, "--file", rosbag_path], cwd=rosbag_folder + '/temp/')
    
    container = av.open(os.path.join(rosbag_folder, 'temp/', os.listdir(rosbag_folder + '/temp/')[0]))

    for idx, frame in enumerate(av.open(os.path.join(rosbag_folder, 'temp/', os.listdir(rosbag_folder + '/temp/')[0])).decode()):
        img = cv2.cvtColor(np.asarray(frame.to_image()), cv2.COLOR_RGB2BGR)
        cv2.imwrite(rosbag_folder + '/rgb/' + str(idx).zfill(8) + '.png', img)

    container.close()

    shutil.rmtree(rosbag_folder + '/temp', ignore_errors=True) 
