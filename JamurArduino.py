import serial
import tensorflow as tf
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime
from firebase import firebase
import requests

import csv
url = 'https://jamur-ca24c.firebaseio.com/'
firebase = firebase.FirebaseApplication(url,None)

#scope = ['https://spreadsheets.google.com/feeds',
         #'https://www.googleapis.com/auth/drive']
#creds = ServiceAccountCredentials.from_json_keyfile_name('learnvocab-c090bda32f33.json',scope)
#client = gspread.authorize(creds)
#sheet = client.open("Data Suhu").sheet1

ser = serial.Serial("/dev/ttyUSB0")

X = tf.placeholder(tf.float32, [None,2])

weights = {
  "layer1" : np.array([[-12.502901  ,   2.0814376 ,  20.61646   ,  -2.4981163 ,7.7447824 , -16.129969  ,  -1.1479825 ,  -3.6694074 , -1.3943328 ,  12.611093  ],
                       [2.4563758 ,   2.1232698 ,   1.1547309 ,  -0.45235634,-11.137693  ,   1.0871675 ,  -0.43982506, -10.733463  ,-4.7602935 ,  -0.52114516]], dtype=np.float32),
  "layer2" : np.array([[ 1.2688447 , -3.8280723 , -1.8669528 , -0.0657398 ,  4.440947  ,-0.49438506, -2.0613477 ,  4.785706  , -3.924797  , -2.7316172 ],
                       [-1.4041699 , -1.2484515 ,  1.0448365 , -0.9665495 , -1.299643  ,2.2467372 , -1.1065305 , -0.4919568 , -2.27616   , -1.2959597 ],
                       [ 1.9777837 ,  0.14167623,  0.8408132 ,  0.12296087, -5.4175334 ,3.1161137 , -2.8741012 , -9.691696  , -4.9442415 , -2.508191  ],
                       [-1.0386454 , -0.07393467, -0.40711743,  0.540365  , -1.773221  ,-2.6459615 , -0.04867569,  2.224603  , -0.5796172 ,  0.4005869 ],
                       [-0.22274388,  1.0990509 ,  0.32237715, -0.8650677 , -1.1379699 ,1.7583554 ,  9.620379  , -2.3303375 ,  7.7740307 ,  1.2337879 ],
                       [-1.9729054 , -2.8890283 , -0.6152815 , -1.1517603 ,  2.7789438 ,-3.0681896 , -1.3224268 ,  7.0305305 , -2.5179415 ,  0.2593467 ],
                       [-0.06101424, -1.0909532 ,  0.21329884, -1.2944535 ,  1.6469818 ,-0.3914194 ,  0.10224427,  0.18477948, -1.4122828 , -0.6391331 ],
                       [-0.5067579 ,  0.20376027,  0.8936112 , -1.3408732 ,  3.4333944 ,-2.4285393 ,  6.7630205 ,  2.7783487 ,  9.963321  , -0.15598455],
                       [ 0.96534127,  0.29329848,  2.4066582 ,  0.3594686 ,  2.0032718 ,-0.6886175 ,  4.8257694 ,  1.9824109 ,  2.525622  ,  0.6111814 ],
                       [ 0.6032972 ,  1.0702218 , -1.0714316 ,  0.684005  , -4.8714013 ,0.12576577,  0.79074377, -4.704713  ,  0.12383524,  0.17654547]],dtype=np.float32),
  "layer3" : np.array([[  1.4284482 ,  -2.1934125 ,  -0.98400223,  -1.0322505 ,2.7697747 ,  -1.0417286 ,  -1.5757545 ,  -2.8903975 ,-1.2048132 ,   0.5329068 ],
                       [ -0.4445942 ,  -1.6292764 ,  -1.2858584 ,   2.8691418 ,4.86168   ,  -1.1248599 ,  -1.6984035 ,  -2.0335443 ,-1.4178658 ,  -0.67849946],
                       [  1.7218382 ,  -1.2510264 ,   0.6764323 ,   0.20741525,1.7621932 ,  -3.8817074 ,   0.36348626,  -1.1054105 ,-0.66064525,  -0.5853588 ],
                       [  1.1177539 ,  -1.1810317 ,  -0.7188865 ,  -1.1543505 ,3.313429  ,  -1.8875289 ,  -1.3004596 ,  -2.3768063 ,0.5792328 ,   1.4887369 ],
                       [  2.0855567 ,  -0.3051247 ,  -0.68396044,  -2.9422507 ,0.76491386,   0.5395975 ,  -0.36065555,   4.2868767 ,-2.52913   ,   0.87720484],
                       [  2.3919842 ,  -0.3648251 ,  -3.2550075 ,  -0.8512516 ,3.492754  ,  -1.6561141 ,   0.37332585,  -3.5832844 ,0.25590593,  -0.27777445],
                       [-10.701511  ,  -1.5859585 ,  -0.36055145,  -1.9931204 ,3.048318  ,   0.41326082,   0.19670886,  -1.9743073 ,3.3601937 ,  -0.56455845],
                       [ -0.18013704,  -0.789926  ,   0.2757648 ,  -5.8217716 ,0.65258783,   0.15768042,  -0.05436259,   3.5176823 ,-6.3876143 ,  -1.4410971 ],
                       [ -5.3624234 ,  -0.47912362,  -1.850377  ,   6.727492  ,3.1759589 ,   1.074988  ,  -3.2444038 ,   0.47267634,-7.1997995 ,   3.9613583 ],
                       [  0.26106814,   0.01989038,  -1.4683579 ,  -0.0236089 ,0.06962655,  -0.9845505 ,   0.94373786,  -0.44162792,0.02221225,  -1.9631178 ]], dtype=np.float32),
  "layer4" : np.array([[ 0.46287066, -4.656665  ],
                       [ 2.2681134 ,  5.151868  ],
                       [-0.16693799,  1.4269941 ],
                       [ 2.9359186 , -0.68731076],
                       [ 2.2853847 ,  9.452858  ],
                       [ 0.64299446,  5.704114  ],
                       [-0.10021187,  0.93792444],
                       [-4.483027  , -0.22715096],
                       [ 3.415164  , -0.32514998],
                       [ 0.7645477 ,  3.7619894 ]], dtype=np.float32)
}

biases = {
  "layer1" : np.array([ 1.7478728 , -2.018559  , -5.7839484 , -0.88063264,  2.4426057 ,3.2799752 , -2.067527  ,  5.3626533 ,  3.1319818 , -2.8152387 ],dtype=np.float32),
  "layer2" : np.array([-1.096459  , -1.6998426 , -0.99487895,  0.3113135 , -0.6299982 ,-0.08440783,  0.46945944,  1.237826  , -3.347416  , -0.45652038],dtype=np.float32),
  "layer3" : np.array([ 0.88573796, -1.8524182 , -2.2749414 ,  0.25873068,  5.057738  ,-2.8489347 , -3.4978576 , -1.9113679 ,  0.96873474, -1.428945  ],dtype=np.float32),
  "layer4" : np.array([ 6.1702795, 12.271734 ], dtype=np.float32)
}

def model(X, weights, biases):
    hidden1 = tf.add(tf.matmul(X, weights["layer1"]), biases["layer1"])
    hidden1 = tf.nn.relu(hidden1)

    hidden2 = tf.add(tf.matmul(hidden1, weights["layer2"]), biases["layer2"])
    hidden2 = tf.nn.sigmoid(hidden2)

    hidden3 = tf.add(tf.matmul(hidden2, weights["layer3"]), biases["layer3"])
    hidden3 = tf.nn.sigmoid(hidden3)

    output = tf.add(tf.matmul(hidden3, weights["layer4"]), biases["layer4"])
    #output = tf.nn.sigmoid(output)
    return output

y_pred = model(X, weights, biases)

with open('data-suhu.csv', 'w', newline='') as f:
    while True:
        data = ser.readline()
        data = data.decode("utf-8").split('\r\n')
        data = list(map(float,data[0].split(',')))
        x_test = np.array([[data[2], data[3]]], dtype=np.float32)
        #x_test = np.array([[25,63]], dtype=np.float32)
        x_test = x_test/100
        with tf.Session() as sess:
            output = sess.run(y_pred, feed_dict={X:x_test})
        #ser.write(output)
        kirim = output.astype(str)[0,0] + ',' + output.astype(str)[0,1] + '\n'
        kirim = kirim.encode('latin1')
        #print(kirim)
        ser.write(kirim)
        #print('Input: ', x_test*100, '  Output:', output)
        print('InTemp:' ,data[2], '  InHum:', data[3], '  OutTemp:', data[0], '  OutHum:', data[1], '  OutVolt: ',output)
        firebase.patch('',{'/jamurANN/suhu':data[2]})
        firebase.patch('',{'/jamurANN/kelembapan':data[3]})
        firebase.patch('',{'/jamurANN/kec-kipas':output.astype(str)[0,0]})
        firebase.patch('',{'/jamurANN/kec-mist':output.astype(str)[0,1]})

        #now=datetime.datetime.now()

        # thewriter = csv.writer(f)
        # thewriter.writerow([now.strftime("%H:%M:%S"),data[2], data[3], data[0], data[1]])
        #row = [now.strftime("%H:%M:%S"),data[2], data[3], data[0], data[1]]
        #sheet.insert_row(row, len(sheet.get_all_values())+1)
