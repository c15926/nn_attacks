## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

# import tensorflow as tf
###insert (220915) ####
import os
import scipy.misc
from scipy.stats import betabinom
######

import tensorflow.compat.v1 as tf
import numpy as np
import time
import cv2
from skimage import io


# from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
# from setup_inception import ImageNet, InceptionModel

#### insert (220915) ####
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.semi_supervised import LabelPropagation
import keras
from keras_preprocessing import image
import matplotlib.pyplot as plt
########

from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi

tf.disable_v2_behavior()

def show(string, img, i, a, c, b, m):
    """
    Show MNSIT digits in the console.
    """

    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    
    ### insert (220915)

    
    test_images = img.reshape(28, 28)
    image_path = os.path.join('/content/drive/MyDrive/c15926/nn_robust_attacks/samples/sampled_iages_%s_(D%d_%d_%d_iter_%d)_%d.jpg' %(string, m, a, a+c, b, i))
    cv2.imwrite(image_path, test_images)
    plt.imshow(test_images)
    plt.show()
#    misc.imsave(image_path, test_images)
    
    #######

    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))
      
    ####### 코드 넣어서 나온 이미지를 따로 저장되도록 -> 구글드라이브에 자기가 지정한 장소에 #######
    ####### 구글 사이트 검색해서 라이브러리를 불러와서 저장하면 됩니다. ######

    #img_file = "/content/drive/MyDrive/c15926/nn_robust_attacks/Samples/" # 표시할 이미지 경로
    #cv2.imwrite('/content/drive/MyDrive/c15926/nn_robust_attacks/Samples/save_image.jpg', img)
    #io.imsave('/content/drive/MyDrive/c15926/nn_robust_attacks/Samples/save_image.jpg', img)

  



def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.test_labels.shape[1])
                

            for j in seq:
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
                print(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    print(inputs.shape)
    print(targets)

    return inputs, targets


if __name__ == "__main__":
    with tf.Session() as sess:

      ### insert (220915) #####
        m = 1
        starting = 0
        num_sample = 1
        num_iter = 500

        flag = True # or False (if 비목표 공격할 경우)
        size_batch = 9 # False일 경우, size_batch = 1로 변경

      ######


        data, model =  MNIST(), MNISTModel("models/mnist", sess)
        #data, model =  CIFAR(), CIFARModel("models/cifar", sess)
        attack = CarliniL2(sess, model, batch_size=1, max_iterations=500, confidence=0)
        #attack = CarliniL0(sess, model, max_iterations=1000, initial_const=10,
        #                   largest_const=15)

        inputs, targets = generate_data(data, samples=3, targeted=False,
                                        start=100, inception=False)
        timestart = time.time()
        adv = attack.attack(inputs, targets)
        timeend = time.time()
        
        print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

        print(adv[0].shape)
        print(inputs[0].shape)

        for i in range(len(adv)):
      

            print("Valid:")

            ###  insert (220915) ###

            show("Original", inputs[i], i, starting, num_sample, num_iter, m)

            #######s
            
            print("Orignal Classification:", model.model.predict(inputs[i:i+1]))

            print("Adversarial:")

            ### insert (220915) ###

            show("Adversarial", adv[i], i , starting, num_sample, num_iter, m)

            ######
            
            print("Adversarial Classification:", model.model.predict(adv[i:i+1]))

            print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)

          