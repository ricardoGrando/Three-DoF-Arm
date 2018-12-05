#!/usr/bin/env python

import sys
import rospy
import roslib
from std_msgs.msg import String
import threading
import random
import time
from random import randint

from topicCartesianState import *
from std_msgs.msg import Float64
from std_srvs.srv import Empty

from sympy import *
import numpy as np
import math
from serviceCartesianState import *


import math
from scipy import linalg
from keras.models import load_model
from keras.models import Model
from utils import *

import matplotlib.pyplot as plt
import matplotlib as mpl

import csv

from utils import *
from numpy import genfromtxt
import glob

PUBLISHER_RATE_SLEEP = 2

pubList =  [    '/three_dof_arm/rotation_position_controller/command',
                '/three_dof_arm/joint1_position_controller/command',
                '/three_dof_arm/joint2_position_controller/command'                
            ]

class main(object):

    def __init__(self, step, path):
        self.linkThreads = []
        self.mutex = threading.Condition()

        for i in range(0, len(pubList)):
            self.linkThreads.append(topicCartesianState(pubList[i], PUBLISHER_RATE_SLEEP, self.mutex, False))
                    
            # init node
            rospy.init_node('cartesianService', anonymous = True)  
       
        for i in range(0, len(pubList)):
            self.linkThreads[i].start()

        self.endEffectorService = serviceCartesianState('/gazebo/get_link_state', "three_dof_arm::end_effector", "world", "link")

        self.defineKinematics()

        # theta1 = 0, theta2 = pi/4, theta3 = pi/2
        # self.theta1 = 0.0
        # self.theta2 = math.pi/4
        # self.theta3 = math.pi/2
        # endEffectorPositionInitial = np.array([1.41421356237656, 0.0, 2.0])
        self.theta1 = 0.0
        self.theta2 = 0
        self.theta3 = math.pi/2
        endEffectorPositionInitial = np.array([1.0, 0.0, 3.0])

        self.nameFile = path+str(step)

        self.trajectory, _ = self.spiral(step, 0.5, endEffectorPositionInitial, 2)
        #print(self.trajectory)
    
    def defineKinematics(self):
        self.t1 = Symbol("t1")
        self.t2 = Symbol("t2")
        self.t3 = Symbol("t3")
        l1 = Symbol("l1")
        l2 = Symbol("l2")
        l3 = Symbol("l3")

        self.x = cos(self.t1)*cos(self.t2)*sin(self.t3)*l3 + cos(self.t1)*sin(self.t2)*cos(self.t3)*l3 + cos(self.t1)*sin(self.t2)*l2
        self.y = sin(self.t1)*cos(self.t2)*sin(self.t3)*l3 + sin(self.t1)*sin(self.t2)*cos(self.t3)*l3 + sin(self.t1)*sin(self.t2)*l2
        self.z = -sin(self.t2)*sin(self.t3)*l3 + cos(self.t2)*cos(self.t3)*l3 + cos(self.t2)*l2 + l1

        self.x = self.x.subs(l1, 2.0)
        self.x = self.x.subs(l2, 1.0)
        self.x = self.x.subs(l3, 1.0)
        self.y = self.y.subs(l1, 2.0)
        self.y = self.y.subs(l2, 1.0)
        self.y = self.y.subs(l3, 1.0)
        self.z = self.z.subs(l1, 2.0)
        self.z = self.z.subs(l2, 1.0)
        self.z = self.z.subs(l3, 1.0)

        self.a00 = diff( self.x , self.t1)
        self.a01 = diff( self.x , self.t2)
        self.a02 = diff( self.x , self.t3)

        self.a10 = diff( self.y , self.t1)
        self.a11 = diff( self.y , self.t2)
        self.a12 = diff( self.y , self.t3)

        self.a20 = diff( self.z , self.t1)
        self.a21 = diff( self.z , self.t2)
        self.a22 = diff( self.z , self.t3)      

    def change(self, a, t1_number, t2_number, t3_number):
        a = a.subs(self.t1, t1_number)
        a = a.subs(self.t2, t2_number)
        a = a.subs(self.t3, t3_number)
        
        return a

    def spiral(self, stepSize, radius, endEffectorPosition, numberSpirals):
        endEffector = np.zeros(shape=(int(2*math.pi*radius/stepSize)*numberSpirals,3))
        #print(endEffector.shape)
        #print(endEffectorPosition.shape)
        endEffector[:] = endEffectorPosition
            
        angle = math.asin(stepSize/(radius))
        
        for i in range(0, endEffector.shape[0]):
            endEffector[i][2] = endEffector[i][2] + radius*math.sin(angle*(i+1))      
            endEffector[i][0] = endEffector[i][0] + radius*math.cos(angle*(i+1)) - radius
            endEffector[i][1] = endEffector[i][1] + radius*stepSize*(i+1)/3 
            
            if i > 0:
                dx = (endEffector[i][0] - endEffector[i-1][0])
                dy = (endEffector[i][1] - endEffector[i-1][1])
                dz = (endEffector[i][2] - endEffector[i-1][2])
                d =  ( dx**2 + dy**2 + dz**2 )**(0.5)
                #print endEffector.shape[0], stepSize, dx, dy, dz, d
        
        return endEffector, d

    def writeToFileDelta(self, delta_x, delta_y, delta_z):
        file = open(self.nameFile+".txt","a")

        #string = str(delta_x)+','+str(delta_y)+','+str(delta_z)
        string = str((delta_x**2+delta_y**2+delta_z**2))
        file.write(str(string+'\n'))
        file.close()

    def jacobianExperiment(self):
        self.angles = np.zeros(shape=(self.trajectory.shape[0], self.trajectory.shape[1]))
        i = 0
        while(i < self.trajectory.shape[0]-1):
            self.defineKinematics()

            self.a00 = self.change(self.a00, self.theta1, self.theta2, self.theta3)
            self.a01 = self.change(self.a01, self.theta1, self.theta2, self.theta3)
            self.a02 = self.change(self.a02, self.theta1, self.theta2, self.theta3)
            self.a10 = self.change(self.a10, self.theta1, self.theta2, self.theta3)
            self.a11 = self.change(self.a11, self.theta1, self.theta2, self.theta3)
            self.a12 = self.change(self.a12, self.theta1, self.theta2, self.theta3)
            self.a20 = self.change(self.a20, self.theta1, self.theta2, self.theta3)
            self.a21 = self.change(self.a21, self.theta1, self.theta2, self.theta3)
            self.a22 = self.change(self.a22, self.theta1, self.theta2, self.theta3)

            x = self.change(self.x, self.theta1, self.theta2, self.theta3)
            y = self.change(self.y, self.theta1, self.theta2, self.theta3)
            z = self.change(self.z, self.theta1, self.theta2, self.theta3)
            
            if i > 0:
                self.writeToFileDelta(self.trajectory[i][0]-x, self.trajectory[i][1]-y, self.trajectory[i][2]-z)
            
            delta_end = np.array([  [self.trajectory[i+1][0]-x], \
                                    [self.trajectory[i+1][1]-y], \
                                    [self.trajectory[i+1][2]-z]
                                ])
            #print("Delta x: "+str(delta_end[0])+"Delta y: "+str(delta_end[1])+"Delta z: "+str(delta_end[2]))

            self.j = np.matrix([[self.a00, self.a01, self.a02], \
                                [self.a10, self.a11, self.a12], \
                                [self.a20, self.a21, self.a22]], dtype='float')

            self.j_inv = np.linalg.inv(self.j)

            delta_angles = np.dot(self.j_inv, delta_end)

            self.theta1 += delta_angles[0]
            self.theta2 += delta_angles[1]
            self.theta3 += delta_angles[2]

            self.angles[i][0] = self.theta1
            self.angles[i][1] = self.theta2
            self.angles[i][2] = self.theta3

            #print(self.angles[i])

            i += 1

    def inverseKinematicsFunction(self, angles, delta_end_effector):
        t1 = Symbol("t1")
        t2 = Symbol("t2")
        t3 = Symbol("t3")
        l1 = Symbol("l1")
        l2 = Symbol("l2")
        l3 = Symbol("l3")

        x = cos(t1)*cos(t2)*sin(t3)*l3 + cos(t1)*sin(t2)*cos(t3)*l3 + cos(t1)*sin(t2)*l2
        y = sin(t1)*cos(t2)*sin(t3)*l3 + sin(t1)*sin(t2)*cos(t3)*l3 + sin(t1)*sin(t2)*l2
        z = -sin(t2)*sin(t3)*l3 + cos(t2)*cos(t3)*l3 + cos(t2)*l2 + l1

        x = x.subs(l1, 2.0)
        x = x.subs(l2, 1.0)
        x = x.subs(l3, 1.0)
        y = y.subs(l1, 2.0)
        y = y.subs(l2, 1.0)
        y = y.subs(l3, 1.0)
        z = z.subs(l1, 2.0)
        z = z.subs(l2, 1.0)
        z = z.subs(l3, 1.0)

        a00 = diff( x , t1)
        a01 = diff( x , t2)
        a02 = diff( x , t3)

        a10 = diff( y , t1)
        a11 = diff( y , t2)
        a12 = diff( y , t3)

        a20 = diff( z , t1)
        a21 = diff( z , t2)
        a22 = diff( z , t3)

        print(angles)

        a00 = a00.subs(t1, angles[0])
        a00 = a00.subs(t2, angles[1])
        a00 = a00.subs(t3, angles[2])
        a01 = a01.subs(t1, angles[0])
        a01 = a01.subs(t2, angles[1])
        a01 = a01.subs(t3, angles[2])
        a02 = a02.subs(t1, angles[0])
        a02 = a02.subs(t2, angles[1])
        a02 = a02.subs(t3, angles[2])
        a10 = a10.subs(t1, angles[0])
        a10 = a10.subs(t2, angles[1])
        a10 = a10.subs(t3, angles[2])
        a11 = a11.subs(t1, angles[0])
        a11 = a11.subs(t2, angles[1])
        a11 = a11.subs(t3, angles[2])
        a12 = a12.subs(t1, angles[0])
        a12 = a12.subs(t2, angles[1])
        a12 = a12.subs(t3, angles[2])
        a20 = a20.subs(t1, angles[0])
        a20 = a20.subs(t2, angles[1])
        a20 = a20.subs(t3, angles[2])
        a21 = a21.subs(t1, angles[0])
        a21 = a21.subs(t2, angles[1])
        a21 = a21.subs(t3, angles[2])
        a22 = a22.subs(t1, angles[0])
        a22 = a22.subs(t2, angles[1])
        a22 = a22.subs(t3, angles[2])        

        j = np.array([ [a00, a01, a02], \
                        [a10, a11, a12], \
                        [a20, a21, a22]
                      ], dtype="float")        

        j_inv = np.linalg.inv(j)

        delta_angles = j_inv.dot(delta_end_effector)        

        new_angles = angles + delta_angles        

        x = x.subs(t1, new_angles[0])
        x = x.subs(t2, new_angles[1])
        x = x.subs(t3, new_angles[2])
        y = y.subs(t1, new_angles[0])
        y = y.subs(t2, new_angles[1])
        y = y.subs(t3, new_angles[2])
        z = z.subs(t1, new_angles[0])
        z = z.subs(t2, new_angles[1])
        z = z.subs(t3, new_angles[2]) 
       
        return new_angles, float(x), float(y), float(z)

    def jacobianExercise(self):
        self.angles = np.zeros(shape=(self.trajectory.shape[0], self.trajectory.shape[1]))
        i = 0
        x = self.trajectory[i][0]
        y = self.trajectory[i][1]
        z = self.trajectory[i][2]

        self.angles[i][0] = self.theta1
        self.angles[i][1] = self.theta2
        self.angles[i][2] = self.theta3

        while(i < self.trajectory.shape[0]-1):            
            
            delta_end = np.array([  self.trajectory[i+1][0]-x,
                                    self.trajectory[i+1][1]-y,
                                    self.trajectory[i+1][2]-z
                                ])

            #delta_end = np.array([0.0, 0, 0.1])
            
            self.angles[i+1], x, y, z = self.inverseKinematicsFunction(self.angles[i], delta_end)
            

            # self.theta1 += delta_angles[0]
            # self.theta2 += delta_angles[1]
            # self.theta3 += delta_angles[2]

            # self.angles[i][0] = self.theta1
            # self.angles[i][1] = self.theta2
            # self.angles[i][2] = self.theta3

            #print(self.angles[i])

            i += 1

    def netExperiment(self, model):
        self.angles = np.zeros(shape=(self.trajectory.shape[0], 3, 1))
        self.angles[:,0] = self.theta1
        self.angles[:,1] = self.theta2
        self.angles[:,2] = self.theta3
        # print(self.angles)

        i = 0
                
        # # full
        # highestList = np.array([    0.05, 0.05, 0.05, 3.14159265359, 3.14159265359, 3.1414962357793392, 0.1745329251, 0.1745329251, 0.1745329251])
        # lowestList = np.array([     -0.05, -0.05, -0.05, -3.14159265359, -3.14159265359, -3.14159265359, -0.1745329251, -0.1745329251, -0.1745329251])

        
        highestList = np.array([0.4999992017079372, 0.4999997022129006, 0.4999990498591971, 3.1414988992338606, 3.1414986177704454, 3.1414997536842444, 0.7853903970228057, 0.7853519733451677, 0.785392912463341])
        lowestList = np.array([-0.4999998315796146, -0.49999960704497365, -0.4999968060261315, -3.1414980107000616, -3.1414989635909167, -3.141498782113911, -0.785345060166046, -0.7853562961740115, -0.785304829597736])       
        
        while(i < self.trajectory.shape[0]-1):
            # self.defineKinematics()            

            # x = self.change(self.x, self.angles[i][0], self.angles[i][1], self.angles[i][2])
            # y = self.change(self.y, self.angles[i][0], self.angles[i][1], self.angles[i][2])
            # z = self.change(self.z, self.angles[i][0], self.angles[i][1], self.angles[i][2])

            l1 = 2
            l2 = 1
            l3 = 1

            self.theta1 = self.angles[i][0]
            self.theta2 = self.angles[i][1]
            self.theta3 = self.angles[i][2]
            
            x = math.cos(self.theta1)*math.cos(self.theta2)*math.sin(self.theta3)*l3 + math.cos(self.theta1)*math.sin(self.theta2)*math.cos(self.theta3)*l3 + math.cos(self.theta1)*math.sin(self.theta2)*l2
            y = math.sin(self.theta1)*math.cos(self.theta2)*math.sin(self.theta3)*l3 + math.sin(self.theta1)*math.sin(self.theta2)*math.cos(self.theta3)*l3 + math.sin(self.theta1)*math.sin(self.theta2)*l2
            z = -math.sin(self.theta2)*math.sin(self.theta3)*l3 + math.cos(self.theta2)*math.cos(self.theta3)*l3 + math.cos(self.theta2)*l2 + l1

            if i > 0:
                self.writeToFileDelta(self.trajectory[i][0]-x, self.trajectory[i][1]-y, self.trajectory[i][2]-z)
            
            predictArray = np.array([   self.trajectory[i+1][0] - x,
                                        self.trajectory[i+1][1] - y,
                                        self.trajectory[i+1][2] - z,                                                                    
                                        self.angles[i][0],
                                        self.angles[i][1],
                                        self.angles[i][2]                            
                                    ])       
            #print(self.trajectory[i])
            # normalize the predict array
            predictArray = normalizeArray(predictArray, highestList[:6], lowestList[:6])
            # the net output
            deltaAnglesPredicted = (np.array(model.predict(predictArray.reshape(1,6)))[0])

            # desnormalize the predicted angles
            deltaAnglesPredicted = desnormalizeArray(deltaAnglesPredicted, highestList[6:], lowestList[6:])
            #print("Net")
            # the new angle plus the delta angles
            #print(deltaAnglesPredicted)
            self.angles[i+1] = deltaAnglesPredicted.reshape(3,1) + self.angles[i]

            #print(self.angles)

            i += 1

            #time.sleep(1)

    def writeToFile(self, deltaCartesian, angles, deltaAngles, writer):    
        aux = np.concatenate((deltaCartesian, angles), axis=0)
        data = np.concatenate((aux, deltaAngles), axis=0)

        writer.writerow(data.tolist())

    def generateData(self, numberSamples, maxDegree, maxDeltaEnd):

        File = "dataset_normal"+str(maxDegree)+"_"+str(maxDeltaEnd)
        datacsvfile = open(File+".csv", 'a')    
        datacsvwriter = csv.writer(datacsvfile)
        
        i = 0
        count = 0
        
        while(i < numberSamples):
            self.theta1 = np.random.uniform(-3.1415, 3.1415)
            self.theta2 = np.random.uniform(-3.1415, 3.1415)
            self.theta3 = np.random.uniform(-3.1415, 3.1415)            
            #self.theta1 = 0.0
            #self.theta2 = math.pi/4
            #self.theta3 = math.pi/2

            #print("t1 "+str(self.theta1))
            #print("t2 "+str(self.theta2))
            #print("t3 "+str(self.theta3))
            
            # self.defineKinematics()

            l1 = 2
            l2 = 1
            l3 = 1
            
            self.x = math.cos(self.theta1)*math.cos(self.theta2)*math.sin(self.theta3)*l3 + math.cos(self.theta1)*math.sin(self.theta2)*math.cos(self.theta3)*l3 + math.cos(self.theta1)*math.sin(self.theta2)*l2
            self.y = math.sin(self.theta1)*math.cos(self.theta2)*math.sin(self.theta3)*l3 + math.sin(self.theta1)*math.sin(self.theta2)*math.cos(self.theta3)*l3 + math.sin(self.theta1)*math.sin(self.theta2)*l2
            self.z = -math.sin(self.theta2)*math.sin(self.theta3)*l3 + math.cos(self.theta2)*math.cos(self.theta3)*l3 + math.cos(self.theta2)*l2 + l1


            # self.x = self.change(self.x, self.theta1, self.theta2, self.theta3)
            # self.y = self.change(self.y, self.theta1, self.theta2, self.theta3)
            # self.z = self.change(self.z, self.theta1, self.theta2, self.theta3)

            old_x = self.x
            old_y = self.y
            old_z = self.z

            #print("x "+str(old_x))
            #print("y "+str(old_y))
            #print("z "+str(old_z))

            delta_theta1 = np.radians(maxDegree)*np.random.normal(0, 0.33)
            delta_theta2 = np.radians(maxDegree)*np.random.normal(0, 0.33)
            delta_theta3 = np.radians(maxDegree)*np.random.normal(0, 0.33)

            #print("delta_t1 "+str(delta_theta1))
            #print("delta_t2 "+str(delta_theta2))
            #print("delta_t3 "+str(delta_theta3))

            if (abs(delta_theta1) < np.radians(maxDegree) and abs(delta_theta2) < np.radians(maxDegree) and abs(delta_theta3) < np.radians(maxDegree)):
                
                self.theta1 += delta_theta1
                self.theta2 += delta_theta2
                self.theta3 += delta_theta3

                if(abs(self.theta1) < 3.1415 and abs(self.theta2) < 3.1415 and abs(self.theta3) < 3.1415):

                    # self.defineKinematics()

                    # self.x = self.change(self.x, self.theta1, self.theta2, self.theta3)
                    # self.y = self.change(self.y, self.theta1, self.theta2, self.theta3)
                    # self.z = self.change(self.z, self.theta1, self.theta2, self.theta3)

                    self.x = math.cos(self.theta1)*math.cos(self.theta2)*math.sin(self.theta3)*l3 + math.cos(self.theta1)*math.sin(self.theta2)*math.cos(self.theta3)*l3 + math.cos(self.theta1)*math.sin(self.theta2)*l2
                    self.y = math.sin(self.theta1)*math.cos(self.theta2)*math.sin(self.theta3)*l3 + math.sin(self.theta1)*math.sin(self.theta2)*math.cos(self.theta3)*l3 + math.sin(self.theta1)*math.sin(self.theta2)*l2
                    self.z = -math.sin(self.theta2)*math.sin(self.theta3)*l3 + math.cos(self.theta2)*math.cos(self.theta3)*l3 + math.cos(self.theta2)*l2 + l1

                    #print("x "+str(self.x))
                    #print("y "+str(self.y))
                    #print("z "+str(self.z))

                    delta_x = self.x-old_x
                    delta_y = self.y-old_y
                    delta_z = self.z-old_z

                    #print("delta_x "+str(delta_x))
                    #print("delta_y "+str(delta_y))
                    #print("delta_z "+str(delta_z))

                    if(abs(delta_x) < maxDeltaEnd and abs(delta_y) < maxDeltaEnd and abs(delta_z) < maxDeltaEnd):
                        if(abs(delta_x) > maxDeltaEnd/10 and abs(delta_y) > maxDeltaEnd/10 and abs(delta_z) > maxDeltaEnd/10):
                            datacsvwriter.writerow([delta_x, delta_y, delta_z, self.theta1, self.theta2, self.theta3, delta_theta1, delta_theta2, delta_theta3])

                            count += 1

            i += 1

            if count % 1000 == 0:
                print("Total: "+str(count)+" Valid: "+str(float(count)/i))

    def plot(self, pathnet, pathjac):
        filesNet = sorted(glob.glob(pathnet+"/*.txt"))
        filesJac = sorted(glob.glob(pathjac+"/*.txt"))
        mseNet = np.zeros(shape=(len(filesNet),))
        mseJac = np.zeros(shape=(len(filesJac),))

        for i in range(0, len(filesNet)):
            with open(filesNet[i]) as f:
                lines = f.readlines()
                for j in range(0, len(lines)):
                    #print(float(lines[j][:-1]))
                    mseNet[i] += float(lines[j][:-1])/float(len(lines))

        for i in range(0, len(filesJac)):
            #print(filesJac)
            with open(filesJac[i]) as f:
                lines = f.readlines()
                for j in range(0, len(lines)):
                    #print(float(lines[j][:-1]))
                    mseJac[i] += float(lines[j][:-1])/float(len(lines))

        
            
        #mpl.rcParams['legend.fontsize'] = 10
        print(mseJac)

        fig = plt.figure()
        x = (range(0, mseJac.shape[0]))
        y = mseJac
        print(len(x))
        print(y.shape[0])

        for i in range(0, len(x)):
            x[i] = float(x[i])/2000 + 0.1005

        plt.plot(x, y, label='mseJac')
        plt.legend()
        
        x = (range(0, mseNet.shape[0]))
        
        for i in range(0, len(x)):
            x[i] = float(x[i])/2000 + 0.1005
        y = mseNet
        plt.plot(x, y, label='mseNet')
        plt.legend()

        plt.show()

    def runOnSimulation(self):
        count = 0
        spawnCount = 0
        while count < self.trajectory.shape[0]:    
            self.mutex.acquire()
            time.sleep(0.1)
            publishersFlag = True
            if count == 1:
                time.sleep(5)
            for i in range(0, len(self.linkThreads)):
                if self.linkThreads[i].flag == True:
                    publishersFlag = False
                    break    
            
            if publishersFlag == True:
                self.endEffectorService.getState()

                #only for Gazebo
                if (count != 0):
                    self.endEffectorService.spawnObject(spawnCount, self.endEffectorService.pos_x, \
                                                self.endEffectorService.pos_y, self.endEffectorService.pos_z)

                spawnCount += 1
                
                self.linkThreads[0].setValue(self.angles[count][0])
                self.linkThreads[1].setValue(self.angles[count][1])
                self.linkThreads[2].setValue(self.angles[count][2])

                count += 1
            
                for i in range(0, len(self.linkThreads)):                                        
                    self.linkThreads[i].setFlag()
                
                self.mutex.notify_all()  

            else:
                self.mutex.wait() 
            
            self.mutex.release()

        for i in range(0, len(linkList)):
            self.linkThreads[i].join()

    def histogramDataset(self, path, max_step):        
        #dataset = np.genfromtxt(path, delimiter=',')
        with open(path) as inFile:
            step_x = list(np.linspace(0, max_step, 200, endpoint=True))
            del(step_x[0])
            step_y = [0 for i in step_x]
            count = 0
            
            reader = csv.reader(inFile)

            for line in reader:
                for i, interval in enumerate(step_x):
                    d = (float(line[0])**2 + float(line[1])**2 + float(line[2])**2)**0.5
                    if d < interval:
                        step_y[i] += 1
                        break
                count+=1
                if count%100 == 0:
                    print(count)
                    if count == 500000:
                        break
            
        print (step_y)
        plt.figure()
        plt.plot(step_x, step_y)
        plt.legend([r'$Step$'])
        #plt.savefig("graphs/{}_step.png".format(dataset_name))
        plt.show()
# i = 0
# step = 0.1

# stepsize = 0.0005
#model = load_model("/home/ricardo/catkin_ws/src/Three_dof_arm/kinematics/model__80_8192_Adam_sigmoid_128_92_64_0.0014767418555882677_0.0019274018703649442_0.00046107815922dataset.csv.h5")

# while(step < 0.5):
#     #m = main(step, "jacobian/")
#     #m.jacobianExperiment()
#     m = main(step, "net/", )
#     m.netExperiment(model)

#     step += stepsize
#     #m.runOnSimulation()

#     print(step)

m = main(0.05, "net/")
#m.jacobianExperiment()
m.jacobianExercise()
#m.netExperiment(model)
m.runOnSimulation()
#m.plot("net/", "jacobian/")
# m.generateData(10000000, 45, 0.5)
# m.histogramDataset("dataset_normal45_0.5.csv", 0.5)