# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 19:49:46 2023

@author: Xiaofei Zhang
@email:csezxf@163.com
# """
import numpy as np
from scipy import stats
import pdb 
import math
import os
import scipy.io as scio
import pandas as pd
import h5py
#######################################Cut Data##################################%
def fNIRS_cut(Low_fNIRS, Hight_fNIRS):
    [nub_scenario, nub_channel, nub_point] = Low_fNIRS.shape
    Low_groups_all = []
    High_groups_all = []
    step = 100
    for i in range(0, nub_point - 300, step):
        Low_groups = Low_fNIRS[:, :, i:i + 400]
        High_groups = Hight_fNIRS[:, :, i:i + 400]
        Low_groups_all.append(Low_groups)
        High_groups_all.append(High_groups)
    Low_groups_all = np.array(Low_groups_all)
    High_groups_all=np.array(High_groups_all)
    Low_groups_all = np.concatenate(Low_groups_all, axis=0)
    [a_line, b_line, c_line] = Low_groups_all.shape
    High_groups_all = np.concatenate(High_groups_all, axis=0)
    DATA_cut = []
    DATA_cut = np.concatenate((Low_groups_all, High_groups_all), axis=0)

    return DATA_cut,a_line

#######################################t test##################################%
def t_test(TH_Hight_DATA, TH_Low_DATA, NUM_CHANNEL,nub_scenario):
    line=15
    fNIRS_MeanValue = np.zeros([nub_scenario, NUM_CHANNEL * 2])
    for i_channel in range(NUM_CHANNEL):
        i_all = 0
        for i_nub_sample in range(nub_scenario):
            # fNIRS_MeanValue=np.zeros([nub_scenario,NUM_CHANNEL*2])
            # print(i_nub_sample)
            # fNIRS_MeanValue=np.zeros([i_all,NUM_CHANNEL*2,nub_point])
            fNIRS_Hight_MeanValue_Feature = np.mean(TH_Hight_DATA[i_nub_sample, i_channel, :])
            TH_Hight_DATA_one = TH_Hight_DATA[1, :, :]
            TH_Low_DATA_one = TH_Low_DATA[1, :, :]
            # print(fNIRS_Hight_MeanValue_Feature)
            fNIRS_Low_MeanValue_Feature = np.mean(TH_Low_DATA[i_nub_sample, i_channel, :])
            fNIRS_MeanValue[i_nub_sample, i_channel * 2] = fNIRS_Hight_MeanValue_Feature
            fNIRS_MeanValue[i_nub_sample, i_channel * 2 + 1] = fNIRS_Low_MeanValue_Feature
    del TH_Low_DATA, TH_Hight_DATA
    # [x_t_test_all,y_t_test_all]=fNIRS_MeanValue.shape
    # y_t_test=y_t_test_all/2  
    t_value_all = []
    for i_t_test in range(line):
        # print(fNIRS_MeanValue.shape)
        t_value, pvalue = stats.ttest_rel(fNIRS_MeanValue[:, i_t_test], fNIRS_MeanValue[:, i_t_test + 1])
        # t_value=np.array(t_value)
        # print(t_value)
        # t_value=c._getattribute_("statistic")
        t_value_all.append(t_value)
    t_value_all_array = np.array(t_value_all)
    # print(c_all)
    t_value_all_array = abs(t_value_all_array)
    t_value_all_array = t_value_all_array[0:15:2]
    return t_value_all_array
#######################################make a new data ##################################%
def fNIRS_t_test(fNIRS_cut,Low_fNIRS,Hight_fNIRS, NUM_CHANNEL, nub_scenario):
    fNIRS_xa = []
    [a, d, c] = fNIRS_cut.shape
    D = t_test(Low_fNIRS, Hight_fNIRS, NUM_CHANNEL, nub_scenario)
    # pdb.set_trace()
    for i_T in range(a):
        xi = fNIRS_cut[i_T, :, :]
        xi = xi.reshape(8, 400)    
        # print(D)
        # print(D.shape)
        D=np.array([1,1,1,1,1,1,1,1])
        F = np.empty(shape=[0, 400])  # 创建空矩阵
        m = 0  # 利用矩阵索引取矩阵每一行元素，初值为0
        for i in range(len(D)):
            f = xi[m, :] * D[m]
            F = np.vstack((F, f))
            m = m + 1
        fNIRS_xa.append(F)
    fNIRS_xa = np.array(fNIRS_xa)
    return fNIRS_xa,a
###############################################################################
def get_event_data(name_event, x,y):
    DataSetfNIRSOneevnt={name_event:{'Data_X':x,'Data_Y':y}}
    del name_event,x,y
    return DataSetfNIRSOneevnt
##################################################################################
def obtain_data(i_event):
    NUM_CHANNEL = 8
    line=15
    #########################obtain data position######################
    Current_Data = os.getcwd()
    Position_Data = Current_Data + '\Data\\Homer3'
    All_Event_Type = ['EmergentAEB', 'left_cut_in', 'right_cut_in','pedestrian_right']
    Num_Event = len(All_Event_Type)
    DataSetfNIRS={}
    # i_event='EmergentAEB'
    # for i_event in All_Event_Type:  #####load each event data
    print('It is dealing with the data of', i_event)
    Event_Position_Data = Position_Data + '\\' + i_event
    Name_Event_People = os.listdir(Event_Position_Data)
    Num_Event_People = len(Name_Event_People)
    Low_fNIRS_All = []
    Hight_fNIRS_All = []
    Low_Risk_Field_All = []
    Hight_Risk_Field_All = []
    Num_All_Sample = 0

    #######################load data################################%
    for i_people in Name_Event_People:  #####load each people data
        People_Position_Data_EmergentAEB = Event_Position_Data + '\\' + i_people
        # print('################################################')
        # print('It is dealing with the data of', People_Position_Data_EmergentAEB)
        ##############################define data#####################%
        People_Data = scio.loadmat(People_Position_Data_EmergentAEB)
        Effective_data = People_Data['Effective_Data']
        Scene_num = People_Data['Scene_num']
        Scene_Num_Int = Scene_num.astype(int)[0][0] - 1
        [Event, Sample_Num, Column] = Effective_data.shape
        Sample_Low_Num = int(Sample_Num / 2)
        Sample_Low_Range = np.zeros(Sample_Low_Num)
        Sample_Hight_Range = np.zeros(Sample_Low_Num)
        Low_fNIRS = np.zeros([Scene_Num_Int, NUM_CHANNEL, Sample_Low_Num])
        Hight_fNIRS = np.zeros([Scene_Num_Int, NUM_CHANNEL, Sample_Low_Num])
        Low_Risk_Field = np.zeros([Scene_Num_Int, Sample_Low_Num])
        Hight_Risk_Field = np.zeros([Scene_Num_Int, Sample_Low_Num])
        #################################################################################################################
        for i_scenario in range(Scene_Num_Int):  #####load each people data
            for i_sample in range(Sample_Low_Num):
                for i_channl in range(NUM_CHANNEL):
                    # (HBR-HBO)sqrt(2)
                    Low_fNIRS[i_scenario, i_channl, i_sample] = (Effective_data[
                                                                    i_scenario, i_sample, i_channl * 2 + 21] -
                                                                    Effective_data[
                                                                    i_scenario, i_sample, i_channl * 2 + 20]) / math.sqrt(2)
                    Hight_fNIRS[i_scenario, i_channl, i_sample] = (Effective_data[
                                                                    i_scenario, i_sample + Sample_Low_Num, i_channl * 2 + 21] -
                                                                    Effective_data[
                                                                    i_scenario, i_sample + Sample_Low_Num, i_channl * 2 + 20]) / math.sqrt(2)
                Low_Risk_Field[i_scenario, i_sample] = Effective_data[i_scenario, i_sample, 18]  #
                Hight_Risk_Field[i_scenario, i_sample] = Effective_data[i_scenario, i_sample + Sample_Low_Num, 18]
        Low_fNIRS_All.append(Low_fNIRS)
        Hight_fNIRS_All.append(Hight_fNIRS)
        Low_Risk_Field_All.append(Low_Risk_Field)
        Hight_Risk_Field_All.append(Hight_Risk_Field)
        Num_All_Sample = Num_All_Sample + Scene_Num_Int
        ###############################Merge  fNIRS data##############################
    Low_fNIRS_All = np.concatenate(Low_fNIRS_All, axis=0)
    Hight_fNIRS_All = np.concatenate(Hight_fNIRS_All, axis=0)
    Low_fNIRS_All_Array = np.array(Low_fNIRS_All, dtype=object)

    Hight_fNIRS_All_Array = np.array(Hight_fNIRS_All, dtype=object)
    Low_Risk_Field_Array = np.array(Low_Risk_Field_All, dtype=object)
    Hight_Risk_Field_Array = np.array(Hight_Risk_Field_All, dtype=object)
    [nub_scenario, nub_channel, nub_point] = Low_fNIRS_All_Array.shape

    i_all = 0

    ######fNIRS dataset slicing######

    DATA_EmergentAEB, a_line = fNIRS_cut(Low_fNIRS_All_Array, Hight_fNIRS_All_Array)

    ######assign weights######

    xa, a = fNIRS_t_test(DATA_EmergentAEB, Low_fNIRS_All_Array, Hight_fNIRS_All_Array, NUM_CHANNEL, nub_scenario)
    # pdb.set_trace()
    y1 = np.zeros((a_line, 1))
    y2 = np.ones((a_line, 1))
    y = np.concatenate((y1, y2), axis=0)
    # pdb.set_trace()
    DataSetfNIRSOneevnt={'Data_X':xa,'Data_Y':y}
    del Low_fNIRS_All_Array, Hight_fNIRS_All_Array,DATA_EmergentAEB,xa,y
    return DataSetfNIRSOneevnt
def save_reslut_to_text(Result_Text,Event_name,Algorithm_Name,Training_Accuracy,Testing_Accuracy):
    Result='The result of '+ Algorithm_Name+'algorithm' +'for the ' +Event_name+ ':\n '
    file = open(Result_Text, "a")
    file.write(Result)
    file.write('Training_Accuracy is:'+str(Training_Accuracy)+ '\n ')
    file.write('Testing_Accuracy is:' +str(Testing_Accuracy) +'\n ')
    file.close()

##################################################################################
def obtain_data_form_data(i_event):
    NUM_CHANNEL = 8
    line=15
    print('It is dealing with the data of', i_event)
    #########################obtain data position######################
    All_Event_Name={'EmergentAEB':'Scenario7','left_cut_in':'Scenario4','right_cut_in':'Scenario5','pedestrian_right':'Scenario10'}
    ScenarioNub=All_Event_Name[i_event]
    Current_Data = os.getcwd()
    Position_Data = Current_Data + '\\Data\\'
    Scenario_Position_Data=Position_Data+ScenarioNub
    Scenario_Data = scio.loadmat(Scenario_Position_Data)

    # Scene_Num_Int=int(Scenario_Data['Event_Nub'])
    Sample_Nub=Scenario_Data['Sub1'].shape[1]
    Sample_Low_Num = int(Sample_Nub / 2)
    Low_fNIRS_All = []
    Hight_fNIRS_All = []
    Low_Risk_Field_All = []
    Hight_Risk_Field_All = []
    for key, event_value in Scenario_Data.items(): 
        if(key[0:3]=='Sub'):
            Scene_Num_Int=int(event_value.shape[0])
            Low_fNIRS = np.zeros([Scene_Num_Int, NUM_CHANNEL, Sample_Low_Num])
            Hight_fNIRS = np.zeros([Scene_Num_Int, NUM_CHANNEL, Sample_Low_Num])
            Low_Risk_Field = np.zeros([Scene_Num_Int, Sample_Low_Num])
            Hight_Risk_Field = np.zeros([Scene_Num_Int, Sample_Low_Num])
             #################################################################################################################
            for i_scenario in range(event_value.shape[0]):  #####load each people data
                for i_sample in range(Sample_Low_Num):
                    for i_channl in range(NUM_CHANNEL):
                        # (HBR-HBO)sqrt(2)
                        Low_fNIRS[i_scenario, i_channl, i_sample] =  (event_value[
                                                                        i_scenario, i_sample, i_channl * 2 + 14] -
                                                                      event_value[
                                                                        i_scenario, i_sample, i_channl * 2+ 13]) / math.sqrt(2)
                        Hight_fNIRS[i_scenario, i_channl, i_sample] = (event_value[
                                                                        i_scenario, i_sample + Sample_Low_Num, i_channl * 2 + 14] -
                                                                        event_value[i_scenario, i_sample + Sample_Low_Num, i_channl * 2+ 13 ]) / math.sqrt(2)
                    Low_Risk_Field[i_scenario, i_sample] =   event_value[i_scenario, i_sample, 12]  #
                    Hight_Risk_Field[i_scenario, i_sample] = event_value[i_scenario, i_sample + Sample_Low_Num, 12]
            Low_fNIRS_All.append(Low_fNIRS)
            Hight_fNIRS_All.append(Hight_fNIRS)
            Low_Risk_Field_All.append(Low_Risk_Field)
            Hight_Risk_Field_All.append(Hight_Risk_Field)
        ###############################Merge  fNIRS data##############################
    Low_fNIRS_All = np.concatenate(Low_fNIRS_All, axis=0)
    Hight_fNIRS_All = np.concatenate(Hight_fNIRS_All, axis=0)
    Low_fNIRS_All_Array = np.array(Low_fNIRS_All, dtype=object)

    Hight_fNIRS_All_Array = np.array(Hight_fNIRS_All, dtype=object)
    Low_Risk_Field_Array = np.array(Low_Risk_Field_All, dtype=object)
    Hight_Risk_Field_Array = np.array(Hight_Risk_Field_All, dtype=object)
    [nub_scenario, nub_channel, nub_point] = Low_fNIRS_All_Array.shape

    i_all = 0
    ######fNIRS dataset slicing######
    DATA_Sample, a_line = fNIRS_cut(Low_fNIRS_All_Array, Hight_fNIRS_All_Array)
    # pdb.set_trace()
    ######assign weights######

    # xa, a = fNIRS_t_test(DATA_EmergentAEB, Low_fNIRS_All_Array, Hight_fNIRS_All_Array, NUM_CHANNEL, nub_scenario)
    xa=DATA_Sample
    # pdb.set_trace()
    # pdb.set_trace()
    y1 = np.zeros((a_line, 1))
    y2 = np.ones((a_line, 1))
    y = np.concatenate((y1, y2), axis=0)
    # pdb.set_trace()
    DataSetfNIRSOneevnt={'Data_X':xa,'Data_Y':y}
    del Low_fNIRS_All_Array, Hight_fNIRS_All_Array,DATA_Sample,xa,y
    return DataSetfNIRSOneevnt

# if __name__ == '__main__':
#     main()
