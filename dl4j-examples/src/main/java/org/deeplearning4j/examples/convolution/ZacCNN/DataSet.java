package org.deeplearning4j.examples.convolution.ZacCNN;

import java.util.HashMap;
import java.util.Map;

public class DataSet {

    private static Map<DataType, Config> dataMap = new HashMap<>();

    static {
        // size: 160M done: good
        dataMap.put(DataType.HAR, new Config.Builder()
                                      .setDataPath("/Users/zber/Desktop/1_Data/Har/demo_train.csv")
                                      .setTestPath("/Users/zber/Desktop/1_Data/Har/demo_test.csv")
                                      .setTaskNum(480)
                                      .setChannel(9)
                                      .setWidth(128)
                                      .setNumClass(6)
                                      .setBatch(16)
                                      .build());
        // size: 100M done: good
        dataMap.put(DataType.MHe, new Config.Builder()
                                      .setDataPath("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Final_Data/og_shuffle/mHealth/train.csv")
                                      .setTestPath("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Final_Data/og_shuffle/mHealth/test.csv")
                                      .setTaskNum(2349)
                                      .setChannel(23)
                                      .setWidth(100)
                                      .setNumClass(12)
                                      .setKernel(50)
                                      .isNormal(true)
                                      .setBatch(16)
                                      .build());
        // size: 900M done: good
        dataMap.put(DataType.PAMA, new Config.Builder()
                                       .setDataPath("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Final_Data/og_shuffle/Pama2/7class/train.csv")
                                       .setTestPath("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Final_Data/og_shuffle/Pama2/7class/test.csv")
                                       .setTaskNum(11397)
                                       .setChannel(9)
                                       .setWidth(512)
                                       .setNumClass(7)
                                       .isNormal(true)
                                       .setBatch(16)
                                       .build());
        // size: 100M done has issue: test label and ac
//        dataMap.put(DataType.PAMA, new Config.Builder()
//                                       .setDataPath("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Final_Data/og_shuffle/Pama2/12class/og_train.csv")
//                                       .setTestPath("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Final_Data/og_shuffle/Pama2/12class/og_test.csv")
//                                       .setTaskNum(4244)
//                                       .setChannel(52)
//                                       .setWidth(25)
//                                       .setNumClass(12)
//                                       .isNormal(true)
//                                       .setBatch(16)
//                                       .setKernel(12)
//                                       .setf1Out(500)
//                                       .build());
        // size: 517M done has issue: test label from -1 to 17 and has 19 class!?
        dataMap.put(DataType.OP, new Config.Builder()
                                     .setDataPath("/Users/zber/Desktop/1_Data/data/OpportunityUCIDataset/train.csv")
                                     .setTestPath("/Users/zber/Desktop/1_Data/data/OpportunityUCIDataset/test.csv")
                                     .setTaskNum(14034)
                                     .setChannel(77)
                                     .setWidth(25)
                                     .setNumClass(11)
                                     .setBatch(16)
                                     .isNormal(true)
                                     .setKernel(10)
                                     .build());
        // size:58m
        dataMap.put(DataType.AAL, new Config.Builder()
            .setDataPath("/Users/zber/Desktop/1_Data/dataset_uci/train.csv")
            .setTestPath("/Users/zber/Desktop/1_Data/dataset_uci/test.csv")
            .setTaskNum(4252)
            .setChannel(1)
            .setWidth(561)
            .setNumClass(6)
            .setBatch(16)
            .setKernel(50)
            .isNormal(true)
            //.setEpoch(40)
            .build());
        dataMap.put(DataType.EMG, new Config.Builder()
            .setDataPath("/Users/zber/Desktop/1_Data/Variety_Experiment/EMG_data_for_gestures-master/train.csv")
            .setTestPath("/Users/zber/Desktop/1_Data/Variety_Experiment/EMG_data_for_gestures-master/test.csv")
            .setTaskNum(8573)
            .setChannel(8)
            .setWidth(20)
            .setNumClass(6)
            .setBatch(16)
            .setKernel(8)
            .isNormal(true)
            .setEpoch(20)
            .build());
        dataMap.put(DataType.EEG, new Config.Builder()
            .setDataPath("/Users/zber/Desktop/1_Data/Variety_Experiment/Eye_state/train.csv")
            .setTestPath("/Users/zber/Desktop/1_Data/Variety_Experiment/Eye_state/test.csv")
            .setTaskNum(1047)
            .setChannel(14)
            .setWidth(25)
            .setNumClass(2)
            .setBatch(16)
            .setKernel(7)
            .isNormal(true)
            .setEpoch(50)
            .build());
        dataMap.put(DataType.EE, new Config.Builder()
            .setDataPath("/Users/zber/Desktop/1_Data/Variety_Experiment/EMG_PhysicalActionDataSet/train.csv")
            .setTestPath("/Users/zber/Desktop/1_Data/Variety_Experiment/EMG_PhysicalActionDataSet/test.csv")
            .setTaskNum(14005)
            .setChannel(8)
            .setWidth(50)
            .setNumClass(10)
            .setBatch(16)
            .setKernel(20)
            .isNormal(true)
            .setEpoch(2) // test it with less epoch
            .build());
        dataMap.put(DataType.EES, new Config.Builder()
            .setDataPath("/Users/zber/Desktop/1_Data/Variety_Experiment/EEG-Eye-State-Prediction/train.csv")
            .setTestPath("/Users/zber/Desktop/1_Data/Variety_Experiment/EEG-Eye-State-Prediction/test.csv")
            .setTaskNum(13904)
            .setChannel(1)
            .setWidth(14)
            .setNumClass(2)
            .setBatch(16)
            .setKernel(7)
            .isNormal(true)
            .setEpoch(5)
            .build());
        dataMap.put(DataType.FALL, new Config.Builder()
            .setDataPath("/Users/zber/Desktop/1_Data/Variety_Experiment/Fall_Detection/train.csv")
            .setTestPath("/Users/zber/Desktop/1_Data/Variety_Experiment/Fall_Detection/test.csv")
            .setTaskNum(10921)
            .setChannel(1)
            .setWidth(604)
            .setNumClass(17)
            .setBatch(16)
            .setKernel(40)
            .isNormal(true)
            .setEpoch(20)
            .build());
        dataMap.put(DataType.SIS, new Config.Builder()
            .setDataPath("/Users/zber/Desktop/1_Data/Variety_Experiment/Fall_Detection/SVD_applied_fall_detection/sisfall/data/train.csv")
            .setTestPath("/Users/zber/Desktop/1_Data/Variety_Experiment/Fall_Detection/SVD_applied_fall_detection/sisfall/data/test.csv")
            .setTaskNum(3390)
            .setChannel(1)
            .setWidth(1350)
            .setNumClass(34)
            .setBatch(16)
            .setKernel(32)
            .isNormal(true)
            .setEpoch(20)
            .build());
    }

    public static Config getNewConfig(DataType dataType) {
        return dataMap.get(dataType).clone();
    }

    public static Config getConfig(DataType dataType) {
        return dataMap.get(dataType);
    }
}
