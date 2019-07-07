package org.deeplearning4j.examples.convolution.ZacCNN;

import java.util.HashMap;
import java.util.Map;

public class DataSet {

    private static Map<DataType, Config> dataMap = new HashMap<>();

    private static final String basePath = "/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Final_Data/";

    static {
        // size: 160M done: good vs image
        dataMap.put(DataType.HAR, new Config.Builder()
                                      .setDataPath(basePath + "nor_data/Har/nor_train.csv")
                                      .setTestPath(basePath + "nor_data/Har/nor_test.csv")
                                      .setTaskNum(7352)
                                      .setChannel(9)
                                      .setWidth(128)
                                      .setNumClass(6)
                                      .setBatch(16)
                                      .build());
        // size: 100M done: good
        dataMap.put(DataType.MHe, new Config.Builder()
                                      .setDataPath(basePath + "nor_data/mHealth/nor_train.csv")
                                      .setTestPath(basePath + "nor_data/mHealth/nor_test.csv")
                                      .setTaskNum(2339)
                                      .setChannel(23)
                                      .setWidth(100)
                                      .setNumClass(12)
                                      .setKernel(50)
//                                      .isNormal(true)
                                      .setBatch(16)
                                      .build());
        // size: 900M done: good
        dataMap.put(DataType.PAMA, new Config.Builder()
                                       .setDataPath(basePath + "nor_data/Pama2/nor_train.csv")
                                       .setTestPath(basePath + "nor_data/Pama2/nor_train.csv")
                                       .setTaskNum(11397)
                                       .setChannel(9)
                                       .setWidth(512)
                                       .setNumClass(7)
//                                       .isNormal(true)
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
        // size: 500M done: good
        dataMap.put(DataType.OP, new Config.Builder()
                                     .setDataPath(basePath + "nor_data/Op/nor_train.csv")
                                     .setTestPath(basePath + "nor_data/Op/nor_test.csv")
                                     .setTaskNum(14046)
                                     .setChannel(77)
                                     .setWidth(25)
                                     .setNumClass(11)
                                     .setBatch(16)
//                                     .isNormal(true)
                                     .setKernel(10)
                                     .build());

        // size: 25m done: good
        dataMap.put(DataType.EMG, new Config.Builder()
                                      .setDataPath(basePath + "nor_data/EMG/nor_train.csv")
                                      .setTestPath(basePath + "nor_data/EMG/nor_test.csv")
                                      .setTaskNum(8563)
                                      .setChannel(8)
                                      .setWidth(20)
                                      .setNumClass(6)
                                      .setBatch(16)
                                      .setKernel(8)
//                                      .isNormal(true)
                                      .setEpoch(20)
                                      .build());
        // size: 116m done
        dataMap.put(DataType.FALL, new Config.Builder()
                                       .setDataPath(basePath + "nor_data/Fall/nor_train.csv")
                                       .setTestPath(basePath + "nor_data/Fall/nor_test.csv")
                                       .setTaskNum(8448)
                                       .setChannel(1)
                                       .setWidth(604)
                                       .setNumClass(11)
                                       .setBatch(16)
                                       .setKernel(42)
//                                       .isNormal(true)
                                       .setEpoch(20)
                                       .build());


        // -----------------------------------

        dataMap.put(DataType.TEST, new Config.Builder()
                                       .setDataPath(basePath + "test/nor_train_t.csv")
                                       .setTestPath(basePath + "test/nor_test_t.csv")
                                       .setTaskNum(120)
                                       .setChannel(9)
                                       .setWidth(128)
//                                       .isNormal(true)
                                       .setNumClass(6)
                                       .setBatch(16)
                                       .setEpoch(10)
                                       .build());

        dataMap.put(DataType.TEST2, new Config.Builder()
                                        .setDataPath(basePath + "test2/nor_train_t2.csv")
                                        .setTestPath(basePath + "test2/nor_test_t2.csv")
                                        .setTaskNum(480)
                                        .setChannel(9)
                                        .setWidth(128)
//                                       .isNormal(true)
                                        .setNumClass(6)
                                        .setBatch(16)
                                        .setEpoch(3)
                                        .build());

        dataMap.put(DataType.TEST3, new Config.Builder()
                                        .setDataPath(basePath + "test3/nor_train_t3.csv")
                                        .setTestPath(basePath + "test3/nor_test_t3.csv")
                                        .setTaskNum(1012)
                                        .setChannel(9)
                                        .setWidth(128)
//                                       .isNormal(true)
                                        .setNumClass(6)
                                        .setBatch(16)
                                        .setEpoch(3)
                                        .build());


        dataMap.put(DataType.TEST4, new Config.Builder()
                                        .setDataPath(basePath + "test4/nor_train_t4.csv")
                                        .setTestPath(basePath + "test4/nor_test_t4.csv")
                                        .setTaskNum(2100)
                                        .setChannel(9)
                                        .setWidth(128)
//                                       .isNormal(true)
                                        .setNumClass(6)
                                        .setBatch(16)
                                        .setEpoch(3)
                                        .build());

        // size: 150M done has issue: class 2 failed
//        dataMap.put(DataType.DF, new Config.Builder()
//                                     .setDataPath("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Final_Data/og_shuffle/Df/train.csv")
//                                     .setTestPath("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Final_Data/og_shuffle/Df/test.csv")
//                                     .setTaskNum(39120)
//                                     .setChannel(9)
//                                     .setWidth(25)
//                                     .setNumClass(2)
//                                     .setBatch(16)
//                                     .setKernel(12)
//                                     .isNormal(true)
//                                     .build());
    }

    public static Config getNewConfig(DataType dataType) {
        return dataMap.get(dataType).clone();
    }

    public static Config getConfig(DataType dataType) {
        return dataMap.get(dataType);
    }
}
