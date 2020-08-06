package org.deeplearning4j.examples.convolution.ZacCNN.Pair;

import java.util.HashMap;
import java.util.Map;

public class DataSet {

    public enum DataType {
        HAR,
        OP,
        MHe,
        //    DF,
        TEST,
        TEST2,
        TEST3,
        TEST4,
        EMG,
        PAMA,
        FALL,
        FINDROID,
        HAR_NPY,
        HAR_GAR
    }

    private static Map<DataType, Config> dataMap = new HashMap<>();

    //    private static final String basePath = "/Users/zhangyu/GoogleDrive/PHD/All_Papers/MDLdroid/Submission/Final_Data/final_data";
    private static final String basePath = SystemRun.dataPath;

    static {
        // size: 160M done: good vs image
        dataMap.put(DataType.HAR, new Config.Builder()

                                      .setDataPath(basePath + "/Har/nor_train.csv")
                                      .setTestPath(basePath + "/Har/nor_test.csv")
                                      .setTaskNum(7352)
                                      .setChannel(9)
                                      .setWidth(128)
                                      .setNumClass(6)
//            .setKernel(14)
                                      .setBatch(16)
                                      .build());

        // size: 900M done: good
        dataMap.put(DataType.PAMA, new Config.Builder()
                                       .setDataPath(basePath + "/Pama2/nor_train.csv")
                                       .setTestPath(basePath + "/Pama2/nor_train.csv")
                                       .setTaskNum(11397)
                                       .setChannel(9)
                                       .setWidth(512)
                                       .setNumClass(7)
                                       .setBatch(16)
                                       .build());


        dataMap.put(DataType.MHe, new Config.Builder()   // kernel 50, width 100  91%   // kernel 30 width 100 96% // 11 class 35 kernel 92%
                                      .setDataPath(basePath + "/mHealth/nor_train.csv")
                                      .setTestPath(basePath + "/mHealth/nor_test.csv")
                                      .setTaskNum(2485)
                                      .setChannel(23)
                                      .setWidth(100)
                                      .setNumClass(11)
                                      .setKernel(35)
                                      .setBatch(16)
                                      .build());


        dataMap.put(DataType.OP, new Config.Builder()
                                     .setDataPath(basePath + "/Op/nor_train.csv")
                                     .setTestPath(basePath + "/Op/nor_test.csv")
                                     .setTaskNum(15735) // 23 15737

                                     .setChannel(77)
                                     .setWidth(23) // 25 - 10 %，23 - 10  89%, 21 -10  0.8964,
                                     .setNumClass(11)
                                     .setBatch(16)
                                     .setKernel(10)
                                     .build());

        dataMap.put(DataType.EMG, new Config.Builder()
                                      .setDataPath(basePath + "/EMG/nor_train.csv")
                                      .setTestPath(basePath + "/EMG/nor_test.csv")
                                      .setTaskNum(12295)
                                      .setChannel(8)
                                      .setWidth(20)
                                      .setNumClass(6)
                                      .setBatch(16)
                                      .setKernel(9)
                                      .build());

        dataMap.put(DataType.FALL, new Config.Builder()
                                       .setDataPath(basePath + "/Fall/nor_train.csv")
                                       .setTestPath(basePath + "/Fall/nor_test.csv")
                                       .setTaskNum(7618)
                                       .setChannel(1)
                                       .setWidth(604)
                                       .setNumClass(8)
                                       .setBatch(16)
                                       .setKernel(42)
                                       .setEpoch(20)
                                       .build());

        dataMap.put(DataType.FINDROID, new Config.Builder()
                                           .setDataPath("/Users/zber/Program_dev/Finger_demo/Data/fing_8train_330.csv")
                                           .setTestPath("/Users/zber/Program_dev/Finger_demo/Data/fing_20test.csv")
                                           .setTaskNum(2640)
                                           .setChannel(6)
                                           .setWidth(150)
                                           .setNumClass(8)
                                           .setBatch(16)
                                           .setKernel(75)
                                           .isNormal(true)
                                           .setEpoch(35)
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


        // --------------------------------------------------------------
        dataMap.put(DataType.HAR_NPY, new Config.Builder()

                                          .setDataPath(basePath + "/Har/nor_train.csv")
                                          .setTestPath(basePath + "/Har/nor_test.csv")
                                          .setTaskNum(7352)

                                          .setChannel(9)
                                          .setWidth(128)
                                          .setNumClass(6)
                                          .setBatch(16)
                                          .build());

        dataMap.put(DataType.HAR_GAR, new Config.Builder()

                                          .setDataPath("/Users/zhangyu/Downloads/har_normal_diff_train.csv")
                                          .setTestPath("/Users/zhangyu/Downloads/har_normal_diff_test.csv")
                                          .setTaskNum(7352)
                                          .setChannel(9)
                                          .setKernel(63)
                                          .setWidth(127)
                                          .setNumClass(6)
                                          .setBatch(16)
                                          .build());

    }

    public static Config getNewConfig(DataType dataType) {
        return dataMap.get(dataType).clone();
    }

    public static Config getConfig(DataType dataType) {
        return dataMap.get(dataType);
    }
}
