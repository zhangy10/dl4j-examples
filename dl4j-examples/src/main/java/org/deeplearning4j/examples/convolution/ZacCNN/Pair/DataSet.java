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

    private static final int batchSize = 16;
//    private static final int batchSize = 64;

    static {
        // size: 160M done: good vs image
        dataMap.put(DataType.HAR, new Config.Builder()

                                      .setDataPath(basePath + "/Har/nor_train.csv")
                                      .setTestPath(basePath + "/Har/nor_test.csv")
                                      .setTaskNum(7352)
                                      .setChannel(9)
                                      .setWidth(128)
                                      .setNumClass(6)
                                      .setKernel(SystemRun.isMobileNet ? 14 : 64)
                                      .setBatch(batchSize)
                                      .build());

        // size: 900M done: good
        dataMap.put(DataType.PAMA, new Config.Builder()
                                       .setDataPath(basePath + "/Pama2/nor_train.csv")
                                       .setTestPath(basePath + "/Pama2/nor_train.csv")
                                       .setTaskNum(11397)
                                       .setChannel(9)
                                       .setWidth(512)
                                       .setNumClass(7)
                                       .setKernel(SystemRun.isMobileNet ? 14 : 64)
                                       .setBatch(batchSize)
                                       .build());


        dataMap.put(DataType.MHe, new Config.Builder()   // kernel 50, width 100  91%   // kernel 30 width 100 96% // 11 class 35 kernel 92%
                                      .setDataPath(basePath + "/mHealth/nor_train.csv")
                                      .setTestPath(basePath + "/mHealth/nor_test.csv")
                                      .setTaskNum(2485)
                                      .setChannel(23)
                                      .setWidth(100)
                                      .setNumClass(11)
                                      .setKernel(SystemRun.isMobileNet ? 12 : 35)
                                      .setBatch(batchSize)
                                      .build());


        dataMap.put(DataType.OP, new Config.Builder()
                                     .setDataPath(basePath + "/Op/nor_train.csv")
                                     .setTestPath(basePath + "/Op/nor_test.csv")
                                     .setTaskNum(15735) // 23 15737

                                     .setChannel(77)
                                     .setWidth(23) // 25 - 10 %，23 - 10  89%, 21 -10  0.8964,
                                     .setNumClass(11)
                                     .setBatch(batchSize)
                                     .setKernel(SystemRun.isMobileNet ? 3 : 10)
                                     .build());

        dataMap.put(DataType.EMG, new Config.Builder()
                                      .setDataPath(basePath + "/EMG/nor_train.csv")
                                      .setTestPath(basePath + "/EMG/nor_test.csv")
                                      // 20 width
                                      .setTaskNum(12295)
                                      .setWidth(20)
                                      .setKernel(SystemRun.isMobileNet ? 3 : 9)

                                      // 100 width
//                                      .setTaskNum(6443)
//                                      .setWidth(100)
//                                      .setKernel(12)

                                      .setChannel(8)
                                      .setNumClass(6)
                                      .setBatch(batchSize)
                                      .build());

        dataMap.put(DataType.FALL, new Config.Builder()
                                       .setDataPath(basePath + "/Fall/nor_train.csv")
                                       .setTestPath(basePath + "/Fall/nor_test.csv")
                                       .setTaskNum(7618)
                                       .setChannel(1)
                                       .setWidth(604)
                                       .setNumClass(8)
                                       .setBatch(batchSize)
                                       .setKernel(SystemRun.isMobileNet ? 14 : 42)
                                       .build());

        dataMap.put(DataType.FINDROID, new Config.Builder()
                                           .setDataPath("/Users/zber/Program_dev/Finger_demo/Data/fing_8train_330.csv")
                                           .setTestPath("/Users/zber/Program_dev/Finger_demo/Data/fing_20test.csv")
                                           .setTaskNum(2640)
                                           .setChannel(6)
                                           .setWidth(150)
                                           .setNumClass(8)
                                           .setBatch(batchSize)
                                           .setKernel(75)
                                           .isNormal(true)
                                           .setEpoch(35)
                                           .build());

    }

    public static Config getNewConfig(DataType dataType) {
        return dataMap.get(dataType).clone();
    }

    public static Config getConfig(DataType dataType) {
        return dataMap.get(dataType);
    }
}
