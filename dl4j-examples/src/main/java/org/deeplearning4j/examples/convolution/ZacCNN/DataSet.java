package org.deeplearning4j.examples.convolution.ZacCNN;

import java.util.HashMap;
import java.util.Map;

public class DataSet {

    private static Map<DataType, Config> dataMap = new HashMap<>();

    static {
        // size: 160M done: good
        dataMap.put(DataType.HAR, new Config.Builder()
                                      .setDataPath("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Renew_data/nor_train.csv")
                                      .setTestPath("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Renew_data/nor_test.csv")
                                      .setTaskNum(7352)
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
        // size: 2G done has issue: test label from -1 to 17 and has 19 class!?
        dataMap.put(DataType.OP, new Config.Builder()
                                     .setDataPath("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Final_Data/og_shuffle/Op/train.csv")
                                     .setTestPath("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Final_Data/og_shuffle/Op/test.csv")
                                     .setTaskNum(58346)
                                     .setChannel(77)
                                     .setWidth(23)
                                     .setNumClass(18)
                                     .setBatch(16)
                                     .isNormal(true)
                                     .setKernel(10)
                                     .build());
        // size: 150M done has issue: class 2 failed
        dataMap.put(DataType.DF, new Config.Builder()
                                     .setDataPath("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Final_Data/og_shuffle/Df/train.csv")
                                     .setTestPath("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Final_Data/og_shuffle/Df/test.csv")
                                     .setTaskNum(39120)
                                     .setChannel(9)
                                     .setWidth(25)
                                     .setNumClass(2)
                                     .setBatch(16)
                                     .setKernel(12)
                                     .isNormal(true)
                                     .build());
    }

    public static Config getNewConfig(DataType dataType) {
        return dataMap.get(dataType).clone();
    }

    public static Config getConfig(DataType dataType) {
        return dataMap.get(dataType);
    }
}
