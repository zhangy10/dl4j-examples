package org.nd4j.examples.ZacTest;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.io.*;

public class ReadArrayTest {

    private static String file = "/Users/zhangyu/Desktop/zac_test";


    public static void main(String[] args) throws Exception{

        INDArray array = Nd4j.readBinary(new File(file));

        System.out.println(array);
    }
}
