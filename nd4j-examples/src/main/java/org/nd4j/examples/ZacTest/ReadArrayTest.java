package org.nd4j.examples.ZacTest;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class ReadArrayTest {

    private static String file = "/Users/zhangyu/Desktop/zac_test";


    public static void main(String[] args) throws Exception{

//        INDArray array = Nd4j.readBinary(new File(file));
//
//        System.out.println(array);


        List<Integer> list = new ArrayList<>();

        int size = 10;
        for (int i = 0; i < size; i++) {
            list.add(i);
        }


        List<Integer> list1 = new ArrayList<>();
        list1.addAll(list.subList(0, list.size()));

        list.removeAll(list1);


        for (int i = 20; i < 3 * size; i++) {
            list.add(i);
        }


    }
}
