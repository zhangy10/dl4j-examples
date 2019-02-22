package org.nd4j.examples.ZacTest;

import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.io.*;

public class ZacTest {


    private static String file = "/Users/zhangyu/Desktop/zac_test";

    public static void main(String[] args) throws Exception{
//        DataOutputStream


//        Nd4j.write();
//
//        Nd4j.saveBinary();
//
//        Nd4j.readBinary()

        int nRows = 1;
        int nColumns = 10;

//        INDArray zeros = Nd4j.zeros(nRows, nColumns);
        INDArray random = Nd4j.rand(nRows, nColumns);
//        INDArray ones = Nd4j.ones(nRows, nColumns, 6);

        File array = new File(file);

        Nd4j.saveBinary(random, array);


//        INDArray aa = random.reshape('c', 2, 5);
//
//
//        INDArray bb = random.reshape('f', 2, 5);

//        System.out.println(zeros);
        System.out.println(random);
//        System.out.println(zeros.shape().toString());
//        System.out.println(ones);
//        System.out.println(ones.shape().toString());
//        System.out.println(aa);
//        System.out.println(bb);

//        INDArray vstack = Nd4j.vstack(ones, zeros);
//
//        vstack = Nd4j.vstack(vstack, vstack);


//        System.out.println(vstack);


//        System.out.println(ones);
//        for (int i = 0; i < vstack.rows(); i++) {
//            for (int j = 0; j < vstack.columns(); j++) {
//                System.out.println(i + " " + j + " " + vstack.getDouble(i, j));
//            }


//        NdIndexIterator iter = new NdIndexIterator(vstack.rows(), nColumns);
//
//        while (iter.hasNext()) {
//            int[] nextIndex = iter.next();

//            double nextVal = vstack.getDouble(nextIndex);
//        }


    }
}
