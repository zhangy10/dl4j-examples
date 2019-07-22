package org.deeplearning4j.examples.convolution.ZacCNN.DistriTest;


import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

import static java.math.RoundingMode.*;

public class DoubleTest {

    static boolean needScale = false;
    static int scale = 8;
    static double up = 1000;


    public static void main(String[] args) {

        Double out = 0.5953496694564819;
        if (needScale) {
            out = BigDecimal.valueOf(out).setScale(scale, HALF_UP).doubleValue();

//            out *= up;
//            out = BigDecimal.valueOf(out).setScale(0, HALF_UP).doubleValue();
        }

        String test = out.toString();
//        String test = "595.0";
        Text t1 = new Text(test);

        String a = "595.0";
        Text t2 = new Text(a);

        int size = 10000;
        int row = 16;

        List<Writable> l1 = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            l1.add(t1);
        }

        List<Writable> l2 = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            l2.add(t2);
        }


        long time = System.currentTimeMillis();
        for (int i = 0; i < row; i++) {
            INDArray a2 = convertToArray(l2);
        }
        System.out.println("Image: " + (System.currentTimeMillis() - time));


        time = System.currentTimeMillis();
        for (int i = 0; i < row; i++) {
            INDArray a1 = convertToArray(l1);
        }
        System.out.println("Sensing: " + (System.currentTimeMillis() - time));
        time = System.currentTimeMillis();

        for (int i = 0; i < row; i++) {
            INDArray a3 = RecordConverter.toArray(l1);
        }
        System.out.println("Converter: " + (System.currentTimeMillis() - time));

        time = System.currentTimeMillis();
        for (int i = 0; i < row; i++) {
            INDArray a2 = convertToArray(l2);
        }
        System.out.println("Image: " + (System.currentTimeMillis() - time));

    }

    public static INDArray convertToArray(List<Writable> list) {
        INDArray a = Nd4j.create(1, list.size());

        int k = 0;
        for (Writable w : list) {
            a.putScalar(0, k, w.toDouble());
            k++;
        }

        return a;
    }
}
