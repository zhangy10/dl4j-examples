package org.deeplearning4j.examples.convolution.ZacCNN.DistriTest;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

public class CompressTest2 {


    public static void main(String[] args) throws Exception {
        String compressed = "/Users/zhangyu/Desktop/compress";
        String data = "/Users/zhangyu/Desktop/test";

        INDArray old = Nd4j.readBinary(new File(data));

        // TODO compressed file is not working for coverting to array
        INDArray newd = Nd4j.readBinary(new File(compressed));

        INDArray r = old.sub(newd);


    }
}
