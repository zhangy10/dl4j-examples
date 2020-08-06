package org.deeplearning4j.examples.convolution.ZacCNN.Pair.Utils;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Message {
    public int id;
    public INDArray parameters;
    public String confJosn;
    public double gradient;
    public int num = 0;

    //        double loss;
    public double w;
}
