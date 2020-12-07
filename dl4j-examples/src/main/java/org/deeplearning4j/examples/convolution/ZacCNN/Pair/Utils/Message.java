package org.deeplearning4j.examples.convolution.ZacCNN.Pair.Utils;

import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Go.CommCount;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Message {

    public static final int FINAL_STATE = 10;
    public static final int COMM_STATE = 11;

    public int id;
    public INDArray parameters;
    public String confJosn;
    public double gradient;
    public int num = 0;
    public String log;

    public int state = COMM_STATE;

    //        double loss;
    public double w;
    public CommCount commCount;
    public double goScale;
    public String output;

    public Message(int id) {
        this.id = id;
    }
}
