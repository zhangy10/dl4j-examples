package org.deeplearning4j.examples.convolution.ZacCNN.Pair.Go;

public class NonIIDRunAll extends NonIIDRun {

    public static void main(String[] args) {

        // start from 2 to 8, select one of 9
        maxTaskNum = 8;
        testTimes = 1;
        // start from 2
        taskNum = 2;

        new NonIIDRunAll();
    }

}
