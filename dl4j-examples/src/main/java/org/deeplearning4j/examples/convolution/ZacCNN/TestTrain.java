package org.deeplearning4j.examples.convolution.ZacCNN;

import junit.framework.Test;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;

public class TestTrain {

    //Global variables to accept the classification results from the background thread.
    double first;
    double second;
    double third;


    public void run() {
        //Get the doubles from params, which is an array so they will be 0,1,2,3

        double pld = 5.1;
        double pwd = 3.5;
        double sld = 1.4;
        double swd = 0.2;


        //Write them in the log
        System.out.println("do in background string pl = " + pld);
        System.out.println("do in background string pw = " + pwd);
        System.out.println("do in background string pw = " + pwd);
        System.out.println("do in background string sw = " + swd);


        //Create input
        INDArray actualInput = Nd4j.zeros(1,4);
        actualInput.putScalar(new int[]{0,0}, pld);
        actualInput.putScalar(new int[]{0,1}, pwd);
        actualInput.putScalar(new int[]{0,2}, sld);
        actualInput.putScalar(new int[]{0,3}, swd);

        //Convert the iris data into 150x4 matrix
        int row=150;
        int col=4;

        double[][] irisMatrix=new double[row][col];
        int i = 0;
        for(int r=0; r<row; r++){
            for( int c=0; c<col; c++){
                irisMatrix[r][c]= IrisDataSet.irisData[i++];
            }
        }


        //Check the array by printing it in the log
        System.out.println(Arrays.deepToString(irisMatrix).replace("], ", "]\n"));

        //Now do the same for the label data
        int rowLabel=150;
        int colLabel=3;

        double[][] twodimLabel=new double[rowLabel][colLabel];
        int ii = 0;
        for(int r=0; r<rowLabel; r++){
            for( int c=0; c<colLabel; c++){
                twodimLabel[r][c]= IrisDataSet.labelData[ii++];
            }
        }

        System.out.println(Arrays.deepToString(twodimLabel).replace("], ", "]\n"));

        //Convert the data matrices into training INDArrays
        INDArray trainingIn = Nd4j.create(irisMatrix);
        INDArray trainingOut = Nd4j.create(twodimLabel);

        //build the layers of the network
        DenseLayer inputLayer = new DenseLayer.Builder()
                .nIn(4)
                .nOut(3)
                .name("Input")
                .build();

        DenseLayer hiddenLayer = new DenseLayer.Builder()
                .nIn(3)
                .nOut(3)
                .name("Hidden")
                .build();

        OutputLayer outputLayer = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(3)
                .nOut(3)
                .name("Output")
                .activation(Activation.SOFTMAX)
                .build();


        NeuralNetConfiguration.Builder nncBuilder = new NeuralNetConfiguration.Builder();
        long seed = 6;
        nncBuilder.seed(seed);
        nncBuilder.activation(Activation.TANH);
        nncBuilder.weightInit(WeightInit.XAVIER);

        // for annotationProcessors
//            nncBuilder.updater(Updater.ADAM);

        NeuralNetConfiguration.ListBuilder listBuilder = nncBuilder.list();
        listBuilder.layer(0, inputLayer);
        listBuilder.layer(1, hiddenLayer);
        listBuilder.layer(2, outputLayer);

        listBuilder.backprop(true);

        MultiLayerNetwork myNetwork = new MultiLayerNetwork(listBuilder.build());
        myNetwork.init();


        //Create a data set from the INDArrays and train the network
        DataSet myData = new DataSet(trainingIn, trainingOut);
        for(int l=0; l<=1000; l++) {
            myNetwork.fit(myData);
        }

        //Evaluate the input data against the model
        INDArray actualOutput = myNetwork.output(actualInput);
        System.out.println("Output: " + actualOutput.toString());

        //Retrieve the three probabilities
        first = actualOutput.getDouble(0,0);
        second = actualOutput.getDouble(0,1);
        third = actualOutput.getDouble(0,2);
    }


    public static void main (String[] args) {
        new TestTrain().run();
    }

}





class IrisDataSet {
    //load the raw Iris data set into a java array.
    //Four measurements per flower, 150 flowers total, 50 of each type
    //Values are petal length, petal width, sepal length, sepal width
    // 5.1,3.5,1.4,0.2 for one data to one label
    static double [] irisData={5.1,3.5,1.4,0.2,4.9,3,1.4,0.2,4.7,3.2,1.3,0.2,4.6,3.1,1.5,0.2,5,3.6,1.4,0.2,
            5.4,3.9,1.7,0.4,4.6,3.4,1.4,0.3,5,3.4,1.5,0.2,4.4,2.9,1.4,0.2,4.9,3.1,1.5,0.1,5.4,3.7,1.5,0.2,
            4.8,3.4,1.6,0.2,4.8,3,1.4,0.1,4.3,3,1.1,0.1,5.8,4,1.2,0.2,5.7,4.4,1.5,0.4,5.4,3.9,1.3,0.4,
            5.1,3.5,1.4,0.3,5.7,3.8,1.7,0.3,5.1,3.8,1.5,0.3,5.4,3.4,1.7,0.2,5.1,3.7,1.5,0.4,4.6,3.6,1,0.2,
            5.1,3.3,1.7,0.5,4.8,3.4,1.9,0.2,5,3,1.6,0.2,5,3.4,1.6,0.4,5.2,3.5,1.5,0.2,5.2,3.4,1.4,0.2,
            4.7,3.2,1.6,0.2,4.8,3.1,1.6,0.2,5.4,3.4,1.5,0.4,5.2,4.1,1.5,0.1,5.5,4.2,1.4,0.2,4.9,3.1,1.5,0.1,
            5,3.2,1.2,0.2,5.5,3.5,1.3,0.2,4.9,3.1,1.5,0.1,4.4,3,1.3,0.2,5.1,3.4,1.5,0.2,5,3.5,1.3,0.3,
            4.5,2.3,1.3,0.3,4.4,3.2,1.3,0.2,5,3.5,1.6,0.6,5.1,3.8,1.9,0.4,4.8,3,1.4,0.3,5.1,3.8,1.6,0.2,
            4.6,3.2,1.4,0.2,5.3,3.7,1.5,0.2,5,3.3,1.4,0.2,7,3.2,4.7,1.4,6.4,3.2,4.5,1.5,6.9,3.1,4.9,1.5,
            5.5,2.3,4,1.3,6.5,2.8,4.6,1.5,5.7,2.8,4.5,1.3,6.3,3.3,4.7,1.6,4.9,2.4,3.3,1,6.6,2.9,4.6,1.3,
            5.2,2.7,3.9,1.4,5,2,3.5,1,5.9,3,4.2,1.5,6,2.2,4,1,6.1,2.9,4.7,1.4,5.6,2.9,3.6,1.3,6.7,3.1,4.4,1.4,
            5.6,3,4.5,1.5,5.8,2.7,4.1,1,6.2,2.2,4.5,1.5,5.6,2.5,3.9,1.1,5.9,3.2,4.8,1.8,6.1,2.8,4,1.3,
            6.3,2.5,4.9,1.5,6.1,2.8,4.7,1.2,6.4,2.9,4.3,1.3,6.6,3,4.4,1.4,6.8,2.8,4.8,1.4,6.7,3,5,1.7,
            6,2.9,4.5,1.5,5.7,2.6,3.5,1,5.5,2.4,3.8,1.1,5.5,2.4,3.7,1,5.8,2.7,3.9,1.2,6,2.7,5.1,1.6,
            5.4,3,4.5,1.5,6,3.4,4.5,1.6,6.7,3.1,4.7,1.5,6.3,2.3,4.4,1.3,5.6,3,4.1,1.3,5.5,2.5,4,1.3,
            5.5,2.6,4.4,1.2,6.1,3,4.6,1.4,5.8,2.6,4,1.2,5,2.3,3.3,1,5.6,2.7,4.2,1.3,5.7,3,4.2,1.2,
            5.7,2.9,4.2,1.3,6.2,2.9,4.3,1.3,5.1,2.5,3,1.1,5.7,2.8,4.1,1.3,6.3,3.3,6,2.5,5.8,2.7,5.1,1.9,
            7.1,3,5.9,2.1,6.3,2.9,5.6,1.8,6.5,3,5.8,2.2,7.6,3,6.6,2.1,4.9,2.5,4.5,1.7,7.3,2.9,6.3,1.8,
            6.7,2.5,5.8,1.8,7.2,3.6,6.1,2.5,6.5,3.2,5.1,2,6.4,2.7,5.3,1.9,6.8,3,5.5,2.1,5.7,2.5,5,2,
            5.8,2.8,5.1,2.4,6.4,3.2,5.3,2.3,6.5,3,5.5,1.8,7.7,3.8,6.7,2.2,7.7,2.6,6.9,2.3,6,2.2,5,1.5,
            6.9,3.2,5.7,2.3,5.6,2.8,4.9,2,7.7,2.8,6.7,2,6.3,2.7,4.9,1.8,6.7,3.3,5.7,2.1,7.2,3.2,6,1.8,
            6.2,2.8,4.8,1.8,6.1,3,4.9,1.8,6.4,2.8,5.6,2.1,7.2,3,5.8,1.6,7.4,2.8,6.1,1.9,7.9,3.8,6.4,2,
            6.4,2.8,5.6,2.2,6.3,2.8,5.1,1.5,6.1,2.6,5.6,1.4,7.7,3,6.1,2.3,6.3,3.4,5.6,2.4,6.4,3.1,5.5,1.8,
            6,3,4.8,1.8,6.9,3.1,5.4,2.1,6.7,3.1,5.6,2.4,6.9,3.1,5.1,2.3,5.8,2.7,5.1,1.9,6.8,3.2,5.9,2.3,
            6.7,3.3,5.7,2.5,6.7,3,5.2,2.3,6.3,2.5,5,1.9,6.5,3,5.2,2,6.2,3.4,5.4,2.3,5.9,3,5.1,1.8};

    //load the iris label data into a one dimensional array (Type 1 = 1,0,0 / Type 2 = 0,1,0 / Type 3 = 0,0,1).
    //Type 1 = I. setosa, Type 2 = I. versicolor, Type 3 = I. virginica
    static double [] labelData={1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,
            1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,
            1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,
            1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,
            0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,
            0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,
            0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,1,0,0,1,
            0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,
            0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,
            0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,
            0,0,1,0,0,1,0,0,1
    };
}
