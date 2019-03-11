package org.deeplearning4j.examples.convolution.ZacCNN.DistriTest;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

public class ModelParamTest {


    public static void main(String[] args) throws Exception {
        // read a model
        File model = new File("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Renew_data/" + "model.bin");
        MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(model);


        File save = new File("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Renew_data/params");
        Nd4j.saveBinary(network.params(), save);

        System.out.println("save params");

        INDArray array = Nd4j.readBinary(save);
        System.out.println(array.shape().toString());

    }
}
