package org.deeplearning4j.examples.convolution.ZacCNN.Pair.Model;

import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Config;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;

public class MDLModel {

    public enum Type {
        LENET1D, MOBILENET_1D, TCN
    }


    public static MultiLayerConfiguration getNetwork(Type type, Config config) {
        MultiLayerConfiguration conf = null;
        switch (type) {
            case LENET1D:
                conf = new LeNet(config).init();
                break;
            case MOBILENET_1D:
                conf = new Mobilenet(config).init();
                break;
            case TCN:
                conf = new TCN(config).init();
                break;
        }
        return conf;
    }

}
