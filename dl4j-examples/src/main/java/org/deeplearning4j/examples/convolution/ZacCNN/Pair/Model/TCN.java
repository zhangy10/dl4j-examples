package org.deeplearning4j.examples.convolution.ZacCNN.Pair.Model;

import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Config;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution1D;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;


/**
 * 1D
 */
public class TCN extends BaseConfig {

//    private int blockN = 3;
//    private int[] paddings = new int[blockN];
//    private int base = 2;

    public TCN(Config config) {
        super(config);

//        for (int i = 0; i < blockN; i++) {
//            paddings[0] = (config.getKernal() - 1) * (int) Math.pow(base, i);
//        }
    }


    @Override
    public MultiLayerConfiguration init() {
        return new NeuralNetConfiguration.Builder().seed(config.getSeed())
                   .activation(Activation.RELU)
//                   .weightInit(new WeightInitNormal()) // better init

                   .weightInit(new NormalDistribution(0.0, 1.0))
                   .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                   .l2(5 * 1e-4)

                   .updater(new Adam(config.getLearnRate()))
                   .list()
                   // block 1
                   .layer(new Convolution1D.Builder()
                              .kernelSize(config.getKernal())
                              .name("Tcnn1")
                              .stride(1)
                              .nIn(config.getChannel())
                              .nOut(config.getC1_out())
//                              .convolutionMode(ConvolutionMode.Causal)
                              .dilation(1)
                              .build())
                   .layer(new Convolution1D.Builder()
                              .name("Tcnn2")
                              .kernelSize(config.getKernal())
                              .stride(1)
                              .nOut(config.getC1_out())
//                              .convolutionMode(ConvolutionMode.Causal)
                              .dilation(1)
                              .build())

//                   // block 2
                   .layer(new Convolution1D.Builder()
                              .kernelSize(config.getKernal())
                              .name("Tcnn3")
                              .stride(1)
                              .nOut(config.getC1_out())
//                              .convolutionMode(ConvolutionMode.Causal)
                              .dilation(2)
                              .build())
                   .layer(new Convolution1D.Builder()
                              .name("Tcnn4")
                              .kernelSize(config.getKernal())
                              .stride(1)
                              .nOut(config.getC1_out())
//                              .convolutionMode(ConvolutionMode.Causal)
                              .dilation(2)
                              .build())

                   .layer(new GlobalPoolingLayer(PoolingType.MAX))

                   // output
                   .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                              .name("output")
                              .nOut(config.getNumClasses())
                              .activation(Activation.SOFTMAX) // radial basis function required
                              .build())
                   .setInputType(
                       InputType.recurrent(config.getChannel(), config.getWidth()))
                   .build();
    }

}
