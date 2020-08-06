package org.deeplearning4j.examples.convolution.ZacCNN.Pair.Model;

import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Config;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;


/**
 * kernel: h x w
 */

public class LeNet extends BaseConfig {


    public LeNet(Config config) {
        super(config);
    }

    @Override
    public MultiLayerConfiguration init() {

        return new NeuralNetConfiguration.Builder().seed(config.getSeed())
                   .activation(Activation.RELU)
//                   .weightInit(new WeightInitNormal()) // better init

                   .weightInit(new NormalDistribution(0.0, 1.0))
                   .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                   .l2(5 * 1e-4)

                   .convolutionMode(ConvolutionMode.Same)
                   .updater(new Adam(config.getLearnRate()))
                   .list()
                   // block 1
                   .layer(new ConvolutionLayer.Builder()
                              .kernelSize(1, config.getKernal())
                              .name("cnn1")
                              .stride(1, 1)
                              .nIn(config.getChannel())
                              .nOut(36)
                              .hasBias(true)
                              .activation(Activation.RELU)
                              .build())
//                   .layer(new BatchNormalization())
                   .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                              .kernelSize(1, config.getPooling())
                              .stride(1, config.getPooling())
                              .name("maxpool1")
                              .build())
                   // block 2
                   .layer(new ConvolutionLayer.Builder()
                              .name("cnn2")
                              .kernelSize(1, config.getKernal())
                              .stride(1, 1)
                              .nOut(72)
                              .hasBias(true)
                              .activation(Activation.RELU)
                              .build())
//                   .layer(new BatchNormalization())
                   .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                              .name("maxpool2")
                              .kernelSize(1, config.getPooling())
                              .stride(1, config.getPooling())
                              .build())
                   // fully connected
                   .layer(new DenseLayer.Builder()
                              .name("ffn1")
                              .activation(Activation.RELU)
                              .hasBias(true)
                              .nOut(300)
                              .build())
                   // output
                   .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                              .name("output")
                              .nOut(config.getNumClasses())
                              .activation(Activation.SOFTMAX) // radial basis function required
                              .build())
                   .setInputType(InputType.convolutional(config.getHeight(), config.getWidth(), config.getChannel()))

                   .build();
    }

}
