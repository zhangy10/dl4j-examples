package org.deeplearning4j.examples.convolution.ZacCNN.Pair.Model;

import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Config;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;


/**
 * 2D
 */
public class Mobilenet extends BaseConfig {


    private static final String TEMP_MODEL_FILENAME = "tempModel";
    private static final String H5_EXTENSION = ".h5";
    private static final String COPY_TO = "/Users/zhangyu/Desktop/";

    public Mobilenet(Config config) {
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

                   .updater(new Adam(config.getLearnRate()))
                   .list()
                   .layer(new ConvolutionLayer.Builder()
                              .name("point1")
                              .kernelSize(1, config.getKernal())
                              .stride(1, 2)
                              .nIn(config.getChannel())
                              .nOut(32)
//                              .hasBias(config.isHasBias())
                              .activation(Activation.RELU).build())
//                   .layer(new BatchNormalization())

                   // block 1
                   .layer(new DepthwiseConvolution2D.Builder().name("depth1")
                              .kernelSize(1, config.getKernal())
                              .stride(1, 1)
                              .depthMultiplier(1)
//                              .nOut(config.getC2_out())
                              .hasBias(true)
                              .activation(Activation.RELU)
                              .build())
//                   .layer(new BatchNormalization())
                   .layer(new ConvolutionLayer.Builder()
                              .name("point1")
                              .kernelSize(1, 1)
                              .stride(1, 1)
                              .nOut(64)
//                              .hasBias(config.isHasBias())
                              .activation(Activation.RELU).build())
//                   .layer(new BatchNormalization())

                   // block 2
                   .layer(new DepthwiseConvolution2D.Builder().name("depth2")
                              .kernelSize(1, config.getKernal())
                              .stride(1, 2)
                              .depthMultiplier(1)
//                              .nOut(config.getC2_out())
//                              .hasBias(config.isHasBias())
                              .activation(Activation.RELU)
                              .build())
//                   .layer(new BatchNormalization())
                   .layer(new ConvolutionLayer.Builder()
                              .name("point2")
                              .kernelSize(1, 1)
                              .stride(1, 1)
                              .nOut(128)
//                              .hasBias(config.isHasBias())
                              .activation(Activation.RELU).build())
//                   .layer(new BatchNormalization())

                   // block 3
                   .layer(new DepthwiseConvolution2D.Builder().name("depth3")
                              .kernelSize(1, config.getKernal())
                              .stride(1, 1)
                              .depthMultiplier(1)
//                              .nOut(config.getC2_out())
//                              .hasBias(config.isHasBias())
                              .activation(Activation.RELU)
                              .build())
//                   .layer(new BatchNormalization())
                   .layer(new ConvolutionLayer.Builder()
                              .name("point3")
                              .kernelSize(1, 1)
                              .stride(1, 1)
                              .nOut(256)
//                              .hasBias(config.isHasBias())
                              .activation(Activation.RELU).build())
//                   .layer(new BatchNormalization())

                   .layer(new GlobalPoolingLayer(PoolingType.AVG))

                   // output
                   .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                              .name("output")
                              .nOut(config.getNumClasses())
                              .activation(Activation.SOFTMAX) // radial basis function required
                              .build())
                   .setInputType(InputType.convolutionalFlat(config.getHeight(), config.getWidth(), config.getChannel()))
                   .build();
    }

}
