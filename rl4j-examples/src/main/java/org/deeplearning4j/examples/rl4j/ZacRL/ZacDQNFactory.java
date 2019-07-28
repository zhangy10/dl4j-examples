package org.deeplearning4j.examples.rl4j.ZacRL;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.rl4j.network.dqn.DQN;
import org.deeplearning4j.rl4j.network.dqn.DQNFactory;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;

public class ZacDQNFactory implements DQNFactory {

    private final DQNFactoryStdDense.Configuration conf;

    private DenseLayer full(String name, int in, int out) {
        return new DenseLayer.Builder().name(name).nIn(in).nOut(out).dist(new NormalDistribution(0, 1)).build();
    }

    public DQN buildDQN(int[] numInputs, int numOutputs) {
        int nIn = 1;
        int[] var4 = numInputs;

        for (int var6 = 0; var6 < numInputs.length; ++var6) {
            int i = var4[var6];
            nIn *= i;
        }

        NeuralNetConfiguration.ListBuilder confB = (new NeuralNetConfiguration.Builder())
                                                       .seed(12345L)
                                                       .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                                       .updater((IUpdater) (this.conf.getUpdater() != null ? this.conf.getUpdater() : new Adam()))
//                                                       .weightInit(WeightInit.XAVIER)
                                                       .weightInit(WeightInit.DISTRIBUTION)
                                                       .dist(new NormalDistribution(0.0, 1.0))
                                                       .activation(Activation.RELU)
                                                       .l2(this.conf.getL2())
                                                       .list()
//                                                       .layer(0, ((org.deeplearning4j.nn.conf.layers.DenseLayer.Builder)((org.deeplearning4j.nn.conf.layers.DenseLayer.Builder)((org.deeplearning4j.nn.conf.layers.DenseLayer.Builder)(new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()).nIn(nIn)).nOut(this.conf.getNumHiddenNodes())).activation(Activation.RELU))
                                                       // input layer
                                                       .layer(0, full("f" + 0, nIn, this.conf.getNumHiddenNodes()));

        // hidden layers
        for (int i = 1; i < this.conf.getNumLayer(); ++i) {
//            confB.layer(i, ((org.deeplearning4j.nn.conf.layers.DenseLayer.Builder) ((org.deeplearning4j.nn.conf.layers.DenseLayer.Builder) ((org.deeplearning4j.nn.conf.layers.DenseLayer.Builder) (new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder()).nIn(this.conf.getNumHiddenNodes())).nOut(this.conf.getNumHiddenNodes())).activation(Activation.RELU)).build());
            confB.layer(i, full("f" + i, this.conf.getNumHiddenNodes(), this.conf.getNumHiddenNodes()));
        }

        // output layer
//        confB.layer(this.conf.getNumLayer(), ((org.deeplearning4j.nn.conf.layers.OutputLayer.Builder) ((org.deeplearning4j.nn.conf.layers.OutputLayer.Builder) ((org.deeplearning4j.nn.conf.layers.OutputLayer.Builder) (new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.MSE)).activation(Activation.IDENTITY)).nIn(this.conf.getNumHiddenNodes())).nOut(numOutputs)).build());
        confB.layer(this.conf.getNumLayer(), new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                                 .name("o1")
                                                 .nOut(numOutputs)
                                                 .activation(Activation.SOFTMAX)
                                                 .build());

        MultiLayerConfiguration mlnconf = confB.pretrain(false).backprop(true).build();
        MultiLayerNetwork model = new MultiLayerNetwork(mlnconf);
        model.init();
        if (this.conf.getListeners() != null) {
            model.setListeners(this.conf.getListeners());
        } else {
            model.setListeners(new TrainingListener[]{new ScoreIterationListener(50)});
        }

        return new DQN(model);
    }

    public ZacDQNFactory(DQNFactoryStdDense.Configuration conf) {
        this.conf = conf;
    }

    public DQNFactoryStdDense.Configuration getConf() {
        return this.conf;
    }

    public boolean equals(Object o) {
        if (o == this) {
            return true;
        } else if (!(o instanceof DQNFactoryStdDense)) {
            return false;
        } else {
            DQNFactoryStdDense other = (DQNFactoryStdDense) o;
            Object this$conf = this.getConf();
            Object other$conf = other.getConf();
            if (this$conf == null) {
                if (other$conf != null) {
                    return false;
                }
            } else if (!this$conf.equals(other$conf)) {
                return false;
            }

            return true;
        }
    }

    public int hashCode() {
        int PRIME = 1;
        int result = 1;
        Object $conf = this.getConf();
        result = result * 59 + ($conf == null ? 43 : $conf.hashCode());
        return result;
    }

    public String toString() {
        return "DQNFactoryStdDense(conf=" + this.getConf() + ")";
    }

}
