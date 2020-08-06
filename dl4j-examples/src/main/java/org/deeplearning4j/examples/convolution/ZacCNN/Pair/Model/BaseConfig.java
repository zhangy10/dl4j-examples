package org.deeplearning4j.examples.convolution.ZacCNN.Pair.Model;

import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Config;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;

public abstract class BaseConfig extends MultiLayerConfiguration {

    protected Config config;

    public static CacheMode cacheMode = CacheMode.NONE;
    public static WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    public static ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;
    public static int normal_std = 1;
    public static int seed = 1234;

    public BaseConfig(Config config) {
        this.config = config;
    }

    public abstract MultiLayerConfiguration init();
}
