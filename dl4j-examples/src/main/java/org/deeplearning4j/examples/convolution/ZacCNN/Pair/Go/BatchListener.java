package org.deeplearning4j.examples.convolution.ZacCNN.Pair.Go;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.TrainingListener;

public interface BatchListener extends TrainingListener {

    void iterationStart(Model model, int iteration, int epoch);

}
