package org.deeplearning4j.examples.rl4j.RL;

import org.deeplearning4j.examples.rl4j.ZacRL.ZacDQNFactory;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Adam;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

public class ZacTrain {


    private static TrainingListener listener = new TrainingListener() {
        @Override
        public void iterationDone(Model model, int i, int i1) {
            //System.out.print("loss: " + model.score() + " batchId: " + i + " epoc: " + i1);
        }

        @Override
        public void onEpochStart(Model model) {

        }

        @Override
        public void onEpochEnd(Model model) {

        }

        @Override
        public void onForwardPass(Model model, List<INDArray> list) {

        }

        @Override
        public void onForwardPass(Model model, Map<String, INDArray> map) {

        }

        @Override
        public void onGradientCalculation(Model model) {

        }

        @Override
        public void onBackwardPass(Model model) {

        }
    };


    public static QLearning.QLConfiguration TOY_QL =
        new QLearning.QLConfiguration(
            123,   //Random seed
            20,//Max step By epoch
            2100, //Max step
            32, //Max size of experience replay
            16,



            //size of batches
            1000000,   //target update (hard)
            10,     //num step noop warmup
            0.00,  //reward scaling
            0.95,  //gamma
            10.0,  //td-error clipping
            0.05f,  //min epsilon
            500,  //num step for eps greedy anneal
            false   //double DQN
        );

// Configuration set
    public static DQNFactoryStdDense.Configuration TOY_NET =
        DQNFactoryStdDense.Configuration.builder()
            .l2(0.01).updater(new Adam(1e-2)).numLayer(2).numHiddenNodes(32)
            .listeners(new TrainingListener[] {listener})
            .build();


    public static void main(String[] args) throws IOException {
        learn();
//        load();
    }


    public static void learn() throws IOException {

        //record the training data in rl4j-data in a new folder
        DataManager manager = new DataManager();

        //define the mdp from toy (toy length)
        //
        ZacMDP mdp = new ZacMDP();


        //define the training
        QLearningDiscreteDense<ZacStep> dql = new QLearningDiscreteDense(mdp, new ZacDQNFactory(TOY_NET), TOY_QL, manager);

       // mdp.setModel(dql);

        //start the training
        dql.train();

        DQNPolicy<ZacStep> pol = dql.getPolicy();


        pol.save("/Users/zber/Desktop/toy_pol2");

        //useless on toy but good practice!
        mdp.close();
    }


    public static void load() throws IOException {

        //showcase serialization by using the trained agent on a new similar mdp (but render it this time)

        //define the mdp from gym (name, render)
        MDP mdp2 = new ZacMDP();

        //load the previous agent
        DQNPolicy<Box> pol2 = DQNPolicy.load("/Users/zber/Desktop/toy_pol2");

        //evaluate the agent
        double rewards = 0;
        int size = 20;
        for (int i = 0; i < size; i++) {
            mdp2.reset();
            double reward = pol2.play(mdp2);
            rewards += reward;
            Logger.getAnonymousLogger().info("Reward: " + reward);
        }

        Logger.getAnonymousLogger().info("average: " + rewards / size);

    }

}
