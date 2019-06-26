package org.deeplearning4j.examples.rl4j.ZacRL;

import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.learning.config.Adam;

import java.io.IOException;
import java.util.logging.Logger;

public class ZacTrain {


    public static QLearning.QLConfiguration TOY_QL =
        new QLearning.QLConfiguration(
            123,   //Random seed
            20,//Max step By epoch
            1000, //Max step
            1000, //Max size of experience replay
            32,    //size of batches
            10,   //target update (hard)
            0,     //num step noop warmup
            0.05,  //reward scaling
            0.99,  //gamma
            10.0,  //td-error clipping
            0.1f,  //min epsilon
            2000,  //num step for eps greedy anneal
            true   //double DQN
        );


    public static DQNFactoryStdDense.Configuration TOY_NET =
        DQNFactoryStdDense.Configuration.builder()
            .l2(0.01).updater(new Adam(1e-2)).numLayer(3).numHiddenNodes(16).build();

    public static void main(String[] args) throws IOException {
        learn();
//        load();
    }


    public static void learn() throws IOException {

        //record the training data in rl4j-data in a new folder
        DataManager manager = new DataManager();

        //define the mdp from toy (toy length)
        ZacMDP mdp = new ZacMDP();

        //define the training
        QLearningDiscreteDense<ZacStep> dql = new QLearningDiscreteDense(mdp, TOY_NET, TOY_QL, manager);

        mdp.setModel(dql);

        //start the training
        dql.train();

        DQNPolicy<ZacStep> pol = dql.getPolicy();

        pol.save("/Users/zhangyu/Desktop/toy_pol2");

        //useless on toy but good practice!
        mdp.close();
    }


    public static void load() throws IOException {

        //showcase serialization by using the trained agent on a new similar mdp (but render it this time)

        //define the mdp from gym (name, render)
        MDP mdp2 = new ZacMDP();

        //load the previous agent
        DQNPolicy<Box> pol2 = DQNPolicy.load("/Users/zhangyu/Desktop/toy_pol2");

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
