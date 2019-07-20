package org.deeplearning4j.examples.rl4j;

import java.io.IOException;
import java.util.logging.Logger;

import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.AsyncNStepQLearningDiscrete;
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.AsyncNStepQLearningDiscreteDense;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.mdp.gym.GymEnv;
import org.deeplearning4j.rl4j.mdp.toy.HardDeteministicToy;
import org.deeplearning4j.rl4j.mdp.toy.SimpleToy;
import org.deeplearning4j.rl4j.mdp.toy.SimpleToyState;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.learning.config.Adam;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/11/16.
 *
 * main example for toy DQN
 *
 */
public class Toy {


    public static QLearning.QLConfiguration TOY_QL =
            new QLearning.QLConfiguration(
                    123,   //Random seed
                    10,//Max step By epoch
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
        simpleToy();
//        hardToy();
        //toyAsyncNstep();
//        loadSimplyToy();
//        loadHardToy();
    }

    public static void simpleToy() throws IOException {

        //record the training data in rl4j-data in a new folder
        DataManager manager = new DataManager();

        //define the mdp from toy (toy length)
        SimpleToy mdp = new SimpleToy(5);

        System.out.println(mdp.getClass());

        //define the training method
//        Learning<SimpleToyState, Integer, DiscreteSpace, IDQN> dql = new QLearningDiscreteDense<SimpleToyState>(mdp, TOY_NET, TOY_QL, manager);
        QLearningDiscreteDense<SimpleToyState>  dql = new QLearningDiscreteDense<SimpleToyState>(mdp, TOY_NET, TOY_QL, manager);


        //enable some logging for debug purposes on toy mdp
        mdp.setFetchable(dql);

        //start the training
        dql.train();

        DQNPolicy<SimpleToyState> pol = dql.getPolicy();

        pol.save("/Users/zhangyu/Desktop/toy_pol1");

        //useless on toy but good practice!
        mdp.close();

    }

    public static void loadSimplyToy() throws IOException {

        //showcase serialization by using the trained agent on a new similar mdp (but render it this time)

        //define the mdp from gym (name, render)
        MDP mdp2 = new SimpleToy(20);

        //load the previous agent
        DQNPolicy<Box> pol2 = DQNPolicy.load("/Users/zhangyu/Desktop/toy_pol1");

        //evaluate the agent
        double rewards = 0;
        for (int i = 0; i < 1000; i++) {
            mdp2.reset();
            double reward = pol2.play(mdp2);
            rewards += reward;
            Logger.getAnonymousLogger().info("Reward: " + reward);
        }

        Logger.getAnonymousLogger().info("average: " + rewards/1000);

    }

    public static void loadHardToy() throws IOException {

        //showcase serialization by using the trained agent on a new similar mdp (but render it this time)

        //define the mdp from gym (name, render)
        MDP mdp2 = new HardDeteministicToy();

        //load the previous agent
        DQNPolicy<Box> pol2 = DQNPolicy.load("/Users/zhangyu/Desktop/toy_pol2");

        //evaluate the agent
        double rewards = 0;
        for (int i = 0; i < 1000; i++) {
            mdp2.reset();
            double reward = pol2.play(mdp2);
            rewards += reward;
            Logger.getAnonymousLogger().info("Reward: " + reward);
        }

        Logger.getAnonymousLogger().info("average: " + rewards/1000);

    }


    public static void hardToy() throws IOException {

        //record the training data in rl4j-data in a new folder
        DataManager manager = new DataManager();

        //define the mdp from toy (toy length)
        MDP mdp = new HardDeteministicToy();

        //define the training
        QLearningDiscreteDense<SimpleToyState> dql = new QLearningDiscreteDense(mdp, TOY_NET, TOY_QL, manager);

        //start the training
        dql.train();

        DQNPolicy<SimpleToyState> pol = dql.getPolicy();

        pol.save("/Users/zhangyu/Desktop/toy_pol2");

        //useless on toy but good practice!
        mdp.close();


    }


    public static AsyncNStepQLearningDiscrete.AsyncNStepQLConfiguration TOY_ASYNC_QL =
        new AsyncNStepQLearningDiscrete.AsyncNStepQLConfiguration(
            123,        //Random seed
            100000,     //Max step By epoch
            80000,      //Max step
            8,          //Number of threads
            5,          //t_max
            100,        //target update (hard)
            0,          //num step noop warmup
            0.1,        //reward scaling
            0.99,       //gamma
            10.0,       //td-error clipping
            0.1f,       //min epsilon
            2000        //num step for eps greedy anneal
        );


    public static void toyAsyncNstep() throws IOException {

        //record the training data in rl4j-data in a new folder
        DataManager manager = new DataManager();

        //define the mdp
        SimpleToy mdp = new SimpleToy(20);

        //define the training
        AsyncNStepQLearningDiscreteDense dql = new AsyncNStepQLearningDiscreteDense<SimpleToyState>(mdp, TOY_NET, TOY_ASYNC_QL, manager);

        //enable some logging for debug purposes on toy mdp
        mdp.setFetchable(dql);

        //start the training
        dql.train();

        //useless on toy but good practice!
        mdp.close();

    }

}
