package org.deeplearning4j.examples.rl4j.ZacRL;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import java.util.List;
import java.util.Map;

public class ZacTrain {


    private static TrainingListener listener = new TrainingListener() {
        @Override
        public void iterationDone(Model model, int batchid, int epoch) {
            System.out.print("loss: " + model.score() + " batchId: " + batchid + " epoc: " + epoch);
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

    static int actionSize = 5;
    static int busySize = 1;

    public static QLearning.QLConfiguration TOY_QL =
        new QLearning.QLConfiguration(
            123,   //Random seed
            // basic info
            3000,//Max step By epoch
            15, //Max step
            40, //Max size of experience replay
            16,

            // ------
            100,   //target update (hard)
            10,     //num step noop warmup
            0.00,  //reward scaling
            0.95,  //gamma
            10.0,  //td-error clipping

            // exploration
            0.1f,  //min epsilon
            10,  //num step for eps greedy anneal
            false   //double DQN
        );

    // Configuration set
    public static DQNFactoryStdDense.Configuration TOY_NET =
        DQNFactoryStdDense.Configuration.builder()
            .l2(0.01).updater(new Adam(1e-2)).numLayer(3).numHiddenNodes(actionSize * 2)
            .listeners(new TrainingListener[]{listener})
            .build();


    public static void main(String[] args) throws Exception {
        start();
    }

//    public static void load() throws IOException {
//
//        //showcase serialization by using the trained agent on a new similar mdp (but render it this time)
//
//        //define the mdp from gym (name, render)
//        MDP mdp2 = new ZacMDP_1();
//
//        //load the previous agent
//        DQNPolicy<Box> pol2 = DQNPolicy.load("/Users/zber/Desktop/toy_pol2");
//
//        //evaluate the agent
//        double rewards = 0;
//        int size = 20;
//        for (int i = 0; i < size; i++) {
//            mdp2.reset();
//            double reward = pol2.play(mdp2);
//            rewards += reward;
//            Logger.getAnonymousLogger().info("Reward: " + reward);
//        }
//
//        Logger.getAnonymousLogger().info("average: " + rewards / size);
//
//    }

    public static void start() throws Exception {
        //record the training data in rl4j-data in a new folder
        DataManager manager = new DataManager();

        ZacMDP mdp = new ZacMDP(actionSize, busySize);

        //define the training
        ZacQLearningImpl<ZacStep> dql = new ZacQLearningImpl(mdp, new ZacDQNFactory(TOY_NET), TOY_QL, manager);
        mdp.setDQN(dql);

        //start the training
        dql.train();

        // save model
        DQNPolicy<ZacStep> pol = (DQNPolicy) dql.getPolicy();
        pol.save("/Users/zhangyu/Desktop/zac_dqn");

        // evaluate
        play(mdp);

        //useless on toy but good practice!
        mdp.close();
    }

    public static void testPlay(ZacMDP dmp) {
        if (dmp.getDQN() == null) {
            return;
        }

//        int leftNodeSize = 2 * (dmp.getActionSize() - 1);
//
//        INDArray startRow = Nd4j.create(1, dmp.getActionSize());
//        startRow.putRow(0, Nd4j.create(dmp.getEnvData()));
//
//        while (leftNodeSize > 0) {
//            // select one node
//            INDArray output = Nd4j.max(dmp.getDQN().getNeuralNet().output(startRow), 1);
//            System.out.println("Env value: " + startRow.toString());
//            System.out.println("Output: " + output);
//            int maxAction = Learning.getMaxAction(output);
//            System.out.println("Select: " + maxAction);
//
//
//
//
//
//            leftNodeSize--;
//        }

    }

    public static void play(ZacMDP dmp) {

        double reward = dmp.getDQN().getPolicy().play(dmp);
        System.out.println("All reward: " + reward + "; actions: " + "");

    }

}
