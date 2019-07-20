package org.deeplearning4j.examples.rl4j.ZacRL;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.NeuralNetFetchable;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscrete;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * ZacStep: state
 * <p>
 * Integer: action id
 * <p>
 * DiscreteSpace: action
 */
public class ZacMDP implements MDP<ZacStep, Integer, DiscreteSpace> {

    // for random initalize each step's values
    final private static int SEED = 1234;

    //TODO 10 steps toy (always +1 reward2 actions),
    // toylong (1000 steps),
    // toyhard (7 actions, +1 only if actiion = (step/100+step)%7,
    // and toyStoch (like last but reward has 0.10 odd to be somewhere else).

    // A toy MDP where the agent should find the maximum to get the reward.

    /**
     * one round has Max_Step steps, and each step has ACTION_SIZE actions, if step is the last one, then this round is finished
     * <p>
     * in our project, step size is node size, action set is how many nodes master can select for each sync (each sync, select one action)
     * <p>
     * if step of each round is the last, then finish...
     */
    final private static int MAX_STEP = 5;
    final private static int ACTION_SIZE = 5; // number of nodes
    final private static int numSleep = 1;

    // this is state: each step will have all action values, step id : Va1, Va2, Va3...
    private static Map<State, Double> valueMap = new HashMap<>();
    private static ZacStep[] states;
    private static Map<State, Double> rewardMap = new HashMap<>();
    private static int round = 0;
    private static double action;

    static {
        CreateState();
        CreateReward();
        states = initStates(MAX_STEP, SEED);
        action = -1.0;
    }

    // this is action: how many actions  (有几种选择)
    private DiscreteSpace actionSpace = new DiscreteSpace(ACTION_SIZE);
    private ObservationSpace<ZacStep> observationSpace = new ArrayObservationSpace(new int[]{ACTION_SIZE});
    // current step
    private ZacStep zacStep;
    // current reply
    private StepReply zacReply;
    // Q learning model reference
    private QLearningDiscrete fetchable;

    private List<String> rewardList = new ArrayList<>();

    private static void CreateState() {
        valueMap.put(State.FREE, 3.0);
        valueMap.put(State.SLEEP, 2.0);
//        valueMap.put(State.SYNC, 1.0);
        valueMap.put(State.DONE, 0.0);
    }

    private static void CreateReward() {
        rewardMap.put(State.FREE, 1.0);
        rewardMap.put(State.SLEEP, -0.5);
//        rewardMap.put(1.0,0.0);
        rewardMap.put(State.DONE, -0.75);
    }


    // initiate
    // free :3  sleep:2 sync:1 done:0
    // how many sleep , how many nodes,

    public static int maxIndex(double[] values) {
        double maxValue = -Double.MIN_VALUE;
        int maxIndex = -1;
        for (int i = 0; i < values.length; i++) {
            if (values[i] > maxValue) {
                maxValue = values[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * count how many sleep in initial array
     */
    private static boolean countSleep(double[] values) {
        int count = 0;
        for (double i : values) {
            if (i == valueMap.get(State.SLEEP)) {
                count++;
            }
        }
        return count == numSleep;
    }

    /**
     * init every step values
     * State  = 环境
     *
     * @param size
     * @param seed
     * @return
     */
    private static ZacStep[] initStates(int size, int seed) {


        Random rd = new Random(seed);
        ZacStep[] states = new ZacStep[size];


//        double[] line1 = new double[] {1.0, 2.0, 3.0, 4.0, 5.0};  //4
//        double[] line2 = new double[] {5.0, 4.0, 3.0, 2.0, 1.0};  //1
//        double[] line3 = new double[] {4.0, 5.0, 3.0, 2.0, 1.0};  //2
//        double[] line4 = new double[] {4.0, 3.0, 5.0, 2.0, 1.0};  //3
//
//
//        states[0] = new ZacStep(line1, 0);
//        states[1] = new ZacStep(line2, 1);
//        states[2] = new ZacStep(line3, 2);
//        states[3] = new ZacStep(line4, 3);


        for (int i = 0; i < size; i++) {
            // init this step with all actions
            double[] values = new double[ACTION_SIZE];
            // init all actions' value
            for (int j = 0; j < ACTION_SIZE; j++) {
                values[j] = valueMap.get(State.FREE);
            }
            values[0] = valueMap.get(State.SLEEP);
//                while(!countSleep(values)){
//                values[rd.nextInt(ACTION_SIZE)] =valueMap.get(State.SLEEP);
//            }
            states[i] = new ZacStep(values, i);
        }

//        zacStep = getStep(round);
        return states;
    }

    private static ZacStep getStep(int round) {
        return states[round].clone();

    }

    public void printTest(NeuralNetFetchable<IDQN> idqn) {
        INDArray input = Nd4j.create(MAX_STEP, ACTION_SIZE);
        for (int i = 0; i < MAX_STEP; i++) {
            input.putRow(i, Nd4j.create(states[i].toArray()));
        }
//        INDArray output = Nd4j.max(idqn.getNeuralNet().output(input), 1);
//        Logger.getAnonymousLogger().info(output.toString());
    }

    public void setModel(QLearningDiscrete fetchable) {
        this.fetchable = fetchable;
    }

    @Override
    public void close() {
        System.out.println();
        for (String log : rewardList) {
            System.out.println(log);
        }
    }

    /**
     * training is done
     * one training round is done
     *
     * @return
     */
    @Override
    public boolean isDone() {
//        if (zacStep.getStep() == MAX_STEP - 1) {
//            System.out.println("[The round done]: " + zacStep.getStep());
//            printTest(fetchable);
//            return true;
//        }
//        return false;
//        if (action == valueMap.get(State.DONE)) {
//            return true;
//        }

        for (double i : zacStep.getValues()) {
            if (i != valueMap.get(State.DONE))
                return false;

        }

//        if (zacReply.getReward() >= 0) {
//            rewardList.add(String.format("[-----------<step: %d, %s>------------]", zacStep.getStep(), zacReply.toString()));
//        }
        return true;


    }

    //if()


    @Override
    public ObservationSpace<ZacStep> getObservationSpace() {
        return observationSpace;
    }

    @Override
    public DiscreteSpace getActionSpace() {
        return actionSpace;
    }

    @Override
    public ZacStep reset() {
        rewardMap.put(State.SLEEP, -0.5);
        action = -1.0;
//        if (round == MAX_STEP - 1) {
//            round = 0;
//            return zacStep = getStep(round);
//        }
//        zacStep = getStep(round);
//        round++;
        zacStep = getStep(0);
        return zacStep;

    }

    /**
     * return a reward value here...
     *
     * @param a
     * @return
     */
    @Override
    public StepReply<ZacStep> step(Integer a) {
        action = zacStep.getValues()[a];
        double reward = 0;
        // select max value to action

        if (isDone()) {
            reward += 1;
        } else if (action == valueMap.get(State.DONE)) {
            reward += rewardMap.get(State.DONE);
        } else if (action == valueMap.get(State.SLEEP)) {
            reward += rewardMap.get(State.SLEEP);

        } else {
            reward -= 0.04;
        }

//        if(action == valueMap.get(State.DONE)) reward += rewardMap.get(State.DONE);
//
//
//        if(action == valueMap.get(State.SLEEP)) reward += rewardMap.get(State.SLEEP);
//

        if (rewardMap.get(State.SLEEP) < 0) {

            rewardMap.put(State.SLEEP, rewardMap.get(State.SLEEP) + 0.1);
        }
        //
//        if (a == maxIndex(zacStep.getValues()))
//            reward += 1;
//
//        reward += rewardMap.get(action);
//        // update the reward of SLEEP as the model is training
//        if (rewardMap.get(valueMap.get(State.SLEEP)) <= 0.5) {
//            rewardMap.put(valueMap.get(State.SLEEP), rewardMap.get(valueMap.get(State.SLEEP)) + 0.1);
//        }
//        zacStep.setValues(a,valueMap.get(State.DONE));
////        if(isDone()) reward += 100;
//        System.out.println("[doing step]: " + zacStep.toString());
//
//
//
//        if (zacStep.getStep() > ACTION_SIZE) {
//            int gap = zacStep.getStep() - ACTION_SIZE;
//            reward -= gap;
//        }


//        double max = zacStep.getValues()[maxIndex(zacStep.getValues())];

//        if (zacStep.getValues()[a] == max ) {
//            reward++;
//        }
        System.out.println("[doing step]: " + zacStep.toString());
        zacStep.setValues(a, valueMap.get(State.DONE));

        zacReply = new StepReply(zacStep, reward, isDone(), new JSONObject("{}"));

        return zacReply;
    }

    @Override
    public ZacMDP newInstance() {
        return new ZacMDP();
    }

    public enum State {
        FREE,
        SLEEP,
        DONE,
//        SYNC
    }
}

