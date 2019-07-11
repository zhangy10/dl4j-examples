package org.deeplearning4j.examples.rl4j.ZacRL;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.NeuralNetFetchable;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;
import java.util.logging.Logger;

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
    final private static int ACTION_SIZE = 5;
    // this is state: each step will have all action values, step id : Va1, Va2, Va3...
    final private static ZacStep[] states = initStates(MAX_STEP, SEED);

    // this is action: how many actions
    private DiscreteSpace actionSpace = new DiscreteSpace(ACTION_SIZE);

    private ObservationSpace<ZacStep> observationSpace = new ArrayObservationSpace(new int[]{ACTION_SIZE});

    // current step
    private ZacStep zacStep;

    // Q learning model reference
    private NeuralNetFetchable<IDQN> fetchable;


    public void printTest(NeuralNetFetchable<IDQN> idqn) {
        INDArray input = Nd4j.create(MAX_STEP, ACTION_SIZE);
        for (int i = 0; i < MAX_STEP; i++) {
            input.putRow(i, Nd4j.create(states[i].toArray()));
        }
        INDArray output = Nd4j.max(idqn.getNeuralNet().output(input), 1);
        Logger.getAnonymousLogger().info(output.toString());
    }

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
     * init every step values
     *
     * @param size
     * @param seed
     * @return
     */
    private static ZacStep[] initStates(int size, int seed) {
        Random rd = new Random(seed);
        ZacStep[] states = new ZacStep[size];
        for (int i = 0; i < size; i++) {
            // init this step with all actions
            double[] values = new double[ACTION_SIZE];
            // init all actions' value
            for (int j = 0; j < ACTION_SIZE; j++) {
                values[j] = rd.nextDouble();
            }
            states[i] = new ZacStep(values, i);
        }
        return states;
    }

    public void setModel(NeuralNetFetchable<IDQN> fetchable) {
        this.fetchable = fetchable;
    }

    @Override
    public void close() {
    }

    /**
     * training is done
     *
     * @return
     */
    @Override
    public boolean isDone() {
        if (zacStep.getStep() == MAX_STEP - 1) {
            System.out.println("[The round done]: " + zacStep.getStep());
            printTest(fetchable);
            return true;
        }
        return false;
    }

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
        return zacStep = states[0];
    }

    /**
     * return a reward value here...
     *
     * @param a
     * @return
     */
    @Override
    public StepReply<ZacStep> step(Integer a) {
        double reward = 0;
        // select max value to action
        if (a == maxIndex(zacStep.getValues()))
            reward += 1;
        zacStep = states[zacStep.getStep() + 1];
        return new StepReply(zacStep, reward, isDone(), new JSONObject("{}"));
    }

    @Override
    public ZacMDP newInstance() {
        return new ZacMDP();
    }
}
