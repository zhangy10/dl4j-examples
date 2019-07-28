package org.deeplearning4j.examples.rl4j.ZacRL;

import org.deeplearning4j.rl4j.learning.StepCountable;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.deeplearning4j.rl4j.policy.EpsGreedy;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Random;

public class ZacGreedy<O extends Encodable, A, AS extends ActionSpace<A>> extends EpsGreedy<O, A, AS> {

    final private Policy<O, A> policy;
    final private MDP<O, A, AS> mdp;
    final private int updateStart;
    final private int epsilonNbStep;
    final private Random rd;
    final private float minEpsilon;
    final private StepCountable learning;

    public ZacGreedy(Policy<O, A> policy, MDP<O, A, AS> mdp, int updateStart, int epsilonNbStep, Random rd, float minEpsilon, StepCountable learning) {
        // do nothing...
        super(policy, mdp, updateStart, epsilonNbStep, rd, minEpsilon, learning);

        this.policy = policy;
        this.mdp = mdp;
        this.updateStart = updateStart;
        this.epsilonNbStep = epsilonNbStep;
        this.rd = rd;
        this.minEpsilon = minEpsilon;
        this.learning = learning;
    }

    public NeuralNet getNeuralNet() {
        return policy.getNeuralNet();
    }

    public A nextAction(INDArray input) {

        ZacMDP currentMDP = (ZacMDP) mdp;

        float ep = getEpsilon();


        if (learning.getStepCounter() % 500 == 1)
            System.out.println("Zac EP value: " + ep + " " + learning.getStepCounter());


        // TODO only random learn
//        if (rd.nextFloat() > ep)
//            return policy.nextAction(input);
//        else {
        return (A) currentMDP.getNextAction();
//        }
    }

    public float getEpsilon() {
        return Math.min(1f, Math.max(minEpsilon, 1f - (learning.getStepCounter() - updateStart) * 1f / epsilonNbStep));
    }


}
