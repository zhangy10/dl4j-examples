package org.deeplearning4j.examples.convolution.ZacCNN.Pair.Go;

import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Config;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Go.base.BaseTrain;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.GradientInfo;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Model.MDLModel;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.SystemRun;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Utils.Message;
import org.deeplearning4j.examples.convolution.ZacCNN.SplitListener;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;

//import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public class TrainGo extends BaseTrain {

    // GoSGD
    private double goScale = 0;

    // GoSGD p for Bernoulli
    private double p = 0.01;

    private CommCount commCount;
    private List<Message> finalAvg = new ArrayList<>();

    public TrainGo(BlockingQueue<Message> sendQueue, int id, Config settings, SplitListener splitListener, String modelFile, double goScale) {
        super(sendQueue, id, settings, splitListener, modelFile);
        this.masterQueue = sendQueue;
        this.id = id;
        this.settings = settings;

        this.splitListener = splitListener;
        this.modelFile = modelFile;

        this.goScale = goScale;
        this.commCount = new CommCount(id);
    }

    public TrainGo(int id, Config settings, int slaveNum, SplitListener splitListener, String modelFile, double goScale) {
        this(null, id, settings, splitListener, modelFile, goScale);
        this.isMaster = true;
        this.slaveNum = slaveNum;
        broadcast = new ArrayList<>();
    }

    private TrainingListener listener = new BatchListener() {

        public void aggregate(Model model, Message msg) {
            INDArray local = model.params().dup();
            local.muli(goScale / (goScale + msg.goScale));
            msg.parameters.muli(msg.goScale / (goScale + msg.goScale));
            // local + remote
            local.addi(msg.parameters);
            model.setParams(local);

            goScale = goScale + msg.goScale;
            commCount.addGet();
        }

        public Message sendModel(Model model) {
            goScale = goScale / 2;
            Message msg = new Message(id);
            msg.parameters = model.params().dup();
            msg.goScale = goScale;
            commCount.addSend();
            return msg;
        }

        @Override
        public void iterationStart(Model model, int iteration, int epoch) {

            List<Message> msgList = new ArrayList<>();
            try {
                int num = getQueue.size();
                while (num > 0) {
                    Message msg = getQueue.take();
                    // mainly for final average
                    if (msg.state == Message.FINAL_STATE) {
                        finalAvg.add(msg);
                    } else {
                        msgList.add(msg);
                    }
                    num--;
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            // aggregate all message
            for (Message msg : msgList) {
                aggregate(model, msg);
                System.out.println("T_" + id + " [Getting & Aggregate message] {from id: " + msg.id + "}, batch: " + batchID + ", goScale: " + goScale);
            }
        }


        @Override
        public void iterationDone(Model model, int iteration, int epoch) {
            epoc = epoch;
            batchID = iteration;

            // get batch time
            DNNModel network = (DNNModel) model;
            long bend = System.currentTimeMillis();
            long time = bend - bstart;
            bstart = bend;
            printInfo("master iteration done: " + iteration + " model score: " + model.score() + " epoch: " + epoch +
                          " learning rate: " + network.getLearningRate(0) + " time: " + time);

            batchTime.add(time);

            // get average loss value
            List<Double> list = epocLoss.get(epoch);
            if (list == null) {
                list = new ArrayList<>();
                epocLoss.put(epoch, list);
            }
            list.add(model.score());

            switch (SystemRun.policy) {
                case Bernoulli:
                    // Go share
                    if (StdRandom.bernoulli(p)) {
                        share(model);
                    }
                    break;
                case BATCH:
                    share(model);
                    break;
//                case HALF_EPOC:
//                    break;
            }

            // add l1 changes
            appendGradient(epoc, model.params().norm1Number().doubleValue());
        }

        double l1start = 0;

        double l1end = 0;

        double l1gradient = 0;

        @Override
        public void onEpochStart(Model model) {
            bstart = System.currentTimeMillis();

            // get L1 start point
            l1start = model.params().norm1Number().doubleValue();
            appendGradient(epoc, l1start);
        }

        @Override
        public void onEpochEnd(Model model) {
            printInfo("[-------------onEpochEnd----------------] batchID: " + batchID);

            // get L1 end point
            l1end = model.params().norm1Number().doubleValue();
            l1gradient = l1start - l1end;
            GradientInfo.append(id, l1start, l1end, l1gradient);

            if (SystemRun.policy == SystemRun.SyncPolicy.EPOC) {
                share(model);
            }

            // update done, and test for accuracy
            if (SystemRun.isTestRound) {
                test((DNNModel) model);
            }
        }

        @Override
        public void onForwardPass(Model model, List<INDArray> activations) {

        }

        @Override
        public void onForwardPass(Model model, Map<String, INDArray> activations) {

        }

        @Override
        public void onGradientCalculation(Model model) {

        }

        @Override
        public void onBackwardPass(Model model) {

        }

        private void share(Model model) {
            printInfo("T_" + id + "[----------SYNC-----------] batchID: " + batchID);
            // share or send model to target
            TrainGo target = GoRun.getRandomTarget(id);
            if (target != null) {
                Message msg = sendModel(model);

                System.out.println("T_" + id + " [Sending message] {from id: " + id + " to " + target.getTaskID() + "}, batch: " + batchID + ", goScale: " + goScale);
                target.getQueue().offer(msg);
            }
        }

    };

    @Override
    protected void loadData() {
        trainIterator = loading(settings.getDataPath(), settings.getTaskNum());
        testIterator = loading(settings.getTestPath(), SystemRun.isIID ? settings.getTaskNum() * GoRun.getNodeNum() : settings.getTaskNum());
    }

    @Override
    protected MultiLayerNetwork getModel() {
        DNNModel model = null;
        MultiLayerConfiguration conf = null;
        if (isMaster) {
            try {
                model = (DNNModel) ModelSerializer.restoreMultiLayerNetwork(existingFile);
            } catch (Exception e) {
                System.out.println("No existing file found, model will be initialzed in MAIN");
            } finally {
                if (model == null) {
                    conf = MDLModel.getNetwork(modelType, settings);
                    model = new DNNModel(conf);
                    model.init();
                }
            }
            System.out.println(model.summary());

            // for test, save model
            try {
                Nd4j.saveBinary(model.params(), new File("/Users/zhangyu/Desktop/cache"));
            } catch (IOException e) {
                e.printStackTrace();
            }

            // send conf to others
            Message message = new Message(id);
            if (conf != null) {
                message.confJosn = conf.toJson();
            }
            // send init to others
            message.parameters = model.params();

            if (broadcast != null) {
                int i = 0;
                for (BlockingQueue send : broadcast) {
                    send.offer(message);
                    i++;
                    System.out.println("send model init to " + i);
                }
            }
        } else {
            if (model == null) {
                try {
                    // read from master
                    Message message = getQueue.take();
                    System.out.println("Thread " + id + " init model.....");
                    conf = MultiLayerConfiguration.fromJson(message.confJosn);
                    model = new DNNModel(conf);

                    // not use copy from master
                    model.init();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
        model.setListeners(listener);
        return model;
    }

    @Override
    protected void afterTrain(String output, Model model) {

        // train done
        if (isMaster) {
            // comm info
            String comm = commCount.toString();
            // L1 gradient print
            String gradients = printAllGradient();

            // take cache first
            for (Message m : finalAvg) {
                output += m.output;
                gradients += m.log;
                comm += m.commCount.toString();
            }

            List<Message> finalMsg = new ArrayList<>();
            // gather all gradient info
            int num = avaliableNum - finalAvg.size();
            while (num > 0) {
                System.out.println("Master is [waiting]... left: " + num + ", already have: " + finalAvg.size());
                Message msg = null;
                try {
                    msg = getQueue.take();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                if (msg != null) {
                    if (msg.state == Message.FINAL_STATE) {
                        num--;
                        output += msg.output;
                        gradients += msg.log;
                        comm += msg.commCount.toString();

                        finalMsg.add(msg);
                        System.out.println("Master is taking... left: " + num);
                    } else {
                        System.out.println("Master remove invalid message from: T_" + msg.id);
                    }
                }
            }

            // go final average for test
            if (SystemRun.isIID) {
                String finalResult = finalAverage(finalMsg, (MultiLayerNetwork) model);
                output += finalResult;
                System.out.println(finalResult);
            } else {
                // Non iid: ecah node will have ecah test model and result
                // time should be the longest like master, different ac will be kept in each log
                // do nothing...
            }


            // add other node's log
            output += "\n\n--------------------\n\n" + comm + "\n-------------------\n\n";
            output += gradients;
            System.out.println(comm);
            System.out.println(gradients);

            if (splitListener != null) {
                splitListener.trainDone(output);
            }
        } else {
            // ac + time + commcount info to master
            // send gradients to master
            Message msg = new Message(id);
            msg.log = printAllGradient();
            msg.output = output;
            msg.commCount = commCount;
            // for final test
            msg.parameters = model.params().dup();

            msg.state = Message.FINAL_STATE;
            System.out.println("T_" + id + " Slave is [FINALY] sending to master...");
            masterQueue.offer(msg);
        }
    }

    private String finalAverage(List<Message> list, MultiLayerNetwork model) {
        String result = "";
        INDArray newP = model.params().dup();
        for (Message m : list) {
            newP.addi(m.parameters);
        }
        // only for cases if other tasks finish earlier than master
        for (Message m : finalAvg) {
            newP.addi(m.parameters);
        }
        newP.divi(list.size() + finalAvg.size() + 1);
        model.setParams(newP);

        // Save model
        printInfo("[Save final model!-----------]");
        try {
            ModelSerializer.writeModel(model, modelFile, true);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Final test
        Evaluation eval = model.evaluate(testIterator);
        result = "[Final Result]:------------------\nEpoc ID: " + epoc + "\n";
        result += eval.stats() + "\n\n";
        result += "[Final result] ac: " + eval.accuracy() + ", pr: " + eval.precision() + ", re: " + eval.recall() + ", f1: " + eval.f1() + "\n\n";
        return result;
    }

}
