package org.deeplearning4j.examples.convolution.ZacCNN;

import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.AdamUpdater;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.InverseSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public class TrainSplit extends Thread {

    private int id;

    private BlockingQueue<Msg> getQueue = new LinkedBlockingQueue<>();

    private BlockingQueue<Msg> sendQueue;

    private List<BlockingQueue<Msg>> broadcast;

    private boolean isMaster = false;

    private Settings settings;

    private int slaveNum;

    private int epoc;

    private Map<Integer, List<Double>> epocLoss = new TreeMap<>();

//    private INDArray w0 = null;

    private boolean isLinked = false;

    public boolean isEnd = false;


    public TrainSplit(BlockingQueue<Msg> sendQueue, int id, Settings settings, boolean isLinked) {
        this(sendQueue, id, settings);
        this.isLinked = isLinked;
    }

    public TrainSplit(BlockingQueue<Msg> sendQueue, int id, Settings settings) {
        this.sendQueue = sendQueue;
        this.id = id;
        this.settings = settings;
    }

    public TrainSplit(int id, Settings settings, int slaveNum, boolean isLinked) {
        this(id, settings, slaveNum);
        this.isLinked = isLinked;
    }

    public TrainSplit(int id, Settings settings, int slaveNum) {
        this(null, id, settings);
        this.isMaster = true;
        this.slaveNum = slaveNum;
        broadcast = new ArrayList<>();
    }

    public BlockingQueue getQueue() {
        return getQueue;
    }

    public void addSlave(BlockingQueue<Msg> queue) {
        if (broadcast != null) {
            broadcast.add(queue);
        }
    }


    private TrainingListener listener = new TrainingListener() {

        @Override
        public void iterationDone(Model model, int iteration, int epoch) {
            epoc = epoch;
            if (isMaster) {
                MultiLayerNetwork network = (MultiLayerNetwork) model;
                System.out.println("master iteration done: " + iteration + " model score: " + model.score() + " epoch: " + epoch +
                                       " learning rate: " + network.getLearningRate(0));
            }
            List<Double> list = epocLoss.get(epoch);
            if (list == null) {
                list = new ArrayList<>();
                epocLoss.put(epoch, list);
            }
            list.add(model.score());
        }

        @Override
        public void onEpochStart(Model model) {

        }

        @Override
        public void onEpochEnd(Model model) {
            // average loss value
            List<Double> lossList = epocLoss.get(epoc);
            double loss = 0;
            for (Double d : lossList) {
                loss += d;
            }
            loss /= lossList.size();

            MultiLayerNetwork network = (MultiLayerNetwork) model;

            // each epoch end, then will do weight sync
            if (isMaster) {
                List<Msg> msgList = new ArrayList<>();
                try {
                    int num = 0;
                    if (isLinked) {
                        num = 1;
                    } else {
                        num = slaveNum;
                    }
                    while (num > 0) {
                        msgList.add(getQueue.take());
                        num--;
                        System.out.println("Master is taking... left: " + num);
                    }
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                // weight sync
//                    List<Double> lossSet = new ArrayList<>();
                INDArray newP = model.params().dup();

                if (isLinked) {
                    Msg newMsg = msgList.get(0);


                    newMsg.num++;
                    newMsg.parameters.muli(newMsg.num - 1);
                    newP.addi(newMsg.parameters);
                    newP.divi(newMsg.num);
                    System.out.println("Master is divided by: [" + newMsg.num + "]");
                } else {
                    // 1.  average SGD
                    for (Msg m : msgList) {
                        newP.addi(m.parameters);
                    }

                    // average, but learning rate will be smaller and smaller
                    newP.divi(msgList.size() + 1);
                }

//                    // 2.  sum by different weight
//                    Msg masterMsg = new Msg();
//                    masterMsg.loss = model.score();
//                    masterMsg.parameters = model.params().dup();
//                    msgList.add(masterMsg);
//
//                    double all = 0;
//                    for (Msg m : msgList) {
//                        all += m.loss;
//                    }
//
//                    for (Msg m : msgList) {
//                        m.w = (all - m.loss) / ((msgList.size() - 1) * all);
//                    }
//
//                    INDArray newP = msgList.get(0).parameters.mul(msgList.get(0).w);
//                    for (int i = 1; i < msgList.size(); i++) {
//                        newP.addi(msgList.get(i).parameters.mul(msgList.get(i).w));
//                    }


                // 4. adam update
//                    INDArray init = Nd4j.zeros(1, model.params().shape()[1] * 2);
//                    GradientUpdater u = new Adam(network.getLearningRate(0)).instantiate(init, false);
//                    INDArray newP = null;
//                    for (int i = 0; i < msgList.size(); i++) {
//                        // Aw here is required
//                        Msg m = msgList.get(i);
//                        INDArray Aw = w0.sub(m.parameters);
//                        u.applyUpdater(Aw, i, 0);
//                        if (i == msgList.size() - 1) {
//                            newP = Aw;
//                        }
//                    }
//                    newP = w0.sub(newP);


                // 3.  update learning rate
//                    Double l = network.getLearningRate(0);
//                    l *= msgList.size() + 1;
//                    network.setLearningRate(l);

                // update model
                // * setParam.assign will make a copy
                model.setParams(newP);
//                w0 = model.params().dup();

                // if last round, will not send the update to slaves
                if (epoc != settings.epoch - 1) {
                    Msg newMsg = new Msg();
                    newMsg.parameters = newP;
//                        newMsg.l = l;
                    int i = 0;
                    for (BlockingQueue queue : broadcast) {
                        queue.offer(newMsg);
                        i++;
                        System.out.println("master sending to " + i);
                    }
                }
            } else {
                Msg msg = new Msg();

                // if linked, need frist get message from sub node, then send to root
                if (isLinked && !isEnd) {
                    try {
                        System.out.println("node is waiting for sub node... thread: " + id);
                        Msg newMsg = getQueue.take();
                        msg.parameters = model.params().dup();

                        // add with sub node weights
                        newMsg.num++;
                        newMsg.parameters.muli(newMsg.num - 1);
                        msg.parameters.addi(newMsg.parameters);
                        msg.parameters.divi(newMsg.num);
                        msg.num = newMsg.num;

                        System.out.println("divide by: [" + msg.num + "] thread: " + id);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                } else {
                    msg.parameters = model.params();
                    msg.num++;
                }

//                msg.loss = model.score();
                msg.id = id;

                System.out.println("Slave is sending... thread: " + id);
                sendQueue.offer(msg);

                Msg newMsg = null;
                try {
                    System.out.println("Slave is waiting for master... thread: " + id);
                    newMsg = getQueue.take();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println("Slave get new P.... thread: " + id);

                INDArray newP = newMsg.parameters;
//                double l = newMsg.l;
//
//                network.setLearningRate(l);
                model.setParams(newP);
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
    };


    @Override
    public void run() {
        long start = System.currentTimeMillis();

        HarReader reader = new HarReader(settings.numLinesToSkip, settings.height,
            settings.width, settings.channel, settings.numClasses, settings.taskNum, settings.delimiter);
        try {
            reader.initialize(new FileSplit(settings.getFile()));
        } catch (Exception e) {
            e.printStackTrace();
        }

        DataSetIterator iterator = new RecordReaderDataSetIterator(reader, settings.batchSize, settings.labelIndex, settings.numClasses);

        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        if (settings.isNoraml) {
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(iterator);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
            iterator.setPreProcessor(normalizer);
        }

        MultiLayerNetwork model = null;
        if (isMaster) {
            MultiLayerConfiguration conf = getModelConf(settings.numClasses, settings.height, settings.width, settings.channel);
            // send conf to others
            Msg message = new Msg();
            message.confJosn = conf.toJson();

            model = new MultiLayerNetwork(conf);
            model.init();
            // send init to others
            message.parameters = model.params();

            // last weight for sync use
//            w0 = model.params().dup();

            if (broadcast != null) {
                int i = 0;
                for (BlockingQueue send : broadcast) {
                    send.offer(message);
                    i++;
                    System.out.println("send model init to " + i);
                }
            }
        } else {
            try {
                // read from master
                Msg message = getQueue.take();
                System.out.println("Thread " + id + " init model.....");
                MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(message.confJosn);
                model = new MultiLayerNetwork(conf);
                // clone will be false in real case
                model.init(message.parameters, true);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        model.setListeners(listener);
        model.fit(iterator, settings.epoch);

        long end = System.currentTimeMillis();
        // evaluate
        if (isMaster) {
            System.out.println("Save model....");
            String basePath = "/Users/zhangyu/Desktop/";
            try {
                ModelSerializer.writeModel(model, basePath + "multi_model.bin", true);
            } catch (IOException e) {
                e.printStackTrace();
            }
            System.out.println("Save model done!!");
            System.out.println("Total time: " + (end - start) / 1000);

            // each epoc average
            List<Double> averageList = new ArrayList<>();
            Iterator<Integer> it = epocLoss.keySet().iterator();
            while (it.hasNext()) {
                int index = it.next();
                List<Double> epocList = epocLoss.get(index);
                int size = epocList.size();
                double average = 0;
                for (int i = 0; i < size; i++) {
                    average += epocList.get(i);
                }
                averageList.add(average / size);
            }

            System.out.println(averageList);
        }
    }

    private ConvolutionLayer convNet(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        if (in == -1) {
            return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nOut(out).biasInit(bias).convolutionMode(ConvolutionMode.Same).build();
        }
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).convolutionMode(ConvolutionMode.Same).build();
    }

    private SubsamplingLayer maxpooling(String name, int[] kernel, int[] stride) {
        return new SubsamplingLayer.Builder(kernel, stride).name(name).build();
    }

    private DenseLayer full(String name, int out, double bias, double dropOut) {
        return new DenseLayer.Builder().name(name).nOut(out).biasInit(bias).dropOut(dropOut).dist(new NormalDistribution(0, 1)).build();
    }


    public MultiLayerConfiguration getModelConf(int numLabels, int height, int width, int channels) {
        double nonZeroBias = 1; //偏差
        double dropOut = 0.8; //随机丢弃比例
        long seed = 42;

        int kernal = 64;
        int pooling = 2;

        // layer settings
        int c1_out = 36;
        int c2_out = 72;
        int f1_out = 300;

        double learnRate = 0.001;
        double gamma = 0.5;

        return new NeuralNetConfiguration.Builder()
                   .seed(seed)
//                                           .weightInit(WeightInit.NORMAL) //根据给定的分布采样参数
                   .weightInit(WeightInit.DISTRIBUTION)
                   .dist(new NormalDistribution(0.0, 1.0)) //均值为0，方差为1.0的正态分布
                   .activation(Activation.RELU)
                   .updater(new Adam(learnRate))
                   // increase by 1% over 0.001
//                   .updater(new Adam(new InverseSchedule(ScheduleType.EPOCH, learnRate, gamma, 1)))
//                .biasUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 2e-2, 0.1, 100000), 0.9))

                   .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                   //采用除以梯度2范数来规范化梯度防止梯度消失或突变
                   .l2(5 * 1e-4)
                   .list() //13层的网络,第1,3层构建了alexnet计算层，目的是对当前输出的结果做平滑处理，参数有相邻核映射数n=5,规范化常亮k=2,指数常量beta=0.75，系数常量alpha=1e-4
                   .layer(0, convNet("c1", channels, c1_out, new int[]{1, kernal}, new int[]{1, 1}, new int[]{0, 0}, 0))
                   // update padding issue
//                                           .layer(0, convNet("c1", channels, 36, new int[]{1, 64}, new int[]{1, 1}, new int[]{0, 32}, 0))
                   .layer(1, maxpooling("m1", new int[]{1, pooling}, new int[]{1, pooling}))
                   .layer(2, convNet("c2", -1, c2_out, new int[]{1, kernal}, new int[]{1, 1}, new int[]{0, 0}, nonZeroBias))
//                                           .layer(2, convNet("c2", -1, 72, new int[]{1, 64}, new int[]{1, 1}, new int[]{0, 16}, nonZeroBias))
                   .layer(3, maxpooling("m2", new int[]{1, pooling}, new int[]{1, pooling}))
                   .layer(4, full("f1", f1_out, nonZeroBias, dropOut))
                   .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                 .name("o1")
                                 .nOut(numLabels)
                                 .activation(Activation.SOFTMAX)
                                 .build())
                   .backprop(true)
                   .setInputType(InputType.convolutional(height, width, channels))
                   .build();
    }

    public static class Settings {
        int epoch = 20;

        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 0;
        int taskNum = 7352;

        char delimiter = ',';

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        // last pos is label
        // by channel, the label index will be the second one, not actual label index
        int labelIndex = 1;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row

        int numClasses = 6;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
        int batchSize = 8;

        // channel * width = inputwidth
        int channel = 9;
        int height = 1;
        int width = 128;

        boolean isNoraml = false;
        MultiLayerConfiguration conf;

        public Settings(int numLinesToSkip, int taskNum) {
            this.numLinesToSkip = numLinesToSkip;
            this.taskNum = taskNum;
        }

        public File getFile() throws Exception {
            return new File("/Users/zhangyu/Desktop/mDeepBoost/Important/Data/Renew_data/nor_train.csv");
        }
    }

    public class Msg {
        int id;
        INDArray parameters;
        String confJosn;
        int num = 0;

        //        double loss;
        double w;
    }

    public static class Pair {
        int start;
        int end;

        public Pair(int start, int end) {
            this.start = start;
            this.end = end;
        }

        @Override
        public String toString() {
            return "[" + start + "/" + end + "]";
        }
    }

    public static void main(String[] args) {
        int taskNum = 15;
        int total = 7352;
        List<Pair> list = getTask(taskNum, total);

        System.out.println(list.toString());

        TrainSplit master = null;
        List<TrainSplit> slaveList = new ArrayList<>();
        for (int i = 0; i < taskNum; i++) {
            Pair fragment = list.get(i);
            if (i == 0) {
                master = new TrainSplit(i, new Settings(fragment.start, fragment.end), taskNum - 1);
            } else {
                TrainSplit slave = new TrainSplit(master.getQueue(), i, new Settings(fragment.start, fragment.end));
                master.addSlave(slave.getQueue());
                slaveList.add(slave);
            }
        }
        master.start();
        for (TrainSplit t : slaveList) {
            t.start();
        }
    }

    public static List<Pair> getTask(int taskNum, int total) {
        int doNum = total / taskNum;
        int rest = total % taskNum;
        int lastNum = doNum + rest;

        List<Pair> list = new ArrayList<>();
        int start = 0;
        for (int i = 0; i < taskNum; i++) {
            if (i == taskNum - 1) {
                list.add(new Pair(start, lastNum));
            } else {
                list.add(new Pair(start, doNum));
            }
            start += doNum;
        }
        return list;
    }
}
