package org.deeplearning4j.examples.convolution.ZacCNN.Pair.Go.base;

import org.deeplearning4j.examples.convolution.ZacCNN.FileUtils;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Config;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.DataSet;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Model.MDLModel;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.SystemRun;
import org.deeplearning4j.examples.convolution.ZacCNN.Pair.Utils.TaskSplit;
import org.deeplearning4j.examples.convolution.ZacCNN.SplitListener;

import java.io.File;
import java.util.*;

public abstract class BaseRun {

    protected static int taskNum;
    protected DataSet.DataType dataType;
    protected MDLModel.Type modelType = MDLModel.Type.LENET1D;
    protected String basePath;
    protected String logPath;
    protected String log = "";
    protected SplitListener donelistener;

    protected String modelFile = "";
    protected String existingFile = "";

    protected int seed = 1234;
    protected String tag = "";

    private static final String TRAIN = "train";
    private static final String TEST = "test";
    private static final String CSV = ".csv";
    private static final String SPLIT = "_";

    // black list, exclude subjects
    public static List<Integer> harBlackList = new ArrayList<>(Arrays.asList(8, 15, 27, 16, 12, 13, 4, 20));
    public static List<Integer> emgBlackList = new ArrayList<>(Arrays.asList(21));
    public static List<Integer> mheBlackList = new ArrayList<>(Arrays.asList(7));

    public static Map<DataSet.DataType, List<Integer>> blackList = new HashMap<>();

    static {
        blackList.put(DataSet.DataType.HAR, harBlackList);
        blackList.put(DataSet.DataType.EMG, emgBlackList);
        blackList.put(DataSet.DataType.MHe, mheBlackList);
    }

    public BaseRun(int taskNum, DataSet.DataType dataType, String basePath, SplitListener donelistener) {
        this(taskNum, dataType, basePath, donelistener, MDLModel.Type.LENET1D);
    }

    public BaseRun(int taskNum, DataSet.DataType dataType, String basePath, SplitListener donelistener, MDLModel.Type modelType) {
        this.taskNum = taskNum;
        this.dataType = dataType;
        this.basePath = basePath;
        this.donelistener = donelistener;
        this.modelType = modelType;
    }

    public void setType(MDLModel.Type modelType) {
        this.modelType = modelType;
    }

    public BaseRun setRandomSeed(int seed) {
        this.seed = seed;
        return this;
    }

    public BaseRun setTag(String tag) {
        this.tag = tag;
        return this;
    }

    protected SplitListener listener = new SplitListener() {
        @Override
        public void trainDone(String output) {
            // save output to files
            FileUtils.write(log + output, logPath);
            if (donelistener != null) {
                donelistener.trainDone(null);
            }
        }
    };

    public void run() {
        String savePath = basePath + taskNum + "_" + dataType + "_" + modelType + "_" + tag;
        modelFile = savePath + "_model.bin";
        logPath = savePath + "_log.txt";
        existingFile = basePath + dataType + "_" + modelType + "_model.bin";

        start();
    }

    protected abstract void start();

    public List<Config> getConfig() {
        List<Config> configs = new ArrayList<>();
        if (SystemRun.isIID) {
            // split task
            List<TaskSplit.Pair> list = TaskSplit.getTask(taskNum, DataSet.getConfig(dataType).getTaskNum());
            log = list.toString() + "\n";
            System.out.println(log);

            for (TaskSplit.Pair pair : list) {
                Config config = DataSet.getNewConfig(dataType);
                config.setTaskRange(pair.start, pair.end);
                configs.add(config);
            }
        } else {
            // read different files
            Config config = DataSet.getConfig(dataType);
            List<File> trainDir = getFiles(new File(config.getDataPath()).listFiles());
            List<File> testDir = getFiles(new File(config.getTestPath()).listFiles());
            configs = selectSubject(taskNum, trainDir, testDir);
        }
        return configs;
    }

    private List<File> getFiles(File[] files) {
        List<File> list = new ArrayList<>();
        // order file name
        Arrays.sort(files, new Comparator<File>() {

            @Override
            public int compare(File f1, File f2) {
                String[] n1 = f1.getName().split(SPLIT);
                String[] n2 = f2.getName().split(SPLIT);
                return Integer.compare(Integer.valueOf(n1[1]), Integer.valueOf(n2[1]));
            }
        });
        // filter invalid file name
        if (files != null) {
            for (File f : files) {
                if (f.getName().contains(TRAIN) || f.getName().contains(TEST)) {
                    list.add(f);
                }
            }
        }
        return list;
    }

    private List<Config> selectSubject(int select, List<File> trainDir, List<File> testDir) {
        List<Config> configs = new ArrayList<>();
        // Random select task number of index
        Random r = new Random(seed);
        List<Integer> index = new ArrayList<>();
        List<Integer> black = blackList.get(dataType);

        int num = taskNum;
        while (num > 0) {
            int next = r.nextInt(trainDir.size());
            if (!index.contains(next) && !black.contains(next)) {
                index.add(next);
                num--;
            }
        }
        log = "[Select subject]: " + index.toString() + "\n";
        System.out.println(log);

        for (int i = 0; i < index.size(); i++) {
            File train = trainDir.get(index.get(i));
            File test = testDir.get(index.get(i));
            String[] fileName = train.getName().replace(CSV, "").split(SPLIT);
            Integer dataNum = Integer.valueOf(fileName[2]);

            Config config = DataSet.getNewConfig(dataType);
            config.setTaskRange(0, dataNum)
                .setDataPath(train.getAbsolutePath())
                .setTestPath(test.getAbsolutePath());
            configs.add(config);
        }
        return configs;
    }

}
