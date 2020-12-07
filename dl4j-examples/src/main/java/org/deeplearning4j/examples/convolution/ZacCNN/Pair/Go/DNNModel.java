package org.deeplearning4j.examples.convolution.ZacCNN.Pair.Go;

import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.Solver;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.CrashReportingUtil;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.heartbeat.utils.TaskUtils;

public class DNNModel extends MultiLayerNetwork {

    public DNNModel(MultiLayerConfiguration conf) {
        super(conf);
    }

    @Override
    public void fit(DataSetIterator iterator) {
        try{
            fitHelper(iterator);
        } catch (OutOfMemoryError e){
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    public synchronized void fitHelper(DataSetIterator iterator){
        // we're wrapping all iterators into AsyncDataSetIterator to provide background prefetch - where appropriate
        DataSetIterator iter;
        boolean destructable = false;
        if (iterator.asyncSupported()) {
            iter = new AsyncDataSetIterator(iterator, Math.min(Nd4j.getAffinityManager().getNumberOfDevices() * 2, 2), true);
            destructable = true;
        } else {
            iter = iterator;
        }

        for (TrainingListener tl : trainingListeners) {
            tl.onEpochStart(this);
        }

        LayerWorkspaceMgr workspaceMgr;
        if(getLayerWiseConfigurations().getTrainingWorkspaceMode() == WorkspaceMode.NONE){
            workspaceMgr = LayerWorkspaceMgr.noWorkspaces();
        } else {
            workspaceMgr = LayerWorkspaceMgr.builder()
                               .with(ArrayType.ACTIVATIONS, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                               .with(ArrayType.INPUT, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                               .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                               .with(ArrayType.BP_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                               .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                               .with(ArrayType.RNN_BP_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                               //Note for updater working memory, we have the option to re-use WS_ALL_LAYERS_ACT or FF/BP_WORKING_MEM
                               // as these should be closed by the time updaters are executed
                               //Generally, WS_ALL_LAYERS_ACT will be the larger of the two, so we'll use this
                               .with(ArrayType.UPDATER_WORKING_MEM, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                               .build();
        }
        workspaceMgr.setHelperWorkspacePointers(helperWorkspaces);

        update(TaskUtils.buildTask(iter));
        if (!iter.hasNext() && iter.resetSupported()) {
            iter.reset();
        }
        long time1 = System.currentTimeMillis();
        while (iter.hasNext()) {

            for (TrainingListener tl : trainingListeners) {
                if (tl instanceof BatchListener) {
                    ((BatchListener) tl).iterationStart(this, getIterationCount(), getEpochCount());
                }
            }

            DataSet next = iter.next();
            long time2 = System.currentTimeMillis();

            lastEtlTime.set((time2 - time1));

            if (next.getFeatures() == null || next.getLabels() == null)
                break;

            // TODO: basically we want to wrap internals of this loop into workspace


            boolean hasMaskArrays = next.hasMaskArrays();

            if (layerWiseConfigurations.getBackpropType() == BackpropType.TruncatedBPTT) {
                doTruncatedBPTT(next.getFeatures(), next.getLabels(), next.getFeaturesMaskArray(),
                    next.getLabelsMaskArray(), workspaceMgr);
            } else {
                if (hasMaskArrays)
                    setLayerMaskArrays(next.getFeaturesMaskArray(), next.getLabelsMaskArray());

                setInput(next.getFeatures());
                setLabels(next.getLabels());

                if (solver == null) {
                    try (MemoryWorkspace wsO = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                        solver = new Solver.Builder().configure(conf()).listeners(getListeners()).model(this)
                                     .build();
                    }
                }

                //TODO CACHE
                solver.optimize(workspaceMgr);
            }

            if (hasMaskArrays)
                clearLayerMaskArrays();

            time1 = System.currentTimeMillis();
            synchronizeIterEpochCounts();
        }

        if (!trainingListeners.isEmpty()) {
            for (TrainingListener tl : trainingListeners) {
                tl.onEpochEnd(this);
            }
        }

        clearLayersStates();

        if (destructable)
            ((AsyncDataSetIterator) iter).shutdown();

        incrementEpochCount();
    }

}
