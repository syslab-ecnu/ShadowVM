package ShadowVM;

import ShadowVM.proto.types.*;
import ShadowVM.spark.rdd.vRDD;
import ShadowVM.spark.rdd.vSourceRDD;
import org.apache.hadoop.util.hash.Hash;
import org.apache.spark.SparkContext;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class Pipeline implements Serializable {
    private final PipelineInfo pipelineInfo;

    private ObjectSchema outputSchema;

    private final ArrayList<Pipeline> prePipelines;

    public Pipeline(PipelineBuilder builder) {
        this.pipelineInfo = builder.getPipelineInfo();
        this.outputSchema = builder.getOutputSchema();
        this.prePipelines = builder.getPrePipelines();
    }

    public Pipeline(String id) {
        pipelineInfo = PipelineInfo.newBuilder().setId(id).build();
        prePipelines = new ArrayList<Pipeline>();
    }

    public ObjectSchema schema() {
        return outputSchema;
    }

    public static Pipeline readCSV(String name, String path, ObjectSchema schema) {
        PipelineBuilder pipeline = new PipelineBuilder("load_"+name);
        HashMap<String, String> sourceArgs = new HashMap<>();
        sourceArgs.put("type", "file");
        sourceArgs.put("path", path);
        pipeline.addOperator(OperatorInfo.Type.Source, sourceArgs, schema);
        HashMap<String, String> loaderArgs = new HashMap<>();
        loaderArgs.put("col_spilter", "|");
        loaderArgs.put("row_spilter", "\n");
        pipeline.addOperator(OperatorInfo.Type.Loader, loaderArgs);
        pipeline.addOperator(OperatorInfo.Type.Sink, new HashMap<String, String>());
        return pipeline.build();
    }

    public PipelineBuilder newPipeline(String id, ObjectSchema schema) {
        PipelineBuilder pipeline = new PipelineBuilder(id);
        HashMap<String, String> args = new HashMap<>();
        args.put("type", "object");
        pipeline.addOperator(OperatorInfo.Type.Source, args, this.outputSchema);
        pipeline.setOutputSchema(schema);
        pipeline.addPrePipelines(this);
        return pipeline;
    }

    public vRDD tovRDD(SparkContext sc, SVMContext svmctx) {
        if (prePipelines.isEmpty()) {
            return new vSourceRDD(sc, 1, 1, this, svmctx);
        }
        vRDD[] preRDDs = new vRDD[prePipelines.size()];
        for (int i = 0; i < prePipelines.size(); i++) {
            preRDDs[i] = prePipelines.get(i).tovRDD(sc, svmctx);
        }

        return new vRDD(sc, preRDDs, 1, 1, this, svmctx);
    }


    public PipelineInfo getPipelineInfo() { return pipelineInfo;}

    public String toString() {
        StringBuilder result = new StringBuilder();
        int i = 0;
        for (Pipeline prePipeline: prePipelines) {
            result.append(prePipeline.toString());
        }
        result.append("========================\n");
        result.append(String.format("Pipeline ID: %s\n", pipelineInfo.getId()));
        result.append("Dependencies: [");
        for (Pipeline prePipeline: prePipelines) {
            result.append(prePipeline.pipelineInfo.getId());
            if (i != prePipelines.size() - 1)
                result.append(",");
            i++;
        }
        result.append("]\n");
        for(OperatorInfo op : pipelineInfo.getWorkflowList()) {
            result.append(String.format("Operator: %s, Args: %s\n", op.getType(), op.getArgsMap().toString()));
        }
        return result.toString();
    }

}
