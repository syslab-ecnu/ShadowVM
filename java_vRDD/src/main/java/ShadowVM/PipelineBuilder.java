package ShadowVM;

import ShadowVM.proto.types.ColumnInfo;
import ShadowVM.proto.types.OperatorInfo;
import ShadowVM.proto.types.PipelineInfo;

import java.util.*;

public class PipelineBuilder {
    private PipelineInfo.Builder builder;
    private ObjectSchema outputSchema;
    private ArrayList<Pipeline> prePipelines;

    private TreeSet<String> columns;
    private Map<String, String> columnMap;

    public PipelineBuilder(String id) {
        builder = PipelineInfo.newBuilder().setId(id);
        prePipelines = new ArrayList<Pipeline>();
        columns = new TreeSet<>();
        columnMap = new HashMap<String, String>();
    }

    public PipelineInfo getPipelineInfo() {
        return builder.build();
    }

    public ObjectSchema getOutputSchema() {
        return outputSchema;
    }

    public ArrayList<Pipeline> getPrePipelines() {
        return prePipelines;
    }

    public Pipeline build() { return new Pipeline(this);}

    public void addOperator(OperatorInfo.Type type, Map<String, String> args, ObjectSchema schema) {
        OperatorInfo op = OperatorInfo.newBuilder()
                .setType(type)
                .setId(builder.getId())
                .putAllArgs(args)
                .setOutput(schema.getObjectInfo())
                .build();
        outputSchema = schema;
        builder.addWorkflow(op);
    }

    public void addOperator(OperatorInfo.Type type, Map<String, String> args) {
        addOperator(type, args, outputSchema);
    }

    public void setOutputSchema(ObjectSchema schema) {
        this.outputSchema = schema;
    }

    public void addPrePipelines(Pipeline...pipelines) {
        int tableIdx = prePipelines.size();
        for (Pipeline p : pipelines) {
            int columnIndx = 0;
            for (ObjectColumn col : p.schema().columns) {
                if (col.columnName != null) {
                    String key = "$" + col.columnName;
                    if (col.columnName.isEmpty())
                        key = "$UNKNOWN";
                    String value = String.format("$(%d.%d)", tableIdx, columnIndx);
                    columns.add(key);
                    columnMap.put(key, value);
                }
                columnIndx++;
            }
            tableIdx++;
        }
        this.prePipelines.addAll(Arrays.asList(pipelines));
    }

    public String replaceExpression(String expr) {
        // FIXME: parser
        String newExpr = expr;
//
//        for (int i = 0; i < prePipelines.size(); i++) {
//            Pipeline p = prePipelines.get(i);
//            ObjectSchema s = p.schema();
//            for (int j = 0; j < s.columns.size(); j++) {
//                String target = "$("+ i +"."+ j +")";
//                String src = "$"+s.columns.get(j).columnName;
//                newExpr = newExpr.replace(src, target);
//                src = "$"+s.table+"."+s.columns.get(j).columnName;
//                newExpr = newExpr.replace(src, target);
//            }
//        }
        for (Iterator iter = columns.descendingIterator(); iter.hasNext(); ) {
            String key = (String)iter.next();
            newExpr = newExpr.replace(key, columnMap.get(key));
        }
        return newExpr;
    }

    public PipelineBuilder filter(String condition) {
        HashMap<String, String> args = new HashMap<>();
        args.put("predication", replaceExpression(condition));
        this.addOperator(OperatorInfo.Type.Filter, args);
        return this;
    }

    public PipelineBuilder map(String... cols) {
        // FIXME: parser
        String expr = "";
        for (int i = 0; i < cols.length; i++) {
            String newCol = replaceExpression(cols[i]);
            String[] parts = newCol.substring(1).split("/.");
            if (parts.length == 1) {
                outputSchema.columns.get(i).columnName = cols[i].substring(1);
            } else {
                outputSchema.columns.get(i).columnName = parts[1];
            }

            if (i != 0) expr += ",";
            expr += newCol;
        }
        HashMap<String, String> args = new HashMap<>();
        args.put("select", expr);
        this.addOperator(OperatorInfo.Type.Map, args);
        return this;
    }

    public PipelineBuilder joinWith(Pipeline joinTable, String leftKey, String rightKey) {
        this.addPrePipelines(joinTable);
        HashMap<String, String> args = new HashMap<>();
        args.put("lkey", replaceExpression(leftKey));
        args.put("rkey", replaceExpression(rightKey));
        this.addOperator(OperatorInfo.Type.JoinWith, args);
        return this;
    }

    public PipelineBuilder reduceBy(String dim, String metric) {
        HashMap<String, String> args = new HashMap<>();
        args.put("dim", replaceExpression(dim));
        args.put("metric", replaceExpression(metric));
        this.addOperator(OperatorInfo.Type.ReduceBy, args);
        return this;
    }

    public PipelineBuilder sink() {
        this.addOperator(OperatorInfo.Type.Sink, new HashMap<String, String>());
        return this;
    }
}
