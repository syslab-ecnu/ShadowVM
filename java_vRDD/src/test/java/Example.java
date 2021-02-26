import ShadowVM.ObjectColumn;
import ShadowVM.ObjectSchema;
import ShadowVM.Pipeline;
import ShadowVM.SVMContext;
import ShadowVM.proto.types.ValType;
import ShadowVM.spark.rdd.vRDD;
import org.apache.spark.SparkContext;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.SparkSession;
import org.junit.Test;

public class Example {
    @Test
    public void example() {
        // 0. Create a RPC server for test
//        RDDTest.createRPCServer();

        // 1. Create Spark Session
        SparkSession ss = SparkSession.builder().master("local").getOrCreate();
        SparkContext sc = ss.sparkContext();

        // 2. Create schema for the table/object
        ObjectSchema dateSchema = new ObjectSchema();
        dateSchema.appendColumn(new ObjectColumn("d_datekey", ValType.Int32));
        dateSchema.appendColumn(new ObjectColumn("d_year", ValType.Int32));
        dateSchema.appendColumn(new ObjectColumn("d_month", ValType.Int32));
        dateSchema.appendColumn(new ObjectColumn("d_day", ValType.Int32));

        ObjectSchema lineorderSchema = new ObjectSchema();
        lineorderSchema.appendColumn(new ObjectColumn("lo_orderkey", ValType.Int32));
        lineorderSchema.appendColumn(new ObjectColumn("lo_linenumber", ValType.Int32));
        lineorderSchema.appendColumn(new ObjectColumn("lo_orderdate", ValType.Int32));
        lineorderSchema.appendColumn(new ObjectColumn("lo_revenue", ValType.Int32));

        // 3. Load the tables
        Pipeline loadDate = Pipeline.readCSV("date", "path-to-date.csv", dateSchema);
        Pipeline loadLineorder = Pipeline.readCSV("lineorder", "path-to-lineorder.csv", lineorderSchema);

        // 4. Build query pipelines
        ObjectSchema queryDateSchema = new ObjectSchema();
        queryDateSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryDateSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryDateSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline queryDate = loadDate.newPipeline("query-date", queryDateSchema)
                .filter("$d_year > 1998")
                .map("$d_datekey", "$d_year")
                .sink()
                .build();

        ObjectSchema queryLineorderSchema = new ObjectSchema();
        queryLineorderSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryLineorderSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryLineorderSchema.appendColumn(new ObjectColumn(ValType.Int32));
        Pipeline queryLineorder = loadLineorder.newPipeline("query-lineorder", queryLineorderSchema)
                .joinWith(queryDate, "$lo_orderdate", "$d_datekey")
                .reduceBy("$d_year", "$lo_revenue/100")
                .sink()
                .build();

        System.out.println(queryLineorder.toString());

        // 5. Submit to Spark
        SVMContext svmCtx = new SVMContext("localhost", 9191, false, false, false);
        vRDD vrdd = queryLineorder.tovRDD(sc, svmCtx);
        vrdd.collect();
    }
}
