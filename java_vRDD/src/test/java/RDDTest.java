import ShadowVM.*;
import ShadowVM.proto.types.*;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;
import org.apache.spark.sql.SparkSession;
import org.junit.Assert;
import org.junit.Test;
import ShadowVM.spark.rdd.*;
import org.apache.spark.*;
import scala.Tuple2;

public class RDDTest {
    Server rpcServer;

    @Test public void TestReplaceExpr() {
        Pipeline lineorder = SSBHelper.LoadLineorder();
        Pipeline date = SSBHelper.LoadDate();
        PipelineBuilder pb = lineorder.newPipeline("testReplace", new ObjectSchema());
        pb = pb.joinWith(date, "", "");
        Assert.assertTrue(
            pb.replaceExpression("$d_year==1997 || $d_year==1998").equals(
                "$(1.4)==1997 || $(1.4)==1998"
            )
        );
        Assert.assertTrue(
                pb.replaceExpression("($lo_revenue-$lo_supplycost)/100").equals(
                        "($(0.12)-$(0.13))/100"
                )
        );
    }

    @Test
    public void TestBuildvRDD() {
        SparkSession ss = SparkSession.builder().master("local").getOrCreate();
        SparkContext sc = ss.sparkContext();
        SVMContext svmCtx = new SVMContext("localhost", 9898, false, false, false);
        Pipeline p = SSBHelper.LoadLineorder();
        vSourceRDD sourceRDD = new vSourceRDD(sc, 1, 1, p, svmCtx);
        vRDD vrdd = new vRDD(sc, new vRDD[]{sourceRDD}, 1, 1, p, svmCtx);

        Object obj = vrdd.collect();
        Tuple2<Object, Pipeline>[] a = (Tuple2<Object, Pipeline>[]) obj;
        for (Tuple2<Object, Pipeline> objectSVMObjectTuple2 : a) {
            System.out.println(objectSVMObjectTuple2._2.getPipelineInfo().getId());
        }
    }

    @Test
    public void TestBuildRelation() {
        SparkSession ss = SparkSession.builder().master("local").getOrCreate();
        SparkContext sc = ss.sparkContext();

        ObjectSchema dateSchema = SSBHelper.getDateSchema();
        ObjectSchema queryDateSchema = SSBHelper.getQuery43DateSchema();

        ObjectSchema lineorderSchema = SSBHelper.getLineorderSchema();
        ObjectSchema queryLineorderSchema = SSBHelper.getQuery43LineorderSchema();
        Pipeline queryCustomer = Pipeline.readCSV("customer", "path/customer", SSBHelper.getCustomerSchema());
        Pipeline date = Pipeline.readCSV("date", "path/date", dateSchema);
        Pipeline queryDate = date
            .newPipeline("query_date", queryDateSchema)
            .filter("$d_year==1997 || $d_year==1998")
            .map("$d_datekey", "d_year")
            .sink()
            .build();

        Pipeline lineorder = Pipeline.readCSV("lineorder", "path/lineorder", lineorderSchema);
        Pipeline queryLineorder = lineorder
            .newPipeline("query_lineorder", queryLineorderSchema)
            .joinWith(queryDate, "$lo_orderdate", "$d_datekey")
            .joinWith(queryCustomer, "$lo_custkey", "$c_custkey")
            .reduceBy("$d_year, $c_nation", "($lo_revenue-$lo_supplycost)/100")
            .sink()
            .build();

        createRPCServer();
        SVMContext svmctx = new SVMContext("localhost", 9191, false, false, false);
        vRDD rdd = queryLineorder.tovRDD(sc, svmctx);
        rdd.collect();
    }

    @Test
    public void TestSSBQ43() {
        SparkSession ss = SparkSession.builder().master("local").getOrCreate();
        SparkContext sc = ss.sparkContext();

        Pipeline p = SSBHelper.Q43();

        createRPCServer();
        SVMContext svmctx = new SVMContext("localhost", 9191, false, false, false);
        vRDD rdd = p.tovRDD(sc, svmctx);
        rdd.collect();
    }

    private static class RunPipelineImpl extends ShadowVMAPIGrpc.ShadowVMAPIImplBase {
        @Override
        public void runPipelineCMD(PipelineInfo pipelineInfo, StreamObserver<Ret> observer) {
            System.out.println(pipelineInfo.getId());
            observer.onNext(Ret.getDefaultInstance());
            observer.onCompleted();
        }
    }

    public void createRPCServer() {
        try {
            rpcServer = ServerBuilder
                    .forPort(9191)
                    .addService(new RunPipelineImpl())
                    .build()
                    .start();
        } catch (Exception e) {
            e.printStackTrace();
        }
        Runtime.getRuntime().addShutdownHook(new Thread() {
            @Override
            public synchronized void start() {
                System.out.println("*** shutting down gRPC server since JVM is shutting down");
                rpcServer.shutdown();
                System.out.println("*** server shut down");
            }
         });
    }
}
