package ShadowVM.spark.rdd

import ShadowVM.{Pipeline, SVMContext}
import ShadowVM.proto.types.ShadowVMAPIGrpc
import ShadowVM.spark.rdd.vRDD.getDirectDependency
import org.apache.spark._
import org.apache.spark.rdd.RDD
import _root_.io.grpc.ManagedChannelBuilder

class vPartition(idx: Int) extends Partition {
  override def index: Int = idx
}

class DirectPartitioner(partitions: Int) extends Partitioner {
  override def numPartitions: Int = partitions
  override def getPartition(key: Any): Int =
    key.asInstanceOf[Int] % numPartitions
}

object vRDD {
  def getDirectDependency(rdds: Seq[vRDD], numPartitions: Int) : Seq[Dependency[_]] =
    rdds.map(rdd => new ShuffleDependency(rdd, new DirectPartitioner(numPartitions)))

  def createSourceRDD(pipeline: Pipeline, sc: SparkContext) : RDD[Pipeline] = {
    sc.makeRDD(Seq(pipeline))
  }
}

class vRDD(
    _sc: SparkContext,
    vRDDs: Array[vRDD],
    numPartitions: Int,
    maxPartitions: Int,
    pipeline: Pipeline,
    svmctx: SVMContext)
  extends RDD[(Int, Pipeline)] (_sc, getDirectDependency(vRDDs, numPartitions)) {

  protected def createStub(host: String, port: Int): ShadowVMAPIGrpc.ShadowVMAPIBlockingStub = {
    val builder = ManagedChannelBuilder.forAddress(host, port)
    builder.usePlaintext()
    ShadowVMAPIGrpc.newBlockingStub(builder.build())
  }

  override def compute(split: Partition, context: TaskContext): Iterator[(Int, Pipeline)] = {
    // TODO: We assume the dependency only contains ONE partition.
    val preObjects =
      dependencies.map(dep => {
        SparkEnv.get.shuffleManager
          .getReader(
              dep.asInstanceOf[ShuffleDependency[Int, Pipeline, Pipeline]].shuffleHandle,
              split.index,
              split.index + 1,
              context)
          .read().asInstanceOf[Iterator[(Int, Pipeline)]].toSeq.head._2
      })

    val stub = createStub(svmctx.host, svmctx.port)
    val pipelineInfo = pipeline.getPipelineInfo.toBuilder
      .setCodegen(svmctx.codegen)
      .setDebug(svmctx.debug)
      .setIsGPU(svmctx.isGPU)
      .build()
    val ret = stub.runPipelineCMD(pipelineInfo)
    if (!ret.getSuccess) {
      println(ret.getInfo)
    }
    (0 until maxPartitions).map((_, pipeline)).toIterator
  }

  override protected def getPartitions: Array[Partition] =
    (0 until numPartitions).map(new vPartition(_)).toArray
}

class vSourceRDD(
    _sc: SparkContext,
    numPartitions: Int,
    maxPartitions: Int,
    pipeline: Pipeline,
    svmctx: SVMContext)
  extends vRDD(_sc, Array(), numPartitions, maxPartitions, pipeline, svmctx) {

  override def compute(split: Partition, context: TaskContext): Iterator[(Int, Pipeline)] = {

    val stub = createStub(svmctx.host, svmctx.port)
    val pipelineInfo = pipeline.getPipelineInfo.toBuilder
      .setCodegen(svmctx.codegen)
      .setDebug(svmctx.debug)
      .setIsGPU(svmctx.isGPU)
      .build()
    val ret = stub.runPipelineCMD(pipelineInfo)
    if (!ret.getSuccess) {
      println(ret.getInfo)
    }
    (0 until maxPartitions).map((_, pipeline)).toIterator
  }
}


