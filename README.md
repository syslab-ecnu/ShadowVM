# ShadowVM

ShadowVM is an accelerator-aware platform to execute data analytics with CPUs and GPUs.
ShadowVM is based on the decoupled architecture, which consists of a lightweight control plane  and an efficient data plane.
The control plane adapts analytical workloads and offloads computing to the data plane that generates CPP/CUDA code for underlying execution.

  

##  Dependencies

- CUDA 9.0/10.0
- clang 6.0
- gRPC 1.20 with its builtin protobuf 3.7
- JDK 8

## Hardware 

The x86 machine with large main memory is required.

- 16 cores or more (recommended)
- 128 GB or higher (strongly recommended)

Pascal or higher architectures GPUs are required.

- NVIDIA Titan RTX (recommended)
- NVIDIA Tesla P40/P100 (recommended)



## Operating System  
- Ubuntu Server 18.04

## Compiling Method

Compile ShadowVM (CPP/CUDA data plane) by using NVIDIA Nsight with the options as follows. 

- Header search path
  - -I/usr/local/cuda-10.0/samples/common/inc 
  - -I/usr/local/include 
- Library search path
  - -L/usr/local/lib 
- Linked libs 
  - -lprotoc
  - -lcuda 
  - -lprotobuf 
  - -lgrpc++ 
  - -lgrpc 
  - -lpthread 
- Misc 
  - c++11 enabled (-std=c++11)
  - -O2
  - fPIC
  - separate complication
  - exclude com/cpp/cpu-runtime.h & cuda-runtime.cu from compiling


 Compile ShadowVM (Java control plane) by using Maven.
 ```shell
 mvn compile
 ```
 
## Quick Start 


Use Spark vRDD as the control plane (see java_vRDD/src/test/java/Example.java).

```java
// Create Spark Session
// 1. Create Spark Session
SparkSession ss = SparkSession.builder().master("local").getOrCreate();
SparkContext sc = ss.sparkContext();

// 2. Create schema for the dataset
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

// 5. Checkout the DAG structure of pipelines  
System.out.println(queryLineorder.toString());

// 5. connect to the ShadowVM data plane
SVMContext svmCtx = new SVMContext("localhost", 50051, false, false, false);

// 6. Submit to Spark & ShadowVM
vRDD vrdd = queryLineorder.tovRDD(sc, svmCtx);
vrdd.collect();
```

Besides, ShadowVM's data plane can also be simply run without the vRDD control plane.
The examples of 13 SSB SQLs are provided in example/.
 
 ```shell
./ShadowVM
 ```
Configuations can be changed in com/cpp/com.cu and example/.

Here are several important parameters along with their meaning.

- home_path_:  the code generation path of ShadowVM's data plane
- include_path_: the CPP/CUDA header path of ShadowVM's data plane
- data_path_: the path to load table files into ShadowVM's data plane



