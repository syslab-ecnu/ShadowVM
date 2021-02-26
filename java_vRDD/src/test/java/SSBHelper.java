import ShadowVM.*;
import ShadowVM.proto.types.OperatorInfo;
import ShadowVM.proto.types.StoType;
import ShadowVM.proto.types.ValType;

import java.io.File;
import java.io.FileInputStream;
import java.util.Map;

public class SSBHelper {
    static public ObjectSchema getDateSchema() {
        ObjectSchema schema = new ObjectSchema("date");
        schema.setStorage(StoType.HostMem);
        schema.appendColumn(new ObjectColumn("d_datekey", ValType.Int32));
        schema.appendColumn(new ObjectColumn("d_date", ValType.String, 18));
        schema.appendColumn(new ObjectColumn("d_dayofweek", ValType.String, 9));
        schema.appendColumn(new ObjectColumn("d_month", ValType.String, 9));
        schema.appendColumn(new ObjectColumn("d_year", ValType.Int32));
        schema.appendColumn(new ObjectColumn("d_yearmonthnum", ValType.String, 8));
        schema.appendColumn(new ObjectColumn("d_yearmonth", ValType.Int32));
        schema.appendColumn(new ObjectColumn("d_daynuminweek", ValType.Int32));
        schema.appendColumn(new ObjectColumn("d_daynuminmonth", ValType.Int32));
        schema.appendColumn(new ObjectColumn("d_daynuminyear", ValType.Int32));
        schema.appendColumn(new ObjectColumn("d_monthnuminyear", ValType.Int32));
        schema.appendColumn(new ObjectColumn("d_weeknuminyear", ValType.Int32));
        schema.appendColumn(new ObjectColumn("d_sellingseason", ValType.String, 12));
        schema.appendColumn(new ObjectColumn("d_lastdayinweekfl", ValType.Int32));
        schema.appendColumn(new ObjectColumn("d_lastdayinmonthfl", ValType.Int32));
        schema.appendColumn(new ObjectColumn("d_holidayfl", ValType.Int32));
        schema.appendColumn(new ObjectColumn("d_weekdayfl", ValType.Int32));
        return schema;
    }

    static public ObjectSchema getLineorderSchema() {
        ObjectSchema schema = new ObjectSchema();
        schema.setStorage(StoType.HostMem);
        schema.appendColumn(new ObjectColumn("lo_orderkey", ValType.Int32));
        schema.appendColumn(new ObjectColumn("lo_linenumber", ValType.Int32));
        schema.appendColumn(new ObjectColumn("lo_custkey", ValType.Int32));
        schema.appendColumn(new ObjectColumn("lo_partkey", ValType.Int32));
        schema.appendColumn(new ObjectColumn("lo_suppkey", ValType.Int32));
        schema.appendColumn(new ObjectColumn("lo_orderdate", ValType.Int32));
        schema.appendColumn(new ObjectColumn("lo_orderpriority", ValType.Int32));
        schema.appendColumn(new ObjectColumn("lo_shippriority", ValType.String, 15));
        schema.appendColumn(new ObjectColumn("lo_quantity", ValType.Int32));
        schema.appendColumn(new ObjectColumn("lo_extendedprice", ValType.Int32));
        schema.appendColumn(new ObjectColumn("lo_ordtotalprice", ValType.Int32));
        schema.appendColumn(new ObjectColumn("lo_discount", ValType.Int32));
        schema.appendColumn(new ObjectColumn("lo_revenue", ValType.Int32));
        schema.appendColumn(new ObjectColumn("lo_supplycost", ValType.Int32));
        schema.appendColumn(new ObjectColumn("lo_tax", ValType.Int32));
        schema.appendColumn(new ObjectColumn("lo_commitdate", ValType.Int32));
        schema.appendColumn(new ObjectColumn("lo_shipmode", ValType.String, 10));
        return schema;
    }

    static public ObjectSchema getCustomerSchema() {
        ObjectSchema schema = new ObjectSchema();
        schema.setStorage(StoType.HostMem);
        schema.appendColumn(new ObjectColumn("c_custkey", ValType.Int32));
        schema.appendColumn(new ObjectColumn("c_name", ValType.String, 25));
        schema.appendColumn(new ObjectColumn("c_address", ValType.String, 25));
        schema.appendColumn(new ObjectColumn("c_city", ValType.String, 10));
        schema.appendColumn(new ObjectColumn("c_nation", ValType.String, 15));
        schema.appendColumn(new ObjectColumn("c_region", ValType.String, 15));
        schema.appendColumn(new ObjectColumn("c_phone", ValType.String, 15));
        schema.appendColumn(new ObjectColumn("c_mktsegment", ValType.String, 15));
        return schema;
    }

    static public ObjectSchema getPartSchema() {
        ObjectSchema schema = new ObjectSchema();
        schema.setStorage(StoType.HostMem);
        schema.appendColumn(new ObjectColumn("p_partkey", ValType.Int32));
        schema.appendColumn(new ObjectColumn("p_name", ValType.String, 22));
        schema.appendColumn(new ObjectColumn("p_mfgr", ValType.String, 8));
        schema.appendColumn(new ObjectColumn("p_category", ValType.String, 8));
        schema.appendColumn(new ObjectColumn("p_brand1", ValType.String, 10));
        schema.appendColumn(new ObjectColumn("p_color", ValType.String, 11));
        schema.appendColumn(new ObjectColumn("p_type", ValType.String, 25));
        schema.appendColumn(new ObjectColumn("p_size", ValType.Int32));
        schema.appendColumn(new ObjectColumn("p_container", ValType.String, 10));
        return schema;
    }

    static public ObjectSchema getSupplierSchema() {
        ObjectSchema schema = new ObjectSchema();
        schema.setStorage(StoType.HostMem);
        schema.appendColumn(new ObjectColumn("s_suppkey", ValType.Int32));
        schema.appendColumn(new ObjectColumn("s_name", ValType.String, 25));
        schema.appendColumn(new ObjectColumn("s_address", ValType.String, 25));
        schema.appendColumn(new ObjectColumn("s_city", ValType.String, 10));
        schema.appendColumn(new ObjectColumn("s_nation", ValType.String, 15));
        schema.appendColumn(new ObjectColumn("s_region", ValType.String, 12));
        schema.appendColumn(new ObjectColumn("s_phone", ValType.String, 15));
        return schema;
    }

    static public ObjectSchema getQuery43DateSchema() {
        ObjectSchema dateOutSchema = new ObjectSchema();
        dateOutSchema.setStorage(StoType.HostMem);
        dateOutSchema.appendColumn(new ObjectColumn(ValType.Int32));
        dateOutSchema.appendColumn(new ObjectColumn(ValType.Int32));
        dateOutSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        return dateOutSchema;
    }

    static public ObjectSchema getQuery43LineorderSchema() {
        ObjectSchema lineorderOutSchema = new ObjectSchema();
        lineorderOutSchema.setStorage(StoType.HostMem);
        for (int i = 0; i < 12; i++)
            lineorderOutSchema.appendColumn(new ObjectColumn(ValType.Int32));
        return lineorderOutSchema;
    }

    static public Pipeline LoadDate() {
        ObjectSchema schema = getDateSchema();
        return Pipeline.readCSV("date", "path/date", schema);
    }

    static public Pipeline LoadLineorder() {
        ObjectSchema schema = getLineorderSchema();
        return Pipeline.readCSV("lineorder", "path/lineorder", schema);
    }

    static public Pipeline LoadCustomer() {
        ObjectSchema schema = getCustomerSchema();
        return Pipeline.readCSV("customer", "path/customer", schema);
    }

    static public Pipeline LoadPart() {
        ObjectSchema schema = getPartSchema();
        return Pipeline.readCSV("part", "path/part", schema);
    }

    static public Pipeline LoadSupplier() {
        ObjectSchema schema = getSupplierSchema();
        return Pipeline.readCSV("supplier", "path/supplier", schema);
    }

    static public Pipeline Q43Date(Pipeline date) {
        ObjectSchema dateOutSchema = new ObjectSchema();
        dateOutSchema.setStorage(StoType.HostMem);
        dateOutSchema.appendColumn(new ObjectColumn(ValType.Int32));
        dateOutSchema.appendColumn(new ObjectColumn(ValType.Int32));
        dateOutSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        return date.newPipeline("q43-date", dateOutSchema)
                .filter("$d_year==1997 || $d_year==1998")
                .map("$d_datekey", "$d_year")
                .sink()
                .build();
    }

    static public Pipeline Q43Part(Pipeline part) {
        ObjectSchema partOutSchema = new ObjectSchema();
        partOutSchema.setStorage(StoType.HostMem);
        partOutSchema.appendColumn(new ObjectColumn(ValType.Int32));
        partOutSchema.appendColumn(new ObjectColumn(ValType.String, 10));
        partOutSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        String partCondition = "StrCmp($p_category,\\\"MFGR#14\\\",8)==0";
        return part.newPipeline("q43-part", partOutSchema)
                .filter(partCondition)
                .map("$p_partkey", "$p_brand1")
                .sink()
                .build();
    }

    static public Pipeline Q43Supplier(Pipeline sup) {
        ObjectSchema supplierOutSchema = new ObjectSchema();
        supplierOutSchema.setStorage(StoType.HostMem);
        supplierOutSchema.appendColumn(new ObjectColumn(ValType.Int32));
        supplierOutSchema.appendColumn(new ObjectColumn(ValType.String, 25));
        supplierOutSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        String supplierCondition = "StrCmp($s_nation,\\\"UNITED STATES\\\",15)==0";
        return sup.newPipeline("q43-supplier", supplierOutSchema)
                .filter(supplierCondition)
                .map("$s_suppkey", "$s_city")
                .sink()
                .build();
    }

    static public Pipeline Q43Customer(Pipeline customer) {
        ObjectSchema customerOutSchema = new ObjectSchema();
        customerOutSchema.setStorage(StoType.HostMem);
        customerOutSchema.appendColumn(new ObjectColumn(ValType.Int32));
        customerOutSchema.appendColumn(new ObjectColumn(ValType.String, 15));
        customerOutSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        String customerCondition = "StrCmp($c_region,\\\"AMERICA\\\",15)==0";
        return customer.newPipeline("q43-customer", customerOutSchema)
                .filter(customerCondition)
                .map("$c_custkey", "$c_nation")
                .sink()
                .build();
    }

    static public Pipeline Q43Lineorder(Pipeline lineorder, Pipeline dateQuery, Pipeline partQuery, Pipeline supplierQuery, Pipeline customerQuery) {
        ObjectSchema lineorderOutSchema = new ObjectSchema();
        lineorderOutSchema.setStorage(StoType.HostMem);
        for (int i = 0; i < 12; i++)
            lineorderOutSchema.appendColumn(new ObjectColumn(ValType.Int32));
        return lineorder.newPipeline("q43", lineorderOutSchema)
                .joinWith(partQuery, "$lo_partkey", "$p_partkey")
                .joinWith(dateQuery, "$lo_orderdate", "$d_datekey")
                .joinWith(supplierQuery, "$lo_suppkey", "$s_suppkey")
                .joinWith(customerQuery, "$lo_custkey", "$c_custkey")
                .reduceBy("$d_year, $s_city, $p_brand1", "($lo_revenue-$lo_supplycost)/100")
                .sink()
                .build();
    }

    static public Pipeline Q11() {
//        select sum(lo_extendedprice*lo_discount) as revenue
//          from lineorder, date
//          where lo_orderdate = d_datekey
//              and d_year = 1993
//              and lo_discount between 1 and 3
//              and lo_quantity < 25

        ObjectSchema queryDateSchema = new ObjectSchema();
        queryDateSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryDateSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline date = LoadDate().newPipeline("q11-date", queryDateSchema)
                .filter("$d_year == 1993")
                .map("$d_datekey")
                .sink()
                .build();
        ObjectSchema q11Schema = new ObjectSchema();
        for (int i = 0; i < 9; i++)
            q11Schema.appendColumn(new ObjectColumn(ValType.Int32));
        Pipeline q11 = LoadLineorder().newPipeline("q11", q11Schema)
                .filter("$lo_discount >= 1 and $lo_discount <= 3 and $lo_quantity < 25")
                .joinWith(date, "$lo_orderdate", "$d_datekey")
                .reduceBy("", "($lo_extendedprice*$lo_discount)/1000")
                .sink()
                .build();
        return q11;
    }

    static public Pipeline Q12() {
//        select sum(lo_extendedprice*lo_discount) as revenue
//           from lineorder, date
//           where lo_orderdate = d_datekey
//              and d_yearmonthnum = 199401
//              and lo_discount between 4 and 6
//              and lo_quantity between 26 and 35

        ObjectSchema queryDateSchema = new ObjectSchema();
        queryDateSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryDateSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline date = LoadDate().newPipeline("q12-date", queryDateSchema)
                .filter("$d_yearmonthnum == 199401")
                .map("$d_datekey")
                .sink()
                .build();
        ObjectSchema outputSchema = new ObjectSchema();
        for (int i = 0; i < 9; i++)
            outputSchema.appendColumn(new ObjectColumn(ValType.Int32));
        Pipeline q = LoadLineorder().newPipeline("q12", outputSchema)
                .filter("$lo_discount >= 4 and $lo_discount <= 6 and $lo_quantity >= 26 and $lo_quantity <=35")
                .joinWith(date, "$lo_orderdate", "$d_datekey")
                .reduceBy("", "($lo_extendedprice*$lo_discount)/1000")
                .sink()
                .build();
        return q;
    }

    static public Pipeline Q13() {
//        select sum(lo_extendedprice*lo_discount) as revenue
//           from lineorder, date
//           where lo_orderdate = d_datekey
//              and d_weeknuminyear = 6 and d_year = 1994
//              and lo_discount between 5 and 7
//              and lo_quantity between 36 and 40
        ObjectSchema queryDateSchema = new ObjectSchema();
        queryDateSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryDateSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline date = LoadDate().newPipeline("q13-date", queryDateSchema)
                .filter("$d_weeknuminyear == 6 and $d_year == 1994")
                .map("$d_datekey")
                .sink()
                .build();
        ObjectSchema outputSchema = new ObjectSchema();
        for (int i = 0; i < 9; i++)
            outputSchema.appendColumn(new ObjectColumn(ValType.Int32));
        Pipeline q = LoadLineorder().newPipeline("q13", outputSchema)
                .filter("$lo_discount >= 5 and $lo_discount <= 7 and $lo_quantity >= 36 and $lo_quantity <= 40")
                .joinWith(date, "$lo_orderdate", "$d_datekey")
                .reduceBy("", "($lo_extendedprice*$lo_discount)/1000")
                .sink()
                .build();
        return q;
    }

    static public Pipeline Q21() {
//    select sum(lo_revenue), d_year, p_brand

//        from lineorder, date, part, supplier
//        where lo_orderdate = d_datekey
//            and lo_partkey = p_partkey
//            and lo_suppkey = s_suppkey
//            and p_category = 'MFGR#12'
//            and s_region = 'AMERICA'
//        group by d_year, p_brand1
//        order by d_year, p_brand1
        ObjectSchema queryDateSchema = new ObjectSchema();
        queryDateSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryDateSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryDateSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline date = LoadDate().newPipeline("q21-date", queryDateSchema)
                .map("$d_datekey", "$d_year")
                .sink()
                .build();

        ObjectSchema queryPartSchema = new ObjectSchema();
        queryPartSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryPartSchema.appendColumn(new ObjectColumn(ValType.String, 10));
        queryPartSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline part = LoadPart().newPipeline("q21-part", queryPartSchema)
                .filter("StrCmp($p_category,\\\"MFGR#12\\\",8)==0")
                .map("$p_partkey", "$p_brand1")
                .sink()
                .build();

        ObjectSchema querySuppSchema = new ObjectSchema();
        querySuppSchema.appendColumn(new ObjectColumn(ValType.Int32));
        querySuppSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline supp = LoadSupplier().newPipeline("q21-supplier", querySuppSchema)
                .filter("StrCmp($s_region,\\\"AMERICA\\\",15)==0")
                .map("$s_suppkey")
                .sink()
                .build();

        ObjectSchema outputSchema = new ObjectSchema();
        for (int i = 0; i < 9; i++)
            outputSchema.appendColumn(new ObjectColumn(ValType.Int32));
        Pipeline p = LoadLineorder().newPipeline("q21", outputSchema)
                .joinWith(date, "$lo_orderdate", "$d_datekey")
                .joinWith(part, "$lo_partkey", "$p_partkey")
                .joinWith(supp, "$lo_suppkey", "$s_suppkey")
                .reduceBy("$d_year,$p_brand1", "$lo_revenue/100")
                .sink()
                .build();
        return p;
    }

    static public Pipeline Q22() {
//        select sum(lo_revenue), d_year, p_brand1
//            from lineorder, date, part, supplier
//            where lo_orderdate = d_datekey
//                and lo_partkey = p_partkey
//                and lo_suppkey = s_suppkey
//                and p_brand1 between 'MFGR#2221' and 'MFGR#2228'
//                and s_region = 'ASIA'
//            group by d_year, p_brand1
//            order by d_year, p_brand1

        ObjectSchema queryDateSchema = new ObjectSchema();
        queryDateSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryDateSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryDateSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline date = LoadDate().newPipeline("q22-date", queryDateSchema)
                .map("$d_datekey", "$d_year")
                .sink()
                .build();

        ObjectSchema queryPartSchema = new ObjectSchema();
        queryPartSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryPartSchema.appendColumn(new ObjectColumn(ValType.String, 10));
        queryPartSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline part = LoadPart().newPipeline("q22-part", queryPartSchema)
                .filter("StrCmp($p_brand1,\\\"MFGR#2221\\\",10)>=0 and StrCmp($p_brand1,\\\"MFGR#2228\\\",10)<=0")
                .map("$p_partkey", "$p_brand1")
                .sink()
                .build();

        ObjectSchema querySuppSchema = new ObjectSchema();
        querySuppSchema.appendColumn(new ObjectColumn(ValType.Int32));
        querySuppSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline supp = LoadSupplier().newPipeline("q22-supplier", querySuppSchema)
                .filter("StrCmp($s_region,\\\"ASIA\\\",15)==0")
                .map("$s_suppkey")
                .sink()
                .build();

        ObjectSchema outputSchema = new ObjectSchema();
        for (int i = 0; i < 9; i++)
            outputSchema.appendColumn(new ObjectColumn(ValType.Int32));
        Pipeline p = LoadLineorder().newPipeline("q22", outputSchema)
                .joinWith(date, "$lo_orderdate", "$d_datekey")
                .joinWith(part, "$lo_partkey", "$p_partkey")
                .joinWith(supp, "$lo_suppkey", "$s_suppkey")
                .reduceBy("$d_year,$p_brand1", "$lo_revenue/100")
                .sink()
                .build();
        return p;
    }

    static public Pipeline Q23() {
//        select sum(lo_revenue), d_year, p_brand1
//            from lineorder, date, part, supplier
//            where lo_orderdate = d_datekey
//                and lo_partkey = p_partkey
//                and lo_suppkey = s_suppkey
//                and p_brand1 = 'MFGR#2221'
//                and s_region = 'EUROPE'
//            group by d_year, p_brand1
//            order by d_year, p_brand1


        ObjectSchema queryDateSchema = new ObjectSchema();
        queryDateSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryDateSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryDateSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline date = LoadDate().newPipeline("q23-date", queryDateSchema)
                .map("$d_datekey", "$d_year")
                .sink()
                .build();

        ObjectSchema queryPartSchema = new ObjectSchema();
        queryPartSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryPartSchema.appendColumn(new ObjectColumn(ValType.String, 10));
        queryPartSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline part = LoadPart().newPipeline("q23-part", queryPartSchema)
                .filter("StrCmp($p_brand1,\\\"MFGR#2221\\\",10)==0")
                .map("$p_partkey", "$p_brand1")
                .sink()
                .build();

        ObjectSchema querySuppSchema = new ObjectSchema();
        querySuppSchema.appendColumn(new ObjectColumn(ValType.Int32));
        querySuppSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline supp = LoadSupplier().newPipeline("q23-supplier", querySuppSchema)
                .filter("StrCmp($s_region,\\\"EUROPE\\\",15)==0")
                .map("$s_suppkey")
                .sink()
                .build();

        ObjectSchema outputSchema = new ObjectSchema();
        for (int i = 0; i < 9; i++)
            outputSchema.appendColumn(new ObjectColumn(ValType.Int32));
        Pipeline p = LoadLineorder().newPipeline("q23", outputSchema)
                .joinWith(date, "$lo_orderdate", "$d_datekey")
                .joinWith(part, "$lo_partkey", "$p_partkey")
                .joinWith(supp, "$lo_suppkey", "$s_suppkey")
                .reduceBy("$d_year,$p_brand1", "$lo_revenue/100")
                .sink()
                .build();
        return p;
    }

    static public Pipeline Q31() {
//        select c_nation, s_nation, d_year, sum(lo_revenue) as revenue
//            from customer, lineorder, supplier, date
//            where lo_custkey = c_custkey
//                and lo_suppkey = s_suppkey
//                and lo_orderdate = d_datekey
//                and c_region = 'ASIA'
//                and s_region = 'ASIA'
//                and d_year >= 1992 and d_year <= 1997
//            group by c_nation, s_nation, d_year
//            order by d_year asc, revenue desc
        ObjectSchema queryDateSchema = new ObjectSchema();
        queryDateSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryDateSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryDateSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline date = LoadDate().newPipeline("q31-date", queryDateSchema)
                .filter("$d_year >= 1992 and $d_year <= 1997")
                .map("$d_datekey", "$d_year")
                .sink()
                .build();

        ObjectSchema queryCustomerSchema = new ObjectSchema();
        queryCustomerSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryCustomerSchema.appendColumn(new ObjectColumn(ValType.String, 15));
        queryCustomerSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline customer = LoadCustomer().newPipeline("q31-customer", queryCustomerSchema)
                .filter("StrCmp($c_region,\\\"ASIA\\\",15)==0")
                .map("$c_custkey", "$c_nation")
                .sink()
                .build();

        ObjectSchema querySuppSchema = new ObjectSchema();
        querySuppSchema.appendColumn(new ObjectColumn(ValType.Int32));
        querySuppSchema.appendColumn(new ObjectColumn(ValType.String, 15));
        querySuppSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline supp = LoadSupplier().newPipeline("q31-supplier", querySuppSchema)
                .filter("StrCmp($s_region,\\\"ASIA\\\",15)==0")
                .map("$s_suppkey", "$s_nation")
                .sink()
                .build();

        ObjectSchema outputSchema = new ObjectSchema();
        for (int i = 0; i < 12; i++)
            outputSchema.appendColumn(new ObjectColumn(ValType.Int32));
        Pipeline p = LoadLineorder().newPipeline("q31", outputSchema)
                .joinWith(supp, "$lo_suppkey", "$s_suppkey")
                .joinWith(date, "$lo_orderdate", "$d_datekey")
                .joinWith(customer, "$lo_custkey", "$c_custkey")
                .reduceBy("$c_nation,$s_nation,$d_year", "$lo_revenue/100")
                .sink()
                .build();
        return p;
    }

    static public Pipeline Q32() {
//        select c_city, s_city, d_year, sum(lo_revenue) as revenue
//            from customer, lineorder, supplier, date
//            where lo_custkey = c_custkey
//                and lo_suppkey = s_suppkey
//                and lo_orderdate = d_datekey
//                and c_nation = 'UNITED STATES'
//                and s_nation = 'UNITED STATES'
//                and d_year >= 1992 and d_year <= 1997
//            group by c_city, s_city, d_year
//            order by d_year asc, revenue desc

        ObjectSchema queryDateSchema = new ObjectSchema();
        queryDateSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryDateSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryDateSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline date = LoadDate().newPipeline("q32-date", queryDateSchema)
                .filter("$d_year >= 1992 and $d_year <= 1997")
                .map("$d_datekey", "$d_year")
                .sink()
                .build();

        ObjectSchema queryCustomerSchema = new ObjectSchema();
        queryCustomerSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryCustomerSchema.appendColumn(new ObjectColumn(ValType.String, 10));
        queryCustomerSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline customer = LoadCustomer().newPipeline("q32-customer", queryCustomerSchema)
                .filter("StrCmp($c_nation,\\\"UNITED STATES\\\",15)==0")
                .map("$c_custkey", "$c_city")
                .sink()
                .build();

        ObjectSchema querySuppSchema = new ObjectSchema();
        querySuppSchema.appendColumn(new ObjectColumn(ValType.Int32));
        querySuppSchema.appendColumn(new ObjectColumn(ValType.String, 10));
        querySuppSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline supp = LoadSupplier().newPipeline("q32-supplier", querySuppSchema)
                .filter("StrCmp($s_nation,\\\"UNITED STATES\\\",15)==0")
                .map("$s_suppkey", "$s_city")
                .sink()
                .build();

        ObjectSchema outputSchema = new ObjectSchema();
        for (int i = 0; i < 12; i++)
            outputSchema.appendColumn(new ObjectColumn(ValType.Int32));
        Pipeline p = LoadLineorder().newPipeline("q32", outputSchema)
                .joinWith(supp, "$lo_suppkey", "$s_suppkey")
                .joinWith(date, "$lo_orderdate", "$d_datekey")
                .joinWith(customer, "$lo_custkey", "$c_custkey")
                .reduceBy("$c_city,$s_city,$d_year", "$lo_revenue/100")
                .sink()
                .build();
        return p;
    }

    static public Pipeline Q33() {
//        select c_city, s_city, d_year, sum(lo_revenue) as revenue
//            from customer, lineorder, supplier, date
//            where lo_custkey = c_custkey
//                and lo_suppkey = s_suppkey
//                and lo_orderdate = d_datekey
//                and c_nation = 'UNITED KINGDOM'
//                and (c_city='UNITED KI1' or c_city='UNITED KI5')
//                and (s_city='UNITED KI1' or s_city='UNITED KI5')
//                and s_nation = 'UNITED KINGDOM'
//                and d_year >= 1992 and d_year <= 1997
//            group by c_city, s_city, d_year
//            order by d_year asc, revenue desc


        ObjectSchema queryDateSchema = new ObjectSchema();
        queryDateSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryDateSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryDateSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline date = LoadDate().newPipeline("q33-date", queryDateSchema)
                .filter("$d_year >= 1992 and $d_year <= 1997")
                .map("$d_datekey", "$d_year")
                .sink()
                .build();

        ObjectSchema queryCustomerSchema = new ObjectSchema();
        queryCustomerSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryCustomerSchema.appendColumn(new ObjectColumn(ValType.String, 10));
        queryCustomerSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline customer = LoadCustomer().newPipeline("q33-customer", queryCustomerSchema)
                .filter("StrCmp($c_nation,\\\"UNITED KINGDOM\\\",15)==0 && (StrCmp($c_city, \\\"UNITED KI1\\\",10)==0 || StrCmp($c_city, \\\"UNITED KI5\\\",10)==0)")
                .map("$c_custkey", "$c_city")
                .sink()
                .build();

        ObjectSchema querySuppSchema = new ObjectSchema();
        querySuppSchema.appendColumn(new ObjectColumn(ValType.Int32));
        querySuppSchema.appendColumn(new ObjectColumn(ValType.String, 10));
        querySuppSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline supp = LoadSupplier().newPipeline("q33-supplier", querySuppSchema)
                .filter("StrCmp($s_nation,\\\"UNITED KINGDOM\\\",15)==0 && (StrCmp($s_city, \\\"UNITED KI1\\\",10)==0 || StrCmp($s_city, \\\"UNITED KI5\\\",10)==0)")
                .map("$s_suppkey", "$s_city")
                .sink()
                .build();

        ObjectSchema outputSchema = new ObjectSchema();
        for (int i = 0; i < 12; i++)
            outputSchema.appendColumn(new ObjectColumn(ValType.Int32));
        Pipeline p = LoadLineorder().newPipeline("q33", outputSchema)
                .joinWith(supp, "$lo_suppkey", "$s_suppkey")
                .joinWith(date, "$lo_orderdate", "$d_datekey")
                .joinWith(customer, "$lo_custkey", "$c_custkey")
                .reduceBy("$c_city,$s_city,$d_year", "$lo_revenue/100")
                .sink()
                .build();
        return p;
    }

    static public Pipeline Q34() {
//        select c_city, s_city, d_year, sum(lo_revenue) as revenue
//            from customer, lineorder, supplier, date
//            where lo_custkey = c_custkey
//                and lo_suppkey = s_suppkey
//                and lo_orderdate = d_datekey
//                and c_nation = 'UNITED KINGDOM'
//                and (c_city='UNITED KI1' or c_city='UNITED KI5')
//                and (s_city='UNITED KI1' or s_city='UNITED KI5')
//                and s_nation = 'UNITED KINGDOM'
//                and d_yearmonth = 'Dec1997'
//            group by c_city, s_city, d_year
//            order by d_year asc, revenue desc

        ObjectSchema queryDateSchema = new ObjectSchema();
        queryDateSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryDateSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryDateSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline date = LoadDate().newPipeline("q34-date", queryDateSchema)
                .filter("StrCmp($d_yearmonth, \\\"Dec1997\\\", 8)==0")
                .map("$d_datekey", "$d_year")
                .sink()
                .build();

        ObjectSchema queryCustomerSchema = new ObjectSchema();
        queryCustomerSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryCustomerSchema.appendColumn(new ObjectColumn(ValType.String, 10));
        queryCustomerSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline customer = LoadCustomer().newPipeline("q34-customer", queryCustomerSchema)
                .filter("StrCmp($c_nation,\\\"UNITED KINGDOM\\\",15)==0 && (StrCmp($c_city, \\\"UNITED KI1\\\",10)==0 || StrCmp($c_city, \\\"UNITED KI5\\\",10)==0)")
                .map("$c_custkey", "$c_city")
                .sink()
                .build();

        ObjectSchema querySuppSchema = new ObjectSchema();
        querySuppSchema.appendColumn(new ObjectColumn(ValType.Int32));
        querySuppSchema.appendColumn(new ObjectColumn(ValType.String, 10));
        querySuppSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline supp = LoadSupplier().newPipeline("q34-supplier", querySuppSchema)
                .filter("StrCmp($s_nation,\\\"UNITED KINGDOM\\\",15)==0 && (StrCmp($s_city, \\\"UNITED KI1\\\",10)==0 || StrCmp($s_city, \\\"UNITED KI5\\\",10)==0)")
                .map("$s_suppkey", "$s_city")
                .sink()
                .build();

        ObjectSchema outputSchema = new ObjectSchema();
        for (int i = 0; i < 12; i++)
            outputSchema.appendColumn(new ObjectColumn(ValType.Int32));
        Pipeline p = LoadLineorder().newPipeline("q34", outputSchema)
                .joinWith(supp, "$lo_suppkey", "$s_suppkey")
                .joinWith(date, "$lo_orderdate", "$d_datekey")
                .joinWith(customer, "$lo_custkey", "$c_custkey")
                .reduceBy("$c_city,$s_city,$d_year", "$lo_revenue/100")
                .sink()
                .build();
        return p;
    }

    static public Pipeline Q41() {
//        select d_year, c_nation, sum(lo_revenue-lo_supplycost) as profit1
//            from date, customer, supplier, part, lineorder
//            where lo_custkey = c_custkey
//                and lo_suppkey = s_suppkey
//                and lo_partkey = p_partkey
//                and lo_orderdate = d_datekey
//                and c_region = 'AMERICA'
//                and s_region = 'AMERICA'
//                and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2')
//            group by d_year, c_nation
//            order by d_year, c_nation
        ObjectSchema queryDateSchema = new ObjectSchema();
        queryDateSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryDateSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryDateSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline date = LoadDate().newPipeline("q41-date", queryDateSchema)
                .map("$d_datekey", "$d_year")
                .sink()
                .build();

        ObjectSchema queryCustomerSchema = new ObjectSchema();
        queryCustomerSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryCustomerSchema.appendColumn(new ObjectColumn(ValType.String, 15));
        queryCustomerSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline customer = LoadCustomer().newPipeline("q41-customer", queryCustomerSchema)
                .filter("StrCmp($c_region,\\\"AMERICA\\\",15)==0")
                .map("$c_custkey", "$c_nation")
                .sink()
                .build();

        ObjectSchema querySuppSchema = new ObjectSchema();
        querySuppSchema.appendColumn(new ObjectColumn(ValType.Int32));
        querySuppSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline supp = LoadSupplier().newPipeline("q41-supplier", querySuppSchema)
                .filter("StrCmp($s_region,\\\"AMERICA\\\",15)==0")
                .map("$s_suppkey")
                .sink()
                .build();

        ObjectSchema queryPartSchema = new ObjectSchema();
        queryPartSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryPartSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline part = LoadPart().newPipeline("q41-part", queryPartSchema)
                .filter("StrCmp($p_mfgr,\\\"MFGR#1\\\",8)==0 || StrCmp($p_mfgr,\\\"MFGR#2\\\",8)==0")
                .map("$p_partkey")
                .sink()
                .build();

        ObjectSchema outputSchema = new ObjectSchema();
        for (int i = 0; i < 12; i++)
            outputSchema.appendColumn(new ObjectColumn(ValType.Int32));
        Pipeline p = LoadLineorder().newPipeline("q41", outputSchema)
                .joinWith(part, "$lo_partkey", "$p_partkey")
                .joinWith(date, "$lo_orderdate", "$d_datekey")
                .joinWith(supp, "$lo_suppkey", "$s_suppkey")
                .joinWith(customer, "$lo_custkey", "$c_custkey")
                .reduceBy("$d_year,$c_nation", "($lo_revenue-$lo_supplycost)/100")
                .sink()
                .build();
        return p;
    }

    static public Pipeline Q42() {
//        select d_year, s_nation, p_category, sum(lo_revenue-lo_supplycost) as profit1
//            from date, customer, supplier, part, lineorder
//            where lo_custkey = c_custkey
//                and lo_suppkey = s_suppkey
//                and lo_partkey = p_partkey
//                and lo_orderdate = d_datekey
//                and c_region = 'AMERICA'
//                and s_region = 'AMERICA'
//                and (d_year = 1997 or d_year = 1998)
//                and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2')
//            group by d_year, s_nation, p_category
//            order by d_year, s_nation, p_category

        ObjectSchema queryDateSchema = new ObjectSchema();
        queryDateSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryDateSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryDateSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline date = LoadDate().newPipeline("q42-date", queryDateSchema)
                .filter("$d_year == 1997 and $d_year == 1998")
                .map("$d_datekey", "$d_year")
                .sink()
                .build();

        ObjectSchema queryCustomerSchema = new ObjectSchema();
        queryCustomerSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryCustomerSchema.appendColumn(new ObjectColumn(ValType.String, 15));
        queryCustomerSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline customer = LoadCustomer().newPipeline("q42-customer", queryCustomerSchema)
                .filter("StrCmp($c_region,\\\"AMERICA\\\",15)==0")
                .map("$c_custkey", "$c_nation")
                .sink()
                .build();

        ObjectSchema querySuppSchema = new ObjectSchema();
        querySuppSchema.appendColumn(new ObjectColumn(ValType.Int32));
        querySuppSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline supp = LoadSupplier().newPipeline("q42-supplier", querySuppSchema)
                .filter("StrCmp($s_region,\\\"AMERICA\\\",15)==0")
                .map("$s_suppkey")
                .sink()
                .build();

        ObjectSchema queryPartSchema = new ObjectSchema();
        queryPartSchema.appendColumn(new ObjectColumn(ValType.Int32));
        queryPartSchema.appendColumn(new ObjectColumn(ValType.String, 8));
        queryPartSchema.appendColumn(new ObjectColumn(ValType.String, 40));
        Pipeline part = LoadPart().newPipeline("q42-part", queryPartSchema)
                .filter("StrCmp($p_mfgr,\\\"MFGR#1\\\",8)==0 || StrCmp($p_mfgr,\\\"MFGR#2\\\",8)==0")
                .map("$p_partkey", "$p_category")
                .sink()
                .build();

        ObjectSchema outputSchema = new ObjectSchema();
        for (int i = 0; i < 12; i++)
            outputSchema.appendColumn(new ObjectColumn(ValType.Int32));
        Pipeline p = LoadLineorder().newPipeline("q42", outputSchema)
                .joinWith(part, "$lo_partkey", "$p_partkey")
                .joinWith(date, "$lo_orderdate", "$d_datekey")
                .joinWith(supp, "$lo_suppkey", "$s_suppkey")
                .joinWith(customer, "$lo_custkey", "$c_custkey")
                .reduceBy("$d_year,$c_nation,$p_category", "($lo_revenue-$lo_supplycost)/100")
                .sink()
                .build();
        return p;
    }

    static public Pipeline Q43() {
        // date
        Pipeline date = SSBHelper.LoadDate();
        Pipeline datePipeline = SSBHelper.Q43Date(date);

        // part
        Pipeline part = SSBHelper.LoadPart();
        Pipeline partPipeline = SSBHelper.Q43Part(part);

        // supplier
        Pipeline supplier = SSBHelper.LoadSupplier();
        Pipeline supplierPipeline = SSBHelper.Q43Supplier(supplier);

        // customer
        Pipeline customer = SSBHelper.LoadCustomer();
        Pipeline customerPipeline = SSBHelper.Q43Customer(customer);

        // lineorder
        Pipeline lineorder = SSBHelper.LoadLineorder();
        Pipeline query = SSBHelper.Q43Lineorder(lineorder, datePipeline, partPipeline, supplierPipeline, customerPipeline);
        return query;
    }


}
