package ShadowVM;

import ShadowVM.ObjectColumn;
import ShadowVM.proto.types.*;

import java.io.Serializable;
import java.util.ArrayList;

public class ObjectSchema implements Serializable {
    private ObjectInfo objectInfo;

    public String table;
    public ArrayList<ObjectColumn> columns;

    public ObjectSchema() {
        objectInfo = ObjectInfo
            .newBuilder()
            .setStorage(StoType.HostMem)
            .build();
        table = "_UNKNOWN_";
        columns = new ArrayList<>();
    }
    public ObjectSchema(String tableName) {
        this();
        table = tableName;
    }
    public ObjectSchema(ObjectInfo obj) {
        objectInfo = obj;
        table = "_UNKNOWN_";
        columns = new ArrayList<>();
        for (ColumnInfo col : obj.getColumnListList()) {
            columns.add(new ObjectColumn("_UNKNOWN_", col.getType(), (int)col.getValSize()));
        }
    }
    public void setStorage(StoType type) {
        objectInfo = objectInfo.toBuilder().setStorage(type).build();
    }

    public void appendColumn(ObjectColumn column) {
        objectInfo = objectInfo.toBuilder().addColumnList(column.columnInfo).build();
        columns.add(column);
    }

    public ObjectInfo getObjectInfo() {
        return objectInfo;
    }

}
