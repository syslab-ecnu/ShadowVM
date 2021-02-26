package ShadowVM;

import ShadowVM.proto.types.*;

import java.io.Serializable;

public class ObjectColumn implements Serializable {
    public ColumnInfo columnInfo;
    public String columnName;
    private int getTypeSize(ValType type) {
        int size = 0;
        switch (type) {
            case Byte: size = 1; break;
            case Int16: size = 2; break;
            case Int32: size = 4; break;
            case Int64: size = 8; break;
            case Float32: size = 4; break;
            case Float64: size = 8; break;
            default:
                // TODO: warning
                System.out.println("Create Column Error");
                break;
        }
        return size;
    }

    public ObjectColumn(ValType type, int size) {
        columnInfo = ColumnInfo.newBuilder()
                .setType(type)
                .setValSize(size)
                .build();
    }

    public ObjectColumn(ValType type) {
        int size = getTypeSize(type);
        columnInfo = ColumnInfo.newBuilder()
                .setType(type)
                .setValSize(size)
                .build();
    }

    public ObjectColumn(String name, ValType type, int size) {
        this(type, size);
        columnName = name;
    }

    public ObjectColumn(String name, ValType type) {
        this(type);
        columnName = name;
    }
}
