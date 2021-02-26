package ShadowVM;

import java.io.Serializable;

public class SVMContext implements Serializable {
    public String host;
    public int port;
    public boolean codegen = false;
    public boolean debug = false;
    public boolean isGPU = false;
    public SVMContext(String host, int port, boolean codegen, boolean debug, boolean isGPU) {
        this.host = host;
        this.port = port;
        this.codegen = codegen;
        this.debug = debug;
        this.isGPU = isGPU;
    }
}
