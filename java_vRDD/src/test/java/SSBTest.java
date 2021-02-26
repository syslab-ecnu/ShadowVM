import ShadowVM.Pipeline;
import org.junit.Assert;
import org.junit.Test;

public class SSBTest {

    @Test
    public void TestBuildSSBQ11() {
        Pipeline p = SSBHelper.Q11();
        System.out.println(p.toString());
        checkExpression(p.toString());
    }

    @Test
    public void TestBuildSSBQ12() {
        Pipeline p = SSBHelper.Q12();
        System.out.println(p.toString());
        checkExpression(p.toString());
    }

    @Test
    public void TestBuildSSBQ13() {
        Pipeline p = SSBHelper.Q13();
        System.out.println(p.toString());
        checkExpression(p.toString());
    }

    @Test
    public void TestBuildSSBQ21() {
        Pipeline p = SSBHelper.Q21();
        System.out.println(p.toString());
        checkExpression(p.toString());
    }

    @Test
    public void TestBuildSSBQ22() {
        Pipeline p = SSBHelper.Q22();
        System.out.println(p.toString());
        checkExpression(p.toString());
    }

    @Test
    public void TestBuildSSBQ23() {
        Pipeline p = SSBHelper.Q23();
        System.out.println(p.toString());
        checkExpression(p.toString());
    }

    @Test
    public void TestBuildSSBQ31() {
        Pipeline p = SSBHelper.Q31();
        System.out.println(p.toString());
        checkExpression(p.toString());
    }

    @Test
    public void TestBuildSSBQ32() {
        Pipeline p = SSBHelper.Q32();
        System.out.println(p.toString());
        checkExpression(p.toString());
    }

    @Test
    public void TestBuildSSBQ33() {
        Pipeline p = SSBHelper.Q32();
        System.out.println(p.toString());
        checkExpression(p.toString());
    }

    @Test
    public void TestBuildSSBQ34() {
        Pipeline p = SSBHelper.Q34();
        System.out.println(p.toString());
        checkExpression(p.toString());
    }

    @Test
    public void TestBuildSSBQ41() {
        Pipeline p = SSBHelper.Q41();
        System.out.println(p.toString());
        checkExpression(p.toString());
    }

    @Test
    public void TestBuildSSBQ42() {
        Pipeline p = SSBHelper.Q42();
        System.out.println(p.toString());
        checkExpression(p.toString());
    }

    @Test
    public void TestBuildSSBQ43() {
        Pipeline p = SSBHelper.Q43();
        System.out.println(p.toString());
        checkExpression(p.toString());
    }

    private void checkExpression(String s) {
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '$' && i != s.length() - 1 && s.charAt(i+1) != '(') {
                System.out.printf("CheckExpression Fail: %s\n", s.substring(i).split(" ")[0]);
                Assert.fail();
            }
        }
    }
}
