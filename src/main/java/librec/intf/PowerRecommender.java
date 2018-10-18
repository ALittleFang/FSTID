package main.java.librec.intf;

import happy.coding.io.Logs;
import happy.coding.io.Strings;
import main.java.librec.data.DataDAO;
import main.java.librec.data.DenseMatrix;
import main.java.librec.data.SparseMatrix;

/**
 * Created by 84064
 * on 2018/3/10
 */
public abstract class PowerRecommender extends SocialRecommender {

    // 影响力矩阵
    protected static SparseMatrix powerMatrix;

    // initialization
    static {
        String powerPath = cf.getPath("dataset.power");
        Logs.debug("Power dataset: {}", Strings.last(powerPath, 38));

        socialDao = new DataDAO(powerPath, rateDao.getUserIds());

        try {
            powerMatrix = socialDao.readData();

            //socialCache = socialMatrix.rowCache(cacheSpec);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }
    public PowerRecommender(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);
    }
}
