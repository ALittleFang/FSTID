package main.java.librec.ranking;

import com.google.common.cache.LoadingCache;
import main.java.librec.data.Configuration;
import main.java.librec.data.DenseMatrix;
import main.java.librec.data.DenseVector;
import main.java.librec.data.SparseMatrix;
import main.java.librec.intf.SocialRecommender;
import main.java.librec.util.Randoms;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

/**
 * FSTID: 结合信任出入度及影响力
 *
 */
@Configuration("binThold, rho, alpha, p, factors, lRate, maxLRate, regU, regI, regB, iters")
public class FSTID extends SocialRecommender {
    /**
     * P、Q-用户的特征矩阵（一般P指用户v，q指用户u）
     * X、Y-项目的特征矩阵（一般X指项目j，Y指项目i）
     */
    private DenseMatrix P,Q,X,Y;

    /**
     * user biases and item biases
     */
    private DenseVector itemBiases;

    /**
     * 用户-项目缓存，项目-用户缓存，用户u的信任用户缓存,用户u被信任的缓存，用户u被影响的缓存
     */
    protected LoadingCache<Integer, List<Integer>> userItemsCache,itemUsersCache,userTrusteeCache,userTrusterCache,powerUserCache;


    private int rho;
    private float alpha, beta, z_out , p ,fac_s,fac_t;

    private List<Integer> powerUsers = new ArrayList<>();


    public FSTID(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);

        isRankingPred = true;
    }

    @Override
    public void initModel() throws Exception {
        numUsers = 1578;
        numItems = 2071;
        P = new DenseMatrix(numUsers, numFactors);
        Q = new DenseMatrix(numUsers, numFactors);

        X = new DenseMatrix(numItems, numFactors);
        Y = new DenseMatrix(numItems, numFactors);

        getFacMatrix(0);
        getFacMatrix(1);
        Q = P.clone();
        Y = X.clone();

        double smallValue=0.01;

        itemBiases = new DenseVector(numItems);
        itemBiases.init(smallValue);

        getPowerUser();

        algoOptions = cf.getParamOptions("FSTID");
        rho = algoOptions.getInt("-rho");
        alpha = algoOptions.getFloat("-alpha");
        beta = algoOptions.getFloat("-beta");
        z_out = algoOptions.getFloat("-t_out");
        p = algoOptions.getFloat("-p");
        fac_s = algoOptions.getFloat("-fs");
        fac_t = algoOptions.getFloat("-ft");

        userItemsCache = trainMatrix.rowColumnsCache(cacheSpec);
        itemUsersCache = trainMatrix.columnRowsCache(cacheSpec);
        userTrusteeCache = socialMatrix.rowColumnsCache(cacheSpec);
        userTrusterCache = socialMatrix.columnRowsCache(cacheSpec);
    }

    @Override
    protected void buildModel() throws Exception {

        for (int iter = 1; iter <= numIters; iter++) {

            loss = 0;

            DenseMatrix PS = new DenseMatrix(numUsers, numFactors);
            DenseMatrix QS = new DenseMatrix(numUsers, numFactors);

            DenseMatrix XS = new DenseMatrix(numItems, numFactors);
            DenseMatrix YS = new DenseMatrix(numItems, numFactors);


            // update throughout each user-item-rating (u, j, ruj) cell
            for (int u : trainMatrix.rows()) {
                List<Integer> ratedItems=null,trusteeUsers = null, trusterUsers=null;
                try {
                    ratedItems = userItemsCache.get(u);
                    trusteeUsers = userTrusteeCache.get(u);
                    trusterUsers = userTrusterCache.get(u);
                    //powerUsers = powerUserCache.get(u);
                } catch (ExecutionException e) {
                    e.printStackTrace();
                }
                double wto = trusteeUsers.size() > 0 ? Math.pow(trusteeUsers.size(), -z_out) : 0;
                double wti = trusterUsers.size() > 0 ? Math.pow(trusterUsers.size(), -z_out) : 0;
                double wp = powerUsers.size() > 0 ? Math.pow(powerUsers.size(), -p) : 0;

                for (int i : ratedItems) {
                    double rui = trainMatrix.get(u, i);

                    // sample a set of items unrated by user u
                    List<Integer> js = new ArrayList<>();
                    int len = 0;
                    while (len < rho) {
                        int j = Randoms.uniform(numItems);
                        if (ratedItems.contains(j) || js.contains(j))
                            continue;

                        js.add(j);
                        len++;
                    }

                    // user similarity
                    double sum_vi = 0;
                    double[] sum_vif = new double[numFactors];
                    int cnt_v = 0;
                    List<Integer> ratingUsers = null;
                    try {
                        ratingUsers = itemUsersCache.get(i);
                    } catch (ExecutionException e) {
                        e.printStackTrace();
                    }
                    for (int v : ratingUsers) {
                        if (u != v) {
                            sum_vi += DenseMatrix.rowMult(P, v, Q, u);
                            cnt_v++;

                            for (int f = 0; f < numFactors; f++) {
                                sum_vif[f] += P.get(v, f);
                            }
                        }
                    }
                    double w_vi = cnt_v > 0 ? Math.pow(cnt_v, -beta) : 0;

                    // item similarity
                    double sum_wi = 0;
                    int cnt_w = 0;
                    double[] sum_wif = new double[numFactors];
                    for (int w : ratedItems) {
                        if (w != i) {
                            sum_wi += DenseMatrix.rowMult(X, w, Y, i);
                            cnt_w++;

                            for (int f = 0; f < numFactors; f++) {
                                sum_wif[f] += X.get(w, f);
                            }
                        }
                    }
                    double w_wi = cnt_w > 0 ? Math.pow(cnt_w, -alpha) : 0;
                    double w_wj = ratedItems.size() > 0 ? Math.pow(ratedItems.size(), -alpha) : 0;

                    // 信任出度用户影响
                    double sum_to_i = 0;
                    double[] sum_to_f = new double[numFactors];
                    for (int t : trusteeUsers) {
                        sum_to_i += DenseMatrix.rowMult(P, t, Y, i);

                        for (int f = 0; f < numFactors; f++) {
                            sum_to_f[f] += P.get(t, f);
                        }
                    }

                    // 信任入度用户影响
                    double sum_ti_i = 0;
                    double[] sum_ti_f = new double[numFactors];
                    for (int t : trusterUsers) {
                        sum_ti_i += DenseMatrix.rowMult(P, t, Y, i);

                        for (int f = 0; f < numFactors; f++) {
                            sum_ti_f[f] += P.get(t, f);
                        }
                    }

                    //对目标用户有影响力的用户来和目标项目进行特征相乘
                    double sum_power_i = 0;
                    double[] sum_power_f = new double[numFactors];
                    for (int t : powerUsers) {
                        sum_power_i += DenseMatrix.rowMult(P, t, Y, i);

                        for (int f = 0; f < numFactors; f++) {
                            sum_power_f[f] += P.get(t, f);
                        }
                    }

                    // update for each item j unrated by user u
                    double[] xs = new double[numFactors];
                    double[] ws = new double[numFactors];
                    double[] t_out = new double[numFactors];
                    double[] t_in = new double[numFactors];
                    double[] ps = new double[numFactors];
                    for (int j : js) {

                        List<Integer> Cj = null;
                        try {
                            Cj = itemUsersCache.get(j);
                        } catch (ExecutionException e) {
                            e.printStackTrace();
                        }
                        double sum_vj = 0;
                        double[] sum_vjf = new double[numFactors];
                        int cnt_j = 0;
                        for (int v : Cj) {
                            sum_vj += DenseMatrix.rowMult(P, v, Q, u);
                            cnt_j++;

                            for (int f = 0; f < numFactors; f++) {
                                sum_vjf[f] += P.get(v, f);
                            }
                        }
                        double w_vj = cnt_j > 0 ? Math.pow(cnt_j, -beta) : 0;

                        double sum_wj = 0;
                        double[] sum_wjf = new double[numFactors];
                        for (int w : ratedItems) {
                            sum_wj += DenseMatrix.rowMult(X, w, Y, j);

                            for (int f = 0; f < numFactors; f++) {
                                sum_wjf[f] += X.get(w, f);
                            }
                        }

                        double sum_to_j = 0;
                        for (int t : trusteeUsers) {
                            sum_to_j += DenseMatrix.rowMult(P, t, Y, j);
                        }

                        double sum_ti_j = 0;
                        for (int t : trusterUsers) {
                            sum_ti_j += DenseMatrix.rowMult(P, t, Y, j);
                        }

                        double sum_power_j = 0;
                        for (int t : powerUsers) {
                            sum_power_j += DenseMatrix.rowMult(P, t, Y, j);
                        }


                        double bi = itemBiases.get(i), bj = itemBiases.get(j);
                        double pui = bi + fac_s * w_vi * sum_vi + (1-fac_s)*w_wi * sum_wi +
                                    fac_t * wto * sum_to_i+ (1-fac_t) * wti * sum_ti_i + wp*sum_power_i;
                        double puj = bj + fac_s * w_vj * sum_vj + (1-fac_s)*w_wj * sum_wj +
                                    fac_t *wto * sum_to_j+ (1-fac_t) * wti * sum_ti_j + wp*sum_power_j;
                        double ruj = 0;
                        double eij = (rui - ruj) - (pui - puj);

                        loss += eij * eij;

                        // update bi
                        itemBiases.add(i, -lRate * (-eij + regB * bi));

                        // update bj
                        itemBiases.add(j, -lRate * (eij - regB * bj));

                        loss += regB * bi * bi - regB * bj * bj;

                        // update quf, yif, yjf
                        for (int f = 0; f < numFactors; f++) {
                            double quf = Q.get(u, f);
                            double yif = Y.get(i, f), yjf = Y.get(j, f);

                            //更新qu
                            double delta = eij * (w_vj * sum_vjf[f] - w_vi * sum_vif[f]) + regU * quf;
                            QS.add(u, f, -lRate * delta);

                            loss += regU * quf * quf;

                            //更新yi
                            delta = eij * (-w_wi * sum_wif[f] - wto * sum_to_f[f]- wti * sum_ti_f[f]- wp * sum_power_f[f]) + regI * yif;
                            YS.add(i, f, -lRate * delta);

                            //更新yj
                            delta = eij * (w_wj * sum_wjf[f] + wto * sum_to_f[f] + wti * sum_ti_f[f] + wp * sum_power_f[f]) - regI * yjf;
                            YS.add(j, f, -lRate * delta);

                            loss += regI * yif * yif - regI * yjf * yjf;

                            xs[f] += eij * (-w_vi) * quf;
                            ws[f] += eij * (w_wj * yjf - w_wi * yif);
                            t_out[f] += eij * wto * (yjf - yif);
                            t_in[f] += eij * wti * (yjf - yif);
                            ps[f] += eij * wp * (yjf - yif);
                        }

                        // update pvf for v in cj
                        for (int v : Cj) {
                            for (int f = 0; f < numFactors; f++) {
                                double pvf = P.get(v, f);
                                double delta = eij * w_vj * Q.get(u, f) - regU * pvf;
                                PS.add(v, f, -lRate * delta);

                                loss -= regU * pvf * pvf;
                            }
                        }

                    }

                    // update pvf for v in Ci
                    for (int v : ratingUsers) {
                        if (v != u) {
                            for (int f = 0; f < numFactors; f++) {
                                double pvf = P.get(v, f);
                                double delta = xs[f] / rho + regU * pvf;
                                PS.add(v, f, -lRate * delta);

                                loss += regU * pvf * pvf;
                            }
                        }
                    }

                    // update xwf for w in Ru
                    for (int w : ratedItems) {
                        if (w != i) {
                            for (int f = 0; f < numFactors; f++) {
                                double xwf = X.get(w, f);
                                double delta = ws[f] / rho + regI * xwf;
                                XS.add(w, f, -lRate * delta);

                                loss += regI * xwf * xwf;
                            }
                        }
                    }

                    // 更新出度用户特征矩阵
                    for (int t : trusteeUsers) {
                        for (int f = 0; f < numFactors; f++) {
                            double ptf = P.get(t, f);
                            double delta = t_out[f] / rho + regU * ptf;
                            PS.add(t, f, -lRate * delta);

                            loss += regU * ptf * ptf;
                        }
                    }

                    // 更新入度用户特征矩阵
                    for (int t : trusterUsers) {
                        for (int f = 0; f < numFactors; f++) {
                            double ptf = P.get(t, f);
                            double delta = t_in[f] / rho + regU * ptf;
                            PS.add(t, f, -lRate * delta);

                            loss += regU * ptf * ptf;
                        }
                    }

                    // 更新影响力用户特征矩阵
                    for (int p : powerUsers) {
                        for (int f = 0; f < numFactors; f++) {
                            double ptf = P.get(p, f);
                            double delta = ps[f] / rho + regU * ptf;
                            PS.add(p, f, -lRate * delta);

                            loss += regU * ptf * ptf;
                        }
                    }
                }
            }

            P = P.add(PS);
            Q = Q.add(QS);
            X = X.add(XS);
            Y = Y.add(YS);

            loss *= 0.5;

            if (isConverged(iter))
                break;
            updateLRate(iter);
        }
    }

    @Override
    public double predict(int u, int i) throws Exception {
        List<Integer> ratingUsers = null, ratedItems=null, trusteeUsers=null,trusterUsers=null;
        try {
            ratingUsers = itemUsersCache.get(i);
            ratedItems = userItemsCache.get(u);
            trusteeUsers = userTrusteeCache.get(u);
            trusterUsers = userTrusterCache.get(u);
        } catch (ExecutionException e) {
            e.printStackTrace();
        }

        // user similarity
        double sum_c = 0;
        int count = 0;
        for (int v : ratingUsers) {
            if (v != u) {
                sum_c += DenseMatrix.rowMult(P, v, Q, u);
                count++;
            }
        }
        double wc = count > 0 ? Math.pow(count, -beta) : 0;

        // item similarity
        double sum_r = 0;
        count = 0;
        for (int w : ratedItems) {
            if (w != i) {
                sum_r += DenseMatrix.rowMult(X, w, Y, i);
                count++;
            }
        }
        double wr = count > 0 ? Math.pow(count, -alpha) : 0;

        // 信任出度
        double sum_to = 0;
        for (int t : trusteeUsers) {
            sum_to += DenseMatrix.rowMult(P, t, Y, i);
        }
        double wto = trusteeUsers.size() > 0 ? Math.pow(trusteeUsers.size(), -z_out) : 0;

        // 信任入度
        double sum_ti = 0;
        for (int t : trusterUsers) {
            sum_ti += DenseMatrix.rowMult(P, t, Y, i);
        }
        double wti = trusterUsers.size() > 0 ? Math.pow(trusterUsers.size(), -z_out) : 0;

        // 影响力
        double sum_power = 0;
        for (int t : powerUsers) {
            sum_power += DenseMatrix.rowMult(P, t, Y, i);
        }
        double pti = powerUsers.size() > 0 ? Math.pow(powerUsers.size(), -p) : 0;

        return itemBiases.get(i) + fac_s*wc * sum_c + (1-fac_s)*wr * sum_r +
                fac_t*wto * sum_to + (1-fac_t) *wti * sum_ti + pti*sum_power;
    }

    public void getFacMatrix(int type) throws Exception {
        String pathname ;
        if(type == 0)
            pathname = cf.getPath("dataset.fstid.user_feature");
        else
            pathname = cf.getPath("dataset.fstid.item_feature");

        File filename = new File(pathname); // 要读取以上路径的input。txt文件
        InputStreamReader reader = new InputStreamReader(new FileInputStream(filename),"UTF-8"); // 建立一个输入流对象reader
        BufferedReader br = new BufferedReader(reader); // 建立一个对象，它把文件内容转成计算机能读懂的语言
        String line = "";
        int num = 0;
        // line = br.readLine();  读入语句
        while (line != null) {
            line = br.readLine(); // 一次读入一行数据
            if( line == null || line.equals("") )
                break;
            String[] token =line.trim().split(",");
            for(int i=0;i<token.length;i++){
                if (type == 0)
                    P.set(num,i,Double.parseDouble(token[i])/100);
                else
                    X.set(num,i,Double.parseDouble(token[i])/100);
            }
            num ++;
        }
        br.close();
    }

    public void getPowerUser() throws Exception {
        File filename = new File(cf.getPath("dataset.fstid.power")); // 要读取以上路径的input。txt文件
        InputStreamReader reader = new InputStreamReader(new FileInputStream(filename),"UTF-8"); // 建立一个输入流对象reader
        BufferedReader br = new BufferedReader(reader); // 建立一个对象，它把文件内容转成计算机能读懂的语言
        String line = "";
        int num = 0;
        // line = br.readLine();  读入语句
        while (line != null) {
            line = br.readLine(); // 一次读入一行数据
            if( line == null || line.equals("") )
                break;
            String[] token =line.trim().split(" ");
            powerUsers.add(Integer.valueOf(token[0]));
        }
        br.close();
    }
}
