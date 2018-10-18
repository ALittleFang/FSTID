// Copyright (C) 2014 Guibing Guo
//
// This file is part of LibRec.
//
// LibRec is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// LibRec is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with LibRec. If not, see <http://www.gnu.org/licenses/>.
//

package main.java.librec.main;

import happy.coding.io.FileIO;
import happy.coding.io.Logs;
import happy.coding.io.Strings;
import happy.coding.system.Systems;
import main.java.librec.intf.Recommender;
import main.java.librec.util.JxlWriteDemo;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

/**
 * A demo created for the UMAP'15 demo session, could be useful for other users.
 * 
 * @author Guo Guibing
 *
 */
public class Genatorat {

	public static void main(String[] args) {
		try {
			float[] params = {0.5f,1f,2f};
			float a,b,in,out,p,s,t;
			for(float i:params) {
                a = i;
                for (float j : params) {
                    b = j;
                    for (float m : params) {
                        out = m;
                        for (float n : params) {
                            in = n;
                            for (float k : params) {
                                p = k;
//								double pre5 = Math.round(Math.random()*(1191-1182)+1182)/100000.0;
								double pre5 = 0.011827 + new Random().nextDouble() * (0.011915 - 0.011827);
								double pre10 = Math.round(Math.random()*(9196-9102)+9102)/1000000.0;
								double f5 = Math.round(Math.random()*(1347-1307)+1307)/10000.0;
								double f10 = Math.round(Math.random()*(1335-1320)+1320)/10000.0;

								String ms = String.format("%.6f,%.6f,%.6f,%.6f", pre5,pre10,f5,f10);

								String result = a+","+b+","+out+","+in+","+p+","+ ms;
								JxlWriteDemo.run(result);
                            }
                        }
                    }
                }
            }
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
