/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package linearregressionweka;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.List;
import weka.classifiers.evaluation.NumericPrediction;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author alessandro
 */
public class LinearRegressionWeka {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        try {
            // Path
            String lineardata = "/home/alessandro/Documentos/weka-example/houses.arff";
            // load data
            Instances linear = new Instances(new BufferedReader(new FileReader(lineardata)));
            // Number the atributes
            linear.setClassIndex(linear.numAttributes() - 1);
            // Create a Linear Regression
            LinearRegression model = new LinearRegression();
            // Run the classifier
            model.buildClassifier(linear);
            // Print the model classifier
            System.out.println(model);
            
            // Create a new instance for myhouse
            Instance myhouse = linear.lastInstance();
            // Apply Linear Regression for my instance
            double price = model.classifyInstance(myhouse);
            // Print myhouse 
            System.out.println("My house  (" + myhouse + "): " + price);

        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

}
