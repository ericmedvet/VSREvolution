package it.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.PruningMultiLayerPerceptron;
import it.units.malelab.jgea.core.util.ArrayTable;
import it.units.malelab.jgea.core.util.ImagePlotters;
import it.units.malelab.jgea.core.util.Table;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author eric on 2021/02/10 for VSREvolution
 */
public class MLPPruningTest {

  public static void main(String[] args) throws IOException {
    nnIOPlots();
  }

  public static void errPlots() throws IOException {
    int[] nOfInputs = {10, 25, 50, 100};
    int[] nOfLayers = {0, 1, 2};
    int[] seeds = IntStream.range(0, 30).toArray();
    double dT = 1d / 60d;
    double totalT = 10d;
    double[] rates = IntStream.range(0, 20).mapToDouble(i -> (double) i * 0.025d).toArray();
    for (int nOfInput : nOfInputs) {
      for (int nOfLayer : nOfLayers) {
        int[] innerLayers = IntStream.range(0, nOfLayer).map(l -> nOfInput).toArray();
        List<String> names = new ArrayList<>();
        names.add("rate");
        for (PruningMultiLayerPerceptron.Context context : PruningMultiLayerPerceptron.Context.values()) {
          for (PruningMultiLayerPerceptron.Criterion criterion : PruningMultiLayerPerceptron.Criterion.values()) {
            names.add(context.name().toLowerCase() + "/" + criterion.name().toLowerCase());
          }
        }
        Table<Double> table = new ArrayTable<>(names);
        for (double rate : rates) {
          List<Double> row = new ArrayList<>();
          row.add(rate);
          for (PruningMultiLayerPerceptron.Context context : PruningMultiLayerPerceptron.Context.values()) {
            for (PruningMultiLayerPerceptron.Criterion criterion : PruningMultiLayerPerceptron.Criterion.values()) {
              double err = 0d;
              for (int seed : seeds) {
                Random r = new Random(seed);
                MultiLayerPerceptron nn = new MultiLayerPerceptron(MultiLayerPerceptron.ActivationFunction.TANH, nOfInput, innerLayers, 1);
                MultiLayerPerceptron pnn = new PruningMultiLayerPerceptron(MultiLayerPerceptron.ActivationFunction.TANH, nOfInput, innerLayers, 1, totalT / 2, context, criterion, rate);
                double[] ws = IntStream.range(0, nn.getParams().length).mapToDouble(i -> r.nextDouble() * 2d - 1d).toArray();
                nn.setParams(ws);
                pnn.setParams(ws);
                for (double t = 0; t < totalT; t = t + dT) {
                  final double finalT = t;
                  double[] inputs = IntStream.range(0, nOfInput).mapToDouble(j -> Math.sin(finalT / (double) (j + 1))).toArray();
                  double localErr = Math.abs(nn.apply(inputs)[0] - pnn.apply(inputs)[0]);
                  if (t > totalT / 2) {
                    err = err + localErr;
                  }
                }
              }
              row.add(err / (double) seeds.length / totalT * 2d);
            }
          }
          table.addRow(row);
        }
        String fileName = String.format(
            "/home/eric/err-vs-rate-i%d-l%d.png",
            nOfInput,
            innerLayers.length
        );
        System.out.println(fileName);
        ImageIO.write(
            ImagePlotters.xyLines(800, 600).apply(table),
            "png",
            new File(fileName));
      }
    }
  }

  public static void nnIOPlots() throws IOException {
    int nOfInput = 2;
    int nOfCalls = 50;
    double dT = 1d / 60d;
    double totalT = 10d;
    double pruningT = 5d;
    int[] innerLayers = new int[]{8, 7, 6};
    double[] rates = new double[]{0.05, 0.10, 0.15, 0.25, 0.33, 0.5};
    for (PruningMultiLayerPerceptron.Context context : PruningMultiLayerPerceptron.Context.values()) {
      for (PruningMultiLayerPerceptron.Criterion criterion : PruningMultiLayerPerceptron.Criterion.values()) {
        MultiLayerPerceptron nn = new MultiLayerPerceptron(MultiLayerPerceptron.ActivationFunction.TANH, nOfInput, innerLayers, 1);
        List<PruningMultiLayerPerceptron> pnns = Arrays.stream(rates).mapToObj(r -> new PruningMultiLayerPerceptron(MultiLayerPerceptron.ActivationFunction.TANH, nOfInput, innerLayers, 1, pruningT, context, criterion, r)).collect(Collectors.toList());
        Random r = new Random(2);
        double[] ws = IntStream.range(0, nn.getParams().length).mapToDouble(i -> r.nextDouble() * 2d - 1d).toArray();
        nn.setParams(ws);
        System.out.println(ws.length);
        pnns.forEach(pnn -> pnn.setParams(ws));

        List<String> names = new ArrayList<>();
        names.add("x");
        names.add("y-nn");
        for (double rate : rates) {
          names.add(String.format(
              "y-pnn-%s-%s-%4.2f",
              criterion.toString().toLowerCase(),
              context.toString().toLowerCase(),
              rate
          ));
        }
        Table<Double> table = new ArrayTable<>(names);

        for (double t = 0; t < totalT; t = t + dT) {
          double[] input = new double[]{Math.sin(0.1 * t), Math.sin(t)};
          List<Double> values = new ArrayList<>();
          values.add(t);
          values.add(nn.apply(input)[0]);
          pnns.forEach(pnn -> values.add(pnn.apply(input)[0]));
          table.addRow(values);
        }

        ImageIO.write(
            ImagePlotters.xyLines(800, 600).apply(table),
            "png",
            new File(String.format(
                "/home/eric/pnns-%s-%s.png",
                criterion.toString().toLowerCase(),
                context.toString().toLowerCase()
            )));
      }
    }
  }

}
