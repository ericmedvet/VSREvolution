package it.units.erallab.mappers;

import it.units.erallab.RealFunction;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;

import java.util.Collections;
import java.util.List;

/**
 * @author eric
 * @created 2020/12/08
 * @project VSREvolution
 */
public class MLP implements ReversableFunction<List<Double>, RealFunction> {

  private final double innerLayerRatio;

  public MLP(double innerLayerRatio) {
    this.innerLayerRatio = innerLayerRatio;
  }

  @Override
  public List<Double> example(RealFunction function) { //TODO probably this should be without argument; function should instead be an arg of the contstructor; or maybe just the size
    int[] innerNeurons = innerLayerRatio == 0 ? new int[0] : new int[]{(int) Math.round((double) function.getInputDim() * innerLayerRatio)};
    return Collections.nCopies(
        MultiLayerPerceptron.countWeights(
            MultiLayerPerceptron.countNeurons(
                function.getInputDim(),
                innerNeurons,
                function.getOutputDim())
        ), 0d);
  }

  @Override
  public RealFunction apply(List<Double> doubles) {
    return null; //TODO fix
  }
}
